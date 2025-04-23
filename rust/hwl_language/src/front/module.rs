use crate::front::check::{
    check_type_contains_compile_value, check_type_contains_type, check_type_contains_value, TypeContainsReason,
};
use crate::front::compile::{
    ArenaPortInterfaces, ArenaPorts, CompileItemContext, CompileRefs, Port, PortInfo, PortInterfaceInfo, Register,
    RegisterInfo, Wire, WireInfo,
};
use crate::front::context::{
    BlockKind, CompileTimeExpressionContext, ExpressionContext, ExtraRegisters, IrBuilderExpressionContext,
};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::domain::{DomainSignal, PortDomain, ValueDomain};
use crate::front::expression::{EvaluatedId, ValueInner};
use crate::front::item::ElaboratedItemParams;
use crate::front::scope::{NamedValue, Scope, ScopeContent, ScopedEntry};
use crate::front::signal::{Polarized, Signal};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, ElaboratedInterfaceView, MaybeUndefined};
use crate::front::variables::VariableValues;
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrIfStatement, IrModule,
    IrModuleChild, IrModuleInfo, IrModuleInstance, IrPort, IrPortConnection, IrPortInfo, IrPorts, IrRegister,
    IrRegisterInfo, IrStatement, IrVariables, IrWire, IrWireInfo, IrWireOrPort,
};
use crate::syntax::ast::{
    self, ClockedBlockReset, ExpressionKind, ForStatement, ModulePortBlock, ModulePortInBlock, ModulePortInBlockKind,
    ModulePortSingleKind, ModuleStatement, ModuleStatementKind, PortDirection, ResetKind,
};
use crate::syntax::ast::{
    Block, ClockedBlock, CombinatorialBlock, DomainKind, Identifier, MaybeIdentifier, ModulePortItem, ModulePortSingle,
    PortConnection, RegDeclaration, RegOutPortMarker, Spanned, SyncDomain, WireDeclaration, WireKind,
};
use crate::syntax::parsed::AstRefModule;
use crate::syntax::pos::Span;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
use crate::util::{result_pair, result_pair_split, ResultExt};
use crate::{new_index_type, throw};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Either};
use std::fmt::Debug;
use std::hash::Hash;
// TODO split this file into header/body

type SignalsInScope = Vec<Spanned<Signal>>;

new_index_type!(pub Connector);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct ConnectorSingle(usize);

type ArenaConnectors = Arena<Connector, ConnectorInfo>;

pub struct ConnectorInfo {
    id: Identifier,
    kind: ConnectorKind,
}

enum ConnectorKind {
    Port {
        direction: Spanned<PortDirection>,
        domain: Spanned<PortDomain<ConnectorSingle>>,
        ty: Spanned<HardwareType>,
        single: ConnectorSingle,
    },
    Interface {
        domain: Spanned<DomainKind<Polarized<ConnectorSingle>>>,
        view: ElaboratedInterfaceView,
        singles: Vec<ConnectorSingle>,
    },
}

pub struct ElaboratedModuleHeader {
    item: AstRefModule,
    params: Option<Vec<(Identifier, CompileValue)>>,
    ports: ArenaPorts,
    port_interfaces: ArenaPortInterfaces,
    ports_ir: Arena<IrPort, IrPortInfo>,
    scope_ports: ScopeContent,
}

pub struct ElaboratedModuleInfo {
    pub module_ast: AstRefModule,
    pub module_ir: IrModule,
    pub connectors: ArenaConnectors,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_module_ports_new(
        self,
        params: ElaboratedItemParams<AstRefModule>,
    ) -> Result<(ArenaConnectors, ElaboratedModuleHeader), ErrorGuaranteed> {
        let ElaboratedItemParams { item, params } = params;
        let &ast::ItemDefModule {
            span: def_span,
            vis: _,
            id: _,
            params: _,
            ref ports,
            body: _,
        } = &self.fixed.parsed[item];

        // reconstruct header scope
        let mut ctx = CompileItemContext::new_empty(self, None);
        let mut vars = VariableValues::new_root(&ctx.variables);
        let scope_params = ctx.rebuild_params_scope(item.into(), &mut vars, &params)?;

        // elaborate ports
        // TODO we actually need a full context here?
        let (connectors, scope_ports, ports_ir) =
            self.elaborate_module_ports_impl(&mut ctx, &scope_params, &mut vars, ports, def_span);
        let scope_ports = scope_ports.into_content();

        let header: ElaboratedModuleHeader = ElaboratedModuleHeader {
            item,
            params,
            ports: ctx.ports,
            port_interfaces: ctx.port_interfaces,
            ports_ir,
            scope_ports,
        };
        Ok((connectors, header))
    }

    pub fn elaborate_module_body_new(&self, ports: ElaboratedModuleHeader) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let ElaboratedModuleHeader {
            item,
            params,
            ports,
            port_interfaces,
            ports_ir,
            scope_ports,
        } = ports;
        let &ast::ItemDefModule {
            span: def_span,
            vis: _,
            ref id,
            params: _,
            ports: _,
            ref body,
        } = &self.fixed.parsed[item];

        self.check_should_stop(id.span())?;

        // rebuild scopes
        let mut ctx = CompileItemContext::new_restore(*self, None, ports, port_interfaces);
        let mut vars = VariableValues::new_root(&ctx.variables);

        let scope_params = ctx.rebuild_params_scope(item.into(), &mut vars, &params)?;
        let scope_ports = Scope::restore_child_from_content(def_span, &scope_params, scope_ports);

        // elaborate the body
        self.elaborate_module_body_impl(ctx, &vars, &scope_ports, id, params.clone(), ports_ir, body)
    }

    fn elaborate_module_ports_impl<'p>(
        &self,
        ctx: &mut CompileItemContext,
        scope_params: &'p Scope<'p>,
        vars: &mut VariableValues,
        ports: &Spanned<Vec<ModulePortItem>>,
        module_def_span: Span,
    ) -> (ArenaConnectors, Scope<'p>, Arena<IrPort, IrPortInfo>) {
        let diags = self.diags;

        let mut scope_ports = Scope::new_child(ports.span.join(Span::single_at(module_def_span.end)), scope_params);

        let mut connectors: ArenaConnectors = Arena::new();
        let mut port_to_single: IndexMap<Port, ConnectorSingle> = IndexMap::new();
        let mut ports_ir = Arena::new();

        let mut next_single = 0;
        let mut next_single = move || {
            let scalar = ConnectorSingle(next_single);
            next_single += 1;
            scalar
        };

        for port_item in &ports.inner {
            match port_item {
                ModulePortItem::Single(port_item) => {
                    let ModulePortSingle { span: _, id, kind } = port_item;
                    match kind {
                        &ModulePortSingleKind::Port { direction, ref kind } => {
                            let (domain, ty) = match &kind.inner {
                                WireKind::Clock => (
                                    Ok(Spanned {
                                        span: kind.span,
                                        inner: PortDomain::Clock,
                                    }),
                                    Ok(Spanned {
                                        span: kind.span,
                                        inner: HardwareType::Clock,
                                    }),
                                ),
                                WireKind::Normal { domain, ty } => (
                                    ctx.eval_port_domain(&scope_ports, domain)
                                        .map(|d| d.map_inner(PortDomain::Kind)),
                                    ctx.eval_expression_as_ty_hardware(&scope_ports, vars, ty, "port"),
                                ),
                            };

                            let entry = result_pair(domain, ty).map(|(domain, ty)| {
                                push_connector_single(
                                    ctx,
                                    &mut ports_ir,
                                    &mut connectors,
                                    &mut port_to_single,
                                    &mut next_single,
                                    id,
                                    direction,
                                    domain,
                                    ty,
                                )
                            });
                            scope_ports.declare(diags, id, entry);
                        }
                        ModulePortSingleKind::Interface {
                            span_keyword: _,
                            domain,
                            interface,
                        } => {
                            let domain = ctx.eval_port_domain(&scope_ports, domain);
                            let interface_view = ctx
                                .eval_expression_as_compile(&scope_ports, vars, interface, "interface")
                                .and_then(|interface| match interface.inner {
                                    CompileValue::InterfaceView(view) => Ok(view),
                                    value => Err(diags.report_simple(
                                        "expected interface view",
                                        interface.span,
                                        format!("got other value `{}`", value.to_diagnostic_string()),
                                    )),
                                });

                            let entry = result_pair(domain, interface_view).and_then(|(domain, view)| {
                                push_connector_interface(
                                    ctx,
                                    &mut ports_ir,
                                    &mut connectors,
                                    &mut port_to_single,
                                    &mut next_single,
                                    id,
                                    domain,
                                    view,
                                )
                            });
                            scope_ports.declare(diags, id, entry);
                        }
                    }
                }
                ModulePortItem::Block(port_item) => {
                    let ModulePortBlock { span: _, domain, ports } = port_item;

                    let domain = ctx.eval_port_domain(&scope_ports, domain);

                    for port in ports {
                        let ModulePortInBlock { span: _, id, kind } = port;

                        match kind {
                            ModulePortInBlockKind::Port { direction, ty } => {
                                let ty = ctx.eval_expression_as_ty_hardware(&scope_ports, vars, ty, "port");

                                let entry = result_pair(domain, ty).map(|(domain, ty)| {
                                    push_connector_single(
                                        ctx,
                                        &mut ports_ir,
                                        &mut connectors,
                                        &mut port_to_single,
                                        &mut next_single,
                                        id,
                                        *direction,
                                        domain.map_inner(PortDomain::Kind),
                                        ty,
                                    )
                                });
                                scope_ports.declare(diags, id, entry);
                            }
                            ModulePortInBlockKind::Interface {
                                span_keyword: _,
                                interface,
                            } => {
                                let interface_view = ctx
                                    .eval_expression_as_compile(&scope_ports, vars, interface, "interface")
                                    .and_then(|interface| match interface.inner {
                                        CompileValue::InterfaceView(view) => Ok(view),
                                        value => Err(diags.report_simple(
                                            "expected interface view",
                                            interface.span,
                                            format!("got other value `{}`", value.to_diagnostic_string()),
                                        )),
                                    });

                                let entry = result_pair(domain, interface_view).and_then(|(domain, view)| {
                                    push_connector_interface(
                                        ctx,
                                        &mut ports_ir,
                                        &mut connectors,
                                        &mut port_to_single,
                                        &mut next_single,
                                        id,
                                        domain,
                                        view,
                                    )
                                });
                                scope_ports.declare(diags, id, entry);
                            }
                        };
                    }
                }
            }
        }

        assert!(ctx.ports.len() >= connectors.len());
        assert_eq!(ctx.ports.len(), ports_ir.len());

        (connectors, scope_ports, ports_ir)
    }

    fn elaborate_module_body_impl(
        &self,
        mut ctx_item: CompileItemContext,
        vars: &VariableValues,
        scope_header: &Scope,
        def_id: &MaybeIdentifier,
        params: Option<Vec<(Identifier, CompileValue)>>,
        ir_ports: Arena<IrPort, IrPortInfo>,
        body: &Block<ModuleStatement>,
    ) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let drivers = Drivers::new(&ctx_item.ports);

        let mut signals_in_scope: SignalsInScope = vec![];
        for (port, port_info) in &ctx_item.ports {
            signals_in_scope.push(Spanned::new(port_info.span, Signal::Port(port)));
        }

        let mut ctx = BodyElaborationContext {
            ctx: &mut ctx_item,
            ir_ports,
            ir_wires: Arena::new(),
            ir_registers: Arena::new(),
            drivers,

            register_initial_values: IndexMap::new(),
            out_port_register_connections: IndexMap::new(),

            pass_1_next_statement_index: 0,
            clocked_block_statement_index_to_process_index: IndexMap::new(),
            children: vec![],
            delayed_error: Ok(()),
        };

        // process declarations
        ctx.elaborate_block(scope_header, vars, &mut signals_in_scope, body, true);

        // stop if any errors have happened so far, we don't want spurious errors about drivers
        ctx.delayed_error?;

        // check driver validness
        // TODO more checking: combinatorial blocks can't read values they will later write,
        //   unless they have already written them
        ctx.pass_2_check_drivers_and_populate_resets()?;

        // create process for registered output ports
        // TODO is it worth trying to avoid this and using the "reg" port markers in verilog?
        if !ctx.out_port_register_connections.is_empty() {
            let mut statements = vec![];

            for (&out_port, &reg) in &ctx.out_port_register_connections {
                let out_port_ir = ctx.ctx.ports[out_port].ir;
                let reg_info = &ctx.ctx.registers[reg];
                let reg_ir = reg_info.ir;
                let statement =
                    IrStatement::Assign(IrAssignmentTarget::port(out_port_ir), IrExpression::Register(reg_ir));
                statements.push(Spanned {
                    span: reg_info.id.span(),
                    inner: statement,
                })
            }

            let process = IrCombinatorialProcess {
                locals: IrVariables::new(),
                block: IrBlock { statements },
            };
            ctx.children
                .push(Child::Finished(IrModuleChild::CombinatorialProcess(process)));
        }

        // return result
        ctx.delayed_error?;
        let processes = ctx
            .children
            .into_iter()
            .map(|c| match c {
                Child::Finished(c) => c,
                Child::Clocked(c) => IrModuleChild::ClockedProcess(c.finish(ctx.ctx)),
            })
            .collect();

        Ok(IrModuleInfo {
            ports: ctx.ir_ports,
            registers: ctx.ir_registers,
            wires: ctx.ir_wires,
            large: ctx_item.large,
            children: processes,
            debug_info_id: def_id.clone(),
            debug_info_generic_args: params,
        })
    }
}

fn push_connector_single(
    ctx: &mut CompileItemContext,
    ports_ir: &mut IrPorts,
    connectors: &mut ArenaConnectors,
    port_to_single: &mut IndexMap<Port, ConnectorSingle>,
    next_single: &mut impl FnMut() -> ConnectorSingle,
    id: &Identifier,
    direction: Spanned<PortDirection>,
    domain: Spanned<PortDomain<Port>>,
    ty: Spanned<HardwareType>,
) -> ScopedEntry {
    let ir_port = ports_ir.push(IrPortInfo {
        name: id.string.clone(),
        direction: direction.inner,
        ty: ty.inner.as_ir(),
        debug_span: id.span,
        debug_info_ty: ty.inner.to_diagnostic_string(),
        debug_info_domain: domain.inner.to_diagnostic_string(ctx),
    });

    let port = ctx.ports.push(PortInfo {
        span: id.span,
        name: id.string.clone(),
        direction,
        domain,
        ty: ty.clone(),
        ir: ir_port,
    });

    let connector_domain = domain.map_inner(|d| d.map_inner(|p| *port_to_single.get(&p).unwrap()));
    let single = next_single();
    connectors.push(ConnectorInfo {
        id: id.clone(),
        kind: ConnectorKind::Port {
            direction,
            domain: connector_domain,
            ty,
            single,
        },
    });
    port_to_single.insert_first(port, single);

    ScopedEntry::Named(NamedValue::Port(port))
}

fn push_connector_interface(
    ctx: &mut CompileItemContext,
    ports_ir: &mut IrPorts,
    connectors: &mut ArenaConnectors,
    port_to_single: &mut IndexMap<Port, ConnectorSingle>,
    next_single: &mut impl FnMut() -> ConnectorSingle,
    id: &Identifier,
    domain: Spanned<DomainKind<Polarized<Port>>>,
    view: ElaboratedInterfaceView,
) -> Result<ScopedEntry, ErrorGuaranteed> {
    let ElaboratedInterfaceView {
        interface,
        view: view_name,
    } = &view;
    let interface = ctx.refs.shared.elaborated_interface_info(*interface)?;
    let view_info = interface.views.get(view_name).unwrap();

    let port_dirs = view_info.port_dirs.as_ref_ok()?;

    let mut declared_ports = vec![];
    let mut singles = vec![];

    for (port_index, (_, port)) in enumerate(&interface.ports) {
        let name = format!("{}_{}", id.string, port.id.string);

        let direction = port_dirs[port_index].1;
        let ty = port.ty.as_ref_ok()?;

        let ir_port = ports_ir.push(IrPortInfo {
            name: name.clone(),
            direction: direction.inner,
            ty: ty.inner.as_ir(),
            debug_span: id.span,
            debug_info_ty: ty.inner.to_diagnostic_string(),
            debug_info_domain: domain.inner.to_diagnostic_string(ctx),
        });
        let port = ctx.ports.push(PortInfo {
            span: id.span,
            name,
            direction,
            domain: domain.map_inner(PortDomain::Kind),
            ty: ty.clone(),
            ir: ir_port,
        });

        declared_ports.push(port);

        let single = next_single();
        singles.push(single);
        port_to_single.insert_first(port, single);
    }

    let port_interface = ctx.port_interfaces.push(PortInterfaceInfo {
        id: id.clone(),
        view: view.clone(),
        domain,
        ports: declared_ports,
    });
    let connector_domain = domain.map_inner(|d| d.map_signal(|p| p.map_inner(|p| *port_to_single.get(&p).unwrap())));
    connectors.push(ConnectorInfo {
        id: id.clone(),
        kind: ConnectorKind::Interface {
            domain: connector_domain,
            view,
            singles,
        },
    });

    Ok(ScopedEntry::Named(NamedValue::PortInterface(port_interface)))
}

struct BodyElaborationContext<'c, 'a, 's> {
    ctx: &'c mut CompileItemContext<'a, 's>,

    ir_ports: Arena<IrPort, IrPortInfo>,
    ir_wires: Arena<IrWire, IrWireInfo>,
    ir_registers: Arena<IrRegister, IrRegisterInfo>,
    drivers: Drivers,

    register_initial_values: IndexMap<Register, Result<Spanned<CompileValue>, ErrorGuaranteed>>,
    out_port_register_connections: IndexMap<Port, Register>,

    pass_1_next_statement_index: usize,
    clocked_block_statement_index_to_process_index: IndexMap<usize, usize>,
    children: Vec<Child>,
    delayed_error: Result<(), ErrorGuaranteed>,
}

enum Child {
    Finished(IrModuleChild),
    Clocked(ChildClockedProcess),
}

struct ChildClockedProcess {
    locals: IrVariables,
    clock_signal: Spanned<DomainSignal>,
    clock_block: IrBlock,
    reset: Option<ChildClockedProcessReset>,
}

impl ChildClockedProcess {
    pub fn finish(self, ctx: &mut CompileItemContext) -> IrClockedProcess {
        let ChildClockedProcess {
            locals,
            clock_signal,
            clock_block,
            reset,
        } = self;

        let (clock_block, async_reset_signal_and_block) = match reset {
            None => {
                // nothing to do here
                (clock_block, None)
            }
            Some(reset) => {
                let ChildClockedProcessReset {
                    kind,
                    signal: reset_signal,
                    reg_inits,
                } = reset;
                let reset_signal_ir = reset_signal.map_inner(|s| ctx.domain_signal_to_ir(s));

                // create reset statements
                let mut reset_statements = vec![];
                for init in reg_inits {
                    let ExtraRegisterInit { span, reg, init } = init;
                    let stmt = IrStatement::Assign(IrAssignmentTarget::register(reg), init);
                    reset_statements.push(Spanned::new(span, stmt));
                }
                let reset_block = IrBlock {
                    statements: reset_statements,
                };

                // put them in the right place
                match kind.inner {
                    ResetKind::Async => {
                        // async reset becomes a separate block, sensitive to the signal
                        (clock_block, Some((reset_signal_ir, reset_block)))
                    }
                    ResetKind::Sync => {
                        // async reset becomes an extra if branch inside the clocked block
                        let if_stmt = IrIfStatement {
                            condition: reset_signal_ir.inner,
                            then_block: reset_block,
                            else_block: Some(clock_block),
                        };
                        let root_block = IrBlock {
                            statements: vec![Spanned::new(reset_signal.span, IrStatement::If(if_stmt))],
                        };
                        (root_block, None)
                    }
                }
            }
        };

        IrClockedProcess {
            locals,
            clock_signal: clock_signal.map_inner(|s| ctx.domain_signal_to_ir(s)),
            clock_block,
            async_reset_signal_and_block,
        }
    }
}

struct ChildClockedProcessReset {
    kind: Spanned<ResetKind>,
    signal: Spanned<DomainSignal>,
    reg_inits: Vec<ExtraRegisterInit>,
}

#[derive(Debug)]
pub struct ExtraRegisterInit {
    pub span: Span,
    pub reg: IrRegister,
    pub init: IrExpression,
}

impl BodyElaborationContext<'_, '_, '_> {
    fn elaborate_block(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        signals: &mut SignalsInScope,
        block: &Block<ModuleStatement>,
        is_root_block: bool,
    ) {
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        let mut scope_inner = Scope::new_child(block.span, scope);
        let mut vars_inner = VariableValues::new_child(vars);

        let signals_len_before = signals.len();
        self.pass_0_declarations(&mut scope_inner, &mut vars_inner, signals, block, is_root_block);
        let signals_len_after = signals.len();

        self.pass_1_processes(&scope_inner, &vars_inner, signals, block);
        assert_eq!(signals_len_after, signals.len());
        signals.truncate(signals_len_before);
    }

    fn pass_0_declarations(
        &mut self,
        scope: &mut Scope,
        vars: &mut VariableValues,
        signals: &mut SignalsInScope,
        body: &Block<ModuleStatement>,
        is_root_block: bool,
    ) {
        let diags = self.ctx.refs.diags;

        let Block { span: _, statements } = body;
        for stmt in statements {
            match &stmt.inner {
                // non declarations, skip
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
                // these will get their own two-phase process during the second pass
                ModuleStatementKind::If(_) => {}
                ModuleStatementKind::For(_) => {}
                ModuleStatementKind::Block(_) => {}
                ModuleStatementKind::ConstBlock(_) => {}
                // declarations
                ModuleStatementKind::CommonDeclaration(decl) => {
                    self.ctx.eval_and_declare_declaration(scope, vars, decl)
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let reg = self.elaborate_module_declaration_reg(scope, vars, decl);
                    let entry = reg.map(|reg_init| {
                        let RegisterInit { reg, init } = reg_init;
                        self.register_initial_values.insert_first(reg, init);
                        self.drivers.reg_drivers.insert_first(reg, IndexMap::new());
                        signals.push(Spanned::new(decl.id.span(), Signal::Register(reg)));
                        ScopedEntry::Named(NamedValue::Register(reg))
                    });

                    scope.maybe_declare(diags, decl.id.as_ref(), entry);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let (wire, process) = result_pair_split(self.elaborate_module_declaration_wire(scope, vars, decl));
                    let process = process.transpose();

                    if let Ok(wire) = wire {
                        let mut drivers = IndexMap::new();
                        if let Some(value) = &decl.value {
                            drivers.insert_first(Driver::WireDeclaration, value.span);
                        }
                        self.drivers.wire_drivers.insert_first(wire, drivers);
                    }

                    if let Some(process) = process {
                        match process {
                            Ok(process) => self
                                .children
                                .push(Child::Finished(IrModuleChild::CombinatorialProcess(process))),
                            Err(e) => self.delayed_error = Err(e),
                        }
                    }

                    let entry = wire.map(|wire| ScopedEntry::Named(NamedValue::Wire(wire)));
                    scope.maybe_declare(diags, decl.id.as_ref(), entry);
                    if let Ok(wire) = wire {
                        signals.push(Spanned::new(decl.id.span(), Signal::Wire(wire)));
                    }
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    // TODO check if this still works in nested blocks, maybe we should only allow this at the top level
                    //   no, we can't really ban this, we need conditional makers for eg. conditional ports
                    // declare register that shadows the outer port, which is exactly what we want
                    match self.elaborate_module_declaration_reg_out_port(scope, vars, decl, is_root_block) {
                        Ok((port, reg_init)) => {
                            let port_drivers = self.drivers.output_port_drivers.get_mut(&port).unwrap();
                            assert!(port_drivers.is_empty());
                            port_drivers.insert_first(Driver::OutputPortConnectionToReg, decl.span);

                            self.drivers.reg_drivers.insert_first(reg_init.reg, IndexMap::new());
                            self.register_initial_values.insert_first(reg_init.reg, reg_init.init);
                            self.out_port_register_connections.insert_first(port, reg_init.reg);

                            let entry = Ok(ScopedEntry::Named(NamedValue::Register(reg_init.reg)));
                            scope.declare(diags, &decl.id, entry);
                            signals.push(Spanned::new(decl.id.span, Signal::Register(reg_init.reg)));
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
            }
        }
    }

    fn pass_1_processes(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        signals: &mut SignalsInScope,
        body: &Block<ModuleStatement>,
    ) {
        let diags = self.ctx.refs.diags;
        let Block { span: _, statements } = body;

        for stmt in statements {
            let stmt_index = self.pass_1_next_statement_index;
            self.pass_1_next_statement_index += 1;

            match &stmt.inner {
                // control flow
                ModuleStatementKind::Block(block) => {
                    self.elaborate_block(scope, vars, signals, block, false);
                }
                ModuleStatementKind::ConstBlock(block) => {
                    let mut ctx_inner = CompileTimeExpressionContext {
                        span: block.span,
                        reason: "const block".to_owned(),
                    };
                    let mut vars_inner = VariableValues::new_child(vars);
                    let _: Result<(), ErrorGuaranteed> =
                        match self.ctx.elaborate_block(&mut ctx_inner, scope, &mut vars_inner, block) {
                            Ok(((), block_end)) => block_end.unwrap_outside_function_and_loop(diags),
                            Err(e) => Err(e),
                        };
                }
                ModuleStatementKind::If(if_stmt) => {
                    match self.ctx.compile_if_statement_choose_block(scope, vars, if_stmt) {
                        Ok(block) => {
                            if let Some(block) = block {
                                self.elaborate_block(scope, vars, signals, block, false);
                            }
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
                ModuleStatementKind::For(for_stmt) => {
                    let for_stmt = Spanned::new(stmt.span, for_stmt);
                    match self.elaborate_for(scope, vars, signals, for_stmt) {
                        Ok(()) => {}
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
                // declarations, already handled
                ModuleStatementKind::CommonDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                // blocks, handle now
                ModuleStatementKind::CombinatorialBlock(block) => {
                    let mut vars_inner = self.new_vars_for_process(vars, signals);

                    let ir_process = self.elaborate_combinatorial_block(&mut vars_inner, scope, stmt_index, block);
                    match ir_process {
                        Ok(ir_process) => self
                            .children
                            .push(Child::Finished(IrModuleChild::CombinatorialProcess(ir_process))),
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
                ModuleStatementKind::ClockedBlock(block) => {
                    let mut vars_inner = self.new_vars_for_process(vars, signals);
                    let ir_process = self.elaborate_clocked_block(&mut vars_inner, scope, stmt_index, block);

                    match ir_process {
                        Ok(ir_process) => {
                            let child_index = self.children.len();
                            self.children.push(Child::Clocked(ir_process));
                            self.clocked_block_statement_index_to_process_index
                                .insert_first(stmt_index, child_index);
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
                ModuleStatementKind::Instance(instance) => {
                    let mut vars_inner = self.new_vars_for_process(vars, signals);
                    let instance_ir = self.elaborate_instance(scope, &mut vars_inner, stmt_index, instance);

                    match instance_ir {
                        Ok(instance_ir) => {
                            self.children
                                .push(Child::Finished(IrModuleChild::ModuleInstance(instance_ir)));
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
            }
        }
    }

    fn elaborate_combinatorial_block(
        &mut self,
        vars: &mut VariableValues,
        scope: &Scope,
        stmt_index: usize,
        block: &CombinatorialBlock,
    ) -> Result<IrCombinatorialProcess, ErrorGuaranteed> {
        let &CombinatorialBlock {
            span: _,
            span_keyword,
            ref block,
        } = block;
        let diags = self.ctx.refs.diags;

        let block_kind = BlockKind::Combinatorial { span_keyword };
        let mut report_assignment = |target: Spanned<Signal>| {
            self.drivers
                .report_assignment(diags, Driver::CombinatorialBlock(stmt_index), target)
        };
        let mut ctx = IrBuilderExpressionContext::new(block_kind, &mut report_assignment);

        let (ir_block, end) = self.ctx.elaborate_block(&mut ctx, scope, vars, block)?;
        let ir_variables = ctx.finish();
        end.unwrap_outside_function_and_loop(diags)?;

        Ok(IrCombinatorialProcess {
            locals: ir_variables,
            block: ir_block,
        })
    }

    fn elaborate_clocked_block(
        &mut self,
        vars: &mut VariableValues,
        scope: &Scope,
        stmt_index: usize,
        block: &ClockedBlock,
    ) -> Result<ChildClockedProcess, ErrorGuaranteed> {
        let &ClockedBlock {
            span: _,
            span_keyword,
            span_domain,
            ref clock,
            ref reset,
            ref block,
        } = block;
        let diags = self.ctx.refs.diags;

        // eval signals
        let clock = self.ctx.eval_expression_as_domain_signal(scope, clock);
        let reset = reset
            .as_ref()
            .map(|reset| {
                reset
                    .as_ref()
                    .map_inner(|reset| {
                        let &ClockedBlockReset { kind, ref signal } = reset;
                        let signal = self.ctx.eval_expression_as_domain_signal(scope, signal)?;
                        Ok(ClockedBlockReset { kind, signal })
                    })
                    .transpose()
            })
            .transpose();
        let (clock, reset) = result_pair(clock, reset)?;

        // check reset domain
        if let Some(reset) = &reset {
            let ClockedBlockReset { kind, signal } = reset.inner;
            match kind.inner {
                // nothing to check for async resets
                // TODO check that the posedge is sync to the clock
                ResetKind::Async => {}
                // check that the reset is sync to the clock
                ResetKind::Sync => {
                    let target = clock.map_inner(|s| ValueDomain::Sync(SyncDomain { clock: s, reset: None }));
                    let source = Spanned::new(reset.span, signal.inner.signal.domain(self.ctx).inner);
                    self.ctx
                        .check_valid_domain_crossing(span_domain, target, source, "sync reset")?;
                }
            }
        };

        // for the domain, a sync reset disappears and we only keep the clock
        let domain = SyncDomain {
            clock: clock.inner,
            reset: reset.as_ref().and_then(|reset| match reset.inner.kind.inner {
                ResetKind::Async => Some(reset.inner.signal.inner),
                ResetKind::Sync => None,
            }),
        };
        let domain = Spanned::new(span_domain, domain);

        let mut elaborate_block = |extra_regs| {
            let block_kind = BlockKind::Clocked {
                span_keyword,
                domain,
                ir_registers: &mut self.ir_registers,
                extra_registers: extra_regs,
            };
            let mut report_assignment = |target: Spanned<Signal>| {
                self.drivers
                    .report_assignment(diags, Driver::ClockedBlock(stmt_index), target)
            };
            let mut ctx = IrBuilderExpressionContext::new(block_kind, &mut report_assignment);

            let (ir_block, end) = self.ctx.elaborate_block(&mut ctx, scope, vars, block)?;
            let ir_variables = ctx.finish();
            end.unwrap_outside_function_and_loop(diags)?;
            Ok((ir_block, ir_variables))
        };

        let ((clock_block, locals), reset) = match reset {
            None => (elaborate_block(ExtraRegisters::NoReset)?, None),
            Some(reset) => {
                let mut reg_inits = vec![];
                let block_result = elaborate_block(ExtraRegisters::WithReset(&mut reg_inits))?;

                let reset = ChildClockedProcessReset {
                    kind: reset.inner.kind,
                    signal: reset.inner.signal,
                    reg_inits,
                };

                (block_result, Some(reset))
            }
        };

        Ok(ChildClockedProcess {
            locals,
            clock_signal: clock,
            clock_block,
            reset,
        })
    }

    /// We want to create new [VariableValues] for each process, to be able to reset the variables that exist and to reset the version counters.
    /// TODO is any of that actually necessary?
    fn new_vars_for_process<'p>(
        &self,
        vars_parent: &'p VariableValues<'p>,
        signals_in_scope: &[Spanned<Signal>],
    ) -> VariableValues<'p> {
        let mut vars = VariableValues::new_child(vars_parent);
        for &signal in signals_in_scope {
            // we're creating a new instance of variables here, and we know none of these signals are duplicate,
            //   so we can unwrap
            let _ = vars.signal_new(self.ctx.refs.diags, signal);
        }
        vars
    }

    fn elaborate_for(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        signals: &mut SignalsInScope,
        for_stmt: Spanned<&ForStatement<ModuleStatement>>,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;
        let &ForStatement {
            span_keyword,
            ref index,
            ref index_ty,
            ref iter,
            ref body,
        } = for_stmt.inner;
        let iter_span = iter.span;

        let mut vars_inner = VariableValues::new_child(vars);
        let index_ty = index_ty
            .as_ref()
            .map(|index_ty| self.ctx.eval_expression_as_ty(scope, &mut vars_inner, index_ty));

        let mut ctx = CompileTimeExpressionContext {
            span: for_stmt.span,
            reason: "module-level for statement".to_string(),
        };
        let iter = self
            .ctx
            .eval_expression_as_for_iterator(&mut ctx, &mut (), scope, &mut vars_inner, iter);

        let index_ty = index_ty.transpose()?;
        let iter = iter?;

        // TODO allow break?
        for index_value in iter {
            self.ctx.refs.check_should_stop(span_keyword)?;
            let index_value = index_value.to_maybe_compile(&mut self.ctx.large);

            if let Some(index_ty) = &index_ty {
                let index_value_spanned = Spanned::new(iter_span, &index_value);
                check_type_contains_value(
                    diags,
                    TypeContainsReason::ForIndexType(index_ty.span),
                    &index_ty.inner,
                    index_value_spanned,
                    false,
                    false,
                )?;
            }

            let mut scope_inner = Scope::new_child(index.span().join(body.span), scope);
            let var = vars_inner.var_new_immutable_init(
                &mut self.ctx.variables,
                index.clone(),
                span_keyword,
                Ok(index_value),
            );
            scope_inner.maybe_declare(diags, index.as_ref(), Ok(ScopedEntry::Named(NamedValue::Variable(var))));

            self.elaborate_block(&scope_inner, &vars_inner, signals, body, false);
        }

        Ok(())
    }

    fn elaborate_instance(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        stmt_index: usize,
        instance: &ast::ModuleInstance,
    ) -> Result<IrModuleInstance, ErrorGuaranteed> {
        let refs = self.ctx.refs;
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;

        let ast::ModuleInstance {
            name,
            span_keyword,
            module,
            port_connections,
        } = instance;

        let elaborated_module = self.ctx.eval_expression_as_module(scope, vars, *span_keyword, module)?;
        let &ElaboratedModuleInfo {
            module_ast,
            module_ir,
            ref connectors,
        } = self.ctx.refs.shared.elaborated_module_info(elaborated_module)?;

        let module_ast = &refs.fixed.parsed[module_ast];

        // eval and check port connections
        // TODO use function parameter matching for ports too?
        //   we need at least reordering and proper error messages
        if connectors.len() != port_connections.inner.len() {
            let diag = Diagnostic::new("mismatched port connections for module instance")
                .add_error(
                    port_connections.span,
                    format!("ports connected here, got {} connections", port_connections.inner.len()),
                )
                .add_info(
                    module_ast.ports.span,
                    format!("module ports declared here, expected {} connections", connectors.len()),
                )
                .finish();
            return Err(diags.report(diag));
        }

        let mut any_port_err = Ok(());

        let mut single_to_signal = IndexMap::new();
        let mut ir_connections = vec![];

        for (connector, connection) in zip_eq(connectors.keys(), &port_connections.inner) {
            match self.elaborate_instance_port_connection(
                scope,
                vars,
                stmt_index,
                connectors,
                &single_to_signal,
                connector,
                connection,
            ) {
                Ok(connections) => {
                    for (single, signal, ir_connection) in connections {
                        single_to_signal.insert_first(single, signal);
                        ir_connections.push(ir_connection);
                    }
                }
                Err(e) => {
                    any_port_err = Err(e);
                }
            }
        }

        any_port_err?;

        Ok(IrModuleInstance {
            name: name.as_ref().map(|name| name.string.clone()),
            module: module_ir,
            port_connections: ir_connections,
        })
    }

    fn elaborate_instance_port_connection(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        stmt_index: usize,
        connectors: &ArenaConnectors,
        prev_single_to_signal: &IndexMap<ConnectorSingle, ConnectionSignal>,
        connector: Connector,
        connection: &Spanned<PortConnection>,
    ) -> Result<Vec<(ConnectorSingle, ConnectionSignal, Spanned<IrPortConnection>)>, ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;

        let PortConnection {
            id: connection_id,
            expr: value_expr,
        } = &connection.inner;
        let ConnectorInfo { id: connector_id, kind } = &connectors[connector];

        // check id match
        if connector_id.string != connection_id.string {
            let diag = Diagnostic::new("mismatched port connection")
                .add_error(
                    connection_id.span,
                    format!("connected here to `{}`", connection_id.string),
                )
                .add_info(
                    connector_id.span,
                    format!("expected connection to `{}`", connector_id.string),
                )
                .footer(Level::Note, "port connection re-ordering is not yet supported")
                .finish();
            return Err(diags.report(diag));
        }

        // replace signals that are earlier ports with their connected value
        let map_domain_kind = |domain_span: Span, domain: DomainKind<Polarized<ConnectorSingle>>| {
            Ok(match domain {
                DomainKind::Const => DomainKind::Const,
                DomainKind::Async => DomainKind::Async,
                DomainKind::Sync(sync) => DomainKind::Sync(sync.try_map_signal(|raw_port| {
                    let mapped_port = match prev_single_to_signal.get(&raw_port.signal) {
                        None => throw!(
                            diags.report_internal_error(connection.span, "failed to get signal for previous port")
                        ),
                        Some(&ConnectionSignal::Dummy(dummy_span)) => {
                            let diag = Diagnostic::new_todo(
                                "dummy port connections that are used in the domain of other ports",
                            )
                            .add_error(dummy_span, "port connected to dummy here")
                            .add_info(domain_span, "port used in a domain here")
                            .finish();
                            throw!(diags.report(diag))
                        }
                        Some(&ConnectionSignal::Expression(expr_span)) => {
                            let diag = Diagnostic::new_todo(
                                "expression port connections that are used in the domain of other ports",
                            )
                            .add_error(expr_span, "port connected to expression here")
                            .add_info(domain_span, "port used in a domain here")
                            .finish();
                            throw!(diags.report(diag))
                        }
                        Some(&ConnectionSignal::Signal(signal)) => Ok(signal),
                    }?;
                    Ok(Polarized {
                        signal: mapped_port.signal,
                        inverted: mapped_port.inverted ^ raw_port.inverted,
                    })
                })?),
            })
        };

        // always try to evaluate as signal for domain replacing purposes
        let signal = match &value_expr.inner {
            ExpressionKind::Dummy => ConnectionSignal::Dummy(value_expr.span),
            _ => match self.ctx.try_eval_expression_as_domain_signal(scope, value_expr, |_| ()) {
                Ok(signal) => ConnectionSignal::Signal(signal.inner),
                Err(Either::Left(())) => ConnectionSignal::Expression(value_expr.span),
                Err(Either::Right(e)) => throw!(e),
            },
        };

        // evaluate the connection differently depending on the port direction
        match kind {
            &ConnectorKind::Port {
                direction,
                domain,
                ref ty,
                single,
            } => {
                let domain = domain
                    .map_inner(|d| match d {
                        PortDomain::Clock => Ok(ValueDomain::Clock),
                        PortDomain::Kind(kind) => {
                            Ok(ValueDomain::from_domain_kind(map_domain_kind(domain.span, kind)?))
                        }
                    })
                    .transpose()?;

                let ir_connection = match direction.inner {
                    PortDirection::Input => {
                        // better dummy port error message
                        if let ExpressionKind::Dummy = value_expr.inner {
                            return Err(diags.report_simple(
                                "dummy connections are only allowed for output ports",
                                value_expr.span,
                                "used dummy connection on input port here",
                            ));
                        }

                        // eval expr
                        // TODO this should not be an internal, this is actually possible now with block expressions
                        let mut report_assignment =
                            report_assignment_internal_error(diags, "module instance port connection");
                        let block_kind = BlockKind::InstancePortConnection {
                            span_connection: connection.span,
                        };
                        let mut ctx = IrBuilderExpressionContext::new(block_kind, &mut report_assignment);
                        let mut ctx_block = ctx.new_ir_block();

                        let mut vars_inner = VariableValues::new_child(vars);
                        let connection_value = self.ctx.eval_expression(
                            &mut ctx,
                            &mut ctx_block,
                            scope,
                            &mut vars_inner,
                            &ty.inner.as_type(),
                            value_expr,
                        )?;

                        // check type
                        let reason = TypeContainsReason::InstancePortInput {
                            span_connection_port_id: connection_id.span,
                            span_port_ty: ty.span,
                        };
                        let check_ty = check_type_contains_value(
                            diags,
                            reason,
                            &ty.inner.as_type(),
                            connection_value.as_ref(),
                            true,
                            false,
                        );

                        // check domain
                        let target_domain = Spanned {
                            span: connection_id.span,
                            inner: domain.inner,
                        };
                        let source_domain = connection_value.as_ref().map_inner(|v| v.domain());
                        let check_domain = self.ctx.check_valid_domain_crossing(
                            connection.span,
                            target_domain,
                            source_domain,
                            "input port connection",
                        );

                        check_ty?;
                        check_domain?;

                        // convert value to ir
                        let connection_value_ir_raw = connection_value
                            .as_ref()
                            .map_inner(|v| {
                                Ok(
                                    v.as_hardware_value(diags, &mut self.ctx.large, value_expr.span, &ty.inner)?
                                        .expr,
                                )
                            })
                            .transpose()?;

                        // build extra wire and process if necessary
                        let connection_value_ir = if !ctx_block.statements.is_empty()
                            || connection_value_ir_raw.inner.contains_variable(&self.ctx.large)
                        {
                            let extra_ir_wire = self.ir_wires.push(IrWireInfo {
                                ty: ty.inner.as_ir(),
                                debug_info_id: MaybeIdentifier::Identifier(connector_id.clone()),
                                debug_info_ty: ty.inner.clone(),
                                debug_info_domain: connection_value.inner.domain().to_diagnostic_string(self.ctx),
                            });

                            ctx_block.statements.push(Spanned {
                                span: connection.span,
                                inner: IrStatement::Assign(
                                    IrAssignmentTarget::wire(extra_ir_wire),
                                    connection_value_ir_raw.inner,
                                ),
                            });
                            let process = IrCombinatorialProcess {
                                locals: ctx.finish(),
                                block: ctx_block,
                            };
                            self.children
                                .push(Child::Finished(IrModuleChild::CombinatorialProcess(process)));

                            IrExpression::Wire(extra_ir_wire)
                        } else {
                            connection_value_ir_raw.inner
                        };

                        IrPortConnection::Input(Spanned {
                            span: value_expr.span,
                            inner: connection_value_ir,
                        })
                    }
                    PortDirection::Output => {
                        // eval expr as dummy, wire or port
                        let build_error = || {
                            diags.report_simple(
                                "output port must be connected to wire or port",
                                value_expr.span,
                                "other value",
                            )
                        };

                        match &value_expr.inner {
                            ExpressionKind::Dummy => IrPortConnection::Output(None),
                            ExpressionKind::Id(id) => {
                                let named = self.ctx.eval_id(scope, id)?;

                                let (signal_ir, signal_target, signal_ty, &signal_domain) = match named.inner {
                                    EvaluatedId::Named(NamedValue::Wire(wire)) => {
                                        let wire_info = &self.ctx.wires[wire];
                                        (
                                            IrWireOrPort::Wire(wire_info.ir),
                                            Signal::Wire(wire),
                                            &wire_info.ty,
                                            &wire_info.domain,
                                        )
                                    }
                                    EvaluatedId::Named(NamedValue::Port(port)) => {
                                        let port_info = &self.ctx.ports[port];
                                        (
                                            IrWireOrPort::Port(port_info.ir),
                                            Signal::Port(port),
                                            &port_info.ty,
                                            &port_info.domain.map_inner(ValueDomain::from_port_domain),
                                        )
                                    }
                                    _ => throw!(build_error()),
                                };

                                // check type
                                let mut any_err = Ok(());
                                let reason = TypeContainsReason::InstancePortOutput {
                                    span_connection_signal_id: value_expr.span,
                                    span_signal_ty: signal_ty.span,
                                };
                                any_err = any_err.and(check_type_contains_type(
                                    diags,
                                    reason,
                                    &signal_ty.inner.as_type(),
                                    Spanned {
                                        span: connection_id.span,
                                        inner: &ty.inner.as_type(),
                                    },
                                    false,
                                ));

                                // check domain
                                any_err = any_err.and(self.ctx.check_valid_domain_crossing(
                                    connection.span,
                                    signal_domain,
                                    domain,
                                    "output port connection",
                                ));

                                // report driver
                                let driver = Driver::InstancePortConnection(stmt_index);
                                let target = Spanned::new(named.span, signal_target);
                                self.drivers.report_assignment(diags, driver, target)?;

                                // success, build connection
                                any_err?;
                                IrPortConnection::Output(Some(signal_ir))
                            }
                            _ => throw!(build_error()),
                        }
                    }
                };

                let spanned_ir_connection = Spanned {
                    span: connection.span,
                    inner: ir_connection,
                };
                Ok(vec![(single, signal, spanned_ir_connection)])
            }
            ConnectorKind::Interface {
                domain: connector_domain,
                view: connector_view,
                singles: connector_singles,
            } => {
                let connector_domain = connector_domain
                    .map_inner(|d| {
                        Ok(ValueDomain::from_domain_kind(map_domain_kind(
                            connector_domain.span,
                            d,
                        )?))
                    })
                    .transpose()?;

                // eval expr
                // TODO this should not be an internal, this is actually possible now with block expressions
                let mut report_assignment = report_assignment_internal_error(diags, "module instance port connection");
                let block_kind = BlockKind::InstancePortConnection {
                    span_connection: connection.span,
                };
                let mut ctx = IrBuilderExpressionContext::new(block_kind, &mut report_assignment);
                let mut ctx_block = ctx.new_ir_block();

                let mut vars_inner = VariableValues::new_child(vars);
                let value = self.ctx.eval_expression_inner(
                    &mut ctx,
                    &mut ctx_block,
                    scope,
                    &mut vars_inner,
                    &Type::Any,
                    value_expr,
                )?;

                // unwrap interface
                let value_interface = match value {
                    ValueInner::PortInterface(port_interface) => port_interface,
                    ValueInner::Value(_) => {
                        let diag = Diagnostic::new("expected interface value")
                            .add_error(value_expr.span, "got non-interface expression")
                            .add_info(connector_id.span, "port defined as interface here")
                            .finish();
                        return Err(diags.report(diag));
                    }
                };
                let value_info = &self.ctx.port_interfaces[value_interface];

                // check interface match (including generics)
                if value_info.view.interface != connector_view.interface {
                    let diag = Diagnostic::new("interface mismatch")
                        .add_error(value_expr.span, "got mismatching interface")
                        .add_info(connector_id.span, "expected interface set here")
                        .finish();
                    return Err(diags.report(diag));
                }

                // check directions and build connections
                let interface_info = self
                    .ctx
                    .refs
                    .shared
                    .elaborated_interface_info(connector_view.interface)?;
                let view_info = interface_info.views.get(&connector_view.view).unwrap();
                let value_view_info = interface_info.views.get(&value_info.view.view).unwrap();

                let mut any_input = false;
                let mut any_output = false;

                let mut result_connections = vec![];

                for i in 0..interface_info.ports.len() {
                    let (_, connector_dir) = &view_info.port_dirs.as_ref_ok()?[i];
                    let (_, value_dir) = &value_view_info.port_dirs.as_ref_ok()?[i];
                    let value_port = value_info.ports[i];
                    let value_port_info = &self.ctx.ports[value_port];
                    let value_port_ir = value_port_info.ir;

                    // check dir
                    if connector_dir.inner != value_dir.inner {
                        let diag = Diagnostic::new(format!(
                            "direction mismatch for interface port `{}`",
                            interface_info.ports[i].id.string
                        ))
                        .add_info(
                            connection_id.span,
                            format!("expected direction {}", connector_dir.inner.diagnostic_string()),
                        )
                        .add_error(
                            value_expr.span,
                            format!("got direction {}", value_dir.inner.diagnostic_string()),
                        )
                        .add_info(connector_dir.span, "expected direction set here")
                        .add_info(value_dir.span, "actual direction set here")
                        .finish();
                        return Err(diags.report(diag));
                    }
                    let dir = connector_dir.inner;

                    // build connection
                    let ir_connection = match dir {
                        PortDirection::Input => {
                            any_input = true;

                            let ir_expr = IrExpression::Port(value_port_ir);
                            IrPortConnection::Input(Spanned::new(value_expr.span, ir_expr))
                        }
                        PortDirection::Output => {
                            any_output = true;

                            // report driver
                            let driver = Driver::InstancePortConnection(stmt_index);
                            let target = Spanned::new(value_expr.span, Signal::Port(value_port));
                            self.drivers.report_assignment(diags, driver, target)?;

                            IrPortConnection::Output(Some(IrWireOrPort::Port(value_port_ir)))
                        }
                    };

                    // build signal
                    let signal = ConnectionSignal::Signal(Polarized {
                        inverted: false,
                        signal: Signal::Port(value_port),
                    });
                    result_connections.push((
                        connector_singles[i],
                        signal,
                        Spanned::new(connection.span, ir_connection),
                    ))
                }

                // check domains
                let value_domain = value_info
                    .domain
                    .map_inner(|d| ValueDomain::from_domain_kind(d.map_signal(|s| s.map_inner(Signal::Port))));
                let mut any_err_domain = Ok(());
                if any_input {
                    let r = self.ctx.check_valid_domain_crossing(
                        connection.span,
                        value_domain,
                        connector_domain,
                        "interface connection with input port",
                    );
                    any_err_domain = any_err_domain.and(r);
                }
                if any_output {
                    let r = self.ctx.check_valid_domain_crossing(
                        connection.span,
                        connector_domain,
                        value_domain,
                        "interface connection with output port",
                    );
                    any_err_domain = any_err_domain.and(r);
                }
                any_err_domain?;

                Ok(result_connections)
            }
        }
    }

    fn pass_2_check_drivers_and_populate_resets(&mut self) -> Result<(), ErrorGuaranteed> {
        // check exactly one valid driver for each signal
        // (all signals have been added the drivers in the first pass, so this also catches signals without any drivers)
        let Drivers {
            output_port_drivers,
            reg_drivers,
            wire_drivers,
        } = &self.drivers;
        let mut any_err = Ok(());

        // wire-like signals, just check
        for (&port, drivers) in output_port_drivers {
            let port_info = &self.ctx.ports[port];
            any_err =
                any_err.and(self.check_drivers_for_port_or_wire("port", &port_info.name, port_info.span, drivers));
        }
        for (&wire, drivers) in wire_drivers {
            let wire_info = &self.ctx.wires[wire];
            any_err = any_err.and(self.check_drivers_for_port_or_wire(
                "wire",
                wire_info.id.string(),
                wire_info.id.span(),
                drivers,
            ));
        }

        // registers: check and collect resets
        for (&reg, drivers) in reg_drivers {
            let reg_info = &self.ctx.registers[reg];
            let decl_span = reg_info.id.span();

            any_err = any_err.and(self.check_drivers_for_reg(decl_span, drivers));

            // TODO allow zero drivers for registers, just turn them into wires with the init expression as the value
            //  (still emit a warning)
            let driver = self.check_exactly_one_driver("register", reg_info.id.string(), decl_span, drivers);
            let maybe_err = match driver {
                Err(e) => Err(e),
                Ok((driver, first_span)) => pull_register_init_into_process(
                    self.ctx,
                    &self.clocked_block_statement_index_to_process_index,
                    &mut self.children,
                    &self.register_initial_values,
                    reg,
                    driver,
                    first_span,
                ),
            };
            any_err = any_err.and(maybe_err);
        }

        any_err
    }

    fn check_drivers_for_port_or_wire(
        &self,
        kind: &str,
        name: &str,
        decl_span: Span,
        drivers: &IndexMap<Driver, Span>,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;

        let mut any_err = Ok(());

        for (&driver, &span) in drivers {
            match driver.kind() {
                DriverKind::ClockedBlock => {
                    let diag = Diagnostic::new(format!("{kind} cannot be driven by clocked block"))
                        .add_error(span, "driven incorrectly here")
                        .add_info(decl_span, format!("`{name}` declared here"))
                        .footer(
                            Level::Help,
                            format!("either drive the {kind} from a combinatorial block or mark it as a register"),
                        )
                        .finish();
                    any_err = Err(diags.report(diag));
                }
                DriverKind::WiredConnection => {
                    // correct, do nothing
                }
            }
        }

        any_err = any_err.and(
            self.check_exactly_one_driver(kind, name, decl_span, drivers)
                .map(|_| ()),
        );
        any_err
    }

    fn check_drivers_for_reg(&self, decl_span: Span, drivers: &IndexMap<Driver, Span>) -> Result<(), ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;

        let mut any_err = Ok(());

        for (&driver, &span) in drivers {
            match driver.kind() {
                DriverKind::ClockedBlock => {
                    // correct, do nothing
                }
                DriverKind::WiredConnection => {
                    let diag = Diagnostic::new("register can only be driven by clocked block")
                        .add_error(span, "driven incorrectly here")
                        .add_info(decl_span, "declared here")
                        .footer(
                            Level::Help,
                            "drive the register from a clocked block or turn it into a wire",
                        )
                        .finish();
                    any_err = Err(diags.report(diag));
                }
            }
        }

        any_err
    }

    fn check_exactly_one_driver(
        &self,
        kind: &str,
        name: &str,
        decl_span: Span,
        drivers: &IndexMap<Driver, Span>,
    ) -> Result<(Driver, Span), ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;

        match drivers.len() {
            0 => {
                let diag = Diagnostic::new(format!("{kind} `{name}` has no driver"))
                    .add_error(decl_span, "declared here")
                    .finish();
                Err(diags.report(diag))
            }
            1 => {
                let (&driver, &first_span) = drivers.iter().single().unwrap();
                Ok((driver, first_span))
            }
            _ => {
                let mut diag = Diagnostic::new(format!("{kind} `{name}` has multiple drivers"));
                for (_, &span) in drivers {
                    diag = diag.add_error(span, "driven here");
                }
                let diag = diag.add_info(decl_span, "declared here").finish();
                Err(diags.report(diag))
            }
        }
    }

    fn elaborate_module_declaration_wire(
        &mut self,
        scope_body: &Scope,
        vars_body: &VariableValues,
        decl: &WireDeclaration,
    ) -> Result<(Wire, Option<IrCombinatorialProcess>), ErrorGuaranteed> {
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;

        let mut vars_inner = VariableValues::new_child(vars_body);

        let WireDeclaration {
            span: _,
            id,
            kind: domain_ty,
            value,
        } = decl;

        // evaluate domain and type
        let (domain, ty) = match &domain_ty.inner {
            WireKind::Clock => (
                Ok(Spanned {
                    span: domain_ty.span,
                    inner: ValueDomain::Clock,
                }),
                Ok(Spanned {
                    span: domain_ty.span,
                    inner: HardwareType::Clock,
                }),
            ),
            WireKind::Normal { domain, ty } => {
                let domain = ctx
                    .eval_domain(scope_body, domain)
                    .map(|d| d.map_inner(ValueDomain::from_domain_kind));
                let ty = ctx.eval_expression_as_ty_hardware(scope_body, &mut vars_inner, ty, "wire");
                (domain, ty)
            }
        };

        // evaluate value
        let value = if let Some(value) = value {
            // TODO better preserve the process order, due to pass1/pass2 the order gets a bit mixed up
            let mut report_assignment = report_assignment_internal_error(diags, "wire declaration value");
            let block_kind = BlockKind::WireValue { span_value: value.span };
            let mut ctx_expr = IrBuilderExpressionContext::new(block_kind, &mut report_assignment);
            let mut ctx_block = ctx_expr.new_ir_block();

            let expected_ty = ty.as_ref_ok().ok().map(|ty| ty.inner.as_type()).unwrap_or(Type::Any);
            let value = ctx.eval_expression(
                &mut ctx_expr,
                &mut ctx_block,
                scope_body,
                &mut vars_inner,
                &expected_ty,
                value,
            );

            let locals = ctx_expr.finish();
            value.map(|value| Some((value, locals, ctx_block)))
        } else {
            Ok(None)
        };

        let domain = domain?;
        let ty = ty?;
        let value = value?;

        let value = value
            .map(|(value, process_locals, process_block)| {
                // check type
                let reason = TypeContainsReason::Assignment {
                    span_target: id.span(),
                    span_target_ty: ty.span,
                };
                let check_ty =
                    check_type_contains_value(diags, reason, &ty.inner.as_type(), value.as_ref(), true, false);

                // check domain
                let value_domain = value.inner.domain();
                let value_domain_spanned = Spanned {
                    span: value.span,
                    inner: value_domain,
                };
                let check_domain =
                    ctx.check_valid_domain_crossing(decl.span, domain, value_domain_spanned, "value to wire");

                check_ty?;
                check_domain?;

                // convert to IR
                let value_ir = value
                    .as_ref()
                    .map_inner(|value_inner| {
                        value_inner.as_hardware_value(diags, &mut ctx.large, value.span, &ty.inner)
                    })
                    .transpose()?;
                Ok((value_ir, process_locals, process_block))
            })
            .transpose()?;

        // build wire
        let ir_wire = self.ir_wires.push(IrWireInfo {
            ty: ty.inner.as_ir(),
            debug_info_id: id.clone(),
            debug_info_ty: ty.inner.clone(),
            debug_info_domain: domain.inner.to_diagnostic_string(ctx),
        });
        let wire = ctx.wires.push(WireInfo {
            id: id.clone(),
            domain,
            ty,
            ir: ir_wire,
        });

        // append assignment to process
        let process_block = if let Some((value, process_locals, mut process_block)) = value {
            let target = IrAssignmentTarget::wire(ir_wire);
            let stmt = IrStatement::Assign(target, value.inner.expr);

            let stmt = Spanned {
                span: value.span,
                inner: stmt,
            };
            process_block.statements.push(stmt);
            Some((process_locals, process_block))
        } else {
            None
        };

        // build final process if necessary
        let process = process_block
            .filter(|(_, b)| !b.statements.is_empty())
            .map(|(locals, block)| IrCombinatorialProcess { locals, block });

        Ok((wire, process))
    }

    fn elaborate_module_declaration_reg(
        &mut self,
        scope_body: &Scope,
        vars_body: &VariableValues,
        decl: &RegDeclaration,
    ) -> Result<RegisterInit, ErrorGuaranteed> {
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;

        let mut vars_inner = VariableValues::new_child(vars_body);

        let RegDeclaration {
            span: _,
            id,
            sync,
            ty,
            init,
        } = decl;

        // evaluate
        let sync = sync
            .as_ref()
            .map_inner(|sync| ctx.eval_domain_sync(scope_body, sync))
            .transpose();
        let ty = ctx.eval_expression_as_ty_hardware(scope_body, &mut vars_inner, ty, "register");
        let init = ctx.eval_expression_as_compile(scope_body, &mut vars_inner, init.as_ref(), "register reset value");

        let sync = sync?;
        let ty = ty?;

        // check type
        let init = init.and_then(|init| {
            let reason = TypeContainsReason::Assignment {
                span_target: id.span(),
                span_target_ty: ty.span,
            };
            check_type_contains_compile_value(diags, reason, &ty.inner.as_type(), init.as_ref(), true)?;
            Ok(init)
        });

        // build register
        let ir_reg = self.ir_registers.push(IrRegisterInfo {
            ty: ty.inner.as_ir(),
            debug_info_id: id.clone(),
            debug_info_ty: ty.inner.clone(),
            debug_info_domain: sync.inner.to_diagnostic_string(ctx),
        });
        let reg = ctx.registers.push(RegisterInfo {
            id: id.clone(),
            domain: sync,
            ty,
            ir: ir_reg,
        });
        Ok(RegisterInit { reg, init })
    }

    fn elaborate_module_declaration_reg_out_port(
        &mut self,
        scope_body: &Scope,
        vars_body: &VariableValues,
        decl: &RegOutPortMarker,
        is_root_block: bool,
    ) -> Result<(Port, RegisterInit), ErrorGuaranteed> {
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;
        let RegOutPortMarker { span: _, id, init } = decl;

        if !is_root_block {
            return Err(diags.report_simple(
                "register output markers are only allowed at the top level of a module",
                decl.span,
                "register output in child block",
            ));
        }

        // find port
        let port = scope_body.find(diags, id).and_then(|port| match port.value {
            &ScopedEntry::Named(NamedValue::Port(port)) => Ok(port),
            _ => {
                let diag = Diagnostic::new("register port marker needs to be on a port")
                    .add_error(id.span, "non-port value here")
                    .add_info(port.defining_span, "declared here")
                    .finish();
                Err(diags.report(diag))
            }
        });

        // evaluate init
        let mut vars_inner = VariableValues::new_child(vars_body);
        let init = ctx.eval_expression_as_compile(scope_body, &mut vars_inner, init, "register reset value");

        let port = port?;
        let port_info = &ctx.ports[port];
        let mut direction_err = Ok(());

        // check port is output
        match port_info.direction.inner {
            PortDirection::Input => {
                let diag = Diagnostic::new("only output ports can be marked as registers")
                    .add_error(id.span, "port marked as register here")
                    .add_info(port_info.direction.span, "port declared as input here")
                    .finish();
                direction_err = Err(diags.report(diag))
            }
            PortDirection::Output => {}
        }

        // check port has sync domain
        let domain = match &port_info.domain.inner {
            PortDomain::Clock => Err("clock"),
            PortDomain::Kind(DomainKind::Async) => Err("async"),
            PortDomain::Kind(DomainKind::Const) => Err("const"),
            PortDomain::Kind(DomainKind::Sync(domain)) => Ok(domain),
        };

        let domain = match domain {
            Ok(domain) => domain,
            Err(actual) => {
                let diag = Diagnostic::new("only synchronous ports can be marked as registers")
                    .add_error(id.span, "port marked as register here")
                    .add_info(port_info.domain.span, format!("port declared as {actual} here"))
                    .finish();
                return Err(diags.report(diag));
            }
        };

        direction_err?;

        // check type
        let init = init.and_then(|init| {
            let reason = TypeContainsReason::Assignment {
                span_target: id.span,
                span_target_ty: port_info.ty.span,
            };
            check_type_contains_compile_value(diags, reason, &port_info.ty.inner.as_type(), init.as_ref(), true)?;
            Ok(init)
        });

        // build register
        let ir_reg = self.ir_registers.push(IrRegisterInfo {
            ty: port_info.ty.inner.as_ir(),
            debug_info_id: MaybeIdentifier::Identifier(id.clone()),
            debug_info_ty: port_info.ty.inner.clone(),
            debug_info_domain: domain.to_diagnostic_string(ctx),
        });
        let reg = ctx.registers.push(RegisterInfo {
            id: MaybeIdentifier::Identifier(id.clone()),
            domain: Spanned {
                span: port_info.domain.span,
                inner: domain.map_signal(|p| p.map_inner(Signal::Port)),
            },
            ty: port_info.ty.clone(),
            ir: ir_reg,
        });
        Ok((port, RegisterInit { reg, init }))
    }
}

enum ConnectionSignal {
    Signal(DomainSignal),
    Dummy(Span),
    Expression(Span),
}

#[derive(Debug)]
struct RegisterInit {
    reg: Register,
    init: Result<Spanned<CompileValue>, ErrorGuaranteed>,
}

/// The usize fields are here to keep different drivers of the same type separate for Eq and Hash,
/// so we can correctly determining whether there are multiple drivers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Driver {
    // wire-like
    WireDeclaration,
    OutputPortConnectionToReg,
    InstancePortConnection(usize),
    CombinatorialBlock(usize),

    // register-like
    ClockedBlock(usize),
}

#[derive(Debug, Copy, Clone)]
pub enum DriverKind {
    ClockedBlock,
    WiredConnection,
}

impl Driver {
    fn kind(&self) -> DriverKind {
        match self {
            Driver::WireDeclaration
            | Driver::CombinatorialBlock(_)
            | Driver::InstancePortConnection(_)
            | Driver::OutputPortConnectionToReg => DriverKind::WiredConnection,
            Driver::ClockedBlock(_) => DriverKind::ClockedBlock,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Drivers {
    // For each signal, for each driver, the first span.
    // This span will be used in error messages in case there are multiple drivers for the same signal.
    output_port_drivers: IndexMap<Port, IndexMap<Driver, Span>>,
    reg_drivers: IndexMap<Register, IndexMap<Driver, Span>>,
    wire_drivers: IndexMap<Wire, IndexMap<Driver, Span>>,
}

impl Drivers {
    pub fn new(ports: &Arena<Port, PortInfo>) -> Self {
        let mut drivers = Drivers {
            output_port_drivers: Default::default(),
            reg_drivers: Default::default(),
            wire_drivers: Default::default(),
        };

        // pre-populate output port entries
        for (port, port_info) in ports {
            match port_info.direction.inner {
                PortDirection::Input => {}
                PortDirection::Output => {
                    drivers.output_port_drivers.entry(port).or_default();
                }
            }
        }

        drivers
    }

    pub fn report_assignment(
        &mut self,
        diags: &Diagnostics,
        driver: Driver,
        target: Spanned<Signal>,
    ) -> Result<(), ErrorGuaranteed> {
        fn record<T: Hash + Eq + Debug>(
            diags: &Diagnostics,
            map: &mut IndexMap<T, IndexMap<Driver, Span>>,
            driver: Driver,
            target: T,
            target_span: Span,
        ) -> Result<(), ErrorGuaranteed> {
            map.get_mut(&target)
                .ok_or_else(|| {
                    diags.report_internal_error(target_span, "failed to record driver, target not yet mapped")
                })?
                .entry(driver)
                .or_insert(target_span);
            Ok(())
        }

        match target.inner {
            Signal::Port(port) => record(diags, &mut self.output_port_drivers, driver, port, target.span),
            Signal::Wire(wire) => record(diags, &mut self.wire_drivers, driver, wire, target.span),
            Signal::Register(reg) => record(diags, &mut self.reg_drivers, driver, reg, target.span),
        }
    }
}

fn report_assignment_internal_error<'a>(
    diags: &'a Diagnostics,
    place: &'a str,
) -> impl FnMut(Spanned<Signal>) -> Result<(), ErrorGuaranteed> + 'a {
    move |target: Spanned<Signal>| {
        Err(diags.report_internal_error(target.span, format!("driving signal within {place}")))
    }
}

fn pull_register_init_into_process(
    ctx: &mut CompileItemContext,
    clocked_block_statement_index_to_process_index: &IndexMap<usize, usize>,
    children: &mut Vec<Child>,
    register_initial_values: &IndexMap<Register, Result<Spanned<CompileValue>, ErrorGuaranteed>>,
    reg: Register,
    driver: Driver,
    driver_first_span: Span,
) -> Result<(), ErrorGuaranteed> {
    let diags = ctx.refs.diags;
    let reg_info = &ctx.registers[reg];

    if let Driver::ClockedBlock(stmt_index) = driver {
        if let Some(&process_index) = clocked_block_statement_index_to_process_index.get(&stmt_index) {
            if let Child::Clocked(process) = &mut children[process_index] {
                if let Some(init) = register_initial_values.get(&reg) {
                    let init = init.as_ref_ok()?;
                    let init_ir = init.inner.as_ir_expression_or_undefined(
                        diags,
                        &mut ctx.large,
                        init.span,
                        &reg_info.ty.inner,
                    )?;

                    match init_ir {
                        MaybeUndefined::Undefined => {
                            // we don't need to reset
                        }
                        MaybeUndefined::Defined(init_ir) => {
                            match &mut process.reset {
                                Some(reset) => {
                                    // check that the reset style matches
                                    let reset_style_err = |block: &str, reg: &str| {
                                        let diag = Diagnostic::new("reset style mismatch")
                                            .add_error(driver_first_span, "block drives register here")
                                            .add_info(reset.kind.span, format!("block defined with {block} reset here"))
                                            .add_info(
                                                reg_info.domain.span,
                                                format!("register defined with {reg} reset here"),
                                            )
                                            .finish();
                                        diags.report(diag)
                                    };
                                    match (reset.kind.inner, reg_info.domain.inner.reset) {
                                        (ResetKind::Async, Some(_)) => {}
                                        (ResetKind::Sync, None) => {}
                                        (ResetKind::Async, None) => return Err(reset_style_err("async", "sync")),
                                        (ResetKind::Sync, Some(_)) => return Err(reset_style_err("sync", "async")),
                                    }

                                    // all good, record the reset value
                                    reset.reg_inits.push(ExtraRegisterInit {
                                        span: init.span,
                                        reg: reg_info.ir,
                                        init: init_ir,
                                    });
                                }
                                None => {
                                    // TODO actually, this is conceptually a bit weird:
                                    //   why is reset a properly of the process, and not (only) the register?
                                    let diag = Diagnostic::new(
                                        "clocked block without reset cannot drive register with reset value",
                                    )
                                    .add_error(driver_first_span, "clocked block drives register here")
                                    .add_info(process.clock_signal.span, "clocked block declared without reset here")
                                    .add_info(init.span, "register reset value defined here")
                                    .footer(
                                        Level::Help,
                                        "either add an reset to the block or use `undef` as the the initial value",
                                    )
                                    .finish();
                                    return Err(diags.report(diag));
                                }
                            }
                        }
                    }
                    return Ok(());
                }
            }
        }
    }

    Err(diags.report_internal_error(
        reg_info.id.span(),
        "failure while pulling reset value into clocked process",
    ))
}
