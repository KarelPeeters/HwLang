use crate::front::check::{TypeContainsReason, check_type_contains_type, check_type_contains_value};
use crate::front::compile::{ArenaPortInterfaces, ArenaPorts, CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::{DomainSignal, PortDomain, ValueDomain};
use crate::front::exit::ExitStack;
use crate::front::expression::{NamedOrValue, ValueInner};
use crate::front::flow::{
    ExtraRegisters, Flow, FlowCompile, FlowCompileContent, FlowHardwareRoot, FlowRoot, FlowRootContent,
    HardwareProcessKind, VariableId,
};
use crate::front::function::CapturedScope;
use crate::front::implication::ValueWithImplications;
use crate::front::interface::ElaboratedInterfacePortInfo;
use crate::front::item::{
    ElaboratedInterface, ElaboratedInterfaceView, ElaboratedItemParams, ElaboratedModule, UniqueDeclaration,
};
use crate::front::scope::{NamedValue, Scope, ScopeContent, ScopedEntry};
use crate::front::signal::{
    Polarized, Port, PortInfo, PortInterfaceInfo, Register, RegisterInfo, Signal, Wire, WireInfo, WireInfoInInterface,
    WireInfoSingle, WireInterfaceInfo, WireOrPort,
};
use crate::front::types::{HardwareType, NonHardwareType, Type, Typed};
use crate::front::value::{CompileValue, MaybeUndefined, SimpleCompileValue, ValueCommon};
use crate::mid::ir::{
    IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression,
    IrIfStatement, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo, IrModuleInternalInstance, IrPort,
    IrPortConnection, IrPortInfo, IrPorts, IrRegister, IrRegisterInfo, IrSignal, IrStatement, IrVariables, IrWire,
    IrWireInfo, IrWireOrPort,
};
use crate::syntax::ast::{
    self, ClockedBlockReset, ExpressionKind, ExtraList, ForStatement, ModuleInstance, ModulePortBlock,
    ModulePortInBlock, ModulePortInBlockKind, ModulePortSingleKind, ModuleStatement, ModuleStatementKind,
    PortDirection, PortSingleKindInner, ResetKind, Visibility, WireDeclarationDomainTyKind, WireDeclarationKind,
};
use crate::syntax::ast::{
    Block, ClockedBlock, CombinatorialBlock, DomainKind, Identifier, MaybeIdentifier, ModulePortItem, ModulePortSingle,
    PortConnection, RegDeclaration, RegOutPortMarker, SyncDomain, WireDeclaration,
};
use crate::syntax::parsed::{AstRefModuleExternal, AstRefModuleInternal};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::BigInt;
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
use crate::util::store::ArcOrRef;
use crate::util::{ResultExt, result_pair, result_pair_split};
use crate::{new_index_type, throw};
use annotate_snippets::Level;
use indexmap::IndexMap;
use indexmap::map::Entry;
use itertools::{Either, Itertools, enumerate};
use std::fmt::Debug;
use std::hash::Hash;

// TODO split this file into header/body
new_index_type!(pub Connector);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct ConnectorSingle(usize);

type ArenaConnectors = Arena<Connector, ConnectorInfo>;

pub struct ConnectorInfo {
    id: Identifier,
    kind: DiagResult<ConnectorKind>,
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
        view: Spanned<ElaboratedInterfaceView>,
        singles: Vec<ConnectorSingle>,
    },
}

#[derive(Debug)]
pub struct ElaboratedModuleHeader<A> {
    pub ast_ref: A,
    debug_info_params: Option<Vec<(String, String)>>,
    ports: ArenaPorts,
    port_interfaces: ArenaPortInterfaces,
    pub ports_ir: Arena<IrPort, IrPortInfo>,

    captured_scope_params: CapturedScope,
    scope_ports: ScopeContent,
    flow_root: FlowRootContent,
    flow: FlowCompileContent,
}

pub struct ElaboratedModuleInternalInfo {
    pub ast_ref: AstRefModuleInternal,
    pub unique: UniqueDeclaration,

    pub module_ir: IrModule,
    pub connectors: ArenaConnectors,
}

pub struct ElaboratedModuleExternalInfo {
    pub ast_ref: AstRefModuleExternal,
    pub module_name: String,
    pub generic_args: Option<Vec<(String, BigInt)>>,
    pub port_names: Vec<String>,
    pub connectors: ArenaConnectors,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_module_ports_new<A>(
        self,
        ast_ref: A,
        def_span: Span,
        params: ElaboratedItemParams,
        captured_scope_params: CapturedScope,
        ports: &Spanned<ExtraList<ModulePortItem>>,
    ) -> DiagResult<(ArenaConnectors, ElaboratedModuleHeader<A>)> {
        let ElaboratedItemParams {
            unique: _,
            params: debug_info_params,
        } = params;

        // reconstruct header scope
        let mut ctx = CompileItemContext::new_empty(self, None);
        let flow_root = FlowRoot::new(self.diags);
        let mut flow = FlowCompile::new_root(&flow_root, def_span, "item declaration");
        let scope_params = captured_scope_params.to_scope(self, &mut flow, def_span)?;

        // elaborate ports
        // TODO we actually need a full context/flow here?
        let (connectors, scope_ports, ports_ir) =
            self.elaborate_module_ports_impl(&mut ctx, &scope_params, &mut flow, ports, def_span)?;
        let scope_ports = scope_ports.into_content();

        let source = self.fixed.source;
        let debug_info_params = debug_info_params.map(|p| {
            p.into_iter()
                .map(|(k, v)| {
                    (
                        k.str(source).to_owned(),
                        v.value_string(&self.shared.elaboration_arenas),
                    )
                })
                .collect_vec()
        });

        let flow = flow.into_content();
        let header = ElaboratedModuleHeader {
            ast_ref,
            debug_info_params,
            ports: ctx.ports,
            port_interfaces: ctx.port_interfaces,
            ports_ir,

            captured_scope_params,
            scope_ports,
            flow_root: flow_root.into_content(),
            flow,
        };
        Ok((connectors, header))
    }

    pub fn elaborate_module_body_new(
        self,
        ports: ElaboratedModuleHeader<AstRefModuleInternal>,
    ) -> DiagResult<IrModuleInfo> {
        let ElaboratedModuleHeader {
            ast_ref,
            debug_info_params,
            ports,
            port_interfaces,
            ports_ir,
            captured_scope_params,
            scope_ports,
            flow_root,
            flow,
        } = ports;
        let &ast::ItemDefModuleInternal {
            span: def_span,
            vis: _,
            id,
            params: _,
            ports: _,
            ref body,
        } = &self.fixed.parsed[ast_ref];

        self.check_should_stop(id.span())?;

        // rebuild scopes
        let ctx = CompileItemContext::new_restore(self, None, ports, port_interfaces);
        let flow_root = FlowRoot::restore(self.diags, flow_root);
        let mut flow = FlowCompile::restore_root(&flow_root, flow);

        let scope_params = captured_scope_params.to_scope(self, &mut flow, def_span)?;
        let scope_ports = Scope::restore_child_from_content(def_span, &scope_params, scope_ports);

        // elaborate the body
        self.elaborate_module_body_impl(ctx, &flow, &scope_ports, id, debug_info_params, ports_ir, body)
    }

    fn elaborate_module_ports_impl<'p, F: Flow>(
        &self,
        ctx: &mut CompileItemContext,
        scope_params: &'p Scope<'p>,
        flow: &mut F,
        ports: &Spanned<ExtraList<ModulePortItem>>,
        module_def_span: Span,
    ) -> DiagResult<(ArenaConnectors, Scope<'p>, Arena<IrPort, IrPortInfo>)> {
        let diags = self.diags;
        let source = self.fixed.source;
        let elab = &self.shared.elaboration_arenas;

        let mut scope_ports = Scope::new_child(ports.span.join(Span::empty_at(module_def_span.end())), scope_params);

        let mut connectors: ArenaConnectors = Arena::new();
        let mut port_to_single: IndexMap<Port, ConnectorSingle> = IndexMap::new();
        let mut ports_ir = Arena::new();
        let mut used_ir_names = IndexMap::new();

        let mut next_single = 0;
        let mut next_single = move || {
            let scalar = ConnectorSingle(next_single);
            next_single += 1;
            scalar
        };

        let mut visit_port_item =
            |ctx: &mut CompileItemContext, scope_ports: &mut Scope, flow: &mut F, port_item: &ModulePortItem| {
                match port_item {
                    ModulePortItem::Single(port_item) => {
                        let &ModulePortSingle { span: _, id, ref kind } = port_item;
                        match *kind {
                            ModulePortSingleKind::Port { direction, ref kind } => {
                                let (domain, ty) = match *kind {
                                    PortSingleKindInner::Clock { span_clock } => (
                                        Ok(Spanned::new(span_clock, PortDomain::Clock)),
                                        Ok(Spanned::new(span_clock, HardwareType::Bool)),
                                    ),
                                    PortSingleKindInner::Normal { domain, ty } => (
                                        ctx.eval_port_domain(scope_ports, flow, domain)
                                            .map(|d| d.map_inner(PortDomain::Kind)),
                                        ctx.eval_expression_as_ty_hardware(scope_ports, flow, ty, "port"),
                                    ),
                                };

                                let entry = push_connector_single(
                                    ctx,
                                    &mut used_ir_names,
                                    &mut ports_ir,
                                    &mut connectors,
                                    &mut port_to_single,
                                    &mut next_single,
                                    id,
                                    direction,
                                    domain,
                                    ty,
                                );
                                scope_ports.declare(diags, Ok(id.spanned_str(source)), entry);
                            }
                            ModulePortSingleKind::Interface {
                                span_keyword: _,
                                domain,
                                interface,
                            } => {
                                let domain = ctx.eval_port_domain(scope_ports, flow, domain);
                                let interface_view = ctx
                                    .eval_expression_as_compile(
                                        scope_ports,
                                        flow,
                                        &Type::InterfaceView,
                                        interface,
                                        Spanned::new(interface.span, "interface view"),
                                    )
                                    .and_then(|view| match view.inner {
                                        CompileValue::Simple(SimpleCompileValue::InterfaceView(inner)) => {
                                            Ok(Spanned::new(view.span, inner))
                                        }
                                        value => Err(diags.report_simple(
                                            "expected interface view",
                                            view.span,
                                            format!("got other value with type `{}`", value.ty().value_string(elab)),
                                        )),
                                    });

                                let entry = push_connector_interface(
                                    ctx,
                                    &mut used_ir_names,
                                    &mut ports_ir,
                                    &mut connectors,
                                    &mut port_to_single,
                                    &mut next_single,
                                    id,
                                    domain,
                                    interface_view,
                                );
                                scope_ports.declare(diags, Ok(id.spanned_str(source)), entry);
                            }
                        }
                    }
                    ModulePortItem::Block(port_item) => {
                        let &ModulePortBlock {
                            span: _,
                            domain,
                            ref ports,
                        } = port_item;

                        let domain = ctx.eval_port_domain(scope_ports, flow, domain);

                        let mut visit_port_item_in_block =
                            |ctx: &mut CompileItemContext,
                             scope_ports: &mut Scope,
                             flow: &mut F,
                             port_item_in_block: &ModulePortInBlock| {
                                let &ModulePortInBlock { span: _, id, ref kind } = port_item_in_block;

                                match *kind {
                                    ModulePortInBlockKind::Port { direction, ty } => {
                                        let domain = domain.map(|d| d.map_inner(PortDomain::Kind));
                                        let ty = ctx.eval_expression_as_ty_hardware(scope_ports, flow, ty, "port");

                                        let entry = push_connector_single(
                                            ctx,
                                            &mut used_ir_names,
                                            &mut ports_ir,
                                            &mut connectors,
                                            &mut port_to_single,
                                            &mut next_single,
                                            id,
                                            direction,
                                            domain,
                                            ty,
                                        );
                                        scope_ports.declare(diags, Ok(id.spanned_str(source)), entry);
                                    }
                                    ModulePortInBlockKind::Interface {
                                        span_keyword: _,
                                        interface,
                                    } => {
                                        let interface_view = ctx
                                            .eval_expression_as_compile(
                                                scope_ports,
                                                flow,
                                                &Type::InterfaceView,
                                                interface,
                                                Spanned::new(interface.span, "interface"),
                                            )
                                            .and_then(|view| match view.inner {
                                                CompileValue::Simple(SimpleCompileValue::InterfaceView(inner)) => {
                                                    Ok(Spanned::new(view.span, inner))
                                                }
                                                value => Err(diags.report_simple(
                                                    "expected interface view",
                                                    view.span,
                                                    format!(
                                                        "got other value with type `{}`",
                                                        value.ty().value_string(elab)
                                                    ),
                                                )),
                                            });

                                        let entry = push_connector_interface(
                                            ctx,
                                            &mut used_ir_names,
                                            &mut ports_ir,
                                            &mut connectors,
                                            &mut port_to_single,
                                            &mut next_single,
                                            id,
                                            domain,
                                            interface_view,
                                        );
                                        scope_ports.declare(diags, Ok(id.spanned_str(source)), entry);
                                    }
                                }
                                Ok(())
                            };

                        ctx.compile_elaborate_extra_list(scope_ports, flow, ports, &mut visit_port_item_in_block)?;
                    }
                }
                Ok(())
            };

        ctx.compile_elaborate_extra_list(&mut scope_ports, flow, &ports.inner, &mut visit_port_item)?;

        Ok((connectors, scope_ports, ports_ir))
    }

    fn elaborate_module_body_impl<'a>(
        &self,
        mut ctx_item: CompileItemContext<'a, '_>,
        flow: &FlowCompile,
        scope_header: &Scope,
        def_id: MaybeIdentifier,
        debug_info_params: Option<Vec<(String, String)>>,
        ir_ports: Arena<IrPort, IrPortInfo>,
        body: &'a Block<ModuleStatement>,
    ) -> DiagResult<IrModuleInfo> {
        let drivers = Drivers::new(&ctx_item.ports);

        let mut signals_in_scope = vec![];
        for (port, _) in &ctx_item.ports {
            signals_in_scope.push(Signal::Port(port));
        }

        let mut ctx = BodyElaborationContext {
            ctx: &mut ctx_item,
            ir_ports,
            ir_wires: Arena::new(),
            ir_registers: Arena::new(),
            drivers,

            register_initial_values: IndexMap::new(),
            out_port_register_connections: IndexMap::new(),

            pass_0_next_statement_index: 0,
            clocked_block_statement_index_to_process_index: IndexMap::new(),
            children: vec![],
            delayed_error: Ok(()),
        };

        // process declarations and collect processes and instantiations
        let (todo, _) = ctx.pass_0_declarations_collect(scope_header, flow, body, true);

        // process the collected processes and instantiations
        ctx.pass_1_elaborate_children(scope_header, flow, todo);

        // stop if any errors have happened so far, we don't want spurious errors about drivers
        ctx.delayed_error?;

        // check that types and domains were inferred for everything
        // TODO maybe this should be ordered after driving checking, now we get "failed to infer" before "no driver"
        ctx.pass_2_check_inferred()?;

        // check driver validness
        // TODO more checking: combinatorial blocks can't read values they will later write,
        //   unless they have already written them
        ctx.pass_2_check_drivers_and_populate_resets()?;

        // create process for registered output ports
        // TODO is it worth trying to avoid this and using the "reg" port markers in verilog?
        // TODO should this be a single combinatorial process or multiple?
        if !ctx.out_port_register_connections.is_empty() {
            let mut statements = vec![];

            for (&out_port, &reg) in &ctx.out_port_register_connections {
                let out_port_ir = ctx.ctx.ports[out_port].ir;
                let reg_info = &ctx.ctx.registers[reg];
                let reg_ir = reg_info.ir;
                let statement = IrStatement::Assign(
                    IrAssignmentTarget::simple(out_port_ir.into()),
                    IrExpression::Signal(IrSignal::Register(reg_ir)),
                );
                statements.push(Spanned {
                    span: reg_info.id.span(),
                    inner: statement,
                })
            }

            let process = IrCombinatorialProcess {
                locals: IrVariables::new(),
                block: IrBlock { statements },
            };
            let child = Child::Finished(IrModuleChild::CombinatorialProcess(process));
            ctx.children.push(Spanned::new(body.span, child));
        }

        // return result
        ctx.delayed_error?;
        let processes = ctx
            .children
            .into_iter()
            .map(|c| {
                let c_inner = match c.inner {
                    Child::Finished(c) => c,
                    Child::Clocked(c) => IrModuleChild::ClockedProcess(c.finish(ctx.ctx)?),
                };
                Ok(Spanned::new(c.span, c_inner))
            })
            .try_collect_all_vec()?;

        let source = self.fixed.source;
        Ok(IrModuleInfo {
            ports: ctx.ir_ports,
            registers: ctx.ir_registers,
            wires: ctx.ir_wires,
            large: ctx_item.large,
            children: processes,
            debug_info_file: source[def_id.span().file].debug_info_path.clone(),
            debug_info_id: def_id.spanned_string(source),
            debug_info_generic_args: debug_info_params,
        })
    }
}

fn push_connector_single(
    ctx: &mut CompileItemContext,
    used_ir_names: &mut IndexMap<String, Span>,
    ports_ir: &mut IrPorts,
    connectors: &mut ArenaConnectors,
    port_to_single: &mut IndexMap<Port, ConnectorSingle>,
    next_single: &mut impl FnMut() -> ConnectorSingle,
    id: Identifier,
    direction: Spanned<PortDirection>,
    domain: DiagResult<Spanned<PortDomain<Port>>>,
    ty: DiagResult<Spanned<HardwareType>>,
) -> DiagResult<ScopedEntry> {
    let diags = ctx.refs.diags;
    let source = ctx.refs.fixed.source;
    let elab = &ctx.refs.shared.elaboration_arenas;

    claim_ir_name(diags, used_ir_names, id.str(source), id.span)?;

    let kind_and_entry = result_pair(domain, ty).map(|(domain, ty)| {
        let ir_port = ports_ir.push(IrPortInfo {
            name: id.str(source).to_owned(),
            direction: direction.inner,
            ty: ty.inner.as_ir(ctx.refs),
            debug_span: id.span,
            debug_info_ty: ty.as_ref().map_inner(|inner| inner.value_string(elab)),
            debug_info_domain: domain.inner.diagnostic_string(ctx),
        });

        let port = ctx.ports.push(PortInfo {
            span: id.span,
            name: id.str(source).to_owned(),
            direction,
            domain,
            ty: ty.clone(),
            ir: ir_port,
        });

        let single = next_single();
        port_to_single.insert_first(port, single);

        let connector_domain = domain.map_inner(|d| d.map_inner(|p| *port_to_single.get(&p).unwrap()));
        let kind = ConnectorKind::Port {
            direction,
            domain: connector_domain,
            ty,
            single,
        };

        let entry = ScopedEntry::Named(NamedValue::Port(port));
        (kind, entry)
    });

    let (kind, entry) = result_pair_split(kind_and_entry);
    connectors.push(ConnectorInfo { id, kind });
    entry
}

fn push_connector_interface(
    ctx: &mut CompileItemContext,
    used_ir_names: &mut IndexMap<String, Span>,
    ports_ir: &mut IrPorts,
    connectors: &mut ArenaConnectors,
    port_to_single: &mut IndexMap<Port, ConnectorSingle>,
    next_single: &mut impl FnMut() -> ConnectorSingle,
    id: Identifier,
    domain: DiagResult<Spanned<DomainKind<Polarized<Port>>>>,
    view: DiagResult<Spanned<ElaboratedInterfaceView>>,
) -> DiagResult<ScopedEntry> {
    let diags = ctx.refs.diags;
    let source = ctx.refs.fixed.source;
    let elab = &ctx.refs.shared.elaboration_arenas;

    let kind_and_entry = result_pair(domain, view).and_then(|(domain, view)| {
        let ElaboratedInterfaceView { interface, view_index } = view.inner;
        let interface = ctx.refs.shared.elaboration_arenas.interface_info(interface);
        let view_info = &interface.views[view_index];

        let port_dirs = view_info.port_dirs.as_ref_ok()?;

        let mut declared_ports = vec![];
        let mut singles = vec![];

        // TODO rework interface port names, to ensure unique but still readable names
        for (port_index, (_, port)) in enumerate(&interface.ports) {
            let id_str = id.str(source);
            let port_id_str = port.id.str(source);
            let name = format!("{id_str}.{port_id_str}");
            let ir_name = format!("{id_str}_{port_id_str}");
            claim_ir_name(diags, used_ir_names, &ir_name, id.span)?;

            let direction = port_dirs[port_index].1;
            let ty = port.ty.as_ref_ok()?;

            let ir_port = ports_ir.push(IrPortInfo {
                name: ir_name,
                direction: direction.inner,
                ty: ty.inner.as_ir(ctx.refs),
                debug_span: id.span,
                debug_info_ty: ty.as_ref().map_inner(|ty| ty.value_string(elab)),
                debug_info_domain: domain.inner.diagnostic_string(ctx),
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
            id,
            view,
            domain,
            ports: declared_ports,
        });
        let connector_domain =
            domain.map_inner(|d| d.map_signal(|p| p.map_inner(|p| *port_to_single.get(&p).unwrap())));
        let kind = ConnectorKind::Interface {
            domain: connector_domain,
            view,
            singles,
        };

        let entry = ScopedEntry::Named(NamedValue::PortInterface(port_interface));
        Ok((kind, entry))
    });

    let (kind, entry) = result_pair_split(kind_and_entry);
    connectors.push(ConnectorInfo { id, kind });
    entry
}

// TODO think about this, and maybe expand to include more things (eg. signals, child modules, ...)
fn claim_ir_name(
    diags: &Diagnostics,
    used_ir_names: &mut IndexMap<String, Span>,
    name: &str,
    span: Span,
) -> DiagResult {
    match used_ir_names.entry(name.to_owned()) {
        Entry::Vacant(entry) => {
            entry.insert(span);
            Ok(())
        }
        Entry::Occupied(entry) => {
            let diag = Diagnostic::new(format!(
                "port with name `{name}` conflicts with earlier port with the same name"
            ))
            .add_error(span, "new port defined here")
            .add_info(*entry.get(), "previous port defined here")
            .finish();
            Err(diags.report(diag))
        }
    }
}

struct BodyElaborationContext<'c, 'a, 's> {
    ctx: &'c mut CompileItemContext<'a, 's>,

    ir_ports: Arena<IrPort, IrPortInfo>,
    ir_wires: Arena<IrWire, IrWireInfo>,
    ir_registers: Arena<IrRegister, IrRegisterInfo>,
    drivers: Drivers,

    register_initial_values: IndexMap<Register, DiagResult<Spanned<MaybeUndefined<CompileValue>>>>,
    out_port_register_connections: IndexMap<Port, Register>,

    pass_0_next_statement_index: usize,
    clocked_block_statement_index_to_process_index: IndexMap<usize, usize>,
    children: Vec<Spanned<Child>>,

    // TODO rename and rethink this
    delayed_error: DiagResult,
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
    pub fn finish(self, ctx: &mut CompileItemContext) -> DiagResult<IrClockedProcess> {
        let ChildClockedProcess {
            locals,
            clock_signal,
            clock_block,
            reset,
        } = self;

        let clock_signal_ir = Spanned::new(clock_signal.span, ctx.domain_signal_to_ir(clock_signal)?);

        let (clock_block, async_reset) = match reset {
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

                let reset_signal_ir = Spanned::new(reset_signal.span, ctx.domain_signal_to_ir(reset_signal)?);

                match kind.inner {
                    ResetKind::Async => {
                        // async reset becomes a separate block, sensitive to the signal
                        let resets = reg_inits
                            .into_iter()
                            .map(|init| {
                                let ExtraRegisterInit { span, reg, init } = init;
                                Spanned::new(span, (reg, init))
                            })
                            .collect_vec();

                        let info = IrAsyncResetInfo {
                            signal: reset_signal_ir,
                            resets,
                        };
                        (clock_block, Some(info))
                    }
                    ResetKind::Sync => {
                        // sync reset just becomes an if branch inside the clocked block
                        let reset_statements = reg_inits
                            .into_iter()
                            .map(|init| {
                                let ExtraRegisterInit { span, reg, init } = init;
                                let stmt = IrStatement::Assign(IrAssignmentTarget::simple(reg.into()), init);
                                Spanned::new(span, stmt)
                            })
                            .collect_vec();
                        let reset_block = IrBlock {
                            statements: reset_statements,
                        };

                        let if_stmt = IrIfStatement {
                            condition: reset_signal_ir.inner.as_expression(&mut ctx.large),
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

        Ok(IrClockedProcess {
            locals,
            clock_signal: clock_signal_ir,
            clock_block,
            async_reset,
        })
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

pub struct ModuleTodo<'a> {
    span: Span,
    scope: ScopeContent,
    flow: FlowCompileContent,
    children: Vec<ModuleChildTodo<'a>>,
}

pub enum ModuleChildTodo<'a> {
    Nested(ModuleTodo<'a>),

    CombinatorialBlock(usize, Spanned<&'a CombinatorialBlock>),
    ClockedBlock(usize, Spanned<&'a ClockedBlock>),
    Instance(usize, Spanned<&'a ModuleInstance>),
}

type PublicDeclarations<'a> = Vec<(
    DiagResult<MaybeIdentifier<Spanned<ArcOrRef<'a, str>>>>,
    DiagResult<ScopedEntry>,
)>;

impl<'a> BodyElaborationContext<'_, 'a, '_> {
    #[must_use]
    fn pass_0_declarations_collect(
        &mut self,
        scope_outer: &Scope,
        flow_outer: &FlowCompile,
        block_outer: &'a Block<ModuleStatement>,
        is_root_block: bool,
    ) -> (ModuleTodo<'a>, PublicDeclarations<'a>) {
        let diags = self.ctx.refs.diags;
        let source = self.ctx.refs.fixed.source;

        let Block { span: _, statements } = block_outer;

        let mut scope = Scope::new_child(block_outer.span, scope_outer);
        let mut flow = flow_outer.new_child_isolated();

        let mut todo_children = vec![];
        let mut pub_declarations = vec![];

        for stmt in statements {
            let stmt_index = self.pass_0_next_statement_index;
            self.pass_0_next_statement_index += 1;

            match &stmt.inner {
                // control flow: evaluate now and visit the children immediately
                ModuleStatementKind::Block(block) => {
                    let (todo, decls) = self.pass_0_declarations_collect(&scope, &flow, block, false);
                    process_todo_and_decls(
                        diags,
                        &mut scope,
                        &mut todo_children,
                        &mut pub_declarations,
                        todo,
                        decls,
                    );
                }
                ModuleStatementKind::If(if_stmt) => {
                    match self.ctx.compile_if_statement_choose_block(&scope, &mut flow, if_stmt) {
                        Ok(block) => {
                            if let Some(block) = block {
                                let (todo, decls) = self.pass_0_declarations_collect(&scope, &flow, block, false);
                                process_todo_and_decls(
                                    diags,
                                    &mut scope,
                                    &mut todo_children,
                                    &mut pub_declarations,
                                    todo,
                                    decls,
                                );
                            }
                        }
                        Err(e) => {
                            self.delayed_error = Err(e);
                        }
                    }
                }
                ModuleStatementKind::For(for_stmt) => {
                    let for_stmt = Spanned::new(stmt.span, for_stmt);
                    match self.elaborate_for(&scope, &flow, for_stmt) {
                        Ok((todo, decls)) => {
                            process_todo_and_decls(
                                diags,
                                &mut scope,
                                &mut todo_children,
                                &mut pub_declarations,
                                todo,
                                decls,
                            );
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }

                // declarations: evaluate and declare now
                ModuleStatementKind::CommonDeclaration(decl) => {
                    self.ctx.eval_and_declare_declaration(&mut scope, &mut flow, decl);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let id = self.ctx.eval_maybe_general_id(&scope, &mut flow, decl.id);

                    let entry = id.as_ref_ok().and_then(|id| {
                        let decl = Spanned::new(stmt.span, decl);
                        let reg_init = self.elaborate_module_declaration_reg(&scope, &flow, id, decl)?;

                        let RegisterInit { reg, init } = reg_init;
                        self.register_initial_values.insert_first(reg, init);
                        self.drivers.reg_drivers.insert_first(reg, Ok(IndexMap::new()));

                        Ok(ScopedEntry::Named(NamedValue::Register(reg)))
                    });

                    if let Err(e) = entry {
                        self.delayed_error = Err(e);
                    }

                    match &decl.vis {
                        Visibility::Public { span: _ } => {
                            pub_declarations.push((id.clone(), entry));
                        }
                        Visibility::Private => {}
                    }

                    scope.maybe_declare(diags, id, entry);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let id = self.ctx.eval_maybe_general_id(&scope, &mut flow, decl.id);

                    let entry = id.as_ref_ok().and_then(|id| {
                        let (named_value, process) =
                            self.elaborate_module_declaration_wire(&scope, &mut flow, id, stmt.span, decl)?;

                        if let Some(process) = process {
                            self.children
                                .push(process.map_inner(|c| Child::Finished(IrModuleChild::CombinatorialProcess(c))));
                        }

                        Ok(ScopedEntry::Named(named_value))
                    });

                    if let Err(e) = entry {
                        self.delayed_error = Err(e);
                    }

                    match &decl.vis {
                        Visibility::Public { span: _ } => {
                            pub_declarations.push((id.clone(), entry));
                        }
                        Visibility::Private => {}
                    }

                    scope.maybe_declare(diags, id, entry);
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    // TODO check if this still works in nested blocks, maybe we should only allow this at the top level
                    //   no, we can't really ban this, we need conditional makers for eg. conditional ports
                    // declare register that shadows the outer port, which is exactly what we want
                    match self.elaborate_module_declaration_reg_out_port(
                        &scope,
                        &mut flow,
                        stmt.span,
                        decl,
                        is_root_block,
                    ) {
                        Ok((port, reg_init)) => {
                            let port_drivers = self.drivers.output_port_drivers.get_mut(&port).unwrap();
                            if let Ok(port_drivers) = port_drivers.as_ref_mut_ok() {
                                assert!(port_drivers.is_empty());
                                port_drivers.insert_first(Driver::OutputPortConnectionToReg, stmt.span);
                            }

                            self.drivers.reg_drivers.insert_first(reg_init.reg, Ok(IndexMap::new()));
                            self.register_initial_values.insert_first(reg_init.reg, reg_init.init);
                            self.out_port_register_connections.insert_first(port, reg_init.reg);

                            scope.declare(
                                diags,
                                Ok(decl.id.spanned_str(source)),
                                Ok(ScopedEntry::Named(NamedValue::Register(reg_init.reg))),
                            );
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }

                // processes: collect and evaluate later, after all declarations have been processed
                ModuleStatementKind::CombinatorialBlock(child) => {
                    todo_children.push(ModuleChildTodo::CombinatorialBlock(
                        stmt_index,
                        Spanned::new(stmt.span, child),
                    ));
                }
                ModuleStatementKind::ClockedBlock(child) => {
                    todo_children.push(ModuleChildTodo::ClockedBlock(
                        stmt_index,
                        Spanned::new(stmt.span, child),
                    ));
                }
                ModuleStatementKind::Instance(child) => {
                    todo_children.push(ModuleChildTodo::Instance(stmt_index, Spanned::new(stmt.span, child)));
                }
            }
        }

        let todo = ModuleTodo {
            span: block_outer.span,
            scope: scope.into_content(),
            flow: flow.into_content(),
            children: todo_children,
        };
        (todo, pub_declarations)
    }

    fn pass_1_elaborate_children(&mut self, scope_parent: &Scope, flow_parent: &FlowCompile, todo: ModuleTodo) {
        let ModuleTodo {
            span: span_block,
            scope,
            flow,
            children,
        } = todo;

        let scope = Scope::restore_child_from_content(span_block, scope_parent, scope);
        let mut flow = FlowCompile::restore_child_isolated(flow_parent, flow);

        for child in children {
            match child {
                ModuleChildTodo::Nested(inner) => {
                    self.pass_1_elaborate_children(&scope, &flow, inner);
                }
                ModuleChildTodo::CombinatorialBlock(stmt_index, block) => {
                    let ir_process = self.elaborate_combinatorial_block(&scope, &flow, stmt_index, block.inner);
                    match ir_process {
                        Ok(ir_process) => {
                            let child = Child::Finished(IrModuleChild::CombinatorialProcess(ir_process));
                            self.children.push(Spanned::new(block.span, child))
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
                ModuleChildTodo::ClockedBlock(stmt_index, block) => {
                    let ir_process = self.elaborate_clocked_block(&scope, &mut flow, stmt_index, block.inner);

                    match ir_process {
                        Ok(ir_process) => {
                            let child_index = self.children.len();
                            let child = Child::Clocked(ir_process);
                            self.children.push(Spanned::new(block.span, child));
                            self.clocked_block_statement_index_to_process_index
                                .insert_first(stmt_index, child_index);
                        }
                        Err(e) => self.delayed_error = Err(e),
                    }
                }
                ModuleChildTodo::Instance(stmt_index, child) => {
                    let child_ir = self.elaborate_instance(&scope, &mut flow, stmt_index, child.inner);
                    match child_ir {
                        Ok(child_ir) => {
                            self.children.push(Spanned::new(child.span, Child::Finished(child_ir)));
                        }
                        Err(e) => {
                            self.delayed_error = Err(e);
                        }
                    }
                }
            }
        }
    }

    fn elaborate_combinatorial_block(
        &mut self,
        scope: &Scope,
        flow_parent: &FlowCompile,
        stmt_index: usize,
        block: &CombinatorialBlock,
    ) -> DiagResult<IrCombinatorialProcess> {
        let &CombinatorialBlock {
            span_keyword,
            ref block,
        } = block;
        let diags = self.ctx.refs.diags;

        // elaborate block
        // TODO maybe unify wires and ports?
        let mut wires_driven = IndexMap::new();
        let mut ports_driven = IndexMap::new();
        let flow_kind = HardwareProcessKind::CombinatorialBlockBody {
            span_keyword,
            wires_driven: &mut wires_driven,
            ports_driven: &mut ports_driven,
        };
        let mut flow = FlowHardwareRoot::new(
            flow_parent,
            block.span,
            flow_kind,
            &mut self.ir_wires,
            &mut self.ir_registers,
        );

        let mut stack = ExitStack::new_root();
        let end = self
            .ctx
            .elaborate_block(scope, &mut flow.as_flow(), &mut stack, block)?;
        end.unwrap_normal(diags, block.span)?;
        let (ir_vars, ir_block) = flow.finish();

        // report drivers
        let driver = Driver::CombinatorialBlock(stmt_index);
        for (wire, span) in wires_driven {
            let _ = self
                .drivers
                .report_assignment(self.ctx, driver, Spanned::new(span, Signal::Wire(wire)));
        }
        for (port, span) in ports_driven {
            let _ = self
                .drivers
                .report_assignment(self.ctx, driver, Spanned::new(span, Signal::Port(port)));
        }

        Ok(IrCombinatorialProcess {
            locals: ir_vars,
            block: ir_block,
        })
    }

    fn elaborate_clocked_block(
        &mut self,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        stmt_index: usize,
        block: &ClockedBlock,
    ) -> DiagResult<ChildClockedProcess> {
        let &ClockedBlock {
            span_keyword,
            span_domain,
            clock,
            reset,
            ref block,
        } = block;
        let diags = self.ctx.refs.diags;

        // eval domain
        let clock = self.ctx.eval_expression_as_domain_signal(scope, flow_parent, clock);
        let reset = reset
            .as_ref()
            .map(|reset| {
                reset
                    .as_ref()
                    .map_inner(|reset| {
                        let &ClockedBlockReset { kind, signal } = reset;
                        let signal = self.ctx.eval_expression_as_domain_signal(scope, flow_parent, signal)?;
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

                    let source_domain = signal.inner.signal.domain(self.ctx, signal.span)?;
                    let source = Spanned::new(reset.span, source_domain.inner);
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

        let mut elaborate_block = |extra_registers| {
            // elaborate block
            let mut registers_driven = IndexMap::new();
            let flow_kind = HardwareProcessKind::ClockedBlockBody {
                span_keyword,
                domain,
                registers_driven: &mut registers_driven,
                extra_registers,
            };
            let mut flow = FlowHardwareRoot::new(
                flow_parent,
                block.span,
                flow_kind,
                &mut self.ir_wires,
                &mut self.ir_registers,
            );

            let mut stack = ExitStack::new_root();
            let end = self
                .ctx
                .elaborate_block(scope, &mut flow.as_flow(), &mut stack, block)?;
            end.unwrap_normal(diags, block.span)?;

            let (ir_vars, ir_block) = flow.finish();

            // report drivers
            for (reg, span) in registers_driven {
                let driver = Driver::ClockedBlock(stmt_index);
                let _ = self
                    .drivers
                    .report_assignment(self.ctx, driver, Spanned::new(span, Signal::Register(reg)));
            }

            Ok((ir_vars, ir_block))
        };

        let ((ir_vars, ir_block), reset) = match reset {
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
            locals: ir_vars,
            clock_signal: clock,
            clock_block: ir_block,
            reset,
        })
    }

    fn elaborate_for(
        &mut self,
        scope_parent: &Scope,
        flow_parent: &FlowCompile,
        for_stmt: Spanned<&'a ForStatement<ModuleStatement>>,
    ) -> DiagResult<(ModuleTodo<'a>, PublicDeclarations<'a>)> {
        let diags = self.ctx.refs.diags;
        let source = self.ctx.refs.fixed.source;
        let elab = &self.ctx.refs.shared.elaboration_arenas;

        let &ForStatement {
            span_keyword,
            index,
            index_ty,
            iter,
            ref body,
        } = for_stmt.inner;
        let iter_span = iter.span;

        let mut scope_for = Scope::new_child(for_stmt.span, scope_parent);
        let mut flow_for = flow_parent.new_child_isolated();

        // header
        let index_ty = index_ty.map(|index_ty| self.ctx.eval_expression_as_ty(&scope_for, &mut flow_for, index_ty));
        let iter = self
            .ctx
            .eval_expression_as_for_iterator(&scope_for, &mut flow_for, iter);

        let index_ty = index_ty.transpose()?;
        let iter = iter?;

        let mut todo_children = vec![];
        let mut pub_declarations = vec![];

        // TODO allow break?
        for index_value in iter {
            self.ctx.refs.check_should_stop(span_keyword)?;

            let index_value = index_value.map_hardware(|h| h.map_expression(|h| self.ctx.large.push_expr(h)));

            if let Some(index_ty) = &index_ty {
                let index_value_spanned = Spanned::new(iter_span, &index_value);
                check_type_contains_value(
                    diags,
                    elab,
                    TypeContainsReason::ForIndexType(index_ty.span),
                    &index_ty.inner,
                    index_value_spanned,
                )?;
            }

            // elaborate the body
            let mut scope_body = Scope::new_child(index.span().join(body.span), &scope_for);
            let mut flow_body = flow_for.new_child_isolated();

            let index_var = flow_body.var_new_immutable_init(
                self.ctx.refs,
                index.span(),
                VariableId::Id(index),
                span_keyword,
                Ok(ValueWithImplications::simple(index_value)),
            )?;
            scope_body.maybe_declare(
                diags,
                Ok(index.spanned_str(source)),
                Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
            );

            let (todo, decls) = self.pass_0_declarations_collect(&scope_body, &flow_body, body, false);

            // wrap the children in an additional layer to keep track of the inner scope
            let scope_body = scope_body.into_content();
            let flow_body = flow_body.into_content();

            let mut todo_children_inner = vec![];
            process_todo_and_decls(
                diags,
                &mut scope_for,
                &mut todo_children_inner,
                &mut pub_declarations,
                todo,
                decls,
            );

            if !todo_children_inner.is_empty() {
                todo_children.push(ModuleChildTodo::Nested(ModuleTodo {
                    span: body.span,
                    scope: scope_body,
                    flow: flow_body,
                    children: todo_children_inner,
                }))
            }
        }

        let todo = ModuleTodo {
            span: for_stmt.span,
            scope: scope_for.into_content(),
            flow: flow_for.into_content(),
            children: todo_children,
        };

        Ok((todo, pub_declarations))
    }

    fn elaborate_instance(
        &mut self,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        stmt_index: usize,
        instance: &ModuleInstance,
    ) -> DiagResult<IrModuleChild> {
        let refs = self.ctx.refs;
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;
        let source = ctx.refs.fixed.source;

        let &ModuleInstance {
            ref name,
            span_keyword,
            module,
            ref port_connections,
        } = instance;

        let elaborated_module = ctx.eval_expression_as_module(scope, flow_parent, span_keyword, module)?;

        let (instance_info, connectors, def_ports_span) = match elaborated_module {
            ElaboratedModule::Internal(module) => {
                let ElaboratedModuleInternalInfo {
                    ast_ref,
                    unique: _,
                    module_ir,
                    connectors,
                } = ctx.refs.shared.elaboration_arenas.module_internal_info(module);
                (
                    ElaboratedModule::Internal(*module_ir),
                    connectors,
                    refs.fixed.parsed[*ast_ref].ports.span,
                )
            }
            ElaboratedModule::External(module) => {
                let ElaboratedModuleExternalInfo {
                    ast_ref,
                    module_name,
                    generic_args,
                    port_names,
                    connectors,
                } = ctx.refs.shared.elaboration_arenas.module_external_info(module);
                (
                    ElaboratedModule::External((module_name, generic_args, port_names)),
                    connectors,
                    refs.fixed.parsed[*ast_ref].ports.span,
                )
            }
        };

        // check that connections are unique
        let mut id_to_connection_and_used: IndexMap<&str, (&Spanned<PortConnection>, bool)> = IndexMap::new();
        for connection in &port_connections.inner {
            match id_to_connection_and_used.entry(connection.inner.id.str(source)) {
                Entry::Vacant(entry) => {
                    entry.insert((connection, false));
                }
                Entry::Occupied(entry) => {
                    let (prev_connection, _) = entry.get();
                    let diag = Diagnostic::new("duplicate connection")
                        .add_info(prev_connection.span, "previous connection here")
                        .add_error(connection.span, "connected again here")
                        .finish();
                    return Err(diags.report(diag));
                }
            }
        }

        // match connectors to connections
        // TODO it's a bit weird that these are evaluated in declaration instead of connection order,
        //   but that's much more convenient for domain resolution
        let mut single_to_signal = IndexMap::new();
        let mut ir_connections = vec![];

        for (connector_index, (connector, connector_info)) in enumerate(connectors) {
            let connector_id_str = connector_info.id.str(source);
            match id_to_connection_and_used.get_mut(connector_id_str) {
                Some((connection, connection_used)) => {
                    if *connection_used {
                        // this should have already been caught during module header elaboration
                        return Err(diags.report_internal_error(connection.span, "connection used twice"));
                    }
                    *connection_used = true;

                    let connections = self.elaborate_instance_port_connection(
                        scope,
                        flow_parent,
                        stmt_index,
                        connector_index,
                        connectors,
                        &single_to_signal,
                        connector,
                        connection,
                    )?;

                    for (single, signal, ir_connection) in connections {
                        single_to_signal.insert_first(single, signal);
                        ir_connections.push(ir_connection);
                    }
                }
                None => {
                    let diag = Diagnostic::new(format!("missing connection for port {connector_id_str}"))
                        .add_error(Span::empty_at(port_connections.span.end()), "connections here")
                        .add_info(connector_info.id.span, "port declared here")
                        .finish();
                    return Err(diags.report(diag));
                }
            }
        }

        let mut any_unused_err = Ok(());
        for (_, &(connection, used)) in id_to_connection_and_used.iter() {
            if !used {
                let diag = Diagnostic::new("connection does not match any port")
                    .add_error(connection.span, "invalid connection here")
                    .add_info(def_ports_span, "ports declared here")
                    .finish();
                any_unused_err = Err(diags.report(diag));
            }
        }
        any_unused_err?;

        let name = name.as_ref().map(|name| name.str(source).to_owned());
        let ir_instance = match instance_info {
            ElaboratedModule::Internal(module_ir) => IrModuleChild::ModuleInternalInstance(IrModuleInternalInstance {
                name,
                module: module_ir,
                port_connections: ir_connections,
            }),
            ElaboratedModule::External((module_name, generic_args, port_names)) => {
                IrModuleChild::ModuleExternalInstance(IrModuleExternalInstance {
                    name,
                    module_name: module_name.clone(),
                    generic_args: generic_args.clone(),
                    port_names: port_names.clone(),
                    port_connections: ir_connections,
                })
            }
        };
        Ok(ir_instance)
    }

    fn elaborate_instance_port_connection(
        &mut self,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        stmt_index: usize,
        connection_index: usize,
        connectors: &ArenaConnectors,
        prev_single_to_signal: &IndexMap<ConnectorSingle, ConnectionSignal>,
        connector: Connector,
        connection: &Spanned<PortConnection>,
    ) -> DiagResult<Vec<(ConnectorSingle, ConnectionSignal, Spanned<IrPortConnection>)>> {
        let diags = self.ctx.refs.diags;
        let source = self.ctx.refs.fixed.source;
        let elab = &self.ctx.refs.shared.elaboration_arenas;

        let &PortConnection {
            id: ref connection_id,
            expr: value_expr,
        } = &connection.inner;
        let value_expr = value_expr.expr();

        let ConnectorInfo { id: connector_id, kind } = &connectors[connector];

        // double-check id match
        if connector_id.str(source) != connection_id.str(source) {
            return Err(diags.report_internal_error(connection.span, "connection name mismatch"));
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
        let signal = match &self.ctx.refs.get_expr(value_expr) {
            ExpressionKind::Dummy => ConnectionSignal::Dummy(value_expr.span),
            _ => {
                let mut flow_domain = flow_parent.new_child_isolated();
                match self
                    .ctx
                    .try_eval_expression_as_domain_signal(scope, &mut flow_domain, value_expr, |_| ())
                {
                    Ok(signal) => ConnectionSignal::Signal(signal.inner),
                    Err(Either::Left(())) => ConnectionSignal::Expression(value_expr.span),
                    Err(Either::Right(e)) => throw!(e),
                }
            }
        };

        // evaluate the connection differently depending on the port direction
        match kind.as_ref_ok()? {
            &ConnectorKind::Port {
                direction,
                domain,
                ref ty,
                single,
            } => {
                let connector_domain = domain
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
                        if let ExpressionKind::Dummy = self.ctx.refs.get_expr(value_expr) {
                            let diag = Diagnostic::new("dummy connections are only allowed for output ports")
                                .add_error(value_expr.span, "dummy connection used here")
                                .add_info(direction.span, "port declared as input here")
                                .finish();
                            return Err(diags.report(diag));
                        }

                        // eval expr
                        let flow_kind = HardwareProcessKind::InstancePortConnection {
                            span_connection: connection.span,
                        };
                        let mut flow = FlowHardwareRoot::new(
                            flow_parent,
                            value_expr.span,
                            flow_kind,
                            &mut self.ir_wires,
                            &mut self.ir_registers,
                        );
                        let connection_value =
                            self.ctx
                                .eval_expression(scope, &mut flow.as_flow(), &ty.inner.as_type(), value_expr)?;
                        let (ir_vars, mut ir_block) = flow.finish();

                        // check type
                        let reason = TypeContainsReason::InstancePortInput {
                            span_connection_port_id: connection_id.span,
                            span_port_ty: ty.span,
                        };
                        let check_ty = check_type_contains_value(
                            diags,
                            elab,
                            reason,
                            &ty.inner.as_type(),
                            connection_value.as_ref(),
                        );

                        // check domain
                        let target_domain = Spanned {
                            span: connection_id.span,
                            inner: connector_domain.inner,
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
                                Ok(v.as_hardware_value_unchecked(
                                    self.ctx.refs,
                                    &mut self.ctx.large,
                                    value_expr.span,
                                    ty.inner.clone(),
                                )?
                                .expr)
                            })
                            .transpose()?;

                        // build extra wire and process if necessary
                        let connection_signal_ir = if ir_block.statements.is_empty()
                            && let IrExpression::Signal(connection_signal) = connection_value_ir_raw.inner
                        {
                            connection_signal
                        } else {
                            let extra_ir_wire = self.ir_wires.push(IrWireInfo {
                                ty: ty.inner.as_ir(self.ctx.refs),
                                debug_info_id: connector_id.spanned_string(source).map_inner(Some),
                                debug_info_ty: ty.inner.clone().value_string(&self.ctx.refs.shared.elaboration_arenas),
                                debug_info_domain: connection_value.inner.domain().diagnostic_string(self.ctx),
                            });

                            ir_block.statements.push(Spanned {
                                span: connection.span,
                                inner: IrStatement::Assign(
                                    IrAssignmentTarget::simple(extra_ir_wire.into()),
                                    connection_value_ir_raw.inner,
                                ),
                            });
                            let process = IrCombinatorialProcess {
                                locals: ir_vars,
                                block: ir_block,
                            };
                            let child = Child::Finished(IrModuleChild::CombinatorialProcess(process));
                            self.children.push(Spanned::new(connection.span, child));

                            IrSignal::Wire(extra_ir_wire)
                        };

                        IrPortConnection::Input(Spanned {
                            span: value_expr.span,
                            inner: connection_signal_ir,
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

                        match self.ctx.refs.get_expr(value_expr) {
                            ExpressionKind::Dummy => IrPortConnection::Output(None),
                            &ExpressionKind::Id(id) => {
                                let id = self.ctx.eval_general_id(scope, flow_parent, id)?;
                                let id = id.as_ref().map_inner(ArcOrRef::as_ref);
                                let named = self.ctx.eval_named_or_value(scope, id)?;

                                let (signal_ir, signal_target, signal_domain, signal_ty) = match named.inner {
                                    NamedOrValue::Named(NamedValue::Wire(wire)) => {
                                        let wire_info = &mut self.ctx.wires[wire];

                                        let wire_domain =
                                            wire_info.suggest_domain(&mut self.ctx.wire_interfaces, connector_domain);
                                        let wire_ty = wire_info.suggest_ty(
                                            self.ctx.refs,
                                            &self.ctx.wire_interfaces,
                                            &mut self.ir_wires,
                                            ty.as_ref(),
                                        );

                                        let wire_domain = wire_domain?;
                                        let wire_ty = wire_ty?;

                                        (
                                            IrWireOrPort::Wire(wire_ty.ir),
                                            Signal::Wire(wire),
                                            wire_domain,
                                            wire_ty.ty,
                                        )
                                    }
                                    NamedOrValue::Named(NamedValue::Port(port)) => {
                                        let port_info = &self.ctx.ports[port];
                                        (
                                            IrWireOrPort::Port(port_info.ir),
                                            Signal::Port(port),
                                            port_info.domain.map_inner(ValueDomain::from_port_domain),
                                            port_info.ty.as_ref(),
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
                                    elab,
                                    reason,
                                    &signal_ty.inner.as_type(),
                                    Spanned {
                                        span: connection_id.span,
                                        inner: &ty.inner.as_type(),
                                    },
                                ));

                                // check domain
                                any_err = any_err.and(self.ctx.check_valid_domain_crossing(
                                    connection.span,
                                    signal_domain,
                                    connector_domain,
                                    "output port connection",
                                ));

                                // report driver
                                let driver = Driver::InstancePortConnection(stmt_index, connection_index, 0);
                                let target = Spanned::new(named.span, signal_target);
                                self.drivers.report_assignment(self.ctx, driver, target)?;

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
                let mut flow_connection = flow_parent.new_child_isolated();
                let value = self
                    .ctx
                    .eval_expression_inner(scope, &mut flow_connection, &Type::Any, value_expr)?;

                // unwrap interface
                // TODO avoid cloning signals vec here
                let (value_interface, value_domain, value_signals) = match value {
                    ValueInner::PortInterface(port_interface) => {
                        let info = &self.ctx.port_interfaces[port_interface];
                        let port_interface = info.view.map_inner(|v| v.interface);
                        let port_domain = info
                            .domain
                            .map_inner(|d| ValueDomain::from_domain_kind(d.map_signal(|s| s.map_inner(Signal::Port))));
                        let port_signals = WireOrPort::Port(&info.ports);
                        (port_interface, port_domain, port_signals)
                    }
                    ValueInner::WireInterface(wire_interface) => {
                        let info = &mut self.ctx.wire_interfaces[wire_interface];
                        let wire_domain = info.suggest_domain(connector_domain)?;
                        // reborrow immutably
                        let info = &self.ctx.wire_interfaces[wire_interface];
                        let wire_signals = WireOrPort::Wire((&info.wires, &info.ir_wires));
                        (info.interface, wire_domain, wire_signals)
                    }
                    ValueInner::Value(_) => {
                        let diag = Diagnostic::new("expected interface value")
                            .add_error(value_expr.span, "got non-interface expression")
                            .add_info(connector_id.span, "port defined as interface here")
                            .finish();
                        return Err(diags.report(diag));
                    }
                };

                // check interface match (including generics)
                if value_interface.inner != connector_view.inner.interface {
                    let diag = Diagnostic::new("interface mismatch")
                        .add_error(value_expr.span, "got mismatching interface")
                        .add_info(
                            connector_view.span,
                            format!(
                                "expected interface `{}` set here",
                                SimpleCompileValue::Interface(connector_view.inner.interface).value_string(elab)
                            ),
                        )
                        .add_info(
                            value_interface.span,
                            format!(
                                "actual interface `{}` set here",
                                SimpleCompileValue::Interface(value_interface.inner).value_string(elab)
                            ),
                        )
                        .finish();
                    return Err(diags.report(diag));
                }

                // check directions and build connections
                let interface_info = self
                    .ctx
                    .refs
                    .shared
                    .elaboration_arenas
                    .interface_info(connector_view.inner.interface);
                let view_info = &interface_info.views[connector_view.inner.view_index];

                let mut any_input = false;
                let mut any_output = false;

                let mut result_connections = vec![];

                for port_index in 0..interface_info.ports.len() {
                    let (_, connector_dir) = &view_info.port_dirs.as_ref_ok()?[port_index];

                    // check direction
                    let (value_dir, value_signal, value_ir) = match value_signals {
                        WireOrPort::Port(ports) => {
                            let port = ports[port_index];
                            let info = &self.ctx.ports[port];
                            (Some(info.direction), Signal::Port(port), IrWireOrPort::Port(info.ir))
                        }
                        WireOrPort::Wire((wires, ir_wires)) => {
                            let wire = wires[port_index];
                            (None, Signal::Wire(wire), IrWireOrPort::Wire(ir_wires[port_index]))
                        }
                    };
                    if let Some(value_dir) = value_dir
                        && connector_dir.inner != value_dir.inner
                    {
                        let diag = Diagnostic::new(format!(
                            "direction mismatch for interface port `{}`",
                            interface_info.ports[port_index].id.str(source)
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
                            IrPortConnection::Input(Spanned::new(value_expr.span, value_ir.as_signal()))
                        }
                        PortDirection::Output => {
                            any_output = true;

                            // report driver
                            let driver = Driver::InstancePortConnection(stmt_index, connection_index, port_index);
                            let target = Spanned::new(value_expr.span, value_signal);
                            self.drivers.report_assignment(self.ctx, driver, target)?;

                            IrPortConnection::Output(Some(value_ir))
                        }
                    };

                    // build signal
                    let signal = ConnectionSignal::Signal(Polarized {
                        inverted: false,
                        signal: value_signal,
                    });
                    result_connections.push((
                        connector_singles[port_index],
                        signal,
                        Spanned::new(connection.span, ir_connection),
                    ))
                }

                // check domains
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

    fn pass_2_check_inferred(&mut self) -> DiagResult {
        let ctx = &*self.ctx;
        let diags = ctx.refs.diags;

        let mut any_err = Ok(());

        for (_, wire_info) in &ctx.wires {
            match wire_info {
                WireInfo::Single(WireInfoSingle { id, domain, typed }) => {
                    if let Ok(None) = domain {
                        any_err = Err(diags.report_simple(
                            "could not infer domain for wire",
                            id.span(),
                            "wire declared here",
                        ));
                    }
                    if let Ok(None) = typed {
                        any_err =
                            Err(diags.report_simple("could not infer type for wire", id.span(), "wire declared here"));
                    }

                    // fill in domain debug info
                    if let (Ok(Some(domain)), Ok(Some(typed))) = (domain, typed) {
                        self.ir_wires[typed.ir].debug_info_domain = domain.inner.diagnostic_string(ctx);
                    }
                }
                WireInfo::Interface(_) => {
                    // nothing to do here, interfaces are checked and debug info is filled in separately
                    // fill in domain debug info
                }
            }
        }

        for (_, info) in &ctx.wire_interfaces {
            let WireInterfaceInfo {
                id,
                domain,
                interface,
                wires: _,
                ir_wires,
            } = info;

            if let Ok(None) = domain {
                any_err = Err(diags.report_simple(
                    "could not infer type for wire interface",
                    id.span(),
                    "wire interface declared here",
                ));
            }
            let _: &Spanned<ElaboratedInterface> = interface;

            // fill in domain debug info
            if let Ok(Some(domain)) = domain {
                let domain_debug_str = domain.inner.diagnostic_string(ctx);
                for &ir in ir_wires {
                    self.ir_wires[ir].debug_info_domain = domain_debug_str.clone();
                }
            }
        }

        for (_, derp) in &ctx.registers {
            let RegisterInfo { id, domain, ty, ir } = derp;

            if let Ok(None) = domain {
                any_err = Err(diags.report_simple(
                    "could not infer domain for register",
                    id.span(),
                    "register declared here",
                ));
            }
            let _: &Spanned<HardwareType> = ty;

            if let Ok(Some(domain)) = domain {
                self.ir_registers[*ir].debug_info_domain = domain.inner.diagnostic_string(ctx);
            }
        }

        any_err
    }

    fn pass_2_check_drivers_and_populate_resets(&mut self) -> DiagResult {
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
            let driver_err = self.check_exactly_one_driver("port", &port_info.name, port_info.span, drivers);
            any_err = any_err.and(driver_err.map(|_| ()))
        }
        for (&wire, drivers) in wire_drivers {
            let wire_info = &self.ctx.wires[wire];
            let wire_name = wire_info.diagnostic_str();
            let driver_err = self.check_exactly_one_driver("wire", wire_name, wire_info.decl_span(), drivers);
            any_err = any_err.and(driver_err.map(|_| ()));
        }

        // registers: check and collect resets
        for (&reg, drivers) in reg_drivers {
            // TODO allow zero drivers for registers, just turn them into wires with the init expression as the value
            //  (still emit a warning)
            let reg_info = &self.ctx.registers[reg];
            let reg_name = reg_info.id.diagnostic_str();
            let driver_err = self.check_exactly_one_driver("register", reg_name, reg_info.id.span(), drivers);

            let maybe_err = match any_err.and(driver_err) {
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

    fn check_exactly_one_driver(
        &self,
        kind: &str,
        name: &str,
        decl_span: Span,
        drivers: &DiagResult<IndexMap<Driver, Span>>,
    ) -> DiagResult<(Driver, Span)> {
        let diags = self.ctx.refs.diags;
        let drivers = drivers.as_ref_ok()?;

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
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        id: &MaybeIdentifier<Spanned<ArcOrRef<str>>>,
        decl_span: Span,
        decl: &WireDeclaration,
    ) -> DiagResult<(NamedValue, Option<Spanned<IrCombinatorialProcess>>)> {
        let ctx = &mut self.ctx;
        let refs = ctx.refs;
        let diags = ctx.refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        // declaration and visibility are handled in the caller
        let &WireDeclaration {
            vis: _,
            span_keyword,
            id: _,
            kind,
        } = decl;
        let id_owned = id
            .as_ref()
            .map_id(|id| id.as_ref().map_inner(|s| s.as_ref().to_owned()));

        // eval domain and value
        match kind {
            WireDeclarationKind::Normal {
                domain_ty,
                assign_span_and_value,
            } => {
                // create wire immediately, we'll fill in the domain and type later
                let wire = ctx.wires.push(WireInfo::Single(WireInfoSingle {
                    id: id_owned,
                    domain: Ok(None),
                    typed: Ok(None),
                }));

                let (domain, ty) = match domain_ty {
                    WireDeclarationDomainTyKind::Clock { span_clock } => (
                        Some(Spanned::new(span_clock, ValueDomain::Clock)),
                        Some(Spanned::new(span_clock, HardwareType::Bool)),
                    ),
                    WireDeclarationDomainTyKind::Normal { domain, ty } => {
                        let domain = domain
                            .map(|domain| {
                                let domain = ctx.eval_domain(scope, flow_parent, domain)?;
                                Ok(domain.map_inner(ValueDomain::from_domain_kind))
                            })
                            .transpose();
                        let ty = ty
                            .map(|ty| ctx.eval_expression_as_ty_hardware(scope, flow_parent, ty, "wire"))
                            .transpose();

                        let domain = domain?;
                        let ty = ty?;

                        (domain, ty)
                    }
                };

                // eval value and infer domain and ty from it
                let process = match assign_span_and_value {
                    None => {
                        // just set the domain and type
                        if let Some(domain) = domain {
                            ctx.wires[wire].suggest_domain(&mut ctx.wire_interfaces, domain)?;
                        }
                        if let Some(ty) = ty.as_ref() {
                            ctx.wires[wire].suggest_ty(refs, &ctx.wire_interfaces, &mut self.ir_wires, ty.as_ref())?;
                        }

                        None
                    }
                    Some((assign_span, value)) => {
                        // eval value
                        let expected_ty = ty.as_ref().map_or(Type::Any, |ty| ty.inner.as_type());

                        let flow_kind = HardwareProcessKind::WireExpression {
                            span_keyword,
                            span_init: value.span,
                        };
                        let mut flow = FlowHardwareRoot::new(
                            flow_parent,
                            value.span,
                            flow_kind,
                            &mut self.ir_wires,
                            &mut self.ir_registers,
                        );
                        let value = ctx.eval_expression(scope, &mut flow.as_flow(), &expected_ty, value)?;
                        let (ir_vars, mut ir_block) = flow.finish();

                        // infer or check domain
                        let value_domain = value.as_ref().map_inner(|v| v.domain());
                        let domain = match domain {
                            None => Ok(value_domain),
                            Some(domain) => ctx
                                .check_valid_domain_crossing(
                                    assign_span,
                                    domain,
                                    value_domain,
                                    "wire declaration value",
                                )
                                .map(|()| domain),
                        };

                        // infer or check type
                        let ty = match ty {
                            None => match value.inner.ty().as_hardware_type(elab) {
                                Ok(ty) => Ok(Spanned::new(value.span, ty)),
                                Err(e) => {
                                    let _: NonHardwareType = e;
                                    let err_msg = format!(
                                        "value with type `{}` cannot be represented in hardware",
                                        value.inner.ty().value_string(elab)
                                    );
                                    let diag = Diagnostic::new("cannot assign non-hardware value to wire")
                                        .add_error(value.span, err_msg)
                                        .add_info(assign_span, "assignment to wire here")
                                        .finish();
                                    Err(diags.report(diag))
                                }
                            },
                            Some(ty) => {
                                let reason = TypeContainsReason::Assignment {
                                    span_target: id.span(),
                                    span_target_ty: ty.span,
                                };
                                check_type_contains_value(diags, elab, reason, &ty.inner.as_type(), value.as_ref())
                                    .map(|()| ty)
                            }
                        };

                        let domain = domain?;
                        let ty = ty?;

                        // create the wire by suggesting the domain and ty
                        let wire_info = &mut ctx.wires[wire];
                        wire_info.suggest_domain(&mut ctx.wire_interfaces, domain)?;
                        let wire_info_typed =
                            wire_info.suggest_ty(refs, &ctx.wire_interfaces, &mut self.ir_wires, ty.as_ref())?;

                        // append final assignment to process
                        // TODO if the value is a signal and there are no statement or locals, skip the process (to avoid delta cycle)
                        // TODO maybe processes should be changed to explicitly list the things they're driving?
                        let expr_hw = value.inner.as_ir_expression_unchecked(
                            refs,
                            &mut ctx.large,
                            value.span,
                            wire_info_typed.ty.inner,
                        )?;
                        let target = IrAssignmentTarget::simple(wire_info_typed.ir.into());

                        let stmt = IrStatement::Assign(target, expr_hw);
                        ir_block.statements.push(Spanned::new(decl_span, stmt));

                        let process = IrCombinatorialProcess {
                            locals: ir_vars,
                            block: ir_block,
                        };
                        let process = Spanned::new(decl_span, process);

                        Some(process)
                    }
                };

                let mut drivers = IndexMap::new();
                if let Some(process) = &process {
                    drivers.insert_first(Driver::WireDeclaration, process.span);
                }
                self.drivers.wire_drivers.insert_first(wire, Ok(drivers));

                Ok((NamedValue::Wire(wire), process))
            }
            WireDeclarationKind::Interface {
                domain,
                span_keyword,
                interface,
            } => {
                // eval domain and interface
                let domain = domain
                    .map(|domain| {
                        let domain = ctx.eval_domain(scope, flow_parent, domain)?;
                        Ok(domain.map_inner(ValueDomain::from_domain_kind))
                    })
                    .transpose();

                let interface = ctx
                    .eval_expression_as_compile(
                        scope,
                        flow_parent,
                        &Type::Interface,
                        interface,
                        Spanned::new(interface.span, "wire interface"),
                    )
                    .and_then(|interface| match interface.inner {
                        CompileValue::Simple(SimpleCompileValue::Interface(interface_inner)) => {
                            Ok(Spanned::new(interface.span, interface_inner))
                        }
                        _ => {
                            let diag = Diagnostic::new("expected interface value")
                                .add_error(interface.span, "got non-interface expression")
                                .add_info(
                                    span_keyword,
                                    "expected an interface because of this wire interface declaration",
                                )
                                .finish();
                            Err(diags.report(diag))
                        }
                    });

                let domain = domain?;
                let interface = interface?;

                // create interface wire
                let wire_interface = ctx.wire_interfaces.push(WireInterfaceInfo {
                    id: id_owned.clone(),
                    domain: Ok(domain),
                    interface,
                    // these will be filled in immediately after this
                    wires: vec![],
                    ir_wires: vec![],
                });

                // create inner wires
                let interface_info = elab.interface_info(interface.inner);
                let mut wires = vec![];
                let mut ir_wires = vec![];
                for (port_index, (port_name, port_info)) in enumerate(&interface_info.ports) {
                    let ElaboratedInterfacePortInfo { id, ty } = port_info;
                    let ty = ty.as_ref_ok()?;

                    let diagnostic_str = format!("{}.{}", id_owned.diagnostic_str(), port_name);
                    let ir_name = format!("{}_{}", id_owned.diagnostic_str(), port_name);

                    let wire_ir_info = IrWireInfo {
                        ty: ty.inner.as_ir(refs),
                        debug_info_id: Spanned::new(id.span, Some(ir_name)),
                        debug_info_ty: ty.inner.value_string(elab),
                        // will be filled in later during the inference checking pass
                        debug_info_domain: String::new(),
                    };
                    let wire_ir = self.ir_wires.push(wire_ir_info);

                    let wire_info = WireInfoInInterface {
                        decl_span,
                        interface: Spanned::new(interface.span, wire_interface),
                        index: port_index,
                        diagnostic_string: diagnostic_str,
                        ir: wire_ir,
                    };
                    let wire = ctx.wires.push(WireInfo::Interface(wire_info));

                    self.drivers.wire_drivers.insert_first(wire, Ok(IndexMap::new()));

                    wires.push(wire);
                    ir_wires.push(wire_ir);
                }

                let wire_interface_info = &mut ctx.wire_interfaces[wire_interface];
                wire_interface_info.wires = wires;
                wire_interface_info.ir_wires = ir_wires;

                Ok((NamedValue::WireInterface(wire_interface), None))
            }
        }
    }

    fn elaborate_module_declaration_reg(
        &mut self,
        scope_body: &Scope,
        flow_body: &FlowCompile,
        id: &MaybeIdentifier<Spanned<ArcOrRef<str>>>,
        decl: Spanned<&RegDeclaration>,
    ) -> DiagResult<RegisterInit> {
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;
        let elab = &ctx.refs.shared.elaboration_arenas;

        let mut flow_inner = flow_body.new_child_isolated();

        // declaration and visibility are handled in the caller
        let &RegDeclaration {
            vis: _,
            span_keyword,
            id: _,
            sync,
            ty,
            init,
        } = decl.inner;

        // evaluate
        let sync = sync
            .map(|sync| {
                sync.map_inner(|sync| ctx.eval_domain_sync(scope_body, &mut flow_inner, sync))
                    .transpose()
            })
            .transpose();

        let ty = ctx.eval_expression_as_ty_hardware(scope_body, &mut flow_inner, ty, "register")?;
        let init_raw = ctx.eval_expression_as_compile_or_undefined(
            scope_body,
            &mut flow_inner,
            &ty.inner.as_type(),
            init,
            Spanned::new(span_keyword, "register reset value"),
        );
        let sync = sync?;

        // check type
        let init = init_raw.and_then(|init_raw| {
            let reason = TypeContainsReason::Assignment {
                span_target: id.span(),
                span_target_ty: ty.span,
            };
            check_type_contains_value(diags, elab, reason, &ty.inner.as_type(), init_raw.as_ref())?;
            Ok(init_raw)
        });

        // build register
        let debug_info_domain = sync
            .as_ref()
            .map_or_else(|| "inferred".to_string(), |sync| sync.inner.diagnostic_string(ctx));
        let ir_reg = self.ir_registers.push(IrRegisterInfo {
            ty: ty.inner.as_ir(ctx.refs),
            debug_info_id: id.spanned_string(),
            debug_info_ty: ty.inner.value_string(elab),
            debug_info_domain,
        });
        let reg = ctx.registers.push(RegisterInfo {
            id: id
                .as_ref()
                .map_id(|id| id.as_ref().map_inner(|s| s.as_ref().to_owned())),
            domain: Ok(sync),
            ty,
            ir: ir_reg,
        });
        Ok(RegisterInit { reg, init })
    }

    fn elaborate_module_declaration_reg_out_port(
        &mut self,
        scope_body: &Scope,
        flow_body: &mut FlowCompile,
        decl_span: Span,
        decl: &RegOutPortMarker,
        is_root_block: bool,
    ) -> DiagResult<(Port, RegisterInit)> {
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;
        let source = ctx.refs.fixed.source;
        let elab = &ctx.refs.shared.elaboration_arenas;

        let &RegOutPortMarker { id, init } = decl;
        let id = id.spanned_str(source);

        if !is_root_block {
            return Err(diags.report_simple(
                "register output markers are only allowed at the top level of a module",
                decl_span,
                "register output in child block",
            ));
        }

        // find port
        let port = scope_body.find(diags, id).and_then(|port| match port.value {
            ScopedEntry::Named(NamedValue::Port(port)) => Ok(port),
            _ => {
                let diag = Diagnostic::new("register port marker needs to be on a port")
                    .add_error(id.span, "non-port value here")
                    .add_info(port.defining_span, "declared here")
                    .finish();
                Err(diags.report(diag))
            }
        });
        let port = port?;
        let port_info = &ctx.ports[port];
        let port_ty = port_info.ty.inner.as_type();

        // evaluate init
        let init_raw = ctx.eval_expression_as_compile_or_undefined(
            scope_body,
            flow_body,
            &port_ty,
            init,
            Spanned::new(init.span, "register reset value"),
        );

        // check port is output
        let port_info = &ctx.ports[port];
        let mut direction_err = Ok(());
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
        let init = init_raw.and_then(|init_raw| {
            let reason = TypeContainsReason::Assignment {
                span_target: id.span,
                span_target_ty: port_info.ty.span,
            };
            check_type_contains_value(diags, elab, reason, &port_ty, init_raw.as_ref())?;
            Ok(init_raw)
        });

        // build register
        let ir_reg = self.ir_registers.push(IrRegisterInfo {
            ty: port_info.ty.inner.as_ir(ctx.refs),
            debug_info_id: id.map_inner(|s| Some(s.to_owned())),
            debug_info_ty: port_info.ty.inner.value_string(elab),
            debug_info_domain: domain.diagnostic_string(ctx),
        });
        let domain_spanned = Spanned {
            span: port_info.domain.span,
            inner: domain.map_signal(|p| p.map_inner(Signal::Port)),
        };
        let reg = ctx.registers.push(RegisterInfo {
            id: MaybeIdentifier::Identifier(id.map_inner(str::to_owned)),
            domain: Ok(Some(domain_spanned)),
            ty: port_info.ty.clone(),
            ir: ir_reg,
        });
        Ok((port, RegisterInit { reg, init }))
    }
}

fn process_todo_and_decls<'a>(
    diags: &Diagnostics,
    scope: &mut Scope,
    todo_children: &mut Vec<ModuleChildTodo<'a>>,
    decls_children: &mut PublicDeclarations<'a>,
    todo: ModuleTodo<'a>,
    decls: PublicDeclarations<'a>,
) {
    if !todo.children.is_empty() {
        todo_children.push(ModuleChildTodo::Nested(todo));
    }

    for (id, entry) in &decls {
        let id_ref = id.as_ref_ok().map(|id| id.as_ref().map_id(|id| id.as_ref()));
        scope.maybe_declare(diags, id_ref, *entry);
    }
    decls_children.extend(decls);
}

enum ConnectionSignal {
    Signal(DomainSignal),
    Dummy(Span),
    Expression(Span),
}

#[derive(Debug)]
struct RegisterInit {
    reg: Register,
    init: DiagResult<Spanned<MaybeUndefined<CompileValue>>>,
}

/// The usize fields are here to keep different drivers of the same type separate for Eq and Hash,
/// so we can correctly determining whether there are multiple drivers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Driver {
    // wire-like
    WireDeclaration,
    OutputPortConnectionToReg,
    InstancePortConnection(usize, usize, usize),
    CombinatorialBlock(usize),

    // register-like
    ClockedBlock(usize),
}

#[derive(Debug, Copy, Clone)]
pub enum DriverKind {
    ClockedBlock,
    WiredConnection,
}

// TODO this can all be simplified
impl Driver {
    fn kind(&self) -> DriverKind {
        match self {
            Driver::WireDeclaration
            | Driver::CombinatorialBlock(_)
            | Driver::InstancePortConnection(_, _, _)
            | Driver::OutputPortConnectionToReg => DriverKind::WiredConnection,
            Driver::ClockedBlock(_) => DriverKind::ClockedBlock,
        }
    }
}

// TODO this can be simplified now that we check driver kind validness at assignment time
// TODO allow multiple non-overlapping drivers for wires, mostly useful for generated instances
#[derive(Debug, Eq, PartialEq)]
struct Drivers {
    // For each signal, for each driver, the first span.
    // This span will be used in error messages in case there are multiple drivers for the same signal.
    output_port_drivers: IndexMap<Port, DiagResult<IndexMap<Driver, Span>>>,
    reg_drivers: IndexMap<Register, DiagResult<IndexMap<Driver, Span>>>,
    wire_drivers: IndexMap<Wire, DiagResult<IndexMap<Driver, Span>>>,
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
                    drivers.output_port_drivers.insert_first(port, Ok(IndexMap::new()));
                }
            }
        }

        drivers
    }

    pub fn report_assignment(
        &mut self,
        ctx: &CompileItemContext,
        driver: Driver,
        target: Spanned<Signal>,
    ) -> DiagResult {
        let diags = ctx.refs.diags;

        // check valid combination
        let driver = match (target.inner, driver.kind()) {
            // correct
            (Signal::Port(_) | Signal::Wire(_), DriverKind::WiredConnection) => Ok(driver),
            (Signal::Register(_), DriverKind::ClockedBlock) => Ok(driver),

            // wrong (this should have been checked during block elaboration already)
            (Signal::Port(_), DriverKind::ClockedBlock) => {
                Err(diags.report_internal_error(target.span, "clocked block driving port"))
            }
            (Signal::Wire(_), DriverKind::ClockedBlock) => {
                Err(diags.report_internal_error(target.span, "clocked block driving wire"))
            }
            (Signal::Register(_), DriverKind::WiredConnection) => {
                Err(diags.report_internal_error(target.span, "wired connection driving register"))
            }
        };

        // record driver
        fn record<T: Hash + Eq + Debug>(
            diags: &Diagnostics,
            map: &mut IndexMap<T, DiagResult<IndexMap<Driver, Span>>>,
            driver: DiagResult<Driver>,
            target: T,
            target_span: Span,
        ) -> DiagResult {
            let inner = map.get_mut(&target).ok_or_else(|| {
                diags.report_internal_error(target_span, "failed to record driver, target was not registered")
            })?;

            match driver {
                Err(e) => {
                    *inner = Err(e);
                    Err(e)
                }
                Ok(driver) => {
                    // TODO maybe we don't need to propagate this error, it's just about a previous incorrect driver
                    let inner = inner.as_ref_mut_ok()?;
                    inner.entry(driver).or_insert(target_span);
                    Ok(())
                }
            }
        }

        let diags = ctx.refs.diags;
        match target.inner {
            Signal::Port(port) => record(diags, &mut self.output_port_drivers, driver, port, target.span),
            Signal::Wire(wire) => record(diags, &mut self.wire_drivers, driver, wire, target.span),
            Signal::Register(reg) => record(diags, &mut self.reg_drivers, driver, reg, target.span),
        }
    }
}

fn pull_register_init_into_process(
    ctx: &mut CompileItemContext,
    clocked_block_statement_index_to_process_index: &IndexMap<usize, usize>,
    children: &mut Vec<Spanned<Child>>,
    register_initial_values: &IndexMap<Register, DiagResult<Spanned<MaybeUndefined<CompileValue>>>>,
    reg: Register,
    driver: Driver,
    driver_first_span: Span,
) -> DiagResult {
    let diags = ctx.refs.diags;
    let reg_info = &ctx.registers[reg];

    let reg_domain = reg_info.domain?.ok_or_else(|| {
        diags.report_internal_error(reg_info.id.span(), "no inferred domain even though there are drivers")
    })?;

    if let Driver::ClockedBlock(stmt_index) = driver
        && let Some(&process_index) = clocked_block_statement_index_to_process_index.get(&stmt_index)
        && let Child::Clocked(process) = &mut children[process_index].inner
        && let Some(init) = register_initial_values.get(&reg)
    {
        let init = init.clone()?;

        match init.inner {
            MaybeUndefined::Undefined => {
                // we don't need to reset
            }
            MaybeUndefined::Defined(init_inner) => {
                let init_ir =
                    init_inner.as_ir_expression_unchecked(ctx.refs, &mut ctx.large, init.span, &reg_info.ty.inner)?;

                match &mut process.reset {
                    Some(reset) => {
                        // check that the reset style matches
                        let reset_style_err = |block: &str, reg: &str| {
                            let diag = Diagnostic::new("reset style mismatch")
                                .add_error(driver_first_span, "block drives register here")
                                .add_info(reset.kind.span, format!("block defined with {block} reset here"))
                                .add_info(reg_domain.span, format!("register defined with {reg} reset here"))
                                .finish();
                            diags.report(diag)
                        };
                        match (reset.kind.inner, reg_domain.inner.reset) {
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
                        let diag =
                            Diagnostic::new("clocked block without reset cannot drive register with reset value")
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

    Err(diags.report_internal_error(
        reg_info.id.span(),
        "failure while pulling reset value into clocked process",
    ))
}
