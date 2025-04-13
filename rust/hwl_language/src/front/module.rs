use crate::front::block::{BlockDomain, TypedIrExpression};
use crate::front::check::{
    check_type_contains_compile_value, check_type_contains_type, check_type_contains_value, check_type_is_bool_compile,
    TypeContainsReason,
};
use crate::front::compile::{
    ArenaPorts, ArenaVariables, CompileItemContext, CompileRefs, Port, PortInfo, Register, RegisterInfo, Wire, WireInfo,
};
use crate::front::context::{
    CompileTimeExpressionContext, ExpressionContext, ExtraRegister, IrBuilderExpressionContext,
};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::expression::EvaluatedId;
use crate::front::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrModule, IrModuleChild,
    IrModuleInfo, IrModuleInstance, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrStatement,
    IrVariables, IrWire, IrWireInfo, IrWireOrPort,
};
use crate::front::misc::{DomainSignal, Polarized, PortDomain, ScopedEntry, Signal, ValueDomain};
use crate::front::scope::{DeclaredValueSingle, Scope};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, MaybeCompile, NamedValue};
use crate::front::variables::VariableValues;
use crate::syntax::ast::{self, ForStatement, IfCondBlockPair, IfStatement, ModuleStatement, ModuleStatementKind};
use crate::syntax::ast::{
    Args, Block, ClockedBlock, CombinatorialBlock, DomainKind, ExpressionKind, Identifier, MaybeIdentifier,
    ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle, PortConnection, PortDirection,
    RegDeclaration, RegOutPortMarker, Spanned, SyncDomain, WireDeclaration, WireKind,
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
use itertools::{enumerate, zip_eq, Either, Itertools};
use std::hash::Hash;

struct BodyElaborationContext<'c, 'a, 's> {
    ctx: &'c mut CompileItemContext<'a, 's>,

    ir_ports: Arena<IrPort, IrPortInfo>,
    ir_wires: Arena<IrWire, IrWireInfo>,
    ir_registers: Arena<IrRegister, IrRegisterInfo>,
    drivers: Drivers,

    register_initial_values: IndexMap<Register, Result<Spanned<CompileValue>, ErrorGuaranteed>>,
    out_port_register_connections: IndexMap<Port, Register>,

    clocked_block_statement_index_to_process_index: IndexMap<usize, usize>,
    processes: Vec<Result<IrModuleChild, ErrorGuaranteed>>,
}

type SignalsInScope = Vec<Spanned<Signal>>;

// TODO rename, this should really be "module ports, viewed from the outside"
new_index_type!(pub InstancePort);
pub type InstancePorts = Arena<InstancePort, PortInfo<InstancePort>>;

pub struct ElaboratedModuleParams {
    module: AstRefModule,
    params: Option<Vec<(Identifier, CompileValue)>>,
}

pub struct ElaboratedModuleHeader {
    module: AstRefModule,
    params: Option<Vec<(Identifier, CompileValue)>>,
    ports: Arena<Port, PortInfo>,
    ports_ir: Arena<IrPort, IrPortInfo>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleElaborationCacheKey {
    module: AstRefModule,
    params: Option<Vec<CompileValue>>,
}

impl ElaboratedModuleParams {
    // TODO these are equivalent, maybe we can remove the layer of redundancy
    pub fn cache_key(&self) -> ModuleElaborationCacheKey {
        let &ElaboratedModuleParams { module, ref params } = self;
        let param_values = params
            .as_ref()
            .map(|params| params.iter().map(|(_, v)| v.clone()).collect());
        ModuleElaborationCacheKey {
            module,
            params: param_values,
        }
    }
}

pub struct ElaboratedModule {
    pub module_ast: AstRefModule,
    pub module_ir: IrModule,
    pub ports: InstancePorts,
}

impl<'s> CompileRefs<'_, 's> {
    pub fn elaborate_module_params_new(
        &self,
        module: AstRefModule,
        args: Option<Args<Option<Identifier>, Spanned<CompileValue>>>,
    ) -> Result<ElaboratedModuleParams, ErrorGuaranteed> {
        let diags = self.diags;
        let &ast::ItemDefModule {
            span: def_span,
            vis: _,
            id: _,
            ref params,
            ports: _,
            body: _,
        } = &self.fixed.parsed[module];

        // create header scope
        let scope_file = self.shared.file_scope(module.file())?;

        let mut ctx = CompileItemContext::new(*self, None, ArenaVariables::new(), ArenaPorts::new());

        // evaluate params/args
        let param_values = match (params, args) {
            (None, None) => None,
            (Some(params), Some(args)) => {
                // We can't use the scope returned here, since it is only valid for the current variables arena,
                //   which will be different during body elaboration. Instead, we'll recreate the scope from the returned parameter values.
                let mut vars = VariableValues::new_root(&ctx.variables);
                let (_, param_values) = ctx.match_args_to_params_and_typecheck(&mut vars, scope_file, params, &args)?;
                Some(param_values)
            }
            _ => throw!(diags.report_internal_error(def_span, "mismatched generic arguments presence")),
        };

        Ok(ElaboratedModuleParams {
            module,
            params: param_values,
        })
    }

    pub fn elaborate_module_ports_new(
        self,
        params: ElaboratedModuleParams,
    ) -> Result<(InstancePorts, ElaboratedModuleHeader), ErrorGuaranteed> {
        let ElaboratedModuleParams { module, params } = params;
        let &ast::ItemDefModule {
            span: def_span,
            vis: _,
            id: _,
            params: _,
            ref ports,
            body: _,
        } = &self.fixed.parsed[module];

        // reconstruct header scope
        let mut ctx = CompileItemContext::new(self, None, ArenaVariables::new(), ArenaPorts::new());
        let mut vars = VariableValues::new_root(&ctx.variables);

        let file_scope = self.shared.file_scope(module.file())?;
        let scope_params = rebuild_params_scope(&mut ctx.variables, &mut vars, file_scope, def_span, &params);

        // TODO we actually need a full context here?
        let (ports_external, ports_ir) =
            self.elaborate_module_ports_impl(&mut ctx, &scope_params, &mut vars, ports, def_span);
        let ports_internal = ctx.ports;

        let header: ElaboratedModuleHeader = ElaboratedModuleHeader {
            module,
            params,
            ports: ports_internal,
            ports_ir,
        };
        Ok((ports_external, header))
    }

    pub fn elaborate_module_body_new(&self, ports: ElaboratedModuleHeader) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let ElaboratedModuleHeader {
            module,
            params,
            ports,
            ports_ir,
        } = ports;
        let &ast::ItemDefModule {
            span: def_span,
            vis: _,
            ref id,
            params: _,
            ports: _,
            ref body,
        } = &self.fixed.parsed[module];

        self.check_should_stop(id.span())?;

        let mut ctx = CompileItemContext::new(*self, None, ArenaVariables::new(), ports);
        let mut vars = VariableValues::new_root(&ctx.variables);

        // rebuild scopes
        let file_scope = self.shared.file_scope(module.file())?;
        let scope_params = rebuild_params_scope(&mut ctx.variables, &mut vars, file_scope, def_span, &params);
        let scope_ports = rebuild_ports_scope(&scope_params, def_span, &ctx.ports);

        self.elaborate_module_body_impl(&mut ctx, &vars, &scope_ports, id, params.clone(), ports_ir, body)
    }

    fn elaborate_module_ports_impl<'p>(
        &self,
        ctx: &mut CompileItemContext,
        scope_params: &'p Scope<'p>,
        vars: &mut VariableValues,
        ports: &Spanned<Vec<ModulePortItem>>,
        module_def_span: Span,
    ) -> (InstancePorts, Arena<IrPort, IrPortInfo>) {
        let diags = self.diags;

        let mut scope_ports = Scope::new_child(ports.span.join(Span::single_at(module_def_span.end)), scope_params);

        let mut ports_external: InstancePorts = Arena::new();
        let mut port_to_external: IndexMap<Port, InstancePort> = IndexMap::new();
        let mut ports_ir = Arena::new();

        for port_item in &ports.inner {
            match port_item {
                ModulePortItem::Single(port_item) => {
                    let &ModulePortSingle {
                        span: _,
                        ref id,
                        direction,
                        ref kind,
                    } = port_item;

                    // eval kind
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
                            ctx.eval_port_domain(&mut scope_ports, domain)
                                .map(|d| d.map_inner(PortDomain::Kind)),
                            ctx.eval_expression_as_ty_hardware(&mut scope_ports, vars, ty, "port"),
                        ),
                    };

                    // build entry
                    let entry = result_pair(domain, ty).map(|(domain, ty)| {
                        let port = push_port(
                            ctx,
                            &mut ports_external,
                            &mut port_to_external,
                            &mut ports_ir,
                            id,
                            direction,
                            domain,
                            ty,
                        );
                        ScopedEntry::Named(NamedValue::Port(port))
                    });

                    scope_ports.declare(diags, id, entry);
                }
                ModulePortItem::Block(port_item) => {
                    let ModulePortBlock { span: _, domain, ports } = port_item;

                    // eval domain
                    let domain = ctx
                        .eval_port_domain(&scope_ports, domain)
                        .map(|d| d.map_inner(PortDomain::Kind));

                    for port in ports {
                        let &ModulePortInBlock {
                            span: _,
                            ref id,
                            direction,
                            ref ty,
                        } = port;

                        // eval ty
                        let ty = ctx.eval_expression_as_ty_hardware(&scope_ports, vars, ty, "port");

                        // build entry
                        let entry = result_pair(domain.as_ref_ok(), ty).map(|(&domain, ty)| {
                            let port = push_port(
                                ctx,
                                &mut ports_external,
                                &mut port_to_external,
                                &mut ports_ir,
                                id,
                                direction,
                                domain,
                                ty,
                            );
                            ScopedEntry::Named(NamedValue::Port(port))
                        });

                        scope_ports.declare(diags, id, entry);
                    }
                }
            }
        }

        assert_eq!(ctx.ports.len(), ports_external.len());
        assert_eq!(ctx.ports.len(), ports_ir.len());

        (ports_external, ports_ir)
    }

    fn elaborate_module_body_impl(
        &self,
        ctx_item: &mut CompileItemContext,
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
            signals_in_scope.push(Spanned::new(port_info.id.span, Signal::Port(port)));
        }

        let mut ctx = BodyElaborationContext {
            ctx: ctx_item,
            ir_ports,
            ir_wires: Arena::new(),
            ir_registers: Arena::new(),
            drivers,

            register_initial_values: IndexMap::new(),
            out_port_register_connections: IndexMap::new(),
            clocked_block_statement_index_to_process_index: IndexMap::new(),
            processes: vec![],
        };

        // process declarations
        ctx.elaborate_block(scope_header, vars, &mut signals_in_scope, body);

        // stop if any errors have happened so far, we don't want spurious errors about drivers
        for p in &ctx.processes {
            p.as_ref_ok()?;
        }

        // check driver validness
        // TODO more checking: combinatorial blocks can't read values they will later write,
        //   unless they have already written them
        ctx.pass_2_check_drivers_and_populate_resets()?;

        // create process for registered output ports
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
            ctx.processes.push(Ok(IrModuleChild::CombinatorialProcess(process)));
        }

        // return result
        let processes = ctx.processes.into_iter().try_collect()?;

        Ok(IrModuleInfo {
            ports: ctx.ir_ports,
            registers: ctx.ir_registers,
            wires: ctx.ir_wires,
            children: processes,
            debug_info_id: def_id.clone(),
            debug_info_generic_args: params,
        })
    }
}

fn rebuild_params_scope<'p>(
    variables: &mut ArenaVariables,
    vars: &mut VariableValues,
    file_scope: &'p Scope<'p>,
    span: Span,
    params: &Option<Vec<(Identifier, CompileValue)>>,
) -> Scope<'p> {
    let mut scope_params = Scope::new_child(span, file_scope);
    if let Some(params) = params {
        for (id, value) in params {
            let var = vars.var_new_immutable_init(
                variables,
                MaybeIdentifier::Identifier(id.clone()),
                id.span,
                MaybeCompile::Compile(value.clone()),
            );
            let declared = DeclaredValueSingle::Value {
                span: id.span,
                value: ScopedEntry::Named(NamedValue::Variable(var)),
            };
            scope_params.declare_already_checked(id.string.clone(), declared);
        }
    }
    scope_params
}

fn rebuild_ports_scope<'p>(params_scope: &'p Scope<'p>, span: Span, ports: &ArenaPorts) -> Scope<'p> {
    let mut scope_ports = Scope::new_child(span, params_scope);

    for (port, port_info) in ports {
        let declared = DeclaredValueSingle::Value {
            span: port_info.id.span,
            value: ScopedEntry::Named(NamedValue::Port(port)),
        };
        scope_ports.declare_already_checked(port_info.id.string.clone(), declared);
    }

    scope_ports
}

fn push_port(
    ctx: &mut CompileItemContext,
    ports_external: &mut InstancePorts,
    port_to_external: &mut IndexMap<Port, InstancePort>,
    ports_ir: &mut Arena<IrPort, IrPortInfo>,
    id: &Identifier,
    direction: Spanned<PortDirection>,
    domain: Spanned<PortDomain<Port>>,
    ty: Spanned<HardwareType>,
) -> Port {
    let ir_port = ports_ir.push(IrPortInfo {
        direction: direction.inner,
        ty: ty.inner.as_ir(),
        debug_info_id: id.clone(),
        debug_info_ty: ty.inner.clone(),
        debug_info_domain: domain.inner.to_diagnostic_string(&ctx),
    });
    let port = ctx.ports.push(PortInfo {
        id: id.clone(),
        direction,
        domain,
        ty: ty.clone(),
        ir: ir_port,
    });

    let port_external = ports_external.push(PortInfo {
        id: id.clone(),
        direction,
        domain: domain.map_inner(|p| p.map_inner(|p| *port_to_external.get(&p).unwrap())),
        ty,
        ir: ir_port,
    });

    port_to_external.insert_first(port, port_external);

    port
}

impl<'s> CompileItemContext<'_, 's> {
    pub fn elaborate_module_header(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        header: &ast::ModuleInstanceHeader,
    ) -> Result<&'s ElaboratedModule, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let ast::ModuleInstanceHeader {
            span: _,
            span_keyword,
            module,
            generic_args,
        } = header;

        // eval module and generics
        let module: Result<Spanned<CompileValue>, ErrorGuaranteed> =
            self.eval_expression_as_compile(&scope, vars, module, "module instance");
        let generic_args = generic_args
            .as_ref()
            .map(|generic_args| {
                generic_args.try_map_inner_all(|a| self.eval_expression_as_compile(&scope, vars, a, "generic arg"))
            })
            .transpose();

        // check that module is indeed a module
        let module = module?;
        let reason = TypeContainsReason::InstanceModule(*span_keyword);
        check_type_contains_compile_value(diags, reason, &Type::Module, module.as_ref(), false)?;
        let module_ast_ref = match module.inner {
            CompileValue::Module(module_eval) => module_eval,
            _ => {
                return Err(
                    diags.report_internal_error(module.span, "expected module, should have already been checked")
                )
            }
        };

        // elaborate module
        // TODO split module header and body elaboration, so we can delay and parallelize body elaboration
        let generic_args = generic_args?;
        self.refs.elaborate_module(module_ast_ref, generic_args)
    }
}

impl BodyElaborationContext<'_, '_, '_> {
    fn elaborate_block(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        signals: &mut SignalsInScope,
        block: &Block<ModuleStatement>,
    ) {
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        let mut scope_inner = Scope::new_child(block.span, scope);
        let mut vars_inner = VariableValues::new_child(vars);

        let signals_len_before = signals.len();
        self.pass_0_declarations(&mut scope_inner, &mut vars_inner, signals, block);
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
                    let (wire, process) = result_pair_split(self.elaborate_module_declaration_wire(&scope, vars, decl));
                    let process = process.transpose();

                    if let Ok(wire) = wire {
                        let mut drivers = IndexMap::new();
                        if let Some(value) = &decl.value {
                            drivers.insert_first(Driver::WireDeclaration, value.span);
                        }
                        self.drivers.wire_drivers.insert_first(wire, drivers);
                    }

                    if let Some(process) = process {
                        self.processes.push(process.map(IrModuleChild::CombinatorialProcess));
                    }

                    let entry = wire.map(|wire| ScopedEntry::Named(NamedValue::Wire(wire)));
                    scope.maybe_declare(diags, decl.id.as_ref(), entry);
                    if let Ok(wire) = wire {
                        signals.push(Spanned::new(decl.id.span(), Signal::Wire(wire)));
                    }
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    // declare register that shadows the outer port, which is exactly what we want
                    match self.elaborate_module_declaration_reg_out_port(&scope, vars, decl) {
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
                        Err(e) => {
                            // don't do anything, as if the marker is not there
                            let _: ErrorGuaranteed = e;
                        }
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

        for (stmt_index, stmt) in enumerate(statements) {
            match &stmt.inner {
                // control flow
                ModuleStatementKind::Block(block) => {
                    self.elaborate_block(scope, vars, signals, block);
                }
                ModuleStatementKind::If(if_stmt) => match self.elaborate_if(scope, vars, signals, if_stmt) {
                    Ok(()) => {}
                    Err(e) => self.processes.push(Err(e)),
                },
                ModuleStatementKind::For(for_stmt) => {
                    let for_stmt = Spanned::new(stmt.span, for_stmt);
                    match self.elaborate_for(scope, vars, signals, for_stmt) {
                        Ok(()) => {}
                        Err(e) => self.processes.push(Err(e)),
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

                    let CombinatorialBlock {
                        span: _,
                        span_keyword: _,
                        block,
                    } = block;

                    let block_domain = BlockDomain::Combinatorial;

                    let mut report_assignment = |target: Spanned<Signal>| {
                        self.drivers
                            .report_assignment(diags, Driver::CombinatorialBlock(stmt_index), target)
                    };
                    let mut ctx = IrBuilderExpressionContext::new(&block_domain, None, &mut report_assignment);

                    let ir_block = self.ctx.elaborate_block(&mut ctx, scope, &mut vars_inner, block);

                    let ir_block = ir_block.and_then(|(ir_block, end)| {
                        let ir_variables = ctx.finish();
                        end.unwrap_outside_function_and_loop(diags)?;
                        let process = IrCombinatorialProcess {
                            locals: ir_variables,
                            block: ir_block,
                        };
                        Ok(IrModuleChild::CombinatorialProcess(process))
                    });

                    self.processes.push(ir_block);
                }
                ModuleStatementKind::ClockedBlock(block) => {
                    let &ClockedBlock {
                        span: _,
                        span_keyword,
                        ref domain,
                        ref block,
                    } = block;

                    let domain = domain
                        .as_ref()
                        .map_inner(|d| self.ctx.eval_domain_sync(scope, d))
                        .transpose();

                    let ir_process = domain.and_then(|domain| {
                        let ir_domain = SyncDomain {
                            clock: self.ctx.domain_signal_to_ir(&domain.inner.clock),
                            reset: self.ctx.domain_signal_to_ir(&domain.inner.reset),
                        };

                        let block_domain = Spanned {
                            span: span_keyword.join(domain.span),
                            inner: domain.inner,
                        };
                        let block_domain = BlockDomain::Clocked(block_domain);

                        let mut vars_inner = self.new_vars_for_process(vars, signals);
                        let mut report_assignment = |target: Spanned<Signal>| {
                            self.drivers
                                .report_assignment(diags, Driver::ClockedBlock(stmt_index), target)
                        };

                        let mut extra_regs = vec![];
                        let mut ctx = IrBuilderExpressionContext::new(
                            &block_domain,
                            Some((&mut self.ir_registers, &mut extra_regs)),
                            &mut report_assignment,
                        );

                        let ir_block = self.ctx.elaborate_block(&mut ctx, scope, &mut vars_inner, block);
                        let ir_variables = ctx.finish();

                        ir_block.and_then(|(ir_block, end)| {
                            end.unwrap_outside_function_and_loop(diags)?;

                            // will be filled in more later, during driver checking and merging
                            let mut on_reset = vec![];
                            for extra_reg in extra_regs {
                                let ExtraRegister { span, reg, init } = extra_reg;
                                if let Some(init) = init {
                                    let stmt = IrStatement::Assign(IrAssignmentTarget::register(reg), init);
                                    on_reset.push(Spanned::new(span, stmt));
                                }
                            }

                            let process = IrClockedProcess {
                                domain: Spanned {
                                    span: domain.span,
                                    inner: ir_domain,
                                },
                                locals: ir_variables,
                                on_clock: ir_block,
                                on_reset: IrBlock { statements: on_reset },
                            };
                            Ok(IrModuleChild::ClockedProcess(process))
                        })
                    });

                    let process_index = self.processes.len();
                    self.processes.push(ir_process);
                    self.clocked_block_statement_index_to_process_index
                        .insert_first(stmt_index, process_index);
                }
                ModuleStatementKind::Instance(instance) => {
                    let mut vars_inner = self.new_vars_for_process(vars, signals);
                    let instance_ir = self.elaborate_instance(scope, &mut vars_inner, stmt_index, instance);
                    self.processes.push(instance_ir.map(IrModuleChild::ModuleInstance));
                }
            }
        }
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

    fn elaborate_if(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        signals: &mut SignalsInScope,
        if_stmt: &IfStatement<ModuleStatement>,
    ) -> Result<(), ErrorGuaranteed> {
        let IfStatement {
            initial_if,
            else_ifs,
            final_else,
        } = if_stmt;

        if self.elaborate_if_pair(scope, vars, signals, initial_if)? {
            return Ok(());
        }
        for else_if in else_ifs {
            if self.elaborate_if_pair(scope, vars, signals, else_if)? {
                return Ok(());
            }
        }
        if let Some(final_else) = final_else {
            self.elaborate_block(scope, vars, signals, final_else);
        }

        Ok(())
    }

    fn elaborate_if_pair(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        signals: &mut SignalsInScope,
        pair: &IfCondBlockPair<ModuleStatement>,
    ) -> Result<bool, ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;
        let IfCondBlockPair {
            span: _,
            span_if,
            cond,
            block,
        } = pair;

        let mut vars_inner = VariableValues::new_child(vars);

        let cond = self
            .ctx
            .eval_expression_as_compile(scope, &mut vars_inner, cond, "module-level if condition")?;
        let reason = TypeContainsReason::IfCondition(*span_if);
        let cond = check_type_is_bool_compile(diags, reason, cond)?;

        if cond {
            self.elaborate_block(scope, &vars_inner, signals, block)
        }

        Ok(cond)
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
            let var =
                vars_inner.var_new_immutable_init(&mut self.ctx.variables, index.clone(), span_keyword, index_value);
            scope_inner.maybe_declare(diags, index.as_ref(), Ok(ScopedEntry::Named(NamedValue::Variable(var))));

            self.elaborate_block(&scope_inner, &vars_inner, signals, body);
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
            header,
            port_connections,
        } = instance;

        let &ElaboratedModule {
            module_ast,
            module_ir,
            ref ports,
        } = ctx.elaborate_module_header(scope, vars, header)?;
        let module_ast = &refs.fixed.parsed[module_ast];

        // eval and check port connections
        // TODO use function parameter matching for ports too?
        //   we need at least reordering and proper error messages
        if ports.len() != port_connections.inner.len() {
            let diag = Diagnostic::new("mismatched port connections for module instance")
                .add_error(
                    port_connections.span,
                    format!("ports connected here, got {} connections", port_connections.inner.len()),
                )
                .add_info(
                    module_ast.ports.span,
                    format!("module ports declared here, expected {} connections", ports.len()),
                )
                .finish();
            return Err(diags.report(diag));
        }

        let mut any_port_err = Ok(());

        let mut port_signals = IndexMap::new();
        let mut ir_connections = vec![];

        for (port, connection) in zip_eq(ports.keys(), &port_connections.inner) {
            match self.elaborate_instance_port_connection(
                &scope,
                vars,
                stmt_index,
                &ports,
                &port_signals,
                port,
                connection,
            ) {
                Ok((signal, connection)) => {
                    port_signals.insert_first(port, signal);
                    ir_connections.push(connection);
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
        ports: &InstancePorts,
        prev_port_signals: &IndexMap<InstancePort, ConnectionSignal>,
        port: InstancePort,
        connection: &Spanned<PortConnection>,
    ) -> Result<(ConnectionSignal, Spanned<IrPortConnection>), ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;

        let PortConnection {
            id: connection_id,
            expr: connection_expr,
        } = &connection.inner;
        let &PortInfo {
            id: ref port_id,
            direction: port_direction,
            domain: ref port_domain_raw,
            ty: ref port_ty_hw,
            ir: _,
        } = &ports[port];

        let port_id = port_id.clone();
        let port_ty_hw = port_ty_hw.clone();
        let port_ty = port_ty_hw.as_ref().map_inner(|ty| ty.as_type());

        // check id match
        if port_id.string != connection_id.string {
            let diag = Diagnostic::new("mismatched port connection")
                .add_error(
                    connection_id.span,
                    format!("connected here to `{}`", connection_id.string),
                )
                .add_info(port_id.span, format!("expected connection to `{}`", port_id.string))
                .footer(Level::Note, "port connection re-ordering is not yet supported")
                .finish();
            return Err(diags.report(diag));
        }

        // replace signals that are earlier ports with their connected value
        let port_domain_span = port_domain_raw.span;
        let port_domain = port_domain_raw
            .map_inner(|port_domain_raw| {
                Ok(match port_domain_raw {
                    PortDomain::Clock => ValueDomain::Clock,
                    PortDomain::Kind(port_domain_raw) => match port_domain_raw {
                        DomainKind::Const => ValueDomain::Const,
                        DomainKind::Async => ValueDomain::Async,
                        DomainKind::Sync(sync) => ValueDomain::Sync(sync.try_map_inner(|raw_port| {
                            let mapped_port = match prev_port_signals.get(&raw_port.signal) {
                                None => throw!(diags.report_internal_error(connection.span, "failed to get signal for previous port")),
                                Some(&ConnectionSignal::Dummy(dummy_span)) => {
                                    let diag = Diagnostic::new("feature not yet implemented: dummy port connections that are used in the domain of other ports")
                                        .add_error(dummy_span, "port connected to dummy here")
                                        .add_info(port_domain_span, "port used in a domain here")
                                        .finish();
                                    throw!(diags.report(diag))
                                }
                                Some(&ConnectionSignal::Expression(expr_span)) => {
                                    let diag = Diagnostic::new("feature not yet implemented: expression port connections that are used in the domain of other ports")
                                        .add_error(expr_span, "port connected to expression here")
                                        .add_info(port_domain_span, "port used in a domain here")
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
                    },
                })
            })
            .transpose();

        // always evaluate as signal for domain replacing purposes
        let signal = match &connection_expr.inner {
            ExpressionKind::Dummy => ConnectionSignal::Dummy(connection_expr.span),
            _ => match self
                .ctx
                .try_eval_expression_as_domain_signal(scope, connection_expr, |_| ())
            {
                Ok(signal) => ConnectionSignal::Signal(signal.inner),
                Err(Either::Left(())) => ConnectionSignal::Expression(connection_expr.span),
                Err(Either::Right(e)) => throw!(e),
            },
        };

        // evaluate the connection differently depending on the port direction
        let ir_connection = match port_direction.inner {
            PortDirection::Input => {
                // better dummy port error message
                if let ExpressionKind::Dummy = connection_expr.inner {
                    return Err(diags.report_simple(
                        "dummy connections are only allowed for output ports",
                        connection_expr.span,
                        "used dummy connection on input port here",
                    ));
                }

                // eval expr
                let mut report_assignment = report_assignment_internal_error(diags, "module instance port connection");
                let mut ctx =
                    IrBuilderExpressionContext::new(&BlockDomain::Combinatorial, None, &mut report_assignment);
                let mut ctx_block = ctx.new_ir_block();

                let mut vars_inner = VariableValues::new_child(vars);
                let connection_value = self.ctx.eval_expression(
                    &mut ctx,
                    &mut ctx_block,
                    scope,
                    &mut vars_inner,
                    &port_ty.inner,
                    connection_expr,
                )?;

                // check type
                let reason = TypeContainsReason::InstancePortInput {
                    span_connection_port_id: connection_id.span,
                    span_port_ty: port_ty.span,
                };
                let check_ty =
                    check_type_contains_value(diags, reason, &port_ty.inner, connection_value.as_ref(), true, false);

                // check domain
                let target_domain = Spanned {
                    span: connection_id.span,
                    inner: &port_domain.as_ref_ok()?.inner,
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
                    .map_inner(|v| Ok(v.as_ir_expression(diags, connection_expr.span, &port_ty_hw.inner)?.expr))
                    .transpose()?;

                // build extra wire and process if necessary
                let connection_value_ir =
                    if !ctx_block.statements.is_empty() || connection_value_ir_raw.inner.contains_variable() {
                        let extra_ir_wire = self.ir_wires.push(IrWireInfo {
                            ty: port_ty_hw.inner.as_ir(),
                            debug_info_id: MaybeIdentifier::Identifier(port_id.clone()),
                            debug_info_ty: port_ty_hw.inner.clone(),
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
                        self.processes.push(Ok(IrModuleChild::CombinatorialProcess(process)));

                        IrExpression::Wire(extra_ir_wire)
                    } else {
                        connection_value_ir_raw.inner
                    };

                IrPortConnection::Input(Spanned {
                    span: connection_expr.span,
                    inner: connection_value_ir,
                })
            }
            PortDirection::Output => {
                // eval expr as dummy, wire or port
                let build_error = || {
                    diags.report_simple(
                        "output port must be connected to wire or port",
                        connection_expr.span,
                        "other value",
                    )
                };

                match &connection_expr.inner {
                    ExpressionKind::Dummy => IrPortConnection::Output(None),
                    ExpressionKind::Id(id) => {
                        let named = self.ctx.eval_id(scope, id)?;

                        let (signal_ir, signal_target, signal_ty, signal_domain) = match named.inner {
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
                            span_connection_signal_id: connection_expr.span,
                            span_signal_ty: signal_ty.span,
                        };
                        any_err = any_err.and(check_type_contains_type(
                            diags,
                            reason,
                            &signal_ty.inner.as_type(),
                            Spanned {
                                span: connection_id.span,
                                inner: &port_ty.inner,
                            },
                            false,
                        ));

                        // check domain
                        any_err = any_err.and(self.ctx.check_valid_domain_crossing(
                            connection.span,
                            signal_domain.as_ref(),
                            Spanned {
                                span: connection_id.span,
                                inner: &port_domain.as_ref_ok()?.inner,
                            },
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
        Ok((signal, spanned_ir_connection))
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
            any_err = any_err.and(self.check_drivers_for_port_or_wire("port", self.ctx.ports[port].id.span, drivers));
        }
        for (&wire, drivers) in wire_drivers {
            any_err = any_err.and(self.check_drivers_for_port_or_wire("wire", self.ctx.wires[wire].id.span(), drivers));
        }

        // registers: check and collect resets
        for (&reg, drivers) in reg_drivers {
            let decl_span = self.ctx.registers[reg].id.span();

            any_err = any_err.and(self.check_drivers_for_reg(decl_span, drivers));

            // TODO allow zero drivers for registers, just turn them into wires with the init expression as the value
            //  (still emit a warning)
            let maybe_err = match self.check_exactly_one_driver("register", self.ctx.registers[reg].id.span(), drivers)
            {
                Err(e) => Err(e),
                Ok(driver) => pull_register_init_into_process(
                    self.ctx,
                    &self.clocked_block_statement_index_to_process_index,
                    &mut self.processes,
                    &self.register_initial_values,
                    reg,
                    driver,
                ),
            };
            any_err = any_err.and(maybe_err);
        }

        any_err
    }

    fn check_drivers_for_port_or_wire(
        &self,
        kind: &str,
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
                        .add_info(decl_span, "declared here")
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

        any_err = any_err.and(self.check_exactly_one_driver(kind, decl_span, drivers).map(|_| ()));
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
        decl_span: Span,
        drivers: &IndexMap<Driver, Span>,
    ) -> Result<Driver, ErrorGuaranteed> {
        let diags = self.ctx.refs.diags;

        match drivers.len() {
            0 => {
                let diag = Diagnostic::new(format!("{kind} has no driver"))
                    .add_error(decl_span, "declared here")
                    .finish();
                Err(diags.report(diag))
            }
            1 => {
                let (&driver, _) = drivers.iter().single().unwrap();
                Ok(driver)
            }
            _ => {
                let mut diag = Diagnostic::new(format!("{kind} has multiple drivers"));
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

        // evaluate
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
                let ty = ctx.eval_expression_as_ty_hardware(&scope_body, &mut vars_inner, ty, "wire");
                (domain, ty)
            }
        };

        // TODO move wire value creation/checking into pass 2, to better preserve the process order
        let mut report_assignment = report_assignment_internal_error(diags, "wire declaration value");
        let mut ctx_expr = IrBuilderExpressionContext::new(&BlockDomain::Combinatorial, None, &mut report_assignment);
        let mut ctx_block = ctx_expr.new_ir_block();

        let expected_ty = ty.as_ref_ok().ok().map(|ty| ty.inner.as_type()).unwrap_or(Type::Any);
        let value: Result<Option<Spanned<MaybeCompile<TypedIrExpression>>>, ErrorGuaranteed> = value
            .as_ref()
            .map(|value| {
                ctx.eval_expression(
                    &mut ctx_expr,
                    &mut ctx_block,
                    scope_body,
                    &mut vars_inner,
                    &expected_ty,
                    value,
                )
            })
            .transpose();

        let domain = domain?;
        let ty = ty?;
        let value = value?;

        let value = value
            .map(|value| {
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
                    ctx.check_valid_domain_crossing(decl.span, domain.as_ref(), value_domain_spanned, "value to wire");

                check_ty?;
                check_domain?;

                // convert to IR
                let value_ir = value.inner.as_ir_expression(diags, value.span, &ty.inner)?;
                Ok(Spanned {
                    span: value.span,
                    inner: value_ir,
                })
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
        if let Some(value) = value {
            let target = IrAssignmentTarget::wire(ir_wire);
            let stmt = IrStatement::Assign(target, value.inner.expr);

            let stmt = Spanned {
                span: value.span,
                inner: stmt,
            };
            ctx_block.statements.push(stmt);
        }

        // build final process if necessary
        let process = if ctx_block.statements.is_empty() {
            None
        } else {
            Some(IrCombinatorialProcess {
                locals: ctx_expr.finish(),
                block: ctx_block,
            })
        };

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
            .map_inner(|sync| ctx.eval_domain_sync(&scope_body, sync))
            .transpose();
        let ty = ctx.eval_expression_as_ty_hardware(&scope_body, &mut vars_inner, ty, "register");
        let init = ctx.eval_expression_as_compile(&scope_body, &mut vars_inner, init.as_ref(), "register reset value");

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
    ) -> Result<(Port, RegisterInit), ErrorGuaranteed> {
        let ctx = &mut self.ctx;
        let diags = ctx.refs.diags;
        let RegOutPortMarker { span: _, id, init } = decl;

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
        let init = ctx.eval_expression_as_compile(&scope_body, &mut vars_inner, init, "register reset value");

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
                inner: domain.map_inner(|p| p.map_inner(Signal::Port)),
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
        fn record<T: Hash + Eq>(
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
    ctx: &CompileItemContext,
    clocked_block_statement_index_to_process_index: &IndexMap<usize, usize>,
    processes: &mut Vec<Result<IrModuleChild, ErrorGuaranteed>>,
    register_initial_values: &IndexMap<Register, Result<Spanned<CompileValue>, ErrorGuaranteed>>,
    reg: Register,
    driver: Driver,
) -> Result<(), ErrorGuaranteed> {
    let diags = ctx.refs.diags;
    let reg_info = &ctx.registers[reg];

    if let Driver::ClockedBlock(stmt_index) = driver {
        if let Some(&process_index) = clocked_block_statement_index_to_process_index.get(&stmt_index) {
            if let IrModuleChild::ClockedProcess(process) = &mut processes[process_index].as_ref_mut_ok()? {
                if let Some(init) = register_initial_values.get(&reg) {
                    let init = init.as_ref_ok()?;
                    let init_ir = init
                        .inner
                        .as_ir_expression_or_undefined(diags, init.span, &reg_info.ty.inner)?
                        .map(|expr| expr.expr);

                    if let Some(init_ir) = init_ir {
                        let stmt = IrStatement::Assign(IrAssignmentTarget::register(reg_info.ir), init_ir);
                        process.on_reset.statements.push(Spanned {
                            span: init.span,
                            inner: stmt,
                        });
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
