use crate::front::block::{BlockDomain, TypedIrExpression, VariableValues};
use crate::front::check::{
    check_type_contains_compile_value, check_type_contains_type, check_type_contains_value, TypeContainsReason,
};
use crate::front::compile::{
    CompileState, ModuleElaborationInfo, Port, PortInfo, Register, RegisterInfo, Wire, WireInfo,
};
use crate::front::context::{ExpressionContext, IrBuilderExpressionContext};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrModuleChild, IrModuleInfo,
    IrModuleInstance, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrStatement, IrVariables,
    IrWire, IrWireInfo, IrWireOrPort,
};
use crate::front::misc::{DomainSignal, Polarized, PortDomain, ScopedEntry, Signal, ValueDomain};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{AssignmentTarget, CompileValue, HardwareValueResult, MaybeCompile, NamedValue};
use crate::syntax::ast;
use crate::syntax::ast::{
    Args, Block, ClockedBlock, CombinatorialBlock, DomainKind, ExpressionKind, GenericParameter, Identifier,
    MaybeIdentifier, ModuleInstance, ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle,
    ModuleStatement, ModuleStatementKind, PortConnection, PortDirection, RegDeclaration, RegOutPortMarker, Spanned,
    SyncDomain, WireDeclaration, WireKind,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
use crate::util::{result_pair, result_pair_split, ResultExt};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Either, Itertools};
use std::hash::Hash;

struct BodyElaborationState<'a, 'b> {
    state: &'a mut CompileState<'b>,

    scope_ports: Scope,
    scope_body: Scope,

    ir_ports: Arena<IrPort, IrPortInfo>,
    ir_wires: Arena<IrWire, IrWireInfo>,
    ir_registers: Arena<IrRegister, IrRegisterInfo>,
    drivers: Drivers,

    register_initial_values: IndexMap<Register, Result<Spanned<CompileValue>, ErrorGuaranteed>>,
    out_port_register_connections: IndexMap<Port, Register>,

    clocked_block_statement_index_to_process_index: IndexMap<usize, usize>,
    processes: Vec<Result<IrModuleChild, ErrorGuaranteed>>,
}

impl CompileState<'_> {
    pub fn elaborate_module_new(
        &mut self,
        elaboration_info: ModuleElaborationInfo,
    ) -> Result<(IrModuleInfo, Vec<Port>), ErrorGuaranteed> {
        let ModuleElaborationInfo { item, args } = elaboration_info;
        let &ast::ItemDefModule {
            span: def_span,
            vis: _,
            id: ref def_id,
            ref params,
            ref ports,
            ref body,
        } = &self.parsed[item];
        let scope_file = self.file_scope(item.file())?;

        let (scope_params, debug_info_generic_args) =
            self.elaborate_module_generics(def_span, scope_file, params, args)?;
        let (scope_ports, ports_vec, ir_ports) = self.elaborate_module_ports(def_span, scope_params, ports);
        let ir_module =
            self.elaborate_module_body(def_id, debug_info_generic_args, &ports_vec, ir_ports, scope_ports, body)?;
        Ok((ir_module, ports_vec))
    }

    fn elaborate_module_generics(
        &mut self,
        def_span: Span,
        file_scope: Scope,
        params: &Option<Spanned<Vec<GenericParameter>>>,
        args: Option<Args<Option<Identifier>, Spanned<CompileValue>>>,
    ) -> Result<(Scope, Option<Vec<(Identifier, CompileValue)>>), ErrorGuaranteed> {
        let diags = self.diags;

        let (params_scope, debug_info_generic_args) = match (params, args) {
            (None, None) => {
                let params_scope = self.scopes.new_child(file_scope, def_span, Visibility::Private);
                (params_scope, None)
            }
            (Some(params), Some(args)) => {
                let span_params = Span::new(params.span.start, def_span.end);
                let (params_scope, param_values) =
                    self.match_args_to_params_and_typecheck(params, &args, file_scope, span_params)?;
                (params_scope, Some(param_values))
            }
            _ => throw!(diags.report_internal_error(def_span, "mismatched generic arguments presence")),
        };

        Ok((params_scope, debug_info_generic_args))
    }

    fn elaborate_module_ports(
        &mut self,
        def_span: Span,
        params_scope: Scope,
        ports: &Spanned<Vec<ModulePortItem>>,
    ) -> (Scope, Vec<Port>, Arena<IrPort, IrPortInfo>) {
        let diags = self.diags;

        let mut ports_vec: Vec<Port> = vec![];
        let mut ir_ports = Arena::default();
        let ports_scope = self.scopes.new_child(params_scope, def_span, Visibility::Private);

        for port_item in &ports.inner {
            match port_item {
                ModulePortItem::Single(port_item) => {
                    let ModulePortSingle {
                        span: _,
                        id,
                        direction,
                        kind,
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
                        WireKind::Normal { domain, ty } => {
                            let no_vars = VariableValues::new_no_vars();
                            (
                                self.eval_port_domain(ports_scope, domain)
                                    .map(|d| d.map_inner(PortDomain::Kind)),
                                self.eval_expression_as_ty_hardware(ports_scope, &no_vars, ty, "port"),
                            )
                        }
                    };

                    // build entry
                    let entry = result_pair(domain, ty).map(|(domain, ty)| {
                        let ir_port = ir_ports.push(IrPortInfo {
                            direction: direction.inner,
                            ty: ty.inner.to_ir(),
                            debug_info_id: id.clone(),
                            debug_info_ty: ty.inner.clone(),
                            debug_info_domain: domain.inner.to_diagnostic_string(self),
                        });
                        let port = self.ports.push(PortInfo {
                            id: id.clone(),
                            direction: *direction,
                            domain,
                            ty,
                            ir: ir_port,
                        });
                        ports_vec.push(port);
                        ScopedEntry::Direct(NamedValue::Port(port))
                    });

                    self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                }
                ModulePortItem::Block(port_item) => {
                    let ModulePortBlock { span: _, domain, ports } = port_item;

                    // eval domain
                    let domain = self
                        .eval_port_domain(ports_scope, domain)
                        .map(|d| d.map_inner(PortDomain::Kind));

                    for port in ports {
                        let ModulePortInBlock {
                            span: _,
                            id,
                            direction,
                            ty,
                        } = port;

                        // eval ty
                        let no_vars = VariableValues::new_no_vars();
                        let ty = self.eval_expression_as_ty_hardware(ports_scope, &no_vars, ty, "port");

                        // build entry
                        let entry = result_pair(domain.as_ref_ok(), ty).map(|(&domain, ty)| {
                            let ir_port = ir_ports.push(IrPortInfo {
                                direction: direction.inner,
                                ty: ty.inner.to_ir(),
                                debug_info_id: id.clone(),
                                debug_info_ty: ty.inner.clone(),
                                debug_info_domain: domain.inner.to_diagnostic_string(self),
                            });
                            let port = self.ports.push(PortInfo {
                                id: id.clone(),
                                direction: *direction,
                                domain,
                                ty,
                                ir: ir_port,
                            });
                            ports_vec.push(port);
                            ScopedEntry::Direct(NamedValue::Port(port))
                        });

                        self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                    }
                }
            }
        }

        (ports_scope, ports_vec, ir_ports)
    }

    fn elaborate_module_body(
        &mut self,
        def_id: &Identifier,
        debug_info_generic_args: Option<Vec<(Identifier, CompileValue)>>,
        ports: &Vec<Port>,
        ir_ports: Arena<IrPort, IrPortInfo>,
        scope_ports: Scope,
        body: &Block<ModuleStatement>,
    ) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let scope_body = self.scopes.new_child(scope_ports, body.span, Visibility::Private);

        let mut state = BodyElaborationState {
            state: self,
            scope_ports,
            scope_body,
            ir_ports,
            ir_wires: Arena::default(),
            ir_registers: Arena::default(),
            drivers: Drivers::default(),

            register_initial_values: IndexMap::new(),
            out_port_register_connections: IndexMap::new(),
            clocked_block_statement_index_to_process_index: IndexMap::new(),
            processes: vec![],
        };

        // process declarations
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        state.pass_0_declarations(body);

        // create driver entries for remaining (= non-reg) output ports
        for &port in ports {
            match state.state.ports[port].direction.inner {
                PortDirection::Input => {
                    // do nothing
                }
                PortDirection::Output => {
                    state.drivers.output_port_drivers.entry(port).or_default();
                }
            }
        }

        // process processes
        state.pass_1_processes(body);

        // stop if any errors have happened so far, we don't want redundant errors about drivers
        for p in &state.processes {
            p.as_ref_ok()?;
        }

        // check driver validness
        // TODO more checking: combinatorial blocks can't read values they will later write,
        //   unless they have already written them
        state.pass_2_check_drivers_and_populate_resets()?;

        // create process for registered output ports
        if !state.out_port_register_connections.is_empty() {
            let mut statements = vec![];

            for (&out_port, &reg) in &state.out_port_register_connections {
                let out_port_ir = state.state.ports[out_port].ir;
                let reg_info = &state.state.registers[reg];
                let reg_ir = reg_info.ir;
                let statement =
                    IrStatement::Assign(IrAssignmentTarget::Port(out_port_ir), IrExpression::Register(reg_ir));
                statements.push(Spanned {
                    span: reg_info.id.span(),
                    inner: statement,
                })
            }

            let process = IrCombinatorialProcess {
                locals: IrVariables::default(),
                block: IrBlock { statements },
            };
            state.processes.push(Ok(IrModuleChild::CombinatorialProcess(process)));
        }

        // return result
        let processes = state.processes.into_iter().try_collect()?;

        Ok(IrModuleInfo {
            ports: state.ir_ports,
            registers: state.ir_registers,
            wires: state.ir_wires,
            children: processes,
            debug_info_id: def_id.clone(),
            debug_info_generic_args,
        })
    }
}

impl BodyElaborationState<'_, '_> {
    fn pass_0_declarations(&mut self, body: &Block<ModuleStatement>) {
        let scope_body = self.scope_body;
        let diags = self.state.diags;

        let Block { span: _, statements } = body;
        for stmt in statements {
            match &stmt.inner {
                // declarations
                ModuleStatementKind::ConstDeclaration(decl) => {
                    self.state.const_eval_and_declare(scope_body, decl);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let reg = self.elaborate_module_declaration_reg(decl);
                    let entry = reg.map(|reg_init| {
                        self.register_initial_values.insert_first(reg_init.reg, reg_init.init);
                        self.drivers.reg_drivers.insert_first(reg_init.reg, IndexMap::new());
                        ScopedEntry::Direct(NamedValue::Register(reg_init.reg))
                    });

                    let state = &mut self.state;
                    state.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let (wire, process) = result_pair_split(self.elaborate_module_declaration_wire(decl));
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

                    let entry = wire.map(|wire| ScopedEntry::Direct(NamedValue::Wire(wire)));
                    self.state.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    // declare register that shadows the outer port, which is exactly what we want
                    match self.elaborate_module_declaration_reg_out_port(decl) {
                        Ok((port, reg_init)) => {
                            let mut port_drivers = IndexMap::new();
                            port_drivers.insert_first(Driver::OutputPortConnectionToReg, decl.span);
                            self.drivers.output_port_drivers.insert_first(port, port_drivers);
                            self.drivers.reg_drivers.insert_first(reg_init.reg, IndexMap::new());
                            self.register_initial_values.insert_first(reg_init.reg, reg_init.init);
                            self.out_port_register_connections.insert_first(port, reg_init.reg);

                            let entry = Ok(ScopedEntry::Direct(NamedValue::Register(reg_init.reg)));
                            self.state.scopes[scope_body].declare(diags, &decl.id, entry, Visibility::Private);
                        }
                        Err(e) => {
                            // don't do anything, as if the marker is not there
                            let _: ErrorGuaranteed = e;
                        }
                    }
                }
                // non declarations, skip
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
            }
        }
    }

    fn pass_1_processes(&mut self, body: &Block<ModuleStatement>) {
        let scope_body = self.scope_body;
        let diags = self.state.diags;

        let Block { span: _, statements } = body;
        for (stmt_index, stmt) in enumerate(statements) {
            match &stmt.inner {
                // declarations, already handled
                ModuleStatementKind::ConstDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                // blocks, handle now
                ModuleStatementKind::CombinatorialBlock(block) => {
                    let CombinatorialBlock {
                        span: _,
                        span_keyword: _,
                        block,
                    } = block;

                    let block_domain = BlockDomain::Combinatorial;
                    let mut report_assignment = |target: Spanned<&AssignmentTarget>| {
                        self.drivers
                            .report_assignment(diags, Driver::CombinatorialBlock(stmt_index), target)
                    };

                    let mut ctx = IrBuilderExpressionContext::new(&block_domain, &mut report_assignment);
                    let ir_block = self
                        .state
                        .elaborate_block(&mut ctx, VariableValues::new(), scope_body, block);

                    let ir_block = ir_block.and_then(|(ir_block, end)| {
                        let ir_variables = ctx.finish();
                        let _: VariableValues = end.unwrap_normal_in_process(diags)?;
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
                        .map_inner(|d| self.state.eval_domain_sync(scope_body, d))
                        .transpose();

                    let ir_process = domain.and_then(|domain| {
                        let ir_domain = SyncDomain {
                            clock: self.state.domain_signal_to_ir(&domain.inner.clock),
                            reset: self.state.domain_signal_to_ir(&domain.inner.reset),
                        };

                        let block_domain = Spanned {
                            span: span_keyword.join(domain.span),
                            inner: domain.inner,
                        };
                        let block_domain = BlockDomain::Clocked(block_domain);
                        let mut report_assignment = |target: Spanned<&AssignmentTarget>| {
                            self.drivers
                                .report_assignment(diags, Driver::ClockedBlock(stmt_index), target)
                        };

                        let mut ctx = IrBuilderExpressionContext::new(&block_domain, &mut report_assignment);
                        let ir_block = self
                            .state
                            .elaborate_block(&mut ctx, VariableValues::new(), scope_body, block);

                        ir_block.and_then(|(ir_block, end)| {
                            let ir_variables = ctx.finish();
                            let _: VariableValues = end.unwrap_normal_in_process(diags)?;

                            let process = IrClockedProcess {
                                domain: Spanned {
                                    span: domain.span,
                                    inner: ir_domain,
                                },
                                locals: ir_variables,
                                on_clock: ir_block,
                                // will be filled in later, during driver checking and merging
                                on_reset: IrBlock { statements: vec![] },
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
                    let instance_ir = self.elaborate_instance(scope_body, stmt_index, instance);
                    self.processes.push(instance_ir.map(IrModuleChild::ModuleInstance));
                }
            }
        }
    }

    fn elaborate_instance(
        &mut self,
        scope_body: Scope,
        stmt_index: usize,
        instance: &ModuleInstance,
    ) -> Result<IrModuleInstance, ErrorGuaranteed> {
        let state = &mut self.state;
        let diags = state.diags;

        let ModuleInstance {
            span: _,
            span_keyword,
            name,
            module,
            generic_args,
            port_connections,
        } = instance;

        // eval module and generics
        let no_vars = VariableValues::new_no_vars();
        let module = state.eval_expression_as_compile(scope_body, &no_vars, module, "module instance");
        let generic_args = generic_args
            .as_ref()
            .map(|generic_args| {
                generic_args
                    .try_map_inner_all(|a| state.eval_expression_as_compile(scope_body, &no_vars, a, "generic arg"))
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
        let module_ast = &state.parsed[module_ast_ref];

        // elaborate module
        let generic_args = generic_args?;
        let elaboration = ModuleElaborationInfo {
            item: module_ast_ref,
            args: generic_args,
        };
        let (module_ir, ports) = self.state.elaborate_module(elaboration)?;

        // eval and check port connections
        // TODO allow port re-ordering as long as domain ordering constraints are respected
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
        let mut ir_connections = IndexMap::new();

        for (&port, connection) in zip_eq(&ports, &port_connections.inner) {
            match self.elaborate_instance_port_connection(scope_body, stmt_index, &port_signals, port, connection) {
                Ok((signal, connection)) => {
                    port_signals.insert_first(port, signal);
                    ir_connections.insert_first(self.state.ports[port].ir, connection);
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
        scope: Scope,
        stmt_index: usize,
        prev_port_signals: &IndexMap<Port, ConnectionSignal>,
        port: Port,
        connection: &Spanned<PortConnection>,
    ) -> Result<(ConnectionSignal, Spanned<IrPortConnection>), ErrorGuaranteed> {
        let diags = self.state.diags;

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
        } = &self.state.ports[port];

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
                                },
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
                .state
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
                let mut ctx = IrBuilderExpressionContext::new(&BlockDomain::Combinatorial, &mut report_assignment);
                let mut ctx_block = ctx.new_ir_block();

                let no_vars = VariableValues::new_no_vars();
                let connection_value = self.state.eval_expression(
                    &mut ctx,
                    &mut ctx_block,
                    scope,
                    &no_vars,
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
                let check_domain = self.state.check_valid_domain_crossing(
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
                            ty: port_ty_hw.inner.to_ir(),
                            debug_info_id: MaybeIdentifier::Identifier(port_id.clone()),
                            debug_info_ty: port_ty_hw.inner.clone(),
                            debug_info_domain: connection_value.inner.domain().to_diagnostic_string(self.state),
                        });

                        ctx_block.statements.push(Spanned {
                            span: connection.span,
                            inner: IrStatement::Assign(
                                IrAssignmentTarget::Wire(extra_ir_wire),
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
                        let named = self.state.eval_id(scope, id)?;

                        let (signal_ir, signal_target, signal_ty, signal_domain) = match named.inner {
                            MaybeCompile::Other(NamedValue::Wire(wire)) => {
                                let wire_info = &self.state.wires[wire];
                                (
                                    IrWireOrPort::Wire(wire_info.ir),
                                    AssignmentTarget::Wire(wire),
                                    &wire_info.ty,
                                    &wire_info.domain,
                                )
                            }
                            MaybeCompile::Other(NamedValue::Port(port)) => {
                                let port_info = &self.state.ports[port];
                                (
                                    IrWireOrPort::Port(port_info.ir),
                                    AssignmentTarget::Port(port),
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
                        any_err = any_err.and(self.state.check_valid_domain_crossing(
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
                        let target = Spanned {
                            span: named.span,
                            inner: &signal_target,
                        };
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
        } = std::mem::take(&mut self.drivers);
        let mut any_err = Ok(());

        // wire-like signals, just check
        for (&port, drivers) in &output_port_drivers {
            any_err = any_err.and(self.check_drivers_for_port_or_wire("port", self.state.ports[port].id.span, drivers));
        }
        for (&wire, drivers) in &wire_drivers {
            any_err =
                any_err.and(self.check_drivers_for_port_or_wire("wire", self.state.wires[wire].id.span(), drivers));
        }

        // registers: check and collect resets
        for (&reg, drivers) in &reg_drivers {
            let decl_span = self.state.registers[reg].id.span();

            any_err = any_err.and(self.check_drivers_for_reg(decl_span, drivers));

            // TODO allow zero drivers for registers, just turn them into wires with the init expression as the value
            //  (still emit a warning)
            match self.check_exactly_one_driver("register", self.state.registers[reg].id.span(), drivers) {
                Err(e) => any_err = Err(e),
                Ok(driver) => any_err = any_err.and(self.pull_register_init_into_process(reg, driver)),
            }
        }

        // put drivers back
        assert_eq!(self.drivers, Drivers::default());
        self.drivers = Drivers {
            output_port_drivers,
            reg_drivers,
            wire_drivers,
        };

        any_err
    }

    fn pull_register_init_into_process(&mut self, reg: Register, driver: Driver) -> Result<(), ErrorGuaranteed> {
        let diags = self.state.diags;
        let reg_info = &self.state.registers[reg];

        if let Driver::ClockedBlock(stmt_index) = driver {
            if let Some(&process_index) = self.clocked_block_statement_index_to_process_index.get(&stmt_index) {
                if let IrModuleChild::ClockedProcess(process) = &mut self.processes[process_index].as_ref_mut_ok()? {
                    if let Some(init) = self.register_initial_values.get(&reg) {
                        let init = init.as_ref_ok()?;

                        // TODO fix duplication with CompileValue::to_ir_expression
                        // TODO move this to where the register is initially visited
                        let init_ir = match init.inner.as_hardware_value(&reg_info.ty.inner) {
                            HardwareValueResult::Success(v) => Some(v),
                            HardwareValueResult::Undefined => None,
                            HardwareValueResult::PartiallyUndefined => {
                                return Err(diags.report_todo(init.span, "partially undefined register reset value"))
                            }
                            HardwareValueResult::Unrepresentable => throw!(diags.report_internal_error(
                                init.span,
                                format!(
                                    "value `{}` has hardware type but is itself not representable",
                                    init.inner.to_diagnostic_string()
                                ),
                            )),
                            HardwareValueResult::InvalidType => throw!(diags.report_internal_error(
                                init.span,
                                format!(
                                    "value `{}` with type `{}` cannot be represented as hardware type `{}`",
                                    init.inner.to_diagnostic_string(),
                                    init.inner.ty().to_diagnostic_string(),
                                    reg_info.ty.inner.as_type().to_diagnostic_string()
                                ),
                            )),
                        };

                        if let Some(init_ir) = init_ir {
                            let stmt = IrStatement::Assign(IrAssignmentTarget::Register(reg_info.ir), init_ir);
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

    fn check_drivers_for_port_or_wire(
        &self,
        kind: &str,
        decl_span: Span,
        drivers: &IndexMap<Driver, Span>,
    ) -> Result<(), ErrorGuaranteed> {
        let mut any_err = Ok(());

        for (&driver, &span) in drivers {
            match driver.kind() {
                DriverKind::ClockedBlock => {
                    let diag = Diagnostic::new(format!("{kind} cannot be driven by clocked block"))
                        .add_error(span, "driven incorrectly here")
                        .add_info(decl_span, "declared here")
                        .footer(
                            Level::Help,
                            format!("either drive the {kind} from a combinatorial block or turn it into a register"),
                        )
                        .finish();
                    any_err = Err(self.state.diags.report(diag));
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
                    any_err = Err(self.state.diags.report(diag));
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
        match drivers.len() {
            0 => {
                let diag = Diagnostic::new(format!("{kind} has no driver"))
                    .add_error(decl_span, "declared here")
                    .finish();
                Err(self.state.diags.report(diag))
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
                Err(self.state.diags.report(diag))
            }
        }
    }

    fn elaborate_module_declaration_wire(
        &mut self,
        decl: &WireDeclaration,
    ) -> Result<(Wire, Option<IrCombinatorialProcess>), ErrorGuaranteed> {
        let state = &mut self.state;
        let scope_body = self.scope_body;
        let diags = state.diags;

        let WireDeclaration {
            span: _,
            id,
            kind: domain_ty,
            value,
        } = decl;

        // evaluate
        let no_vars = VariableValues::new_no_vars();
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
                let domain = state
                    .eval_domain(scope_body, domain)
                    .map(|d| d.map_inner(ValueDomain::from_domain_kind));
                let ty = state.eval_expression_as_ty_hardware(scope_body, &no_vars, ty, "wire");
                (domain, ty)
            }
        };

        // TODO move wire value creation/checking into pass 2, to better preserve the process order
        let mut report_assignment = report_assignment_internal_error(diags, "wire declaration value");
        let mut ctx = IrBuilderExpressionContext::new(&BlockDomain::Combinatorial, &mut report_assignment);
        let mut ctx_block = ctx.new_ir_block();

        let expected_ty = ty.as_ref_ok().ok().map(|ty| ty.inner.as_type()).unwrap_or(Type::Any);
        let value: Result<Option<Spanned<MaybeCompile<TypedIrExpression>>>, ErrorGuaranteed> = value
            .as_ref()
            .map(|value| state.eval_expression(&mut ctx, &mut ctx_block, scope_body, &no_vars, &expected_ty, value))
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
                let check_domain = state.check_valid_domain_crossing(
                    decl.span,
                    domain.as_ref(),
                    value_domain_spanned,
                    "value to wire",
                );

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
            ty: ty.inner.to_ir(),
            debug_info_id: id.clone(),
            debug_info_ty: ty.inner.clone(),
            debug_info_domain: domain.inner.to_diagnostic_string(state),
        });
        let wire = state.wires.push(WireInfo {
            id: id.clone(),
            domain,
            ty,
            ir: ir_wire,
        });

        // append assignment to process
        if let Some(value) = value {
            let target = IrAssignmentTarget::Wire(ir_wire);
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
                locals: ctx.finish(),
                block: ctx_block,
            })
        };

        Ok((wire, process))
    }

    fn elaborate_module_declaration_reg(&mut self, decl: &RegDeclaration) -> Result<RegisterInit, ErrorGuaranteed> {
        let state = &mut self.state;
        let diags = state.diags;
        let scope_body = self.scope_body;

        let RegDeclaration {
            span: _,
            id,
            sync,
            ty,
            init,
        } = decl;

        // evaluate
        let no_vars = VariableValues::new_no_vars();
        let sync = sync
            .as_ref()
            .map_inner(|sync| state.eval_domain_sync(scope_body, sync))
            .transpose();
        let ty = state.eval_expression_as_ty_hardware(scope_body, &no_vars, ty, "register");
        let init = state.eval_expression_as_compile(scope_body, &no_vars, init.as_ref(), "register reset value");

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
            ty: ty.inner.to_ir(),
            debug_info_id: id.clone(),
            debug_info_ty: ty.inner.clone(),
            debug_info_domain: sync.inner.to_diagnostic_string(state),
        });
        let reg = state.registers.push(RegisterInfo {
            id: id.clone(),
            domain: sync,
            ty,
            ir: ir_reg,
        });
        Ok(RegisterInit { reg, init })
    }

    fn elaborate_module_declaration_reg_out_port(
        &mut self,
        decl: &RegOutPortMarker,
    ) -> Result<(Port, RegisterInit), ErrorGuaranteed> {
        let state = &mut self.state;
        let diags = state.diags;
        let scope_body = self.scope_body;
        let scope_ports = self.scope_ports;

        let RegOutPortMarker { span: _, id, init } = decl;

        // find port (looking only at the port scope to avoid shadowing or hitting outer identifiers)
        let port = state.scopes[scope_ports].find_immediate_str(diags, &id.string, Visibility::Private)?;
        let port = match port.value {
            &ScopedEntry::Direct(NamedValue::Port(port)) => Ok(port),
            _ => Err(diags.report_internal_error(id.span, "found non-port in port scope")),
        };

        // evaluate init
        let no_vars = VariableValues::new_no_vars();
        let init = state.eval_expression_as_compile(scope_body, &no_vars, init, "register reset value");

        let init = init;
        let port = port?;

        let port_info = &state.ports[port];
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
            ty: port_info.ty.inner.to_ir(),
            debug_info_id: MaybeIdentifier::Identifier(id.clone()),
            debug_info_ty: port_info.ty.inner.clone(),
            debug_info_domain: domain.to_diagnostic_string(state),
        });
        let reg = state.registers.push(RegisterInfo {
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

#[derive(Default, Debug, Eq, PartialEq)]
struct Drivers {
    // For each signal, for each driver, the first span.
    // This span will be used in error messages in case there are multiple drivers for the same signal.
    output_port_drivers: IndexMap<Port, IndexMap<Driver, Span>>,
    reg_drivers: IndexMap<Register, IndexMap<Driver, Span>>,
    wire_drivers: IndexMap<Wire, IndexMap<Driver, Span>>,
}

impl Drivers {
    pub fn report_assignment(
        &mut self,
        diags: &Diagnostics,
        driver: Driver,
        target: Spanned<&AssignmentTarget>,
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

        match *target.inner {
            AssignmentTarget::Port(port) => record(diags, &mut self.output_port_drivers, driver, port, target.span),
            AssignmentTarget::Wire(wire) => record(diags, &mut self.wire_drivers, driver, wire, target.span),
            AssignmentTarget::Register(reg) => record(diags, &mut self.reg_drivers, driver, reg, target.span),
            AssignmentTarget::Variable(_) => Err(diags.report_todo(target.span, "variable assignment")),
        }
    }
}

fn report_assignment_internal_error<'a>(
    diags: &'a Diagnostics,
    place: &'a str,
) -> impl FnMut(Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed> + 'a {
    move |target: Spanned<&AssignmentTarget>| {
        Err(diags.report_internal_error(target.span, format!("driving signal within {place}")))
    }
}
