use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::block::{BlockDomain, VariableValues};
use crate::new::check::{check_type_contains_compile_value, check_type_contains_value, TypeContainsReason};
use crate::new::compile::{
    CompileState, ConstantInfo, ModuleElaboration, ModuleElaborationInfo, Port, PortInfo, Register, RegisterInfo, Wire,
    WireInfo,
};
use crate::new::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrModuleInfo, IrPort,
    IrPortInfo, IrProcess, IrRegister, IrRegisterInfo, IrStatement, IrVariables, IrWire, IrWireInfo,
};
use crate::new::misc::{PortDomain, ScopedEntry, ValueDomain};
use crate::new::types::HardwareType;
use crate::new::value::{AssignmentTarget, CompileValue, HardwareValueResult, MaybeCompile, NamedValue};
use crate::syntax::ast;
use crate::syntax::ast::{
    Block, ClockedBlock, CombinatorialBlock, DomainKind, GenericParameter, Identifier, MaybeIdentifier,
    ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle, ModuleStatement, ModuleStatementKind,
    PortDirection, PortKind, RegDeclaration, RegOutPortMarker, Spanned, SyncDomain, WireDeclaration,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
use crate::util::{result_pair, result_pair_split, ResultExt};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Itertools};
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
    port_register_connections: IndexMap<Port, Register>,

    clocked_block_statement_index_to_process_index: IndexMap<usize, usize>,
    processes: Vec<Result<Spanned<IrProcess>, ErrorGuaranteed>>,
}

impl CompileState<'_> {
    pub fn elaborate_module_new(
        &mut self,
        module_elaboration: ModuleElaboration,
    ) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let ModuleElaborationInfo { item, args } = self.elaborated_modules[module_elaboration].clone();
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
        let (scope_ports, ir_ports) = self.elaborate_module_ports(def_span, scope_params, ports);
        self.elaborate_module_body(def_id, debug_info_generic_args, ir_ports, scope_ports, body)
    }

    fn elaborate_module_generics(
        &mut self,
        def_span: Span,
        file_scope: Scope,
        params: &Option<Spanned<Vec<GenericParameter>>>,
        args: Option<Vec<CompileValue>>,
    ) -> Result<(Scope, Option<Vec<(Identifier, CompileValue)>>), ErrorGuaranteed> {
        let diags = self.diags;
        let params_scope = self.scopes.new_child(file_scope, def_span, Visibility::Private);

        let debug_info_generic_args = match (params, args) {
            (None, None) => None,
            (Some(params), Some(args)) => {
                if params.inner.len() != args.len() {
                    throw!(diags.report_internal_error(params.span, "mismatched generic argument count"));
                }

                let mut debug_info_generic_args = vec![];

                for (param, arg) in zip_eq(&params.inner, args) {
                    let no_vars = VariableValues::new_no_vars();
                    let param_ty = self.eval_expression_as_ty(params_scope, &no_vars, &param.ty);

                    let entry = param_ty.and_then(|param_ty| {
                        if param_ty.inner.contains_type(&arg.ty()) {
                            let param = self.parameters.push(ConstantInfo {
                                id: MaybeIdentifier::Identifier(param.id.clone()),
                                value: arg.clone(),
                            });
                            Ok(ScopedEntry::Direct(NamedValue::Parameter(param)))
                        } else {
                            let e = diags.report_internal_error(
                                param.ty.span,
                                format!(
                                    "invalid generic argument: type `{}` does not contain value `{}`",
                                    param_ty.inner.to_diagnostic_string(),
                                    arg.to_diagnostic_string()
                                ),
                            );
                            Err(e)
                        }
                    });

                    self.scopes[params_scope].declare(diags, &param.id, entry, Visibility::Private);
                    debug_info_generic_args.push((param.id.clone(), arg));
                }

                Some(debug_info_generic_args)
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
    ) -> (Scope, Arena<IrPort, IrPortInfo>) {
        let diags = self.diags;

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
                        PortKind::Clock => (
                            Ok(Spanned {
                                span: kind.span,
                                inner: PortDomain::Clock,
                            }),
                            Ok(Spanned {
                                span: kind.span,
                                inner: HardwareType::Clock,
                            }),
                        ),
                        PortKind::Normal { domain, ty } => {
                            let no_vars = VariableValues::new_no_vars();
                            (
                                self.eval_domain(ports_scope, &domain)
                                    .map(|d| d.map_inner(|d| PortDomain::Kind(d))),
                                self.eval_expression_as_ty_hardware(ports_scope, &no_vars, ty, "port"),
                            )
                        }
                    };

                    // build entry
                    let entry = result_pair(domain, ty).and_then(|(domain, ty)| {
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

                        Ok(ScopedEntry::Direct(NamedValue::Port(port)))
                    });

                    self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                }
                ModulePortItem::Block(port_item) => {
                    let ModulePortBlock { span: _, domain, ports } = port_item;

                    // eval domain
                    let domain = self
                        .eval_domain(ports_scope, &domain)
                        .map(|d| d.map_inner(|d| PortDomain::Kind(d)));

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
                        let entry = result_pair(domain.as_ref_ok(), ty).and_then(|(domain, ty)| {
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
                                domain: domain.clone(),
                                ty,
                                ir: ir_port,
                            });
                            Ok(ScopedEntry::Direct(NamedValue::Port(port)))
                        });

                        self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                    }
                }
            }
        }

        (ports_scope, ir_ports)
    }

    fn elaborate_module_body(
        &mut self,
        def_id: &Identifier,
        debug_info_generic_args: Option<Vec<(Identifier, CompileValue)>>,
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
            port_register_connections: IndexMap::new(),
            clocked_block_statement_index_to_process_index: IndexMap::new(),
            processes: vec![],
        };

        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        state.pass_0_declarations(body);
        state.pass_1_processes(body);

        // stop if any errors have happened so far, we don't want redundant errors about drivers
        for p in &state.processes {
            p.as_ref_ok()?;
        }

        // check driver validness
        // TODO more checking: combinatorial blocks can't read values they will later write,
        //   unless they have already written them
        state.pass_2_check_drivers_and_populate_resets()?;

        // return result
        let processes = state.processes.into_iter().try_collect()?;

        Ok(IrModuleInfo {
            ports: state.ir_ports,
            registers: state.ir_registers,
            wires: state.ir_wires,
            processes,
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
                        self.processes.push(process.map(|p| Spanned {
                            span: decl.span,
                            inner: p,
                        }));
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
                            self.port_register_connections.insert_first(port, reg_init.reg);

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
        let state = &mut self.state;
        let scope_body = self.scope_body;
        let diags = state.diags;

        let Block { span: _, statements } = body;
        for (statement_index, stmt) in enumerate(statements) {
            match &stmt.inner {
                // declarations, already handled
                ModuleStatementKind::ConstDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                // blocks, handle now
                ModuleStatementKind::CombinatorialBlock(block) => {
                    let &CombinatorialBlock {
                        span: _,
                        span_keyword: _,
                        ref block,
                    } = block;

                    let block_domain = BlockDomain::Combinatorial;
                    let mut ir_locals = Arena::default();
                    let mut report_assignment = |target: Spanned<&AssignmentTarget>| {
                        self.drivers
                            .report_assignment(diags, Driver::CombinatorialBlock(statement_index), target)
                    };

                    let mut condition_domains = vec![];
                    let ir_block = state.elaborate_ir_block(
                        &mut report_assignment,
                        &mut ir_locals,
                        VariableValues::new(),
                        &block_domain,
                        &mut condition_domains,
                        scope_body,
                        block,
                    );
                    assert!(condition_domains.is_empty());

                    let ir_process = ir_block.map(|(block, _vars)| {
                        let ir_body = IrCombinatorialProcess {
                            locals: ir_locals,
                            block,
                        };
                        IrProcess::Combinatorial(ir_body)
                    });
                    self.processes.push(ir_process.map(|p| Spanned {
                        span: block.span,
                        inner: p,
                    }));
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
                        .map_inner(|d| state.eval_domain_sync(scope_body, d))
                        .transpose();

                    let ir_process = domain.and_then(|domain| {
                        let mut ir_locals = Arena::default();
                        let ir_domain = SyncDomain {
                            clock: state.domain_signal_to_ir(&domain.inner.clock),
                            reset: state.domain_signal_to_ir(&domain.inner.reset),
                        };

                        let block_domain = Spanned {
                            span: span_keyword.join(domain.span),
                            inner: domain.inner.clone(),
                        };
                        let block_domain = BlockDomain::Clocked(block_domain);
                        let mut report_assignment = |target: Spanned<&AssignmentTarget>| {
                            self.drivers
                                .report_assignment(diags, Driver::ClockedBlock(statement_index), target)
                        };

                        let mut condition_domains = vec![];
                        let (ir_block, _vars) = state.elaborate_ir_block(
                            &mut report_assignment,
                            &mut ir_locals,
                            VariableValues::new(),
                            &block_domain,
                            &mut condition_domains,
                            scope_body,
                            block,
                        )?;

                        if !condition_domains.is_empty() {
                            throw!(diags.report_internal_error(
                                stmt.span,
                                "unexpected recorded condition domains in clocked block"
                            ));
                        }

                        let ir_process = IrClockedProcess {
                            locals: ir_locals,
                            domain: Spanned {
                                span: domain.span,
                                inner: ir_domain,
                            },
                            on_clock: ir_block,
                            // will be filled in later, during driver checking
                            on_reset: IrBlock { statements: vec![] },
                        };
                        Ok(IrProcess::Clocked(ir_process))
                    });

                    let process_index = self.processes.len();
                    self.processes.push(ir_process.map(|p| Spanned {
                        span: block.span,
                        inner: p,
                    }));
                    self.clocked_block_statement_index_to_process_index
                        .insert_first(statement_index, process_index);
                }
                ModuleStatementKind::Instance(_) => {
                    let e = diags.report_todo(stmt.span, "module instance");
                    self.processes.push(Err(e));
                }
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

            // TODO allow zero drivers for registers, just create a dummy process for the reset value?
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
                if let IrProcess::Clocked(process) = &mut self.processes[process_index].as_ref_mut_ok()?.inner {
                    if let Some(init) = self.register_initial_values.get(&reg) {
                        let init = init.as_ref_ok()?;

                        // TODO fix duplication with lowering in block
                        let init_ir = match init.inner.as_hardware_value() {
                            HardwareValueResult::Success(v) => Some(v),
                            HardwareValueResult::Undefined => None,
                            HardwareValueResult::PartiallyUndefined => {
                                return Err(diags.report_todo(init.span, "partially undefined register reset value"))
                            }
                            HardwareValueResult::Unrepresentable => {
                                // TODO fix this duplication
                                let reason =
                                    "compile time value fits in hardware type but is not convertible to hardware value";
                                return Err(diags.report_internal_error(init.span, reason));
                            }
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
    ) -> Result<(Wire, Option<IrProcess>), ErrorGuaranteed> {
        let state = &mut self.state;
        let scope_body = self.scope_body;
        let diags = state.diags;

        let WireDeclaration {
            span: _,
            id,
            sync,
            ty,
            value,
        } = decl;

        // evaluate
        // TODO allow clock wires
        let domain = state.eval_domain(scope_body, sync);
        let no_vars = VariableValues::new_no_vars();
        let ty = state.eval_expression_as_ty_hardware(scope_body, &no_vars, ty, "wire");

        let value = value
            .as_ref()
            .map(|value| state.eval_expression(scope_body, &no_vars, value))
            .transpose();

        let domain = domain?;
        let ty = ty?;
        let value = value?;

        if let Some(value) = &value {
            // check type
            let reason = TypeContainsReason::Assignment {
                span_assignment: decl.span,
                span_target_ty: ty.span,
            };
            let check_ty = check_type_contains_value(diags, reason, &ty.inner.as_type(), value.as_ref(), true);

            // check domain
            let value_domain = value.inner.domain();
            let value_domain_spanned = Spanned {
                span: value.span,
                inner: value_domain,
            };
            let target_domain = domain.clone().map_inner(|d| ValueDomain::from_domain_kind(d));
            let check_domain = state.check_valid_domain_crossing(
                decl.span,
                target_domain.as_ref(),
                value_domain_spanned,
                "value to wire",
            );

            check_ty?;
            check_domain?;
        }

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

        // build process
        let mut ir_statements = vec![];
        let process = value
            .map(|value| {
                let target = IrAssignmentTarget::Wire(ir_wire);
                let source = match value.inner {
                    MaybeCompile::Compile(_) => throw!(diags.report_todo(value.span, "compile-time wire value")),
                    MaybeCompile::Other(v) => v.expr,
                };
                let stmt = IrStatement::Assign(target, source);
                ir_statements.push(Ok(Spanned {
                    span: value.span,
                    inner: stmt,
                }));

                let ir_statements = ir_statements.into_iter().try_collect()?;
                let ir_process = IrCombinatorialProcess {
                    locals: IrVariables::default(),
                    block: IrBlock {
                        statements: ir_statements,
                    },
                };
                let ir_process = IrProcess::Combinatorial(ir_process);

                Ok(ir_process)
            })
            .transpose()?;

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
                span_assignment: decl.span,
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
                span_assignment: decl.span,
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
                inner: domain.clone(),
            },
            ty: port_info.ty.clone(),
            ir: ir_reg,
        });
        Ok((port, RegisterInit { reg, init }))
    }
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

        match target.inner {
            &AssignmentTarget::Port(port) => record(diags, &mut self.output_port_drivers, driver, port, target.span),
            &AssignmentTarget::Wire(wire) => record(diags, &mut self.wire_drivers, driver, wire, target.span),
            &AssignmentTarget::Register(reg) => record(diags, &mut self.reg_drivers, driver, reg, target.span),
            &AssignmentTarget::Variable(_) => Err(diags.report_todo(target.span, "variable assignment")),
        }
    }
}
