use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::{CompileState, ConstantInfo, ModuleElaboration, ModuleElaborationInfo, Port, PortInfo, Register, RegisterInfo, Wire, WireInfo};
use crate::new::ir::{IrAssignmentTarget, IrBlock, IrExpression, IrLocals, IrModuleInfo, IrPort, IrPortInfo, IrProcess, IrProcessBody, IrRegister, IrRegisterInfo, IrStatement, IrWire, IrWireInfo};
use crate::new::misc::{DomainSignal, PortDomain, ScopedEntry, ValueDomain};
use crate::new::types::HardwareType;
use crate::new::value::{CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast;
use crate::syntax::ast::{Block, ClockedBlock, CombinatorialBlock, DomainKind, GenericParameter, MaybeIdentifier, ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle, ModuleStatement, ModuleStatementKind, PortDirection, PortKind, RegDeclaration, RegOutPortMarker, Spanned, SyncDomain, WireDeclaration};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::{result_pair, result_pair_split, ResultExt};
use indexmap::IndexMap;
use itertools::{zip_eq, Itertools};

pub struct BodyElaborationState<'a, 'b> {
    pub state: &'a mut CompileState<'b>,

    pub scope_ports: Scope,
    pub scope_body: Scope,

    pub ir_ports: Arena<IrPort, IrPortInfo>,
    pub ir_wires: Arena<IrWire, IrWireInfo>,
    pub ir_registers: Arena<IrRegister, IrRegisterInfo>,

    pub register_initial_values: IndexMap<Register, Result<CompileValue, ErrorGuaranteed>>,
    pub port_register_connections: IndexMap<Port, Register>,

    pub processes: Vec<Result<IrProcess, ErrorGuaranteed>>,
}

impl CompileState<'_> {
    pub fn elaborate_module_new(&mut self, module_elaboration: ModuleElaboration) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let ModuleElaborationInfo { item, args } = self.elaborated_modules[module_elaboration].clone();
        let &ast::ItemDefModule { span: def_span, vis: _, id: _, ref params, ref ports, ref body } = &self.parsed[item];
        let scope_file = self.file_scope(item.file())?;

        let scope_params = self.elaborate_module_generics(def_span, scope_file, params, args)?;
        let (scope_ports, ir_ports) = self.elaborate_module_ports(def_span, scope_params, ports);
        self.elaborate_module_body(ir_ports, scope_ports, body)
    }

    fn elaborate_module_generics(&mut self, def_span: Span, file_scope: Scope, params: &Option<Spanned<Vec<GenericParameter>>>, args: Option<Vec<CompileValue>>) -> Result<Scope, ErrorGuaranteed> {
        let diags = self.diags;
        let params_scope = self.scopes.new_child(file_scope, def_span, Visibility::Private);

        match (params, args) {
            (None, None) => {}
            (Some(params), Some(args)) => {
                if params.inner.len() != args.len() {
                    throw!(diags.report_internal_error(params.span, "mismatched generic argument count"));
                }

                for (param, arg) in zip_eq(&params.inner, args) {
                    let param_ty = self.eval_expression_as_ty(params_scope, &param.ty);

                    let entry = param_ty.and_then(|param_ty| {
                        if param_ty.contains_type(&arg.ty()) {
                            let param = self.parameters.push(ConstantInfo {
                                id: MaybeIdentifier::Identifier(param.id.clone()),
                                value: arg,
                            });
                            Ok(ScopedEntry::Direct(NamedValue::Parameter(param)))
                        } else {
                            let e = diags.report_internal_error(
                                param.ty.span,
                                format!(
                                    "invalid generic argument: type `{}` does not contain value `{}`",
                                    param_ty.to_diagnostic_string(),
                                    arg.to_diagnostic_string()
                                ),
                            );
                            Err(e)
                        }
                    });

                    self.scopes[params_scope].declare(diags, &param.id, entry, Visibility::Private);
                }
            }
            _ => throw!(diags.report_internal_error(def_span, "mismatched generic arguments presence")),
        };

        Ok(params_scope)
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
                    let ModulePortSingle { span: _, id, direction, kind } = port_item;

                    // eval kind
                    let (domain, ty) = match &kind.inner {
                        PortKind::Clock => (
                            Ok(Spanned { span: kind.span, inner: PortDomain::Clock }),
                            Ok(Spanned { span: kind.span, inner: HardwareType::Clock }),
                        ),
                        PortKind::Normal { domain, ty } => (
                            self.eval_domain(ports_scope, &domain.inner)
                                .map(|d| Spanned { span: domain.span, inner: PortDomain::Kind(d) }),
                            self.eval_expression_as_ty_hardware(ports_scope, ty, "port")
                                .map(|t| Spanned { span: ty.span, inner: t }),
                        ),
                    };

                    // build entry
                    let entry = result_pair(domain, ty).and_then(|(domain, ty)| {
                        let ir_port = ir_ports.push(IrPortInfo {
                            name: id.string.clone(),
                            direction: direction.inner,
                            ty: ty.inner.to_ir(),
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
                    let domain = self.eval_domain(ports_scope, &domain.inner)
                        .map(|d| Spanned { span: domain.span, inner: PortDomain::Kind(d) });

                    for port in ports {
                        let ModulePortInBlock { span: _, id, direction, ty } = port;

                        // eval ty
                        let ty = self.eval_expression_as_ty_hardware(ports_scope, ty, "port")
                            .map(|t| Spanned { span: ty.span, inner: t });

                        // build entry
                        let entry = result_pair(domain.as_ref_ok(), ty).and_then(|(domain, ty)| {
                            let ir_port = ir_ports.push(IrPortInfo {
                                name: id.string.clone(),
                                direction: direction.inner,
                                ty: ty.inner.to_ir(),
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

            register_initial_values: IndexMap::new(),
            port_register_connections: IndexMap::new(),
            processes: vec![],
        };

        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        state.first_pass_declarations(body);
        state.second_pass_processes(body);

        // check driver validness
        // TODO skip if any bad enough errors happened before
        // TODO implement

        // pull in register initial values
        // TODO

        // return result
        let processes = state.processes.into_iter().try_collect()?;
        Ok(IrModuleInfo {
            ports: state.ir_ports,
            registers: state.ir_registers,
            wires: state.ir_wires,
            processes,
        })
    }

    // TODO move
    fn domain_signal_to_ir(&self, signal: &DomainSignal) -> IrExpression {
        match signal {
            &DomainSignal::Port(port) => IrExpression::Port(self.ports[port].ir),
            &DomainSignal::Wire(wire) => IrExpression::Wire(self.wires[wire].ir),
            &DomainSignal::Register(reg) => IrExpression::Register(self.registers[reg].ir),
            DomainSignal::BoolNot(inner) =>
                IrExpression::BoolNot(Box::new(self.domain_signal_to_ir(inner))),
        }
    }
}

impl BodyElaborationState<'_, '_> {
    fn first_pass_declarations(&mut self, body: &Block<ModuleStatement>) {
        let scope_body = self.scope_body;
        let diags = self.state.diags;

        let Block { span: _, statements } = body;
        for stmt in statements {
            match &stmt.inner {
                // declarations
                ModuleStatementKind::ConstDeclaration(decl) => {
                    let entry = self.state.const_eval_and_check(self.scope_body, decl)
                        .map(|value| {
                            let cst = self.state.constants.push(ConstantInfo {
                                id: decl.id.clone(),
                                value,
                            });
                            ScopedEntry::Direct(NamedValue::Constant(cst))
                        });
                    self.state.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let reg = self.elaborate_module_declaration_reg(decl);
                    let entry = reg.map(|reg_init| {
                        self.register_initial_values.insert_first(reg_init.reg, reg_init.init);

                        ScopedEntry::Direct(NamedValue::Register(reg_init.reg))
                    });

                    let state = &mut self.state;
                    state.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let (wire, process) =
                        result_pair_split(self.elaborate_module_declaration_wire(decl));

                    if let Some(process) = process.transpose() {
                        self.processes.push(process);
                    }

                    let entry = wire.map(|wire| ScopedEntry::Direct(NamedValue::Wire(wire)));
                    self.state.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    // declare register that shadows the outer port, which is exactly what we want
                    match self.elaborate_module_declaration_reg_out_port(decl) {
                        Ok((port, reg_init)) => {
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

    fn second_pass_processes(&mut self, body: &Block<ModuleStatement>) {
        let state = &mut self.state;
        let scope_body = self.scope_body;
        let diags = state.diags;

        let Block { span: _, statements } = body;
        for stmt in statements {
            match &stmt.inner {
                // declarations, already handled
                ModuleStatementKind::ConstDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                // blocks, handle now
                ModuleStatementKind::CombinatorialBlock(block) => {
                    let &CombinatorialBlock { span: _, span_keyword, ref block } = block;

                    let domain = Spanned {
                        span: span_keyword,
                        inner: DomainKind::Async,
                    };

                    let mut ir_locals = Arena::default();
                    let ir_block = state.elaborate_ir_block(&mut ir_locals, domain.as_ref(), scope_body, block);
                    let ir_process = ir_block.map(|block| {
                        let ir_body = IrProcessBody { locals: ir_locals, block };
                        IrProcess::Combinatorial(ir_body)
                    });
                    self.processes.push(ir_process);
                }
                ModuleStatementKind::ClockedBlock(block) => {
                    let &ClockedBlock { span: _, span_keyword, ref domain, ref block } = block;

                    let domain = domain.as_ref()
                        .map_inner(|d| state.eval_domain_sync(scope_body, d))
                        .transpose();

                    let ir_process = domain.and_then(|domain| {
                        let mut ir_locals = Arena::default();
                        let ir_domain = SyncDomain {
                            clock: state.domain_signal_to_ir(&domain.inner.clock),
                            reset: state.domain_signal_to_ir(&domain.inner.reset),
                        };

                        let domain = Spanned {
                            span: span_keyword.join(domain.span),
                            inner: DomainKind::Sync(domain.inner),
                        };

                        state.elaborate_ir_block(&mut ir_locals, domain.as_ref(), scope_body, block)
                            .map(|block| {
                                let ir_body = IrProcessBody { locals: ir_locals, block };
                                IrProcess::Clocked(ir_domain, ir_body)
                            })
                    });
                    self.processes.push(ir_process);
                }
                ModuleStatementKind::Instance(_) => {
                    let e = diags.report_todo(stmt.span, "module instance");
                    self.processes.push(Err(e));
                }
            }
        }
    }

    fn elaborate_module_declaration_wire(&mut self, decl: &WireDeclaration) -> Result<(Wire, Option<IrProcess>), ErrorGuaranteed> {
        let state = &mut self.state;
        let scope_body = self.scope_body;
        let diags = state.diags;

        let WireDeclaration { span: _, id, sync, ty, value } = decl;
        let domain_span = sync.span;
        let ty_span = ty.span;

        // evaluate
        // TODO allow clock wires
        let domain = sync.as_ref()
            .map_inner(|sync| state.eval_domain(scope_body, sync))
            .transpose();
        let ty = state.eval_expression_as_ty_hardware(scope_body, ty, "wire");

        let mut ir_locals = IrLocals::default();
        let mut ir_statements = vec![];
        let value = value.as_ref()
            .map(|value| {
                let eval = state.eval_expression_as_ir(&mut ir_locals, &mut ir_statements, scope_body, value)?;
                Ok(Spanned { span: value.span, inner: eval })
            })
            .transpose();

        let domain = domain?;
        let ty = ty?;
        let value = value?;

        if let Some(value) = &value {
            let ty_spanned = Spanned { span: ty_span, inner: &ty.as_type() };

            // check type and get domain
            let (err_ty, value_domain) = match &value.inner {
                MaybeCompile::Compile(c) => {
                    let value_spanned = Spanned { span: value.span, inner: c };
                    let err_ty = state.check_type_contains_compile_value(decl.span, ty_spanned, value_spanned);
                    (err_ty, &ValueDomain::CompileTime)
                }
                MaybeCompile::Other(v) => {
                    let value_ty_spanned = Spanned { span: value.span, inner: &v.ty.as_type() };
                    let err_ty = state.check_type_contains_type(decl.span, ty_spanned, value_ty_spanned);
                    (err_ty, &v.domain)
                }
            };

            // check domain
            let domain = domain.clone().map_inner(|d| ValueDomain::from_domain_kind(d));
            let value_domain_spanned = Spanned { span: value.span, inner: value_domain };
            let err_domain = state.check_valid_domain_crossing(decl.span, domain.as_ref(), value_domain_spanned);

            err_ty?;
            err_domain?;
        }

        // build wire
        let ir_wire = self.ir_wires.push(IrWireInfo {
            debug_name: id.string().map(String::from),
            ty: ty.to_ir(),
        });
        let wire = state.wires.push(WireInfo {
            id: id.clone(),
            domain,
            ty: Spanned { span: ty_span, inner: ty },
            ir: ir_wire,
        });

        // build process
        let process = value.map(|value| {
            let target = IrAssignmentTarget::Wire(ir_wire);
            let source = match value.inner {
                MaybeCompile::Compile(_) => throw!(diags.report_todo(value.span, "compile-time wire value")),
                MaybeCompile::Other(v) => v.expr,
            };
            ir_statements.push(Ok(IrStatement::Assign(target, source)));

            let ir_statements = ir_statements.into_iter().try_collect()?;
            let ir_body = IrProcessBody { locals: ir_locals, block: IrBlock { statements: ir_statements } };
            let ir_process = IrProcess::Combinatorial(ir_body);

            Ok(ir_process)
        }).transpose()?;

        Ok((wire, process))
    }

    fn elaborate_module_declaration_reg(&mut self, decl: &RegDeclaration) -> Result<RegisterInit, ErrorGuaranteed> {
        let state = &mut self.state;
        let scope_body = self.scope_body;

        let RegDeclaration { span: _, id, sync, ty, init } = decl;
        let sync_span = sync.span;
        let ty_span = ty.span;
        let init_span = init.span;

        // evaluate
        let sync = state.eval_domain_sync(scope_body, &sync.inner);
        let ty = state.eval_expression_as_ty_hardware(scope_body, ty, "register");
        let init = state.eval_expression_as_compile(scope_body, init.as_ref(), "register reset value");

        let sync = sync?;
        let ty = ty?;

        // check type
        let init = init.and_then(|init| {
            let ty_spanned = Spanned { span: ty_span, inner: &ty.as_type() };
            let init_spanned = Spanned { span: init_span, inner: &init };
            state.check_type_contains_compile_value(decl.span, ty_spanned, init_spanned)?;
            Ok(init)
        });

        // build register
        let ir_reg = self.ir_registers.push(IrRegisterInfo {
            debug_name: id.string().map(String::from),
            ty: ty.to_ir(),
        });
        let reg = state.registers.push(RegisterInfo {
            id: id.clone(),
            domain: Spanned { span: sync_span, inner: sync },
            ty: Spanned { span: ty_span, inner: ty },
            ir: ir_reg,
        });
        Ok(RegisterInit { reg, init })
    }

    fn elaborate_module_declaration_reg_out_port(&mut self, decl: &RegOutPortMarker) -> Result<(Port, RegisterInit), ErrorGuaranteed> {
        let state = &mut self.state;
        let scope_body = self.scope_body;
        let scope_ports = self.scope_ports;

        let RegOutPortMarker { span: _, id, init } = decl;

        // find port (looking only at the port scope to avoid shadowing or hitting outer identifiers)
        let port = state.scopes[scope_ports].find_immediate_str(state.diags, &id.string, Visibility::Private)?;
        let port = match port.value {
            &ScopedEntry::Direct(NamedValue::Port(port)) => Ok(port),
            _ => Err(state.diags.report_internal_error(id.span, "found non-port in port scope")),
        };

        let init = state.eval_expression_as_compile(scope_body, init, "register reset value")
            .map(|i| Spanned { span: init.span, inner: i });
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
                direction_err = Err(state.diags.report(diag))
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
                return Err(state.diags.report(diag));
            }
        };

        direction_err?;

        // check type
        let init = init.and_then(|init| {
            let ty_spanned = Spanned { span: port_info.ty.span, inner: &port_info.ty.inner.as_type() };
            state.check_type_contains_compile_value(decl.span, ty_spanned, init.as_ref())?;
            Ok(init.inner)
        });

        // build register
        let ir_reg = self.ir_registers.push(IrRegisterInfo {
            debug_name: Some(id.string.to_string()),
            ty: port_info.ty.inner.to_ir(),
        });
        let reg = state.registers.push(RegisterInfo {
            id: MaybeIdentifier::Identifier(id.clone()),
            domain: Spanned { span: port_info.domain.span, inner: domain.clone() },
            ty: port_info.ty.clone(),
            ir: ir_reg,
        });
        Ok((port, RegisterInit { reg, init }))
    }
}

#[derive(Debug)]
struct RegisterInit {
    reg: Register,
    init: Result<CompileValue, ErrorGuaranteed>,
}
