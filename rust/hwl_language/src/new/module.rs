use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::{CompileState, MaybeUndefined, ModuleElaboration, ModuleElaborationInfo, Port, PortInfo, Register, RegisterInfo, Wire, WireInfo};
use crate::new::ir::IrModuleInfo;
use crate::new::misc::{PortDomain, ScopedEntry};
use crate::new::types::HardwareType;
use crate::new::value::{CompileValue, ScopedValue};
use crate::syntax::ast;
use crate::syntax::ast::{Block, DomainKind, GenericParameter, ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle, ModuleStatement, ModuleStatementKind, PortDirection, PortKind, RegDeclaration, RegOutPortMarker, Spanned, WireDeclaration};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::data::IndexMapExt;
use crate::util::{result_pair, ResultExt};
use indexmap::IndexMap;
use itertools::zip_eq;

impl CompileState<'_> {
    pub fn elaborate_module_new(&mut self, module_elaboration: ModuleElaboration) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let ModuleElaborationInfo { item, args } = self.elaborated_modules[module_elaboration].clone();
        let &ast::ItemDefModule { span: def_span, vis: _, id: _, ref params, ref ports, ref body } = &self.parsed[item];
        let scope_file = self.file_scope(item.file())?;

        let scope_params = self.elaborate_module_generics(def_span, scope_file, params, args)?;
        let scope_ports = self.elaborate_module_ports(def_span, scope_params, ports);
        self.elaborate_module_body(scope_ports, body)
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
                            Ok(ScopedEntry::Direct(ScopedValue::Compile(CompileValue::Type(param_ty))))
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

    fn elaborate_module_ports(&mut self, def_span: Span, params_scope: Scope, ports: &Spanned<Vec<ModulePortItem>>) -> Scope {
        let diags = self.diags;
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
                        let port = self.ports.push(PortInfo {
                            def_id_span: id.span,
                            direction: *direction,
                            domain,
                            ty,
                        });
                        Ok(ScopedEntry::Direct(ScopedValue::Port(port)))
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
                            let port = self.ports.push(PortInfo {
                                def_id_span: id.span,
                                direction: *direction,
                                domain: domain.clone(),
                                ty,
                            });
                            Ok(ScopedEntry::Direct(ScopedValue::Port(port)))
                        });

                        self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                    }
                }
            }
        }

        ports_scope
    }

    fn elaborate_module_body(&mut self, scope_ports: Scope, body: &Block<ModuleStatement>) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let diags = self.diags;
        let scope_body = self.scopes.new_child(scope_ports, body.span, Visibility::Private);
        let Block { span: _, statements } = body;

        // first pass: populate scope with declarations
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        let mut register_initial_values = IndexMap::new();
        let mut port_register_connections = IndexMap::new();

        for stmt in statements {
            match &stmt.inner {
                // declarations
                ModuleStatementKind::ConstDeclaration(decl) => {
                    let entry = self.const_eval_and_check(scope_body, decl)
                        .map(|value| ScopedEntry::Direct(ScopedValue::Compile(value)));
                    self.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let reg = self.elaborate_module_declaration_reg(scope_body, decl);
                    let entry = reg.map(|reg_init| {
                        register_initial_values.insert_first(reg_init.reg, reg_init.init);

                        ScopedEntry::Direct(ScopedValue::Register(reg_init.reg))
                    });
                    self.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let wire = self.elaborate_module_declaration_wire(scope_body, decl);
                    let entry = wire.map(|wire| ScopedEntry::Direct(ScopedValue::Wire(wire)));
                    self.scopes[scope_body].maybe_declare(diags, decl.id.as_ref(), entry, Visibility::Private);
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    // declare register that shadows the outer port, which is exactly what we want
                    match self.elaborate_module_declaration_reg_out_port(scope_ports, scope_body, decl) {
                        Ok((port, reg_init)) => {
                            register_initial_values.insert_first(reg_init.reg, reg_init.init);
                            port_register_connections.insert_first(port, reg_init.reg);

                            let entry = Ok(ScopedEntry::Direct(ScopedValue::Register(reg_init.reg)));
                            self.scopes[scope_body].declare(diags, &decl.id, entry, Visibility::Private);
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

        // second pass: codegen for the actual blocks
        for stmt in statements {
            match &stmt.inner {
                // declarations, already handled
                ModuleStatementKind::ConstDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                // non declarations, skip
                ModuleStatementKind::CombinatorialBlock(_) => throw!(diags.report_todo(stmt.span, "combinatorial block")),
                ModuleStatementKind::ClockedBlock(_) => throw!(diags.report_todo(stmt.span, "clocked block")),
                ModuleStatementKind::Instance(_) => throw!(diags.report_todo(stmt.span, "instance")),
            }
        }

        // check driver validness
        throw!(diags.report_internal_error(body.span, "check module drivers"));

        let _ = register_initial_values;
        let _ = port_register_connections;

        // return result
        // Ok(IrModuleInfo {
        //     ports: Arena::default(),
        //     registers: Arena::default(),
        //     wires: Arena::default(),
        //     processes: vec![],
        // })
    }

    fn elaborate_module_declaration_wire(&mut self, scope_body: Scope, decl: &WireDeclaration) -> Result<Wire, ErrorGuaranteed> {
        let WireDeclaration { span: _, id, sync, ty, value } = decl;
        let domain_span = sync.span;
        let ty_span = ty.span;

        // evaluate
        let domain = self.eval_domain(scope_body, &sync.inner);
        let ty = self.eval_expression_as_ty_hardware(scope_body, ty, "wire");
        let value = value.as_ref()
            .map(|value| Ok((value.span, self.eval_expression(scope_body, value)?)))
            .transpose();

        let domain = domain?;
        let ty = ty?;
        let value = value?;

        // check type
        if let Some((value_span, value)) = value {
            let ty_spanned = Spanned { span: ty_span, inner: &ty.as_type() };
            let value_spanned = Spanned { span: value_span, inner: &value };
            self.check_type_contains_value(decl.span, ty_spanned, value_spanned)?;
        }

        // build wire
        let wire = self.wires.push(WireInfo {
            def_id_span: id.span(),
            domain: Spanned { span: domain_span, inner: domain },
            ty: Spanned { span: ty_span, inner: ty },
        });

        Ok(wire)
    }

    fn elaborate_module_declaration_reg(&mut self, scope_body: Scope, decl: &RegDeclaration) -> Result<RegisterInit, ErrorGuaranteed> {
        let RegDeclaration { span: _, id, sync, ty, init } = decl;
        let ty_span = ty.span;
        let init_span = init.span;

        // evaluate
        let sync = self.eval_domain_sync(scope_body, &sync.inner);
        let ty = self.eval_expression_as_ty_hardware(scope_body, ty, "register");
        let init = self.eval_expression_as_compile_or_undefined(scope_body, init.as_ref(), "register reset value");

        let sync = sync?;
        let ty = ty?;

        // check type
        let init = init.and_then(|init| {
            match &init {
                MaybeUndefined::Undefined => {}
                MaybeUndefined::Defined(init) => {
                    let ty_spanned = Spanned { span: ty_span, inner: &ty.as_type() };
                    let init_spanned = Spanned { span: init_span, inner: init };
                    self.check_type_contains_compile_value(decl.span, ty_spanned, init_spanned)?;
                }
            }
            Ok(init)
        });

        // build register
        let reg = self.registers.push(RegisterInfo {
            def_id_span: id.span(),
            domain: sync,
            ty: Spanned { span: ty_span, inner: ty },
        });
        Ok(RegisterInit { reg, init })
    }

    fn elaborate_module_declaration_reg_out_port(&mut self, scope_ports: Scope, scope_body: Scope, decl: &RegOutPortMarker) -> Result<(Port, RegisterInit), ErrorGuaranteed> {
        let RegOutPortMarker { span: _, id, init } = decl;

        // find port (looking only at the port scope to avoid shadowing or hitting outer identifiers)
        let port = self.scopes[scope_ports].find_immediate_str(self.diags, &id.string, Visibility::Private)?;
        let port = match port.value {
            &ScopedEntry::Direct(ScopedValue::Port(port)) => Ok(port),
            _ => Err(self.diags.report_internal_error(id.span, "found non-port in port scope")),
        };

        let init = self.eval_expression_as_compile_or_undefined(scope_body, init, "register reset value")
            .map(|i| Spanned { span: init.span, inner: i });
        let port = port?;

        let port_info = &self.ports[port];
        let mut direction_err = Ok(());

        // check port is output
        match port_info.direction.inner {
            PortDirection::Input => {
                let diag = Diagnostic::new("only output ports can be marked as registers")
                    .add_error(id.span, "port marked as register here")
                    .add_info(port_info.direction.span, "port declared as input here")
                    .finish();
                direction_err = Err(self.diags.report(diag))
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
                return Err(self.diags.report(diag));
            }
        };

        direction_err?;

        // check type
        let init = init.and_then(|init| {
            match &init.inner {
                MaybeUndefined::Undefined => {}
                MaybeUndefined::Defined(init_compile) => {
                    let ty_spanned = Spanned { span: port_info.ty.span, inner: &port_info.ty.inner.as_type() };
                    let init_spanned = Spanned { span: init.span, inner: init_compile };
                    self.check_type_contains_compile_value(decl.span, ty_spanned, init_spanned)?;
                }
            }
            Ok(init.inner)
        });

        // build register
        let reg = self.registers.push(RegisterInfo {
            def_id_span: id.span,
            domain: domain.clone(),
            ty: port_info.ty.clone(),
        });
        Ok((port, RegisterInit { reg, init }))
    }
}

#[derive(Debug)]
struct RegisterInit {
    reg: Register,
    init: Result<MaybeUndefined<CompileValue>, ErrorGuaranteed>,
}