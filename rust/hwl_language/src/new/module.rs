use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::{CompileState, ModuleElaboration, ModuleElaborationInfo, PortInfo};
use crate::new::ir::IrModuleInfo;
use crate::new::misc::{ScopedEntry, ValueDomain};
use crate::new::types::Type;
use crate::new::value::{CompileValue, ScopedValue};
use crate::syntax::ast;
use crate::syntax::ast::{Block, GenericParameter, ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle, ModuleStatement, ModuleStatementKind, PortKind, Spanned};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::ResultExt;
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
                        if param_ty.contains_value(&arg) {
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
                    let (domain, ty_eval, ty_span) = match &kind.inner {
                        PortKind::Clock => (
                            Ok(ValueDomain::Clock),
                            Ok(Type::Clock),
                            kind.span,
                        ),
                        PortKind::Normal { domain, ty } => (
                            self.eval_domain(ports_scope, &domain.inner)
                                .map(ValueDomain::from_domain_kind),
                            self.eval_expression_as_ty(ports_scope, ty),
                            ty.span,
                        ),
                    };

                    // check type
                    let entry = ty_eval.and_then(|ty_raw| {
                        let ty = self.check_port_ty(ty_raw, ty_span)?;
                        let port = self.ports.push(PortInfo {
                            def_id_span: id.span,
                            direction: direction.inner,
                            domain: domain?,
                            ty,
                        });
                        Ok(ScopedEntry::Direct(ScopedValue::Port(port)))
                    });

                    self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                }
                ModulePortItem::Block(port_item) => {
                    let ModulePortBlock { span: _, domain, ports } = port_item;

                    let domain = self.eval_domain(ports_scope, &domain.inner)
                        .map(ValueDomain::from_domain_kind);

                    for port in ports {
                        let ModulePortInBlock { span: _, id, direction, ty } = port;

                        let ty_eval = self.eval_expression_as_ty(ports_scope, ty);

                        // check type
                        let entry = ty_eval.and_then(|ty_raw| {
                            let ty = self.check_port_ty(ty_raw, ty.span)?;

                            let port = self.ports.push(PortInfo {
                                def_id_span: id.span,
                                direction: direction.inner,
                                domain: domain.as_ref_ok()?.clone(),
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

        let _ = scope_body;

        // first pass: populate scope with declarations
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        for stmt in statements {
            match &stmt.inner {
                // declarations
                ModuleStatementKind::ConstDeclaration(_) => throw!(diags.report_todo(stmt.span, "const declaration")),
                ModuleStatementKind::RegDeclaration(_) => throw!(diags.report_todo(stmt.span, "reg declaration")),
                ModuleStatementKind::WireDeclaration(_) => throw!(diags.report_todo(stmt.span, "wire declaration")),
                ModuleStatementKind::RegOutPortMarker(_) => throw!(diags.report_todo(stmt.span, "reg out port marker")),
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

        // return result
        // Ok(IrModuleInfo {
        //     ports: Arena::default(),
        //     registers: Arena::default(),
        //     wires: Arena::default(),
        //     processes: vec![],
        // })
    }

    fn check_port_ty(&self, ty: Type, ty_span: Span) -> Result<Type, ErrorGuaranteed> {
        match ty.hardware_bit_width() {
            Some(_bit_width) => Ok(ty),
            None => Err(self.diags.report_simple(
                "port type must be representable in hardware",
                ty_span,
                format!("type `{}` is not representable in hardware", ty.to_diagnostic_string()),
            )),
        }
    }
}