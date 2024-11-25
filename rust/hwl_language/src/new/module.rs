use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::{CompileState, ModuleElaboration, ModuleElaborationInfo, PortInfo};
use crate::new::ir::IrModuleInfo;
use crate::new::misc::{ScopedEntry, TypeOrValue, ValueDomain};
use crate::new::types::Type;
use crate::new::value::{CompileValue, ScopedValue};
use crate::syntax::ast;
use crate::syntax::ast::{Block, GenericParameter, GenericParameterKind, ModulePortBlock, ModulePortInBlock, ModulePortItem, ModulePortSingle, ModuleStatement, ModuleStatementKind, PortKind, Spanned};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::arena::Arena;
use itertools::{zip_eq, Itertools};

impl CompileState<'_> {
    pub fn elaborate_module_new(&mut self, module_elaboration: ModuleElaboration) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let ModuleElaborationInfo { item, args } = self.elaborated_modules[module_elaboration].clone();
        let &ast::ItemDefModule { span: def_span, vis: _, id: _, ref params, ref ports, ref body } = &self.parsed[item];
        let scope_file = self.file_scope(item.file())?;

        let scope_params = self.elaborate_module_generics(def_span, scope_file, params, args)?;
        let scope_ports = self.elaborate_module_ports(def_span, scope_params, ports);
        self.elaborate_module_body(scope_ports, body)
    }

    fn elaborate_module_generics(&mut self, def_span: Span, file_scope: Scope, params: &Option<Spanned<Vec<GenericParameter>>>, args: Option<Vec<TypeOrValue<CompileValue>>>) -> Result<Scope, ErrorGuaranteed> {
        let diags = self.diags;
        let params_scope = self.scopes.new_child(file_scope, def_span, Visibility::Private);

        match (params, args) {
            (None, None) => {}
            (Some(params), Some(args)) => {
                if params.inner.len() != args.len() {
                    throw!(diags.report_internal_error(params.span, "mismatched generic argument count"));
                }

                for (param, arg) in zip_eq(&params.inner, args) {
                    let entry = match (&param.kind, arg) {
                        (&GenericParameterKind::Type(param_span), TypeOrValue::Type(arg)) => {
                            let _: Span = param_span;
                            TypeOrValue::Type(arg)
                        }
                        (GenericParameterKind::Value(param_ty), TypeOrValue::Value(arg)) => {
                            let param_ty_span = param_ty.span;
                            let param_ty = self.eval_expression_as_ty(params_scope, param_ty);

                            if !param_ty.contains_value(&arg).unwrap_or(true) {
                                diags.report_internal_error(
                                    param_ty_span,
                                    format!(
                                        "invalid generic argument: type `{}` does not contain value `{}`",
                                        param_ty.to_diagnostic_string(),
                                        arg.to_diagnostic_string()
                                    ),
                                );
                            }

                            TypeOrValue::Value(ScopedValue::Compile(arg))
                        }

                        (_, TypeOrValue::Error(e)) => TypeOrValue::Error(e),

                        (GenericParameterKind::Value(_), TypeOrValue::Type(_)) |
                        (GenericParameterKind::Type(_), TypeOrValue::Value(_)) => {
                            throw!(diags.report_internal_error(param.span, "mismatched generic argument kind"))
                        }
                    };

                    let entry = ScopedEntry::Direct(entry);
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

                    let (domain, ty_raw, ty_span) = match &kind.inner {
                        PortKind::Clock => (
                            ValueDomain::Clock,
                            Type::Clock,
                            kind.span,
                        ),
                        PortKind::Normal { domain, ty } => (
                            ValueDomain::from_domain_kind(self.eval_domain(ports_scope, &domain.inner)),
                            self.eval_expression_as_ty(ports_scope, ty),
                            ty.span,
                        ),
                    };

                    let ty = self.check_port_ty(ty_raw, ty_span);

                    let port = self.ports.push(PortInfo {
                        def_id_span: id.span,
                        direction: direction.inner,
                        domain,
                        ty,
                    });
                    let entry = ScopedEntry::Direct(TypeOrValue::Value(ScopedValue::Port(port)));
                    self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                }
                ModulePortItem::Block(port_item) => {
                    let ModulePortBlock { span: _, domain, ports } = port_item;

                    let domain = ValueDomain::from_domain_kind(self.eval_domain(ports_scope, &domain.inner));

                    for port in ports {
                        let ModulePortInBlock { span: _, id, direction, ty } = port;

                        let ty_raw = self.eval_expression_as_ty(ports_scope, ty);
                        let ty = self.check_port_ty(ty_raw, ty.span);

                        let port = self.ports.push(PortInfo {
                            def_id_span: id.span,
                            direction: direction.inner,
                            domain: domain.clone(),
                            ty,
                        });
                        let entry = ScopedEntry::Direct(TypeOrValue::Value(ScopedValue::Port(port)));
                        self.scopes[ports_scope].declare(diags, id, entry, Visibility::Private);
                    }
                }
            }
        }

        ports_scope
    }

    fn elaborate_module_body(&mut self, scope_ports: Scope, body: &Block<ModuleStatement>) -> Result<IrModuleInfo, ErrorGuaranteed> {
        let scope_body = self.scopes.new_child(scope_ports, body.span, Visibility::Private);
        let Block { span: _, statements } = body;

        // first pass: populate scope with declarations
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future declarations
        for stmt in statements {
            match &stmt.inner {
                // declarations
                ModuleStatementKind::ConstDeclaration(_) => todo!(),
                ModuleStatementKind::RegDeclaration(_) => todo!(),
                ModuleStatementKind::WireDeclaration(_) => todo!(),
                ModuleStatementKind::RegOutPortMarker(_) => todo!(),
                // non declarations, skip
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
            }
        }

        // second pass: codegen for the actual blocks
        // TODO

        // check driver validness
        // TODO

        // return result
        Ok(IrModuleInfo {
            ports: Arena::default(),
            registers: Arena::default(),
            wires: Arena::default(),
            processes: vec![],
        })
    }

    fn check_port_ty(&self, ty: Type, ty_span: Span) -> Type {
        match ty.hardware_bit_width() {
            Err(e) => Type::Error(e),
            Ok(Some(_bit_width)) => ty,
            Ok(None) => Type::Error(self.diags.report_simple(
                "port type must be representable in hardware",
                ty_span,
                format!("type `{}` is not representable in hardware", ty.to_diagnostic_string()),
            )),
        }
    }
}