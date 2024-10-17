use crate::data::compiled::{Constant, ConstantInfo, FunctionSignatureInfo, GenericParameter, GenericTypeParameterInfo, GenericValueParameterInfo, Item, ItemChecked, ModulePortInfo, ModuleSignatureInfo};
use crate::data::diagnostic::{Diagnostic, ErrorGuaranteed};
use crate::data::parsed::ModulePortAstReference;
use crate::front::checking::DomainUserControlled;
use crate::front::common::{ExpressionContext, ScopedEntry, TypeOrValue, ValueDomainKind};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{Constructor, EnumTypeInfo, GenericArguments, GenericParameters, MaybeConstructor, NominalTypeUnique, StructTypeInfo, Type};
use crate::front::values::{FunctionReturnValue, ModuleValueInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{ConstDeclaration, DomainKind, EnumVariant, GenericParameterKind, ItemDefEnum, ItemDefFunction, ItemDefModule, ItemDefStruct, ItemDefType, ItemImport, PortKind, Spanned, StructField, SyncDomain};
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use itertools::enumerate;

impl CompileState<'_, '_> {
    // TODO this signature is wrong: items are not always type constructors
    // TODO clarify: this resolves the _signature_, not the body, right?
    //   for type aliases it appears to resolve the body too.
    pub fn resolve_item_signature_new(&mut self, item: Item) -> MaybeConstructor<TypeOrValue> {
        // check that this is indeed a new query
        assert!(self.compiled[item].signature.is_none());

        // item lookup
        let item_info = &self.compiled[item];
        let item_ast_reference = item_info.ast_ref;
        let item_ast = self.parsed.item_ast(item_ast_reference);
        let file_scope = match self.compiled.file_scopes.get(&item_info.ast_ref.file).unwrap() {
            Ok(scope_file) => scope_file.scope_inner_import,
            &Err(e) => return MaybeConstructor::Error(e),
        };

        // always use a static context for types
        let ctx = &ExpressionContext::NotFunctionBody;

        // actual resolution
        match *item_ast {
            // resolving import signatures doesn't make sense
            ast::Item::Import(ItemImport { span, .. }) => {
                let e = self.diags.report_internal_error(span, "import item should not be resolved directly");
                MaybeConstructor::Error(e)
            }
            // type definitions
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, ref params, ref inner }) => {
                self.resolve_new_generic_def(item, file_scope, params.as_ref(), |s, _args, scope_inner| {
                    TypeOrValue::Type(s.eval_expression_as_ty(scope_inner, inner))
                })
            }
            ast::Item::Struct(ItemDefStruct { span, vis: _, id: _, ref params, ref fields }) => {
                self.resolve_new_generic_def(item, file_scope, params.as_ref(), |s, args, scope_inner| {
                    // map fields
                    let mut fields_map = IndexMap::new();
                    for field in fields {
                        let StructField { span: _, id: field_id, ty } = field;
                        let field_ty = s.eval_expression_as_ty(scope_inner, ty);

                        let prev = fields_map.insert(field_id.string.clone(), (field_id, field_ty));
                        if let Some(prev) = prev {
                            let diag = Diagnostic::new_defined_twice("struct field", span, field_id, prev.0);
                            let err = s.diags.report(diag);
                            return TypeOrValue::Type(Type::Error(err));
                        }
                    }

                    // result
                    let ty = StructTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        fields: fields_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    TypeOrValue::Type(Type::Struct(ty))
                })
            }
            ast::Item::Enum(ItemDefEnum { span, vis: _, id: _, ref params, ref variants }) => {
                self.resolve_new_generic_def(item, file_scope, params.as_ref(), |s, args, scope_inner| {
                    // map variants
                    let mut variants_map = IndexMap::new();
                    for variant in variants {
                        let EnumVariant { span: _, id: variant_id, content } = variant;

                        let content = content.as_ref()
                            .map(|content| s.eval_expression_as_ty(scope_inner, content));

                        let prev = variants_map.insert(variant_id.string.clone(), (variant_id, content));
                        if let Some(prev) = prev {
                            let diag = Diagnostic::new_defined_twice("enum variant", span, variant_id, prev.0);
                            let err = s.diags.report(diag);
                            return TypeOrValue::Type(Type::Error(err));
                        }
                    }

                    // result
                    let ty = EnumTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        variants: variants_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    TypeOrValue::Type(Type::Enum(ty))
                })
            }
            // value definitions
            ast::Item::Module(ItemDefModule { span: _, vis: _, id: _, ref params, ref ports, ref body }) => {
                self.resolve_new_generic_def(item, file_scope, params.as_ref(), |s, args, scope_inner| {
                    // yet another sub-scope for the ports that refer to each other
                    let scope_ports = s.compiled.scopes.new_child(scope_inner, ports.span.join(body.span), Visibility::Private);

                    // map ports
                    let mut port_vec = vec![];

                    for (port_index, port) in enumerate(&ports.inner) {
                        let ast::ModulePort { span: _, id: port_id, direction, kind, } = port;

                        let module_port_info = ModulePortInfo {
                            ast: ModulePortAstReference {
                                item: item_ast_reference,
                                index: port_index,
                            },
                            direction: direction.inner,
                            kind: match &kind.inner {
                                PortKind::Clock => PortKind::Clock,
                                PortKind::Normal { domain: sync, ty } => {
                                    PortKind::Normal {
                                        domain: match &sync.inner {
                                            DomainKind::Async => DomainKind::Async,
                                            DomainKind::Sync(SyncDomain { clock, reset }) => {
                                                let clock = s.eval_expression_as_value(ctx, scope_ports, clock);
                                                let reset = s.eval_expression_as_value(ctx, scope_ports, reset);
                                                DomainKind::Sync(SyncDomain { clock, reset })
                                            }
                                        },
                                        ty: s.eval_expression_as_ty(scope_ports, ty),
                                    }
                                }
                            },
                        };
                        let module_port = s.compiled.module_ports.push(module_port_info);
                        port_vec.push(module_port);

                        s.compiled.scopes[scope_ports].declare(
                            s.diags,
                            &port_id,
                            ScopedEntry::Direct(MaybeConstructor::Immediate(TypeOrValue::Value(Value::ModulePort(module_port)))),
                            Visibility::Private,
                        );
                    }

                    // result
                    let module_info = ModuleSignatureInfo { scope_ports };
                    s.compiled.module_info.insert_first(item, module_info);

                    let module_ty_info = ModuleValueInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        ports: port_vec,
                    };

                    TypeOrValue::Value(Value::Module(module_ty_info))
                })
            }
            ast::Item::Const(ref cst) => {
                let cst = self.process_const_declaration(file_scope, ctx, cst);
                MaybeConstructor::Immediate(TypeOrValue::Value(Value::Constant(cst)))
            },
            ast::Item::Function(ItemDefFunction { span: _, vis: _, id: _, ref params, ref ret_ty, body: _ }) => {
                self.resolve_new_generic_def(item, file_scope, Some(params), |s, args, scope_inner| {
                    // no need to use args for anything, they are mostly used for nominal type uniqueness
                    //   which does not apply to functions
                    let _ = args;

                    let ret_ty = match ret_ty {
                        None => Type::Unit,
                        Some(ret_ty) => s.eval_expression_as_ty(scope_inner, ret_ty),
                    };

                    // keep scope for later
                    let info = FunctionSignatureInfo {
                        scope_inner,
                        ret_ty: ret_ty.clone(),
                    };
                    s.compiled.function_info.insert_first(item, info);

                    // result
                    TypeOrValue::Value(Value::FunctionReturn(FunctionReturnValue { item, ret_ty }))
                })
            }
            ast::Item::Interface(_) =>
                MaybeConstructor::Error(self.diags.report_todo(item_ast.common_info().span_short, "interface definition")),
        }
    }

    fn resolve_new_generic_def<T>(
        &mut self,
        item: Item,
        scope_outer: Scope,
        params: Option<&Spanned<Vec<ast::GenericParameter>>>,
        build: impl FnOnce(&mut Self, GenericArguments, Scope) -> T,
    ) -> MaybeConstructor<T> {
        let item_span = self.parsed.item_ast(self.compiled[item].ast_ref).common_info().span_full;
        let scope_inner = self.compiled.scopes.new_child(scope_outer, item_span, Visibility::Private);

        match params {
            None => {
                // there are no parameters, just map directly
                // the scope still needs to be "nested" since the builder needs an owned scope
                let arguments = GenericArguments { vec: vec![] };
                MaybeConstructor::Immediate(build(self, arguments, scope_inner))
            }
            Some(params) => {
                // build inner scope
                let mut parameters = vec![];
                let mut arguments = vec![];

                for param_ast in &params.inner {
                    // TODO name arg/param better
                    let (param, arg) = match &param_ast.kind {
                        &GenericParameterKind::Type(_span) => {
                            let param = self.compiled.generic_type_params.push(GenericTypeParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                            });
                            (GenericParameter::Type(param), TypeOrValue::Type(Type::GenericParameter(param)))
                        }
                        GenericParameterKind::Value(ty_expr) => {
                            let ty = self.eval_expression_as_ty(scope_inner, ty_expr);
                            let param = self.compiled.generic_value_params.push(GenericValueParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                                ty,
                                ty_span: ty_expr.span,
                            });
                            (GenericParameter::Value(param), TypeOrValue::Value(Value::GenericParameter(param)))
                        }
                    };

                    parameters.push(param);
                    arguments.push(arg.clone());

                    // TODO should we nest scopes here, or is incremental declaration in a single scope equivalent?
                    let entry = ScopedEntry::Direct(MaybeConstructor::Immediate(arg));
                    self.compiled[scope_inner].declare(self.diags, &param_ast.id, entry, Visibility::Private);
                }

                let parameters = GenericParameters { vec: parameters };
                let arguments = GenericArguments { vec: arguments };

                // build inner
                let inner = build(self, arguments, scope_inner);

                // result
                let ty_constr = Constructor { parameters, inner };
                MaybeConstructor::Constructor(ty_constr)
            }
        }
    }

    // TODO delay body processing
    pub fn process_const_declaration<V>(&mut self, scope: Scope, ctx: &ExpressionContext, decl: &ConstDeclaration<V>) -> Constant {
        let &ConstDeclaration { span: _, vis: _, ref id, ref ty, ref value } = decl;

        let ty_eval = self.eval_expression_as_ty(scope, ty);
        let value_eval = self.eval_expression_as_value(ctx, scope, value);

        // check ty
        let value_eval = match self.check_type_contains(Some(ty.span), value.span, &ty_eval, &value_eval) {
            Ok(()) => value_eval,
            Err(e) => Value::Error(e),
        };

        // check domain
        let _: Result<(), ErrorGuaranteed> = self.check_domain_assign(
            id.span(),
            &ValueDomainKind::Const,
            value.span,
            &self.domain_of_value(value.span, &value_eval),
            DomainUserControlled::Source,
            "const value must be const",
        );

        // declare constant
        let cst = self.compiled.constants.push(ConstantInfo {
            defining_id: id.clone(),
            ty: ty_eval.clone(),
            value: value_eval,
        });

        cst
    }

    pub fn check_item_body(&mut self, item: Item) -> ItemChecked {
        assert!(self.compiled[item].body.is_none());

        // TODO remove this, no point in forcing this to be two-phase
        let item_signature = self.compiled[item].signature.as_ref()
            .expect("item should already have been checked");
        let item_signature_err = match item_signature {
            &MaybeConstructor::Error(e) => Some(e),
            _ => None,
        };

        let item_ast = self.parsed.item_ast(self.compiled[item].ast_ref);
        let item_span = item_ast.common_info().span_short;

        match item_ast {
            // these items are fully defined by their type, which was already checked earlier
            ast::Item::Import(_) | ast::Item::Type(_) | ast::Item::Struct(_) | ast::Item::Enum(_) => ItemChecked::None,
            // TODO delay const body checking until this phase, to cut more loops
            ast::Item::Const(_) => ItemChecked::None,
            ast::Item::Function(item_ast) =>
                ItemChecked::Function(self.check_function_body(item, item_ast)),
            ast::Item::Module(item_ast) =>
                ItemChecked::Module(self.check_module_body(item, item_ast)),
            ast::Item::Interface(_) => {
                match item_signature_err {
                    Some(e) => ItemChecked::Error(e),
                    None => ItemChecked::Error(self.diags.report_todo(item_span, "interface body")),
                }
            }
        }
    }
}