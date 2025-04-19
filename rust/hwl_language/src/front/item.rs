use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::function::{CapturedScope, FunctionBody, FunctionValue};
use crate::front::module::ElaboratedModuleInfo;
use crate::front::scope::{DeclaredValueSingle, ScopedEntry};
use crate::front::scope::{NamedValue, Scope};
use crate::front::value::{CompileValue, Value};
use crate::front::variables::VariableValues;
use crate::syntax::ast::{
    CommonDeclaration, ConstDeclaration, FunctionDeclaration, Identifier, Item, ItemDeclaration, ItemDefInterface,
    ItemDefModule, MaybeIdentifier, ModuleInstanceItem, Spanned, TypeDeclaration,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModule};

pub struct ElaboratedItemParams<I> {
    pub item: I,
    pub params: Option<Vec<(Identifier, CompileValue)>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedItemKey<I> {
    item: I,
    params: Option<Vec<CompileValue>>,
}

impl<I: Copy> ElaboratedItemParams<I> {
    pub fn cache_key(&self) -> ElaboratedItemKey<I> {
        let &ElaboratedItemParams { item, ref params } = self;
        let param_values = params
            .as_ref()
            .map(|params| params.iter().map(|(_, v)| v.clone()).collect());
        ElaboratedItemKey {
            item,
            params: param_values,
        }
    }
}

impl<'s> CompileItemContext<'_, 's> {
    pub fn eval_item_new(&mut self, item: AstRefItem) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let file_scope = self.refs.shared.file_scope(item.file())?;

        let item_ast = &self.refs.fixed.parsed[item];
        let span_short = item_ast.common_info().span_short;
        self.refs.check_should_stop(span_short)?;

        match item_ast {
            Item::Import(item_inner) => {
                let reason = "import items should have been resolved in a separate pass already";
                Err(diags.report_internal_error(item_inner.span, reason))
            }
            Item::Instance(instance) => {
                let &ModuleInstanceItem {
                    span: _,
                    span_keyword,
                    ref module,
                } = instance;
                let mut vars = VariableValues::new_root(&self.variables);
                let _: &ElaboratedModuleInfo =
                    self.eval_expression_as_module(file_scope, &mut vars, span_keyword, module)?;
                Ok(CompileValue::UNIT)
            }
            Item::CommonDeclaration(decl) => {
                let ItemDeclaration { vis: _, decl } = decl;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_declaration(file_scope, &mut vars, decl)
            }
            Item::Module(module) => {
                let ItemDefModule {
                    span: _,
                    vis: _,
                    id,
                    params,
                    ports,
                    body,
                } = module;
                let item = AstRefModule::new_unchecked(item, module);

                match params {
                    None => {
                        let item_params = ElaboratedItemParams { item, params: None };
                        let (elab_id, _) = self.refs.elaborate_module(item_params)?;
                        Ok(CompileValue::Module(elab_id))
                    }
                    Some(params) => {
                        let func = FunctionValue {
                            decl_span: id.span(),
                            scope_captured: CapturedScope::from_file_scope(item.file()),
                            params: params.clone(),
                            body: Spanned {
                                span: ports.span.join(body.span),
                                inner: FunctionBody::ModulePortsAndBody(item),
                            },
                        };
                        Ok(CompileValue::Function(func))
                    }
                }
            }
            Item::Interface(interface) => {
                let ItemDefInterface {
                    span: _,
                    vis: _,
                    id,
                    params,
                    span_body,
                    port_types: _,
                    views: _,
                } = interface;
                let item = AstRefInterface::new_unchecked(item, interface);

                match params {
                    None => {
                        let item_params = ElaboratedItemParams { item, params: None };
                        let (elab_id, _) = self.refs.elaborate_interface(item_params)?;
                        Ok(CompileValue::Interface(elab_id))
                    }
                    Some(params) => {
                        let func = FunctionValue {
                            decl_span: id.span(),
                            scope_captured: CapturedScope::from_file_scope(item.file()),
                            params: params.clone(),
                            body: Spanned {
                                span: *span_body,
                                inner: FunctionBody::Interface(item),
                            },
                        };
                        Ok(CompileValue::Function(func))
                    }
                }
            }
        }
    }

    pub fn rebuild_params_scope(
        &mut self,
        item: AstRefItem,
        vars: &mut VariableValues,
        params: &Option<Vec<(Identifier, CompileValue)>>,
    ) -> Result<Scope<'s>, ErrorGuaranteed> {
        let file_scope = self.refs.shared.file_scope(item.file())?;
        let full_span = self.refs.fixed.parsed[item].common_info().span_full;

        let mut scope_params = Scope::new_child(full_span, file_scope);
        if let Some(params) = &params {
            for (id, value) in params {
                let var = vars.var_new_immutable_init(
                    &mut self.variables,
                    MaybeIdentifier::Identifier(id.clone()),
                    id.span,
                    Value::Compile(value.clone()),
                );
                let declared = DeclaredValueSingle::Value {
                    span: id.span,
                    value: ScopedEntry::Named(NamedValue::Variable(var)),
                };
                scope_params.declare_already_checked(id.string.clone(), declared);
            }
        }
        Ok(scope_params)
    }

    pub fn eval_declaration(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclaration,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;

        match decl {
            CommonDeclaration::Type(decl) => {
                let TypeDeclaration {
                    span: _,
                    id,
                    params,
                    body,
                } = decl;

                match params {
                    None => {
                        let ty = self.eval_expression_as_ty(scope, vars, body)?;
                        Ok(CompileValue::Type(ty.inner))
                    }
                    Some(params) => {
                        let func = FunctionValue {
                            decl_span: id.span(),
                            scope_captured: CapturedScope::from_scope(diags, scope, vars)?,
                            params: params.clone(),
                            body: Spanned {
                                span: body.span,
                                inner: FunctionBody::TypeAliasExpr(body.clone()),
                            },
                        };
                        Ok(CompileValue::Function(func))
                    }
                }
            }
            CommonDeclaration::Const(decl) => {
                let ConstDeclaration { span: _, id, ty, value } = decl;

                let ty = ty
                    .as_ref()
                    .map(|ty| self.eval_expression_as_ty(scope, vars, ty))
                    .transpose();
                let value = self.eval_expression_as_compile(scope, vars, value, "const value");

                let ty = ty?;
                let value = value?;

                // check type
                if let Some(ty) = ty {
                    let reason = TypeContainsReason::Assignment {
                        span_target: id.span(),
                        span_target_ty: ty.span,
                    };
                    check_type_contains_compile_value(diags, reason, &ty.inner, value.as_ref(), true)?;
                };

                Ok(value.inner)
            }
            CommonDeclaration::Struct(_) => Err(diags.report_todo(decl.info().1.span(), "struct declaration")),
            CommonDeclaration::Enum(_) => Err(diags.report_todo(decl.info().1.span(), "enum declaration")),
            CommonDeclaration::Function(decl) => {
                let FunctionDeclaration {
                    span: _,
                    id,
                    params,
                    ret_ty,
                    body,
                } = decl;

                let body_inner = FunctionBody::FunctionBodyBlock {
                    body: body.clone(),
                    ret_ty: ret_ty.as_ref().map(|ret_ty| Box::new(ret_ty.clone())),
                };
                let function = FunctionValue {
                    decl_span: id.span(),
                    scope_captured: CapturedScope::from_scope(diags, scope, vars)?,
                    params: params.clone(),
                    body: Spanned {
                        span: body.span,
                        inner: body_inner,
                    },
                };
                Ok(CompileValue::Function(function))
            }
        }
    }

    pub fn eval_and_declare_declaration(
        &mut self,
        scope: &mut Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclaration,
    ) {
        let diags = self.refs.diags;
        let (decl_span, id) = decl.info();
        let entry = self.eval_declaration(scope, vars, decl).map(|v| {
            let var = vars.var_new_immutable_init(&mut self.variables, id.clone(), decl_span, Value::Compile(v));
            ScopedEntry::Named(NamedValue::Variable(var))
        });
        scope.maybe_declare(diags, id.as_ref(), entry);
    }
}
