use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::{ArenaPorts, ArenaVariables, CompileItemContext, CompileRefs};
use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::function::{CapturedScope, FunctionBody, FunctionValue};
use crate::front::scope::{DeclaredValueSingle, ScopedEntry};
use crate::front::scope::{NamedValue, Scope};
use crate::front::value::{CompileValue, Value};
use crate::front::variables::VariableValues;
use crate::syntax::ast::{
    Args, CommonDeclaration, ConstDeclaration, FunctionDeclaration, Identifier, Item, ItemDeclaration, MaybeIdentifier,
    Parameter, Spanned, TypeDeclaration,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModule};
use crate::syntax::pos::Span;
use crate::throw;

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

impl CompileRefs<'_, '_> {
    pub fn elaborate_item_params<I: Copy + Into<AstRefItem>>(
        &self,
        item: I,
        params: &Option<Spanned<Vec<Parameter>>>,
        args: Option<Args<Option<Identifier>, Spanned<CompileValue>>>,
    ) -> Result<ElaboratedItemParams<I>, ErrorGuaranteed> {
        let diags = self.diags;

        let def_span = self.fixed.parsed[item.into()].common_info().span_short;
        let scope_file = self.shared.file_scope(item.into().file())?;
        let mut ctx = CompileItemContext::new(*self, None, ArenaVariables::new(), ArenaPorts::new());

        let param_values = match (params, args) {
            (None, None) => None,
            (Some(params), Some(args)) => {
                // We can't use the scope returned here, since it is only valid for the current variables arena,
                //   which will be different during body elaboration.
                //   Instead, we'll recreate the scope from the returned parameter values.
                let mut vars = VariableValues::new_root(&ctx.variables);
                let (_, param_values) = ctx.match_args_to_params_and_typecheck(&mut vars, scope_file, params, &args)?;
                Some(param_values)
            }
            _ => throw!(diags.report_internal_error(def_span, "mismatched generic arguments presence")),
        };

        Ok(ElaboratedItemParams {
            item,
            params: param_values,
        })
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
                // TODO this is a bit weird, we're not actually evaluating the item
                let mut vars = VariableValues::new_root(&self.variables);
                self.elaborate_module_header(file_scope, &mut vars, instance)?;
                Ok(CompileValue::UNIT)
            }
            Item::CommonDeclaration(decl) => {
                let ItemDeclaration { vis: _, decl } = decl;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_declaration(file_scope, &mut vars, decl)
            }
            Item::Module(module) => {
                let ast_ref = AstRefModule::new_unchecked(item, module);

                // elaborate modules without params immediately, for earlier error messages
                // TODO this is a strange place to do this, modules don't _really_ participate in item evaluation
                // TODO this should respect the elaboration set settings
                if let Some(args) = args_if_empty_params(span_short, module.params.as_ref()) {
                    let _ = self.refs.elaborate_module(ast_ref, args);
                }

                Ok(CompileValue::Module(ast_ref))
            }
            Item::Interface(interface) => {
                let ast_ref = AstRefInterface::new_unchecked(item, interface);

                if let Some(args) = args_if_empty_params(span_short, interface.params.as_ref()) {
                    let _ = self.refs.elaborate_interface(ast_ref, args);
                }

                Ok(CompileValue::Interface(ast_ref))
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

fn args_if_empty_params(
    span_short: Span,
    params: Option<&Spanned<Vec<Parameter>>>,
) -> Option<Option<Args<Option<Identifier>, Spanned<CompileValue>>>> {
    match params {
        None => Some(None),
        Some(params) => {
            if params.inner.is_empty() {
                let args = Args {
                    span: span_short,
                    inner: vec![],
                };
                Some(Some(args))
            } else {
                None
            }
        }
    }
}
