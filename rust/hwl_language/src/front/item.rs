use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::function::{CapturedScope, FunctionBody, FunctionValue};
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::value::{CompileValue, Value};
use crate::front::variables::VariableValues;
use crate::syntax::ast::{
    Args, CommonDeclaration, ConstDeclaration, FunctionDeclaration, Item, ItemDeclaration, Spanned, TypeDeclaration,
};
use crate::syntax::parsed::{AstRefItem, AstRefModule};

impl CompileItemContext<'_, '_> {
    pub fn eval_item_new(&mut self, item: AstRefItem) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let file_scope = self.refs.shared.file_scope(item.file())?;

        let item_ast = &self.refs.fixed.parsed[item];
        self.refs.check_should_stop(item_ast.common_info().span_short)?;

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
                let ast_ref = AstRefModule::new_unchecked(item);

                // elaborate modules without params immediately, for earlier error messages
                // TODO this is a strange place to do this, modules don't _really_ participate in item evaluation
                match &module.params {
                    None => {
                        let _ = self.refs.elaborate_module(ast_ref, None);
                    }
                    Some(params) => {
                        if params.inner.is_empty() {
                            let args = Args {
                                span: module.id.span(),
                                inner: vec![],
                            };
                            let _ = self.refs.elaborate_module(ast_ref, Some(args));
                        }
                    }
                }

                Ok(CompileValue::Module(ast_ref))
            }
            Item::Interface(x) => Err(diags.report_todo(x.span, "interface item")),
        }
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
