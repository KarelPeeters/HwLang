use crate::front::assignment::VariableValues;
use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::function::{FunctionBody, FunctionValue};
use crate::front::misc::ScopedEntry;
use crate::front::scope::Scope;
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast::{Args, ConstDeclaration, Item, ItemDefFunction, TypeDeclaration};
use crate::syntax::parsed::{AstRefItem, AstRefModule};
use crate::syntax::source::FileId;

impl CompileItemContext<'_, '_> {
    pub fn eval_item_new(&mut self, item: AstRefItem) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let file_scope = self.refs.shared.file_scope(item.file())?;

        match &self.refs.fixed.parsed[item] {
            Item::Import(item_inner) => {
                let reason = "import items should have been resolved in a separate pass already";
                Err(diags.report_internal_error(item_inner.span, reason))
            }
            Item::Instance(instance) => {
                // TODO this is a bit weird, we're not actually evaluating the item
                self.elaborate_module_header(file_scope, instance)?;
                Ok(CompileValue::UNIT)
            }
            Item::Const(item_inner) => self.const_eval(file_scope, item_inner),
            Item::Type(item_inner) => self.ty_eval(Some(item.file()), file_scope, item_inner),
            Item::Struct(item_inner) => Err(diags.report_todo(item_inner.span, "visit item kind Struct")),
            Item::Enum(item_inner) => Err(diags.report_todo(item_inner.span, "visit item kind Enum")),
            Item::Function(item_inner) => {
                let ItemDefFunction {
                    span: _,
                    vis: _,
                    id,
                    params,
                    ret_ty,
                    body,
                } = item_inner;

                // TODO top-level item functions are also trivial items, we didn't really need to construct any of this yet
                let function = FunctionValue {
                    decl_span: id.span,
                    scope_captured: item.file(),
                    params: params.clone(),
                    body_span: body.span,
                    body: FunctionBody::FunctionBodyBlock {
                        body: body.clone(),
                        ret_ty: ret_ty.as_ref().map(|ret_ty| Box::new(ret_ty.clone())),
                    },
                };
                Ok(CompileValue::Function(function))
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
                                span: module.id.span,
                                inner: vec![],
                            };
                            let _ = self.refs.elaborate_module(ast_ref, Some(args));
                        }
                    }
                }

                Ok(CompileValue::Module(ast_ref))
            }
        }
    }

    // TODO create combined infrastructure for all these declarations that are allowed
    //   at all levels (file, module, function, block, ...)
    pub fn ty_eval<V>(
        &mut self,
        file_scope: Option<FileId>,
        scope: &Scope,
        decl: &TypeDeclaration<V>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let TypeDeclaration {
            span: _,
            vis: _,
            id,
            params,
            body,
        } = decl;

        match params {
            None => {
                let vars = VariableValues::new_no_vars();
                let ty = self.eval_expression_as_ty(scope, &vars, body)?;
                Ok(CompileValue::Type(ty.inner))
            }
            Some(params) => match file_scope {
                None => Err(diags.report_todo(id.span(), "parametrized type that captures non-file scope")),
                Some(file_scope) => {
                    let func = FunctionValue {
                        decl_span: id.span(),
                        scope_captured: file_scope,
                        params: params.clone(),
                        body_span: body.span,
                        body: FunctionBody::TypeAliasExpr(body.clone()),
                    };
                    Ok(CompileValue::Function(func))
                }
            },
        }
    }

    pub fn ty_eval_and_declare<V>(&mut self, scope: &mut Scope, decl: &TypeDeclaration<V>) {
        let diags = self.refs.diags;
        let entry = self
            .ty_eval(None, scope, decl)
            .map(|v| ScopedEntry::Value(MaybeCompile::Compile(v)));
        scope.maybe_declare(diags, decl.id.as_ref(), entry);
    }

    pub fn const_eval<V>(
        &mut self,
        scope: &Scope,
        decl: &ConstDeclaration<V>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let ConstDeclaration {
            span: _,
            vis: _,
            id,
            ty,
            value,
        } = decl;
        let vars = VariableValues::new_no_vars();

        let ty = ty
            .as_ref()
            .map(|ty| self.eval_expression_as_ty(scope, &vars, ty))
            .transpose();
        let value = self.eval_expression_as_compile(scope, &vars, value, "const value");

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

    pub fn const_eval_and_declare<V>(&mut self, scope: &mut Scope, decl: &ConstDeclaration<V>) {
        let diags = self.refs.diags;
        let entry = self
            .const_eval(scope, decl)
            .map(|v| ScopedEntry::Value(MaybeCompile::Compile(v)));
        scope.maybe_declare(diags, decl.id.as_ref(), entry);
    }
}
