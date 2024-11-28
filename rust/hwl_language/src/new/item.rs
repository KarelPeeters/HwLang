use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::parsed::{AstRefItem, AstRefModule};
use crate::new::compile::{CompileState, ElaborationStackEntry, ModuleElaborationInfo};
use crate::new::function::{FunctionBody, FunctionValue, ReturnType};
use crate::new::value::CompileValue;
use crate::syntax::ast::{ConstDeclaration, Item, ItemDefModule, ItemDefType, Spanned};
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;

impl CompileState<'_> {
    pub fn eval_item_as_ty_or_value(&mut self, item: AstRefItem) -> Result<&CompileValue, ErrorGuaranteed> {
        // the cache lookup is written in a strange way to workaround borrow checker limitations when returning values
        if !self.items.contains_key(&item) {
            let result = self.check_compile_loop(ElaborationStackEntry::Item(item), |s| {
                s.eval_item_as_ty_or_value_new(item)
            }).unwrap_or_else(|e| Err(e));

            self.items.insert_first(item, result).as_ref_ok()
        } else {
            self.items.get(&item).unwrap().as_ref_ok()
        }
    }

    fn eval_item_as_ty_or_value_new(&mut self, item: AstRefItem) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.diags;
        let file_scope = self.file_scope(item.file())?;

        match &self.parsed[item] {
            // imports were already handled in a separate import resolution pass
            Item::Import(item_inner) => {
                Err(diags.report_internal_error(
                    item_inner.span,
                    "import items should have been resolved in a separate pass already",
                ))
            }
            Item::Const(item_inner) => {
                let ConstDeclaration { span: _, vis: _, id: _, ty, value } = item_inner;

                let ty = ty.as_ref()
                    .map(|ty| Ok(Spanned { span: ty.span, inner: self.eval_expression_as_ty(file_scope, ty)? }))
                    .transpose();

                let value_eval = self.eval_expression_as_compile(file_scope, value)?;
                let ty = ty?;

                // check type
                if let Some(ty) = ty {
                    if !ty.inner.contains_value(&value_eval) {
                        // TODO common type diagnostic formatting
                        let diag = Diagnostic::new("const value does not fit in type")
                            .add_error(item_inner.span, "const value does not fit in type")
                            .add_info(ty.span, format!("type `{}` defined here", ty.inner.to_diagnostic_string()))
                            .add_info(value.span, format!("value `{}` defined here", value_eval.to_diagnostic_string()))
                            .finish();
                        return Err(diags.report(diag));
                    }
                };

                Ok(value_eval)
            }
            Item::Type(item_inner) => {
                let ItemDefType { span: _, vis: _, id: _, params, inner } = item_inner;
                match params {
                    None => Ok(CompileValue::Type(self.eval_expression_as_ty(file_scope, inner)?)),
                    Some(params) => {
                        let func = FunctionValue {
                            outer_scope: file_scope.clone(),
                            item,
                            params: params.clone(),
                            ret_ty: ReturnType::Type,
                            body_span: inner.span,
                            body: FunctionBody::Expression(inner.clone()),
                        };
                        Ok(CompileValue::Function(func))
                    }
                }
            }
            Item::Struct(item_inner) => Err(diags.report_todo(item_inner.span, "visit item kind Struct")),
            Item::Enum(item_inner) => Err(diags.report_todo(item_inner.span, "visit item kind Enum")),
            Item::Function(item_inner) => Err(diags.report_todo(item_inner.span, "visit item kind Function")),
            Item::Module(item_inner) => {
                let ItemDefModule { span: _, vis: _, id: _, params, ports: _, body: _ } = item_inner;

                match params {
                    None => {
                        let elaboration = self.elaborated_modules.push(ModuleElaborationInfo {
                            item: AstRefModule::new_unchecked(item),
                            args: None,
                        });
                        let ir_module = self.elaborate_module(elaboration)?;
                        Ok(CompileValue::Module(ir_module))
                    }
                    Some(_) =>
                        Err(diags.report_todo(item_inner.span, "visit item kind Module with params")),
                }
            }
            Item::Interface(item) =>
                Err(diags.report_todo(item.span, "visit item kind Interface")),
        }
    }
}