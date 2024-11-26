use crate::data::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::data::parsed::{AstRefItem, AstRefModule};
use crate::new::compile::{CompileState, ElaborationStackEntry, ModuleElaborationInfo};
use crate::new::function::{FunctionBody, FunctionValue};
use crate::new::misc::TypeOrValue;
use crate::new::value::CompileValue;
use crate::syntax::ast::{ConstDeclaration, Item, ItemDefModule, ItemDefType, Spanned};
use crate::util::data::IndexMapExt;

impl CompileState<'_> {
    pub fn eval_item_as_ty_or_value(&mut self, item: AstRefItem) -> &TypeOrValue<CompileValue> {
        // the cache lookup is written in a strange way to workaround borrow checker limitations when returning values
        if !self.items.contains_key(&item) {
            let result = self.check_compile_loop(ElaborationStackEntry::Item(item), |s| {
                s.eval_item_as_ty_or_value_new(item)
            }).unwrap_or_else(|e| TypeOrValue::Error(e));

            self.items.insert_first(item, result)
        } else {
            self.items.get(&item).unwrap()
        }
    }

    fn eval_item_as_ty_or_value_new(&mut self, item: AstRefItem) -> TypeOrValue<CompileValue> {
        let diags = self.diags;
        let file_scope = match self.file_scope(item.file()) {
            Ok(file_scope) => file_scope,
            Err(e) => return e.into(),
        };

        match &self.parsed[item] {
            // imports were already handled in a separate import resolution pass
            Item::Import(item) => {
                diags.report_internal_error(item.span, "import items should have been resolved in a separate pass already").into()
            }
            Item::Const(item) => {
                let ConstDeclaration { span: _, vis: _, id: _, ty, value } = item;

                let ty = ty.as_ref().map(|ty| Spanned {
                    span: ty.span,
                    inner: self.eval_expression_as_ty(file_scope, ty),
                });

                let value_raw = match self.eval_expression_as_value_compile(file_scope, value) {
                    Ok(value_raw) => value_raw,
                    Err(e) => return e.into(),
                };

                let value = if let Some(ty) = ty {
                    match ty.inner.contains_value(&value_raw) {
                        Ok(true) => value_raw,
                        Ok(false) => {
                            // TODO common type diagnostic formatting
                            let diag = Diagnostic::new("const value does not fit in type")
                                .add_error(item.span, "const value does not fit in type")
                                .add_info(ty.span, format!("type `{}` defined here", ty.inner.to_diagnostic_string()))
                                .add_info(value.span, format!("value `{}` defined here", value_raw.to_diagnostic_string()))
                                .finish();
                            return diags.report(diag).into();
                        }
                        Err(e) => return e.into(),
                    }
                } else {
                    value_raw
                };

                TypeOrValue::Value(value)
            }
            Item::Type(item) => {
                let ItemDefType { span: _, vis: _, id: _, params, inner } = item;
                match params {
                    None => TypeOrValue::Type(self.eval_expression_as_ty(file_scope, inner)),
                    Some(_) => {
                        let func = FunctionValue {
                            outer_scope: file_scope.clone(),
                            body: FunctionBody::Type,
                        };
                        TypeOrValue::Value(CompileValue::Function(func))
                    },
                }
            }
            Item::Struct(item) => diags.report_todo(item.span, "visit item kind Struct").into(),
            Item::Enum(item) => diags.report_todo(item.span, "visit item kind Enum").into(),
            Item::Function(item) => diags.report_todo(item.span, "visit item kind Function").into(),
            Item::Module(item_module) => {
                let ItemDefModule { span: _, vis: _, id: _, params, ports: _, body: _ } = item_module;

                match params {
                    None => {
                        let elaboration = self.elaborated_modules.push(ModuleElaborationInfo {
                            item: AstRefModule::new_unchecked(item),
                            args: None,
                        });
                        let module = self.elaborate_module(elaboration);

                        module.map_or_else(
                            |e| e.into(),
                            |module| TypeOrValue::Value(CompileValue::Module(module)),
                        )
                    }
                    Some(_) => {
                        diags.report_todo(item_module.span, "visit item kind Module with params").into()
                    }
                }
            }
            Item::Interface(item) => {
                diags.report_todo(item.span, "visit item kind Interface").into()
            }
        }
    }
}