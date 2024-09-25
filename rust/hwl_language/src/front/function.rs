use crate::data::compiled::{FunctionChecked, FunctionSignatureInfo, Item, VariableInfo};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::common::{ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::{CompileState, ResolveResult};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::Type;
use crate::front::values::Value;
use crate::syntax::ast::{Block, BlockStatement, BlockStatementKind, ItemDefFunction, VariableDeclaration};

impl CompileState<'_, '_> {
    pub fn check_function_body(&mut self, func_item: Item, funct_ast: &ItemDefFunction) -> ResolveResult<FunctionChecked> {
        let ItemDefFunction { span: _, vis: _, id: _, params: _, ret_ty: _, body } = funct_ast;
        let &FunctionSignatureInfo { scope_inner, ref ret_ty } = self.compiled.function_info.get(&func_item)
            .expect("signature and info should be resolved by now");
        let ret_ty = ret_ty.clone();

        // TODO check return type (and control flow)
        // TODO check control flow
        let _ = ret_ty;

        self.visit_function_block(func_item, scope_inner, body)?;

        Ok(FunctionChecked {})
    }

    fn visit_function_block(
        &mut self,
        func_item: Item,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> ResolveResult<()> {
        let diag = self.diag;
        let scope = self.compiled.scopes.new_child(parent_scope, block.span, Visibility::Private);

        for statement in &block.statements {
            match &statement.inner {
                BlockStatementKind::VariableDeclaration(decl) => {
                    let VariableDeclaration { span, mutable, id, ty, init } = decl;
                    let span = *span;
                    let mutable = *mutable;

                    // evaluate
                    let ty_eval = match ty {
                        Some(ty) => self.eval_expression_as_ty(scope, ty)?,
                        None => Type::Error(diag.report_todo(span, "variable without type")),
                    };
                    let init_eval = match init {
                        Some(init) => self.eval_expression_as_value(scope, init)?,
                        None => Value::Error(diag.report_todo(span, "variable without init")),
                    };

                    // type check
                    if let (Some(ty), Some(init)) = (ty, init) {
                        let _: Result<(), ErrorGuaranteed> = self.check_type_contains(ty.span, init.span, &ty_eval, &init_eval);
                    }

                    // declare
                    let info = VariableInfo { defining_item: func_item, defining_id: id.clone(), ty: ty_eval.clone(), mutable };
                    let variable = self.compiled.variables.push(info);
                    let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(variable))));
                    self.compiled[scope].maybe_declare(diag, id, entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(assignment) => {
                    diag.report_todo(assignment.span, "assignment in function body");
                }
                BlockStatementKind::Expression(expression) => {
                    diag.report_todo(expression.span, "expression in function body");
                }
            }
        }

        Ok(())
    }
}
