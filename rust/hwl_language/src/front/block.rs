use crate::data::compiled::VariableInfo;
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::common::{ExpressionContext, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::Type;
use crate::front::values::Value;
use crate::syntax::ast::{Block, BlockStatement, BlockStatementKind, VariableDeclaration};

impl CompileState<'_, '_> {
    pub fn visit_block(
        &mut self,
        ctx: &ExpressionContext,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> () {
        let diags = self.diags;
        let scope = self.compiled.scopes.new_child(parent_scope, block.span, Visibility::Private);

        for statement in &block.statements {
            match &statement.inner {
                BlockStatementKind::VariableDeclaration(decl) => {
                    let VariableDeclaration { span, mutable, id, ty, init } = decl;
                    let span = *span;
                    let mutable = *mutable;

                    // evaluate
                    let ty_eval = match ty {
                        Some(ty) => self.eval_expression_as_ty(scope, ty),
                        None => Type::Error(diags.report_todo(span, "variable without type")),
                    };
                    let init_eval = match init {
                        Some(init) => self.eval_expression_as_value(ctx, scope, init),
                        None => Value::Error(diags.report_todo(span, "variable without init")),
                    };

                    // type check
                    if let (Some(ty), Some(init)) = (ty, init) {
                        let _: Result<(), ErrorGuaranteed> = self.check_type_contains(ty.span, init.span, &ty_eval, &init_eval);
                    }

                    // declare
                    let info = VariableInfo { defining_id: id.clone(), ty: ty_eval.clone(), mutable };
                    let variable = self.compiled.variables.push(info);
                    let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(variable))));
                    self.compiled[scope].maybe_declare(diags, id, entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(assignment) => {
                    diags.report_todo(assignment.span, "assignment in function body");
                }
                BlockStatementKind::Expression(expression) => {
                    let _ = self.eval_expression_as_value(ctx, scope, expression);
                }
            }
        }
    }
}
