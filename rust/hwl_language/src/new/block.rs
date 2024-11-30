use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::CompileState;
use crate::new::expression::ExpressionContext;
use crate::new::ir::{IrBlock, IrVariable};
use crate::new::misc::DomainSignal;
use crate::new::value::{ExpressionValue, NamedValue};
use crate::syntax::ast::{Assignment, Block, BlockStatement, BlockStatementKind, DomainKind, Spanned};
use crate::throw;

pub struct IrContext;

impl ExpressionContext for IrContext {
    type T = IrVariable;

    fn eval_scoped(self, s: &CompileState, n: Spanned<NamedValue>) -> Result<ExpressionValue<Self::T>, ErrorGuaranteed> {
        Err(s.diags.report_todo(n.span, "eval scoped value in IrContext"))
    }
}

impl CompileState<'_> {
    pub fn elaborate_ir_block(&mut self, parent_scope: Scope, domain: DomainKind<DomainSignal>, block: &Block<BlockStatement>) -> Result<IrBlock, ErrorGuaranteed> {
        let Block { span: _, statements } = block;

        let scope = self.scopes.new_child(parent_scope, block.span, Visibility::Private);
        let mut statements_ir = vec![];

        for stmt in statements {
            match &stmt.inner {
                BlockStatementKind::ConstDeclaration(_) => throw!(self.diags.report_todo(stmt.span, "statement kind ConstDeclaration")),
                BlockStatementKind::VariableDeclaration(_) => throw!(self.diags.report_todo(stmt.span, "statement kind VariableDeclaration")),
                BlockStatementKind::Assignment(stmt) => {
                    let Assignment { span: _, op, target, value } = stmt;

                    if op.inner.is_some() {
                        throw!(self.diags.report_todo(stmt.span, "compound assignment"));
                    }

                    throw!(self.diags.report_todo(stmt.span, "assignment"));

                    // TODO how to implement compile-time variables?
                    //   * always emit all reads/writes (if possible and using hardware types)
                    //   * if not possible; record that this variable is compile-time only
                    //   * keep a shadow map of variables to compile-time values,
                    //       merge at the end of blocks that depend on runtime
                    // let target = self.eval_expression_as_assign_target(scope, target);
                    // let value = self.eval_expression::<IrContext>(scope, value);

                    // TODO check that we can read from valid domain
                    // TODO check that we can write to target domain
                    // TODO record write
                }
                BlockStatementKind::Expression(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Expression")),
                BlockStatementKind::Block(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Block")),
                BlockStatementKind::If(_) => throw!(self.diags.report_todo(stmt.span, "statement kind If")),
                BlockStatementKind::While(_) => throw!(self.diags.report_todo(stmt.span, "statement kind While")),
                BlockStatementKind::For(_) => throw!(self.diags.report_todo(stmt.span, "statement kind For")),
                BlockStatementKind::Return(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Return")),
                BlockStatementKind::Break(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Break")),
                BlockStatementKind::Continue => throw!(self.diags.report_todo(stmt.span, "statement kind Continue")),
            };
        }

        let result = IrBlock {
            statements: statements_ir,
        };
        Ok(result)
    }
}
