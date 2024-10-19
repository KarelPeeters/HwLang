use crate::data::compiled::VariableInfo;
use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::module_body::LowerStatement;
use crate::front::checking::DomainUserControlled;
use crate::front::common::{ExpressionContext, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::Type;
use crate::front::values::Value;
use crate::syntax::ast;
use crate::syntax::ast::{Block, BlockStatement, BlockStatementKind, Spanned, VariableDeclaration};

impl CompileState<'_, '_> {
    #[must_use]
    pub fn visit_block(
        &mut self,
        ctx: &ExpressionContext,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> Vec<LowerStatement> {
        let diags = self.diags;
        let scope = self.compiled.scopes.new_child(parent_scope, block.span, Visibility::Private);

        let mut result_statements = vec![];

        for statement in &block.statements {
            match &statement.inner {
                BlockStatementKind::ConstDeclaration(decl) => {
                    self.process_and_declare_const(scope, decl, Visibility::Private);
                }
                BlockStatementKind::VariableDeclaration(decl) => {
                    let VariableDeclaration { span: _, mutable, id, ty, init } = decl;
                    let mutable = *mutable;

                    let ty_eval = ty.as_ref().map(|ty| {
                        let inner = self.eval_expression_as_ty(scope, ty);
                        Spanned { span: ty.span, inner }
                    });
                    let init_unchecked = self.eval_expression_as_value(ctx, scope, init);

                    // check or infer type
                    let (ty_eval, init_eval) = match ty_eval {
                        None => (self.type_of_value(init.span, &init_unchecked), init_unchecked),
                        Some(ty_eval) => {
                            match self.check_type_contains(Some(ty_eval.span), init.span, &ty_eval.inner, &init_unchecked) {
                                Ok(()) => (ty_eval.inner, init_unchecked),
                                Err(e) => (Type::Error(e), Value::Error(e)),
                            }
                        }
                    };

                    // check domain
                    if let Some(domain) = ctx.domain_kind() {
                        let init_domain = self.domain_of_value(init.span, &init_eval);
                        let _: Result<(), ErrorGuaranteed> = self.check_domain_assign(
                            decl.span,
                            &domain,
                            init.span,
                            &init_domain,
                            DomainUserControlled::Source,
                            "variable initializer must be assignable to context domain",
                        );
                    }

                    // declare
                    let info = VariableInfo { defining_id: id.clone(), ty: ty_eval.clone(), mutable };
                    let variable = self.compiled.variables.push(info);
                    let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(variable))));
                    self.compiled[scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(assignment) => {
                    let &ast::Assignment { span: _, op, ref target, ref value } = assignment;

                    let target = self.eval_expression_as_value(ctx, scope, target);
                    let value = self.eval_expression_as_value(ctx, scope, value);

                    if op.inner.is_some() {
                        let e = diags.report_todo(assignment.span, "assignment with operator");
                        result_statements.push(LowerStatement::Error(e));
                        continue;
                    }

                    match (target, value) {
                        (Value::ModulePort(target), Value::ModulePort(value)) => {
                            let sync = match ctx {
                                ExpressionContext::CombinatorialBlock => None,
                                ExpressionContext::ClockedBlock(sync) => Some(sync.as_ref().map_inner(|d| *d)),
                                _ => {
                                    let e = diags.report_todo(assignment.span, "assignment outside of clocked or combinatorial block");
                                    result_statements.push(LowerStatement::Error(e));
                                    continue;
                                }
                            };

                            let stmt = match self.check_assign_port_port(sync, assignment, target, value) {
                                Ok(()) => LowerStatement::PortPortAssignment(target, value),
                                Err(e) => LowerStatement::Error(e),
                            };
                            result_statements.push(stmt);
                        }
                        (Value::Error(e), _) | (_, Value::Error(e)) => {
                            result_statements.push(LowerStatement::Error(e));
                        }
                        _ => {
                            let err = self.diags.report_todo(statement.span, "general assignment");
                            result_statements.push(LowerStatement::Error(err));
                        }
                    }
                }
                BlockStatementKind::Expression(expression) => {
                    match ctx {
                        ExpressionContext::FunctionBody { .. } => {
                            // TODO control flow reachability and return checking
                            let _ = self.eval_expression_as_value(ctx, scope, expression);
                        }
                        _ => {
                            diags.report_todo(statement.span, "expression statement outside of function body");
                        }
                    }
                }
            }
        }

        result_statements
    }
}
