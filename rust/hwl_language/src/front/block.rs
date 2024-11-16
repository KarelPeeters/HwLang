use crate::data::compiled::VariableInfo;
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::module_body::{LowerBlock, LowerIfStatement, LowerStatement};
use crate::front::checking::DomainUserControlled;
use crate::front::common::{ContextDomain, ExpressionContext, ScopeValue, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::module::MaybeDriverCollector;
use crate::front::scope::Visibility;
use crate::front::solver::Solver;
use crate::front::types::Type;
use crate::front::value::{BoundedRangeInfo, RangeInfo, Value, ValueAccess, ValueContent, ValueDomain};
use crate::syntax::ast;
use crate::syntax::ast::{Block, BlockStatement, BlockStatementKind, ElseIfPair, ForStatement, Spanned, VariableDeclaration, WhileStatement};

#[derive(Debug, Copy, Clone)]
pub enum AccessDirection {
    Read,
    Write,
}

impl CompileState<'_, '_> {
    #[must_use]
    pub fn visit_block(
        &mut self,
        ctx_outer: &ExpressionContext,
        collector: &mut MaybeDriverCollector,
        solver: &mut Solver,
        block: &Block<BlockStatement>,
    ) -> LowerBlock {
        let scope_inner = self.compiled.scopes.new_child(ctx_outer.scope, block.span, Visibility::Private);
        let ctx_inner = &ctx_outer.with_scope(scope_inner);

        let mut lower_statements = vec![];

        for statement in &block.statements {
            let lower_statement = self.visit_statement(ctx_inner, collector, solver, statement);

            if let Some(lower_statement) = lower_statement {
                lower_statements.push(Spanned {
                    span: statement.span,
                    inner: lower_statement,
                });
            }
        }

        LowerBlock {
            statements: lower_statements,
        }
    }

    fn visit_statement(
        &mut self,
        ctx: &ExpressionContext,
        collector: &mut MaybeDriverCollector,
        solver: &mut Solver,
        statement: &BlockStatement,
    ) -> Option<LowerStatement> {
        let diags = self.diags;

        match &statement.inner {
            BlockStatementKind::ConstDeclaration(decl) => {
                self.process_and_declare_const(ctx.scope, solver, decl, Visibility::Private);
                None
            }
            BlockStatementKind::VariableDeclaration(decl) => {
                let VariableDeclaration { span: _, mutable, id, ty, init } = decl;
                let mutable = *mutable;

                let ty = ty.as_ref().map(|ty| {
                    let inner = self.eval_expression_as_ty(ctx.scope, solver, ty);
                    Spanned { span: ty.span, inner }
                });
                let init_raw = self.eval_expression_as_value(ctx, solver, init);

                // check or infer type
                let (ty, init) = match ty {
                    None => (init_raw.ty_default.clone(), init_raw),
                    Some(ty) => {
                        match self.require_type_contains_value(Some(ty.span), init_raw.origin, &ty.inner, &init_raw) {
                            Ok(()) => (ty.inner, init_raw),
                            Err(e) => (ty.inner, Value::error(e, init_raw.origin)),
                        }
                    }
                };

                // check domain
                let domain = match ctx.domain {
                    ContextDomain::Specific(domain) => {
                        let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
                            decl.span,
                            &domain.inner,
                            init.origin,
                            &init.domain,
                            DomainUserControlled::Source,
                            "variable initializer must be assignable to context domain",
                        );
                        domain.inner.clone()
                    }
                    ContextDomain::Passthrough => {
                        // TODO infer the domain form _all_ assigned values, not just the initial one
                        //   (this is a lattice fixedpoint problem)
                        init.domain.clone()
                    }
                };

                // declare
                let info = VariableInfo {
                    defining_id: id.clone(),
                    ty: ty.clone(),
                    mutable,
                    domain: domain.clone(),
                };
                let variable = self.compiled.variables.push(info);
                let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(ScopeValue::Variable(variable))));
                self.compiled[ctx.scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);

                None
            }
            BlockStatementKind::Assignment(assignment) => {
                let &ast::Assignment { span: _, op, ref target, ref value } = assignment;

                let target_eval = self.eval_expression_as_value(ctx, solver, target);
                let value_eval = self.eval_expression_as_value(ctx, solver, value);

                let mut any_err = Ok(());

                // check type
                any_err = any_err.and(self.require_type_contains_value(Some(target.span), value.span, &target_eval.ty_default, &value_eval));

                // check target writable
                let read_err = match target_eval.access {
                    ValueAccess::ReadOnly => {
                        Err(diags.report_simple("invalid assignment: target is readonly", target.span, "expected writable expression, is readonly"))
                    }
                    ValueAccess::WriteOnlyPort(port) => {
                        collector.report_write_output_port(diags, port, target.span);
                        Ok(())
                    }
                    ValueAccess::WriteReadWire(wire) => {
                        collector.report_write_wire(diags, wire, target.span);
                        Ok(())
                    }
                    ValueAccess::WriteReadRegister(reg) => {
                        collector.report_write_reg(diags, reg, target.span);
                        Ok(())
                    }
                    ValueAccess::WriteReadVariable(_) => Ok(()),
                    ValueAccess::Error(e) => Err(e),
                };
                any_err = any_err.and(read_err);

                // check value readable
                let write_err = match value_eval.access {
                    ValueAccess::WriteOnlyPort(_) => {
                        Err(diags.report_todo(value.span, "invalid assignment: read from output port"))
                    }
                    ValueAccess::ReadOnly => Ok(()),
                    ValueAccess::WriteReadWire(_) => Ok(()),
                    ValueAccess::WriteReadRegister(_) => Ok(()),
                    ValueAccess::WriteReadVariable(_) => Ok(()),
                    ValueAccess::Error(e) => Err(e),
                };
                any_err = any_err.and(write_err);

                // check domain
                match ctx.domain {
                    ContextDomain::Passthrough => {
                        // target/value need to be compatible, without additional constraints
                        any_err = any_err.and(self.check_domain_crossing(
                            target.span,
                            &target_eval.domain,
                            value.span,
                            &value_eval.domain,
                            DomainUserControlled::Both,
                            "assignment target domain must be assignable from value domain",
                        ))
                    }
                    ContextDomain::Specific(domain) => {
                        // target/domain and domain/value both need to be compatible
                        any_err = any_err.and(self.check_domain_crossing(
                            target.span,
                            &target_eval.domain,
                            domain.span,
                            domain.inner,
                            DomainUserControlled::Target,
                            "assignment target domain must be assignable from context domain",
                        ));
                        any_err = any_err.and(self.check_domain_crossing(
                            domain.span,
                            domain.inner,
                            value.span,
                            &value_eval.domain,
                            DomainUserControlled::Source,
                            "assignment context domain must be assignable from value domain",
                        ));
                    }
                };

                // operator assignments are not implemented yet
                if op.inner.is_some() {
                    any_err = Err(diags.report_todo(assignment.span, "assignment with operator"));
                }

                // result
                let lower_statement = match any_err {
                    Ok(()) => {
                        LowerStatement::Assignment { target, value }
                    }
                    Err(e) => LowerStatement::Error(e),
                };
                Some(lower_statement)
            }
            BlockStatementKind::Expression(expression) => {
                let value = self.eval_expression_as_value(ctx, solver, expression);
                Some(LowerStatement::Expression(Spanned { span: expression.span, inner: value }))
            }
            BlockStatementKind::Block(stmt) => {
                let block = self.visit_block(ctx, collector, solver, stmt);
                Some(LowerStatement::Block(block))
            }
            BlockStatementKind::If(ref stmt) => {
                let ast::IfStatement { cond, then_block, else_if_pairs, else_block } = stmt;

                // evaluate conditions and blocks
                let cond_eval = self.eval_expression_as_value(ctx, solver, cond);
                let _: Result<_, ErrorGuaranteed> = self.require_type_contains_value(None, cond.span, &Type::Boolean, &cond_eval);

                let lower_then_block = self.visit_block(ctx, collector, solver, then_block);
                let mut pairs = vec![];

                for pair in else_if_pairs {
                    let ElseIfPair { span: _, cond, block } = pair;

                    // TODO check domain of conditional? there should be a more general system to do this
                    let pair_cond_eval = self.eval_expression_as_value(ctx, solver, cond);
                    let _: Result<_, ErrorGuaranteed> = self.require_type_contains_value(None, cond.span, &Type::Boolean, &pair_cond_eval);

                    let pair_block = self.visit_block(ctx, collector, solver, block);

                    pairs.push(Spanned { span: pair.span, inner: (Spanned { span: cond.span, inner: pair_cond_eval }, pair_block) });
                }

                let else_block = else_block.as_ref().map(|else_block| {
                    self.visit_block(ctx, collector, solver, else_block)
                });

                // construct statement, in reverse order
                let mut next_else = else_block;
                for pair in pairs.into_iter().rev() {
                    let (c, b) = pair.inner;
                    next_else = Some(LowerBlock {
                        statements: vec![
                            Spanned {
                                span: pair.span,
                                inner: LowerStatement::If(LowerIfStatement {
                                    condition: c,
                                    then_block: b,
                                    else_block: next_else,
                                })
                            }
                        ]
                    })
                }
                let lower_statement = LowerStatement::If(LowerIfStatement {
                    condition: Spanned { span: cond.span, inner: cond_eval },
                    then_block: lower_then_block,
                    else_block: next_else,
                });
                Some(lower_statement)
            }
            BlockStatementKind::While(statement_while) => {
                let WhileStatement { cond, body } = statement_while;

                let cond_eval = self.eval_expression_as_value(ctx, solver, cond);
                let _: Result<_, ErrorGuaranteed> = self.require_type_contains_value(None, cond.span, &Type::Boolean, &cond_eval);

                let lower_body = self.visit_block(ctx, collector, solver, body);

                let _ = cond_eval;
                let _ = lower_body;
                let lower_statement = LowerStatement::While;
                Some(lower_statement)
            }

            BlockStatementKind::For(ref statement_for) => {
                let ForStatement { index, index_ty, iter, body } = statement_for;

                // define index variable
                let iter_span = iter.span;
                if let Some(index_ty) = index_ty {
                    diags.report_todo(index_ty.span, "for loop index type");
                }

                let iter = self.eval_expression_as_value(ctx, solver, iter);

                // require compile-time range
                let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
                    iter_span,
                    &iter.domain,
                    iter_span,
                    &ValueDomain::CompileTime,
                    DomainUserControlled::Source,
                    "for loop range must be compile-time",
                );

                // require bounded range
                let index_range = match iter.content {
                    ValueContent::Range(RangeInfo { start_inc, end_inc }) => {
                        match (start_inc, end_inc) {
                            (Some(start_inc), Some(end_inc)) => {
                                BoundedRangeInfo { start_inc: start_inc.content, end_inc: end_inc.content }
                            }
                            (_, _) => {
                                let e = diags.report_simple("for loop over unbounded range", iter_span, "expected bounded range");
                                let e = solver.error_int(e);
                                BoundedRangeInfo { start_inc: e, end_inc: e }
                            }
                        }
                    }
                    _ => {
                        let iter_ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &iter.ty_default);
                        let e = diags.report_simple("for loop over non-range target", iter_span, format!("expected range, actual type {}", iter_ty_str));
                        let e = solver.error_int(e);
                        BoundedRangeInfo { start_inc: e, end_inc: e }
                    }
                };

                let index_var = self.compiled.variables.push(VariableInfo {
                    defining_id: index.clone(),
                    ty: Type::Integer(index_range.into_general_range()),
                    mutable: false,
                    domain: ValueDomain::CompileTime,
                });
                let index_value = ScopeValue::Variable(index_var);

                let scope_index = self.compiled.scopes.new_child(ctx.scope, body.span, Visibility::Private);
                let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(index_value)));
                self.compiled[scope_index].maybe_declare(diags, index.as_ref(), entry, Visibility::Private);

                // typecheck body
                let body = self.visit_block(&ctx.with_scope(scope_index), collector, solver, &body);

                let _ = index_range;
                let _ = body;
                let lower_statement = LowerStatement::For;
                Some(lower_statement)
            }
            BlockStatementKind::Return(ref ret_value) => {
                let ret_value = ret_value.as_ref()
                    .map(|v| Spanned { span: v.span, inner: self.eval_expression_as_value(ctx, solver, v) });

                let lower_statement = match ctx.function_return_ty {
                    Some(ret_ty) => {
                        match (ret_value, ret_ty.inner) {
                            (None, Type::Unit | Type::Error(_)) => {
                                // accept
                                LowerStatement::Return(None)
                            }
                            (None, _) => {
                                let diag = Diagnostic::new("missing return value")
                                    .add_info(ret_ty.span, "function return type defined here")
                                    .add_error(statement.span, "missing return value here")
                                    .finish();
                                let e = diags.report(diag);
                                LowerStatement::Return(Some(Value::error(e, statement.span)))
                            }
                            (Some(ret_value), ret_ty_inner) => {
                                match self.require_type_contains_value(Some(ret_ty.span), ret_value.span, ret_ty_inner, &ret_value.inner) {
                                    Ok(()) => LowerStatement::Return(Some(ret_value.inner)),
                                    Err(e) => LowerStatement::Return(Some(Value::error(e, ret_value.span))),
                                }
                            }
                        }
                    }
                    None => {
                        let e = diags.report_simple("return outside function body", statement.span, "return");
                        LowerStatement::Error(e)
                    }
                };
                Some(lower_statement)
            }
            BlockStatementKind::Break(_) => {
                let e = diags.report_todo(statement.span, "break statement");
                Some(LowerStatement::Error(e))
            }
            BlockStatementKind::Continue => {
                let e = diags.report_todo(statement.span, "continue statement");
                Some(LowerStatement::Error(e))
            }
        }
    }

    // TODO make this a parameter of evaluate_expression instead?
    // // TODO this could be much faster with proper LRValue-style tracking
    // // TODO most of this is either covered by the `readable` field on value,
    // //   the checking and collecting should move to evaluate_expression 
    // pub fn check_value_usable_as_direction(&self, ctx: &ExpressionContext, collector: &mut MaybeDriverCollector, value_span: Span, value: &Value, dir: AccessDirection) -> Result<(), ErrorGuaranteed> {
    //     let diags = self.diags;
    // 
    //     const READ_LABEL: &str = "reading here";
    //     const WRITE_LABEL: &str = "writing here";
    // 
    //     let if_write_simple_error = |kind: &str| {
    //         match dir {
    //             AccessDirection::Read => Ok(()),
    //             AccessDirection::Write => {
    //                 let e = diags.report_simple(
    //                     format!("invalid write: {kind} is never writable"),
    //                     value_span,
    //                     WRITE_LABEL,
    //                 );
    //                 Err(e)
    //             }
    //         }
    //     };
    //     let if_write_simple_error_with_location = |kind: &str, def_span: Span| {
    //         match dir {
    //             AccessDirection::Read => Ok(()),
    //             AccessDirection::Write => {
    //                 let diag = Diagnostic::new(format!("invalid write: {kind} is never writable"))
    //                     .add_error(value_span, WRITE_LABEL)
    //                     .add_info(def_span, format!("{kind} declared here"))
    //                     .finish();
    //                 Err(diags.report(diag))
    //             }
    //         }
    //     };
    // 
    //     match value {
    //         // propagate error
    //         &Value::Error(e) => Err(e),
    // 
    //         // maybe assignable, depending on the details and context
    //         &Value::ModulePort(port) => {
    //             match (dir, self.compiled[port].direction) {
    //                 (AccessDirection::Read, PortDirection::Input) => Ok(()),
    //                 (AccessDirection::Write, PortDirection::Output) => {
    //                     collector.report_write_output_port(diags, port, value_span);
    //                     Ok(())
    //                 }
    //                 (AccessDirection::Read, PortDirection::Output) =>
    //                     Err(diags.report_simple("invalid read: cannot read from output port", value_span, READ_LABEL)),
    //                 (AccessDirection::Write, PortDirection::Input) =>
    //                     Err(diags.report_simple("invalid write: cannot write to input port", value_span, WRITE_LABEL)),
    //             }
    //         }
    //         &Value::Wire(wire) => {
    //             match dir {
    //                 AccessDirection::Read => Ok(()),
    //                 AccessDirection::Write => {
    //                     // wires already having an declaration value is reported as an error higher up,
    //                     //   for consistency with ports, regs and non-initialized wires
    //                     collector.report_write_wire(diags, wire, value_span);
    //                     Ok(())
    //                 }
    //             }
    //         }
    //         &Value::Register(reg) => {
    //             match dir {
    //                 AccessDirection::Read => Ok(()),
    //                 AccessDirection::Write => {
    //                     collector.report_write_reg(diags, reg, value_span);
    //                     Ok(())
    //                 }
    //             }
    //         }
    //         &Value::Variable(var) => {
    //             match dir {
    //                 AccessDirection::Read => Ok(()),
    //                 AccessDirection::Write => {
    //                     let info = &self.compiled.variables[var];
    //                     if info.mutable {
    //                         Ok(())
    //                     } else {
    //                         let diag = Diagnostic::new("invalid write: variable is not mutable")
    //                             .add_error(value_span, WRITE_LABEL)
    //                             .add_info(info.defining_id.span(), "variable declared here")
    //                             .finish();
    //                         Err(diags.report(diag))
    //                     }
    //                 }
    //             }
    //         }
    // 
    //         // only readable
    //         Value::Never => if_write_simple_error("_never_ value"),
    //         Value::Unit => if_write_simple_error("unit value"),
    //         Value::Undefined => if_write_simple_error("undefined value"),
    //         Value::BoolConstant(_) => if_write_simple_error("boolean constant"),
    //         Value::IntConstant(_) => if_write_simple_error("integer constant value"),
    //         Value::StringConstant(_) => if_write_simple_error("string constant value"),
    //         Value::Module(_) => if_write_simple_error("module value"),
    // 
    //         // TODO function call args should be checked for read/write
    //         // TODO re-implement all of this as a LRValue system, that checks for read/write
    //         //   once it's converted to the underlying value
    //         Value::FunctionReturn(_) => if_write_simple_error("function return value"),
    // 
    //         &Value::GenericParameter(param) =>
    //             if_write_simple_error_with_location("generic parameter", self.compiled[param].defining_id.span),
    //         &Value::Constant(cst) =>
    //             if_write_simple_error_with_location("constant", self.compiled[cst].defining_id.span()),
    // 
    //         // recursively check sub-expressions for read-ability
    //         Value::Range(info) => {
    //             let RangeInfo { start_inc, end_inc } = info;
    // 
    //             let mut r = Ok(());
    //             if let Some(start) = start_inc {
    //                 r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, start, AccessDirection::Read));
    //             }
    //             if let Some(end) = end_inc {
    //                 r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, end, AccessDirection::Read));
    //             }
    // 
    //             r = r.and(if_write_simple_error("range expression"));
    //             r
    //         }
    //         Value::Binary(_, left, right) => {
    //             let mut r = self.check_value_usable_as_direction(ctx, collector, value_span, left, AccessDirection::Read);
    //             r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, right, AccessDirection::Read));
    //             r = r.and(if_write_simple_error("binary expression"));
    //             r
    //         }
    //         Value::UnaryNot(x) => {
    //             let mut r = self.check_value_usable_as_direction(ctx, collector, value_span, x, AccessDirection::Read);
    //             r = r.and(if_write_simple_error("unary not expression"));
    //             r
    //         }
    //         Value::ArrayAccess { result_ty: _, base, indices } => {
    //             let mut r = self.check_value_usable_as_direction(ctx, collector, value_span, base, dir);
    //             for index in indices {
    //                 match index {
    //                     &ArrayAccessIndex::Error(e) =>
    //                         r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, &Value::Error(e), AccessDirection::Read)),
    //                     ArrayAccessIndex::Single(index) =>
    //                         r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, index, AccessDirection::Read)),
    //                     ArrayAccessIndex::Range(BoundedRangeInfo { start_inc, end_inc }) => {
    //                         r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, start_inc, AccessDirection::Read));
    //                         r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, end_inc, AccessDirection::Read));
    //                     }
    //                 }
    //             }
    //             r
    //         }
    //         Value::ArrayLiteral { result_ty: _, operands } => {
    //             let mut r = Ok(());
    //             for operand in operands {
    //                 r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, &operand.value, AccessDirection::Read));
    //             }
    //             r = r.and(if_write_simple_error("array literal"));
    //             r
    //         }
    //     }
    // }
}
