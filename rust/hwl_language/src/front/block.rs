use crate::data::compiled::{VariableDomain, VariableInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::module_body::{LowerBlock, LowerIfStatement, LowerStatement};
use crate::front::checking::DomainUserControlled;
use crate::front::common::{ContextDomain, ExpressionContext, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::module::MaybeDriverCollector;
use crate::front::scope::Visibility;
use crate::front::types::{IntegerTypeInfo, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{Block, BlockStatement, BlockStatementKind, ElseIfPair, ForStatement, PortDirection, Spanned, VariableDeclaration, WhileStatement};
use crate::syntax::pos::Span;

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
        block: &Block<BlockStatement>,
    ) -> LowerBlock {
        let scope_inner = self.compiled.scopes.new_child(ctx_outer.scope, block.span, Visibility::Private);
        let ctx_inner = &ctx_outer.with_scope(scope_inner);

        let mut lower_statements = vec![];

        for statement in &block.statements {
            let lower_statement = self.visit_statement(ctx_inner, collector, statement);

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

    fn visit_statement(&mut self, ctx: &ExpressionContext, collector: &mut MaybeDriverCollector, statement: &BlockStatement) -> Option<LowerStatement> {
        let diags = self.diags;

        match &statement.inner {
            BlockStatementKind::ConstDeclaration(decl) => {
                self.process_and_declare_const(ctx.scope, decl, Visibility::Private);
                None
            }
            BlockStatementKind::VariableDeclaration(decl) => {
                let VariableDeclaration { span: _, mutable, id, ty, init } = decl;
                let mutable = *mutable;

                let ty_eval = ty.as_ref().map(|ty| {
                    let inner = self.eval_expression_as_ty(ctx.scope, ty);
                    Spanned { span: ty.span, inner }
                });
                let init_unchecked = self.eval_expression_as_value(ctx, collector, init);

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
                match ctx.domain {
                    ContextDomain::Specific(domain) => {
                        let init_domain = self.domain_of_value(init.span, &init_eval);
                        let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
                            decl.span,
                            &domain.inner,
                            init.span,
                            &init_domain,
                            DomainUserControlled::Source,
                            "variable initializer must be assignable to context domain",
                        );
                    }
                    ContextDomain::Passthrough => {}
                }

                let domain = match ctx.domain {
                    ContextDomain::Specific(domain) => VariableDomain::Known(domain.inner.clone()),
                    ContextDomain::Passthrough => VariableDomain::Unknown,
                };

                // declare
                let info = VariableInfo {
                    defining_id: id.clone(),
                    ty: ty_eval.clone(),
                    mutable,
                    domain,
                };
                let variable = self.compiled.variables.push(info);
                let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(variable))));
                self.compiled[ctx.scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);

                None
            }
            BlockStatementKind::Assignment(assignment) => {
                let &ast::Assignment { span: _, op, ref target, ref value } = assignment;

                let target_eval = self.eval_expression_as_value(ctx, collector, target);
                let value_eval = self.eval_expression_as_value(ctx, collector, value);

                let mut any_err = Ok(());

                // check type
                let target_ty = self.type_of_value(target.span, &target_eval);
                any_err = any_err.and(self.check_type_contains(Some(target.span), value.span, &target_ty, &value_eval));

                // check read_write
                any_err = any_err.and(self.check_value_usable_as_direction(ctx, collector, target.span, &target_eval, AccessDirection::Write));
                any_err = any_err.and(self.check_value_usable_as_direction(ctx, collector, value.span, &value_eval, AccessDirection::Read));

                // check domain
                let target_domain = self.domain_of_value(target.span, &target_eval);
                let value_domain = self.domain_of_value(value.span, &value_eval);

                match ctx.domain {
                    ContextDomain::Passthrough => {
                        // target/value need to be compatible, without additional constraints
                        any_err = any_err.and(self.check_domain_crossing(
                            target.span,
                            &target_domain,
                            value.span,
                            &value_domain,
                            DomainUserControlled::Both,
                            "assignment target domain must be assignable from value domain",
                        ))
                    }
                    ContextDomain::Specific(domain) => {
                        // target/domain and domain/value both need to be compatible
                        any_err = any_err.and(self.check_domain_crossing(
                            target.span,
                            &target_domain,
                            domain.span,
                            domain.inner,
                            DomainUserControlled::Target,
                            "assignment target domain must be assignable from context domain",
                        ));
                        any_err = any_err.and(self.check_domain_crossing(
                            domain.span,
                            domain.inner,
                            value.span,
                            &value_domain,
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
                        LowerStatement::Assignment {
                            target: Spanned { span: target.span, inner: target_eval },
                            value: Spanned { span: value.span, inner: value_eval },
                        }
                    }
                    Err(e) => LowerStatement::Error(e),
                };
                Some(lower_statement)
            }
            BlockStatementKind::Expression(expression) => {
                let value = self.eval_expression_as_value(ctx, collector, expression);
                Some(LowerStatement::Expression(Spanned { span: expression.span, inner: value }))
            }
            BlockStatementKind::Block(stmt) => {
                let block = self.visit_block(ctx, collector, stmt);
                Some(LowerStatement::Block(block))
            }
            BlockStatementKind::If(ref stmt) => {
                let ast::IfStatement { cond, then_block, else_if_pairs, else_block } = stmt;

                // evaluate conditions and blocks
                let cond_eval = self.eval_expression_as_value(ctx, collector, cond);
                let _: Result<_, ErrorGuaranteed> = self.check_type_contains(None, cond.span, &Type::Boolean, &cond_eval);

                let lower_then_block = self.visit_block(ctx, collector, then_block);
                let mut pairs = vec![];

                for pair in else_if_pairs {
                    let ElseIfPair { span: _, cond, block } = pair;

                    let pair_cond_eval = self.eval_expression_as_value(ctx, collector, cond);
                    let _: Result<_, ErrorGuaranteed> = self.check_type_contains(None, cond.span, &Type::Boolean, &pair_cond_eval);

                    let pair_block = self.visit_block(ctx, collector, block);

                    pairs.push(Spanned { span: pair.span, inner: (Spanned { span: cond.span, inner: pair_cond_eval }, pair_block) });
                }

                let else_block = else_block.as_ref().map(|else_block| {
                    self.visit_block(ctx, collector, else_block)
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

                let cond_eval = self.eval_expression_as_value(ctx, collector, cond);
                let _: Result<_, ErrorGuaranteed> = self.check_type_contains(None, cond.span, &Type::Boolean, &cond_eval);

                let lower_body = self.visit_block(ctx, collector, body);

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

                let iter = self.eval_expression_as_value(ctx, collector, iter);
                let iter = self.require_int_range_direct(iter_span, &iter);

                let (start, end) = match &iter {
                    &Err(e) => (Value::Error(e), Value::Error(e)),
                    Ok(info) => {
                        // avoid duplicate error if both ends are missing
                        let report_unbounded = || diags.report_simple("for loop over unbounded range", iter_span, "range");

                        let &RangeInfo { ref start_inc, ref end_inc } = info;
                        let (start, end) = match (start_inc, end_inc) {
                            (Some(start), Some(end)) => (start.as_ref().clone(), end.as_ref().clone()),
                            (Some(start), None) => (start.as_ref().clone(), Value::Error(report_unbounded())),
                            (None, Some(end)) => (Value::Error(report_unbounded()), end.as_ref().clone()),
                            (None, None) => {
                                let e = report_unbounded();
                                (Value::Error(e), Value::Error(e))
                            }
                        };
                        (start, end)
                    }
                };

                // TODO require range to be compile-time
                let index_range = Value::Range(RangeInfo {
                    start_inc: Some(Box::new(start)),
                    end_inc: Some(Box::new(end)),
                });
                let index_var = self.compiled.variables.push(VariableInfo {
                    defining_id: index.clone(),
                    ty: Type::Integer(IntegerTypeInfo { range: Box::new(index_range) }),
                    mutable: false,
                    // TODO once we require the range to be compile-time this can be compile-time too
                    domain: VariableDomain::Unknown,
                });

                let scope_index = self.compiled.scopes.new_child(ctx.scope, body.span, Visibility::Private);
                let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(index_var))));
                self.compiled[scope_index].maybe_declare(diags, index.as_ref(), entry, Visibility::Private);

                // typecheck body
                let body = self.visit_block(&ctx.with_scope(scope_index), collector, &body);

                let _ = index_range;
                let _ = body;
                let lower_statement = LowerStatement::For;
                Some(lower_statement)
            }
            BlockStatementKind::Return(ref ret_value) => {
                let ret_value = ret_value.as_ref()
                    .map(|v| Spanned { span: v.span, inner: self.eval_expression_as_value(ctx, collector, v) });

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
                                LowerStatement::Return(Some(Value::Error(e)))
                            }
                            (Some(ret_value), ret_ty_inner) => {
                                match self.check_type_contains(Some(ret_ty.span), ret_value.span, ret_ty_inner, &ret_value.inner) {
                                    Ok(()) => LowerStatement::Return(Some(ret_value.inner)),
                                    Err(e) => LowerStatement::Return(Some(Value::Error(e))),
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

    // TODO this could be much faster with proper LRValue-style tracking
    pub fn check_value_usable_as_direction(&self, ctx: &ExpressionContext, collector: &mut MaybeDriverCollector, value_span: Span, value: &Value, dir: AccessDirection) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;

        const READ_LABEL: &str = "reading here";
        const WRITE_LABEL: &str = "writing here";

        let if_write_simple_error = |kind: &str| {
            match dir {
                AccessDirection::Read => Ok(()),
                AccessDirection::Write => {
                    let e = diags.report_simple(
                        format!("invalid write: {kind} is never writable"),
                        value_span,
                        WRITE_LABEL,
                    );
                    Err(e)
                }
            }
        };
        let if_write_simple_error_with_location = |kind: &str, def_span: Span| {
            match dir {
                AccessDirection::Read => Ok(()),
                AccessDirection::Write => {
                    let diag = Diagnostic::new(format!("invalid write: {kind} is never writable"))
                        .add_error(value_span, WRITE_LABEL)
                        .add_info(def_span, format!("{kind} declared here"))
                        .finish();
                    Err(diags.report(diag))
                }
            }
        };

        match value {
            // propagate error
            &Value::Error(e) => Err(e),

            // maybe assignable, depending on the details and context
            &Value::ModulePort(port) => {
                match (dir, self.compiled[port].direction) {
                    (AccessDirection::Read, PortDirection::Input) => Ok(()),
                    (AccessDirection::Write, PortDirection::Output) => {
                        collector.report_write_output_port(diags, port, value_span);
                        Ok(())
                    }
                    (AccessDirection::Read, PortDirection::Output) =>
                        Err(diags.report_simple("invalid read: cannot read from output port", value_span, READ_LABEL)),
                    (AccessDirection::Write, PortDirection::Input) =>
                        Err(diags.report_simple("invalid write: cannot write to input port", value_span, WRITE_LABEL)),
                }
            }
            &Value::Wire(wire) => {
                match dir {
                    AccessDirection::Read => Ok(()),
                    AccessDirection::Write => {
                        // wires already having an declaration value is reported as an error higher up,
                        //   for consistency with ports, regs and non-initialized wires
                        collector.report_write_wire(diags, wire, value_span);
                        Ok(())
                    }
                }
            }
            &Value::Register(reg) => {
                match dir {
                    AccessDirection::Read => Ok(()),
                    AccessDirection::Write => {
                        collector.report_write_reg(diags, reg, value_span);
                        Ok(())
                    }
                }
            }
            &Value::Variable(var) => {
                match dir {
                    AccessDirection::Read => Ok(()),
                    AccessDirection::Write => {
                        let info = &self.compiled.variables[var];
                        if info.mutable {
                            Ok(())
                        } else {
                            let diag = Diagnostic::new("invalid write: variable is not mutable")
                                .add_error(value_span, WRITE_LABEL)
                                .add_info(info.defining_id.span(), "variable declared here")
                                .finish();
                            Err(diags.report(diag))
                        }
                    }
                }
            }

            // only readable
            Value::Never => if_write_simple_error("_never_ value"),
            Value::Unit => if_write_simple_error("unit value"),
            Value::Undefined => if_write_simple_error("undefined value"),
            Value::BoolConstant(_) => if_write_simple_error("boolean constant"),
            Value::IntConstant(_) => if_write_simple_error("integer constant value"),
            Value::StringConstant(_) => if_write_simple_error("string constant value"),
            Value::Module(_) => if_write_simple_error("module value"),

            // TODO function call args should be checked for read/write
            // TODO re-implement all of this as a LRValue system, that checks for read/write
            //   once it's converted to the underlying value
            Value::FunctionReturn(_) => if_write_simple_error("function return value"),

            &Value::GenericParameter(param) =>
                if_write_simple_error_with_location("generic parameter", self.compiled[param].defining_id.span),
            &Value::Constant(cst) =>
                if_write_simple_error_with_location("constant", self.compiled[cst].defining_id.span()),

            // recursively check sub-expressions for read-ability
            Value::Range(info) => {
                let RangeInfo { start_inc, end_inc } = info;

                let mut r = Ok(());
                if let Some(start) = start_inc {
                    r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, start, AccessDirection::Read));
                }
                if let Some(end) = end_inc {
                    r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, end, AccessDirection::Read));
                }

                r = r.and(if_write_simple_error("range expression"));
                r
            }
            Value::Binary(_, left, right) => {
                let mut r = self.check_value_usable_as_direction(ctx, collector, value_span, left, AccessDirection::Read);
                r = r.and(self.check_value_usable_as_direction(ctx, collector, value_span, right, AccessDirection::Read));
                r = r.and(if_write_simple_error("binary expression"));
                r
            }
            Value::UnaryNot(x) => {
                let mut r = self.check_value_usable_as_direction(ctx, collector, value_span, x, AccessDirection::Read);
                r = r.and(if_write_simple_error("unary not expression"));
                r
            }
        }
    }
}
