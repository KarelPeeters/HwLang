use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::{CompileState, ElaborationStackEntry};
use crate::new::misc::{DomainSignal, ScopedEntry};
use crate::new::types::{IntRange, Type};
use crate::new::value::{CompileValue, ExpressionValue};
use crate::syntax::ast;
use crate::syntax::ast::{DomainKind, Expression, ExpressionKind, IntPattern, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::util::ResultDoubleExt;
use itertools::Itertools;
use num_bigint::BigInt;
use std::convert::identity;

impl CompileState<'_> {
    pub fn eval_expression(&mut self, scope: Scope, expr: &Expression) -> Result<ExpressionValue, ErrorGuaranteed> {
        match &expr.inner {
            ExpressionKind::Dummy =>
                Err(self.diags.report_todo(expr.span, "expr kind Dummy")),
            ExpressionKind::Wrapped(inner) =>
                self.eval_expression(scope, inner),
            ExpressionKind::Id(id) => {
                match self.scopes[scope].find(&self.scopes, self.diags, id, Visibility::Private)?.value {
                    &ScopedEntry::Item(item) =>
                        Ok(ExpressionValue::Compile(self.eval_item_as_ty_or_value(item)?.clone())),
                    ScopedEntry::Direct(scoped) =>
                        Ok(ExpressionValue::from_scoped(scoped.clone()))
                }
            }

            ExpressionKind::TypeFunc(_, _) => Err(self.diags.report_todo(expr.span, "expr kind TypeFunc")),
            ExpressionKind::IntPattern(ref pattern) =>
                match pattern {
                    IntPattern::Hex(_) =>
                        Err(self.diags.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        Err(self.diags.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(pattern_raw) => {
                        let pattern_clean = pattern_raw.replace("_", "");
                        match pattern_clean.parse::<BigInt>() {
                            Ok(value) => Ok(ExpressionValue::Compile(CompileValue::Int(value))),
                            Err(_) => Err(self.diags.report_internal_error(expr.span, "failed to parse int-pattern")),
                        }
                    }
                }
            &ExpressionKind::BoolLiteral(literal) =>
                Ok(ExpressionValue::Compile(CompileValue::Bool(literal))),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) =>
                Ok(ExpressionValue::Compile(CompileValue::String(literal.clone()))),

            ExpressionKind::ArrayLiteral(_) => Err(self.diags.report_todo(expr.span, "expr kind ArrayLiteral")),
            ExpressionKind::TupleLiteral(_) => Err(self.diags.report_todo(expr.span, "expr kind TupleLiteral")),
            ExpressionKind::StructLiteral(_) => Err(self.diags.report_todo(expr.span, "expr kind StructLiteral")),
            ExpressionKind::RangeLiteral(literal) => {
                let &ast::RangeLiteral { end_inclusive, ref start, ref end } = literal;

                let start = start.as_ref().map(|start| self.eval_expression_as_compile(scope, start));
                let end = end.as_ref().map(|end| self.eval_expression_as_compile(scope, end));

                // TODO reduce code duplication
                let start_inc = match start.transpose() {
                    Ok(None) => None,
                    Ok(Some(CompileValue::Int(start))) => Some(start),
                    Ok(Some(start)) => {
                        let e = self.diags.report_simple(
                            "range bound must be integer",
                            expr.span,
                            format!("got `{}`", start.to_diagnostic_string()),
                        );
                        return Err(e);
                    }
                    Err(e) => return Err(e),
                };
                let end = match end.transpose() {
                    Ok(None) => None,
                    Ok(Some(CompileValue::Int(start))) => Some(start),
                    Ok(Some(start)) => {
                        let e = self.diags.report_simple(
                            "range bound must be integer",
                            expr.span,
                            format!("got `{}`", start.to_diagnostic_string()),
                        );
                        return Err(e);
                    }
                    Err(e) => return Err(e),
                };

                let end_inc = if end_inclusive {
                    match end {
                        Some(end) => Some(end),
                        None => {
                            let e = self.diags.report_internal_error(expr.span, "inclusive range must have end");
                            return Err(e);
                        }
                    }
                } else {
                    end.map(|end| end - 1)
                };

                let range = IntRange { start_inc, end_inc };
                Ok(ExpressionValue::Compile(CompileValue::IntRange(range)))
            }
            ExpressionKind::UnaryOp(_, _) => Err(self.diags.report_todo(expr.span, "expr kind UnaryOp")),
            ExpressionKind::BinaryOp(_, _, _) => Err(self.diags.report_todo(expr.span, "expr kind BinaryOp")),
            ExpressionKind::TernarySelect(_, _, _) => Err(self.diags.report_todo(expr.span, "expr kind TernarySelect")),
            ExpressionKind::ArrayIndex(_, _) => Err(self.diags.report_todo(expr.span, "expr kind ArrayIndex")),
            ExpressionKind::DotIdIndex(_, _) => Err(self.diags.report_todo(expr.span, "expr kind DotIdIndex")),
            ExpressionKind::DotIntIndex(_, _) => Err(self.diags.report_todo(expr.span, "expr kind DotIntIndex")),
            ExpressionKind::Call(target, args) => {
                // evaluate target and args
                let target = self.eval_expression_as_compile(scope, target)?;
                // TODO allow runtime expressions in arguments
                let args = args.map_inner(|arg| self.eval_expression_as_compile(scope, arg));

                // report errors for invalid target and args
                //   (only after both have been evaluated to get all diagnostics)
                let target = match target {
                    CompileValue::Function(f) => f,
                    _ => {
                        let e = self.diags.report_simple(
                            "call target must be function",
                            expr.span,
                            format!("got `{}`", target.to_diagnostic_string()),
                        );
                        return Err(e);
                    }
                };
                let args = args.try_map_inner(identity)?;

                // actually do the call
                let entry = ElaborationStackEntry::FunctionCall(expr.span, args.clone());
                self.check_compile_loop(entry, |s| {
                    target.call_compile_time(s, args)
                        .map(ExpressionValue::Compile)
                }).flatten_err()
            }
            ExpressionKind::Builtin(ref args) => self.eval_builtin(scope, expr.span, args),
        }
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin(&mut self, scope: Scope, expr_span: Span, args: &Spanned<Vec<Expression>>) -> Result<ExpressionValue, ErrorGuaranteed> {
        // evaluate args
        let args_eval: Vec<CompileValue> = args.inner.iter()
            .map(|arg| self.eval_expression_as_compile(scope, arg))
            .try_collect()?;

        if let (Some(CompileValue::String(a0)), Some(CompileValue::String(a1))) = (args_eval.get(0), args_eval.get(1)) {
            let rest = &args_eval[2..];
            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) =>
                    return Ok(ExpressionValue::Compile(CompileValue::Type(Type::Any))),
                ("type", "bool", []) =>
                    return Ok(ExpressionValue::Compile(CompileValue::Type(Type::Bool))),
                ("type", "int_range", [range]) => {
                    if let CompileValue::IntRange(range) = range {
                        return Ok(ExpressionValue::Compile(CompileValue::Type(Type::Int(range.clone()))));
                    }
                }

                ("value", "undefined", []) =>
                    return Ok(ExpressionValue::Compile(CompileValue::Undefined)),

                // fallthrough into err
                _ => {}
            }
        }

        let diag = Diagnostic::new("invalid __builtin args")
            .snippet(expr_span)
            .add_error(args.span, "invalid args")
            .finish()
            .finish();
        Err(self.diags.report(diag))
    }

    // TODO add reason, maybe even span
    pub fn eval_expression_as_compile(&mut self, scope: Scope, expr: &Expression) -> Result<CompileValue, ErrorGuaranteed> {
        match self.eval_expression(scope, expr)? {
            ExpressionValue::Compile(c) => Ok(c),
            _ => Err(self.diags.report_simple("expected compile-time constant", expr.span, "got runtime expression")),
        }
    }

    // TODO add reason, maybe even span
    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> Result<Type, ErrorGuaranteed> {
        // TODO unify this message with the one when a normal type-check fails
        match self.eval_expression(scope, expr)? {
            ExpressionValue::Compile(CompileValue::Type(ty)) => Ok(ty),
            _ => Err(self.diags.report_simple("expected type, got value", expr.span, "got value")),
        }
    }

    // TODO add reason, maybe even span
    pub fn eval_expression_as_assign_target(&mut self, scope: Scope, expr: &Expression) -> () {
        let _ = (scope, expr);
        todo!()
    }

    // TODO add reason, maybe even span
    pub fn eval_expression_as_domain_signal(&mut self, scope: Scope, expr: &Expression) -> DomainSignal {
        let _ = (scope, expr);
        todo!()
    }

    pub fn eval_domain(&mut self, scope: Scope, domain: &DomainKind<Box<Expression>>) -> DomainKind<DomainSignal> {
        match domain {
            DomainKind::Async => DomainKind::Async,
            DomainKind::Sync(domain) => {
                DomainKind::Sync(SyncDomain {
                    clock: self.eval_expression_as_domain_signal(scope, &domain.clock),
                    reset: self.eval_expression_as_domain_signal(scope, &domain.reset),
                })
            }
        }
    }
}