use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::CompileState;
use crate::new::misc::{DomainSignal, ScopedEntry, TypeOrValue, TypeOrValueNoError};
use crate::new::types::{IntRange, Type};
use crate::new::value::{CompileValue, ExpressionValue, ScopedValue};
use crate::syntax::ast;
use crate::syntax::ast::{DomainKind, Expression, ExpressionKind, IntPattern, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use itertools::Itertools;
use num_bigint::BigInt;

impl CompileState<'_> {
    pub fn eval_expression_as_ty_or_value(&mut self, scope: Scope, expr: &Expression) -> TypeOrValue<ExpressionValue> {
        match &expr.inner {
            ExpressionKind::Dummy =>
                self.diags.report_todo(expr.span, "expr kind Dummy").into(),
            ExpressionKind::Wrapped(inner) =>
                self.eval_expression_as_ty_or_value(scope, inner),
            ExpressionKind::Id(id) => {
                match self.scopes[scope].find(&self.scopes, self.diags, id, Visibility::Private) {
                    Ok(found) => match found.value {
                        &ScopedEntry::Item(item) =>
                            self.eval_item_as_ty_or_value(item)
                                .clone()
                                .map_value(ExpressionValue::Compile),
                        ScopedEntry::Direct(direct) => match direct {
                            TypeOrValue::Type(ty) => TypeOrValue::Type(ty.clone()),
                            TypeOrValue::Value(v) => match v {
                                ScopedValue::Compile(v) => TypeOrValue::Value(ExpressionValue::Compile(v.clone())),
                                &ScopedValue::Port(v) => TypeOrValue::Value(ExpressionValue::Port(v)),
                                &ScopedValue::Wire(v) => TypeOrValue::Value(ExpressionValue::Wire(v)),
                                &ScopedValue::Register(v) => TypeOrValue::Value(ExpressionValue::Register(v)),
                                &ScopedValue::Variable(v) => TypeOrValue::Value(ExpressionValue::Variable(v)),
                            },
                            &TypeOrValue::Error(e) => TypeOrValue::Error(e),
                        },
                    },
                    Err(e) => TypeOrValue::Error(e),
                }
            }

            ExpressionKind::TypeFunc(_, _) => self.diags.report_todo(expr.span, "expr kind TypeFunc").into(),
            ExpressionKind::IntPattern(ref pattern) =>
                match pattern {
                    IntPattern::Hex(_) =>
                        TypeOrValue::Error(self.diags.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        TypeOrValue::Error(self.diags.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(pattern_raw) => {
                        let pattern_clean = pattern_raw.replace("_", "");
                        match pattern_clean.parse::<BigInt>() {
                            Ok(value) => TypeOrValue::Value(ExpressionValue::Compile(CompileValue::Int(value))),
                            Err(_) => self.diags.report_internal_error(expr.span, "failed to parse int-pattern").into(),
                        }
                    }
                }
            &ExpressionKind::BoolLiteral(literal) =>
                TypeOrValue::Value(ExpressionValue::Compile(CompileValue::Bool(literal))),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) =>
                TypeOrValue::Value(ExpressionValue::Compile(CompileValue::String(literal.clone()))),

            ExpressionKind::ArrayLiteral(_) => self.diags.report_todo(expr.span, "expr kind ArrayLiteral").into(),
            ExpressionKind::TupleLiteral(_) => self.diags.report_todo(expr.span, "expr kind TupleLiteral").into(),
            ExpressionKind::StructLiteral(_) => self.diags.report_todo(expr.span, "expr kind StructLiteral").into(),
            ExpressionKind::RangeLiteral(literal) => {
                let &ast::RangeLiteral { end_inclusive, ref start, ref end } = literal;

                let start = start.as_ref().map(|start| self.eval_expression_as_value_compile(scope, start));
                let end = end.as_ref().map(|end| self.eval_expression_as_value_compile(scope, end));

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
                        return TypeOrValue::Value(ExpressionValue::Error(e));
                    }
                    Err(e) => return TypeOrValue::Value(ExpressionValue::Error(e)),
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
                        return TypeOrValue::Value(ExpressionValue::Error(e));
                    }
                    Err(e) => return TypeOrValue::Value(ExpressionValue::Error(e)),
                };

                let end_inc = if end_inclusive {
                    match end {
                        Some(end) => Some(end),
                        None => {
                            let e = self.diags.report_internal_error(expr.span, "inclusive range must have end");
                            return TypeOrValue::Value(ExpressionValue::Error(e));
                        }
                    }
                } else {
                    end.map(|end| end - 1)
                };

                let range = IntRange { start_inc, end_inc };
                TypeOrValue::Value(ExpressionValue::Compile(CompileValue::IntRange(range)))
            },
            ExpressionKind::UnaryOp(_, _) => self.diags.report_todo(expr.span, "expr kind UnaryOp").into(),
            ExpressionKind::BinaryOp(_, _, _) => self.diags.report_todo(expr.span, "expr kind BinaryOp").into(),
            ExpressionKind::TernarySelect(_, _, _) => self.diags.report_todo(expr.span, "expr kind TernarySelect").into(),
            ExpressionKind::ArrayIndex(_, _) => self.diags.report_todo(expr.span, "expr kind ArrayIndex").into(),
            ExpressionKind::DotIdIndex(_, _) => self.diags.report_todo(expr.span, "expr kind DotIdIndex").into(),
            ExpressionKind::DotIntIndex(_, _) => self.diags.report_todo(expr.span, "expr kind DotIntIndex").into(),
            ExpressionKind::Call(target, args) => {
                let target = self.eval_expression_as_value(scope, target);
                let args = args.map_inner(|arg| self.eval_expression_as_ty_or_value(scope, arg));

                let target = match target {
                    ExpressionValue::Error(e) => return TypeOrValue::Error(e),
                    ExpressionValue::Compile(c) => {
                        match c {
                            CompileValue::Function(f) => f,
                            _ => {
                                let e = self.diags.report_simple(
                                    "call target must be function",
                                    expr.span,
                                    format!("got `{}`", c.to_diagnostic_string()),
                                );
                                return TypeOrValue::Error(e);
                            }
                        }
                    }
                    _ => {
                        let e = self.diags.report_simple(
                            "call target must be compile-time constant function",
                            expr.span,
                            "got runtime value",
                        );
                        return TypeOrValue::Error(e);
                    }
                };

                // TODO implement compile-time function calling for now
                // TODO in the future, allow runtime expressions in call args
                let _ = (target, args);
                self.diags.report_todo(expr.span, "expr kind Call").into()
            }
            ExpressionKind::Builtin(ref args) => self.eval_builtin(scope, expr.span, args),
        }
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin(&mut self, scope: Scope, expr_span: Span, args: &Spanned<Vec<Expression>>) -> TypeOrValue<ExpressionValue> {
        let mut args_eval = Ok(vec![]);

        // evaluate args
        for arg in &args.inner {
            let eval = match self.eval_expression_as_ty_or_value(scope, arg) {
                TypeOrValue::Type(ty) => TypeOrValue::Type(ty),
                TypeOrValue::Value(v) => {
                    let err_value = |s| {
                        self.diags.report_simple("__builtin arguments must be compile-time constants", arg.span, format!("got {}", s))
                    };

                    match v {
                        ExpressionValue::Error(e) => TypeOrValue::Error(e),
                        ExpressionValue::Compile(v) => TypeOrValue::Value(v),
                        ExpressionValue::Port(_) => err_value("port").into(),
                        ExpressionValue::Wire(_) => err_value("wire").into(),
                        ExpressionValue::Register(_) => err_value("register").into(),
                        ExpressionValue::Variable(_) => err_value("variable").into(),
                        ExpressionValue::RuntimeExpression { .. } => err_value("runtime expression").into(),
                    }
                }
                TypeOrValue::Error(e) => TypeOrValue::Error(e),
            };

            match (&mut args_eval, eval) {
                (Ok(args_eval), TypeOrValue::Type(ty)) => args_eval.push(TypeOrValueNoError::Type(ty)),
                (Ok(args_eval), TypeOrValue::Value(v)) => args_eval.push(TypeOrValueNoError::Value(v)),
                (&mut Err(e), _) | (_, TypeOrValue::Error(e)) => args_eval = Err(e),
            }
        }
        let args_eval = match args_eval {
            Ok(args_eval) => args_eval,
            Err(e) => return TypeOrValue::Error(e),
        };

        if let (Some(TypeOrValueNoError::Value(CompileValue::String(a0))), Some(TypeOrValueNoError::Value(CompileValue::String(a1)))) = (args_eval.get(0), args_eval.get(1)) {
            let rest = &args_eval[2..];
            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return TypeOrValue::Type(Type::Any),
                ("type", "bool", []) => return TypeOrValue::Type(Type::Bool),
                ("type", "int_range", [range]) => {
                    if let TypeOrValueNoError::Value(CompileValue::IntRange(range)) = range {
                        return TypeOrValue::Type(Type::Int(range.clone()));
                    }
                }

                ("value", "undefined", []) => return TypeOrValue::Value(ExpressionValue::Compile(CompileValue::Undefined)),

                // fallthrough into err
                _ => {}
            }
        }

        let diag = Diagnostic::new("invalid __builtin args")
            .snippet(expr_span)
            .add_error(args.span, "invalid args")
            .finish()
            .finish();
        TypeOrValue::Error(self.diags.report(diag))
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> Type {
        match self.eval_expression_as_ty_or_value(scope, expr) {
            TypeOrValue::Type(ty) => ty,
            TypeOrValue::Value(_) => {
                let e = self.diags.report_simple("expected type, got value", expr.span, "got value");
                Type::Error(e)
            }
            TypeOrValue::Error(e) => Type::Error(e),
        }
    }

    pub fn eval_expression_as_assign_target(&mut self, scope: Scope, expr: &Expression) -> () {
        let _ = (scope, expr);
        todo!()
    }

    pub fn eval_expression_as_value(&mut self, scope: Scope, expr: &Expression) -> ExpressionValue {
        let _ = (scope, expr);
        let eval = self.eval_expression_as_ty_or_value(scope, expr);

        match eval {
            TypeOrValue::Type(_) => {
                ExpressionValue::Error(self.diags.report_simple("expected value, got type", expr.span, "got type"))
            }
            TypeOrValue::Value(v) => v,
            TypeOrValue::Error(e) => e.into(),
        }
    }

    pub fn eval_expression_as_value_compile(&mut self, scope: Scope, expr: &Expression) -> Result<CompileValue, ErrorGuaranteed> {
        let eval = self.eval_expression_as_value(scope, expr);
        match eval {
            ExpressionValue::Error(e) => Err(e),
            ExpressionValue::Compile(c) => Ok(c),
            _ => Err(self.diags.report_simple("expected compile-time constant", expr.span, "got runtime expression")),
        }
    }

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