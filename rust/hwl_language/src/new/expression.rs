use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::block::VariableValues;
use crate::new::compile::{CompileState, ElaborationStackEntry};
use crate::new::misc::{DomainSignal, ScopedEntry};
use crate::new::types::{HardwareType, IntRange, Type};
use crate::new::value::{AssignmentTarget, CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast;
use crate::syntax::ast::{
    DomainKind, Expression, ExpressionKind, Identifier, IntPattern, PortDirection, Spanned, SyncDomain, UnaryOp,
};
use crate::syntax::pos::Span;
use crate::util::{Never, ResultDoubleExt};
use itertools::Itertools;
use num_bigint::BigInt;
use std::convert::identity;

// TODO rethink this abstraction, and don't overcomplicate things
//   (look at use cases of eval_expression, split it up if that's simpler)
pub trait ExpressionContext {
    type T;

    fn eval_named(
        &mut self,
        s: &CompileState,
        vars: &VariableValues,
        span_use: Span,
        span_def: Span,
        named: NamedValue,
    ) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed>;
    fn bool_not(
        &mut self,
        s: &CompileState,
        span_use: Span,
        inner: Self::T,
    ) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed>;
}

pub struct CompileNamedContext<'s> {
    pub reason: &'s str,
}

impl ExpressionContext for CompileNamedContext<'_> {
    type T = Never;

    fn eval_named(
        &mut self,
        s: &CompileState,
        vars: &VariableValues,
        span_use: Span,
        span_def: Span,
        named: NamedValue,
    ) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed> {
        let build_err = |actual: &str| {
            let diag = Diagnostic::new(format!("{} must be compile-time value", self.reason))
                .add_error(span_use, format!("got `{}`", actual))
                .add_info(span_def, "defined here")
                .finish();
            s.diags.report(diag)
        };

        match named {
            NamedValue::Constant(c) => Ok(MaybeCompile::Compile(s.constants[c].value.clone())),
            NamedValue::Parameter(p) => Ok(MaybeCompile::Compile(s.parameters[p].value.clone())),
            NamedValue::Variable(v) => {
                let assigned = vars.get(s.diags, span_use, v)?;
                match &assigned.value {
                    MaybeCompile::Compile(v) => Ok(MaybeCompile::Compile(v.clone())),
                    MaybeCompile::Other(_) => {
                        let diag = Diagnostic::new(format!("{} must be compile-time value", self.reason))
                            .add_error(span_use, "got variable")
                            .add_info(
                                assigned.event.span(),
                                "variable was assigned a non-compile-time value here",
                            )
                            .finish();
                        Err(s.diags.report(diag))
                    }
                }
            }
            NamedValue::Port(_) => Err(build_err("port")),
            NamedValue::Wire(_) => Err(build_err("wire")),
            NamedValue::Register(_) => Err(build_err("register")),
        }
    }

    fn bool_not(&mut self, _: &CompileState, _: Span, _: Self::T) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed> {
        todo!()
    }
}

impl CompileState<'_> {
    pub fn eval_id(
        &mut self,
        scope: Scope,
        id: &Identifier,
    ) -> Result<Spanned<MaybeCompile<NamedValue>>, ErrorGuaranteed> {
        let found = self.scopes[scope].find(&self.scopes, self.diags, id, Visibility::Private)?;
        let def_span = found.defining_span;
        let result = match found.value {
            &ScopedEntry::Item(item) => MaybeCompile::Compile(self.eval_item_as_ty_or_value(item)?.clone()),
            &ScopedEntry::Direct(value) => MaybeCompile::Other(value),
        };
        Ok(Spanned {
            span: def_span,
            inner: result,
        })
    }

    // TODO better handling of scoped, in particular params and variables:
    //   * params should always be converted to compile-time constants
    //   * variables should be turned into compile-time constants or ir values depending on the context
    //   * these need to happen early enough that followup expression can handle them (eg.
    pub fn eval_expression<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
    ) -> Result<MaybeCompile<C::T>, ErrorGuaranteed> {
        match &expr.inner {
            ExpressionKind::Dummy => Err(self.diags.report_todo(expr.span, "expr kind Dummy")),
            ExpressionKind::Undefined => Ok(MaybeCompile::Compile(CompileValue::Undefined)),
            ExpressionKind::Wrapped(inner) => self.eval_expression(ctx, scope, vars, inner),
            ExpressionKind::Id(id) => {
                let eval = self.eval_id(scope, id)?;
                match eval.inner {
                    MaybeCompile::Compile(c) => Ok(MaybeCompile::Compile(c)),
                    MaybeCompile::Other(value) => ctx.eval_named(self, vars, expr.span, eval.span, value),
                }
            }
            ExpressionKind::TypeFunc(_, _) => Err(self.diags.report_todo(expr.span, "expr kind TypeFunc")),
            ExpressionKind::IntPattern(ref pattern) => match pattern {
                IntPattern::Hex(_) => Err(self.diags.report_todo(expr.span, "hex int-pattern expression")),
                IntPattern::Bin(_) => Err(self.diags.report_todo(expr.span, "bin int-pattern expression")),
                IntPattern::Dec(pattern_raw) => {
                    let pattern_clean = pattern_raw.replace("_", "");
                    match pattern_clean.parse::<BigInt>() {
                        Ok(value) => Ok(MaybeCompile::Compile(CompileValue::Int(value))),
                        Err(_) => Err(self
                            .diags
                            .report_internal_error(expr.span, "failed to parse int-pattern")),
                    }
                }
            },
            &ExpressionKind::BoolLiteral(literal) => Ok(MaybeCompile::Compile(CompileValue::Bool(literal))),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) => Ok(MaybeCompile::Compile(CompileValue::String(literal.clone()))),

            ExpressionKind::ArrayLiteral(_) => Err(self.diags.report_todo(expr.span, "expr kind ArrayLiteral")),
            ExpressionKind::TupleLiteral(_) => Err(self.diags.report_todo(expr.span, "expr kind TupleLiteral")),
            ExpressionKind::StructLiteral(_) => Err(self.diags.report_todo(expr.span, "expr kind StructLiteral")),
            ExpressionKind::RangeLiteral(literal) => {
                let &ast::RangeLiteral {
                    end_inclusive,
                    ref start,
                    ref end,
                } = literal;

                let start = start
                    .as_ref()
                    .map(|start| self.eval_expression_as_compile(scope, vars, start, "range start"));
                let end = end
                    .as_ref()
                    .map(|end| self.eval_expression_as_compile(scope, vars, end, "range end"));

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
                            let e = self
                                .diags
                                .report_internal_error(expr.span, "inclusive range must have end");
                            return Err(e);
                        }
                    }
                } else {
                    end.map(|end| end - 1)
                };

                let range = IntRange { start_inc, end_inc };
                Ok(MaybeCompile::Compile(CompileValue::IntRange(range)))
            }
            ExpressionKind::UnaryOp(op, inner) => {
                match op {
                    UnaryOp::Neg => Err(self.diags.report_todo(expr.span, "expr kind UnaryOp Neg")),
                    UnaryOp::Not => {
                        let inner = self.eval_expression(ctx, scope, vars, inner)?;

                        // TODO unified type checking for this MaybeCompile thing, see all usages of check_type_contains_compile_value
                        match inner {
                            MaybeCompile::Compile(c) => match c {
                                // TODO propagate or undefined?
                                CompileValue::Undefined => Ok(MaybeCompile::Compile(CompileValue::Undefined)),
                                CompileValue::Bool(b) => Ok(MaybeCompile::Compile(CompileValue::Bool(!b))),
                                _ => todo!("type-check first, then report internal error here"),
                            },
                            MaybeCompile::Other(v) => ctx.bool_not(self, expr.span, v),
                        }
                    }
                }
            }
            ExpressionKind::BinaryOp(_, _, _) => Err(self.diags.report_todo(expr.span, "expr kind BinaryOp")),
            ExpressionKind::TernarySelect(_, _, _) => Err(self.diags.report_todo(expr.span, "expr kind TernarySelect")),
            ExpressionKind::ArrayIndex(_, _) => Err(self.diags.report_todo(expr.span, "expr kind ArrayIndex")),
            ExpressionKind::DotIdIndex(_, _) => Err(self.diags.report_todo(expr.span, "expr kind DotIdIndex")),
            ExpressionKind::DotIntIndex(_, _) => Err(self.diags.report_todo(expr.span, "expr kind DotIntIndex")),
            ExpressionKind::Call(target, args) => {
                // evaluate target and args
                let target = self.eval_expression_as_compile(scope, vars, target, "call target")?;
                // TODO allow runtime expressions in arguments
                let args = args.map_inner(|arg| self.eval_expression_as_compile(scope, vars, arg, "call argument"));

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
                self.check_compile_loop(entry, |s| target.call_compile_time(s, args).map(MaybeCompile::Compile))
                    .flatten_err()
            }
            ExpressionKind::Builtin(ref args) => {
                Ok(MaybeCompile::Compile(self.eval_builtin(scope, vars, expr.span, args)?))
            }
        }
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin(
        &mut self,
        scope: Scope,
        vars: &VariableValues,
        expr_span: Span,
        args: &Spanned<Vec<Expression>>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        // evaluate args
        let args_eval: Vec<CompileValue> = args
            .inner
            .iter()
            .map(|arg| self.eval_expression_as_compile(scope, vars, arg, "builtin argument"))
            .try_collect()?;

        if let (Some(CompileValue::String(a0)), Some(CompileValue::String(a1))) = (args_eval.get(0), args_eval.get(1)) {
            let rest = &args_eval[2..];
            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return Ok(CompileValue::Type(Type::Any)),
                ("type", "bool", []) => return Ok(CompileValue::Type(Type::Bool)),
                ("type", "int_range", [range]) => {
                    if let CompileValue::IntRange(range) = range {
                        return Ok(CompileValue::Type(Type::Int(range.clone())));
                    }
                }
                // fallthrough into err
                _ => {}
            }
        }

        let diag = Diagnostic::new("invalid builtin arguments")
            .snippet(expr_span)
            .add_error(args.span, "invalid args")
            .finish()
            .finish();
        Err(self.diags.report(diag))
    }

    pub fn eval_expression_as_compile(
        &mut self,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
        reason: &str,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        match self.eval_expression(&mut CompileNamedContext { reason }, scope, vars, expr)? {
            MaybeCompile::Compile(c) => Ok(c),
            MaybeCompile::Other(never) => never.unreachable(),
        }
    }

    pub fn eval_expression_as_ty(
        &mut self,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
    ) -> Result<Type, ErrorGuaranteed> {
        // TODO unify this message with the one when a normal type-check fails
        match self.eval_expression_as_compile(scope, vars, expr, "type")? {
            CompileValue::Type(ty) => Ok(ty),
            _ => Err(self
                .diags
                .report_simple("expected type, got value", expr.span, "got value")),
        }
    }

    pub fn eval_expression_as_ty_hardware(
        &mut self,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
        reason: &str,
    ) -> Result<HardwareType, ErrorGuaranteed> {
        let ty = self.eval_expression_as_ty(scope, vars, expr)?;
        ty.as_hardware_type().ok_or_else(|| {
            self.diags.report_simple(
                format!("{} type must be representable in hardware", reason),
                expr.span,
                format!("got `{}`", ty.to_diagnostic_string()),
            )
        })
    }

    pub fn eval_expression_as_assign_target(
        &mut self,
        scope: Scope,
        expr: &Expression,
    ) -> Result<AssignmentTarget, ErrorGuaranteed> {
        let build_err = |actual: &str| {
            self.diags
                .report_simple("expected assignment target", expr.span, format!("got `{}`", actual))
        };

        match &expr.inner {
            ExpressionKind::Id(id) => match self.eval_id(scope, id)?.inner {
                MaybeCompile::Compile(_) => Err(build_err("compile-time constant")),
                MaybeCompile::Other(s) => match s {
                    NamedValue::Constant(_) => Err(build_err("constant")),
                    NamedValue::Parameter(_) => Err(build_err("parameter")),
                    NamedValue::Variable(v) => Ok(AssignmentTarget::Variable(v)),
                    NamedValue::Port(p) => {
                        let direction = self.ports[p].direction;
                        match direction.inner {
                            PortDirection::Input => Err(build_err("input port")),
                            PortDirection::Output => Ok(AssignmentTarget::Port(p)),
                        }
                    }
                    NamedValue::Wire(w) => Ok(AssignmentTarget::Wire(w)),
                    NamedValue::Register(r) => Ok(AssignmentTarget::Register(r)),
                },
            },
            ExpressionKind::ArrayIndex(_, _) => {
                Err(self.diags.report_todo(expr.span, "assignment target array index"))?
            }
            ExpressionKind::DotIdIndex(_, _) => {
                Err(self.diags.report_todo(expr.span, "assignment target dot id index"))?
            }
            ExpressionKind::DotIntIndex(_, _) => {
                Err(self.diags.report_todo(expr.span, "assignment target dot int index"))?
            }
            _ => Err(build_err("other expression")),
        }
    }

    pub fn eval_expression_as_domain_signal(
        &mut self,
        scope: Scope,
        expr: &Expression,
    ) -> Result<DomainSignal, ErrorGuaranteed> {
        // TODO expand to allow general expressions (which then probably create implicit signals)?

        let diags = self.diags;
        let build_err = |actual: &str| {
            self.diags
                .report_simple("expected domain signal", expr.span, format!("got `{}`", actual))
        };

        match &expr.inner {
            ExpressionKind::UnaryOp(UnaryOp::Not, inner) => {
                let inner = self.eval_expression_as_domain_signal(scope, inner)?;
                Ok(DomainSignal::BoolNot(Box::new(inner)))
            }
            ExpressionKind::Id(id) => {
                let value = self.eval_id(scope, id)?;
                match value.inner {
                    MaybeCompile::Compile(_) => Err(build_err("compile-time value")),
                    MaybeCompile::Other(s) => match s {
                        NamedValue::Constant(_) => Err(build_err("constant")),
                        NamedValue::Parameter(_) => Err(build_err("parameter")),
                        NamedValue::Variable(_) => Err(build_err("variable")),
                        NamedValue::Port(p) => Ok(DomainSignal::Port(p)),
                        NamedValue::Wire(w) => Ok(DomainSignal::Wire(w)),
                        NamedValue::Register(r) => Ok(DomainSignal::Register(r)),
                    },
                }
            }
            _ => Err(diags.report_simple("expected domain signal (port, wire, reg)", expr.span, "got expression")),
        }
    }

    pub fn eval_domain_sync(
        &mut self,
        scope: Scope,
        domain: &SyncDomain<Box<Expression>>,
    ) -> Result<SyncDomain<DomainSignal>, ErrorGuaranteed> {
        Ok(SyncDomain {
            clock: self.eval_expression_as_domain_signal(scope, &domain.clock)?,
            reset: self.eval_expression_as_domain_signal(scope, &domain.reset)?,
        })
    }

    pub fn eval_domain(
        &mut self,
        scope: Scope,
        domain: &DomainKind<Box<Expression>>,
    ) -> Result<DomainKind<DomainSignal>, ErrorGuaranteed> {
        match domain {
            DomainKind::Async => Ok(DomainKind::Async),
            DomainKind::Sync(domain) => self.eval_domain_sync(scope, domain).map(DomainKind::Sync),
        }
    }
}
