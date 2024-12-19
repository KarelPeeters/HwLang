use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::block::{TypedIrExpression, VariableValues};
use crate::new::check::{check_type_contains_value, TypeContainsReason};
use crate::new::compile::{CompileState, ElaborationStackEntry};
use crate::new::ir::IrExpression;
use crate::new::misc::{DomainSignal, ScopedEntry};
use crate::new::types::{HardwareType, IncRange, Type};
use crate::new::value::{AssignmentTarget, CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast;
use crate::syntax::ast::{
    DomainKind, Expression, ExpressionKind, Identifier, IntPattern, PortDirection, Spanned, SyncDomain, UnaryOp,
};
use crate::syntax::pos::Span;
use crate::util::ResultDoubleExt;
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::convert::identity;

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

    // TODO return COW to save some allocations?
    pub fn eval_expression(
        &mut self,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
    ) -> Result<Spanned<MaybeCompile<TypedIrExpression>>, ErrorGuaranteed> {
        let diags = self.diags;

        let result = match &expr.inner {
            ExpressionKind::Dummy => Err(diags.report_todo(expr.span, "expr kind Dummy")),
            ExpressionKind::Undefined => Ok(MaybeCompile::Compile(CompileValue::Undefined)),
            ExpressionKind::Type => Ok(MaybeCompile::Compile(CompileValue::Type(Type::Type))),
            ExpressionKind::Wrapped(inner) => Ok(self.eval_expression(scope, vars, inner)?.inner),
            ExpressionKind::Id(id) => {
                let eval = self.eval_id(scope, id)?;
                match eval.inner {
                    MaybeCompile::Compile(c) => Ok(MaybeCompile::Compile(c)),
                    MaybeCompile::Other(value) => match value {
                        NamedValue::Constant(cst) => Ok(MaybeCompile::Compile(self.constants[cst].value.clone())),
                        NamedValue::Parameter(param) => Ok(MaybeCompile::Compile(self.parameters[param].value.clone())),
                        NamedValue::Variable(var) => Ok(vars.get(diags, expr.span, var)?.value.clone()),
                        NamedValue::Port(port) => Ok(MaybeCompile::Other(self.ports[port].typed_ir_expr())),
                        NamedValue::Wire(wire) => Ok(MaybeCompile::Other(self.wires[wire].typed_ir_expr())),
                        NamedValue::Register(reg) => Ok(MaybeCompile::Other(self.registers[reg].typed_ir_expr())),
                    },
                }
            }
            ExpressionKind::TypeFunc(_, _) => Err(diags.report_todo(expr.span, "expr kind TypeFunc")),
            ExpressionKind::IntPattern(ref pattern) => match pattern {
                IntPattern::Hex(_) => Err(diags.report_todo(expr.span, "hex int-pattern expression")),
                IntPattern::Bin(_) => Err(diags.report_todo(expr.span, "bin int-pattern expression")),
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

            ExpressionKind::ArrayLiteral(_) => Err(diags.report_todo(expr.span, "expr kind ArrayLiteral")),
            ExpressionKind::TupleLiteral(_) => Err(diags.report_todo(expr.span, "expr kind TupleLiteral")),
            ExpressionKind::StructLiteral(_) => Err(diags.report_todo(expr.span, "expr kind StructLiteral")),
            ExpressionKind::RangeLiteral(literal) => {
                let &ast::RangeLiteral {
                    end_inclusive,
                    ref start,
                    ref end,
                } = literal;

                let start = start.as_ref().map(|start| {
                    Ok(self
                        .eval_expression_as_compile(scope, vars, start, "range start")?
                        .inner)
                });
                let end = end
                    .as_ref()
                    .map(|end| Ok(self.eval_expression_as_compile(scope, vars, end, "range end")?.inner));

                // TODO reduce code duplication
                let start_inc = match start.transpose() {
                    Ok(None) => None,
                    Ok(Some(CompileValue::Int(start))) => Some(start),
                    Ok(Some(start)) => {
                        let e = diags.report_simple(
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
                        let e = diags.report_simple(
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

                let range = IncRange { start_inc, end_inc };
                Ok(MaybeCompile::Compile(CompileValue::IntRange(range)))
            }
            ExpressionKind::UnaryOp(op, operand) => match op.inner {
                UnaryOp::Neg => Err(diags.report_todo(expr.span, "expr kind UnaryOp Neg")),
                UnaryOp::Not => {
                    let operand = self.eval_expression(scope, vars, operand)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        &Type::Bool,
                        operand.as_ref(),
                        false,
                    )?;

                    match operand.inner {
                        MaybeCompile::Compile(c) => match c {
                            CompileValue::Bool(b) => Ok(MaybeCompile::Compile(CompileValue::Bool(!b))),
                            _ => Err(diags.report_internal_error(expr.span, "expected bool for unary not")),
                        },
                        MaybeCompile::Other(v) => {
                            let result = TypedIrExpression {
                                ty: HardwareType::Bool,
                                domain: v.domain,
                                expr: IrExpression::BoolNot(Box::new(v.expr)),
                            };
                            Ok(MaybeCompile::Other(result))
                        }
                    }
                }
            },
            ExpressionKind::BinaryOp(_, _, _) => Err(diags.report_todo(expr.span, "expr kind BinaryOp")),
            ExpressionKind::TernarySelect(_, _, _) => Err(diags.report_todo(expr.span, "expr kind TernarySelect")),
            ExpressionKind::ArrayIndex(base, args) => {
                // eval base and args
                let base = self.eval_expression(scope, vars, base);
                let args = args.map_inner(|a| self.eval_expression(scope, vars, a));

                let base = base?;
                let args = args.try_map_inner(identity)?;

                // disallow named args
                let mut result_named = Ok(());
                for arg in &args.inner {
                    if let Some(name) = &arg.name {
                        result_named = Err(diags.report_todo(name.span, "named array dimensions"));
                    }
                }
                result_named?;

                // loop over all indices, folding the indexing operations
                let mut curr = base;
                for arg in args.inner {
                    let curr_span = curr.span;
                    let curr_inner = match (curr.inner, arg.value.inner) {
                        (MaybeCompile::Compile(curr), MaybeCompile::Compile(index)) => {
                            match curr {
                                // declare new array type
                                CompileValue::Type(curr) => {
                                    let dim_len = match index {
                                        CompileValue::Int(dim_len) => BigUint::try_from(dim_len).map_err(|e| {
                                            diags.report_simple(
                                                "array dimension length cannot be negative",
                                                arg.span,
                                                format!("got value `{}`", e.into_original()),
                                            )
                                        })?,
                                        _ => {
                                            return Err(diags.report_simple(
                                                "array dimension length must be an integer",
                                                arg.span,
                                                format!("got value `{}`", index.to_diagnostic_string()),
                                            ));
                                        }
                                    };

                                    let result_ty = Type::Array(Box::new(curr), dim_len);
                                    MaybeCompile::Compile(CompileValue::Type(result_ty))
                                }
                                // index into compile-time array
                                CompileValue::Array(curr) => {
                                    match index {
                                        CompileValue::Int(index) => {
                                            let valid_range = 0..curr.len();
                                            let index = index.to_usize()
                                                .filter(|index| valid_range.contains(&index))
                                                .ok_or_else(|| {
                                                    diags.report_simple(
                                                        "array index out of bounds",
                                                        arg.span,
                                                        format!("got index `{index}`, valid range for this array is `{valid_range:?}`"),
                                                    )
                                                })?;
                                            // TODO avoid clone, both of this value and of the entire array?
                                            MaybeCompile::Compile(curr[index].clone())
                                        }
                                        CompileValue::IntRange(range) => {
                                            let check_valid = |x: &Option<BigInt>, default: usize| {
                                                // for slices, the valid range is inclusive
                                                let valid_range = 0..=curr.len();
                                                match x {
                                                    None => Ok(default),
                                                    Some(x) => x.to_usize()
                                                        .filter(|index| valid_range.contains(&index))
                                                        .ok_or_else(|| {
                                                            diags.report_simple(
                                                                "array slice range out of bounds",
                                                                arg.span,
                                                                format!("got slice range `{range:?}`, valid range for this array is `{valid_range:?}`"),
                                                            )
                                                        })
                                                }
                                            };

                                            let start = check_valid(&range.start_inc, 0)?;
                                            let end = check_valid(&range.end_inc, curr.len())?;

                                            if start > end {
                                                return Err(diags.report_internal_error(
                                                    arg.span,
                                                    format!("invalid decreasing range `{range:?}`"),
                                                ));
                                            }

                                            // TODO avoid clone, both of this slice and of the entire array?
                                            MaybeCompile::Compile(CompileValue::Array(curr[start..end].to_vec()))
                                        }
                                        _ => {
                                            return Err(diags.report_simple(
                                                "array index must be an integer or an integer range",
                                                arg.span,
                                                format!("got value `{}`", index.to_diagnostic_string()),
                                            ));
                                        }
                                    }
                                }
                                _ => {
                                    return Err(diags.report_simple(
                                        "array index on invalid target",
                                        arg.span,
                                        format!("target `{}` here", curr.to_diagnostic_string()),
                                    ));
                                }
                            }
                        }
                        (curr, index) => {
                            let curr = curr.to_ir_expression(diags, curr_span)?;
                            let index = index.to_ir_expression(diags, arg.span)?;

                            let _ = (curr, index);
                            return Err(diags.report_todo(arg.span, "runtime array indexing/slicing"));
                        }
                    };

                    // TODO this is a bit of sketchy span, but maybe the best we can do
                    curr = Spanned {
                        span: curr.span.join(arg.span),
                        inner: curr_inner,
                    }
                }

                Ok(curr.inner)
            }
            ExpressionKind::DotIdIndex(_, _) => Err(diags.report_todo(expr.span, "expr kind DotIdIndex")),
            ExpressionKind::DotIntIndex(_, _) => Err(diags.report_todo(expr.span, "expr kind DotIntIndex")),
            ExpressionKind::Call(target, args) => {
                // evaluate target and args
                let target = self.eval_expression_as_compile(scope, vars, target, "call target")?;
                // TODO allow runtime expressions in arguments
                let args = args.map_inner(|arg| {
                    Ok(self
                        .eval_expression_as_compile(scope, vars, arg, "call argument")?
                        .inner)
                });

                // report errors for invalid target and args
                //   (only after both have been evaluated to get all diagnostics)
                let target = match target.inner {
                    CompileValue::Function(f) => f,
                    _ => {
                        let e = diags.report_simple(
                            "call target must be function",
                            expr.span,
                            format!("got `{}`", target.inner.to_diagnostic_string()),
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
        };

        result.map(|result| Spanned {
            span: expr.span,
            inner: result,
        })
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
            .map(|arg| Ok(self.eval_expression_as_compile(scope, vars, arg, "builtin argument")?.inner))
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
    ) -> Result<Spanned<CompileValue>, ErrorGuaranteed> {
        match self.eval_expression(scope, vars, expr)?.inner {
            MaybeCompile::Compile(c) => Ok(Spanned { span: expr.span, inner: c }),
            MaybeCompile::Other(ir_expr) => Err(self.diags.report_simple(
                format!("{reason} must be a compile-time value"),
                expr.span,
                format!("got value with domain `{}`", ir_expr.domain.to_diagnostic_string(self)),
            )),
        }
    }

    pub fn eval_expression_as_ty(
        &mut self,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
    ) -> Result<Spanned<Type>, ErrorGuaranteed> {
        // TODO unify this message with the one when a normal type-check fails
        match self.eval_expression_as_compile(scope, vars, expr, "type")?.inner {
            CompileValue::Type(ty) => Ok(Spanned { span: expr.span, inner: ty }),
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
    ) -> Result<Spanned<HardwareType>, ErrorGuaranteed> {
        let ty = self.eval_expression_as_ty(scope, vars, expr)?.inner;
        let ty_hw = ty.as_hardware_type().ok_or_else(|| {
            self.diags.report_simple(
                format!("{} type must be representable in hardware", reason),
                expr.span,
                format!("got `{}`", ty.to_diagnostic_string()),
            )
        })?;
        Ok(Spanned { span: expr.span, inner: ty_hw })
    }

    pub fn eval_expression_as_assign_target(
        &mut self,
        scope: Scope,
        expr: &Expression,
    ) -> Result<Spanned<AssignmentTarget>, ErrorGuaranteed> {
        let build_err = |actual: &str| {
            self.diags
                .report_simple("expected assignment target", expr.span, format!("got `{}`", actual))
        };

        let result = match &expr.inner {
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
        };

        Ok(Spanned { span: expr.span, inner: result? })
    }

    pub fn eval_expression_as_domain_signal(
        &mut self,
        scope: Scope,
        expr: &Expression,
    ) -> Result<Spanned<DomainSignal>, ErrorGuaranteed> {
        // TODO expand to allow general expressions (which then probably create implicit signals)?

        let diags = self.diags;
        let build_err = |actual: &str| {
            self.diags
                .report_simple("expected domain signal", expr.span, format!("got `{}`", actual))
        };

        let result = match &expr.inner {
            ExpressionKind::UnaryOp(
                Spanned {
                    span: _,
                    inner: UnaryOp::Not,
                },
                inner,
            ) => {
                let inner = self.eval_expression_as_domain_signal(scope, inner)?.inner;
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
        };

        Ok(Spanned { span: expr.span, inner: result? })
    }

    pub fn eval_domain_sync(
        &mut self,
        scope: Scope,
        domain: &SyncDomain<Box<Expression>>,
    ) -> Result<SyncDomain<DomainSignal>, ErrorGuaranteed> {
        let clock = self.eval_expression_as_domain_signal(scope, &domain.clock);
        let reset = self.eval_expression_as_domain_signal(scope, &domain.reset);

        Ok(SyncDomain {
            clock: clock?.inner,
            reset: reset?.inner,
        })
    }

    pub fn eval_domain(
        &mut self,
        scope: Scope,
        domain: &Spanned<DomainKind<Box<Expression>>>,
    ) -> Result<Spanned<DomainKind<DomainSignal>>, ErrorGuaranteed> {
        let result = match &domain.inner {
            DomainKind::Async => Ok(DomainKind::Async),
            DomainKind::Sync(domain) => self.eval_domain_sync(scope, domain).map(DomainKind::Sync),
        };

        Ok(Spanned { span: domain.span, inner: result? })
    }
}
