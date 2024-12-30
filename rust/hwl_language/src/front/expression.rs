use crate::front::block::{TypedIrExpression, VariableValues};
use crate::front::check::{check_type_contains_value, check_type_is_bool, check_type_is_int, TypeContainsReason};
use crate::front::compile::{CompileState, ElaborationStackEntry, Port};
use crate::front::context::{CompileTimeExpressionContext, ExpressionContext};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrBoolBinaryOp, IrExpression, IrIntBinaryOp};
use crate::front::misc::{DomainSignal, Polarized, ScopedEntry, Signal, ValueDomain};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type};
use crate::front::value::{AssignmentTarget, CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast;
use crate::syntax::ast::{
    BinaryOp, DomainKind, Expression, ExpressionKind, Identifier, IntPattern, PortDirection, Spanned, SyncDomain,
    UnaryOp,
};
use crate::syntax::pos::Span;
use crate::util::{Never, ResultDoubleExt};
use itertools::{Either, Itertools};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Pow, ToPrimitive};
use std::cmp::{max, min};
use std::convert::identity;
use std::ops::Sub;

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
    pub fn eval_expression<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
    ) -> Result<Spanned<MaybeCompile<TypedIrExpression>>, ErrorGuaranteed> {
        let diags = self.diags;

        let result = match &expr.inner {
            ExpressionKind::Dummy => {
                // if dummy expressions were allowed, the caller would have checked for them already
                Err(diags.report_simple(
                    "dummy expression not allowed in this context",
                    expr.span,
                    "dummy expression used here",
                ))
            }
            ExpressionKind::Undefined => Ok(MaybeCompile::Compile(CompileValue::Undefined)),
            ExpressionKind::Type => Ok(MaybeCompile::Compile(CompileValue::Type(Type::Type))),
            ExpressionKind::Wrapped(inner) => Ok(self.eval_expression(ctx, ctx_block, scope, vars, inner)?.inner),
            ExpressionKind::Id(id) => {
                let eval = self.eval_id(scope, id)?;
                match eval.inner {
                    MaybeCompile::Compile(c) => Ok(MaybeCompile::Compile(c)),
                    MaybeCompile::Other(value) => match value {
                        NamedValue::Constant(cst) => Ok(MaybeCompile::Compile(self.constants[cst].value.clone())),
                        NamedValue::Parameter(param) => Ok(self.parameters[param].value.clone()),
                        NamedValue::Variable(var) => Ok(vars.get(diags, expr.span, var)?.value.clone()),
                        NamedValue::Port(port) => {
                            let port_info = &self.ports[port];
                            match port_info.direction.inner {
                                PortDirection::Input => Ok(MaybeCompile::Other(port_info.typed_ir_expr())),
                                PortDirection::Output => {
                                    Err(self.diags.report_todo(expr.span, "read back from output port"))
                                }
                            }
                        }
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
                UnaryOp::Neg => {
                    let operand = self.eval_expression(ctx, ctx_block, scope, vars, operand)?;
                    let operand = check_type_is_int(diags, TypeContainsReason::Operator(op.span), operand)?;

                    match operand.inner {
                        MaybeCompile::Compile(c) => Ok(MaybeCompile::Compile(CompileValue::Int(-c))),
                        MaybeCompile::Other(v) => {
                            let result_range = ClosedIncRange {
                                start_inc: -v.ty.end_inc,
                                end_inc: -v.ty.start_inc,
                            };
                            let result_expr = IrExpression::IntBinary(
                                IrIntBinaryOp::Sub,
                                Box::new(IrExpression::Int(BigInt::ZERO)),
                                Box::new(v.expr),
                            );

                            let result = TypedIrExpression {
                                ty: HardwareType::Int(result_range),
                                domain: v.domain,
                                expr: result_expr,
                            };
                            Ok(MaybeCompile::Other(result))
                        }
                    }
                }
                UnaryOp::Not => {
                    let operand = self.eval_expression(ctx, ctx_block, scope, vars, operand)?;

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
            ExpressionKind::BinaryOp(op, left, right) => {
                let left = self.eval_expression(ctx, ctx_block, scope, vars, left);
                let right = self.eval_expression(ctx, ctx_block, scope, vars, right);
                self.eval_binary_expression(expr.span, op, left?, right?)
            }
            ExpressionKind::TernarySelect(_, _, _) => Err(diags.report_todo(expr.span, "expr kind TernarySelect")),
            ExpressionKind::ArrayIndex(base, args) => {
                // eval base and args
                let base = self.eval_expression(ctx, ctx_block, scope, vars, base);
                let args = args.map_inner(|a| self.eval_expression(ctx, ctx_block, scope, vars, a));

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
                    let curr_inner = match pair_compile(diags, curr, arg.value)? {
                        MaybeCompile::Compile((curr, index)) => {
                            match curr.inner {
                                // declare new array type
                                CompileValue::Type(curr) => {
                                    let dim_len = match index.inner {
                                        CompileValue::Int(dim_len) => BigUint::try_from(dim_len).map_err(|e| {
                                            diags.report_simple(
                                                "array dimension length cannot be negative",
                                                index.span,
                                                format!("got value `{}`", e.into_original()),
                                            )
                                        })?,
                                        _ => {
                                            return Err(diags.report_simple(
                                                "array dimension length must be an integer",
                                                index.span,
                                                format!("got value `{}`", index.inner.to_diagnostic_string()),
                                            ));
                                        }
                                    };

                                    let result_ty = Type::Array(Box::new(curr), dim_len);
                                    MaybeCompile::Compile(CompileValue::Type(result_ty))
                                }
                                // index into compile-time array
                                CompileValue::Array(curr) => {
                                    match index.inner {
                                        CompileValue::Int(index_inner) => {
                                            let valid_range = 0..curr.len();
                                            let index = index_inner.to_usize()
                                                .filter(|index| valid_range.contains(&index))
                                                .ok_or_else(|| {
                                                    diags.report_simple(
                                                        "array index out of bounds",
                                                        index.span,
                                                        format!("got index `{index_inner}`, valid range for this array is `{valid_range:?}`"),
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
                                                                index.span,
                                                                format!("got slice range `{range:?}`, valid range for this array is `{valid_range:?}`"),
                                                            )
                                                        })
                                                }
                                            };

                                            let start = check_valid(&range.start_inc, 0)?;
                                            let end = check_valid(&range.end_inc, curr.len())?;

                                            if start > end {
                                                return Err(diags.report_internal_error(
                                                    index.span,
                                                    format!("invalid decreasing range `{range:?}`"),
                                                ));
                                            }

                                            // TODO avoid clone, both of this slice and of the entire array?
                                            MaybeCompile::Compile(CompileValue::Array(curr[start..end].to_vec()))
                                        }
                                        _ => {
                                            return Err(diags.report_simple(
                                                "array index must be an integer or an integer range",
                                                index.span,
                                                format!("got value `{}`", index.inner.to_diagnostic_string()),
                                            ));
                                        }
                                    }
                                }
                                _ => {
                                    return Err(diags.report_simple(
                                        "array index on invalid target",
                                        index.span,
                                        format!("target `{}` here", curr.inner.to_diagnostic_string()),
                                    ));
                                }
                            }
                        }
                        MaybeCompile::Other((curr, index)) => {
                            let _ = (curr, index);
                            return Err(diags.report_todo(arg.span, "runtime array indexing/slicing"));
                        }
                    };

                    // TODO this is a bit of sketchy span, but maybe the best we can do
                    curr = Spanned {
                        span: curr_span.join(arg.span),
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
                let args = args.map_inner(|arg| self.eval_expression(ctx, ctx_block, scope, vars, arg));

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
                let entry = ElaborationStackEntry::FunctionCall(expr.span, self.not_eq_stack());
                let (result_block, result_value) = self
                    .check_compile_loop(entry, |s| target.call(s, ctx, args))
                    .flatten_err()?;

                let result_block_spanned = Spanned {
                    span: expr.span,
                    inner: result_block,
                };
                ctx.push_ir_statement_block(ctx_block, result_block_spanned);

                Ok(result_value)
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

    // Proofs of the validness of the integer ranges can be found in `int_range_proofs.py`.
    fn eval_binary_expression(
        &mut self,
        expr_span: Span,
        op: &Spanned<BinaryOp>,
        left: Spanned<MaybeCompile<TypedIrExpression>>,
        right: Spanned<MaybeCompile<TypedIrExpression>>,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let diags = self.diags;

        let op_reason = TypeContainsReason::Operator(op.span);

        // TODO extract even more common int boilerplate
        let check_both_int = |left, right| {
            let left = check_type_is_int(diags, op_reason, left);
            let right = check_type_is_int(diags, op_reason, right);
            Ok((left?, right?))
        };
        let ir_int_binary =
            |range, left: Spanned<TypedIrExpression<_>>, right: Spanned<TypedIrExpression<_>>| TypedIrExpression {
                ty: HardwareType::Int(range),
                domain: left.inner.domain.join(&right.inner.domain),
                expr: IrExpression::IntBinary(
                    IrIntBinaryOp::Add,
                    Box::new(left.inner.expr),
                    Box::new(right.inner.expr),
                ),
            };

        let impl_bool_op = |left, right, op: IrBoolBinaryOp| {
            let left = check_type_is_bool(diags, op_reason, left);
            let right = check_type_is_bool(diags, op_reason, right);

            let left = left?;
            let right = right?;

            match pair_compile_bool(left, right) {
                MaybeCompile::Compile((left, right)) => Ok(MaybeCompile::Compile(CompileValue::Bool(
                    op.eval(left.inner, right.inner),
                ))),
                MaybeCompile::Other((left, right)) => {
                    let result = TypedIrExpression {
                        ty: HardwareType::Bool,
                        domain: left.inner.domain.join(&right.inner.domain),
                        expr: IrExpression::BoolBinary(op, Box::new(left.inner.expr), Box::new(right.inner.expr)),
                    };
                    Ok(MaybeCompile::Other(result))
                }
            }
        };

        match op.inner {
            // (int, int)
            BinaryOp::Add => {
                let (left, right) = check_both_int(left, right)?;
                match pair_compile_int(left, right) {
                    MaybeCompile::Compile((left, right)) => {
                        Ok(MaybeCompile::Compile(CompileValue::Int(left.inner + right.inner)))
                    }
                    MaybeCompile::Other((left, right)) => {
                        let range = ClosedIncRange {
                            start_inc: &left.inner.ty.start_inc + &right.inner.ty.start_inc,
                            end_inc: &left.inner.ty.end_inc + &right.inner.ty.end_inc,
                        };
                        Ok(MaybeCompile::Other(ir_int_binary(range, left, right)))
                    }
                }
            }
            BinaryOp::Sub => {
                let (left, right) = check_both_int(left, right)?;
                match pair_compile_int(left, right) {
                    MaybeCompile::Compile((left, right)) => {
                        Ok(MaybeCompile::Compile(CompileValue::Int(left.inner - right.inner)))
                    }
                    MaybeCompile::Other((left, right)) => {
                        let range = ClosedIncRange {
                            start_inc: &left.inner.ty.start_inc - &right.inner.ty.end_inc,
                            end_inc: &left.inner.ty.end_inc - &right.inner.ty.start_inc,
                        };
                        Ok(MaybeCompile::Other(ir_int_binary(range, left, right)))
                    }
                }
            }
            BinaryOp::Mul => {
                let (left, right) = check_both_int(left, right)?;
                match pair_compile_int(left, right) {
                    MaybeCompile::Compile((left, right)) => {
                        Ok(MaybeCompile::Compile(CompileValue::Int(left.inner * right.inner)))
                    }
                    MaybeCompile::Other((left, right)) => {
                        // calculate valid range
                        let extremes = [
                            &left.inner.ty.start_inc * &right.inner.ty.start_inc,
                            &left.inner.ty.start_inc * &right.inner.ty.end_inc,
                            &left.inner.ty.end_inc * &right.inner.ty.start_inc,
                            &left.inner.ty.end_inc * &right.inner.ty.end_inc,
                        ];
                        let range = ClosedIncRange {
                            start_inc: extremes.iter().min().unwrap().clone(),
                            end_inc: extremes.iter().max().unwrap().clone(),
                        };
                        Ok(MaybeCompile::Other(ir_int_binary(range, left, right)))
                    }
                }
            }
            // (int, non-zero int)
            BinaryOp::Div => Err(diags.report_todo(expr_span, "binary op Div")),
            BinaryOp::Mod => Err(diags.report_todo(expr_span, "binary op Mod")),
            // (nonzero int, non-negative int) or (non-negative int, positive int)
            BinaryOp::Pow => {
                let (base, exp) = check_both_int(left, right)?;

                let zero = BigInt::ZERO;
                let base_range = base.inner.range();
                let exp_range = exp.inner.range();

                // check exp >= 0
                if exp_range.start_inc < &zero {
                    let diag = Diagnostic::new("invalid power operation")
                        .add_error(expr_span, "exponent must be non-negative")
                        .add_info(exp.span, format!("exponent range is `{}`", exp_range))
                        .finish();
                    return Err(diags.report(diag));
                }

                // check not 0 ** 0
                if base_range.contains(&&zero) && exp_range.contains(&&zero) {
                    let diag = Diagnostic::new("invalid power operation `0 ** 0`")
                        .add_error(expr_span, "base and exponent can both be zero")
                        .add_info(base.span, format!("base range is `{}`", base_range))
                        .add_info(exp.span, format!("exponent range is `{}`", exp_range))
                        .finish();
                    return Err(diags.report(diag));
                }

                match pair_compile_int(base, exp) {
                    MaybeCompile::Compile((base, exp)) => {
                        let exp = BigUint::try_from(exp.inner)
                            .map_err(|_| diags.report_internal_error(exp.span, "got negative exp"))?;

                        let result = base.inner.pow(&exp);
                        Ok(MaybeCompile::Compile(CompileValue::Int(result)))
                    }
                    MaybeCompile::Other((base, exp)) => {
                        let exp_start_inc = BigUint::try_from(&exp.inner.ty.start_inc)
                            .map_err(|_| diags.report_internal_error(exp.span, "got negative exp start"))?;
                        let exp_end_inc = BigUint::try_from(&exp.inner.ty.end_inc)
                            .map_err(|_| diags.report_internal_error(exp.span, "got negative exp end"))?;

                        let mut result_min = min(
                            base.inner.ty.start_inc.clone().pow(&exp_start_inc),
                            base.inner.ty.start_inc.clone().pow(&exp_end_inc),
                        );
                        let mut result_max = max(
                            base.inner.ty.start_inc.clone().pow(&exp_end_inc),
                            base.inner.ty.end_inc.clone().pow(&exp_end_inc),
                        );

                        // If base is negative, even/odd powers can cause extremes.
                        // To guard this, try the next highest exponent too if it exists.
                        if exp_end_inc > BigUint::ZERO {
                            let end_exp_sub_one = exp_end_inc.sub(&BigUint::one());
                            result_min = min(result_min, base.inner.ty.start_inc.clone().pow(&end_exp_sub_one));
                            result_max = max(result_max, base.inner.ty.start_inc.clone().pow(&end_exp_sub_one));
                        }

                        let range = ClosedIncRange {
                            start_inc: result_min,
                            end_inc: result_max,
                        };
                        Ok(MaybeCompile::Other(ir_int_binary(range, base, exp)))
                    }
                }
            }
            // (bool, bool)
            BinaryOp::BoolAnd => impl_bool_op(left, right, IrBoolBinaryOp::And),
            BinaryOp::BoolOr => impl_bool_op(left, right, IrBoolBinaryOp::Or),
            BinaryOp::BoolXor => impl_bool_op(left, right, IrBoolBinaryOp::Xor),
            // (T, T)
            BinaryOp::CmpEq => Err(diags.report_todo(expr_span, "binary op CmpEq")),
            BinaryOp::CmpNeq => Err(diags.report_todo(expr_span, "binary op CmpNeq")),
            BinaryOp::CmpLt => Err(diags.report_todo(expr_span, "binary op CmpLt")),
            BinaryOp::CmpLte => Err(diags.report_todo(expr_span, "binary op CmpLte")),
            BinaryOp::CmpGt => Err(diags.report_todo(expr_span, "binary op CmpGt")),
            BinaryOp::CmpGte => Err(diags.report_todo(expr_span, "binary op CmpGte")),
            // (int, range)
            BinaryOp::In => Err(diags.report_todo(expr_span, "binary op In")),

            // TODO boolean arrays?
            BinaryOp::BitAnd => Err(diags.report_todo(expr_span, "binary op BitAnd")),
            BinaryOp::BitOr => Err(diags.report_todo(expr_span, "binary op BitOr")),
            BinaryOp::BitXor => Err(diags.report_todo(expr_span, "binary op BitXor")),
            // TODO (boolean array, non-negative int) and maybe (non-negative int, non-negative int)
            BinaryOp::Shl => Err(diags.report_todo(expr_span, "binary op Shl")),
            BinaryOp::Shr => Err(diags.report_todo(expr_span, "binary op Shr")),
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
            .map(|arg| {
                Ok(self
                    .eval_expression_as_compile(scope, vars, arg, "builtin argument")?
                    .inner)
            })
            .try_collect()?;

        if let (Some(CompileValue::String(a0)), Some(CompileValue::String(a1))) = (args_eval.get(0), args_eval.get(1)) {
            let rest = &args_eval[2..];
            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return Ok(CompileValue::Type(Type::Any)),
                ("type", "bool", []) => return Ok(CompileValue::Type(Type::Bool)),
                ("type", "Range", []) => return Ok(CompileValue::Type(Type::Range)),
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
        let mut ctx = CompileTimeExpressionContext;
        let mut ctx_block = ();

        match self.eval_expression(&mut ctx, &mut ctx_block, scope, vars, expr)?.inner {
            MaybeCompile::Compile(c) => Ok(Spanned {
                span: expr.span,
                inner: c,
            }),
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
            CompileValue::Type(ty) => Ok(Spanned {
                span: expr.span,
                inner: ty,
            }),
            value => Err(self.diags.report_simple(
                "expected type, got value",
                expr.span,
                format!("got value `{}`", value.to_diagnostic_string()),
            )),
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
        Ok(Spanned {
            span: expr.span,
            inner: ty_hw,
        })
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

        Ok(Spanned {
            span: expr.span,
            inner: result?,
        })
    }

    pub fn eval_expression_as_domain_signal(
        &mut self,
        scope: Scope,
        expr: &Expression,
    ) -> Result<Spanned<DomainSignal>, ErrorGuaranteed> {
        let build_err = |actual: &str| {
            self.diags
                .report_simple("expected domain signal", expr.span, format!("got `{}`", actual))
        };
        self.try_eval_expression_as_domain_signal(scope, expr, build_err)
            .map_err(|e| e.into_inner())
    }

    pub fn try_eval_expression_as_domain_signal<E>(
        &mut self,
        scope: Scope,
        expr: &Expression,
        build_err: impl Fn(&str) -> E,
    ) -> Result<Spanned<DomainSignal>, Either<E, ErrorGuaranteed>> {
        // TODO expand to allow general expressions again (which then probably create implicit signals)?
        let result = match &expr.inner {
            ExpressionKind::UnaryOp(
                Spanned {
                    span: _,
                    inner: UnaryOp::Not,
                },
                inner,
            ) => {
                let inner = self
                    .eval_expression_as_domain_signal(scope, inner)
                    .map_err(|e| Either::Right(e))?
                    .inner;
                Ok(inner.invert())
            }
            ExpressionKind::Id(id) => {
                let value = self.eval_id(scope, id).map_err(|e| Either::Right(e))?;
                match value.inner {
                    MaybeCompile::Compile(_) => Err(build_err("compile-time value")),
                    MaybeCompile::Other(s) => match s {
                        NamedValue::Constant(_) => Err(build_err("constant")),
                        NamedValue::Parameter(_) => Err(build_err("parameter")),
                        NamedValue::Variable(_) => Err(build_err("variable")),
                        NamedValue::Port(p) => Ok(Polarized::new(Signal::Port(p))),
                        NamedValue::Wire(w) => Ok(Polarized::new(Signal::Wire(w))),
                        NamedValue::Register(r) => Ok(Polarized::new(Signal::Register(r))),
                    },
                }
            }
            _ => Err(build_err("expression")),
        };

        let result = result.map_err(|e| Either::Left(e))?;
        Ok(Spanned {
            span: expr.span,
            inner: result,
        })
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

        Ok(Spanned {
            span: domain.span,
            inner: result?,
        })
    }

    pub fn eval_port_domain(
        &mut self,
        scope: Scope,
        domain: &Spanned<DomainKind<Box<Expression>>>,
    ) -> Result<Spanned<DomainKind<Polarized<Port>>>, ErrorGuaranteed> {
        let result = self.eval_domain(scope, domain)?;

        Ok(Spanned {
            span: result.span,
            inner: match result.inner {
                DomainKind::Async => DomainKind::Async,
                DomainKind::Sync(sync) => DomainKind::Sync(sync.try_map_inner(|signal| {
                    signal.try_map_inner(|signal| match signal {
                        Signal::Port(port) => Ok(port),
                        Signal::Wire(_) => {
                            Err(self.diags.report_internal_error(domain.span, "expected port, got wire"))
                        }
                        Signal::Register(_) => Err(self
                            .diags
                            .report_internal_error(domain.span, "expected port, got register")),
                    })
                })?),
            },
        })
    }
}

// TODO reduce boilerplate with a trait?
fn pair_compile(
    diags: &Diagnostics,
    left: Spanned<MaybeCompile<TypedIrExpression>>,
    right: Spanned<MaybeCompile<TypedIrExpression>>,
) -> Result<
    MaybeCompile<
        (Spanned<TypedIrExpression>, Spanned<TypedIrExpression>),
        (Spanned<CompileValue>, Spanned<CompileValue>),
    >,
    ErrorGuaranteed,
> {
    pair_compile_general(left, right, |x| x.inner.to_ir_expression(diags, x.span))
}

fn pair_compile_int(
    left: Spanned<MaybeCompile<TypedIrExpression<ClosedIncRange<BigInt>>, BigInt>>,
    right: Spanned<MaybeCompile<TypedIrExpression<ClosedIncRange<BigInt>>, BigInt>>,
) -> MaybeCompile<
    (
        Spanned<TypedIrExpression<ClosedIncRange<BigInt>>>,
        Spanned<TypedIrExpression<ClosedIncRange<BigInt>>>,
    ),
    (Spanned<BigInt>, Spanned<BigInt>),
> {
    let result = pair_compile_general(left, right, |x| {
        Result::<_, Never>::Ok(TypedIrExpression {
            ty: ClosedIncRange::single(x.inner.clone()),
            domain: ValueDomain::CompileTime,
            expr: IrExpression::Int(x.inner),
        })
    });
    match result {
        Ok(result) => result,
        Err(never) => never.unreachable(),
    }
}

fn pair_compile_bool(
    left: Spanned<MaybeCompile<TypedIrExpression<()>, bool>>,
    right: Spanned<MaybeCompile<TypedIrExpression<()>, bool>>,
) -> MaybeCompile<(Spanned<TypedIrExpression<()>>, Spanned<TypedIrExpression<()>>), (Spanned<bool>, Spanned<bool>)> {
    let result = pair_compile_general(left, right, |x| {
        Result::<_, Never>::Ok(TypedIrExpression {
            ty: (),
            domain: ValueDomain::CompileTime,
            expr: IrExpression::Bool(x.inner),
        })
    });
    match result {
        Ok(result) => result,
        Err(never) => never.unreachable(),
    }
}

fn pair_compile_general<T, C, E>(
    left: Spanned<MaybeCompile<T, C>>,
    right: Spanned<MaybeCompile<T, C>>,
    to_other: impl Fn(Spanned<C>) -> Result<T, E>,
) -> Result<MaybeCompile<(Spanned<T>, Spanned<T>), (Spanned<C>, Spanned<C>)>, E> {
    match (left.inner, right.inner) {
        (MaybeCompile::Compile(left_inner), MaybeCompile::Compile(right_inner)) => Ok(MaybeCompile::Compile((
            Spanned {
                span: left.span,
                inner: left_inner,
            },
            Spanned {
                span: right.span,
                inner: right_inner,
            },
        ))),
        (left_inner, right_inner) => {
            let left_inner = match left_inner {
                MaybeCompile::Compile(left_inner) => to_other(Spanned {
                    span: left.span,
                    inner: left_inner,
                }),
                MaybeCompile::Other(left_inner) => Ok(left_inner),
            };
            let right_inner = match right_inner {
                MaybeCompile::Compile(right_inner) => to_other(Spanned {
                    span: right.span,
                    inner: right_inner,
                }),
                MaybeCompile::Other(right_inner) => Ok(right_inner),
            };

            let left_inner = left_inner?;
            let right_inner = right_inner?;

            Ok(MaybeCompile::Other((
                Spanned {
                    span: left.span,
                    inner: left_inner,
                },
                Spanned {
                    span: right.span,
                    inner: right_inner,
                },
            )))
        }
    }
}
