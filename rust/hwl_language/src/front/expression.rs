use crate::front::block::{TypedIrExpression, VariableValues};
use crate::front::check::{
    check_type_contains_type, check_type_contains_value, check_type_is_bool, check_type_is_int,
    check_type_is_int_compile, check_type_is_uint_compile, TypeContainsReason,
};
use crate::front::compile::{CompileState, ElaborationStackEntry, Port};
use crate::front::context::{CompileTimeExpressionContext, ExpressionContext};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{
    IrAssignmentTarget, IrBoolBinaryOp, IrExpression, IrIntArithmeticOp, IrIntCompareOp, IrStatement, IrVariable,
    IrVariableInfo,
};
use crate::front::misc::{DomainSignal, Polarized, ScopedEntry, Signal, ValueDomain};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{AssignmentTarget, CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{
    ArrayLiteralElement, BinaryOp, DomainKind, Expression, ExpressionKind, Identifier, IntLiteral, MaybeIdentifier,
    PortDirection, RangeLiteral, Spanned, SyncDomain, UnaryOp,
};
use crate::syntax::pos::Span;
use crate::util::iter::IterExt;
use crate::util::{Never, ResultDoubleExt};
use itertools::{enumerate, Either};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Num, One, Pow, Signed, ToPrimitive, Zero};
use std::cmp::{max, min};
use std::ops::Sub;
use unwrap_match::unwrap_match;

impl CompileState<'_> {
    pub fn eval_id(
        &mut self,
        scope: Scope,
        id: &Identifier,
    ) -> Result<Spanned<MaybeCompile<NamedValue>>, ErrorGuaranteed> {
        let found = self.scopes[scope].find(&self.scopes, self.diags, id, Visibility::Private)?;
        let def_span = found.defining_span;
        let result = match *found.value {
            ScopedEntry::Item(item) => MaybeCompile::Compile(self.eval_item_as_ty_or_value(item)?.clone()),
            ScopedEntry::Direct(value) => MaybeCompile::Other(value),
        };
        Ok(Spanned {
            span: def_span,
            inner: result,
        })
    }

    pub fn eval_expression<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<Spanned<MaybeCompile<TypedIrExpression>>, ErrorGuaranteed> {
        let value = self.eval_expression_inner(ctx, ctx_block, scope, vars, expected_ty, expr)?;
        Ok(Spanned {
            span: expr.span,
            inner: value,
        })
    }

    // TODO return COW to save some allocations?
    fn eval_expression_inner<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let diags = self.diags;

        match &expr.inner {
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
            ExpressionKind::Wrapped(inner) => Ok(self
                .eval_expression(ctx, ctx_block, scope, vars, expected_ty, inner)?
                .inner),
            ExpressionKind::Id(id) => {
                let eval = self.eval_id(scope, id)?;
                match eval.inner {
                    MaybeCompile::Compile(c) => Ok(MaybeCompile::Compile(c)),
                    MaybeCompile::Other(value) => match value {
                        NamedValue::Constant(cst) => Ok(MaybeCompile::Compile(self.constants[cst].value.clone())),
                        NamedValue::Parameter(param) => Ok(self.parameters[param].value.clone()),
                        NamedValue::Variable(var) => Ok(vars.get(diags, expr.span, var)?.value.clone()),
                        NamedValue::Port(port) => {
                            ctx.check_ir_context(diags, expr.span, "port")?;

                            let port_info = &self.ports[port];
                            match port_info.direction.inner {
                                PortDirection::Input => Ok(MaybeCompile::Other(port_info.typed_ir_expr())),
                                PortDirection::Output => {
                                    Err(self.diags.report_todo(expr.span, "read back from output port"))
                                }
                            }
                        }
                        NamedValue::Wire(wire) => {
                            ctx.check_ir_context(diags, expr.span, "wire")?;

                            let wire_info = &self.wires[wire];
                            let value_stored = store_ir_expression_in_dedicated_variable(
                                ctx,
                                ctx_block,
                                diags,
                                expr.span,
                                &wire_info.id,
                                wire_info.typed_ir_expr(),
                            )?;
                            Ok(MaybeCompile::Other(value_stored.to_general_expression()))
                        }
                        NamedValue::Register(reg) => {
                            ctx.check_ir_context(diags, expr.span, "register")?;

                            let reg_info = &self.registers[reg];
                            let value_stored = store_ir_expression_in_dedicated_variable(
                                ctx,
                                ctx_block,
                                diags,
                                expr.span,
                                &reg_info.id,
                                reg_info.typed_ir_expr(),
                            )?;
                            Ok(MaybeCompile::Other(value_stored.to_general_expression()))
                        }
                    },
                }
            }
            ExpressionKind::TypeFunction => Ok(MaybeCompile::Compile(CompileValue::Type(Type::Function))),
            ExpressionKind::IntLiteral(ref pattern) => {
                let value = match pattern {
                    IntLiteral::Binary(s_raw) => {
                        let s_clean = s_raw[2..].replace('_', "");
                        BigUint::from_str_radix(&s_clean, 2).unwrap()
                    }
                    IntLiteral::Decimal(s_raw) => {
                        let s_clean = s_raw.replace('_', "");
                        BigUint::from_str_radix(&s_clean, 10).unwrap()
                    }
                    IntLiteral::Hexadecimal(s) => {
                        let s_hex = s[2..].replace('_', "");
                        BigUint::from_str_radix(&s_hex, 16).unwrap()
                    }
                };
                Ok(MaybeCompile::Compile(CompileValue::Int(BigInt::from(value))))
            }
            &ExpressionKind::BoolLiteral(literal) => Ok(MaybeCompile::Compile(CompileValue::Bool(literal))),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) => Ok(MaybeCompile::Compile(CompileValue::String(literal.clone()))),

            ExpressionKind::ArrayLiteral(values) => {
                // intentionally ignore the length, the caller can pass "0" when they have no opinion on it
                let expected_ty_inner = match expected_ty {
                    Type::Array(inner, _len) => &**inner,
                    _ => &Type::Any,
                };

                // evaluate
                let values = values
                    .iter()
                    .map(|v| {
                        let expected_ty_curr = if v.inner.spread.is_some() {
                            &Type::Array(Box::new(expected_ty_inner.clone()), BigUint::zero())
                        } else {
                            expected_ty_inner
                        };

                        Ok(ArrayLiteralElement {
                            spread: v.inner.spread,
                            value: self.eval_expression(
                                ctx,
                                ctx_block,
                                scope,
                                vars,
                                expected_ty_curr,
                                &v.inner.value,
                            )?,
                        })
                    })
                    .try_collect_all_vec()?;

                // combine into compile or non-compile value
                let first_non_compile_span = values
                    .iter()
                    .find(|v| !matches!(v.value.inner, MaybeCompile::Compile(_)))
                    .map(|v| v.span());
                if let Some(first_non_compile_span) = first_non_compile_span {
                    // at least one non-compile, turn everything into IR
                    let expected_ty_inner_hw = expected_ty_inner.as_hardware_type().ok_or_else(|| {
                        let message = format!(
                            "hardware array literal has inferred inner type `{}` which is not representable in hardware",
                            expected_ty_inner.to_diagnostic_string()
                        );
                        let diag = Diagnostic::new("hardware array type needs to be representable in hardware")
                            .add_error(expr.span, message)
                            .add_info(first_non_compile_span, "necessary because this array element is not a compile-time value, which forces the entire array to be hardware")
                            .finish();
                        diags.report(diag)
                    })?;

                    let mut result_domain = ValueDomain::CompileTime;
                    let mut result_exprs = vec![];

                    for elem in values {
                        let value_ir =
                            elem.value
                                .inner
                                .as_ir_expression(diags, elem.value.span, &expected_ty_inner_hw)?;
                        result_domain = result_domain.join(&value_ir.domain);
                        result_exprs.push(ArrayLiteralElement {
                            spread: elem.spread,
                            value: value_ir.expr,
                        });
                    }

                    let result_len = result_exprs.len();
                    let result_expr = IrExpression::ArrayLiteral(expected_ty_inner_hw.to_ir(), result_exprs);
                    Ok(MaybeCompile::Other(TypedIrExpression {
                        ty: HardwareType::Array(Box::new(expected_ty_inner_hw), BigUint::from(result_len)),
                        domain: result_domain,
                        expr: result_expr,
                    }))
                } else {
                    // all compile, create compile value
                    let mut result = vec![];
                    for v in values {
                        let v_inner = unwrap_match!(v.value.inner, MaybeCompile::Compile(v) => v);
                        if let Some(spread_span) = v.spread {
                            match v_inner {
                                CompileValue::Array(v_inner) => result.extend(v_inner),
                                _ => {
                                    return Err(diags.report_todo(
                                        spread_span,
                                        "the compile-time spread only works for fully known arrays for now",
                                    ))
                                }
                            }
                        } else {
                            result.push(v_inner);
                        }
                    }
                    Ok(MaybeCompile::Compile(CompileValue::Array(result)))
                }
            }
            ExpressionKind::TupleLiteral(values) => {
                let expected_tys_inner = match expected_ty {
                    Type::Tuple(tys) if tys.len() == values.len() => Some(tys),
                    _ => None,
                };

                // evaluate
                let values = values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let expected_ty_i = expected_tys_inner.map_or(&Type::Any, |tys| &tys[i]);
                        self.eval_expression(ctx, ctx_block, scope, vars, expected_ty_i, v)
                    })
                    .try_collect_all_vec()?;

                // combine into compile or non-compile value
                let first_non_compile = values
                    .iter()
                    .find(|v| !matches!(v.inner, MaybeCompile::Compile(_)))
                    .map(|v| v.span);
                if let Some(first_non_compile) = first_non_compile {
                    // at least one non-compile, turn everything into IR
                    let mut result_ty = vec![];
                    let mut result_domain = ValueDomain::CompileTime;
                    let mut result_expr = vec![];

                    for (i, value) in enumerate(values) {
                        let expected_ty_inner = if let Some(expected_tys_inner) = expected_tys_inner {
                            expected_tys_inner[i].clone()
                        } else {
                            value.inner.ty()
                        };
                        let expected_ty_inner_hw = expected_ty_inner.as_hardware_type().ok_or_else(|| {
                            let message = format!(
                                "tuple element has inferred type `{}` which is not representable in hardware",
                                expected_ty_inner.to_diagnostic_string()
                            );
                            let diag = Diagnostic::new("hardware tuple elements need to be representable in hardware")
                                .add_error(value.span, message)
                                .add_info(first_non_compile, "necessary because this other tuple element is not a compile-time value, which forces the entire tuple to be hardware")
                                .finish();
                            diags.report(diag)
                        })?;

                        let value_ir = value.inner.as_ir_expression(diags, value.span, &expected_ty_inner_hw)?;
                        result_ty.push(value_ir.ty);
                        result_domain = result_domain.join(&value_ir.domain);
                        result_expr.push(value_ir.expr);
                    }

                    Ok(MaybeCompile::Other(TypedIrExpression {
                        ty: HardwareType::Tuple(result_ty),
                        domain: result_domain,
                        expr: IrExpression::TupleLiteral(result_expr),
                    }))
                } else if values
                    .iter()
                    .all(|v| matches!(v.inner, MaybeCompile::Compile(CompileValue::Type(_))))
                {
                    // all type
                    let tys = values
                        .into_iter()
                        .map(|v| unwrap_match!(v.inner, MaybeCompile::Compile(CompileValue::Type(v)) => v))
                        .collect();
                    Ok(MaybeCompile::Compile(CompileValue::Type(Type::Tuple(tys))))
                } else {
                    // all compile
                    let values = values
                        .into_iter()
                        .map(|v| unwrap_match!(v.inner, MaybeCompile::Compile(v) => v))
                        .collect();
                    Ok(MaybeCompile::Compile(CompileValue::Tuple(values)))
                }
            }
            ExpressionKind::StructLiteral(_) => Err(diags.report_todo(expr.span, "expr kind StructLiteral")),
            ExpressionKind::RangeLiteral(literal) => {
                let mut eval_bound = |bound: &Expression, name: &str, op_span: Span| {
                    let bound = self.eval_expression_as_compile(scope, vars, bound, &format!("range {name}"))?;
                    let reason = TypeContainsReason::Operator(op_span);
                    check_type_is_int_compile(diags, reason, bound)
                };

                match *literal {
                    RangeLiteral::ExclusiveEnd {
                        op_span,
                        ref start,
                        ref end,
                    } => {
                        let start = start
                            .as_ref()
                            .map(|start| eval_bound(start, "start", op_span))
                            .transpose();
                        let end = end.as_ref().map(|end| eval_bound(end, "end", op_span)).transpose();

                        Ok(MaybeCompile::Compile(CompileValue::IntRange(IncRange {
                            start_inc: start?,
                            end_inc: end?.map(|end| end - 1),
                        })))
                    }
                    RangeLiteral::InclusiveEnd {
                        op_span,
                        ref start,
                        ref end,
                    } => {
                        let start = start
                            .as_ref()
                            .map(|start| eval_bound(start, "start", op_span))
                            .transpose();
                        let end = eval_bound(end, "end", op_span);

                        Ok(MaybeCompile::Compile(CompileValue::IntRange(IncRange {
                            start_inc: start?,
                            end_inc: Some(end?),
                        })))
                    }
                    RangeLiteral::Length {
                        op_span,
                        ref start,
                        ref len,
                    } => {
                        // TODO support runtime starts here too (so they can be full values),
                        //   for now those are special-cased in the array indexing evaluation.
                        //   Maybe we want real support for mixed compile/runtime compounds,
                        //     eg. arrays, tuples, ranges, ...
                        let start = eval_bound(start, "start", op_span);
                        let length = eval_bound(len, "length", op_span);

                        let start = start?;
                        Ok(MaybeCompile::Compile(CompileValue::IntRange(IncRange {
                            end_inc: Some(&start + length? - 1),
                            start_inc: Some(start),
                        })))
                    }
                }
            }
            ExpressionKind::UnaryOp(op, operand) => match op.inner {
                UnaryOp::Neg => {
                    let operand = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, operand)?;
                    let operand = check_type_is_int(diags, TypeContainsReason::Operator(op.span), operand)?;

                    match operand.inner {
                        MaybeCompile::Compile(c) => Ok(MaybeCompile::Compile(CompileValue::Int(-c))),
                        MaybeCompile::Other(v) => {
                            let result_range = ClosedIncRange {
                                start_inc: -v.ty.end_inc,
                                end_inc: -v.ty.start_inc,
                            };
                            let result_expr = IrExpression::IntArithmetic(
                                IrIntArithmeticOp::Sub,
                                result_range.clone(),
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
                    let operand = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, operand)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        &Type::Bool,
                        operand.as_ref(),
                        false,
                        false,
                    )?;

                    match operand.inner {
                        MaybeCompile::Compile(c) => match c {
                            // TODO support boolean array
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
            &ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, left);
                let right = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, right);
                self.eval_binary_expression(expr.span, op, left?, right?)
            }
            ExpressionKind::TernarySelect(_, _, _) => Err(diags.report_todo(expr.span, "expr kind TernarySelect")),
            ExpressionKind::ArrayIndex(base, indices) => {
                self.eval_array_index_expression(ctx, ctx_block, scope, vars, base, indices)
            }
            ExpressionKind::DotIdIndex(_, _) => Err(diags.report_todo(expr.span, "expr kind DotIdIndex")),
            ExpressionKind::DotIntIndex(_, _) => Err(diags.report_todo(expr.span, "expr kind DotIntIndex")),
            ExpressionKind::Call(target, args) => {
                // evaluate target and args
                let target = self.eval_expression_as_compile(scope, vars, target, "call target")?;
                let args =
                    args.try_map_inner_all(|arg| self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, arg));

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
                let args = args?;

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
        }
    }

    fn eval_array_index_expression<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        base: &Expression,
        indices: &Spanned<Vec<Expression>>,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let diags = self.diags;

        // eval base and args
        let base = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, base);
        let indices = indices
            .inner
            .iter()
            .map(|index| {
                // special case range with length
                if let &ExpressionKind::RangeLiteral(RangeLiteral::Length {
                    op_span,
                    ref start,
                    ref len,
                }) = &index.inner
                {
                    let reason = TypeContainsReason::Operator(op_span);
                    let start = self
                        .eval_expression(ctx, ctx_block, scope, vars, &Type::Any, start)
                        .and_then(|start| check_type_is_int(diags, reason, start));
                    let len = self
                        .eval_expression_as_compile(scope, vars, len, "range length")
                        .and_then(|len| check_type_is_uint_compile(diags, reason, len));

                    let (start, len) = (start?, len?);
                    match start.inner {
                        MaybeCompile::Compile(start) => {
                            // decay back to normal range for fully compile-time
                            let range = IncRange {
                                end_inc: Some(&start + &BigInt::from(len.clone()) - 1),
                                start_inc: Some(start),
                            };
                            let range_compile = MaybeCompile::Compile(CompileValue::IntRange(range));
                            Ok(Spanned {
                                span: index.span,
                                inner: range_compile,
                            })
                        }
                        MaybeCompile::Other(start) => {
                            // preserve special case
                            let range = IrArrayIndexKind::SliceRange { start, len };
                            Ok(Spanned {
                                span: index.span,
                                inner: MaybeCompile::Other(range),
                            })
                        }
                    }
                } else {
                    let index_eval = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, index)?;

                    let index_inner = match index_eval.inner {
                        MaybeCompile::Compile(x) => MaybeCompile::Compile(x),
                        MaybeCompile::Other(x) => MaybeCompile::Other(IrArrayIndexKind::IndexSingle(x)),
                    };

                    Ok(Spanned {
                        span: index_eval.span,
                        inner: index_inner,
                    })
                }
            })
            .try_collect_all_vec();

        let base = base?;
        let indices = indices?;

        // loop over all indices, folding the indexing operations
        let mut curr = base;
        for index in indices {
            curr = self.eval_array_index_expression_step(curr, index)?;
        }
        Ok(curr.inner)
    }

    fn eval_array_index_expression_step(
        &mut self,
        curr: Spanned<MaybeCompile<TypedIrExpression>>,
        index: Spanned<MaybeCompile<IrArrayIndexKind>>,
    ) -> Result<Spanned<MaybeCompile<TypedIrExpression>>, ErrorGuaranteed> {
        let diags = self.diags;

        let next_inner = match (curr.inner, index.inner) {
            (MaybeCompile::Compile(curr_inner), MaybeCompile::Compile(index_inner)) => {
                let index = Spanned {
                    span: index.span,
                    inner: index_inner,
                };

                match curr_inner {
                    CompileValue::Type(curr) => {
                        // declare new array type
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
                    CompileValue::Array(curr_inner) => {
                        // simple compile-time array indexing
                        match check_array_index(diags, curr.span, BigUint::from(curr_inner.len()), index)? {
                            CheckedArrayIndex::Single { index } => {
                                let index = index.to_usize().unwrap();
                                MaybeCompile::Compile(curr_inner[index].clone())
                            }
                            CheckedArrayIndex::Slice { start, len } => {
                                let start = start.to_usize().unwrap();
                                let len = len.to_usize().unwrap();
                                MaybeCompile::Compile(CompileValue::Array(curr_inner[start..start + len].to_vec()))
                            }
                        }
                    }
                    _ => {
                        return Err(diags.report_simple(
                            "array index on invalid target, must be type or array",
                            index.span,
                            format!("target `{}` here", curr_inner.to_diagnostic_string()),
                        ));
                    }
                }
            }
            (curr_inner, index_inner) => {
                let index = Spanned {
                    span: index.span,
                    inner: index_inner,
                };

                let (curr_ty_inner, curr_len) = match curr_inner.ty() {
                    Type::Array(curr_ty_inner, curr_len) => (*curr_ty_inner, curr_len),
                    _ => {
                        return Err(diags.report_simple(
                            "array index on invalid target, must be array",
                            curr.span,
                            format!("target `{}` here", curr_inner.ty().to_diagnostic_string()),
                        ));
                    }
                };

                let curr_ty_inner_hw = curr_ty_inner.as_hardware_type().ok_or_else(|| todo!())?;
                let curr_ir = curr_inner.as_ir_expression(
                    diags,
                    curr.span,
                    &HardwareType::Array(Box::new(curr_ty_inner_hw.clone()), curr_len.clone()),
                )?;
                let base = Box::new(curr_ir.expr);

                let result = match index.inner {
                    MaybeCompile::Compile(index_inner) => {
                        let index = Spanned {
                            span: index.span,
                            inner: index_inner,
                        };
                        match check_array_index(diags, curr.span, curr_len, index)? {
                            CheckedArrayIndex::Single { index } => {
                                let expr = IrExpression::ArrayIndex {
                                    base,
                                    index: Box::new(IrExpression::Int(BigInt::from(index))),
                                };
                                TypedIrExpression {
                                    ty: curr_ty_inner_hw,
                                    domain: curr_ir.domain,
                                    expr,
                                }
                            }
                            CheckedArrayIndex::Slice { start, len } => {
                                let expr = IrExpression::ArraySlice {
                                    base,
                                    start: Box::new(IrExpression::Int(BigInt::from(start))),
                                    len: len.clone(),
                                };
                                TypedIrExpression {
                                    ty: HardwareType::Array(Box::new(curr_ty_inner_hw), len),
                                    domain: curr_ir.domain,
                                    expr,
                                }
                            }
                        }
                    }
                    MaybeCompile::Other(index_inner) => {
                        match index_inner {
                            IrArrayIndexKind::IndexSingle(index_inner) => {
                                // check type and range
                                let reason = TypeContainsReason::ArrayIndex { span_index: index.span };
                                let valid_range = IncRange {
                                    start_inc: Some(BigInt::ZERO),
                                    end_inc: Some(BigInt::from(curr_len) - 1),
                                };
                                let value_ty = Spanned {
                                    span: index.span,
                                    inner: &index_inner.ty.as_type(),
                                };
                                check_type_contains_type(diags, reason, &Type::Int(valid_range), value_ty, false)?;

                                // build expression
                                let expr = IrExpression::ArrayIndex {
                                    base,
                                    index: Box::new(index_inner.expr),
                                };
                                TypedIrExpression {
                                    ty: curr_ty_inner_hw,
                                    domain: curr_ir.domain.join(&index_inner.domain),
                                    expr,
                                }
                            }
                            IrArrayIndexKind::SliceRange { start, len } => {
                                // check ranges
                                let min_start = &start.ty.start_inc;
                                let max_end_inc = &start.ty.end_inc + BigInt::from(len.clone()) - 1;

                                let valid_start_range = ClosedIncRange {
                                    start_inc: BigInt::ZERO,
                                    end_inc: BigInt::from(curr_len.clone()),
                                };
                                let valid_end_inc_range = ClosedIncRange {
                                    start_inc: BigInt::from(-1),
                                    end_inc: BigInt::from(curr_len) - 1,
                                };

                                let start_valid = valid_start_range.contains(&min_start);
                                let end_valid = valid_end_inc_range.contains(&max_end_inc);

                                if !start_valid || !end_valid {
                                    let invalid = match (start_valid, end_valid) {
                                        (false, false) => "start and end",
                                        (false, true) => "start",
                                        (true, false) => "end",
                                        (true, true) => unreachable!(),
                                    };

                                    let msg_error = format!(
                                        "got slice range with start range `{}` and length `{}`,resulting in total range `{}`",
                                        start.ty,
                                        len,
                                        ClosedIncRange {
                                            start_inc: min_start,
                                            end_inc: &max_end_inc,
                                        }
                                    );
                                    let msg_info = format!(
                                        "for this array the valid start range is `{}` and end range is `{}`",
                                        valid_start_range, valid_end_inc_range
                                    );

                                    let diag = Diagnostic::new(format!("array slice range {invalid} out of bounds"))
                                        .add_error(index.span, msg_error)
                                        .add_info(curr.span, msg_info)
                                        .finish();
                                    return Err(diags.report(diag));
                                }

                                // build result
                                let expr = IrExpression::ArraySlice {
                                    base,
                                    start: Box::new(start.expr),
                                    len: len.clone(),
                                };
                                TypedIrExpression {
                                    ty: HardwareType::Array(Box::new(curr_ty_inner_hw), len),
                                    domain: curr_ir.domain.join(&start.domain),
                                    expr,
                                }
                            }
                        }
                    }
                };

                MaybeCompile::Other(result)
            }
        };

        // TODO this is a bit of sketchy span, but maybe the best we can do
        Ok(Spanned {
            span: curr.span.join(index.span),
            inner: next_inner,
        })
    }

    // Proofs of the validness of the integer ranges can be found in `int_range_proofs.py`.
    pub fn eval_binary_expression(
        &mut self,
        expr_span: Span,
        op: Spanned<BinaryOp>,
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
        fn ir_binary_arith(
            op: IrIntArithmeticOp,
            range: ClosedIncRange<BigInt>,
            left: Spanned<TypedIrExpression<ClosedIncRange<BigInt>>>,
            right: Spanned<TypedIrExpression<ClosedIncRange<BigInt>>>,
        ) -> TypedIrExpression {
            let result_expr =
                IrExpression::IntArithmetic(op, range.clone(), Box::new(left.inner.expr), Box::new(right.inner.expr));
            TypedIrExpression {
                ty: HardwareType::Int(range),
                domain: left.inner.domain.join(&right.inner.domain),
                expr: result_expr,
            }
        }

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

        let impl_int_compare_op = |left, right, op: IrIntCompareOp| {
            let left = check_type_is_int(diags, op_reason, left);
            let right = check_type_is_int(diags, op_reason, right);

            let left = left?;
            let right = right?;

            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => Ok(MaybeCompile::Compile(CompileValue::Bool(
                    op.eval(&left.inner, &right.inner),
                ))),
                MaybeCompile::Other((left, right)) => {
                    // TODO warning if the result is always true/false (depending on the ranges)
                    //   or maybe just return a compile-time value again?
                    let result = TypedIrExpression {
                        ty: HardwareType::Bool,
                        domain: left.inner.domain.join(&right.inner.domain),
                        expr: IrExpression::IntCompare(op, Box::new(left.inner.expr), Box::new(right.inner.expr)),
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
                        Ok(MaybeCompile::Other(ir_binary_arith(
                            IrIntArithmeticOp::Add,
                            range,
                            left,
                            right,
                        )))
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
                        Ok(MaybeCompile::Other(ir_binary_arith(
                            IrIntArithmeticOp::Sub,
                            range,
                            left,
                            right,
                        )))
                    }
                }
            }
            BinaryOp::Mul => {
                let right = check_type_is_int(diags, op_reason, right);
                match left.inner.ty() {
                    Type::Array(left_ty_inner, left_len) => {
                        let right = right?;
                        let right_inner = match right.inner {
                            MaybeCompile::Compile(right_inner) => right_inner,
                            MaybeCompile::Other(_) => {
                                return Err(diags.report_simple(
                                    "array repetition right hand side must be compile-time value",
                                    right.span,
                                    "got non-compile-time value here",
                                ));
                            }
                        };
                        let right_inner = BigUint::try_from(right_inner).map_err(|right_inner| {
                            diags.report_simple(
                                "array repetition right hand side cannot be negative",
                                right.span,
                                format!("got value `{}`", right_inner.into_original()),
                            )
                        })?;
                        let right_inner = right_inner.to_usize().ok_or_else(|| {
                            diags.report_simple(
                                "array repetition right hand side too large",
                                right.span,
                                format!("got value `{}`", right_inner),
                            )
                        })?;

                        match left.inner {
                            MaybeCompile::Compile(CompileValue::Array(left_inner)) => {
                                // do the repetition at compile-time
                                // TODO check for overflow (everywhere)
                                let mut result = Vec::with_capacity(left_inner.len() * right_inner);
                                for _ in 0..right_inner {
                                    result.extend_from_slice(&left_inner);
                                }
                                Ok(MaybeCompile::Compile(CompileValue::Array(result)))
                            }
                            MaybeCompile::Compile(_) => Err(diags.report_internal_error(
                                left.span,
                                "compile-time value with type array is not actually an array",
                            )),
                            MaybeCompile::Other(value) => {
                                // implement runtime repetition through spread array literal
                                let element = ArrayLiteralElement {
                                    spread: Some(op.span),
                                    value: value.expr,
                                };
                                let elements = vec![element; right_inner];

                                let left_ty_inner_hw = left_ty_inner.as_hardware_type().unwrap();
                                Ok(MaybeCompile::Other(TypedIrExpression {
                                    ty: HardwareType::Array(Box::new(left_ty_inner_hw.clone()), left_len * right_inner),
                                    domain: value.domain,
                                    expr: IrExpression::ArrayLiteral(left_ty_inner_hw.to_ir(), elements),
                                }))
                            }
                        }
                    }
                    Type::Int(_) => {
                        let left = check_type_is_int(diags, op_reason, left).expect("int, already checked");
                        let right = right?;
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
                                Ok(MaybeCompile::Other(ir_binary_arith(
                                    IrIntArithmeticOp::Mul,
                                    range,
                                    left,
                                    right,
                                )))
                            }
                        }
                    }
                    _ => Err(diags.report_simple(
                        "left hand side of multiplication must be an array or an integer",
                        left.span,
                        format!("got value with type `{}`", left.inner.ty().to_diagnostic_string()),
                    )),
                }
            }
            // (int, non-zero int)
            BinaryOp::Div => {
                let (left, right) = check_both_int(left, right)?;

                // check nonzero
                if right.inner.range().contains(&&BigInt::ZERO) {
                    let diag = Diagnostic::new("division by zero is not allowed")
                        .add_error(
                            right.span,
                            format!(
                                "right hand side has range `{}` which contains zero",
                                right.inner.range()
                            ),
                        )
                        .add_info(op.span, "for operator here")
                        .finish();
                    return Err(diags.report(diag));
                }
                let right_positive = right.inner.range().start_inc.is_positive();

                match pair_compile_int(left, right) {
                    MaybeCompile::Compile((left, right)) => {
                        let result = left.inner.div_floor(&right.inner);
                        Ok(MaybeCompile::Compile(CompileValue::Int(result)))
                    }
                    MaybeCompile::Other((left, right)) => {
                        let a_min = &left.inner.ty.start_inc;
                        let a_max = &left.inner.ty.end_inc;
                        let b_min = &right.inner.ty.start_inc;
                        let b_max = &right.inner.ty.end_inc;
                        let range = if right_positive {
                            ClosedIncRange {
                                start_inc: min(a_min.div_floor(b_max), a_min.div_floor(b_min)),
                                end_inc: max(a_max.div_floor(b_max), a_max.div_floor(b_min)),
                            }
                        } else {
                            ClosedIncRange {
                                start_inc: min(a_max.div_floor(b_max), a_max.div_floor(b_min)),
                                end_inc: max(a_min.div_floor(b_max), a_min.div_floor(b_min)),
                            }
                        };

                        Ok(MaybeCompile::Other(ir_binary_arith(
                            IrIntArithmeticOp::Div,
                            range,
                            left,
                            right,
                        )))
                    }
                }
            }
            BinaryOp::Mod => {
                let (left, right) = check_both_int(left, right)?;

                // check nonzero
                if right.inner.range().contains(&&BigInt::ZERO) {
                    let diag = Diagnostic::new("modulo by zero is not allowed")
                        .add_error(
                            right.span,
                            format!(
                                "right hand side has range `{}` which contains zero",
                                right.inner.range()
                            ),
                        )
                        .add_info(op.span, "for operator here")
                        .finish();
                    return Err(diags.report(diag));
                }
                let right_positive = right.inner.range().start_inc.is_positive();

                match pair_compile_int(left, right) {
                    MaybeCompile::Compile((left, right)) => {
                        let result = left.inner.mod_floor(&right.inner);
                        Ok(MaybeCompile::Compile(CompileValue::Int(result)))
                    }
                    MaybeCompile::Other((left, right)) => {
                        let range = if right_positive {
                            ClosedIncRange {
                                start_inc: BigInt::ZERO,
                                end_inc: &right.inner.ty.end_inc - 1,
                            }
                        } else {
                            ClosedIncRange {
                                start_inc: &right.inner.ty.start_inc + 1,
                                end_inc: BigInt::ZERO,
                            }
                        };

                        Ok(MaybeCompile::Other(ir_binary_arith(
                            IrIntArithmeticOp::Mod,
                            range,
                            left,
                            right,
                        )))
                    }
                }
            }
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
                        Ok(MaybeCompile::Other(ir_binary_arith(
                            IrIntArithmeticOp::Pow,
                            range,
                            base,
                            exp,
                        )))
                    }
                }
            }
            // (bool, bool)
            BinaryOp::BoolAnd => impl_bool_op(left, right, IrBoolBinaryOp::And),
            BinaryOp::BoolOr => impl_bool_op(left, right, IrBoolBinaryOp::Or),
            BinaryOp::BoolXor => impl_bool_op(left, right, IrBoolBinaryOp::Xor),
            // (T, T)
            BinaryOp::CmpEq => impl_int_compare_op(left, right, IrIntCompareOp::Eq),
            BinaryOp::CmpNeq => impl_int_compare_op(left, right, IrIntCompareOp::Neq),
            BinaryOp::CmpLt => impl_int_compare_op(left, right, IrIntCompareOp::Lt),
            BinaryOp::CmpLte => impl_int_compare_op(left, right, IrIntCompareOp::Lte),
            BinaryOp::CmpGt => impl_int_compare_op(left, right, IrIntCompareOp::Gt),
            BinaryOp::CmpGte => impl_int_compare_op(left, right, IrIntCompareOp::Gte),
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
        let args_eval = args
            .inner
            .iter()
            .map(|arg| {
                Ok(self
                    .eval_expression_as_compile(scope, vars, arg, "builtin argument")?
                    .inner)
            })
            .try_collect_all_vec()?;

        if let (Some(CompileValue::String(a0)), Some(CompileValue::String(a1))) = (args_eval.get(0), args_eval.get(1)) {
            let rest = &args_eval[2..];
            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return Ok(CompileValue::Type(Type::Any)),
                ("type", "bool", []) => return Ok(CompileValue::Type(Type::Bool)),
                ("type", "Range", []) => return Ok(CompileValue::Type(Type::Range)),
                ("type", "int_range", [CompileValue::IntRange(range)]) => {
                    return Ok(CompileValue::Type(Type::Int(range.clone())));
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
        let mut ctx = CompileTimeExpressionContext {
            span: expr.span,
            reason: reason.to_owned(),
        };
        let mut ctx_block = ();

        let value_eval = self
            .eval_expression(&mut ctx, &mut ctx_block, scope, vars, &Type::Any, expr)?
            .inner;
        match value_eval {
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

fn store_ir_expression_in_dedicated_variable<C: ExpressionContext>(
    ctx: &mut C,
    ctx_block: &mut C::Block,
    diags: &Diagnostics,
    span_expr: Span,
    id: &MaybeIdentifier,
    value: TypedIrExpression,
) -> Result<TypedIrExpression<HardwareType, IrVariable>, ErrorGuaranteed> {
    let ir_variable_info = IrVariableInfo {
        ty: value.ty.to_ir(),
        debug_info_id: id.clone(),
    };
    let ir_variable = ctx.new_ir_variable(diags, span_expr, ir_variable_info)?;
    let ir_statement = IrStatement::Assign(IrAssignmentTarget::Variable(ir_variable), value.expr);
    ctx.push_ir_statement(
        diags,
        ctx_block,
        Spanned {
            span: span_expr,
            inner: ir_statement,
        },
    )?;

    let stored_value = TypedIrExpression {
        ty: value.ty,
        domain: value.domain,
        expr: ir_variable,
    };
    Ok(stored_value)
}

enum IrArrayIndexKind {
    IndexSingle(TypedIrExpression),
    SliceRange {
        start: TypedIrExpression<ClosedIncRange<BigInt>>,
        len: BigUint,
    },
}

enum CheckedArrayIndex {
    Single { index: BigUint },
    Slice { start: BigUint, len: BigUint },
}

fn check_array_index(
    diags: &Diagnostics,
    array_span: Span,
    array_len: BigUint,
    index: Spanned<CompileValue>,
) -> Result<CheckedArrayIndex, ErrorGuaranteed> {
    match index.inner {
        CompileValue::Int(index_inner) => {
            let valid_range_index = BigInt::ZERO..BigInt::from(array_len);
            if !valid_range_index.contains(&index_inner) {
                return Err(diags.report_simple(
                    "array index out of bounds",
                    index.span,
                    format!("got index `{index_inner}`, valid index range for this array is `{valid_range_index:?}`"),
                ));
            }
            let index = index_inner.to_biguint().unwrap();

            Ok(CheckedArrayIndex::Single { index })
        }
        CompileValue::IntRange(range) => {
            // because we're using inclusive ranges (they are more convenient for math),
            //   the slicing ranges become a bit weird
            let valid_range_start_inc = BigInt::ZERO..=BigInt::from(array_len.clone());
            let valid_range_end_inc = BigInt::from(-1i32)..BigInt::from(array_len.clone());

            let IncRange { start_inc, end_inc } = &range;

            let start_valid = start_inc
                .as_ref()
                .map_or(true, |start_inc| valid_range_start_inc.contains(start_inc));
            let end_valid = end_inc
                .as_ref()
                .map_or(true, |end_inc| valid_range_end_inc.contains(end_inc));
            if !start_valid || !end_valid {
                let invalid = match (start_valid, end_valid) {
                    (false, false) => "start and end",
                    (false, true) => "start",
                    (true, false) => "end",
                    (true, true) => unreachable!(),
                };

                let diag = Diagnostic::new(format!("array slice range {invalid} out of bounds"))
                    .add_error(index.span, format!("got slice range `{range}`"))
                    .add_info(array_span, format!("for this array the valid start range is `{valid_range_start_inc:?}` and end range is `{valid_range_end_inc:?}`"))
                    .finish();
                return Err(diags.report(diag));
            }

            let start_inc = match &range.start_inc {
                None => BigUint::ZERO,
                Some(start_inc) => start_inc.to_biguint().unwrap(),
            };
            let end_ex = match &range.end_inc {
                None => array_len.clone(),
                Some(end_inc) => (end_inc + 1u32).to_biguint().unwrap(),
            };

            if start_inc > end_ex {
                return Err(diags.report_internal_error(
                    index.span,
                    format!("invalid decreasing range `{range:?}` turned into `{start_inc}..{end_ex}` "),
                ));
            }

            let len = end_ex - &start_inc;
            Ok(CheckedArrayIndex::Slice { start: start_inc, len })
        }
        _ => Err(diags.report_simple(
            "array index must be an integer or an integer range",
            index.span,
            format!(
                "got value `{}` with type `{}`",
                index.inner.to_diagnostic_string(),
                index.inner.ty().to_diagnostic_string()
            ),
        )),
    }
}
