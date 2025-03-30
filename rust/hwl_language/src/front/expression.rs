use crate::front::assignment::{AssignmentTarget, AssignmentTargetBase, ValueVersioned, VariableValues};
use crate::front::block::TypedIrExpression;
use crate::front::check::{
    check_type_contains_value, check_type_is_bool, check_type_is_int, check_type_is_int_compile,
    check_type_is_int_hardware, check_type_is_uint_compile, TypeContainsReason,
};
use crate::front::compile::{CompileState, ElaborationStackEntry, Port};
use crate::front::context::{CompileTimeExpressionContext, ExpressionContext};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::implication::{ClosedIncRangeMulti, Implication, ImplicationOp, Implications};
use crate::front::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrBoolBinaryOp, IrExpression, IrIntArithmeticOp, IrIntCompareOp,
    IrStatement, IrVariable, IrVariableInfo,
};
use crate::front::misc::{DomainSignal, Polarized, ScopedEntry, Signal, SignalOrVariable, ValueDomain};
use crate::front::scope::{Scope, Visibility};
use crate::front::steps::{ArrayStep, ArrayStepCompile, ArrayStepHardware, ArraySteps};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{
    ArrayLiteralElement, BinaryOp, DomainKind, Expression, ExpressionKind, Identifier, IntLiteral, MaybeIdentifier,
    PortDirection, RangeLiteral, Spanned, SyncDomain, UnaryOp,
};
use crate::syntax::pos::Span;
use crate::util::data::vec_concat;
use crate::util::iter::IterExt;
use crate::util::{Never, ResultDoubleExt};
use itertools::{enumerate, Either};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{Num, One, Pow, Signed, ToPrimitive, Zero};
use std::cmp::{max, min};
use std::ops::Sub;
use unwrap_match::unwrap_match;

#[derive(Debug)]
pub struct ExpressionWithImplications {
    pub value: MaybeCompile<TypedIrExpression>,
    pub value_versioned: Option<ValueVersioned>,
    pub implications: Implications,
}

impl ExpressionWithImplications {
    pub fn simple(value: MaybeCompile<TypedIrExpression>) -> Self {
        Self {
            value,
            value_versioned: None,
            implications: Implications::default(),
        }
    }
}

#[derive(Debug)]
pub enum EvaluatedId {
    Named(NamedValue),
    Value(MaybeCompile<TypedIrExpression>),
}

impl CompileState<'_> {
    pub fn eval_id(&mut self, scope: Scope, id: &Identifier) -> Result<Spanned<EvaluatedId>, ErrorGuaranteed> {
        let found = self.state.scopes[scope].find(&self.state.scopes, self.diags, id, Visibility::Private)?;
        let def_span = found.defining_span;
        let result = match found.value {
            &ScopedEntry::Named(value) => EvaluatedId::Named(value),
            &ScopedEntry::Item(item) => EvaluatedId::Value(MaybeCompile::Compile(self.eval_item(item)?.clone())),
            ScopedEntry::Value(value) => EvaluatedId::Value(value.clone()),
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
        Ok(self
            .eval_expression_with_implications(ctx, ctx_block, scope, vars, expected_ty, expr)?
            .map_inner(|r| r.value))
    }

    pub fn eval_expression_with_implications<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<Spanned<ExpressionWithImplications>, ErrorGuaranteed> {
        let value = self.eval_expression_inner(ctx, ctx_block, scope, vars, expected_ty, expr)?;
        Ok(Spanned {
            span: expr.span,
            inner: value,
        })
    }

    // TODO return COW to save some allocations?
    // TODO maybe this should return an abstract expression value,
    //   that can then be written (as target), read (as value), typeof-ed, gotten implications, ...
    //   that's awkward for expressions that create statements though, eg. calls
    //   maybe those should push their statements to virtual blocks, and only actually add them once read?
    fn eval_expression_inner<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<ExpressionWithImplications, ErrorGuaranteed> {
        let diags = self.diags;

        let result_simple: MaybeCompile<TypedIrExpression> = match &expr.inner {
            ExpressionKind::Dummy => {
                // if dummy expressions were allowed, the caller would have checked for them already
                return Err(diags.report_simple(
                    "dummy expression not allowed in this context",
                    expr.span,
                    "dummy expression used here",
                ));
            }
            ExpressionKind::Undefined => MaybeCompile::Compile(CompileValue::Undefined),
            ExpressionKind::Type => MaybeCompile::Compile(CompileValue::Type(Type::Type)),
            ExpressionKind::Wrapped(inner) => {
                return Ok(self
                    .eval_expression_with_implications(ctx, ctx_block, scope, vars, expected_ty, inner)?
                    .inner)
            }
            ExpressionKind::Id(id) => {
                match self.eval_id(scope, id)?.inner {
                    EvaluatedId::Value(value) => value,
                    EvaluatedId::Named(value) => match value {
                        // TODO report error when combinatorial block
                        //   reads something it has not written yet but will later write to
                        // TODO more generally, report combinatorial cycles
                        NamedValue::Variable(var) => {
                            let value = vars.get(diags, expr.span, var)?.value.clone();

                            match value {
                                MaybeCompile::Compile(value) => MaybeCompile::Compile(value),
                                MaybeCompile::Other(value) => {
                                    let versioned = vars.value_versioned(SignalOrVariable::Variable(var));
                                    return Ok(apply_implications(ctx, versioned, value));
                                }
                            }
                        }
                        NamedValue::Port(port) => {
                            ctx.check_ir_context(diags, expr.span, "port")?;

                            let port_info = &self.state.ports[port];
                            return match port_info.direction.inner {
                                PortDirection::Input => {
                                    let versioned = vars.value_versioned(SignalOrVariable::Signal(Signal::Port(port)));
                                    Ok(apply_implications(ctx, versioned, port_info.typed_ir_expr()))
                                }
                                PortDirection::Output => {
                                    Err(self.diags.report_todo(expr.span, "read back from output port"))
                                }
                            };
                        }
                        NamedValue::Wire(wire) => {
                            ctx.check_ir_context(diags, expr.span, "wire")?;

                            let wire_info = &self.state.wires[wire];
                            let value_stored = store_ir_expression_in_dedicated_variable(
                                ctx,
                                ctx_block,
                                diags,
                                expr.span,
                                &wire_info.id,
                                wire_info.typed_ir_expr(),
                            )?;
                            let value_expr_raw = value_stored.to_general_expression();

                            let versioned = vars.value_versioned(SignalOrVariable::Signal(Signal::Wire(wire)));
                            return Ok(apply_implications(ctx, versioned, value_expr_raw));
                        }
                        NamedValue::Register(reg) => {
                            ctx.check_ir_context(diags, expr.span, "register")?;

                            let reg_info = &self.state.registers[reg];
                            let value_stored = store_ir_expression_in_dedicated_variable(
                                ctx,
                                ctx_block,
                                diags,
                                expr.span,
                                &reg_info.id,
                                reg_info.typed_ir_expr(),
                            )?;
                            let value_expr_raw = value_stored.to_general_expression();

                            let versioned = vars.value_versioned(SignalOrVariable::Signal(Signal::Register(reg)));
                            return Ok(apply_implications(ctx, versioned, value_expr_raw));
                        }
                    },
                }
            }
            ExpressionKind::TypeFunction => MaybeCompile::Compile(CompileValue::Type(Type::Function)),
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
                MaybeCompile::Compile(CompileValue::Int(BigInt::from(value)))
            }
            &ExpressionKind::BoolLiteral(literal) => MaybeCompile::Compile(CompileValue::Bool(literal)),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) => MaybeCompile::Compile(CompileValue::String(literal.clone())),

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
                        let expected_ty_curr = match &v.inner {
                            ArrayLiteralElement::Single(_) => expected_ty_inner,
                            ArrayLiteralElement::Spread(_, _) => {
                                &Type::Array(Box::new(expected_ty_inner.clone()), BigUint::zero())
                            }
                        };

                        v.inner
                            .map_inner(|value_inner| {
                                self.eval_expression(ctx, ctx_block, scope, vars, expected_ty_curr, value_inner)
                            })
                            .transpose()
                    })
                    .try_collect_all_vec()?;

                // combine into compile or non-compile value
                let first_non_compile_span = values
                    .iter()
                    .find(|v| !matches!(v.value().inner, MaybeCompile::Compile(_)))
                    .map(|v| v.span());
                if let Some(first_non_compile_span) = first_non_compile_span {
                    // at least one non-compile, turn everything into IR
                    let expected_ty_inner_hw = expected_ty_inner.as_hardware_type().ok_or_else(|| {
                        // TODO clarify that inferred type comes from outside, not the expression itself
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
                    let mut result_len = BigUint::zero();

                    for elem in values {
                        let (elem_ir, domain, elem_len) = match elem {
                            ArrayLiteralElement::Single(elem_inner) => {
                                let value_ir =
                                    elem_inner
                                        .inner
                                        .as_ir_expression(diags, elem_inner.span, &expected_ty_inner_hw)?;

                                check_type_contains_value(
                                    diags,
                                    TypeContainsReason::Operator(expr.span),
                                    expected_ty_inner,
                                    Spanned {
                                        span: elem_inner.span,
                                        inner: &MaybeCompile::Other(value_ir.clone()),
                                    },
                                    true,
                                    true,
                                )?;

                                (
                                    IrArrayLiteralElement::Single(value_ir.expr),
                                    value_ir.domain,
                                    BigUint::one(),
                                )
                            }
                            ArrayLiteralElement::Spread(_, elem_inner) => {
                                let value_ir =
                                    elem_inner
                                        .inner
                                        .as_ir_expression(diags, elem_inner.span, &expected_ty_inner_hw)?;

                                let len = match value_ir.ty() {
                                    Type::Array(_, len) => len,
                                    _ => BigUint::zero(),
                                };
                                check_type_contains_value(
                                    diags,
                                    TypeContainsReason::Operator(expr.span),
                                    &Type::Array(Box::new(expected_ty_inner.clone()), len.clone()),
                                    Spanned {
                                        span: elem_inner.span,
                                        inner: &MaybeCompile::Other(value_ir.clone()),
                                    },
                                    true,
                                    true,
                                )?;

                                (IrArrayLiteralElement::Spread(value_ir.expr), value_ir.domain, len)
                            }
                        };

                        result_domain = result_domain.join(&domain);
                        result_exprs.push(elem_ir);
                        result_len += elem_len;
                    }

                    let result_expr =
                        IrExpression::ArrayLiteral(expected_ty_inner_hw.to_ir(), result_len.clone(), result_exprs);
                    MaybeCompile::Other(TypedIrExpression {
                        ty: HardwareType::Array(Box::new(expected_ty_inner_hw), result_len),
                        domain: result_domain,
                        expr: result_expr,
                    })
                } else {
                    // all compile, create compile value
                    let mut result = vec![];
                    for elem in values {
                        match elem {
                            ArrayLiteralElement::Single(elem_inner) => {
                                let elem_inner = unwrap_match!(elem_inner.inner, MaybeCompile::Compile(v) => v);
                                result.push(elem_inner);
                            }
                            ArrayLiteralElement::Spread(span_spread, elem_inner) => {
                                let elem_inner = unwrap_match!(elem_inner.inner, MaybeCompile::Compile(v) => v);
                                let elem_inner_array = match elem_inner {
                                    CompileValue::Array(elem_inner) => elem_inner,
                                    _ => {
                                        return Err(diags.report_todo(
                                            span_spread,
                                            "compile-time spread only works for fully known arrays for now",
                                        ))
                                    }
                                };
                                result.extend(elem_inner_array)
                            }
                        }
                    }
                    MaybeCompile::Compile(CompileValue::Array(result))
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

                    MaybeCompile::Other(TypedIrExpression {
                        ty: HardwareType::Tuple(result_ty),
                        domain: result_domain,
                        expr: IrExpression::TupleLiteral(result_expr),
                    })
                } else if values
                    .iter()
                    .all(|v| matches!(v.inner, MaybeCompile::Compile(CompileValue::Type(_))))
                {
                    // all type
                    let tys = values
                        .into_iter()
                        .map(|v| unwrap_match!(v.inner, MaybeCompile::Compile(CompileValue::Type(v)) => v))
                        .collect();
                    MaybeCompile::Compile(CompileValue::Type(Type::Tuple(tys)))
                } else {
                    // all compile
                    let values = values
                        .into_iter()
                        .map(|v| unwrap_match!(v.inner, MaybeCompile::Compile(v) => v))
                        .collect();
                    MaybeCompile::Compile(CompileValue::Tuple(values))
                }
            }
            ExpressionKind::StructLiteral(_) => return Err(diags.report_todo(expr.span, "expr kind StructLiteral")),
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

                        let range = IncRange {
                            start_inc: start?,
                            end_inc: end?.map(|end| end - 1),
                        };
                        MaybeCompile::Compile(CompileValue::IntRange(range))
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

                        let range = IncRange {
                            start_inc: start?,
                            end_inc: Some(end?),
                        };
                        MaybeCompile::Compile(CompileValue::IntRange(range))
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
                        let range = IncRange {
                            end_inc: Some(&start + length? - 1),
                            start_inc: Some(start),
                        };
                        MaybeCompile::Compile(CompileValue::IntRange(range))
                    }
                }
            }
            ExpressionKind::UnaryOp(op, operand) => match op.inner {
                UnaryOp::Neg => {
                    let operand = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, operand)?;
                    let operand = check_type_is_int(diags, TypeContainsReason::Operator(op.span), operand)?;

                    match operand.inner {
                        MaybeCompile::Compile(c) => MaybeCompile::Compile(CompileValue::Int(-c)),
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
                            MaybeCompile::Other(result)
                        }
                    }
                }
                UnaryOp::Not => {
                    let operand =
                        self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, operand)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        &Type::Bool,
                        operand.as_ref().map_inner(|operand| &operand.value),
                        false,
                        false,
                    )?;

                    match operand.inner.value {
                        MaybeCompile::Compile(c) => match c {
                            // TODO support boolean array
                            CompileValue::Bool(b) => MaybeCompile::Compile(CompileValue::Bool(!b)),
                            _ => return Err(diags.report_internal_error(expr.span, "expected bool for unary not")),
                        },
                        MaybeCompile::Other(v) => {
                            let result = TypedIrExpression {
                                ty: HardwareType::Bool,
                                domain: v.domain,
                                expr: IrExpression::BoolNot(Box::new(v.expr)),
                            };
                            let result = MaybeCompile::Other(result);

                            return Ok(ExpressionWithImplications {
                                value: result,
                                value_versioned: None,
                                implications: operand.inner.implications.invert(),
                            });
                        }
                    }
                }
            },
            &ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, left);
                let right = self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, right);
                return eval_binary_expression(self.diags, expr.span, op, left?, right?);
            }
            ExpressionKind::TernarySelect(_, _, _) => {
                return Err(diags.report_todo(expr.span, "expr kind TernarySelect"))
            }
            ExpressionKind::ArrayIndex(base, indices) => {
                self.eval_array_index_expression(ctx, ctx_block, scope, vars, base, indices)?
            }
            ExpressionKind::DotIdIndex(_, _) => return Err(diags.report_todo(expr.span, "expr kind DotIdIndex")),
            ExpressionKind::DotIntIndex(_, _) => return Err(diags.report_todo(expr.span, "expr kind DotIntIndex")),
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

                result_value
            }
            ExpressionKind::Builtin(ref args) => self.eval_builtin(ctx, ctx_block, scope, vars, expr.span, args)?,
        };

        Ok(ExpressionWithImplications::simple(result_simple))
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
        let base = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, base);
        let steps = indices
            .inner
            .iter()
            .map(|index| self.eval_expression_as_array_step(ctx, ctx_block, scope, vars, index))
            .try_collect_all_vec();

        let base = base?;
        let steps = ArraySteps::new(steps?);

        steps.apply_to_value(self.diags, base)
    }

    fn eval_expression_as_array_step<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        index: &Expression,
    ) -> Result<Spanned<ArrayStep>, ErrorGuaranteed> {
        let diags = self.diags;

        // special case range with length, it can have a hardware start index
        let step = if let &ExpressionKind::RangeLiteral(RangeLiteral::Length {
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

            let start = start?;
            let len = len?;

            match start.inner {
                MaybeCompile::Compile(start) => {
                    ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, len: Some(len) })
                }
                MaybeCompile::Other(start) => ArrayStep::Other(ArrayStepHardware::ArraySlice { start, len }),
            }
        } else {
            let index_eval = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, index)?;

            match index_eval.transpose() {
                MaybeCompile::Compile(index_or_slice) => match index_or_slice.inner {
                    CompileValue::Int(index) => ArrayStep::Compile(ArrayStepCompile::ArrayIndex(index)),
                    CompileValue::IntRange(range) => {
                        let start = range.start_inc.clone().unwrap_or(BigInt::ZERO);
                        match &range.end_inc {
                            None => ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, len: None }),
                            Some(end_inc) => {
                                let slice_len = (end_inc - &start + 1u32).to_biguint().ok_or_else(|| {
                                    diags.report_internal_error(
                                        index.span,
                                        format!("slice range end cannot be below start, got range `{range}`"),
                                    )
                                })?;
                                ArrayStep::Compile(ArrayStepCompile::ArraySlice {
                                    start,
                                    len: Some(slice_len),
                                })
                            }
                        }
                    }
                    _ => {
                        return Err(diags.report_simple(
                            "array index needs to be an int or a range",
                            index_or_slice.span,
                            format!("got `{}`", index_or_slice.inner.to_diagnostic_string()),
                        ));
                    }
                },
                MaybeCompile::Other(index) => {
                    // TODO make this error message better, specifically refer to non-compile-time index
                    let reason = TypeContainsReason::ArrayIndex { span_index: index.span };
                    let index = check_type_is_int_hardware(diags, reason, index)?;
                    ArrayStep::Other(ArrayStepHardware::ArrayIndex(index.inner))
                }
            }
        };

        Ok(Spanned {
            span: index.span,
            inner: step,
        })
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expr_span: Span,
        args: &Spanned<Vec<Expression>>,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let diags = self.diags;

        // evaluate args
        let args_eval = args
            .inner
            .iter()
            .map(|arg| {
                Ok(self
                    .eval_expression(ctx, ctx_block, scope, vars, &Type::Any, arg)?
                    .inner)
            })
            .try_collect_all_vec()?;

        if let (
            Some(MaybeCompile::Compile(CompileValue::String(a0))),
            Some(MaybeCompile::Compile(CompileValue::String(a1))),
        ) = (args_eval.get(0), args_eval.get(1))
        {
            let rest = &args_eval[2..];
            let mut print_compile = |v: &MaybeCompile<TypedIrExpression>| {
                let value_str = match v {
                    // TODO print strings without quotes
                    MaybeCompile::Compile(v) => v.to_diagnostic_string(),
                    // TODO less ugly formatting for TypedIrExpression
                    MaybeCompile::Other(v) => {
                        let TypedIrExpression { ty, domain, expr: _ } = v;
                        let ty_str = ty.to_diagnostic_string();
                        let domain_str = domain.to_diagnostic_string(self.state);
                        format!("TypedIrExpression {{ ty: {ty_str}, domain: {domain_str}, expr: _ , }}")
                    }
                };
                self.print_handler.println(&value_str);
            };

            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return Ok(MaybeCompile::Compile(CompileValue::Type(Type::Any))),
                ("type", "bool", []) => return Ok(MaybeCompile::Compile(CompileValue::Type(Type::Bool))),
                ("type", "str", []) => return Ok(MaybeCompile::Compile(CompileValue::Type(Type::String))),
                ("type", "Range", []) => return Ok(MaybeCompile::Compile(CompileValue::Type(Type::Range))),
                ("type", "int_range", [MaybeCompile::Compile(CompileValue::IntRange(range))]) => {
                    return Ok(MaybeCompile::Compile(CompileValue::Type(Type::Int(range.clone()))));
                }
                ("fn", "typeof", [value]) => return Ok(MaybeCompile::Compile(CompileValue::Type(value.ty()))),
                ("fn", "print_compile", [value]) => {
                    print_compile(value);
                    return Ok(MaybeCompile::Compile(CompileValue::Tuple(vec![])));
                }
                ("fn", "print", [value]) => {
                    if ctx.is_ir_context() {
                        if let MaybeCompile::Compile(CompileValue::String(value)) = value {
                            let stmt = Spanned::new(expr_span, IrStatement::PrintLn(value.clone()));
                            ctx.push_ir_statement(diags, ctx_block, stmt)?;
                            return Ok(MaybeCompile::Compile(CompileValue::Tuple(vec![])));
                        }
                        // fallthough
                    } else {
                        print_compile(value);
                        return Ok(MaybeCompile::Compile(CompileValue::Tuple(vec![])));
                    }
                }
                ("fn", "unsafe_cast_clock", [MaybeCompile::Other(v)]) => match v.ty {
                    HardwareType::Bool => {
                        return Ok(MaybeCompile::Other(TypedIrExpression {
                            ty: HardwareType::Clock,
                            domain: ValueDomain::Clock,
                            expr: v.expr.clone(),
                        }));
                    }
                    _ => {}
                },
                // TODO add assert_com/assert_sim, error_com/error_sim
                // TODO is there an elegant general way to have both variants of a bunch of similar functions?
                // fallthrough into err
                _ => {}
            }
        }

        // TODO this causes a strange error message when people call eg. int_range with non-compile args
        let diag = Diagnostic::new("invalid builtin arguments")
            .snippet(expr_span)
            .add_error(args.span, "invalid args")
            .finish()
            .finish();
        Err(diags.report(diag))
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
                format!(
                    "got value with domain `{}`",
                    ir_expr.domain.to_diagnostic_string(self.state)
                ),
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

    // TODO move typechecks here, immediately returning expected type if any
    pub fn eval_expression_as_assign_target<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: Scope,
        vars: &VariableValues,
        expr: &Expression,
    ) -> Result<Spanned<AssignmentTarget>, ErrorGuaranteed> {
        let diags = self.diags;
        let build_err =
            |actual: &str| diags.report_simple("expected assignment target", expr.span, format!("got `{}`", actual));

        let result = match &expr.inner {
            ExpressionKind::Id(id) => match self.eval_id(scope, id)?.inner {
                EvaluatedId::Value(_) => return Err(build_err("value")),
                EvaluatedId::Named(s) => match s {
                    NamedValue::Variable(v) => {
                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Variable(v)))
                    }
                    NamedValue::Port(p) => {
                        let direction = self.state.ports[p].direction;
                        match direction.inner {
                            PortDirection::Input => return Err(build_err("input port")),
                            PortDirection::Output => {
                                AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Port(p)))
                            }
                        }
                    }
                    NamedValue::Wire(w) => {
                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Wire(w)))
                    }
                    NamedValue::Register(r) => {
                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Register(r)))
                    }
                },
            },
            ExpressionKind::ArrayIndex(inner_target, indices) => {
                let inner_target = self.eval_expression_as_assign_target(ctx, ctx_block, scope, vars, inner_target);
                let array_steps = indices
                    .inner
                    .iter()
                    .map(|index| self.eval_expression_as_array_step(ctx, ctx_block, scope, vars, index))
                    .try_collect_all_vec();

                let inner_target = inner_target?;
                let array_steps = ArraySteps::new(array_steps?);

                let AssignmentTarget {
                    base: inner_base,
                    array_steps: inner_array_steps,
                } = inner_target.inner;
                if !inner_array_steps.is_empty() {
                    return Err(diags.report_todo(expr.span, "combining target expressions"));
                }

                AssignmentTarget {
                    base: inner_base,
                    array_steps,
                }
            }
            ExpressionKind::DotIdIndex(_, _) => {
                return Err(diags.report_todo(expr.span, "assignment target dot id index"))?
            }
            ExpressionKind::DotIntIndex(_, _) => {
                return Err(diags.report_todo(expr.span, "assignment target dot int index"))?
            }
            _ => return Err(build_err("other expression")),
        };

        Ok(Spanned {
            span: expr.span,
            inner: result,
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
                    EvaluatedId::Value(_) => Err(build_err("value")),
                    EvaluatedId::Named(s) => match s {
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
    let ir_target = IrAssignmentTarget::variable(ir_variable);
    let ir_statement = IrStatement::Assign(ir_target, value.expr);
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

// Proofs of the validness of the integer ranges can be found in `int_range_proofs.py`.
pub fn eval_binary_expression(
    diags: &Diagnostics,
    expr_span: Span,
    op: Spanned<BinaryOp>,
    left: Spanned<ExpressionWithImplications>,
    right: Spanned<ExpressionWithImplications>,
) -> Result<ExpressionWithImplications, ErrorGuaranteed> {
    let op_reason = TypeContainsReason::Operator(op.span);

    let check_both_int = |left, right| {
        let left = check_type_is_int(diags, op_reason, left);
        let right = check_type_is_int(diags, op_reason, right);
        Ok((left?, right?))
    };
    let eval_binary_bool = |op, left, right| eval_binary_bool(diags, op_reason, op, left, right);
    let eval_binary_int_compare = |op, left, right| eval_binary_int_compare(diags, op_reason, op, left, right);

    let result_simple: MaybeCompile<_> = match op.inner {
        // (int, int)
        BinaryOp::Add => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value), right.map_inner(|e| e.value))?;
            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => {
                    MaybeCompile::Compile(CompileValue::Int(left.inner + right.inner))
                }
                MaybeCompile::Other((left, right)) => {
                    let range = ClosedIncRange {
                        start_inc: &left.inner.ty.start_inc + &right.inner.ty.start_inc,
                        end_inc: &left.inner.ty.end_inc + &right.inner.ty.end_inc,
                    };
                    MaybeCompile::Other(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Add,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Sub => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value), right.map_inner(|e| e.value))?;
            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => {
                    MaybeCompile::Compile(CompileValue::Int(left.inner - right.inner))
                }
                MaybeCompile::Other((left, right)) => {
                    let range = ClosedIncRange {
                        start_inc: &left.inner.ty.start_inc - &right.inner.ty.end_inc,
                        end_inc: &left.inner.ty.end_inc - &right.inner.ty.start_inc,
                    };
                    MaybeCompile::Other(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Sub,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Mul => {
            let right = check_type_is_int(diags, op_reason, right.map_inner(|e| e.value));
            match left.inner.value.ty() {
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

                    match left.inner.value {
                        MaybeCompile::Compile(CompileValue::Array(left_inner)) => {
                            // do the repetition at compile-time
                            // TODO check for overflow (everywhere)
                            let mut result = Vec::with_capacity(left_inner.len() * right_inner);
                            for _ in 0..right_inner {
                                result.extend_from_slice(&left_inner);
                            }
                            MaybeCompile::Compile(CompileValue::Array(result))
                        }
                        MaybeCompile::Compile(_) => {
                            return Err(diags.report_internal_error(
                                left.span,
                                "compile-time value with type array is not actually an array",
                            ))
                        }
                        MaybeCompile::Other(value) => {
                            // implement runtime repetition through spread array literal
                            let element = IrArrayLiteralElement::Spread(value.expr);
                            let elements = vec![element; right_inner];

                            let left_ty_inner_hw = left_ty_inner.as_hardware_type().unwrap();
                            let result_len = left_len * right_inner;
                            MaybeCompile::Other(TypedIrExpression {
                                ty: HardwareType::Array(Box::new(left_ty_inner_hw.clone()), result_len.clone()),
                                domain: value.domain,
                                expr: IrExpression::ArrayLiteral(left_ty_inner_hw.to_ir(), result_len, elements),
                            })
                        }
                    }
                }
                Type::Int(_) => {
                    let left =
                        check_type_is_int(diags, op_reason, left.map_inner(|e| e.value)).expect("int, already checked");
                    let right = right?;
                    match pair_compile_int(left, right) {
                        MaybeCompile::Compile((left, right)) => {
                            MaybeCompile::Compile(CompileValue::Int(left.inner * right.inner))
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
                            MaybeCompile::Other(build_binary_int_arithmetic_op(
                                IrIntArithmeticOp::Mul,
                                range,
                                left,
                                right,
                            ))
                        }
                    }
                }
                _ => {
                    return Err(diags.report_simple(
                        "left hand side of multiplication must be an array or an integer",
                        left.span,
                        format!("got value with type `{}`", left.inner.value.ty().to_diagnostic_string()),
                    ))
                }
            }
        }
        // (int, non-zero int)
        BinaryOp::Div => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value), right.map_inner(|e| e.value))?;

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
                    MaybeCompile::Compile(CompileValue::Int(result))
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

                    MaybeCompile::Other(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Div,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Mod => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value), right.map_inner(|e| e.value))?;

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
                    MaybeCompile::Compile(CompileValue::Int(result))
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

                    MaybeCompile::Other(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Mod,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        // (nonzero int, non-negative int) or (non-negative int, positive int)
        BinaryOp::Pow => {
            let (base, exp) = check_both_int(left.map_inner(|e| e.value), right.map_inner(|e| e.value))?;

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
                    MaybeCompile::Compile(CompileValue::Int(result))
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
                    MaybeCompile::Other(build_binary_int_arithmetic_op(IrIntArithmeticOp::Pow, range, base, exp))
                }
            }
        }
        // (bool, bool)
        // TODO these should short-circuit, so delay evaluation of right
        BinaryOp::BoolAnd => return eval_binary_bool(left, right, IrBoolBinaryOp::And),
        BinaryOp::BoolOr => return eval_binary_bool(left, right, IrBoolBinaryOp::Or),
        BinaryOp::BoolXor => return eval_binary_bool(left, right, IrBoolBinaryOp::Xor),
        // (T, T)
        BinaryOp::CmpEq => return eval_binary_int_compare(left, right, IrIntCompareOp::Eq),
        BinaryOp::CmpNeq => return eval_binary_int_compare(left, right, IrIntCompareOp::Neq),
        BinaryOp::CmpLt => return eval_binary_int_compare(left, right, IrIntCompareOp::Lt),
        BinaryOp::CmpLte => return eval_binary_int_compare(left, right, IrIntCompareOp::Lte),
        BinaryOp::CmpGt => return eval_binary_int_compare(left, right, IrIntCompareOp::Gt),
        BinaryOp::CmpGte => return eval_binary_int_compare(left, right, IrIntCompareOp::Gte),
        // (int, range)
        BinaryOp::In => return Err(diags.report_todo(expr_span, "binary op In")),
        // (bool, bool)
        // TODO support boolean arrays
        BinaryOp::BitAnd => return eval_binary_bool(left, right, IrBoolBinaryOp::And),
        BinaryOp::BitOr => return eval_binary_bool(left, right, IrBoolBinaryOp::Or),
        BinaryOp::BitXor => return eval_binary_bool(left, right, IrBoolBinaryOp::Xor),
        // TODO (boolean array, non-negative int) and maybe (non-negative int, non-negative int),
        //   and maybe even negative shift amounts?
        BinaryOp::Shl => return Err(diags.report_todo(expr_span, "binary op Shl")),
        BinaryOp::Shr => return Err(diags.report_todo(expr_span, "binary op Shr")),
    };

    Ok(ExpressionWithImplications::simple(result_simple))
}

fn eval_binary_bool(
    diags: &Diagnostics,
    op_reason: TypeContainsReason,
    left: Spanned<ExpressionWithImplications>,
    right: Spanned<ExpressionWithImplications>,
    op: IrBoolBinaryOp,
) -> Result<ExpressionWithImplications, ErrorGuaranteed> {
    fn build_bool_gate(
        f: impl Fn(bool) -> bool,
        inner_eval: ExpressionWithImplications,
        inner_ir: TypedIrExpression<()>,
    ) -> ExpressionWithImplications {
        match (f(false), f(true)) {
            // constants
            (false, false) => ExpressionWithImplications::simple(MaybeCompile::Compile(CompileValue::Bool(false))),
            (true, true) => ExpressionWithImplications::simple(MaybeCompile::Compile(CompileValue::Bool(true))),
            // pass gate
            (false, true) => inner_eval,
            // not gate
            (true, false) => ExpressionWithImplications {
                value: MaybeCompile::Other(TypedIrExpression {
                    ty: HardwareType::Bool,
                    domain: inner_ir.domain,
                    expr: IrExpression::BoolNot(Box::new(inner_ir.expr)),
                }),
                value_versioned: None,
                implications: inner_eval.implications.invert(),
            },
        }
    }

    let left_value = check_type_is_bool(diags, op_reason, left.as_ref().map_inner(|e| e.value.clone()));
    let right_value = check_type_is_bool(diags, op_reason, right.as_ref().map_inner(|e| e.value.clone()));

    let left_value = left_value?;
    let right_value = right_value?;

    match (left_value.inner, right_value.inner) {
        // full compile-tim eval
        (MaybeCompile::Compile(left), MaybeCompile::Compile(right)) => {
            let result = op.eval(left, right);
            Ok(ExpressionWithImplications::simple(MaybeCompile::Compile(
                CompileValue::Bool(result),
            )))
        }
        // partial compile-time eval
        (MaybeCompile::Compile(left_value), MaybeCompile::Other(right_value)) => {
            Ok(build_bool_gate(|b| op.eval(left_value, b), right.inner, right_value))
        }
        (MaybeCompile::Other(left_value), MaybeCompile::Compile(right_value)) => {
            Ok(build_bool_gate(|b| op.eval(b, right_value), left.inner, left_value))
        }
        // full hardware
        (MaybeCompile::Other(left_value), MaybeCompile::Other(right_value)) => {
            let expr = TypedIrExpression {
                ty: HardwareType::Bool,
                domain: left_value.domain.join(&right_value.domain),
                expr: IrExpression::BoolBinary(op, Box::new(left_value.expr), Box::new(right_value.expr)),
            };

            let implications = match op {
                IrBoolBinaryOp::And => Implications {
                    if_true: vec_concat([left.inner.implications.if_true, right.inner.implications.if_true]),
                    if_false: vec![],
                },
                IrBoolBinaryOp::Or => Implications {
                    if_true: vec![],
                    if_false: vec_concat([left.inner.implications.if_false, right.inner.implications.if_false]),
                },
                IrBoolBinaryOp::Xor => Implications::default(),
            };

            Ok(ExpressionWithImplications {
                value: MaybeCompile::Other(expr),
                value_versioned: None,
                implications,
            })
        }
    }
}

fn eval_binary_int_compare(
    diags: &Diagnostics,
    op_reason: TypeContainsReason,
    left: Spanned<ExpressionWithImplications>,
    right: Spanned<ExpressionWithImplications>,
    op: IrIntCompareOp,
) -> Result<ExpressionWithImplications, ErrorGuaranteed> {
    let left_versioned = left.inner.value_versioned;
    let right_versioned = right.inner.value_versioned;

    let left = check_type_is_int(diags, op_reason, left.map_inner(|e| e.value));
    let right = check_type_is_int(diags, op_reason, right.map_inner(|e| e.value));
    let left = left?;
    let right = right?;

    match pair_compile_int(left, right) {
        MaybeCompile::Compile((left, right)) => {
            let result = op.eval(&left.inner, &right.inner);
            Ok(ExpressionWithImplications::simple(MaybeCompile::Compile(
                CompileValue::Bool(result),
            )))
        }
        MaybeCompile::Other((left, right)) => {
            // TODO warning if the result is always true/false (depending on the ranges)
            //   or maybe just return a compile-time value again?

            // build implications
            let lv = left_versioned;
            let rv = right_versioned;
            let lr = left.inner.ty;
            let rr = right.inner.ty;
            let implications = match op {
                IrIntCompareOp::Lt => implications_lt(lv, lr, rv, rr),
                IrIntCompareOp::Lte => implications_lte(lv, lr, rv, rr),
                IrIntCompareOp::Gt => implications_lt(rv, rr, lv, lr),
                IrIntCompareOp::Gte => implications_lte(rv, rr, lv, lr),
                IrIntCompareOp::Eq => implications_eq(lv, lr, rv, rr),
                IrIntCompareOp::Neq => implications_eq(lv, lr, rv, rr).invert(),
            };

            // build the resulting expression
            let result = TypedIrExpression {
                ty: HardwareType::Bool,
                domain: left.inner.domain.join(&right.inner.domain),
                expr: IrExpression::IntCompare(op, Box::new(left.inner.expr), Box::new(right.inner.expr)),
            };
            let eval = ExpressionWithImplications {
                value: MaybeCompile::Other(result),
                value_versioned: None,
                implications,
            };
            Ok(eval)
        }
    }
}

fn build_binary_int_arithmetic_op(
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

fn implications_lt(
    left: Option<ValueVersioned>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersioned>,
    right_range: ClosedIncRange<BigInt>,
) -> Implications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new(left, ImplicationOp::Lt, right_range.end_inc));
        if_false.push(Implication::new(left, ImplicationOp::Gt, right_range.start_inc - 1));
    }
    if let Some(right) = right {
        if_true.push(Implication::new(right, ImplicationOp::Gt, left_range.start_inc));
        if_false.push(Implication::new(right, ImplicationOp::Lt, left_range.end_inc + 1));
    }

    Implications { if_true, if_false }
}

fn implications_lte(
    left: Option<ValueVersioned>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersioned>,
    right_range: ClosedIncRange<BigInt>,
) -> Implications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new(left, ImplicationOp::Lt, right_range.end_inc + 1));
        if_false.push(Implication::new(left, ImplicationOp::Gt, right_range.start_inc));
    }
    if let Some(right) = right {
        if_true.push(Implication::new(right, ImplicationOp::Gt, left_range.start_inc - 1));
        if_false.push(Implication::new(right, ImplicationOp::Lt, left_range.end_inc));
    }

    Implications { if_true, if_false }
}

fn implications_eq(
    left: Option<ValueVersioned>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersioned>,
    right_range: ClosedIncRange<BigInt>,
) -> Implications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new(left, ImplicationOp::Lt, &right_range.end_inc + 1));
        if_true.push(Implication::new(left, ImplicationOp::Gt, &right_range.start_inc - 1));

        if let Some(right) = right_range.as_single() {
            if_false.push(Implication::new(left, ImplicationOp::Neq, right.clone()));
        }
    }

    if let Some(right) = right {
        if_true.push(Implication::new(right, ImplicationOp::Lt, &left_range.end_inc + 1));
        if_true.push(Implication::new(right, ImplicationOp::Gt, &left_range.start_inc - 1));

        if let Some(left) = left_range.as_single() {
            if_false.push(Implication::new(right, ImplicationOp::Neq, left.clone()));
        }
    }

    Implications { if_true, if_false }
}

fn apply_implications<C: ExpressionContext>(
    ctx: &C,
    versioned: Option<ValueVersioned>,
    expr_raw: TypedIrExpression,
) -> ExpressionWithImplications {
    let versioned = match versioned {
        None => return ExpressionWithImplications::simple(MaybeCompile::Other(expr_raw)),
        Some(versioned) => versioned,
    };

    let value_expr = if let HardwareType::Int(ty) = &expr_raw.ty {
        let mut range = ClosedIncRangeMulti::from_range(ty.clone());
        ctx.for_each_implication(versioned, |implication| {
            range.apply_implication(implication.op, &implication.right);
        });

        match range.to_range() {
            // TODO support never type or maybe specifically empty ranges
            None => expr_raw,
            Some(range) => TypedIrExpression {
                ty: HardwareType::Int(range.clone()),
                domain: expr_raw.domain,
                expr: IrExpression::ConstrainIntRange(range, Box::new(expr_raw.expr)),
            },
        }
    } else {
        expr_raw
    };

    ExpressionWithImplications {
        value: MaybeCompile::Other(value_expr),
        value_versioned: Some(versioned),
        implications: Implications::default(),
    }
}
