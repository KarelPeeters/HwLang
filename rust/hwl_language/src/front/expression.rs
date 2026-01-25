use crate::front::assignment::AssignmentTarget;
use crate::front::check::{
    TypeContainsReason, check_hardware_type_for_bit_operation, check_type_contains_value, check_type_is_bool,
    check_type_is_int, check_type_is_int_hardware, check_type_is_string_compile, check_type_is_uint,
    check_type_is_uint_compile,
};
use crate::front::compile::{CompileItemContext, CompileRefs, StackEntry};
use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::{DomainSignal, ValueDomain};
use crate::front::flow::{ExtraRegisters, FlowKind, ValueVersion, VariableId};
use crate::front::function::{FunctionBits, FunctionBitsKind, FunctionBody, FunctionValue, error_unique_mismatch};
use crate::front::implication::{BoolImplications, HardwareValueWithImplications, Implication, ValueWithImplications};
use crate::front::item::{ElaboratedInterfaceView, ElaboratedModule, FunctionItemBody};
use crate::front::scope::{NamedValue, Scope, ScopedEntry};
use crate::front::signal::{Polarized, Port, PortInterface, Signal, WireInterface};
use crate::front::steps::{ArrayStep, ArrayStepCompile, ArrayStepHardware, ArraySteps};
use crate::front::types::{HardwareType, NonHardwareType, Type, TypeBool, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, EnumValue, HardwareInt, HardwareValue, MaybeCompile, MaybeUndefined,
    MixedCompoundValue, NotCompile, RangeValue, SimpleCompileValue, Value, ValueCommon,
};
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIntArithmeticOp,
    IrIntCompareOp, IrLargeArena, IrRegisterInfo, IrSignal, IrStatement, IrVariableInfo,
};
use crate::syntax::ast::{
    Arg, Args, ArrayComprehension, ArrayLiteralElement, BinaryOp, BlockExpression, DomainKind, DotIndexKind,
    Expression, ExpressionKind, GeneralIdentifier, Identifier, IntLiteral, MaybeIdentifier, PortDirection,
    RangeLiteral, RegisterDelay, SyncDomain, UnaryOp,
};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::throw;
use crate::util::big_int::{AnyInt, BigInt, BigUint};
use crate::util::data::{VecExt, vec_concat};

use crate::front::exit::ExitStack;
use crate::front::flow::{Flow, HardwareProcessKind};
use crate::front::module::ExtraRegisterInit;
use crate::front::range_arithmetic::{
    multi_range_binary_add, multi_range_binary_div, multi_range_binary_mod, multi_range_binary_mul,
    multi_range_binary_pow, multi_range_binary_sub, multi_range_unary_neg,
};
use crate::util::iter::IterExt;
use crate::util::range::{ClosedNonEmptyRange, NonEmptyRange, Range};
use crate::util::range_multi::{AnyMultiRange, ClosedNonEmptyMultiRange, MultiRange};
use crate::util::store::ArcOrRef;
use crate::util::{ResultDoubleExt, result_pair};
use itertools::Either;
use std::sync::Arc;
use unwrap_match::unwrap_match;

// TODO better name
#[derive(Debug)]
pub enum ValueInner {
    Value(ValueWithImplications),
    PortInterface(PortInterface),
    WireInterface(WireInterface),
}

// TODO rename/remove
#[derive(Debug)]
pub enum NamedOrValue {
    Named(NamedValue),
    ItemValue(CompileValue),
}

impl<'a> CompileItemContext<'a, '_> {
    pub fn eval_general_id(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        id: GeneralIdentifier,
    ) -> DiagResult<Spanned<ArcOrRef<'a, str>>> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        match id {
            GeneralIdentifier::Simple(id) => Ok(id.spanned_str(self.refs.fixed.source).map_inner(ArcOrRef::Ref)),
            GeneralIdentifier::FromString(span, expr) => {
                let value =
                    self.eval_expression_as_compile(scope, flow, &Type::String, expr, Spanned::new(span, "id string"))?;
                let value = check_type_is_string_compile(diags, elab, TypeContainsReason::Operator(span), value)?;

                Ok(Spanned::new(id.span(), ArcOrRef::Arc(value)))
            }
        }
    }

    pub fn eval_maybe_general_id(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        id: MaybeIdentifier<GeneralIdentifier>,
    ) -> DiagResult<MaybeIdentifier<Spanned<ArcOrRef<'a, str>>>> {
        match id {
            MaybeIdentifier::Dummy { span } => Ok(MaybeIdentifier::Dummy { span }),
            MaybeIdentifier::Identifier(id) => {
                let id = self.eval_general_id(scope, flow, id)?;
                Ok(MaybeIdentifier::Identifier(id))
            }
        }
    }

    // TODO remove, everyone calling this should call scope.find directly
    pub fn eval_named_or_value(&mut self, scope: &Scope, id: Spanned<&str>) -> DiagResult<Spanned<NamedOrValue>> {
        let diags = self.refs.diags;

        let found = scope.find(diags, id)?;
        let def_span = found.defining_span;
        let result = match found.value {
            ScopedEntry::Named(value) => NamedOrValue::Named(value),
            ScopedEntry::Item(item) => {
                // TODO do we need to push a stack entry here? maybe it's better to move that into eval_item,
                //   so no once can accidentally forget
                let entry = StackEntry::ItemUsage(id.span);
                let value = self.recurse(entry, |s| Ok(s.eval_item(item)?.clone())).flatten_err()?;
                NamedOrValue::ItemValue(value)
            }
        };
        Ok(Spanned {
            span: def_span,
            inner: result,
        })
    }

    pub fn eval_expression(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr: Expression,
    ) -> DiagResult<Spanned<Value>> {
        Ok(self
            .eval_expression_with_implications(scope, flow, expected_ty, expr)?
            .map_inner(ValueWithImplications::into_value))
    }

    pub fn eval_expression_with_implications(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr: Expression,
    ) -> DiagResult<Spanned<ValueWithImplications>> {
        let value = self.eval_expression_inner(scope, flow, expected_ty, expr)?;
        match value {
            ValueInner::Value(v) => Ok(Spanned::new(expr.span, v)),
            ValueInner::PortInterface(_) | ValueInner::WireInterface(_) => Err(self.refs.diags.report_error_simple(
                "interface instance expression not allowed here",
                expr.span,
                "this expression evaluates to an interface instance",
            )),
        }
    }

    // TODO return COW to save some allocations?
    // TODO maybe this should return an abstract expression value,
    //   that can then be written (as target), read (as value), typeof-ed, gotten implications, ...
    //   that's awkward for expressions that create statements though, eg. calls
    //   maybe those should push their statements to virtual blocks, and only actually add them once read?
    // TODO rename the other one to `eval_expression_value`, and this one to `eval_expression`
    pub fn eval_expression_inner(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr: Expression,
    ) -> DiagResult<ValueInner> {
        let refs = self.refs;
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;
        let elab = &refs.shared.elaboration_arenas;

        let result_simple = match refs.get_expr(expr) {
            &ExpressionKind::ParseError(_) => {
                return Err(diags.report_error_internal(expr.span, "cannot evaluate parse error"));
            }
            ExpressionKind::Dummy => {
                // if dummy expressions were allowed, the caller would have checked for them already
                return Err(diags.report_error_simple(
                    "dummy expression not allowed in this context",
                    expr.span,
                    "dummy expression used here",
                ));
            }
            ExpressionKind::Undefined => {
                flow.require_hardware(expr.span, "undefined expression")?;

                let expected_ty_hw = expected_ty.as_hardware_type(elab).map_err(|_: NonHardwareType| {
                    diags.report_error_simple(
                        "undefined expression requires hardware type",
                        expr.span,
                        format!("inferred type is `{}`", expected_ty.value_string(elab)),
                    )
                })?;

                let expr = self
                    .large
                    .push_expr(IrExpressionLarge::Undefined(expected_ty_hw.as_ir(refs)));
                Value::Hardware(HardwareValue {
                    ty: expected_ty_hw,
                    domain: ValueDomain::CompileTime,
                    expr,
                })
            }
            ExpressionKind::Type => Value::new_ty(Type::Type),
            ExpressionKind::TypeFunction => Value::new_ty(Type::Function),
            ExpressionKind::Builtin => Value::new_ty(Type::Builtin),
            &ExpressionKind::Wrapped(inner) => {
                return self.eval_expression_inner(scope, flow, expected_ty, inner);
            }
            ExpressionKind::Block(block_expr) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = block_expr;

                let mut scope_inner = Scope::new_child(expr.span, scope);

                let mut stack = ExitStack::new_in_block_expression(expr.span);
                let end = self.elaborate_block_statements(&mut scope_inner, flow, &mut stack, statements)?;
                end.unwrap_normal(diags, expr.span)?;

                self.eval_expression(&scope_inner, flow, expected_ty, expression)?.inner
            }
            &ExpressionKind::Id(id) => {
                let id = self.eval_general_id(scope, flow, id)?;
                let id = id.as_ref().map_inner(ArcOrRef::as_ref);

                let result = match self.eval_named_or_value(scope, id)?.inner {
                    NamedOrValue::ItemValue(value) => {
                        return Ok(ValueInner::Value(ValueWithImplications::simple(Value::from(value))));
                    }
                    NamedOrValue::Named(value) => match value {
                        NamedValue::Variable(var) => {
                            flow.var_eval(refs, &mut self.large, Spanned::new(expr.span, var))?
                        }
                        NamedValue::Port(port) => {
                            flow.signal_eval(self, Spanned::new(expr.span, Signal::Port(port)))?
                        }
                        NamedValue::Wire(wire) => {
                            flow.signal_eval(self, Spanned::new(expr.span, Signal::Wire(wire)))?
                        }
                        NamedValue::Register(reg) => {
                            if let FlowKind::Hardware(flow) = flow.kind_mut() {
                                if let HardwareProcessKind::ClockedBlockBody { domain, .. } = flow.block_kind() {
                                    let reg_info = &mut self.registers[reg];
                                    reg_info.suggest_domain(Spanned::new(expr.span, domain.inner));
                                }
                            }

                            flow.signal_eval(self, Spanned::new(expr.span, Signal::Register(reg)))?
                        }
                        NamedValue::PortInterface(interface) => {
                            // we don't need a hardware context yet, only when we actually access an actual port
                            return Ok(ValueInner::PortInterface(interface));
                        }
                        NamedValue::WireInterface(interface) => {
                            // we don't need a hardware context yet, only when we actually access an actual port
                            return Ok(ValueInner::WireInterface(interface));
                        }
                    },
                };
                return Ok(ValueInner::Value(result));
            }
            ExpressionKind::IntLiteral(pattern) => {
                // TODO is there a way to move this parsing into the tokenizer? at least move this code there,
                //   the risk of divergence is huge
                let value = match *pattern {
                    IntLiteral::Binary { span } => {
                        let raw = source.span_str(span);
                        let clean = raw[2..].replace('_', "");
                        BigUint::from_str_radix(&clean, 2)
                            .map_err(|_| diags.report_error_internal(expr.span, "failed to parse int"))?
                    }
                    IntLiteral::Decimal { span } => {
                        let raw = source.span_str(span);
                        let clean = raw.replace('_', "");
                        BigUint::from_str_radix(&clean, 10)
                            .map_err(|_| diags.report_error_internal(expr.span, "failed to parse int"))?
                    }
                    IntLiteral::Hexadecimal { span } => {
                        let raw = source.span_str(span);
                        let s_hex = raw[2..].replace('_', "");
                        BigUint::from_str_radix(&s_hex, 16)
                            .map_err(|_| diags.report_error_internal(expr.span, "failed to parse int"))?
                    }
                };
                Value::new_int(BigInt::from(value))
            }
            &ExpressionKind::BoolLiteral(literal) => Value::new_bool(literal),
            ExpressionKind::StringLiteral(pieces) => self.eval_string_literal(scope, flow, pieces)?,
            ExpressionKind::ArrayLiteral(values) => {
                self.eval_array_literal(scope, flow, expected_ty, expr.span, values)?
            }
            ExpressionKind::TupleLiteral(values) => self.eval_tuple_literal(scope, flow, expected_ty, values)?,
            ExpressionKind::RangeLiteral(literal) => {
                let mut eval_bound = |bound: Expression, op_span: Span| {
                    let bound_span = bound.span;
                    let bound = self.eval_expression(scope, flow, &Type::Int(MultiRange::open()), bound)?;

                    let reason = TypeContainsReason::Operator(op_span);
                    let bound = check_type_is_int(diags, elab, reason, bound)?;

                    Ok(Spanned::new(bound_span, bound))
                };

                match *literal {
                    RangeLiteral::ExclusiveEnd {
                        op_span,
                        ref start,
                        ref end,
                    } => {
                        let start = start.map(|start| eval_bound(start, op_span)).transpose();
                        let end = end.map(|end| eval_bound(end, op_span)).transpose();

                        let start = start?;
                        let end = end?;

                        // check validness
                        if let (Some(start), Some(end)) = (&start, &end) {
                            let start_range = start.inner.range();
                            let end_range = end.inner.range();

                            #[allow(clippy::nonminimal_bool)]
                            if !(start_range.enclosing_range().end - 1 <= *end_range.enclosing_range().start) {
                                let msg_start = message_range_or_single("start", &start_range, None);
                                let msg_end = message_range_or_single("end", &end_range, None);
                                let diag = DiagnosticError::new(
                                    "range requires that start <= end",
                                    op_span,
                                    "range constructed here",
                                )
                                .add_info(start.span, msg_start)
                                .add_info(end.span, msg_end)
                                .report(diags);
                                return Err(diag);
                            }
                        }

                        let start = start.map(|s| s.inner);
                        let end = end.map(|e| e.inner);
                        let range = RangeValue::Normal(Range { start, end });
                        Value::Compound(MixedCompoundValue::Range(range))
                    }
                    RangeLiteral::InclusiveEnd {
                        op_span,
                        start,
                        end_inc,
                    } => {
                        // eval bounds
                        let start = start.map(|start| eval_bound(start, op_span)).transpose();
                        let end_inc = eval_bound(end_inc, op_span);

                        let start = start?;
                        let end_inc = end_inc?;

                        // check validness
                        if let Some(start) = &start {
                            let start_range = start.inner.range();
                            let end_inc_range = end_inc.inner.range();

                            #[allow(clippy::nonminimal_bool)]
                            if !(start_range.enclosing_range().end - 1 <= end_inc_range.enclosing_range().start + 1) {
                                let msg_start = message_range_or_single("start", &start_range, None);
                                let msg_end = message_range_or_single("end", &end_inc_range, None);
                                let diag = DiagnosticError::new(
                                    "inclusive range requires that start <= end_inc + 1",
                                    op_span,
                                    "inclusive range constructed here",
                                )
                                .add_info(start.span, msg_start)
                                .add_info(end_inc.span, msg_end)
                                .report(diags);
                                return Err(diag);
                            }
                        }

                        // map end to exclusive
                        let end = match end_inc.inner {
                            MaybeCompile::Compile(end_inc) => MaybeCompile::Compile(end_inc + 1),
                            MaybeCompile::Hardware(end_inc) => {
                                let end_range = end_inc.ty.map(|x| x + 1);

                                let end_expr = IrExpressionLarge::IntArithmetic(
                                    IrIntArithmeticOp::Add,
                                    end_range.enclosing_range().cloned(),
                                    end_inc.expr,
                                    IrExpression::Int(BigInt::ONE),
                                );
                                MaybeCompile::Hardware(HardwareInt {
                                    ty: end_range,
                                    domain: end_inc.domain,
                                    expr: self.large.push_expr(end_expr),
                                })
                            }
                        };

                        let start = start.map(|s| s.inner);
                        let range = RangeValue::Normal(Range { start, end: Some(end) });
                        Value::Compound(MixedCompoundValue::Range(range))
                    }
                    RangeLiteral::Length { op_span, start, length } => {
                        let start = eval_bound(start, op_span);

                        let range_uint = MultiRange::from(Range {
                            start: Some(BigInt::ZERO),
                            end: None,
                        });
                        let length = self
                            .eval_expression(scope, flow, &Type::Int(range_uint), length)
                            .and_then(|length| {
                                let reason = TypeContainsReason::Operator(op_span);
                                check_type_is_uint(diags, elab, reason, length)
                            });

                        let start = start?;
                        let length = length?;

                        let range = match (start.inner, length) {
                            (MaybeCompile::Hardware(start), MaybeCompile::Compile(length)) => {
                                // preserve full information to allow for hardware slicing
                                RangeValue::HardwareStartLength { start, length }
                            }
                            (start, length) => {
                                // decay to normal range by calculating `end = start + length`
                                let length = match length {
                                    MaybeCompile::Compile(length) => MaybeCompile::Compile(BigInt::from(length)),
                                    MaybeCompile::Hardware(length) => MaybeCompile::Hardware(HardwareInt {
                                        ty: length.ty.map(BigInt::from),
                                        domain: length.domain,
                                        expr: length.expr,
                                    }),
                                };
                                let end = match pair_compile_int(start.clone(), length) {
                                    MaybeCompile::Compile((start, length)) => MaybeCompile::Compile(start + length),
                                    MaybeCompile::Hardware((start, length)) => {
                                        let range = multi_range_binary_add(&start.ty, &length.ty);
                                        let result = build_binary_int_arithmetic_op(
                                            IrIntArithmeticOp::Add,
                                            &mut self.large,
                                            range,
                                            start,
                                            length,
                                        );
                                        MaybeCompile::Hardware(result)
                                    }
                                };

                                // length is unsigned, so end >= start always holds
                                RangeValue::Normal(Range {
                                    start: Some(start),
                                    end: Some(end),
                                })
                            }
                        };

                        Value::Compound(MixedCompoundValue::Range(range))
                    }
                }
            }
            ExpressionKind::ArrayComprehension(array_comprehension) => {
                let &ArrayComprehension {
                    body,
                    index,
                    span_keyword,
                    iter,
                } = array_comprehension;

                let expected_ty_inner = match expected_ty {
                    Type::Array(inner, _) => inner,
                    _ => &Type::Any,
                };

                let iter_eval = self.eval_expression_as_for_iterator(scope, flow, iter)?;

                // this is only a lower bound,
                //   there might be spread elements in the literal which expand to multiple values
                let len_lower_bound = match iter_eval.len() {
                    None => {
                        return Err(diags.report_error_simple(
                            "array comprehension over infinite iterator would never finish",
                            iter.span,
                            "this iterator is infinite",
                        ));
                    }
                    Some(len) => usize::try_from(len).unwrap_or(0),
                };

                let mut values = Vec::with_capacity(len_lower_bound);

                for index_value in iter_eval {
                    self.refs.check_should_stop(expr.span)?;

                    let index_value = index_value.map_hardware(|h| h.map_expression(|h| self.large.push_expr(h)));
                    let index_var = flow.var_new_immutable_init(
                        refs,
                        index.span(),
                        VariableId::Id(index),
                        span_keyword,
                        Ok(ValueWithImplications::simple(index_value)),
                    )?;

                    let scope_span = body.span().join(index.span());
                    let mut scope_body = Scope::new_child(scope_span, scope);
                    scope_body.maybe_declare(
                        diags,
                        Ok(index.spanned_str(source)),
                        Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
                    );

                    let value = body
                        .map_inner(|body_expr| self.eval_expression(&scope_body, flow, expected_ty_inner, body_expr))
                        .transpose()?;
                    values.push(value);
                }

                array_literal_combine_values(refs, flow, &mut self.large, expr.span, values)?
            }

            &ExpressionKind::UnaryOp(op, operand) => match op.inner {
                UnaryOp::Plus => {
                    let operand = self.eval_expression_with_implications(scope, flow, &Type::Any, operand)?;
                    let _ = check_type_is_int(
                        diags,
                        elab,
                        TypeContainsReason::Operator(op.span),
                        operand.clone().map_inner(ValueWithImplications::into_value),
                    )?;
                    return Ok(ValueInner::Value(operand.inner));
                }
                UnaryOp::Neg => {
                    let operand = self.eval_expression(scope, flow, &Type::Any, operand)?;
                    let operand_int = check_type_is_int(diags, elab, TypeContainsReason::Operator(op.span), operand)?;

                    match operand_int {
                        MaybeCompile::Compile(c) => Value::new_int(-c),
                        MaybeCompile::Hardware(v) => {
                            let range = multi_range_unary_neg(&v.ty);
                            let result_expr = self.large.push_expr(IrExpressionLarge::IntArithmetic(
                                IrIntArithmeticOp::Sub,
                                range.enclosing_range().cloned(),
                                IrExpression::Int(BigInt::ZERO),
                                v.expr,
                            ));

                            let result = HardwareValue {
                                ty: HardwareType::Int(range),
                                domain: v.domain,
                                expr: result_expr,
                            };
                            Value::Hardware(result)
                        }
                    }
                }
                UnaryOp::Not => {
                    let operand = self.eval_expression_with_implications(scope, flow, &Type::Any, operand)?;
                    let operand_bool = check_type_is_bool(diags, elab, TypeContainsReason::Operator(op.span), operand)?;

                    match operand_bool {
                        MaybeCompile::Compile(c) => Value::new_bool(!c),
                        MaybeCompile::Hardware(v) => {
                            let result = HardwareValue {
                                ty: HardwareType::Bool,
                                domain: v.value.domain,
                                expr: self.large.push_expr(IrExpressionLarge::BoolNot(v.value.expr)),
                            };
                            let result_with_implications = HardwareValueWithImplications {
                                value: result,
                                version: None,
                                implications: v.implications.invert(),
                            };
                            return Ok(ValueInner::Value(Value::Hardware(result_with_implications)));
                        }
                    }
                }
            },
            &ExpressionKind::BinaryOp(op, left, right) => {
                let left = self.eval_expression_with_implications(scope, flow, &Type::Any, left);
                let right = self.eval_expression_with_implications(scope, flow, &Type::Any, right);
                let result = eval_binary_expression(refs, &mut self.large, expr.span, op, left?, right?)?;
                return Ok(ValueInner::Value(result));
            }
            &ExpressionKind::ArrayIndex {
                span_brackets,
                base,
                ref indices,
            } => {
                // eval base
                let base = self.eval_expression(scope, flow, &Type::Any, base)?;

                // TODO support nd arrays
                // evaluate index
                let index = indices
                    .single_ref()
                    .ok_or_else(|| diags.report_error_todo(expr.span, "multidimensional array indexing"))?;
                let index = match index {
                    &ArrayLiteralElement::Single(index) => index,
                    ArrayLiteralElement::Spread(_, _) => {
                        return Err(diags.report_error_todo(index.span(), "multidimensional array indexing"));
                    }
                };

                let index_step = self.eval_expression_as_array_step(scope, flow, index)?;
                let steps = ArraySteps::new(vec![Spanned::new(span_brackets, index_step)]);

                // apply steps to value
                steps.apply_to_value(self.refs, &mut self.large, base)?
            }
            &ExpressionKind::ArrayType {
                span_brackets: _,
                ref lengths,
                inner_ty,
            } => {
                // TODO add syntax for "any-length" lists, maybe `[_]inner`, that also scales nicely?
                // TODO support nd arrays
                // evaluate length
                let length = lengths
                    .single_ref()
                    .ok_or_else(|| diags.report_error_todo(expr.span, "multidimensional arrays"))?;
                let length = match length {
                    &ArrayLiteralElement::Single(length) => length,
                    ArrayLiteralElement::Spread(_, _) => {
                        return Err(diags.report_error_todo(length.span(), "multidimensional arrays"));
                    }
                };

                let length = match refs.get_expr(length) {
                    ExpressionKind::Dummy => None,
                    _ => {
                        let ty_uint = Type::Int(MultiRange::from(Range {
                            start: Some(BigInt::ZERO),
                            end: None,
                        }));
                        let length = self.eval_expression_as_compile(
                            scope,
                            flow,
                            &ty_uint,
                            length,
                            Spanned::new(expr.span, "array type length"),
                        )?;
                        let reason = TypeContainsReason::ArrayLen { span_len: length.span };
                        let length = check_type_is_uint_compile(diags, elab, reason, length)?;
                        Some(length)
                    }
                };

                // eval inner type and construct new array type
                let inner_ty = self.eval_expression_as_ty(scope, flow, inner_ty)?;
                Value::new_ty(Type::Array(Arc::new(inner_ty.inner), length))
            }
            &ExpressionKind::DotIndex(base, index) => match index {
                DotIndexKind::Id(index) => {
                    return self.eval_dot_id_index(scope, flow, expected_ty, expr.span, base, index);
                }
                DotIndexKind::Int { span: index_span } => {
                    let base_eval = self.eval_expression_inner(scope, flow, &Type::Any, base)?;
                    let index = source.span_str(index_span);

                    let index_int = BigUint::from_str_radix(index, 10)
                        .map_err(|_| diags.report_error_internal(expr.span, "failed to parse int"))?;
                    let err_not_tuple = |ty: &str| {
                        DiagnosticError::new(
                            "indexing into non-tuple type",
                            index_span,
                            "attempt to index into non-tuple type here",
                        )
                        .add_info(base.span, format!("base has type `{ty}`"))
                        .report(diags)
                    };
                    let err_index_out_of_bounds = |len: usize| {
                        DiagnosticError::new(
                            "tuple index out of bounds",
                            index_span,
                            format!("index `{index_int}` is out of bounds"),
                        )
                        .add_info(base.span, format!("base is tuple with length `{len}`"))
                        .report(diags)
                    };

                    // TODO use common step logic once that supports tuple indexing
                    match base_eval {
                        ValueInner::Value(Value::Compound(MixedCompoundValue::Tuple(inner))) => {
                            let index = index_int
                                .as_usize_if_lt(inner.len())
                                .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                            inner[index].clone()
                        }
                        ValueInner::Value(Value::Simple(SimpleCompileValue::Type(Type::Tuple(Some(inner))))) => {
                            let index = index_int
                                .as_usize_if_lt(inner.len())
                                .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                            Value::new_ty(inner[index].clone())
                        }
                        ValueInner::Value(Value::Hardware(value)) => {
                            let value = value.value;
                            match value.ty {
                                HardwareType::Tuple(inner_tys) => {
                                    let index = index_int
                                        .as_usize_if_lt(inner_tys.len())
                                        .ok_or_else(|| err_index_out_of_bounds(inner_tys.len()))?;

                                    let expr = IrExpressionLarge::TupleIndex {
                                        base: value.expr,
                                        index,
                                    };
                                    Value::Hardware(HardwareValue {
                                        ty: inner_tys[index].clone(),
                                        domain: value.domain,
                                        expr: self.large.push_expr(expr),
                                    })
                                }
                                _ => return Err(err_not_tuple(&value.ty.value_string(elab))),
                            }
                        }
                        ValueInner::Value(v) => return Err(err_not_tuple(&v.ty().value_string(elab))),
                        ValueInner::PortInterface(_) | ValueInner::WireInterface(_) => {
                            return Err(err_not_tuple("interface instance"));
                        }
                    }
                }
            },
            &ExpressionKind::Call(target, ref args) => {
                // eval target
                let target = self.eval_expression_as_compile(
                    scope,
                    flow,
                    &Type::Any,
                    target,
                    Spanned::new(expr.span, "call target"),
                )?;

                // handle special cases that can't immediately evaluate their arguments
                match target.inner {
                    Value::Simple(SimpleCompileValue::Type(Type::Builtin)) => {
                        // builtin
                        let value = self.eval_builtin(scope, flow, expr.span, target.span, args)?;
                        return Ok(ValueInner::Value(ValueWithImplications::simple(value)));
                    }
                    Value::Simple(SimpleCompileValue::Type(Type::Type)) => {
                        // typeof operator
                        let ty = self.eval_type_of(scope, flow, expr.span, args)?;
                        return Ok(ValueInner::Value(ValueWithImplications::new_ty(ty)));
                    }
                    _ => {
                        // fallthrough into normal call logic
                    }
                }

                // eval args
                let args_eval = args
                    .inner
                    .iter()
                    .map(|arg| {
                        // TODO pass an actual expected type in cases where we know it (eg. struct/enum construction)
                        let arg_value = self.eval_expression_with_implications(scope, flow, &Type::Any, arg.value)?;

                        Ok(Arg {
                            span: arg.span,
                            name: arg.name.map(|id| Spanned::new(id.span, id.str(source))),
                            value: arg_value,
                        })
                    })
                    .try_collect_all_vec();
                let args = args_eval.map(|inner| Args { span: args.span, inner });

                self.eval_call(flow, expected_ty, expr.span, target.as_ref(), args)?
            }
            &ExpressionKind::UnsafeValueWithDomain(value, domain) => {
                // evaluate value and domain
                let flow_hw = flow.require_hardware(expr.span, "domain cast")?;

                let value = {
                    // evaluate the value with disabled domain checks
                    // (the whole point of this operation is to bypass domain checking)
                    // TODO is this not accidentally disabling eg. writes to unrelated signals too?
                    let mut flow_hw_inner = flow_hw.new_child_scoped_without_domain_checks();
                    self.eval_expression(scope, &mut flow_hw_inner, expected_ty, value)
                };

                let mut flow_domain = flow.new_child_compile(domain.span, "domain");
                let domain = self.eval_domain(scope, &mut flow_domain, domain);

                let value = value?;
                let domain = domain?;

                // convert value to hardware
                let ty = match value.inner.ty().as_hardware_type(elab) {
                    Ok(ty) => ty,
                    Err(NonHardwareType) => {
                        let diag = DiagnosticError::new(
                            "value must be representable in hardware for domain cast",
                            value.span,
                            format!("got non-hardware type `{}`", value.inner.ty().value_string(elab)),
                        )
                        .report(diags);
                        return Err(diag);
                    }
                };
                let value_expr = value
                    .inner
                    .as_ir_expression_unchecked(refs, &mut self.large, value.span, &ty)?;

                // set domain
                Value::Hardware(HardwareValue {
                    ty,
                    domain: ValueDomain::from_domain_kind(domain.inner),
                    expr: value_expr,
                })
            }
            ExpressionKind::RegisterDelay(reg_delay) => {
                let &RegisterDelay {
                    span_keyword,
                    value,
                    init,
                } = reg_delay;

                // eval
                let value = self.eval_expression(scope, flow, expected_ty, value);
                let init = self.eval_expression_as_compile_or_undefined(
                    scope,
                    flow,
                    expected_ty,
                    init,
                    Spanned::new(span_keyword, "register init"),
                );
                let (value, init) = result_pair(value, init)?;

                // figure out type
                let value_ty = value.inner.ty();
                let init_ty = init.inner.ty();
                let ty = value_ty.union(&init_ty);
                let ty_hw = ty.as_hardware_type(elab).map_err(|_| {
                    DiagnosticError::new(
                        "register type must be representable in hardware",
                        span_keyword,
                        format!("got non-hardware type `{}`", ty.value_string(elab)),
                    )
                    .add_info(value.span, format!("from combining `{}`", value_ty.value_string(elab)))
                    .add_info(init.span, format!("from combining `{}`", init_ty.value_string(elab)))
                    .report(diags)
                })?;

                let flow = flow.require_hardware(expr.span, "register expression")?;

                // convert values to hardware
                let value =
                    value
                        .inner
                        .as_hardware_value_unchecked(refs, &mut self.large, value.span, ty_hw.clone())?;
                let init_span = init.span;
                let init = init
                    .map_inner(|inner| match inner {
                        MaybeUndefined::Undefined => Ok(MaybeUndefined::Undefined),
                        MaybeUndefined::Defined(inner) => Ok(MaybeUndefined::Defined(
                            inner.as_ir_expression_unchecked(refs, &mut self.large, init_span, &ty_hw)?,
                        )),
                    })
                    .transpose()?;

                // create variable to hold the result
                let var_info = IrVariableInfo {
                    ty: ty_hw.as_ir(refs),
                    debug_info_span: span_keyword,
                    debug_info_id: Some("reg_delay".to_owned()),
                };
                let ir_var = flow.new_ir_variable(var_info);

                // create register to act as the delay storage
                let (clocked_domain, clocked_registers, ir_registers) =
                    flow.check_clocked_block(span_keyword, "register expression")?;
                let reg_info = IrRegisterInfo {
                    ty: ty_hw.as_ir(refs),
                    debug_info_id: Spanned::new(span_keyword, None),
                    debug_info_ty: ty_hw.clone().value_string(elab),
                    debug_info_domain: clocked_domain.inner.diagnostic_string(self),
                };
                let reg = ir_registers.push(reg_info);

                // record the reset value if any
                match clocked_registers {
                    ExtraRegisters::NoReset => match init.inner {
                        MaybeUndefined::Undefined => {
                            // we can't reset, but luckily we don't need to
                        }
                        MaybeUndefined::Defined(_) => {
                            let _ = DiagnosticError::new(
                                "registers without reset cannot have an initial value",
                                init.span,
                                "attempt to create a register with init here",
                            )
                            .add_info(
                                clocked_domain.span,
                                "the current clocked block is defined without reset here",
                            )
                            .add_footer_hint("either add a reset to the block or use `undef` as the initial value")
                            .report(diags);
                        }
                    },
                    ExtraRegisters::WithReset(extra_registers) => {
                        match init.inner {
                            MaybeUndefined::Undefined => {
                                // we can reset but we don't need to
                            }
                            MaybeUndefined::Defined(init_inner) => {
                                extra_registers.push(ExtraRegisterInit {
                                    span: init.span,
                                    reg,
                                    init: init_inner,
                                });
                            }
                        }
                    }
                }

                // do the right shuffle operations
                let stmt_load = IrStatement::Assign(
                    IrAssignmentTarget::simple(ir_var.into()),
                    IrExpression::Signal(IrSignal::Register(reg)),
                );
                flow.push_ir_statement(Spanned::new(span_keyword, stmt_load));
                let stmt_store = IrStatement::Assign(IrAssignmentTarget::simple(reg.into()), value.expr);
                flow.push_ir_statement(Spanned::new(span_keyword, stmt_store));

                // return the variable, now containing the previous value of the register
                Value::Hardware(HardwareValue {
                    ty: ty_hw,
                    domain: ValueDomain::Sync(clocked_domain.inner),
                    expr: IrExpression::Variable(ir_var),
                })
            }
        };

        Ok(ValueInner::Value(ValueWithImplications::simple(result_simple)))
    }

    pub fn eval_call(
        &mut self,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr_span: Span,
        target: Spanned<&CompileValue>,
        args: DiagResult<Args<Option<Spanned<&str>>, Spanned<ValueWithImplications>>>,
    ) -> DiagResult<Value> {
        let refs = self.refs;
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        match target.inner {
            Value::Simple(SimpleCompileValue::Function(target_function)) => {
                // normal function call
                // TODO should we do the recursion marker here or inside of the call function?
                let entry = StackEntry::FunctionCall(expr_span);
                self.recurse(entry, |s| {
                    s.call_function(flow, expected_ty, target.span, expr_span, target_function, args?)
                })
                .flatten_err()
            }

            // handle some special type calls
            Value::Simple(SimpleCompileValue::Type(Type::Int(range))) => {
                let result = eval_int_ty_call(refs, expr_span, Spanned::new(target.span, range), args?)?;
                Ok(Value::new_ty(result))
            }
            Value::Simple(SimpleCompileValue::Type(Type::Tuple(None))) => {
                let result = eval_tuple_ty_call(refs, args?)?;
                Ok(Value::new_ty(result))
            }

            // not a valid call target
            _ => Err(diags.report_error_simple(
                "call target must be function",
                expr_span,
                format!("got value with type `{}`", target.inner.ty().value_string(elab)),
            )),
        }
    }

    fn eval_dot_id_index(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr_span: Span,
        base: Expression,
        index: Identifier,
    ) -> DiagResult<ValueInner> {
        // TODO make sure users don't accidentally define fields/variants/functions with the same name
        // TODO add array.len, type.int_start, type.int_end, type.int_ranges

        let refs = self.refs;
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let base_eval = self.eval_expression_inner(scope, flow, &Type::Any, base)?;
        let index_str = index.str(refs.fixed.source);

        // interface fields
        let base_eval = match base_eval {
            ValueInner::PortInterface(port_interface) => {
                // get the underlying port
                let port_interface_info = &self.port_interfaces[port_interface];
                let interface_info = self
                    .refs
                    .shared
                    .elaboration_arenas
                    .interface_info(port_interface_info.view.inner.interface);

                let port_index = interface_info.ports.get_index_of(index_str).ok_or_else(|| {
                    DiagnosticError::new(
                        format!("port `{index_str}` not found on interface"),
                        index.span,
                        "attempt to access port here",
                    )
                    .add_info(port_interface_info.view.span, "port interface set here")
                    .add_info(interface_info.id.span(), "interface declared here")
                    .report(diags)
                })?;
                let port = port_interface_info.ports[port_index];

                let flow = flow.require_hardware(expr_span, "port access")?;
                let port_eval = flow.signal_eval(self, Spanned::new(expr_span, Signal::Port(port)))?;
                return Ok(ValueInner::Value(port_eval));
            }
            ValueInner::WireInterface(wire_interface) => {
                let wire_interface_info = &self.wire_interfaces[wire_interface];
                let interface_info = self
                    .refs
                    .shared
                    .elaboration_arenas
                    .interface_info(wire_interface_info.interface.inner);

                let wire_index = interface_info.ports.get_index_of(index_str).ok_or_else(|| {
                    DiagnosticError::new(
                        format!("port `{index_str}` not found on interface"),
                        index.span,
                        "attempt to access port here",
                    )
                    .add_info(wire_interface_info.interface.span, "wire interface set here")
                    .add_info(interface_info.id.span(), "interface declared here")
                    .report(diags)
                })?;
                let wire = wire_interface_info.wires[wire_index];

                let flow = flow.require_hardware(expr_span, "wire access")?;
                let wire_eval = flow.signal_eval(self, Spanned::new(expr_span, Signal::Wire(wire)))?;
                return Ok(ValueInner::Value(wire_eval));
            }
            ValueInner::Value(base_eval) => base_eval,
        };

        // interface views
        if let &Value::Simple(SimpleCompileValue::Interface(base_interface)) = &base_eval {
            let info = self.refs.shared.elaboration_arenas.interface_info(base_interface);
            let (view_index, _) = info.get_view(diags, self.refs.fixed.source, index)?;

            let interface_view = ElaboratedInterfaceView {
                interface: base_interface,
                view_index,
            };
            let result = Value::Simple(SimpleCompileValue::InterfaceView(interface_view));
            return Ok(ValueInner::Value(result));
        }

        // common type attributes
        if let Value::Simple(SimpleCompileValue::Type(ty)) = &base_eval {
            match index_str {
                "size_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, elab, Spanned::new(base.span, ty))?;
                    let width = ty_hw.as_ir(refs).size_bits();
                    let result = Value::new_int(width.into());
                    return Ok(ValueInner::Value(result));
                }
                // TODO all of these should return functions with a single params,
                //   without the need for scope capturing
                "to_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, elab, Spanned::new(base.span, ty))?;
                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::ToBits,
                    };
                    let result = Value::Simple(SimpleCompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                "from_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, elab, Spanned::new(base.span, ty))?;

                    if !ty_hw.every_bit_pattern_is_valid(refs) {
                        let diag = DiagnosticError::new(
                            "from_bits is only allowed for types where every bit pattern is valid",
                            base.span,
                            format!("got type `{}` with invalid bit patterns", ty_hw.value_string(elab)),
                        )
                        .add_footer_hint("consider using a target type where every bit pattern is valid")
                        .add_footer_hint("if you know the bits are valid for this type, use `from_bits_unsafe` instead")
                        .report(diags);
                        return Err(diag);
                    }

                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::FromBits { is_unsafe: false },
                    };
                    let result = Value::Simple(SimpleCompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                "from_bits_unsafe" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, elab, Spanned::new(base.span, ty))?;
                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::FromBits { is_unsafe: true },
                    };
                    let result = Value::Simple(SimpleCompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                _ => {}
            }

            // array type attributes
            if let Type::Array(ty_inner, ty_len) = ty {
                match index_str {
                    "inner" => {
                        let result = Value::new_ty((**ty_inner).clone());
                        return Ok(ValueInner::Value(result));
                    }
                    "len" => {
                        return if let Some(ty_len) = ty_len {
                            let result = Value::new_int(BigInt::from(ty_len));
                            Ok(ValueInner::Value(result))
                        } else {
                            let diag = DiagnosticError::new(
                                "cannot get length of array type with unknown length",
                                index.span,
                                "trying to get length here",
                            )
                            .add_info(base.span, format!("array type is `{}`", ty.value_string(elab)))
                            .report(diags);
                            Err(diag)
                        };
                    }
                    _ => {}
                }
            }
        }

        // struct new
        if let &Value::Simple(SimpleCompileValue::Type(Type::Struct(elab))) = &base_eval
            && index_str == "new"
        {
            let result = Value::Simple(SimpleCompileValue::Function(FunctionValue::StructNew(elab)));
            return Ok(ValueInner::Value(result));
        }

        let base_item_function = match &base_eval {
            Value::Simple(SimpleCompileValue::Function(FunctionValue::User(func))) => match &func.body.inner {
                FunctionBody::ItemBody(body) => Some(body),
                _ => None,
            },
            _ => None,
        };
        if let Some(&FunctionItemBody::Struct(unique, _)) = base_item_function
            && index_str == "new"
        {
            let func = FunctionValue::StructNewInfer(unique);
            let result = Value::Simple(SimpleCompileValue::Function(func));
            return Ok(ValueInner::Value(result));
        }

        // enum variants
        let eval_enum = |elab| {
            let info = self.refs.shared.elaboration_arenas.enum_info(elab);
            let variant_index = info.find_variant(diags, Spanned::new(index.span, index_str))?;
            let variant_info = &info.variants[variant_index];

            let result = match &variant_info.payload_ty {
                None => Value::Compound(MixedCompoundValue::Enum(EnumValue {
                    ty: elab,
                    variant: variant_index,
                    payload: None,
                })),
                Some(_) => Value::Simple(SimpleCompileValue::Function(FunctionValue::EnumNew(
                    elab,
                    variant_index,
                ))),
            };
            Ok(ValueInner::Value(result))
        };

        if let &Value::Simple(SimpleCompileValue::Type(Type::Enum(elab))) = &base_eval {
            return eval_enum(elab);
        }
        if let Some(&FunctionItemBody::Enum(unique, _)) = base_item_function {
            return if let &Type::Enum(expected) = expected_ty {
                let expected_info = self.refs.shared.elaboration_arenas.enum_info(expected);
                if expected_info.unique == unique {
                    eval_enum(expected)
                } else {
                    Err(
                        error_unique_mismatch("enum", base.span, expected_info.unique.id().span(), unique.id().span())
                            .report(diags),
                    )
                }
            } else {
                // TODO check if there is any possible variant for this index string,
                //   otherwise we'll get confusing and delayed error messages
                // TODO this is indeed confusing, eg. print(Option.None) is a function,
                //   this gets especially confusing when merging into a hardware value
                let func = FunctionValue::EnumNewInfer(unique, Arc::new(index_str.to_owned()));
                Ok(ValueInner::Value(Value::Simple(SimpleCompileValue::Function(func))))
            };
        }

        // struct fields
        let base_ty = base_eval.ty();
        if let Type::Struct(base_ty_struct) = base_ty {
            let info = self.refs.shared.elaboration_arenas.struct_info(base_ty_struct);
            let field_index = info.fields.get_index_of(index_str).ok_or_else(|| {
                DiagnosticError::new(
                    "field not found",
                    index.span,
                    "attempt to access non-existing field here",
                )
                .add_info(base.span, format!("base has type `{}`", base_ty.value_string(elab)))
                .add_info(info.span_body, "struct fields declared here")
                .report(diags)
            })?;

            let result = match base_eval {
                Value::Simple(_) => return Err(diags.report_error_internal(expr_span, "expected struct value")),
                Value::Compound(base_eval) => match base_eval {
                    MixedCompoundValue::Struct(base_eval) => base_eval.fields[field_index].clone(),
                    _ => return Err(diags.report_error_internal(expr_span, "expected struct compile value")),
                },
                Value::Hardware(base_eval) => {
                    let base_eval = base_eval.value;
                    match base_eval.ty {
                        HardwareType::Struct(elab) => {
                            let elab_info = self.refs.shared.elaboration_arenas.struct_info(elab.inner());
                            let field_types = elab_info.fields_hw.as_ref().unwrap();

                            let expr = IrExpressionLarge::TupleIndex {
                                base: base_eval.expr,
                                index: field_index,
                            };
                            Value::Hardware(HardwareValue {
                                ty: field_types[field_index].clone(),
                                domain: base_eval.domain,
                                expr: self.large.push_expr(expr),
                            })
                        }
                        _ => return Err(diags.report_error_internal(expr_span, "expected struct hardware value")),
                    }
                }
            };

            return Ok(ValueInner::Value(ValueWithImplications::simple(result)));
        }

        // array length
        if let Type::Array(_, value_len) = &base_ty
            && index_str == "len"
        {
            return if let Some(value_len) = value_len {
                let result = Value::new_int(BigInt::from(value_len));
                Ok(ValueInner::Value(result))
            } else {
                Err(diags.report_error_internal(expr_span, "value has type with unknown array length"))
            };
        }

        // fallthrough into error
        let diag = DiagnosticError::new(
            "invalid dot index expression",
            index.span,
            format!("no attribute found with name `{index_str}`"),
        )
        .add_info(base.span, format!("base has type `{}`", base_ty.value_string(elab)))
        .report(diags);
        Err(diag)
    }

    fn eval_array_literal(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr_span: Span,
        values: &[ArrayLiteralElement<Expression>],
    ) -> DiagResult<Value> {
        // intentionally ignore the length, the caller can pass "0" when they have no opinion on it
        // TODO if we stop ignoring the length at some point, then we can infer lengths in eg. `[false] * _`
        let expected_ty_inner = match expected_ty {
            Type::Array(inner, _len) => &**inner,
            _ => &Type::Any,
        };

        // evaluate
        let values = values
            .iter()
            .map(|v| {
                let expected_ty_curr = match v {
                    ArrayLiteralElement::Single(_) => expected_ty_inner,
                    ArrayLiteralElement::Spread(_, _) => &Type::Array(Arc::new(expected_ty_inner.clone()), None),
                };

                v.map_inner(|value_inner| self.eval_expression(scope, flow, expected_ty_curr, value_inner))
                    .transpose()
            })
            .try_collect_all_vec()?;

        // combine into compile or non-compile value
        array_literal_combine_values(self.refs, flow, &mut self.large, expr_span, values)
    }

    fn eval_tuple_literal(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        values: &Vec<Expression>,
    ) -> DiagResult<Value> {
        let expected_tys_inner = if let Type::Tuple(Some(tys)) = expected_ty
            && tys.len() == values.len()
        {
            Some(tys)
        } else {
            None
        };

        let values = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let expected_ty_i = expected_tys_inner.map_or(&Type::Any, |tys| &tys[i]);
                Ok(self.eval_expression(scope, flow, expected_ty_i, v)?.inner)
            })
            .try_collect_all_vec()?;

        Ok(Value::Compound(MixedCompoundValue::Tuple(values)))
    }

    fn eval_expression_as_array_step(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        index: Expression,
    ) -> DiagResult<ArrayStep> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        // special case range with length, it can have a hardware start index
        // TODO cleanup, this special case is probably no longer necessary
        let step = if let &ExpressionKind::RangeLiteral(RangeLiteral::Length {
            op_span,
            start,
            length: len,
        }) = self.refs.get_expr(index)
        {
            let reason = TypeContainsReason::Operator(op_span);
            let start = self
                .eval_expression(scope, flow, &Type::Any, start)
                .and_then(|start| check_type_is_int(diags, elab, reason, start));

            let ty_uint = Type::Int(MultiRange::from(Range {
                start: Some(BigInt::ZERO),
                end: None,
            }));
            let len = self
                .eval_expression_as_compile(scope, flow, &ty_uint, len, Spanned::new(index.span, "range length"))
                .and_then(|len| check_type_is_uint_compile(diags, elab, reason, len));

            let start = start?;
            let len = len?;

            match start {
                MaybeCompile::Compile(start) => ArrayStep::Compile(ArrayStepCompile::ArraySlice {
                    start,
                    length: Some(len),
                }),
                MaybeCompile::Hardware(start) => {
                    ArrayStep::Hardware(ArrayStepHardware::ArraySlice { start, length: len })
                }
            }
        } else {
            let index = self.eval_expression(scope, flow, &Type::Any, index)?;
            let index_span = index.span;

            let index_ty = index.inner.ty();
            let err_wrong_type = || {
                diags.report_error_simple(
                    "array index must be integer or range",
                    index.span,
                    format!("got type `{}`", index_ty.value_string(elab)),
                )
            };
            let err_hardware_not_len = || {
                DiagnosticError::new(
                    "hardware slicing only supports `start+..length` ranges",
                    index.span,
                    "got non-length slice",
                )
                .report(diags)
            };

            match index.inner {
                Value::Simple(index) => match index {
                    SimpleCompileValue::Int(index) => ArrayStep::Compile(ArrayStepCompile::ArrayIndex(index)),
                    _ => return Err(err_wrong_type()),
                },
                Value::Compound(index) => match index {
                    MixedCompoundValue::Range(range) => match range {
                        RangeValue::Normal(Range { start, end }) => {
                            let start = match start {
                                None => BigInt::ZERO,
                                Some(start) => match start {
                                    MaybeCompile::Compile(start) => start,
                                    MaybeCompile::Hardware(_) => return Err(err_hardware_not_len()),
                                },
                            };

                            let length = match end {
                                None => None,
                                Some(end) => match end {
                                    MaybeCompile::Compile(end) => {
                                        Some(BigUint::try_from(&end - &start).map_err(|_| {
                                            diags.report_error_simple(
                                                "array slice end out of bounds",
                                                index_span,
                                                format!("slice end cannot be negative, got {end}"),
                                            )
                                        })?)
                                    }
                                    MaybeCompile::Hardware(_) => return Err(err_hardware_not_len()),
                                },
                            };

                            ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, length })
                        }
                        RangeValue::HardwareStartLength { start, length } => {
                            ArrayStep::Hardware(ArrayStepHardware::ArraySlice { start, length })
                        }
                    },
                    _ => return Err(err_wrong_type()),
                },
                Value::Hardware(index) => {
                    // TODO make this error message better, specifically refer to non-compile-time index
                    let reason = TypeContainsReason::ArrayIndex { span_index: index_span };
                    let index_int = check_type_is_int_hardware(diags, elab, reason, Spanned::new(index_span, index))?;
                    ArrayStep::Hardware(ArrayStepHardware::ArrayIndex(index_int))
                }
            }
        };

        Ok(step)
    }

    pub fn eval_expression_as_compile(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr: Expression,
        reason: Spanned<&'static str>,
    ) -> DiagResult<Spanned<CompileValue>> {
        // TODO should we allow compile-time writes to the outside?
        //   right now we intentionally don't because that might be confusing in eg. function params or types
        let mut flow_inner = flow.new_child_compile(reason.span, reason.inner);
        let value = self.eval_expression(scope, &mut flow_inner, expected_ty, expr)?.inner;

        let value = CompileValue::try_from(&value).map_err(|_: NotCompile| {
            self.refs.diags.report_error_simple(
                format!("{} must be a compile-time value", reason.inner),
                expr.span,
                "got hardware value",
            )
        })?;
        Ok(Spanned {
            span: expr.span,
            inner: value,
        })
    }

    pub fn eval_expression_as_compile_or_undefined(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr: Expression,
        reason: Spanned<&'static str>,
    ) -> DiagResult<Spanned<MaybeUndefined<CompileValue>>> {
        let diags = self.refs.diags;

        if let ExpressionKind::Undefined = self.refs.get_expr(expr) {
            Ok(Spanned {
                span: expr.span,
                inner: MaybeUndefined::Undefined,
            })
        } else {
            // TODO should we allow compile-time writes to the outside?
            //   right now we intentionally don't because that might be confusing in eg. function params or types
            let mut flow_inner = flow.new_child_compile(reason.span, reason.inner);
            let value_eval = self.eval_expression(scope, &mut flow_inner, expected_ty, expr)?.inner;

            match CompileValue::try_from(&value_eval) {
                Ok(value_eval) => Ok(Spanned {
                    span: expr.span,
                    inner: MaybeUndefined::Defined(value_eval),
                }),
                Err(NotCompile) => Err(diags.report_error_simple(
                    format!("{} must be a compile-time value", reason.inner),
                    expr.span,
                    "got hardware value",
                )),
            }
        }
    }

    pub fn eval_expression_as_ty(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
    ) -> DiagResult<Spanned<Type>> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        // TODO unify this message with the one when a normal type-check fails
        match self
            .eval_expression_as_compile(scope, flow, &Type::Type, expr, Spanned::new(expr.span, "type"))?
            .inner
        {
            CompileValue::Simple(SimpleCompileValue::Type(ty)) => Ok(Spanned {
                span: expr.span,
                inner: ty,
            }),
            value => {
                let mut diag = DiagnosticError::new(
                    "expected type, got value",
                    expr.span,
                    format!("got value with type `{}`", value.ty().value_string(elab)),
                );

                if let ExpressionKind::TupleLiteral(_) = self.refs.get_expr(expr) {
                    diag = diag.add_footer_hint("tuple types are written `Tuple(...)`, not `(...)`")
                }

                Err(diag.report(diags))
            }
        }
    }

    pub fn eval_expression_as_ty_hardware(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
        reason: &str,
    ) -> DiagResult<Spanned<HardwareType>> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let ty = self.eval_expression_as_ty(scope, flow, expr)?.inner;
        let ty_hw = ty.as_hardware_type(elab).map_err(|_| {
            diags.report_error_simple(
                format!("{reason} type must be representable in hardware"),
                expr.span,
                format!("got type `{}`", ty.value_string(elab)),
            )
        })?;
        Ok(Spanned {
            span: expr.span,
            inner: ty_hw,
        })
    }

    // TODO move typechecks here, immediately returning expected type if any
    pub fn eval_expression_as_assign_target(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
    ) -> DiagResult<Spanned<AssignmentTarget>> {
        let diags = self.refs.diags;

        // TODO include definition site (at least for named values)
        let build_err =
            |actual: &str| diags.report_error_simple("expected assignment target", expr.span, format!("got {actual}"));

        let result = match *self.refs.get_expr(expr) {
            ExpressionKind::Id(id) => {
                let id = self.eval_general_id(scope, flow, id)?;
                let id = id.as_ref().map_inner(ArcOrRef::as_ref);

                match self.eval_named_or_value(scope, id)?.inner {
                    NamedOrValue::ItemValue(_) => return Err(build_err("item")),
                    NamedOrValue::Named(s) => match s {
                        NamedValue::Variable(v) => AssignmentTarget::simple(Spanned::new(expr.span, v.into())),
                        NamedValue::Port(port) => {
                            // check direction
                            let direction = self.ports[port].direction;
                            match direction.inner {
                                PortDirection::Input => return Err(build_err("input port")),
                                PortDirection::Output => {}
                            }

                            AssignmentTarget::simple(Spanned::new(expr.span, port.into()))
                        }
                        NamedValue::PortInterface(_) | NamedValue::WireInterface(_) => {
                            return Err(build_err("interface instance"));
                        }
                        NamedValue::Wire(w) => AssignmentTarget::simple(Spanned::new(expr.span, w.into())),
                        NamedValue::Register(r) => AssignmentTarget::simple(Spanned::new(expr.span, r.into())),
                    },
                }
            }
            ExpressionKind::ArrayIndex {
                span_brackets,
                base,
                ref indices,
            } => {
                // eval base
                let base = self.eval_expression_as_assign_target(scope, flow, base)?;

                // eval index
                let index = indices
                    .single_ref()
                    .ok_or_else(|| diags.report_error_todo(expr.span, "multidimensional arrays"))?;
                let index = match index {
                    &ArrayLiteralElement::Single(index) => index,
                    ArrayLiteralElement::Spread(_, _) => {
                        return Err(diags.report_error_todo(index.span(), "multidimensional arrays"));
                    }
                };

                let index_step = self.eval_expression_as_array_step(scope, flow, index)?;
                let index_step = Spanned::new(span_brackets, index_step);

                // wrap result
                let AssignmentTarget {
                    base: base_inner,
                    array_steps: mut inner_array_steps,
                } = base.inner;
                inner_array_steps.steps.push(index_step);
                AssignmentTarget {
                    base: base_inner,
                    array_steps: inner_array_steps,
                }
            }
            ExpressionKind::DotIndex(base, index) => {
                match index {
                    DotIndexKind::Id(index) => {
                        match self.refs.get_expr(base) {
                            &ExpressionKind::Id(base) => {
                                let base = self.eval_general_id(scope, flow, base)?;
                                let base = base.as_ref().map_inner(ArcOrRef::as_ref);

                                match self.eval_named_or_value(scope, base)?.inner {
                                    NamedOrValue::Named(NamedValue::PortInterface(base)) => {
                                        // get port
                                        let port_interface_info = &self.port_interfaces[base];
                                        let interface_info = self
                                            .refs
                                            .shared
                                            .elaboration_arenas
                                            .interface_info(port_interface_info.view.inner.interface);
                                        let (port_index, _) =
                                            interface_info.get_port(diags, self.refs.fixed.source, index)?;
                                        let port = port_interface_info.ports[port_index];

                                        // check direction
                                        let direction = self.ports[port].direction;
                                        match direction.inner {
                                            PortDirection::Input => return Err(build_err("input port")),
                                            PortDirection::Output => {}
                                        }

                                        AssignmentTarget::simple(Spanned::new(expr.span, port.into()))
                                    }
                                    NamedOrValue::Named(NamedValue::WireInterface(base)) => {
                                        // get port
                                        let wire_interface_info = &self.wire_interfaces[base];
                                        let interface_info = self
                                            .refs
                                            .shared
                                            .elaboration_arenas
                                            .interface_info(wire_interface_info.interface.inner);
                                        let (wire_index, _) =
                                            interface_info.get_port(diags, self.refs.fixed.source, index)?;
                                        let wire = wire_interface_info.wires[wire_index];

                                        AssignmentTarget::simple(Spanned::new(expr.span, wire.into()))
                                    }
                                    _ => {
                                        return Err(diags.report_error_simple(
                                            "dot index is only allowed on port/wire interfaces",
                                            base.span,
                                            "got other named value here",
                                        ));
                                    }
                                }
                            }
                            _ => {
                                return Err(diags.report_error_simple(
                                    "dot index is only allowed on port/wire interfaces",
                                    base.span,
                                    "got other expression here",
                                ));
                            }
                        }
                    }
                    DotIndexKind::Int { span: _ } => {
                        return Err(diags.report_error_todo(expr.span, "assignment target dot int index"))?;
                    }
                }
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
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
    ) -> DiagResult<Spanned<DomainSignal>> {
        let diags = self.refs.diags;
        let build_err =
            |actual: &str| diags.report_error_simple("expected domain signal", expr.span, format!("got `{actual}`"));
        self.try_eval_expression_as_domain_signal(scope, flow, expr, build_err)
            .map_err(|e| e.into_inner())
    }

    pub fn try_eval_expression_as_domain_signal<E>(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
        build_err: impl Fn(&str) -> E,
    ) -> Result<Spanned<DomainSignal>, Either<E, DiagError>> {
        // TODO expand to allow general expressions again (which then probably create implicit signals)?
        let result = match *self.refs.get_expr(expr) {
            ExpressionKind::UnaryOp(
                Spanned {
                    span: _,
                    inner: UnaryOp::Not,
                },
                inner,
            ) => {
                let inner = self
                    .try_eval_expression_as_domain_signal(scope, flow, inner, build_err)?
                    .inner;
                Ok(inner.invert())
            }
            ExpressionKind::Id(id) => {
                let id = self.eval_general_id(scope, flow, id).map_err(Either::Right)?;
                let id = id.as_ref().map_inner(ArcOrRef::as_ref);

                let value = self.eval_named_or_value(scope, id).map_err(|e| Either::Right(e))?;
                match value.inner {
                    NamedOrValue::ItemValue(_) => Err(build_err("item")),
                    NamedOrValue::Named(s) => match s {
                        NamedValue::Variable(_) => Err(build_err("variable")),
                        NamedValue::Port(p) => Ok(Polarized::new(Signal::Port(p))),
                        NamedValue::PortInterface(_) | NamedValue::WireInterface(_) => {
                            Err(build_err("interface instance"))
                        }
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
        scope: &Scope,
        flow: &mut impl Flow,
        domain: SyncDomain<Expression>,
    ) -> DiagResult<SyncDomain<DomainSignal>> {
        let SyncDomain { clock, reset } = domain;
        let clock = self.eval_expression_as_domain_signal(scope, flow, clock);
        let reset = reset.map(|reset| self.eval_expression_as_domain_signal(scope, flow, reset));
        Ok(SyncDomain {
            clock: clock?.inner,
            reset: reset.transpose()?.map(|r| r.inner),
        })
    }

    pub fn eval_domain(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        domain: Spanned<DomainKind<Expression>>,
    ) -> DiagResult<Spanned<DomainKind<DomainSignal>>> {
        let result = match domain.inner {
            DomainKind::Const => Ok(DomainKind::Const),
            DomainKind::Async => Ok(DomainKind::Async),
            DomainKind::Sync(domain) => self.eval_domain_sync(scope, flow, domain).map(DomainKind::Sync),
        };

        Ok(Spanned {
            span: domain.span,
            inner: result?,
        })
    }

    pub fn eval_port_domain(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        domain: Spanned<DomainKind<Expression>>,
    ) -> DiagResult<Spanned<DomainKind<Polarized<Port>>>> {
        let diags = self.refs.diags;
        let result = self.eval_domain(scope, flow, domain)?;

        Ok(Spanned {
            span: result.span,
            inner: match result.inner {
                DomainKind::Const => DomainKind::Const,
                DomainKind::Async => DomainKind::Async,
                DomainKind::Sync(sync) => DomainKind::Sync(sync.try_map_signal(|signal| {
                    signal.try_map_inner(|signal| match signal {
                        Signal::Port(port) => Ok(port),
                        Signal::Wire(_) => Err(diags.report_error_internal(domain.span, "expected port, got wire")),
                        Signal::Register(_) => {
                            Err(diags.report_error_internal(domain.span, "expected port, got register"))
                        }
                    })
                })?),
            },
        })
    }

    pub fn eval_expression_as_for_iterator(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        iter: Expression,
    ) -> DiagResult<ForIterator> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let iter = self.eval_expression(scope, flow, &Type::Any, iter)?;
        let iter_span = iter.span;

        let result = match iter.inner {
            Value::Compound(MixedCompoundValue::Range(iter)) => {
                // TODO allow hardware-length range?
                fn expect_compile<C, H>(
                    diags: &Diagnostics,
                    iter_span: Span,
                    v: MaybeCompile<C, H>,
                ) -> Result<C, DiagError> {
                    match v {
                        MaybeCompile::Compile(v) => Ok(v),
                        MaybeCompile::Hardware(_) => Err(diags.report_error_simple(
                            "iterator range must be compile-time value",
                            iter_span,
                            "got hardware value here",
                        )),
                    }
                }

                match iter {
                    RangeValue::Normal(Range { start, end }) => {
                        // check compile
                        let start = start.map(|start| expect_compile(diags, iter_span, start)).transpose()?;
                        let end = end.map(|end| expect_compile(diags, iter_span, end)).transpose()?;

                        // check start
                        let start = start.ok_or_else(|| {
                            let range = Range {
                                start: None,
                                end: end.clone(),
                            };
                            diags.report_error_simple(
                                "iterator range must have start value",
                                iter_span,
                                format!("got range `{}`", range),
                            )
                        })?;

                        ForIterator::Int { next: start, end }
                    }
                    RangeValue::HardwareStartLength { start, length: _ } => {
                        let _: HardwareInt = start;
                        return Err(diags.report_error_simple(
                            "iterator range must be compile-time value",
                            iter_span,
                            "got hardware value here",
                        ));
                    }
                }
            }
            Value::Simple(SimpleCompileValue::Array(array)) => ForIterator::CompileArray { next: 0, array },
            Value::Hardware(HardwareValue {
                ty: HardwareType::Array(ty_inner, len),
                domain,
                expr: array_expr,
            }) => {
                let base = HardwareValue {
                    ty: (Arc::unwrap_or_clone(ty_inner), len),
                    domain,
                    expr: array_expr,
                };
                ForIterator::HardwareArray {
                    next: BigUint::ZERO,
                    base,
                }
            }
            _ => {
                throw!(diags.report_error_simple(
                    "invalid for loop iterator type, must be range or array",
                    iter.span,
                    format!("iterator has type `{}`", iter.inner.ty().value_string(elab))
                ))
            }
        };

        Ok(result)
    }

    pub fn eval_expression_as_module(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        span_keyword: Span,
        expr: Expression,
    ) -> DiagResult<ElaboratedModule> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let eval =
            self.eval_expression_as_compile(scope, flow, &Type::Module, expr, Spanned::new(expr.span, "module"))?;

        let reason = TypeContainsReason::InstanceModule(span_keyword);
        check_type_contains_value(diags, elab, reason, &Type::Module, eval.as_ref())?;

        match eval.inner {
            CompileValue::Simple(SimpleCompileValue::Module(elab)) => Ok(elab),
            _ => Err(diags.report_error_internal(eval.span, "expected module, should have already been checked")),
        }
    }
}

/// Evaluate a call to `int` or `uint`, which is how integer range constraints are expressed.
/// The arguments must be compile time, and either:
/// * a single int, the bitwidth
/// * a list of ranges which together form the multi-range
fn eval_int_ty_call(
    refs: CompileRefs,
    span_call: Span,
    target: Spanned<&MultiRange<BigInt>>,
    args: Args<Option<Spanned<&str>>, Spanned<ValueWithImplications>>,
) -> DiagResult<Type> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;
    let args_span = args.span;

    // int calls should only work for `int` and `uint`, detect which of these it is here
    let target_signed = match target.inner.as_single_range() {
        Some(NonEmptyRange { start: None, end: None }) => true,
        Some(NonEmptyRange {
            start: Some(BigInt::ZERO),
            end: None,
        }) => false,
        _ => {
            let diag = DiagnosticError::new(
                "base type must be int or uint for int type constraining",
                span_call,
                "attempt to constrain int type here",
            )
            .add_info(
                target.span,
                format!(
                    "base type `{}` here",
                    Type::Int(target.inner.clone()).value_string(elab)
                ),
            )
            .report(diags);
            return Err(diag);
        }
    };

    // check that args are unnamed and compile-time
    let args = args
        .inner
        .iter()
        .map(|arg| {
            let Arg { span: _, name, value } = arg;
            if let Some(name) = name {
                return Err(diags.report_error_simple(
                    "expected unnamed arguments for int type",
                    name.span,
                    "got named arg here",
                ));
            }

            let arg_value = CompileValue::try_from(&value.inner).map_err(|_: NotCompile| {
                diags.report_error_simple(
                    "expected compile-time argument for int type",
                    value.span,
                    "got hardware value here",
                )
            })?;
            Ok(Spanned::new(value.span, arg_value))
        })
        .try_collect_all_vec()?;

    // if single integer arg, interpret as bitwidth
    if let Some(arg) = args.single_ref() {
        if let CompileValue::Simple(SimpleCompileValue::Int(width)) = &arg.inner {
            let width = BigUint::try_from(width.clone()).map_err(|width| {
                diags.report_error_simple(
                    format!("the bitwidth of an integer type cannot be negative, got `{width}`"),
                    arg.span,
                    "got negative bitwidth here",
                )
            })?;

            let range = if target_signed {
                let width_m1 = BigUint::try_from(width - 1u8).map_err(|_| {
                    diags.report_error_simple(
                        "zero-width signed integers are not allowed",
                        arg.span,
                        "got width zero here",
                    )
                })?;

                let pow = BigUint::pow_2_to(&width_m1);
                Range {
                    start: Some(-&pow),
                    end: Some(BigInt::from(pow)),
                }
            } else {
                Range {
                    start: Some(BigInt::ZERO),
                    end: Some(BigInt::from(BigUint::pow_2_to(&width))),
                }
            };
            return Ok(Type::Int(MultiRange::from(range)));
        }
    }

    // all args must be ranges, union them together
    let mut result = MultiRange::EMPTY;
    for arg in args {
        let arg_range = match arg.inner {
            CompileValue::Compound(CompileCompoundValue::Range(range)) => range,
            _ => {
                let diag = DiagnosticError::new(
                    "int type constraint must be a single int int or multiple int ranges",
                    arg.span,
                    format!("got value with type `{}` here", arg.inner.ty().value_string(elab)),
                )
                .report(diags);
                return Err(diag);
            }
        };
        result = result.union(&MultiRange::from(arg_range));
    }

    // check that the new range is a subrange of the base type
    let base_ty_name = match target_signed {
        true => "int",
        false => "uint",
    };
    if let Some(enclosing_range) = result.enclosing_range() {
        if !target.inner.contains_range(enclosing_range) {
            let diag = DiagnosticError::new(
                "int range must be a subrange of the base type",
                args_span,
                format!("new range `{result}` is not a subrange"),
            )
            .add_info(target.span, format!("base type {base_ty_name}"))
            .report(diags);
            return Err(diag);
        }
    }

    Ok(Type::Int(result))
}

fn eval_tuple_ty_call(
    refs: CompileRefs,
    args: Args<Option<Spanned<&str>>, Spanned<ValueWithImplications>>,
) -> DiagResult<Type> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    // check that args are unnamed and types
    let args = args
        .inner
        .iter()
        .map(|arg| {
            let Arg { span: _, name, value } = arg;
            if let Some(name) = name {
                return Err(diags.report_error_simple(
                    "expected unnamed arguments for tuple type",
                    name.span,
                    "got named arg here",
                ));
            }

            match &value.inner {
                Value::Simple(SimpleCompileValue::Type(ty)) => Ok(ty.clone()),
                _ => Err(diags.report_error_simple(
                    "expected type",
                    value.span,
                    format!("got value with type `{}` here", value.inner.ty().value_string(elab)),
                )),
            }
        })
        .try_collect_all_vec()?;

    Ok(Type::Tuple(Some(Arc::new(args))))
}

pub enum ForIterator {
    Int {
        next: BigInt,
        // exclusive
        end: Option<BigInt>,
    },
    CompileArray {
        next: usize,
        array: Arc<Vec<CompileValue>>,
    },
    HardwareArray {
        next: BigUint,
        base: HardwareValue<(HardwareType, BigUint)>,
    },
}

impl ForIterator {
    pub fn len(&self) -> Option<BigUint> {
        match self {
            ForIterator::Int { next, end } => end.as_ref().map(|end| BigUint::try_from(end - next).unwrap()),
            ForIterator::CompileArray { next, array } => Some(BigUint::from(array.len() - *next)),
            ForIterator::HardwareArray { next, base } => {
                let (_, len) = &base.ty;
                Some(BigUint::try_from(len - next).unwrap())
            }
        }
    }
}

impl Iterator for ForIterator {
    type Item = Value<SimpleCompileValue, MixedCompoundValue, HardwareValue<HardwareType, IrExpressionLarge>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ForIterator::Int { next, end } => {
                if end.as_ref().is_none_or(|end| &*next < end) {
                    let curr = Value::new_int(next.clone());
                    *next += 1;
                    Some(curr)
                } else {
                    None
                }
            }
            ForIterator::CompileArray { next, array } => {
                if *next < array.len() {
                    let curr = array[*next].clone();
                    *next += 1;
                    Some(Value::from(curr))
                } else {
                    None
                }
            }
            ForIterator::HardwareArray { next, base } => {
                let HardwareValue {
                    ty: (ty_inner, ty_len),
                    domain,
                    expr: base_expr,
                } = &*base;

                if &*next < ty_len {
                    let index_expr = IrExpression::Int(BigInt::from(next.clone()));
                    *next += 1u8;

                    let element_expr = IrExpressionLarge::ArrayIndex {
                        base: base_expr.clone(),
                        index: index_expr,
                    };
                    Some(Value::Hardware(HardwareValue {
                        ty: (*ty_inner).clone(),
                        domain: *domain,
                        expr: element_expr,
                    }))
                } else {
                    None
                }
            }
        }
    }
}

fn pair_compile_int<L: AnyInt, R: AnyInt>(
    left: MaybeCompile<L, HardwareValue<ClosedNonEmptyMultiRange<L>>>,
    right: MaybeCompile<R, HardwareValue<ClosedNonEmptyMultiRange<R>>>,
) -> MaybeCompile<
    (L, R),
    (
        HardwareValue<ClosedNonEmptyMultiRange<L>>,
        HardwareValue<ClosedNonEmptyMultiRange<R>>,
    ),
> {
    fn f<T: AnyInt>(x: T) -> HardwareValue<ClosedNonEmptyMultiRange<T>> {
        HardwareValue {
            ty: ClosedNonEmptyMultiRange::single(x.clone()),
            domain: ValueDomain::Const,
            expr: IrExpression::Int(x.into()),
        }
    }

    pair_compile(left, right, |x: L| f(x), |x: R| f(x))
}

fn pair_compile<LC, LH, RC, RH>(
    left: MaybeCompile<LC, LH>,
    right: MaybeCompile<RC, RH>,
    f_left: impl FnOnce(LC) -> LH,
    f_right: impl FnOnce(RC) -> RH,
) -> MaybeCompile<(LC, RC), (LH, RH)> {
    match (left, right) {
        (MaybeCompile::Compile(left), MaybeCompile::Compile(right)) => MaybeCompile::Compile((left, right)),
        (left, right) => {
            let left = match left {
                MaybeCompile::Compile(left) => f_left(left),
                MaybeCompile::Hardware(left) => left,
            };
            let right = match right {
                MaybeCompile::Compile(right) => f_right(right),
                MaybeCompile::Hardware(right) => right,
            };
            MaybeCompile::Hardware((left, right))
        }
    }
}

pub fn eval_binary_expression(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    expr_span: Span,
    op: Spanned<BinaryOp>,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
) -> DiagResult<ValueWithImplications> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    let op_reason = TypeContainsReason::Operator(op.span);

    let left_span = left.span;
    let right_span = right.span;

    let check_both_int = |left, right| {
        let left = check_type_is_int(diags, elab, op_reason, left);
        let right = check_type_is_int(diags, elab, op_reason, right);
        Ok((left?, right?))
    };
    let eval_binary_bool = |large, left, right, op| eval_binary_bool(refs, large, op_reason, left, right, op);
    let eval_binary_int_compare =
        |large, left, right, op| eval_binary_int_compare(refs, large, op_reason, left, right, op);

    let result_simple: Value<_> = match op.inner {
        // (int, int)
        BinaryOp::Add => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;
            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => Value::new_int(left + right),
                MaybeCompile::Hardware((left, right)) => {
                    let range = multi_range_binary_add(&left.ty, &right.ty);
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Add, large, range, left, right);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }
        BinaryOp::Sub => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;
            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => Value::new_int(left - right),
                MaybeCompile::Hardware((left, right)) => {
                    let range = multi_range_binary_sub(&left.ty, &right.ty);
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Sub, large, range, left, right);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }
        BinaryOp::Mul => {
            // TODO do we want to keep using multiplication as the "array repeat" syntax?
            //   if so, maybe allow tuples on the right side for multidimensional repeating
            let right = check_type_is_int(diags, elab, op_reason, right.map_inner(|e| e.into_value()));
            match left.inner.ty() {
                Type::Array(left_ty_inner, left_len) => {
                    let left_len = left_len.expect("array value has known length");

                    let right = right?;
                    let right_inner = match right {
                        MaybeCompile::Compile(right_inner) => right_inner,
                        MaybeCompile::Hardware(_) => {
                            return Err(diags.report_error_simple(
                                "array repetition right hand side must be compile-time value",
                                right_span,
                                "got non-compile-time value here",
                            ));
                        }
                    };
                    let right_inner = BigUint::try_from(right_inner).map_err(|right_inner| {
                        diags.report_error_simple(
                            "array repetition right hand side cannot be negative",
                            right_span,
                            format!("got value `{right_inner}`"),
                        )
                    })?;
                    let right_inner = usize::try_from(right_inner).map_err(|right_inner| {
                        diags.report_error_simple(
                            "array repetition right hand side too large",
                            right_span,
                            format!("got value `{right_inner}`"),
                        )
                    })?;

                    match left.inner.into_value() {
                        Value::Simple(SimpleCompileValue::Array(left_inner)) => {
                            // do the repetition at compile-time
                            // TODO check for overflow (everywhere)
                            let mut result = Vec::with_capacity(left_inner.len() * right_inner);
                            for _ in 0..right_inner {
                                result.extend_from_slice(&left_inner);
                            }
                            Value::Simple(SimpleCompileValue::Array(Arc::new(result)))
                        }
                        Value::Hardware(value) => {
                            // implement runtime repetition through spread array literal
                            let element = IrArrayLiteralElement::Spread(value.expr);
                            let elements = vec![element; right_inner];

                            let left_ty_inner_hw = left_ty_inner.as_hardware_type(elab).unwrap();
                            let result_len = left_len * right_inner;
                            let result_expr = IrExpressionLarge::ArrayLiteral(
                                left_ty_inner_hw.as_ir(refs),
                                result_len.clone(),
                                elements,
                            );
                            Value::Hardware(HardwareValue {
                                ty: HardwareType::Array(Arc::new(left_ty_inner_hw.clone()), result_len),
                                domain: value.domain,
                                expr: large.push_expr(result_expr),
                            })
                        }
                        Value::Simple(_) | Value::Compound(_) => {
                            return Err(diags.report_error_internal(
                                left.span,
                                "compile-time value with type array is not actually an array",
                            ));
                        }
                    }
                }
                Type::Int(_) => {
                    let left = check_type_is_int(diags, elab, op_reason, left.map_inner(|e| e.into_value()))
                        .expect("int, already checked");
                    let right = right?;
                    match pair_compile_int(left, right) {
                        MaybeCompile::Compile((left, right)) => Value::new_int(left * right),
                        MaybeCompile::Hardware((left, right)) => {
                            let range = multi_range_binary_mul(&left.ty, &right.ty);
                            let result =
                                build_binary_int_arithmetic_op(IrIntArithmeticOp::Mul, large, range, left, right);
                            Value::Hardware(HardwareValue::from(result))
                        }
                    }
                }
                _ => {
                    return Err(diags.report_error_simple(
                        "left hand side of multiplication must be an array or an integer",
                        left.span,
                        format!("got value with type `{}`", left.inner.ty().value_string(elab)),
                    ));
                }
            }
        }
        // (int, non-zero int)
        BinaryOp::Div => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;

            // check nonzero
            let right_range = right.range();
            if right_range.contains(&BigInt::ZERO) {
                let msg_right = message_range_or_single("right", &right_range, Some("which contains zero"));
                let diag = DiagnosticError::new("division by zero is not allowed", right_span, msg_right)
                    .add_info(op.span, "for operator here")
                    .report(diags);
                return Err(diag);
            }

            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => {
                    let result = left.div_floor(&right).unwrap();
                    Value::new_int(result)
                }
                MaybeCompile::Hardware((left, right)) => {
                    let range = multi_range_binary_div(&left.ty, &right.ty).expect("already checked for zero");
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Div, large, range, left, right);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }
        BinaryOp::Mod => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;

            // check nonzero
            let right_range = right.range();
            if right_range.contains(&BigInt::ZERO) {
                let msg_right = message_range_or_single("right", &right_range, Some("which contains zero"));
                let diag = DiagnosticError::new("modulo by zero is not allowed", right_span, msg_right)
                    .add_info(op.span, "for operator here")
                    .report(diags);
                return Err(diag);
            }

            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => {
                    let result = left.mod_floor(&right).unwrap();
                    Value::new_int(result)
                }
                MaybeCompile::Hardware((left, right)) => {
                    let range = multi_range_binary_mod(&left.ty, &right.ty).expect("already checked for zero");
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Mod, large, range, left, right);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }
        // (nonzero int, non-negative int) or (non-negative int, positive int)
        BinaryOp::Pow => {
            let base_span = left_span;
            let exp_span = right_span;
            let (base, exp) = check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;

            let zero = BigInt::ZERO;
            let base_range = base.range();
            let exp_range = exp.range();

            // check exp >= 0
            if *exp_range.enclosing_range().start < zero {
                let diag = DiagnosticError::new("invalid power operation", expr_span, "exponent must be non-negative")
                    .add_info(exp_span, message_range_or_single("exponent", &exp_range, None))
                    .report(diags);
                return Err(diag);
            }

            // check not 0 ** 0
            // TODO is there actually a reason to ban this? `1` makes sense as the answer
            if base_range.contains(&zero) && exp_range.contains(&zero) {
                let msg_base = message_range_or_single("base", &base_range, Some("which contains zero"));
                let msg_exp = message_range_or_single("exponent", &exp_range, Some("which contains zero"));
                let diag = DiagnosticError::new(
                    "invalid power operation `0 ** 0`",
                    expr_span,
                    "base and exponent could both be zero",
                )
                .add_info(base_span, msg_base)
                .add_info(exp_span, msg_exp)
                .report(diags);
                return Err(diag);
            }

            match pair_compile_int(base, exp) {
                MaybeCompile::Compile((base, exp)) => {
                    let exp = BigUint::try_from(exp)
                        .map_err(|_| diags.report_error_internal(exp_span, "got negative exp"))?;
                    Value::new_int(base.pow(&exp))
                }
                MaybeCompile::Hardware((base, exp)) => {
                    // we checked that exp is non-negative earlier
                    let exp_range = exp.ty.clone().map(|b| BigUint::try_from(b).unwrap());

                    let range = multi_range_binary_pow(&base.ty, &exp_range).expect("already checked for 0**0");
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Pow, large, range, base, exp);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }

        BinaryOp::Shl => {
            let left = check_type_is_int(diags, elab, op_reason, left.map_inner(|e| e.into_value()));
            let right = check_type_is_uint(diags, elab, op_reason, right.map_inner(|e| e.into_value()));

            let left = left?;
            let right = right?;

            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => {
                    let factor = BigUint::pow_2_to(&right);
                    Value::new_int(left * factor)
                }
                MaybeCompile::Hardware((left, right)) => {
                    let factor_range =
                        multi_range_binary_pow(&ClosedNonEmptyMultiRange::single(BigInt::TWO), &right.ty)
                            .expect("non-zero expr");
                    let result_range = multi_range_binary_mul(&left.ty, &factor_range);

                    let right = HardwareInt::from(right);
                    let result =
                        build_binary_int_arithmetic_op(IrIntArithmeticOp::Shl, large, result_range, left, right);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }
        BinaryOp::Shr => {
            let left = check_type_is_int(diags, elab, op_reason, left.map_inner(|e| e.into_value()));
            let right = check_type_is_uint(diags, elab, op_reason, right.map_inner(|e| e.into_value()));

            let left = left?;
            let right = right?;

            match pair_compile_int(left, right) {
                MaybeCompile::Compile((left, right)) => {
                    let divisor = BigUint::pow_2_to(&right);
                    let result = left.div_floor(&BigInt::from(divisor)).expect("non-zero div");
                    Value::new_int(result)
                }
                MaybeCompile::Hardware((left, right)) => {
                    let divisor_range =
                        multi_range_binary_pow(&ClosedNonEmptyMultiRange::single(BigInt::TWO), &right.ty)
                            .expect("non-zero expr");
                    let result_range = multi_range_binary_div(&left.ty, &divisor_range).expect("non-zero divisor");

                    let right = HardwareInt::from(right);
                    let result =
                        build_binary_int_arithmetic_op(IrIntArithmeticOp::Shr, large, result_range, left, right);
                    Value::Hardware(HardwareValue::from(result))
                }
            }
        }

        // (bool, bool)
        // TODO these should short-circuit, so delay evaluation of right
        BinaryOp::BoolAnd => return eval_binary_bool(large, left, right, IrBoolBinaryOp::And),
        BinaryOp::BoolOr => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Or),
        BinaryOp::BoolXor => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Xor),
        // (T, T)
        // TODO expand eq/neq to bools/tuples/strings/structs/enums, for the latter only if the type is the same
        BinaryOp::CmpEq => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Eq),
        BinaryOp::CmpNeq => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Neq),
        BinaryOp::CmpLt => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Lt),
        BinaryOp::CmpLte => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Lte),
        BinaryOp::CmpGt => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Gt),
        BinaryOp::CmpGte => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Gte),
        // (int, range) or (T:Eq, array)
        // TODO share code with match "in" pattern
        // TODO for hardware ranges, also check if start <(=) end, otherwise this might have false positives
        BinaryOp::In => return Err(diags.report_error_todo(expr_span, "binary op In")),
        // (bool, bool)
        // TODO support boolean arrays
        BinaryOp::BitAnd => return eval_binary_bool(large, left, right, IrBoolBinaryOp::And),
        BinaryOp::BitOr => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Or),
        BinaryOp::BitXor => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Xor),
    };

    Ok(ValueWithImplications::simple(result_simple))
}

fn eval_binary_bool(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    op_reason: TypeContainsReason,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
    op: IrBoolBinaryOp,
) -> DiagResult<ValueWithImplications> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    let left = check_type_is_bool(diags, elab, op_reason, left);
    let right = check_type_is_bool(diags, elab, op_reason, right);

    let left = left?;
    let right = right?;

    let result = match eval_binary_bool_typed(large, op, left, right) {
        MaybeCompile::Compile(v) => Value::new_bool(v),
        MaybeCompile::Hardware(v) => Value::Hardware(v.map_type(|_: TypeBool| HardwareType::Bool)),
    };
    Ok(result)
}

pub fn eval_binary_bool_typed(
    large: &mut IrLargeArena,
    op: IrBoolBinaryOp,
    left: MaybeCompile<bool, HardwareValueWithImplications<TypeBool>>,
    right: MaybeCompile<bool, HardwareValueWithImplications<TypeBool>>,
) -> MaybeCompile<bool, HardwareValueWithImplications<TypeBool>> {
    match (left, right) {
        // full compile-time eval
        (MaybeCompile::Compile(left), MaybeCompile::Compile(right)) => MaybeCompile::Compile(op.eval(left, right)),

        // partial compile-time eval
        (MaybeCompile::Compile(left), MaybeCompile::Hardware(right)) => {
            build_unary_bool_gate(large, right, |b| op.eval(left, b))
        }
        (MaybeCompile::Hardware(left), MaybeCompile::Compile(right)) => {
            build_unary_bool_gate(large, left, |b| op.eval(b, right))
        }

        // full hardware
        // TODO these could be improved for multi-integer ranges,
        //   we actually need to join/intersect allowed ranges per version
        (MaybeCompile::Hardware(left), MaybeCompile::Hardware(right)) => {
            let expr = HardwareValue {
                ty: TypeBool,
                domain: left.value.domain.join(right.value.domain),
                expr: large.push_expr(IrExpressionLarge::BoolBinary(op, left.value.expr, right.value.expr)),
            };

            let implications = match op {
                IrBoolBinaryOp::And => BoolImplications {
                    if_true: vec_concat([left.implications.if_true, right.implications.if_true]),
                    if_false: vec![],
                },
                IrBoolBinaryOp::Or => BoolImplications {
                    if_true: vec![],
                    if_false: vec_concat([left.implications.if_false, right.implications.if_false]),
                },
                IrBoolBinaryOp::Xor => BoolImplications::new(None),
            };

            MaybeCompile::Hardware(HardwareValueWithImplications {
                value: expr,
                version: None,
                implications,
            })
        }
    }
}

fn build_unary_bool_gate(
    large: &mut IrLargeArena,
    value: HardwareValueWithImplications<TypeBool>,
    op: impl Fn(bool) -> bool,
) -> MaybeCompile<bool, HardwareValueWithImplications<TypeBool>> {
    match (op(false), op(true)) {
        // constants
        (false, false) => MaybeCompile::Compile(false),
        (true, true) => MaybeCompile::Compile(true),
        // pass gate
        (false, true) => MaybeCompile::Hardware(value),
        // not gate
        (true, false) => MaybeCompile::Hardware(HardwareValueWithImplications {
            value: HardwareValue {
                ty: TypeBool,
                domain: value.value.domain,
                expr: large.push_expr(IrExpressionLarge::BoolNot(value.value.expr)),
            },
            version: None,
            implications: value.implications.invert(),
        }),
    }
}

fn eval_binary_int_compare(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    op_reason: TypeContainsReason,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
    op: IrIntCompareOp,
) -> DiagResult<ValueWithImplications> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    let left_int = check_type_is_int(
        diags,
        elab,
        op_reason,
        left.clone().map_inner(ValueWithImplications::into_value),
    );
    let right_int = check_type_is_int(
        diags,
        elab,
        op_reason,
        right.clone().map_inner(ValueWithImplications::into_value),
    );
    let left_int = left_int?;
    let right_int = right_int?;

    // TODO spans are getting unnecessarily complicated, eg. why does this try to propagate spans?
    match pair_compile_int(left_int, right_int) {
        MaybeCompile::Compile((left, right)) => {
            let result = op.eval(&left, &right);
            Ok(ValueWithImplications::new_bool(result))
        }
        MaybeCompile::Hardware((left_int, right_int)) => {
            let lv = match left.inner {
                ValueWithImplications::Simple(_) | ValueWithImplications::Compound(_) => None,
                ValueWithImplications::Hardware(v) => v.version,
            };
            let rv = match right.inner {
                ValueWithImplications::Simple(_) | ValueWithImplications::Compound(_) => None,
                ValueWithImplications::Hardware(v) => v.version,
            };

            // TODO warning if the result is always true/false (depending on the ranges)
            //   or maybe just return a compile-time value again?
            // build implications
            let lr = left_int.ty.enclosing_range();
            let rr = right_int.ty.enclosing_range();
            let implications = match op {
                IrIntCompareOp::Lt => implications_lt(false, lv, lr, rv, rr),
                IrIntCompareOp::Lte => implications_lt(true, lv, lr, rv, rr),
                IrIntCompareOp::Gt => implications_lt(false, rv, rr, lv, lr),
                IrIntCompareOp::Gte => implications_lt(true, rv, rr, lv, lr),
                IrIntCompareOp::Eq => implications_eq(lv, lr, rv, rr),
                IrIntCompareOp::Neq => implications_eq(lv, lr, rv, rr).invert(),
            };

            // build the resulting expression
            let result = HardwareValue {
                ty: HardwareType::Bool,
                domain: left_int.domain.join(right_int.domain),
                expr: large.push_expr(IrExpressionLarge::IntCompare(op, left_int.expr, right_int.expr)),
            };
            Ok(Value::Hardware(HardwareValueWithImplications {
                value: result,
                version: None,
                implications,
            }))
        }
    }
}

fn build_binary_int_arithmetic_op(
    op: IrIntArithmeticOp,
    large: &mut IrLargeArena,
    range: ClosedNonEmptyMultiRange<BigInt>,
    left: HardwareInt,
    right: HardwareInt,
) -> HardwareInt {
    let result_expr = IrExpressionLarge::IntArithmetic(op, range.enclosing_range().cloned(), left.expr, right.expr);
    HardwareInt {
        ty: range,
        domain: left.domain.join(right.domain),
        expr: large.push_expr(result_expr),
    }
}

// TODO move these to the implications module
// TODO rework implications entirely first
fn implications_lt(
    eq: bool,
    left: Option<ValueVersion>,
    left_range: ClosedNonEmptyRange<&BigInt>,
    right: Option<ValueVersion>,
    right_range: ClosedNonEmptyRange<&BigInt>,
) -> BoolImplications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(implication_lt_is(true, eq, left, right_range));
        if_false.push(implication_lt_is(false, eq, left, right_range));
    }

    #[allow(clippy::nonminimal_bool)]
    if let Some(right) = right {
        // flip left/right by inverting the eval and eq flags
        if_true.push(implication_lt_is(!true, !eq, right, left_range));
        if_false.push(implication_lt_is(!false, !eq, right, left_range));
    }

    BoolImplications { if_true, if_false }
}

/// Construct the implication for `left <(=) right == eval`. `eq` indicates whether the comparison is `<` or `<=`.
fn implication_lt_is(eval: bool, eq: bool, left: ValueVersion, right: ClosedNonEmptyRange<&BigInt>) -> Implication {
    let range = match (eval, eq) {
        // left <= right
        (true, true) => Range {
            start: None,
            end: Some(right.end.clone()),
        },
        // left < right
        (true, false) => Range {
            start: None,
            end: Some(right.end - 1),
        },
        // left > right
        (false, true) => Range {
            start: Some(right.start + 1),
            end: None,
        },
        // left >= right
        (false, false) => Range {
            start: Some(right.start.clone()),
            end: None,
        },
    };
    Implication::new_int(left, MultiRange::from(range))
}

fn implications_eq(
    left: Option<ValueVersion>,
    left_range: ClosedNonEmptyRange<&BigInt>,
    right: Option<ValueVersion>,
    right_range: ClosedNonEmptyRange<&BigInt>,
) -> BoolImplications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new_int(
            left,
            MultiRange::from(Range::from(right_range.cloned())),
        ));

        if let Some(right_single) = right_range.as_single() {
            if_false.push(Implication::new_int(
                left,
                MultiRange::from(Range::single(right_single.clone())).complement(),
            ));
        }
    }

    if let Some(right) = right {
        if_true.push(Implication::new_int(
            right,
            MultiRange::from(Range::from(left_range.cloned())),
        ));

        if let Some(left_single) = left_range.as_single() {
            if_false.push(Implication::new_int(
                right,
                MultiRange::from(Range::single(left_single.clone())).complement(),
            ));
        }
    }

    BoolImplications { if_true, if_false }
}

fn array_literal_combine_values(
    refs: CompileRefs,
    flow: &mut impl Flow,
    large: &mut IrLargeArena,
    expr_span: Span,
    values: Vec<ArrayLiteralElement<Spanned<Value>>>,
) -> DiagResult<Value> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    let first_non_compile = values
        .iter()
        .find(|v| CompileValue::try_from(&v.value().inner).is_err())
        .map(|v| v.value());

    if let Some(first_non_compile) = first_non_compile {
        // at least one non-compile, turn everything into IR
        let ty_inner = values
            .iter()
            .map(|v| match v {
                ArrayLiteralElement::Single(v) => Ok(v.inner.ty()),
                ArrayLiteralElement::Spread(_, v) => {
                    let v_ty = v.inner.ty();
                    match v_ty {
                        Type::Array(ty_inner, _len) => Ok(Arc::unwrap_or_clone(ty_inner)),
                        _ => Err(diags.report_error_simple(
                            "spread operator requires an array value",
                            v.span,
                            format!("got non-array type `{}`", v_ty.value_string(elab)),
                        )),
                    }
                }
            })
            .try_fold(Type::Undefined, |a, t| Ok(a.union(&t?)))?;

        // TODO improve error message:
        //   * if mix of types that cases result type any, report pair
        //   * if any element non-hardware, report that element
        let ty_inner_hw = ty_inner.as_hardware_type(elab).map_err(|_| {
            let message = format!(
                "hardware array literal has inner type `{}` which is not representable in hardware",
                ty_inner.value_string(elab)
            );
            DiagnosticError::new(
                "hardware array type needs to be representable in hardware",
                expr_span,
                message,
            )
            .add_info(
                first_non_compile.span,
                format!(
                    "first non-compile value with type `{}`",
                    first_non_compile.inner.ty().value_string(elab)
                ),
            )
            .report(diags)
        })?;
        let ty_inner_hw = Arc::new(ty_inner_hw);

        let mut result_domain = ValueDomain::CompileTime;
        let mut result_len = BigUint::ZERO;
        let mut result_exprs = Vec::with_capacity(values.len());

        for elem in values {
            let (elem_len, elem_domain, elem_expr) = match elem {
                ArrayLiteralElement::Single(elem_inner) => {
                    let elem_domain = elem_inner.inner.domain();
                    let elem_expr =
                        elem_inner
                            .inner
                            .as_ir_expression_unchecked(refs, large, elem_inner.span, &ty_inner_hw)?;

                    (BigUint::ONE, elem_domain, IrArrayLiteralElement::Single(elem_expr))
                }
                ArrayLiteralElement::Spread(_, elem_inner) => {
                    let elem_len = unwrap_match!(elem_inner.inner.ty(), Type::Array(_, len) => len)
                        .expect("array value has known length");
                    let elem_domain = elem_inner.inner.domain();
                    let elem_expr = elem_inner.inner.as_ir_expression_unchecked(
                        refs,
                        large,
                        elem_inner.span,
                        &HardwareType::Array(ty_inner_hw.clone(), elem_len.clone()),
                    )?;

                    (elem_len, elem_domain, IrArrayLiteralElement::Spread(elem_expr))
                }
            };

            result_len += elem_len;
            result_domain = result_domain.join(elem_domain);
            result_exprs.push(elem_expr);
        }

        let result_expr = IrExpressionLarge::ArrayLiteral(ty_inner_hw.as_ir(refs), result_len.clone(), result_exprs);
        let result_value = HardwareValue {
            ty: HardwareType::Array(ty_inner_hw, result_len),
            domain: result_domain,
            expr: large.push_expr(result_expr),
        };

        // store result in variable to prevent large duplicate expressions
        let flow = flow.require_hardware(expr_span, "array literal containing hardware values")?;
        let result_value = flow.store_hardware_value_in_new_ir_variable(refs, expr_span, None, result_value);

        Ok(Value::Hardware(result_value.map_expression(IrExpression::Variable)))
    } else {
        // all compile, create compile value
        let mut result = Vec::with_capacity(values.len());
        for elem in values {
            match elem {
                ArrayLiteralElement::Single(elem_inner) => {
                    let elem_inner = CompileValue::try_from(&elem_inner.inner).unwrap();
                    result.push(elem_inner);
                }
                ArrayLiteralElement::Spread(span_spread, elem_inner) => {
                    let elem_inner = CompileValue::try_from(&elem_inner.inner).unwrap();
                    let elem_inner_array = match elem_inner {
                        CompileValue::Simple(SimpleCompileValue::Array(elem_inner)) => elem_inner,
                        _ => {
                            return Err(diags.report_error_internal(span_spread, "expected array value"));
                        }
                    };
                    result.extend(elem_inner_array.iter().cloned())
                }
            }
        }
        Ok(Value::Simple(SimpleCompileValue::Array(Arc::new(result))))
    }
}

fn message_range_or_single(name: &str, range: &impl AnyMultiRange<BigInt>, suffix_if_multiple: Option<&str>) -> String {
    let (suffix_sep, suffix) = match suffix_if_multiple {
        None => ("", ""),
        Some(suffix) => (" ", suffix),
    };

    let range = range.as_multi_range();
    match range.as_single() {
        None => format!("{name} has range {range}{suffix_sep}{suffix}"),
        Some(single) => format!("{name} is {single}"),
    }
}
