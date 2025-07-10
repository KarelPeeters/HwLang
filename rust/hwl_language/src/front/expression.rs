use crate::front::assignment::{AssignmentTarget, AssignmentTargetBase};
use crate::front::check::{
    check_hardware_type_for_bit_operation, check_type_contains_compile_value, check_type_contains_value,
    check_type_is_bool, check_type_is_int, check_type_is_int_compile, check_type_is_int_hardware, check_type_is_string,
    check_type_is_uint_compile, TypeContainsReason,
};
use crate::front::compile::{CompileItemContext, CompileRefs, StackEntry};
use crate::front::diagnostic::{DiagError, DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::{DomainSignal, ValueDomain};
use crate::front::flow::{ExtraRegisters, ValueVersion};
use crate::front::function::{error_unique_mismatch, FunctionBits, FunctionBitsKind, FunctionBody, FunctionValue};
use crate::front::implication::{
    BoolImplications, HardwareValueWithImplications, Implication, ImplicationOp, ValueWithImplications,
};
use crate::front::item::{ElaboratedModule, FunctionItemBody};
use crate::front::scope::{NamedValue, Scope, ScopedEntry};
use crate::front::signal::{Polarized, Port, PortInterface, Signal, WireInterface};
use crate::front::steps::{ArrayStep, ArrayStepCompile, ArrayStepHardware, ArraySteps};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, ElaboratedInterfaceView, HardwareValue, MaybeUndefined, Value};
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIntArithmeticOp,
    IrIntCompareOp, IrLargeArena, IrRegisterInfo, IrStatement, IrVariableInfo,
};
use crate::syntax::ast::{
    Arg, Args, ArrayComprehension, ArrayLiteralElement, BinaryOp, BlockExpression, DomainKind, Expression,
    ExpressionKind, GeneralIdentifier, Identifier, IntLiteral, MaybeIdentifier, PortDirection, RangeLiteral,
    RegisterDelay, Spanned, StringPiece, SyncDomain, UnaryOp,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::data::{vec_concat, VecExt};

use crate::front::flow::{Flow, FlowKind, HardwareProcessKind};
use crate::front::module::ExtraRegisterInit;
use crate::syntax::token::apply_string_literal_escapes;
use crate::util::iter::IterExt;
use crate::util::store::ArcOrRef;
use crate::util::{result_pair, Never, ResultDoubleExt, ResultNeverExt};
use annotate_snippets::Level;
use itertools::{enumerate, Either};
use std::cmp::{max, min};
use std::ops::Sub;
use std::sync::Arc;
use unwrap_match::unwrap_match;

// TODO better name
#[derive(Debug)]
pub enum ValueInner {
    Value(ValueWithImplications),
    PortInterface(PortInterface),
    WireInterface(WireInterface),
}

#[derive(Debug)]
pub enum NamedOrValue {
    Named(NamedValue),
    Value(Value),
}

impl<'a> CompileItemContext<'a, '_> {
    pub fn eval_general_id(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        id: GeneralIdentifier,
    ) -> DiagResult<Spanned<ArcOrRef<'a, str>>> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        match id {
            GeneralIdentifier::Simple(id) => Ok(id.spanned_str(source).map_inner(ArcOrRef::Ref)),
            GeneralIdentifier::FromString(span, expr) => {
                let value = self.eval_expression_as_compile(scope, flow, &Type::String, expr, "id string")?;
                let value = check_type_is_string(diags, TypeContainsReason::Operator(span), value)?;

                Ok(Spanned::new(span, ArcOrRef::Arc(value)))
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
            MaybeIdentifier::Dummy(span) => Ok(MaybeIdentifier::Dummy(span)),
            MaybeIdentifier::Identifier(id) => {
                let id = self.eval_general_id(scope, flow, id)?;
                Ok(MaybeIdentifier::Identifier(id))
            }
        }
    }

    pub fn eval_named_or_value(&mut self, scope: &Scope, id: Spanned<&str>) -> DiagResult<Spanned<NamedOrValue>> {
        let diags = self.refs.diags;

        let found = scope.find(diags, id)?;
        let def_span = found.defining_span;
        let result = match *found.value {
            ScopedEntry::Named(value) => NamedOrValue::Named(value),
            ScopedEntry::Item(item) => {
                let entry = StackEntry::ItemUsage(id.span);
                self.recurse(entry, |s| {
                    Ok(NamedOrValue::Value(Value::Compile(s.eval_item(item)?.clone())))
                })
                .flatten_err()?
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
            .map_inner(|r| match r {
                Value::Compile(v) => Value::Compile(v),
                Value::Hardware(v) => Value::Hardware(v.value),
            }))
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
            ValueInner::PortInterface(_) | ValueInner::WireInterface(_) => Err(self.refs.diags.report_simple(
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

        let result_simple = match refs.get_expr(expr) {
            ExpressionKind::Dummy => {
                // if dummy expressions were allowed, the caller would have checked for them already
                return Err(diags.report_simple(
                    "dummy expression not allowed in this context",
                    expr.span,
                    "dummy expression used here",
                ));
            }
            ExpressionKind::Undefined => Value::Compile(CompileValue::Undefined),
            ExpressionKind::Type => Value::Compile(CompileValue::Type(Type::Type)),
            &ExpressionKind::Wrapped(inner) => {
                return self.eval_expression_inner(scope, flow, expected_ty, inner);
            }
            ExpressionKind::Block(block_expr) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = block_expr;

                let mut scope_inner = Scope::new_child(expr.span, scope);

                // TODO propagate return type?
                let end = self.elaborate_block_statements(&mut scope_inner, flow, None, statements)?;
                end.unwrap_outside_function_and_loop(diags)?;

                self.eval_expression(&scope_inner, flow, expected_ty, expression)?.inner
            }
            &ExpressionKind::Id(id) => {
                let id = self.eval_general_id(scope, flow, id)?;
                let id = id.as_ref().map_inner(ArcOrRef::as_ref);

                let result = match self.eval_named_or_value(scope, id)?.inner {
                    NamedOrValue::Value(value) => {
                        return Ok(ValueInner::Value(ValueWithImplications::simple(value)));
                    }
                    NamedOrValue::Named(value) => match value {
                        NamedValue::Variable(var) => flow.var_eval(self, Spanned::new(expr.span, var))?,
                        NamedValue::Port(port) => {
                            // TODO check domain, are we allowed to read this in the current context?
                            let flow = flow.check_hardware(expr.span, "port access")?;
                            Value::Hardware(flow.signal_eval(self, Spanned::new(expr.span, Signal::Port(port)))?)
                        }
                        NamedValue::Wire(wire) => {
                            // TODO check domain, are we allowed to read this in the current context?
                            let flow = flow.check_hardware(expr.span, "wire access")?;
                            Value::Hardware(flow.signal_eval(self, Spanned::new(expr.span, Signal::Wire(wire)))?)
                        }
                        NamedValue::Register(reg) => {
                            // TODO check domain, are we allowed to read this in the current context?
                            let flow = flow.check_hardware(expr.span, "register access")?;

                            let reg_info = &mut self.registers[reg];
                            if let HardwareProcessKind::ClockedBlockBody { domain, .. } = flow.block_kind() {
                                reg_info.suggest_domain(Spanned::new(expr.span, domain.inner));
                            }

                            Value::Hardware(flow.signal_eval(self, Spanned::new(expr.span, Signal::Register(reg)))?)
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
                return Ok(ValueInner::Value(
                    result.map_hardware(|v| HardwareValueWithImplications::simple_version(v)),
                ));
            }
            ExpressionKind::TypeFunction => Value::Compile(CompileValue::Type(Type::Function)),
            ExpressionKind::IntLiteral(ref pattern) => {
                let value = match *pattern {
                    IntLiteral::Binary(raw) => {
                        let raw = source.span_str(raw);
                        let clean = raw[2..].replace('_', "");
                        BigUint::from_str_radix(&clean, 2)
                            .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?
                    }
                    IntLiteral::Decimal(raw) => {
                        let raw = source.span_str(raw);
                        let clean = raw.replace('_', "");
                        BigUint::from_str_radix(&clean, 10)
                            .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?
                    }
                    IntLiteral::Hexadecimal(raw) => {
                        let raw = source.span_str(raw);
                        let s_hex = raw[2..].replace('_', "");
                        BigUint::from_str_radix(&s_hex, 16)
                            .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?
                    }
                };
                Value::Compile(CompileValue::Int(BigInt::from(value)))
            }
            &ExpressionKind::BoolLiteral(literal) => Value::Compile(CompileValue::Bool(literal)),
            ExpressionKind::StringLiteral(pieces) => {
                let mut s = String::new();

                for &piece in pieces {
                    match piece {
                        StringPiece::Literal(span) => {
                            let raw = source.span_str(span);
                            let escaped = apply_string_literal_escapes(raw);
                            s.push_str(escaped.as_ref());
                        }
                        StringPiece::Substitute(value) => {
                            let value = self.eval_expression(scope, flow, &Type::Any, value)?;

                            let value_inner = match value.inner {
                                Value::Compile(v) => v,
                                Value::Hardware(_) => {
                                    return Err(
                                        diags.report_todo(value.span, "string substitution for hardware values")
                                    );
                                }
                            };

                            let tmp;
                            let value_str = match &value_inner {
                                CompileValue::Undefined => "undef",
                                &CompileValue::Bool(b) => {
                                    if b {
                                        "true"
                                    } else {
                                        "false"
                                    }
                                }
                                CompileValue::Int(i) => {
                                    tmp = i.to_string();
                                    tmp.as_str()
                                }
                                CompileValue::String(s) => s.as_str(),
                                _ => {
                                    let msg = format!(
                                        "string substitution for values with type {}",
                                        value_inner.ty().diagnostic_string()
                                    );
                                    return Err(diags.report_todo(value.span, msg));
                                }
                            };
                            s.push_str(value_str);
                        }
                    }
                }

                Value::Compile(CompileValue::String(Arc::new(s)))
            }
            ExpressionKind::ArrayLiteral(values) => {
                self.eval_array_literal(scope, flow, expected_ty, expr.span, values)?
            }
            ExpressionKind::TupleLiteral(values) => self.eval_tuple_literal(scope, flow, expected_ty, values)?,
            ExpressionKind::RangeLiteral(literal) => {
                let mut eval_bound = |bound: Expression, bound_name: &'static str, op_span: Span| {
                    let bound =
                        self.eval_expression_as_compile(scope, flow, &Type::Int(IncRange::OPEN), bound, bound_name)?;
                    let reason = TypeContainsReason::Operator(op_span);
                    check_type_is_int_compile(diags, reason, bound)
                };

                match *literal {
                    RangeLiteral::ExclusiveEnd {
                        op_span,
                        ref start,
                        ref end,
                    } => {
                        let start = start.map(|start| eval_bound(start, "range start", op_span)).transpose();
                        let end = end.map(|end| eval_bound(end, "range end", op_span)).transpose();

                        let range = IncRange {
                            start_inc: start?,
                            end_inc: end?.map(|end| end - 1),
                        };
                        Value::Compile(CompileValue::IntRange(range))
                    }
                    RangeLiteral::InclusiveEnd { op_span, start, end } => {
                        let start = start.map(|start| eval_bound(start, "range start", op_span)).transpose();
                        let end = eval_bound(end, "range end", op_span);

                        let range = IncRange {
                            start_inc: start?,
                            end_inc: Some(end?),
                        };
                        Value::Compile(CompileValue::IntRange(range))
                    }
                    RangeLiteral::Length { op_span, start, len } => {
                        // TODO support runtime starts here too (so they can be full values),
                        //   for now those are special-cased in the array indexing evaluation.
                        //   Maybe we want real support for mixed compile/runtime compounds,
                        //     eg. arrays, tuples, ranges, ...
                        let start = eval_bound(start, "range start", op_span);
                        let length = eval_bound(len, "range length", op_span);

                        let start = start?;
                        let range = IncRange {
                            end_inc: Some(&start + length? - 1),
                            start_inc: Some(start),
                        };
                        Value::Compile(CompileValue::IntRange(range))
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
                let len_lower_bound = match iter_eval.len_if_finite() {
                    None => {
                        return Err(diags.report_simple(
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

                    let index_value = index_value.to_maybe_compile(&mut self.large);
                    let index_var = flow.var_new_immutable_init(index, span_keyword, Ok(index_value));

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

                array_literal_combine_values(refs, &mut self.large, expr.span, expected_ty_inner, values)?
            }

            &ExpressionKind::UnaryOp(op, operand) => match op.inner {
                UnaryOp::Plus => {
                    let operand = self.eval_expression_with_implications(scope, flow, &Type::Any, operand)?;
                    let _ = check_type_is_int(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        operand.clone().map_inner(ValueWithImplications::into_value),
                    )?;
                    return Ok(ValueInner::Value(operand.inner));
                }
                UnaryOp::Neg => {
                    let operand = self.eval_expression(scope, flow, &Type::Any, operand)?;
                    let operand_int = check_type_is_int(diags, TypeContainsReason::Operator(op.span), operand)?;

                    match operand_int.inner {
                        Value::Compile(c) => Value::Compile(CompileValue::Int(-c)),
                        Value::Hardware(v) => {
                            let result_range = ClosedIncRange {
                                start_inc: -v.ty.end_inc,
                                end_inc: -v.ty.start_inc,
                            };
                            let result_expr = self.large.push_expr(IrExpressionLarge::IntArithmetic(
                                IrIntArithmeticOp::Sub,
                                result_range.clone(),
                                IrExpression::Int(BigInt::ZERO),
                                v.expr,
                            ));

                            let result = HardwareValue {
                                ty: HardwareType::Int(result_range),
                                domain: v.domain,
                                expr: result_expr,
                            };
                            Value::Hardware(result)
                        }
                    }
                }
                UnaryOp::Not => {
                    let operand = self.eval_expression_with_implications(scope, flow, &Type::Any, operand)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        &Type::Bool,
                        operand.clone().map_inner(ValueWithImplications::into_value).as_ref(),
                        false,
                        false,
                    )?;

                    match operand.inner {
                        Value::Compile(c) => match c {
                            // TODO support boolean array
                            CompileValue::Bool(b) => Value::Compile(CompileValue::Bool(!b)),
                            _ => return Err(diags.report_internal_error(expr.span, "expected bool for unary not")),
                        },
                        Value::Hardware(v) => {
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
            &ExpressionKind::ArrayIndex(base, ref indices) => {
                self.eval_array_index_expression(scope, flow, base, indices)?
            }
            &ExpressionKind::ArrayType(ref lens, base) => {
                let lens = lens
                    .inner
                    .iter()
                    .map(|&len| {
                        let len = match len {
                            ArrayLiteralElement::Single(len) => len,
                            ArrayLiteralElement::Spread(_, _) => {
                                return Err(diags.report_todo(len.span(), "spread in array type lengths"))
                            }
                        };
                        let len_expected_ty = Type::Int(IncRange {
                            start_inc: Some(BigInt::ZERO),
                            end_inc: None,
                        });
                        let len =
                            self.eval_expression_as_compile(scope, flow, &len_expected_ty, len, "array type length")?;
                        let reason = TypeContainsReason::ArrayLen { span_len: len.span };
                        check_type_is_uint_compile(diags, reason, len)
                    })
                    .try_collect_all_vec();
                let base = self.eval_expression_as_ty(scope, flow, base);

                let lengths = lens?;
                let base = base?;

                // apply lengths inside-out
                let result = lengths
                    .into_iter()
                    .rev()
                    .fold(base.inner, |acc, len| Type::Array(Arc::new(acc), len));
                Value::Compile(CompileValue::Type(result))
            }
            &ExpressionKind::DotIdIndex(base, index) => {
                return self.eval_dot_id_index(scope, flow, expected_ty, expr.span, base, index);
            }
            &ExpressionKind::DotIntIndex(base, index_span) => {
                let base_eval = self.eval_expression_inner(scope, flow, &Type::Any, base)?;
                let index = source.span_str(index_span);

                let index_int = BigUint::from_str_radix(index, 10)
                    .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?;
                let err_not_tuple = |ty: &str| {
                    let diag = Diagnostic::new("indexing into non-tuple type")
                        .add_error(index_span, "attempt to index into non-tuple type here")
                        .add_info(base.span, format!("base has type `{ty}`"))
                        .finish();
                    diags.report(diag)
                };
                let err_index_out_of_bounds = |len: usize| {
                    let diag = Diagnostic::new("tuple index out of bounds")
                        .add_error(index_span, format!("index `{index_int}` is out of bounds"))
                        .add_info(base.span, format!("base is tuple with length `{len}`"))
                        .finish();
                    diags.report(diag)
                };

                match base_eval {
                    ValueInner::Value(Value::Compile(CompileValue::Tuple(inner))) => {
                        let index = index_int
                            .as_usize_if_lt(inner.len())
                            .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                        Value::Compile(inner[index].clone())
                    }
                    ValueInner::Value(Value::Compile(CompileValue::Type(Type::Tuple(inner)))) => {
                        let index = index_int
                            .as_usize_if_lt(inner.len())
                            .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                        Value::Compile(CompileValue::Type(inner[index].clone()))
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
                                    index: index.into(),
                                };
                                Value::Hardware(HardwareValue {
                                    ty: inner_tys[index].clone(),
                                    domain: value.domain,
                                    expr: self.large.push_expr(expr),
                                })
                            }
                            _ => return Err(err_not_tuple(&value.ty.diagnostic_string())),
                        }
                    }
                    ValueInner::Value(v) => return Err(err_not_tuple(&v.ty().diagnostic_string())),
                    ValueInner::PortInterface(_) | ValueInner::WireInterface(_) => {
                        return Err(err_not_tuple("interface instance"))
                    }
                }
            }
            &ExpressionKind::Call(target, ref args) => {
                // eval target
                let target = self.eval_expression_as_compile(scope, flow, &Type::Any, target, "call target")?;

                // eval args
                let args_eval = args
                    .inner
                    .iter()
                    .map(|arg| {
                        // TODO pass an actual expected type in cases where we know it (eg. struct/enum construction)
                        let arg_value = self.eval_expression(scope, flow, &Type::Any, arg.value)?;

                        Ok(Arg {
                            span: arg.span,
                            name: arg.name.map(|id| Spanned::new(id.span, id.str(source))),
                            value: arg_value,
                        })
                    })
                    .try_collect_all_vec();
                let args = args_eval.map(|inner| Args { span: args.span, inner });

                // check that the target is a function
                let target_inner = match target.inner {
                    CompileValue::Type(Type::Int(range)) => {
                        // handle integer calls here
                        let result = eval_int_ty_call(diags, expr.span, Spanned::new(target.span, range), args?)?;
                        return Ok(ValueInner::Value(Value::Compile(CompileValue::Type(Type::Int(result)))));
                    }
                    CompileValue::Function(f) => f,
                    _ => {
                        let e = diags.report_simple(
                            "call target must be function",
                            expr.span,
                            format!("got `{}`", target.inner.diagnostic_string()),
                        );
                        return Err(e);
                    }
                };

                let args = args?;

                // actually do the call
                // TODO should we do the recursion marker here or inside of the call function?
                let entry = StackEntry::FunctionCall(expr.span);
                let value = self
                    .recurse(entry, |s| {
                        s.call_function(flow, expected_ty, target.span, expr.span, &target_inner, args)
                    })
                    .flatten_err()?;

                value
            }
            ExpressionKind::Builtin(ref args) => self.eval_builtin(scope, flow, expr.span, args)?,
            &ExpressionKind::UnsafeValueWithDomain(value, domain) => {
                let value = self.eval_expression(scope, flow, expected_ty, value);

                let mut flow_domain = flow.new_child_compile(domain.span, "domain");
                let domain = self.eval_domain(scope, &mut flow_domain, domain);

                let value = value?;
                let domain = domain?;

                match value.inner {
                    // casting compile values is useless but not harmful
                    Value::Compile(value) => Value::Compile(value),
                    Value::Hardware(value) => {
                        // TODO warn if this call is redundant, eg. if the domain would already be safely assignable
                        Value::Hardware(HardwareValue {
                            ty: value.ty,
                            domain: ValueDomain::from_domain_kind(domain.inner),
                            expr: value.expr,
                        })
                    }
                }
            }
            ExpressionKind::RegisterDelay(reg_delay) => {
                let &RegisterDelay {
                    span_keyword,
                    value,
                    init,
                } = reg_delay;

                // eval
                let value = self.eval_expression(scope, flow, expected_ty, value);
                let init = self.eval_expression_as_compile(scope, flow, expected_ty, init, "register init");
                let (value, init) = result_pair(value, init)?;

                // figure out type
                let value_ty = value.inner.ty();
                let init_ty = init.inner.ty();
                let ty = value_ty.union(&init_ty, true);
                let ty_hw = ty.as_hardware_type(refs).map_err(|_| {
                    let diag = Diagnostic::new("register type must be representable in hardware")
                        .add_error(
                            span_keyword,
                            format!("got non-hardware type `{}`", ty.diagnostic_string()),
                        )
                        .add_info(value.span, format!("from combining `{}`", value_ty.diagnostic_string()))
                        .add_info(init.span, format!("from combining `{}`", init_ty.diagnostic_string()))
                        .finish();
                    diags.report(diag)
                })?;

                let flow = flow.check_hardware(expr.span, "register expression")?;

                // convert values to hardware
                let value = value
                    .inner
                    .as_hardware_value(refs, &mut self.large, value.span, &ty_hw)?;
                let init = init
                    .as_ref()
                    .map_inner(|inner| inner.as_ir_expression_or_undefined(refs, &mut self.large, init.span, &ty_hw))
                    .transpose()?;

                // create variable to hold the result
                let debug_info_id = || Spanned::new(span_keyword, None);
                let var_info = IrVariableInfo {
                    ty: ty_hw.as_ir(refs),
                    debug_info_id: debug_info_id(),
                };
                let ir_var = flow.new_ir_variable(var_info);

                // create register to act as the delay storage
                let (clocked_domain, clocked_registers, ir_registers) = flow.check_clocked_block()?;
                let reg_info = IrRegisterInfo {
                    ty: ty_hw.as_ir(refs),
                    debug_info_id: debug_info_id(),
                    debug_info_ty: ty_hw.clone(),
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
                            let diag = Diagnostic::new("registers without reset cannot have an initial value")
                                .add_error(init.span, "attempt to create a register with init here")
                                .add_info(
                                    clocked_domain.span,
                                    "the current clocked block is defined without reset here",
                                )
                                .footer(
                                    Level::Help,
                                    "either add an reset to the block or use `undef` as the the initial value",
                                )
                                .finish();
                            let _ = diags.report(diag);
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
                let stmt_load = IrStatement::Assign(IrAssignmentTarget::variable(ir_var), IrExpression::Register(reg));
                flow.push_ir_statement(Spanned::new(span_keyword, stmt_load));
                let stmt_store = IrStatement::Assign(IrAssignmentTarget::register(reg), value.expr);
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
        let refs = self.refs;
        let diags = self.refs.diags;

        let base_eval = self.eval_expression_inner(scope, flow, &Type::Any, base)?;
        let index_str = index.str(self.refs.fixed.source);

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
                    let diag = Diagnostic::new(format!("port `{}` not found on interface", index_str))
                        .add_error(index.span, "attempt to access port here")
                        .add_info(port_interface_info.view.span, "port interface set here")
                        .add_info(interface_info.id.span(), "interface declared here")
                        .finish();
                    diags.report(diag)
                })?;
                let port = port_interface_info.ports[port_index];

                // TODO check domain, are we allowed to read this in the current context?
                let flow = flow.check_hardware(expr_span, "port access")?;
                let port_eval = flow.signal_eval(self, Spanned::new(expr_span, Signal::Port(port)))?;
                return Ok(ValueInner::Value(Value::Hardware(
                    HardwareValueWithImplications::simple_version(port_eval),
                )));
            }
            ValueInner::WireInterface(wire_interface) => {
                let wire_interface_info = &self.wire_interfaces[wire_interface];
                let interface_info = self
                    .refs
                    .shared
                    .elaboration_arenas
                    .interface_info(wire_interface_info.interface.inner);

                let wire_index = interface_info.ports.get_index_of(index_str).ok_or_else(|| {
                    let diag = Diagnostic::new(format!("port `{}` not found on interface", index_str))
                        .add_error(index.span, "attempt to access port here")
                        .add_info(wire_interface_info.interface.span, "wire interface set here")
                        .add_info(interface_info.id.span(), "interface declared here")
                        .finish();
                    diags.report(diag)
                })?;
                let wire = wire_interface_info.wires[wire_index];

                // TODO check domain, are we allowed to read this in the current context?
                let flow = flow.check_hardware(expr_span, "wire access")?;
                let wire_eval = flow.signal_eval(self, Spanned::new(expr_span, Signal::Wire(wire)))?;
                return Ok(ValueInner::Value(Value::Hardware(
                    HardwareValueWithImplications::simple_version(wire_eval),
                )));
            }
            ValueInner::Value(base_eval) => base_eval,
        };

        // interface views
        if let &Value::Compile(CompileValue::Interface(base_interface)) = &base_eval {
            let info = self.refs.shared.elaboration_arenas.interface_info(base_interface);
            let (view_index, _) = info.get_view(diags, self.refs.fixed.source, index)?;

            let interface_view = ElaboratedInterfaceView {
                interface: base_interface,
                view_index,
            };
            let result = Value::Compile(CompileValue::InterfaceView(interface_view));
            return Ok(ValueInner::Value(result));
        }

        // common type attributes
        if let Value::Compile(CompileValue::Type(ty)) = &base_eval {
            match index_str {
                "size_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(refs, Spanned::new(base.span, ty))?;
                    let width = ty_hw.as_ir(refs).size_bits();
                    let result = Value::Compile(CompileValue::Int(width.into()));
                    return Ok(ValueInner::Value(result));
                }
                // TODO all of these should return functions with a single params,
                //   without the need for scope capturing
                "to_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(refs, Spanned::new(base.span, ty))?;
                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::ToBits,
                    };
                    let result = Value::Compile(CompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                "from_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(refs, Spanned::new(base.span, ty))?;

                    if !ty_hw.every_bit_pattern_is_valid(refs) {
                        let diag =
                            Diagnostic::new("from_bits is only allowed for types where every bit pattern is valid")
                                .add_error(
                                    base.span,
                                    format!("got type `{}` with invalid bit patterns", ty_hw.diagnostic_string()),
                                )
                                .footer(
                                    Level::Help,
                                    "consider using use target type where every bit pattern is valid",
                                )
                                .footer(
                                    Level::Help,
                                    "if you know the bits are valid for this type, use `from_bits_unsafe´ instead",
                                )
                                .finish();
                        return Err(diags.report(diag));
                    }

                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::FromBits,
                    };
                    let result = Value::Compile(CompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                "from_bits_unsafe" => {
                    let ty_hw = check_hardware_type_for_bit_operation(refs, Spanned::new(base.span, ty))?;
                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::FromBits,
                    };
                    let result = Value::Compile(CompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                _ => {}
            }
        }

        // struct new
        if let &Value::Compile(CompileValue::Type(Type::Struct(elab))) = &base_eval {
            if index_str == "new" {
                let result = Value::Compile(CompileValue::Function(FunctionValue::StructNew(elab)));
                return Ok(ValueInner::Value(result));
            }
        }

        let base_item_function = match &base_eval {
            Value::Compile(CompileValue::Function(FunctionValue::User(func))) => match &func.body.inner {
                FunctionBody::ItemBody(body) => Some(body),
                _ => None,
            },
            _ => None,
        };
        if let Some(&FunctionItemBody::Struct(unique, _)) = base_item_function {
            if index_str == "new" {
                let func = FunctionValue::StructNewInfer(unique);
                let result = Value::Compile(CompileValue::Function(func));
                return Ok(ValueInner::Value(result));
            }
        }

        // enum variants
        let eval_enum = |elab| {
            let info = self.refs.shared.elaboration_arenas.enum_info(elab);
            let variant_index = info.find_variant(diags, Spanned::new(index.span, index_str))?;
            let (_, content_ty) = &info.variants[variant_index];

            let result = match content_ty {
                None => CompileValue::Enum(elab, (variant_index, None)),
                Some(_) => CompileValue::Function(FunctionValue::EnumNew(elab, variant_index)),
            };

            Ok(ValueInner::Value(Value::Compile(result)))
        };

        if let &Value::Compile(CompileValue::Type(Type::Enum(elab))) = &base_eval {
            return eval_enum(elab);
        }
        if let Some(&FunctionItemBody::Enum(unique, _)) = base_item_function {
            return if let &Type::Enum(expected) = expected_ty {
                let expected_info = self.refs.shared.elaboration_arenas.enum_info(expected);
                if expected_info.unique == unique {
                    eval_enum(expected)
                } else {
                    Err(diags.report(error_unique_mismatch(
                        "struct",
                        base.span,
                        expected_info.unique.span_id(),
                        unique.span_id(),
                    )))
                }
            } else {
                // TODO check if there is any possible variant for this index string,
                //   otherwise we'll get confusing and delayed error messages
                let func = FunctionValue::EnumNewInfer(unique, Arc::new(index_str.to_owned()));
                Ok(ValueInner::Value(Value::Compile(CompileValue::Function(func))))
            };
        }

        // struct fields
        let base_ty = base_eval.ty();
        if let Type::Struct(elab) = base_ty {
            let info = self.refs.shared.elaboration_arenas.struct_info(elab);
            let field_index = info.fields.get_index_of(index_str).ok_or_else(|| {
                let diag = Diagnostic::new("field not found")
                    .add_info(base.span, format!("base has type `{}`", base_ty.diagnostic_string()))
                    .add_error(index.span, "attempt to access non-existing field here")
                    .add_info(info.span_body, "struct fields declared here")
                    .finish();
                diags.report(diag)
            })?;

            let result = match base_eval {
                Value::Compile(base_eval) => match base_eval {
                    CompileValue::Struct(_, field_values) => Value::Compile(field_values[field_index].clone()),
                    _ => return Err(diags.report_internal_error(expr_span, "expected struct compile value")),
                },
                Value::Hardware(base_eval) => {
                    let base_eval = base_eval.value;
                    match base_eval.ty {
                        HardwareType::Struct(elab) => {
                            let elab_info = self.refs.shared.elaboration_arenas.struct_info(elab.inner());
                            let field_types = elab_info.fields_hw.as_ref().unwrap();

                            let expr = IrExpressionLarge::TupleIndex {
                                base: base_eval.expr,
                                index: field_index.into(),
                            };
                            Value::Hardware(HardwareValue {
                                ty: field_types[field_index].clone(),
                                domain: base_eval.domain,
                                expr: self.large.push_expr(expr),
                            })
                        }
                        _ => return Err(diags.report_internal_error(expr_span, "expected struct hardware value")),
                    }
                }
            };

            return Ok(ValueInner::Value(ValueWithImplications::simple(result)));
        }

        // fallthrough into error
        let diag = Diagnostic::new("invalid dot index expression")
            .add_info(base.span, format!("base has type `{}`", base_ty.diagnostic_string()))
            .add_error(index.span, format!("no attribute found with with name `{index_str}`"))
            .finish();
        Err(diags.report(diag))
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
                    ArrayLiteralElement::Spread(_, _) => {
                        &Type::Array(Arc::new(expected_ty_inner.clone()), BigUint::ZERO)
                    }
                };

                v.map_inner(|value_inner| self.eval_expression(scope, flow, expected_ty_curr, value_inner))
                    .transpose()
            })
            .try_collect_all_vec()?;

        // combine into compile or non-compile value
        array_literal_combine_values(self.refs, &mut self.large, expr_span, expected_ty_inner, values)
    }

    fn eval_tuple_literal(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        values: &Vec<Expression>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        let expected_tys_inner = match expected_ty {
            Type::Tuple(tys) if tys.len() == values.len() => Some(tys),
            _ => None,
        };

        // evaluate
        let values = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let expected_ty_i = expected_tys_inner.map_or(&Type::Any, |tys| &tys[i]);
                self.eval_expression(scope, flow, expected_ty_i, v)
            })
            .try_collect_all_vec()?;

        // combine into compile or non-compile value
        let first_non_compile = values
            .iter()
            .find(|v| !matches!(v.inner, Value::Compile(_)))
            .map(|v| v.span);
        Ok(if let Some(first_non_compile) = first_non_compile {
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
                let expected_ty_inner_hw = expected_ty_inner.as_hardware_type(self.refs).map_err(|_| {
                    let message = format!(
                        "tuple element has inferred type `{}` which is not representable in hardware",
                        expected_ty_inner.diagnostic_string()
                    );
                    let diag = Diagnostic::new("hardware tuple elements need to be representable in hardware")
                        .add_error(value.span, message)
                        .add_info(first_non_compile, "necessary because this other tuple element is not a compile-time value, which forces the entire tuple to be hardware")
                        .finish();
                    diags.report(diag)
                })?;

                let value_ir =
                    value
                        .inner
                        .as_hardware_value(self.refs, &mut self.large, value.span, &expected_ty_inner_hw)?;
                result_ty.push(value_ir.ty);
                result_domain = result_domain.join(value_ir.domain);
                result_expr.push(value_ir.expr);
            }

            Value::Hardware(HardwareValue {
                ty: HardwareType::Tuple(Arc::new(result_ty)),
                domain: result_domain,
                expr: self.large.push_expr(IrExpressionLarge::TupleLiteral(result_expr)),
            })
        } else if values
            .iter()
            .all(|v| matches!(v.inner, Value::Compile(CompileValue::Type(_))))
        {
            // all type
            let tys = values
                .into_iter()
                .map(|v| unwrap_match!(v.inner, Value::Compile(CompileValue::Type(v)) => v))
                .collect();
            Value::Compile(CompileValue::Type(Type::Tuple(Arc::new(tys))))
        } else {
            // all compile
            let values = values
                .into_iter()
                .map(|v| unwrap_match!(v.inner, Value::Compile(v) => v))
                .collect();
            Value::Compile(CompileValue::Tuple(Arc::new(values)))
        })
    }

    fn eval_array_index_expression(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        base: Expression,
        indices: &Spanned<Vec<Expression>>,
    ) -> DiagResult<Value> {
        let base = self.eval_expression(scope, flow, &Type::Any, base);
        let steps = indices
            .inner
            .iter()
            .map(|&index| self.eval_expression_as_array_step(scope, flow, index))
            .try_collect_all_vec();

        let base = base?;
        let steps = ArraySteps::new(steps?);

        steps.apply_to_value(self.refs, &mut self.large, base)
    }

    fn eval_expression_as_array_step(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        index: Expression,
    ) -> DiagResult<Spanned<ArrayStep>> {
        let diags = self.refs.diags;

        // special case range with length, it can have a hardware start index
        let step = if let &ExpressionKind::RangeLiteral(RangeLiteral::Length { op_span, start, len }) =
            self.refs.get_expr(index)
        {
            let reason = TypeContainsReason::Operator(op_span);
            let start = self
                .eval_expression(scope, flow, &Type::Any, start)
                .and_then(|start| check_type_is_int(diags, reason, start));

            let len_expected_ty = Type::Int(IncRange {
                start_inc: Some(BigInt::ZERO),
                end_inc: None,
            });
            let len = self
                .eval_expression_as_compile(scope, flow, &len_expected_ty, len, "range length")
                .and_then(|len| check_type_is_uint_compile(diags, reason, len));

            let start = start?;
            let len = len?;

            match start.inner {
                Value::Compile(start) => ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, len: Some(len) }),
                Value::Hardware(start) => ArrayStep::Hardware(ArrayStepHardware::ArraySlice { start, len }),
            }
        } else {
            let index_eval = self.eval_expression(scope, flow, &Type::Any, index)?;

            match index_eval.transpose() {
                Value::Compile(index_or_slice) => match index_or_slice.inner {
                    CompileValue::Int(index) => ArrayStep::Compile(ArrayStepCompile::ArrayIndex(index)),
                    CompileValue::IntRange(range) => {
                        let start = range.start_inc.clone().unwrap_or(BigInt::ZERO);
                        match &range.end_inc {
                            None => ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, len: None }),
                            Some(end_inc) => {
                                let slice_len = BigUint::try_from(end_inc - &start + 1).map_err(|_| {
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
                            format!("got `{}`", index_or_slice.inner.diagnostic_string()),
                        ));
                    }
                },
                Value::Hardware(index) => {
                    // TODO make this error message better, specifically refer to non-compile-time index
                    let reason = TypeContainsReason::ArrayIndex { span_index: index.span };
                    let index = check_type_is_int_hardware(diags, reason, index)?;
                    ArrayStep::Hardware(ArrayStepHardware::ArrayIndex(index.inner))
                }
            }
        };

        Ok(Spanned {
            span: index.span,
            inner: step,
        })
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr_span: Span,
        args: &Spanned<Vec<Expression>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        // evaluate args
        let args_eval = args
            .inner
            .iter()
            .map(|&arg| Ok(self.eval_expression(scope, flow, &Type::Any, arg)?.inner))
            .try_collect_all_vec()?;

        if let (Some(Value::Compile(CompileValue::String(a0))), Some(Value::Compile(CompileValue::String(a1)))) =
            (args_eval.get(0), args_eval.get(1))
        {
            let rest = &args_eval[2..];
            let print_compile = |v: &Value| {
                let value_str = match v {
                    // TODO print strings without quotes
                    Value::Compile(v) => v.diagnostic_string(),
                    // TODO less ugly formatting for HardwareValue
                    Value::Hardware(v) => {
                        let HardwareValue { ty, domain, expr: _ } = v;
                        let ty_str = ty.diagnostic_string();
                        let domain_str = domain.diagnostic_string(self);
                        format!("HardwareValue {{ ty: {ty_str}, domain: {domain_str}, expr: _, }}")
                    }
                };
                self.refs.print_handler.println(&value_str);
            };

            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return Ok(Value::Compile(CompileValue::Type(Type::Any))),
                ("type", "bool", []) => return Ok(Value::Compile(CompileValue::Type(Type::Bool))),
                ("type", "str", []) => return Ok(Value::Compile(CompileValue::Type(Type::String))),
                ("type", "Range", []) => return Ok(Value::Compile(CompileValue::Type(Type::Range))),
                ("type", "int", []) => {
                    return Ok(Value::Compile(CompileValue::Type(Type::Int(IncRange::OPEN))));
                }
                ("fn", "typeof", [value]) => return Ok(Value::Compile(CompileValue::Type(value.ty()))),
                ("fn", "print", [value]) => {
                    match flow.kind_mut() {
                        FlowKind::Compile(_) => {
                            // TODO record similarly to diagnostics, where they can be deterministically printed later
                            print_compile(value);
                            return Ok(Value::Compile(CompileValue::unit()));
                        }
                        FlowKind::Hardware(flow) => {
                            if let Value::Compile(CompileValue::String(value)) = value {
                                let stmt = Spanned::new(expr_span, IrStatement::PrintLn((**value).clone()));
                                flow.push_ir_statement(stmt);
                                return Ok(Value::Compile(CompileValue::unit()));
                            }
                            // fallthough
                        }
                    }
                }
                (
                    "fn",
                    "assert",
                    &[Value::Compile(CompileValue::Bool(cond)), Value::Compile(CompileValue::String(ref msg))],
                ) => {
                    return if cond {
                        Ok(Value::Compile(CompileValue::unit()))
                    } else {
                        Err(diags.report_simple(
                            format!("assertion failed with message {:?}", msg),
                            expr_span,
                            "failed here",
                        ))
                    }
                }
                ("fn", "assert", [Value::Hardware(_), Value::Compile(CompileValue::String(_))]) => {
                    return Err(diags.report_todo(expr_span, "runtime assert"));
                }
                ("fn", "unsafe_bool_to_clock", [v]) => match v.ty() {
                    Type::Bool => {
                        let expr = v.as_hardware_value(self.refs, &mut self.large, expr_span, &HardwareType::Bool)?;
                        return Ok(Value::Hardware(HardwareValue {
                            ty: HardwareType::Bool,
                            domain: ValueDomain::Clock,
                            expr: expr.expr.clone(),
                        }));
                    }
                    _ => {}
                },
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
        scope: &Scope,
        flow: &mut impl Flow,
        expected_ty: &Type,
        expr: Expression,
        reason: &'static str,
    ) -> DiagResult<Spanned<CompileValue>> {
        let diags = self.refs.diags;

        // TODO should we allow compile-time writes to the outside?
        //   right now we intentionally don't because that might be confusing in eg. function params or types
        let mut flow_inner = flow.new_child_compile(expr.span, reason);
        let value_eval = self.eval_expression(scope, &mut flow_inner, expected_ty, expr)?.inner;

        match value_eval {
            Value::Compile(c) => Ok(Spanned {
                span: expr.span,
                inner: c,
            }),
            // TODO maybe this can be an internal error now, the flow already prevents hardware values
            Value::Hardware(ir_expr) => Err(diags.report_simple(
                format!("{reason} must be a compile-time value"),
                expr.span,
                format!("got value with domain `{}`", ir_expr.domain.diagnostic_string(self)),
            )),
        }
    }

    pub fn eval_expression_as_ty(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
    ) -> DiagResult<Spanned<Type>> {
        let diags = self.refs.diags;

        // TODO unify this message with the one when a normal type-check fails
        match self
            .eval_expression_as_compile(scope, flow, &Type::Type, expr, "type")?
            .inner
        {
            CompileValue::Type(ty) => Ok(Spanned {
                span: expr.span,
                inner: ty,
            }),
            value => Err(diags.report_simple(
                "expected type, got value",
                expr.span,
                format!("got value `{}`", value.diagnostic_string()),
            )),
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

        let ty = self.eval_expression_as_ty(scope, flow, expr)?.inner;
        let ty_hw = ty.as_hardware_type(self.refs).map_err(|_| {
            diags.report_simple(
                format!("{} type must be representable in hardware", reason),
                expr.span,
                format!("got type `{}`", ty.diagnostic_string()),
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
            |actual: &str| diags.report_simple("expected assignment target", expr.span, format!("got {}", actual));

        let result = match self.refs.get_expr(expr) {
            &ExpressionKind::Id(id) => {
                let id = self.eval_general_id(scope, flow, id)?;
                let id = id.as_ref().map_inner(ArcOrRef::as_ref);

                match self.eval_named_or_value(scope, id)?.inner {
                    NamedOrValue::Value(_) => return Err(build_err("value")),
                    NamedOrValue::Named(s) => match s {
                        NamedValue::Variable(v) => {
                            AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Variable(v)))
                        }
                        NamedValue::Port(port) => {
                            // check direction
                            let direction = self.ports[port].direction;
                            match direction.inner {
                                PortDirection::Input => return Err(build_err("input port")),
                                PortDirection::Output => {}
                            }

                            AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Port(port)))
                        }
                        NamedValue::PortInterface(_) | NamedValue::WireInterface(_) => {
                            return Err(build_err("interface instance"))
                        }
                        NamedValue::Wire(w) => {
                            AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Wire(w)))
                        }
                        NamedValue::Register(r) => {
                            AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Register(r)))
                        }
                    },
                }
            }
            &ExpressionKind::ArrayIndex(inner_target, ref indices) => {
                let inner_target = self.eval_expression_as_assign_target(scope, flow, inner_target);
                let array_steps = indices
                    .inner
                    .iter()
                    .map(|&index| self.eval_expression_as_array_step(scope, flow, index))
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
            &ExpressionKind::DotIdIndex(base, index) => match self.refs.get_expr(base) {
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
                            let (port_index, _) = interface_info.get_port(diags, self.refs.fixed.source, index)?;
                            let port = port_interface_info.ports[port_index];

                            // check direction
                            let direction = self.ports[port].direction;
                            match direction.inner {
                                PortDirection::Input => return Err(build_err("input port")),
                                PortDirection::Output => {}
                            }

                            AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Port(port)))
                        }
                        NamedOrValue::Named(NamedValue::WireInterface(base)) => {
                            // get port
                            let wire_interface_info = &self.wire_interfaces[base];
                            let interface_info = self
                                .refs
                                .shared
                                .elaboration_arenas
                                .interface_info(wire_interface_info.interface.inner);
                            let (wire_index, _) = interface_info.get_port(diags, self.refs.fixed.source, index)?;
                            let wire = wire_interface_info.wires[wire_index];

                            AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Wire(wire)))
                        }
                        _ => {
                            return Err(diags.report_simple(
                                "dot index is only allowed on port/wire interfaces",
                                base.span,
                                "got other named value here",
                            ));
                        }
                    }
                }
                _ => {
                    return Err(diags.report_simple(
                        "dot index is only allowed on port/wire interfaces",
                        base.span,
                        "got other expression here",
                    ))
                }
            },
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
        scope: &Scope,
        flow: &mut impl Flow,
        expr: Expression,
    ) -> DiagResult<Spanned<DomainSignal>> {
        let diags = self.refs.diags;
        let build_err =
            |actual: &str| diags.report_simple("expected domain signal", expr.span, format!("got `{}`", actual));
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
                    NamedOrValue::Value(_) => Err(build_err("value")),
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
                        Signal::Wire(_) => Err(diags.report_internal_error(domain.span, "expected port, got wire")),
                        Signal::Register(_) => {
                            Err(diags.report_internal_error(domain.span, "expected port, got register"))
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

        let iter = self.eval_expression(scope, flow, &Type::Any, iter)?;
        let iter_span = iter.span;

        let result = match iter.inner {
            Value::Compile(CompileValue::IntRange(iter)) => {
                let IncRange { start_inc, end_inc } = iter;
                let start_inc = match start_inc {
                    Some(start_inc) => start_inc,
                    None => {
                        return Err(diags.report_simple(
                            "for loop iterator range must have start value",
                            iter_span,
                            format!(
                                "got range `{}`",
                                IncRange {
                                    start_inc: None,
                                    end_inc
                                }
                            ),
                        ))
                    }
                };

                ForIterator::Int {
                    next: start_inc,
                    end_inc,
                }
            }
            Value::Compile(CompileValue::Array(array)) => ForIterator::CompileArray { next: 0, array },
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
                throw!(diags.report_simple(
                    "invalid for loop iterator type, must be range or array",
                    iter.span,
                    format!("iterator has type `{}`", iter.inner.ty().diagnostic_string())
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

        let eval = self.eval_expression_as_compile(scope, flow, &Type::Module, expr, "module")?;

        let reason = TypeContainsReason::InstanceModule(span_keyword);
        check_type_contains_compile_value(diags, reason, &Type::Module, eval.as_ref(), false)?;

        match eval.inner {
            CompileValue::Module(elab) => Ok(elab),
            _ => Err(diags.report_internal_error(eval.span, "expected module, should have already been checked")),
        }
    }
}

fn eval_int_ty_call(
    diags: &Diagnostics,
    span_call: Span,
    target_range: Spanned<IncRange<BigInt>>,
    args: Args<Option<Spanned<&str>>, Spanned<Value>>,
) -> DiagResult<IncRange<BigInt>> {
    // ensure single unnamed compile-time arg
    let arg = args.inner.single().ok_or_else(|| {
        diags.report_simple(
            "expected single argument for int type",
            args.span,
            "got multiple args here",
        )
    })?;

    let Arg { span: _, name, value } = arg;
    if let Some(name) = name {
        return Err(diags.report_simple(
            "expected unnamed argument for int type",
            name.span,
            "got named arg here",
        ));
    }
    let arg = match value.inner {
        Value::Compile(value_inner) => Spanned::new(value.span, value_inner),
        Value::Hardware(_) => {
            return Err(diags.report_simple(
                "expected compile-time argument for int type",
                value.span,
                "got hardware value here",
            ))
        }
    };

    // int calls should only work for `int` and `uint`, detect which of these it is here
    let target_signed = match target_range.inner {
        IncRange {
            start_inc: None,
            end_inc: None,
        } => true,
        IncRange {
            start_inc: Some(BigInt::ZERO),
            end_inc: None,
        } => false,
        _ => {
            let diag = Diagnostic::new("base type must be int or uint for int type constraining")
                .add_error(span_call, "attempt to constrain int type here")
                .add_info(
                    target_range.span,
                    format!("base type `{}` here", Type::Int(target_range.inner).diagnostic_string()),
                )
                .finish();
            return Err(diags.report(diag));
        }
    };

    let result = match arg.inner {
        CompileValue::Int(width) => {
            // int arg, this is the number of bits in `int(bits)` or `uint(bits)`
            let width = BigUint::try_from(width).map_err(|width| {
                diags.report_simple(
                    format!("the bitwidth of an integer type cannot be negative, got `{width}`"),
                    arg.span,
                    "got negative bitwidth here",
                )
            })?;

            if target_signed {
                let width_m1 = BigUint::try_from(width - 1).map_err(|_| {
                    diags.report_simple(
                        "zero-width signed integers are not allowed",
                        arg.span,
                        "got width zero here",
                    )
                })?;

                let pow = BigUint::pow_2_to(&width_m1);
                IncRange {
                    start_inc: Some(-&pow),
                    end_inc: Some(pow - 1),
                }
            } else {
                IncRange {
                    start_inc: Some(BigInt::ZERO),
                    end_inc: Some(BigUint::pow_2_to(&width) - 1),
                }
            }
        }
        CompileValue::IntRange(new_range) => {
            // int range arg, this is the new range
            if !target_range.inner.contains_range(&new_range) {
                let base_ty_name = match target_signed {
                    true => "int",
                    false => "uint",
                };
                let diag = Diagnostic::new("int range must be a subrange of the base type")
                    .add_error(arg.span, format!("new range `{}` is not a subrange", new_range))
                    .add_info(arg.span, format!("base type {base_ty_name}"))
                    .finish();
                return Err(diags.report(diag));
            }

            new_range
        }
        _ => {
            let diag = Diagnostic::new("int type constraining must be an int or int range")
                .add_error(
                    arg.span,
                    format!("got invalid value `{}` here", arg.inner.diagnostic_string()),
                )
                .finish();
            return Err(diags.report(diag));
        }
    };
    Ok(result)
}

pub enum ForIterator {
    Int {
        next: BigInt,
        end_inc: Option<BigInt>,
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
    pub fn len_if_finite(&self) -> Option<BigUint> {
        match self {
            ForIterator::Int { next, end_inc } => end_inc
                .as_ref()
                .map(|end_inc| BigUint::try_from(end_inc - next + 1u32).unwrap()),
            ForIterator::CompileArray { next, array } => Some(BigUint::from(array.len() - *next)),
            ForIterator::HardwareArray { next, base } => {
                let (_, len) = &base.ty;
                Some(BigUint::try_from(len - next).unwrap())
            }
        }
    }
}

impl Iterator for ForIterator {
    type Item = Value<CompileValue, HardwareValue<HardwareType, IrExpressionLarge>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ForIterator::Int { next, end_inc } => {
                if let Some(end_inc) = end_inc {
                    if next > end_inc {
                        return None;
                    }
                }

                let curr = Value::Compile(CompileValue::Int(next.clone()));
                *next += 1;
                Some(curr)
            }
            ForIterator::CompileArray { next, array } => {
                if *next < array.len() {
                    let curr = array[*next].clone();
                    *next += 1;
                    Some(Value::Compile(curr))
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

                if &*next >= ty_len {
                    return None;
                }
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
            }
        }
    }
}

fn pair_compile_int(
    left: Spanned<Value<BigInt, HardwareValue<ClosedIncRange<BigInt>>>>,
    right: Spanned<Value<BigInt, HardwareValue<ClosedIncRange<BigInt>>>>,
) -> Value<
    (Spanned<BigInt>, Spanned<BigInt>),
    (
        Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
        Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
    ),
> {
    pair_compile_general(left, right, |x| {
        Result::<_, Never>::Ok(HardwareValue {
            ty: ClosedIncRange::single(x.inner.clone()),
            domain: ValueDomain::CompileTime,
            expr: IrExpression::Int(x.inner),
        })
    })
    .remove_never()
}

fn pair_compile_general<C, T, E>(
    left: Spanned<Value<C, T>>,
    right: Spanned<Value<C, T>>,
    to_other: impl Fn(Spanned<C>) -> Result<T, E>,
) -> Result<Value<(Spanned<C>, Spanned<C>), (Spanned<T>, Spanned<T>)>, E> {
    match (left.inner, right.inner) {
        (Value::Compile(left_inner), Value::Compile(right_inner)) => Ok(Value::Compile((
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
                Value::Compile(left_inner) => to_other(Spanned {
                    span: left.span,
                    inner: left_inner,
                }),
                Value::Hardware(left_inner) => Ok(left_inner),
            };
            let right_inner = match right_inner {
                Value::Compile(right_inner) => to_other(Spanned {
                    span: right.span,
                    inner: right_inner,
                }),
                Value::Hardware(right_inner) => Ok(right_inner),
            };

            let left_inner = left_inner?;
            let right_inner = right_inner?;

            Ok(Value::Hardware((
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

// Proofs of the validness of the integer ranges can be found in `int_range_proofs.py`.
pub fn eval_binary_expression(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    expr_span: Span,
    op: Spanned<BinaryOp>,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
) -> DiagResult<ValueWithImplications> {
    let diags = refs.diags;
    let op_reason = TypeContainsReason::Operator(op.span);

    let check_both_int = |left, right| {
        let left = check_type_is_int(diags, op_reason, left);
        let right = check_type_is_int(diags, op_reason, right);
        Ok((left?, right?))
    };
    let eval_binary_bool = |large, left, right, op| eval_binary_bool(diags, large, op_reason, left, right, op);
    let eval_binary_int_compare =
        |large, left, right, op| eval_binary_int_compare(diags, large, op_reason, left, right, op);

    let result_simple: Value<_> = match op.inner {
        // (int, int)
        BinaryOp::Add => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;
            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => Value::Compile(CompileValue::Int(left.inner + right.inner)),
                Value::Hardware((left, right)) => {
                    let range = ClosedIncRange {
                        start_inc: &left.inner.ty.start_inc + &right.inner.ty.start_inc,
                        end_inc: &left.inner.ty.end_inc + &right.inner.ty.end_inc,
                    };
                    Value::Hardware(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Add,
                        large,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Sub => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;
            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => Value::Compile(CompileValue::Int(left.inner - right.inner)),
                Value::Hardware((left, right)) => {
                    let range = ClosedIncRange {
                        start_inc: &left.inner.ty.start_inc - &right.inner.ty.end_inc,
                        end_inc: &left.inner.ty.end_inc - &right.inner.ty.start_inc,
                    };
                    Value::Hardware(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Sub,
                        large,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Mul => {
            // TODO do we want to keep using multiplication as the "array repeat" syntax?
            //   if so, maybe allow tuples on the right side for multidimensional repeating
            let right = check_type_is_int(diags, op_reason, right.map_inner(|e| e.into_value()));
            match left.inner.ty() {
                Type::Array(left_ty_inner, left_len) => {
                    let right = right?;
                    let right_inner = match right.inner {
                        Value::Compile(right_inner) => right_inner,
                        Value::Hardware(_) => {
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
                            format!("got value `{}`", right_inner),
                        )
                    })?;
                    let right_inner = usize::try_from(right_inner).map_err(|right_inner| {
                        diags.report_simple(
                            "array repetition right hand side too large",
                            right.span,
                            format!("got value `{}`", right_inner),
                        )
                    })?;

                    match left.inner.into_value() {
                        Value::Compile(CompileValue::Array(left_inner)) => {
                            // do the repetition at compile-time
                            // TODO check for overflow (everywhere)
                            let mut result = Vec::with_capacity(left_inner.len() * right_inner);
                            for _ in 0..right_inner {
                                result.extend_from_slice(&left_inner);
                            }
                            Value::Compile(CompileValue::Array(Arc::new(result)))
                        }
                        Value::Compile(_) => {
                            return Err(diags.report_internal_error(
                                left.span,
                                "compile-time value with type array is not actually an array",
                            ))
                        }
                        Value::Hardware(value) => {
                            // implement runtime repetition through spread array literal
                            let element = IrArrayLiteralElement::Spread(value.expr);
                            let elements = vec![element; right_inner];

                            let left_ty_inner_hw = left_ty_inner.as_hardware_type(refs).unwrap();
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
                    }
                }
                Type::Int(_) => {
                    let left = check_type_is_int(diags, op_reason, left.map_inner(|e| e.into_value()))
                        .expect("int, already checked");
                    let right = right?;
                    match pair_compile_int(left, right) {
                        Value::Compile((left, right)) => Value::Compile(CompileValue::Int(left.inner * right.inner)),
                        Value::Hardware((left, right)) => {
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
                            let result =
                                build_binary_int_arithmetic_op(IrIntArithmeticOp::Mul, large, range, left, right);
                            Value::Hardware(result)
                        }
                    }
                }
                _ => {
                    return Err(diags.report_simple(
                        "left hand side of multiplication must be an array or an integer",
                        left.span,
                        format!("got value with type `{}`", left.inner.ty().diagnostic_string()),
                    ))
                }
            }
        }
        // (int, non-zero int)
        BinaryOp::Div => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;

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
            let right_positive = right.inner.range().start_inc > &BigInt::ZERO;

            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => {
                    let result = left.inner.div_floor(&right.inner).unwrap();
                    Value::Compile(CompileValue::Int(result))
                }
                Value::Hardware((left, right)) => {
                    let a_min = &left.inner.ty.start_inc;
                    let a_max = &left.inner.ty.end_inc;
                    let b_min = &right.inner.ty.start_inc;
                    let b_max = &right.inner.ty.end_inc;
                    let range = if right_positive {
                        ClosedIncRange {
                            start_inc: min(a_min.div_floor(b_max).unwrap(), a_min.div_floor(b_min).unwrap()),
                            end_inc: max(a_max.div_floor(b_max).unwrap(), a_max.div_floor(b_min).unwrap()),
                        }
                    } else {
                        ClosedIncRange {
                            start_inc: min(a_max.div_floor(b_max).unwrap(), a_max.div_floor(b_min).unwrap()),
                            end_inc: max(a_min.div_floor(b_max).unwrap(), a_min.div_floor(b_min).unwrap()),
                        }
                    };

                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Div, large, range, left, right);
                    Value::Hardware(result)
                }
            }
        }
        BinaryOp::Mod => {
            let (left, right) =
                check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;

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
            let right_positive = right.inner.range().start_inc > &BigInt::ZERO;

            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => {
                    let result = left.inner.mod_floor(&right.inner).unwrap();
                    Value::Compile(CompileValue::Int(result))
                }
                Value::Hardware((left, right)) => {
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

                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Mod, large, range, left, right);
                    Value::Hardware(result)
                }
            }
        }
        // (nonzero int, non-negative int) or (non-negative int, positive int)
        BinaryOp::Pow => {
            let (base, exp) = check_both_int(left.map_inner(|e| e.into_value()), right.map_inner(|e| e.into_value()))?;

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
                Value::Compile((base, exp)) => {
                    let exp = BigUint::try_from(exp.inner)
                        .map_err(|_| diags.report_internal_error(exp.span, "got negative exp"))?;

                    let result = base.inner.pow(&exp);
                    Value::Compile(CompileValue::Int(result))
                }
                Value::Hardware((base, exp)) => {
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
                        let end_exp_sub_one = BigUint::try_from(exp_end_inc.sub(&BigUint::ONE)).unwrap();
                        result_min = min(result_min, base.inner.ty.start_inc.clone().pow(&end_exp_sub_one));
                        result_max = max(result_max, base.inner.ty.start_inc.clone().pow(&end_exp_sub_one));
                    }

                    let range = ClosedIncRange {
                        start_inc: result_min,
                        end_inc: result_max,
                    };
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Pow, large, range, base, exp);
                    Value::Hardware(result)
                }
            }
        }
        // (bool, bool)
        // TODO these should short-circuit, so delay evaluation of right
        BinaryOp::BoolAnd => return eval_binary_bool(large, left, right, IrBoolBinaryOp::And),
        BinaryOp::BoolOr => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Or),
        BinaryOp::BoolXor => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Xor),
        // (T, T)
        // TODO expand eq/neq to bools/tuples/structs/enums, for the latter only if the type is the same
        BinaryOp::CmpEq => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Eq),
        BinaryOp::CmpNeq => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Neq),
        BinaryOp::CmpLt => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Lt),
        BinaryOp::CmpLte => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Lte),
        BinaryOp::CmpGt => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Gt),
        BinaryOp::CmpGte => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Gte),
        // (int, range)
        // TODO share code with match "in" pattern
        BinaryOp::In => return Err(diags.report_todo(expr_span, "binary op In")),
        // (bool, bool)
        // TODO support boolean arrays
        BinaryOp::BitAnd => return eval_binary_bool(large, left, right, IrBoolBinaryOp::And),
        BinaryOp::BitOr => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Or),
        BinaryOp::BitXor => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Xor),
        // TODO (boolean array, non-negative int) and maybe (non-negative int, non-negative int),
        //   and maybe even negative shift amounts?
        BinaryOp::Shl => return Err(diags.report_todo(expr_span, "binary op Shl")),
        BinaryOp::Shr => return Err(diags.report_todo(expr_span, "binary op Shr")),
    };

    Ok(ValueWithImplications::simple(result_simple))
}

fn eval_binary_bool(
    diags: &Diagnostics,
    large: &mut IrLargeArena,
    op_reason: TypeContainsReason,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
    op: IrBoolBinaryOp,
) -> DiagResult<ValueWithImplications> {
    fn build_bool_gate(
        f: impl Fn(bool) -> bool,
        large: &mut IrLargeArena,
        inner_eval: HardwareValueWithImplications,
        inner_ir: HardwareValue<()>,
    ) -> ValueWithImplications {
        match (f(false), f(true)) {
            // constants
            (false, false) => ValueWithImplications::simple(Value::Compile(CompileValue::Bool(false))),
            (true, true) => ValueWithImplications::simple(Value::Compile(CompileValue::Bool(true))),
            // pass gate
            (false, true) => ValueWithImplications::Hardware(inner_eval),
            // not gate
            (true, false) => ValueWithImplications::Hardware(HardwareValueWithImplications {
                value: HardwareValue {
                    ty: HardwareType::Bool,
                    domain: inner_ir.domain,
                    expr: large.push_expr(IrExpressionLarge::BoolNot(inner_ir.expr)),
                },
                version: None,
                implications: inner_eval.implications.invert(),
            }),
        }
    }

    let left_value = check_type_is_bool(
        diags,
        op_reason,
        left.clone().map_inner(ValueWithImplications::into_value),
    );
    let right_value = check_type_is_bool(
        diags,
        op_reason,
        right.clone().map_inner(ValueWithImplications::into_value),
    );

    let left_value = left_value?;
    let right_value = right_value?;

    match (left_value.inner, right_value.inner) {
        // full compile-tim eval
        (Value::Compile(left), Value::Compile(right)) => {
            let result = CompileValue::Bool(op.eval(left, right));
            Ok(ValueWithImplications::simple(Value::Compile(result)))
        }
        // partial compile-time eval
        (Value::Compile(left_value), Value::Hardware(right_value)) => Ok(build_bool_gate(
            |b| op.eval(left_value, b),
            large,
            right.inner.unwrap_hardware(),
            right_value,
        )),
        (Value::Hardware(left_value), Value::Compile(right_value)) => Ok(build_bool_gate(
            |b| op.eval(b, right_value),
            large,
            left.inner.unwrap_hardware(),
            left_value,
        )),
        // full hardware
        (Value::Hardware(left_value), Value::Hardware(right_value)) => {
            let expr = HardwareValue {
                ty: HardwareType::Bool,
                domain: left_value.domain.join(right_value.domain),
                expr: large.push_expr(IrExpressionLarge::BoolBinary(op, left_value.expr, right_value.expr)),
            };

            let left_inner = left.inner.unwrap_hardware();
            let right_inner = right.inner.unwrap_hardware();

            let implications = match op {
                IrBoolBinaryOp::And => BoolImplications {
                    if_true: vec_concat([left_inner.implications.if_true, right_inner.implications.if_true]),
                    if_false: vec![],
                },
                IrBoolBinaryOp::Or => BoolImplications {
                    if_true: vec![],
                    if_false: vec_concat([left_inner.implications.if_false, right_inner.implications.if_false]),
                },
                IrBoolBinaryOp::Xor => BoolImplications::default(),
            };

            Ok(ValueWithImplications::Hardware(HardwareValueWithImplications {
                value: expr,
                version: None,
                implications,
            }))
        }
    }
}

fn eval_binary_int_compare(
    diags: &Diagnostics,
    large: &mut IrLargeArena,
    op_reason: TypeContainsReason,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
    op: IrIntCompareOp,
) -> DiagResult<ValueWithImplications> {
    let left_int = check_type_is_int(
        diags,
        op_reason,
        left.clone().map_inner(ValueWithImplications::into_value),
    );
    let right_int = check_type_is_int(
        diags,
        op_reason,
        right.clone().map_inner(ValueWithImplications::into_value),
    );
    let left_int = left_int?;
    let right_int = right_int?;

    // TODO spans are getting unnecessarily complicated, eg. why does this try to propagate spans?
    match pair_compile_int(left_int, right_int) {
        Value::Compile((left, right)) => {
            let result = op.eval(&left.inner, &right.inner);
            Ok(ValueWithImplications::simple(Value::Compile(CompileValue::Bool(
                result,
            ))))
        }
        Value::Hardware((left_int, right_int)) => {
            let lv = match left.inner {
                ValueWithImplications::Compile(_) => None,
                ValueWithImplications::Hardware(v) => v.version,
            };
            let rv = match right.inner {
                ValueWithImplications::Compile(_) => None,
                ValueWithImplications::Hardware(v) => v.version,
            };

            // TODO warning if the result is always true/false (depending on the ranges)
            //   or maybe just return a compile-time value again?
            // build implications
            let lr = left_int.inner.ty;
            let rr = right_int.inner.ty;
            let implications = match op {
                IrIntCompareOp::Lt => implications_lt(lv, lr, rv, rr),
                IrIntCompareOp::Lte => implications_lte(lv, lr, rv, rr),
                IrIntCompareOp::Gt => implications_lt(rv, rr, lv, lr),
                IrIntCompareOp::Gte => implications_lte(rv, rr, lv, lr),
                IrIntCompareOp::Eq => implications_eq(lv, lr, rv, rr),
                IrIntCompareOp::Neq => implications_eq(lv, lr, rv, rr).invert(),
            };

            // build the resulting expression
            let result = HardwareValue {
                ty: HardwareType::Bool,
                domain: left_int.inner.domain.join(right_int.inner.domain),
                expr: large.push_expr(IrExpressionLarge::IntCompare(
                    op,
                    left_int.inner.expr,
                    right_int.inner.expr,
                )),
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
    range: ClosedIncRange<BigInt>,
    left: Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
    right: Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
) -> HardwareValue {
    let result_expr = IrExpressionLarge::IntArithmetic(op, range.clone(), left.inner.expr, right.inner.expr);
    HardwareValue {
        ty: HardwareType::Int(range),
        domain: left.inner.domain.join(right.inner.domain),
        expr: large.push_expr(result_expr),
    }
}

fn implications_lt(
    left: Option<ValueVersion>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersion>,
    right_range: ClosedIncRange<BigInt>,
) -> BoolImplications {
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

    BoolImplications { if_true, if_false }
}

fn implications_lte(
    left: Option<ValueVersion>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersion>,
    right_range: ClosedIncRange<BigInt>,
) -> BoolImplications {
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

    BoolImplications { if_true, if_false }
}

fn implications_eq(
    left: Option<ValueVersion>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersion>,
    right_range: ClosedIncRange<BigInt>,
) -> BoolImplications {
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

    BoolImplications { if_true, if_false }
}

fn array_literal_combine_values(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    expr_span: Span,
    expected_ty_inner: &Type,
    values: Vec<ArrayLiteralElement<Spanned<Value>>>,
) -> DiagResult<Value> {
    let diags = refs.diags;

    let first_non_compile_span = values
        .iter()
        .find(|v| !matches!(v.value().inner, Value::Compile(_)))
        .map(|v| v.span());
    if let Some(first_non_compile_span) = first_non_compile_span {
        // at least one non-compile, turn everything into IR
        let expected_ty_inner = match expected_ty_inner {
            Type::Any => {
                // infer type based on elements
                let mut ty_joined = Type::Undefined;
                for value in &values {
                    let value_ty = match value {
                        ArrayLiteralElement::Single(value) => value.inner.ty(),
                        ArrayLiteralElement::Spread(_, values) => match values.inner.ty() {
                            Type::Array(ty, _) => Arc::unwrap_or_clone(ty),
                            _ => Type::Undefined,
                        },
                    };
                    ty_joined = ty_joined.union(&value_ty, false);
                }
                ty_joined
            }
            _ => expected_ty_inner.clone(),
        };

        let expected_ty_inner_hw = expected_ty_inner.as_hardware_type(refs).map_err(|_| {
            // TODO clarify that inferred type comes from outside, not the expression itself
            let message = format!(
                "hardware array literal has inferred inner type `{}` which is not representable in hardware",
                expected_ty_inner.diagnostic_string()
            );
            let diag = Diagnostic::new("hardware array type needs to be representable in hardware")
                .add_error(expr_span, message)
                .add_info(first_non_compile_span, "necessary because this array element is not a compile-time value, which forces the entire array to be hardware")
                .finish();
            diags.report(diag)
        })?;

        let mut result_domain = ValueDomain::CompileTime;
        let mut result_exprs = Vec::with_capacity(values.len());
        let mut result_len = BigUint::ZERO;

        for elem in values {
            let (elem_ir, domain, elem_len) = match elem {
                ArrayLiteralElement::Single(elem_inner) => {
                    let value_ir =
                        elem_inner
                            .inner
                            .as_hardware_value(refs, large, elem_inner.span, &expected_ty_inner_hw)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(expr_span),
                        &expected_ty_inner,
                        Spanned {
                            span: elem_inner.span,
                            inner: &Value::Hardware(value_ir.clone()),
                        },
                        true,
                        true,
                    )?;

                    (
                        IrArrayLiteralElement::Single(value_ir.expr),
                        value_ir.domain,
                        BigUint::ONE,
                    )
                }
                ArrayLiteralElement::Spread(_, elem_inner) => {
                    let value_ir =
                        elem_inner
                            .inner
                            .as_hardware_value(refs, large, elem_inner.span, &expected_ty_inner_hw)?;

                    let len = match value_ir.ty() {
                        Type::Array(_, len) => len,
                        _ => BigUint::ZERO,
                    };
                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(expr_span),
                        &Type::Array(Arc::new(expected_ty_inner.clone()), len.clone()),
                        Spanned {
                            span: elem_inner.span,
                            inner: &Value::Hardware(value_ir.clone()),
                        },
                        true,
                        true,
                    )?;

                    (IrArrayLiteralElement::Spread(value_ir.expr), value_ir.domain, len)
                }
            };

            result_domain = result_domain.join(domain);
            result_exprs.push(elem_ir);
            result_len += elem_len;
        }

        let result_expr =
            IrExpressionLarge::ArrayLiteral(expected_ty_inner_hw.as_ir(refs), result_len.clone(), result_exprs);
        Ok(Value::Hardware(HardwareValue {
            ty: HardwareType::Array(Arc::new(expected_ty_inner_hw), result_len),
            domain: result_domain,
            expr: large.push_expr(result_expr),
        }))
    } else {
        // all compile, create compile value
        let mut result = Vec::with_capacity(values.len());
        for elem in values {
            match elem {
                ArrayLiteralElement::Single(elem_inner) => {
                    let elem_inner = unwrap_match!(elem_inner.inner, Value::Compile(v) => v);
                    result.push(elem_inner);
                }
                ArrayLiteralElement::Spread(span_spread, elem_inner) => {
                    let elem_inner = unwrap_match!(elem_inner.inner, Value::Compile(v) => v);
                    let elem_inner_array = match elem_inner {
                        CompileValue::Array(elem_inner) => elem_inner,
                        _ => {
                            return Err(diags.report_todo(
                                span_spread,
                                "compile-time spread only works for fully known arrays for now",
                            ))
                        }
                    };
                    result.extend(elem_inner_array.iter().cloned())
                }
            }
        }
        Ok(Value::Compile(CompileValue::Array(Arc::new(result))))
    }
}
