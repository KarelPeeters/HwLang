use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::common::ValueDomainKind;
use crate::front::driver::{CompileState, EvalTrueError};
use crate::front::types::{IntegerTypeInfo, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::{BinaryOp, PortKind, SyncDomain};
use crate::syntax::pos::Span;
use crate::try_opt_result;
use num_bigint::BigInt;
use num_traits::One;

impl CompileState<'_, '_> {
    pub fn domain_of_value(&self, span: Span, value: &Value) -> ValueDomainKind {
        let diags = self.diags;

        match value {
            &Value::Error(e) => ValueDomainKind::Error(e),
            &Value::ModulePort(port) => {
                match &self.compiled[port].kind {
                    PortKind::Clock => ValueDomainKind::Clock,
                    PortKind::Normal { domain, ty: _ } => ValueDomainKind::from_domain_kind(domain.clone()),
                }
            }
            &Value::GenericParameter(_) => ValueDomainKind::Const,
            &Value::Never => ValueDomainKind::Const,
            &Value::Unit => ValueDomainKind::Const,
            &Value::InstConstant(_) => ValueDomainKind::Const,
            Value::Range(info) => {
                let RangeInfo { start, end, end_inclusive: _ } = info;

                let start = start.as_ref().map(|v| self.domain_of_value(span, v));
                let end = end.as_ref().map(|v| self.domain_of_value(span, v));

                match (start, end) {
                    (None, None) =>
                        ValueDomainKind::Const,
                    (Some(single), None) | (None, Some(single)) =>
                        single,
                    (Some(start), Some(end)) =>
                        self.merge_domains(span, &start, &end),
                }
            }
            Value::Binary(_, left, right) =>
                self.merge_domains(span, &self.domain_of_value(span, left), &self.domain_of_value(span, right)),
            Value::UnaryNot(inner) =>
                self.domain_of_value(span, inner),
            // TODO just join all argument domains
            &Value::FunctionReturn(_) =>
                ValueDomainKind::Error(diags.report_todo(span, "domain of function return value")),
            &Value::Module(_) =>
                ValueDomainKind::Error(diags.report_simple("cannot get domain of module value", span, "module")),
            &Value::Wire =>
                ValueDomainKind::Error(diags.report_todo(span, "domain of wire value")),
            &Value::Register(reg) =>
                ValueDomainKind::Sync(self.compiled[reg].domain.clone()),
            // TODO this is a bit confusing, the origin of the variable matters!
            &Value::Variable(_) =>
                ValueDomainKind::Error(diags.report_todo(span, "domain of variable value")),
        }
    }

    /// Merge two sync domains, as they would be if they were used as part of a single expression.
    /// Reports an error if this is not possible.
    pub fn merge_domains(&self, span: Span, left: &ValueDomainKind, right: &ValueDomainKind) -> ValueDomainKind {
        match (left, right) {
            // propagate errors
            (&ValueDomainKind::Error(e), _) | (_, &ValueDomainKind::Error(e)) => ValueDomainKind::Error(e),
            // const can merge with anything and become that other domain
            (ValueDomainKind::Const, other) | (other, ValueDomainKind::Const) => other.clone(),
            // sync can merge if both domains match
            (ValueDomainKind::Sync(left), ValueDomainKind::Sync(right)) => {
                match sync_domains_equal(left, right) {
                    Ok(SyncDomainsEqual::Equal) => ValueDomainKind::Sync(left.clone()),
                    Ok(SyncDomainsEqual::NotEqual(reason)) => {
                        let label = format!(
                            "{}: domains {} and {}",
                            reason,
                            self.compiled.sync_kind_to_readable_string(&self.source, &ValueDomainKind::Sync(left.clone())),
                            self.compiled.sync_kind_to_readable_string(&self.source, &ValueDomainKind::Sync(right.clone())),
                        );
                        let e = self.diags.report_simple("cannot merge different sync domains", span, label);
                        ValueDomainKind::Error(e)
                    }
                    Err(e) => ValueDomainKind::Error(e),
                }
            }
            (ValueDomainKind::Clock, _) | (_, ValueDomainKind::Clock) => {
                ValueDomainKind::Error(self.diags.report_simple("cannot merge clock domain with anything", span, "clock domain"))
            }
            (ValueDomainKind::Async, _) | (_, ValueDomainKind::Async) => {
                ValueDomainKind::Error(self.diags.report_simple("cannot merge async domain with anything", span, "async domain"))
            }
        }
    }

    // TODO double-check that all of these type-checking functions do their error handling correctly
    //   and don't short-circuit unnecessarily, preventing multiple errors
    // TODO change this to be type-type based instead of type-value
    // TODO move error formatting out of this function, it depends too much on the context and spans are not always available
    pub fn check_type_contains(&self, span_ty: Option<Span>, span_value: Span, ty: &Type, value: &Value) -> Result<(), ErrorGuaranteed> {
        match (ty, value) {
            // propagate errors, we can't just silently ignore them:
            //   downstream compiler code might actually depend on the type matching
            (&Type::Error(e), _) | (_, &Value::Error(e)) => return Err(e),

            // any contains everything
            (&Type::Any, _) => return Ok(()),

            // basic types that only contain themselves
            (Type::Unit, Value::Unit) => return Ok(()),
            (Type::Never, Value::Never) => return Ok(()),
            // TODO differentiate between arbitrary open, half-open, ...
            (Type::Range, Value::Range(_)) => return Ok(()),

            // integer range check
            (Type::Integer(IntegerTypeInfo { range }), value) => {
                // only continue checking if the value has at least a range, which means that it's an integer
                match self.range_of_value(span_value, value) {
                    Ok(Some(_)) => {
                        if let Value::Range(range) = range.as_ref() {
                            // check that the value fits in the required range
                            let &RangeInfo { ref start, ref end, end_inclusive } = range;
                            if let Some(start) = start {
                                let cond = Value::Binary(BinaryOp::CmpLte, start.clone(), Box::new(value.clone()));
                                self.require_value_true_for_type_check(span_ty, span_value, &cond)?;
                            }
                            if let Some(end) = end {
                                let cmp_op = if end_inclusive { BinaryOp::CmpLte } else { BinaryOp::CmpLt };
                                let cond = Value::Binary(cmp_op, Box::new(value.clone()), end.clone());
                                self.require_value_true_for_type_check(span_ty, span_value, &cond)?;
                            }
                            return Ok(());
                        }
                    }
                    // fallthrough into error
                    Ok(None) => {}
                    Err(e) => return Err(e),
                }
            }

            // type-type checks
            // TODO type equality is not good enough (and should be removed entirely), eg. this does not support broadcasting
            (ty, &Value::GenericParameter(param)) => {
                if ty == &self.compiled[param].ty {
                    return Ok(());
                }
            }
            (ty, &Value::ModulePort(port)) => {
                // TODO weird, will solve itself once we switch to type-type
                if let PortKind::Normal { domain: _, ty: ref port_ty } = self.compiled[port].kind {
                    if ty == port_ty {
                        return Ok(());
                    }
                }
            }
            (ty, &Value::Variable(var)) => {
                if ty == &self.compiled[var].ty {
                    return Ok(());
                }
            }
            (ty, &Value::Register(reg)) => {
                if ty == &self.compiled[reg].ty {
                    return Ok(());
                }
            }

            // unary not does not change the type
            // TODO check that the input is a boolean or bits?
            (ty, Value::UnaryNot(inner)) => {
                return self.check_type_contains(span_ty, span_value, ty, inner);
            }

            // fallthrough into error
            _ => {}
        };

        let ty_str = self.compiled.type_to_readable_str(self.source, ty);
        let value_str = self.compiled.value_to_readable_str(self.source, value);
        let title = format!("type mismatch: value {} does not match type {}", value_str, ty_str);
        let err = Diagnostic::new(title)
            .add_error(span_value, "value used here")
            .add_info_maybe(span_ty, "type defined here")
            .finish();
        Err(self.diags.report(err))
    }

    pub fn require_value_true_for_range(&self, span_range: Span, value: &Value) -> Result<(), ErrorGuaranteed> {
        self.try_eval_bool_true(span_range, value).map_err(|e| {
            let value_str = self.compiled.value_to_readable_str(self.source, value);
            let title = format!("range valid check failed: value {} {}", value_str, e.to_message());
            let err = Diagnostic::new(title)
                .add_error(span_range, "when checking that this range is non-decreasing")
                .finish();
            self.diags.report(err).into()
        })
    }

    pub fn require_value_true_for_type_check(&self, span_ty: Option<Span>, span_value: Span, value: &Value) -> Result<(), ErrorGuaranteed> {
        self.try_eval_bool_true(span_value, value).map_err(|e| {
            let value_str = self.compiled.value_to_readable_str(self.source, value);
            let title = format!("type check failed: value {} {}", value_str, e.to_message());
            // TODO include the type of the value and the target type, with generics it's not always obvious
            let err = Diagnostic::new(title)
                .add_error(span_value, "when type checking this value")
                .add_info_maybe(span_ty, "against this type")
                .finish();
            self.diags.report(err).into()
        })
    }

    pub fn try_eval_bool_true(&self, origin: Span, value: &Value) -> Result<(), EvalTrueError> {
        match self.try_eval_bool(origin, value) {
            Err(e) => {
                // accept errors as correct to prevent further downstream errors
                let _: ErrorGuaranteed = e;
                Ok(())
            }
            Ok(Some(true)) => Ok(()),
            Ok(Some(false)) => Err(EvalTrueError::False),
            Ok(None) => Err(EvalTrueError::Unknown),
        }
    }

    // TODO what is the general algorithm for this? equivalence graphs?
    // TODO add boolean-proving cache
    // TODO it it possible to keep boolean proving and type inference separate? it probably is
    //   if we don't allow user-defined type selections
    //   even then, we can probably brute-force our way through those relatively easily
    // TODO convert lte/gte into +1/-1 fixes instead?
    // TODO convert inclusive/exclusive into +1/-1 fixes instead?
    // TODO check lt, lte, gt, gte, ... all together elegantly
    // TODO return true for vacuous truths, eg. comparisons between empty ranges?
    pub fn try_eval_bool(&self, origin: Span, value: &Value) -> Result<Option<bool>, ErrorGuaranteed> {
        let result = self.try_eval_bool_inner(origin, value);
        if self.log_const_eval {
            eprintln!(
                "try_eval_bool({}) -> {:?}",
                self.compiled.value_to_readable_str(self.source, value),
                result
            );
        }
        result
    }

    pub fn try_eval_bool_inner(&self, origin: Span, value: &Value) -> Result<Option<bool>, ErrorGuaranteed> {
        // TODO this is wrong, we should be returning None a lot more, eg. if the ranges of the operands are not tight
        match *value {
            Value::Error(e) => return Err(e),
            Value::Binary(binary_op, ref left, ref right) => {
                let left = try_opt_result!(self.range_of_value(origin, left)?);
                let right = try_opt_result!(self.range_of_value(origin, right)?);

                let compare_lt = |allow_eq: bool| {
                    let end_delta = if left.end_inclusive { 0 } else { 1 };
                    let left_end = value_as_int(left.end.as_ref()?)? - end_delta;
                    let right_start = value_as_int(right.start.as_ref()?)?;

                    if allow_eq {
                        Some(&left_end <= right_start)
                    } else {
                        Some(&left_end < right_start)
                    }
                };

                match binary_op {
                    BinaryOp::CmpLt => return Ok(compare_lt(false)),
                    BinaryOp::CmpLte => return Ok(compare_lt(true)),
                    _ => {}
                }
            }
            // TODO support more values
            _ => {}
        }
        Ok(None)
    }

    // TODO This needs to be split up into a tight and loose range:
    //   we _know_ all values in the tight range are reachable,
    //   and we _know_ no values outside of the loose range are
    pub fn range_of_value(&self, origin: Span, value: &Value) -> Result<Option<RangeInfo<Box<Value>>>, ErrorGuaranteed> {
        let result = self.range_of_value_inner(origin, value)?;
        let result_simplified = result.clone().map(|r| {
            RangeInfo {
                start: r.start.map(|v| Box::new(simplify_value(*v))),
                end: r.end.map(|v| Box::new(simplify_value(*v))),
                end_inclusive: r.end_inclusive,
            }
        });
        if self.log_const_eval {
            let result_str = result.as_ref()
                .map_or("None".to_string(), |r| self.compiled.range_to_readable_str(self.source, &r));
            let result_simple_str = result_simplified.as_ref()
                .map_or("None".to_string(), |r| self.compiled.range_to_readable_str(self.source, &r));
            eprintln!(
                "range_of_value({}) -> {:?} -> {:?}",
                self.compiled.value_to_readable_str(self.source, value),
                result_str,
                result_simple_str,
            );
        }
        Ok(result_simplified)
    }

    fn range_of_value_inner(&self, origin: Span, value: &Value) -> Result<Option<RangeInfo<Box<Value>>>, ErrorGuaranteed> {
        // TODO if range ends are themselves params with ranges, assume the worst case
        //   although that misses things like (n < n+1)
        fn ty_as_range(ty: &Type) -> Result<Option<RangeInfo<Box<Value>>>, ErrorGuaranteed> {
            match ty {
                &Type::Error(e) => Err(e),
                Type::Integer(IntegerTypeInfo { range }) => {
                    match range.as_ref() {
                        &Value::Error(e) => Err(e),
                        Value::Range(range) => Ok(Some(range.clone())),
                        _ => Ok(None),
                    }
                }
                _ => Ok(None)
            }
        }

        // TODO for unchecked type, return unchecked value?

        match *value {
            Value::Error(e) => Err(e),
            // params have types which we can use to extract a range
            Value::GenericParameter(param) => ty_as_range(&self.compiled[param].ty),
            Value::Unit => Ok(None),

            // TODO return the same as error?
            Value::Never => Ok(None),
            // a single integer corresponds to the range containing only that integer
            // TODO should we generate an inclusive or exclusive range here?
            //   this will become moot once we switch to +1 deltas
            Value::InstConstant(ref value) => Ok(Some(RangeInfo {
                start: Some(Box::new(Value::InstConstant(value.clone()))),
                end: Some(Box::new(Value::InstConstant(value + 1u32))),
                end_inclusive: false,
            })),
            Value::ModulePort(port) => {
                match &self.compiled[port].kind {
                    PortKind::Clock => Ok(None),
                    PortKind::Normal { domain: _, ty } => ty_as_range(ty),
                }
            }
            Value::Binary(op, ref left, ref right) => {
                let left = try_opt_result!(self.range_of_value(origin, left)?);
                let right = try_opt_result!(self.range_of_value(origin, right)?);

                // TODO replace end_inclusive with a +1 delta to get rid of this trickiness forever
                if left.end_inclusive || right.end_inclusive {
                    return Ok(None);
                }

                match op {
                    BinaryOp::Add => {
                        let range = RangeInfo {
                            start: option_pair(left.start.as_ref(), right.start.as_ref())
                                .map(|(left_start, right_start)|
                                    Box::new(Value::Binary(BinaryOp::Add, left_start.clone(), right_start.clone()))
                                ),
                            end: option_pair(left.end.as_ref(), right.end.as_ref())
                                .map(|(left_end, right_end)|
                                    Box::new(Value::Binary(BinaryOp::Add, left_end.clone(), right_end.clone()))
                                ),
                            end_inclusive: false,
                        };
                        Ok(Some(range))
                    }
                    BinaryOp::Sub => {
                        let range = RangeInfo {
                            start: option_pair(left.start.as_ref(), right.end.as_ref())
                                .map(|(left_start, right_end)| {
                                    let right_end_inclusive = Box::new(Value::Binary(BinaryOp::Sub, right_end.clone(), Box::new(Value::InstConstant(BigInt::one()))));
                                    Box::new(Value::Binary(BinaryOp::Sub, left_start.clone(), right_end_inclusive))
                                }),
                            end: option_pair(left.end.as_ref(), right.start.as_ref())
                                .map(|(left_end, right_start)|
                                    Box::new(Value::Binary(BinaryOp::Sub, left_end.clone(), right_start.clone()))
                                ),
                            end_inclusive: false,
                        };
                        Ok(Some(range))
                    }
                    BinaryOp::Pow => {
                        // check that exponent is non-negative
                        let right_start = right.start.as_ref().ok_or_else(|| {
                            let right_str = self.compiled.range_to_readable_str(self.source, &right);
                            let title = format!("power exponent cannot be negative, got range without lower bound {:?}", right_str);
                            let err = Diagnostic::new(title)
                                .add_error(origin, "while checking this expression")
                                .finish();
                            self.diags.report(err)
                        })?;
                        let cond = Value::Binary(BinaryOp::CmpLte, Box::new(Value::InstConstant(BigInt::ZERO)), right_start.clone());
                        self.try_eval_bool_true(origin, &cond)
                            .map_err(|e| {
                                let cond_str = self.compiled.value_to_readable_str(self.source, &cond);
                                let title = format!("power exponent range check failed: value {} {}", cond_str, e.to_message());
                                let err = Diagnostic::new(title)
                                    .add_error(origin, "while checking this expression")
                                    .finish();
                                self.diags.report(err)
                            })?;

                        let left_start = try_opt_result!(left.start);

                        // if base is >0, then the result is >0 too
                        // TODO this range can be improved a lot: consider cases +,0,- separately
                        let base_positive = self.try_eval_bool(origin, &Value::Binary(BinaryOp::CmpLt, Box::new(Value::InstConstant(BigInt::ZERO)), left_start))?;
                        if base_positive == Some(true) {
                            Ok(Some(RangeInfo {
                                start: Some(Box::new(Value::InstConstant(BigInt::one()))),
                                end: None,
                                end_inclusive: false,
                            }))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(None),
                }
            }
            Value::UnaryNot(_) => Ok(None),
            Value::Range(_) => panic!("range can't itself have a range type"),
            Value::FunctionReturn(ref ret) => ty_as_range(&ret.ret_ty),
            Value::Module(_) => panic!("module can't have a range type"),
            // TODO get their types
            Value::Wire => Ok(None),
            Value::Register(reg) => ty_as_range(&self.compiled[reg].ty),
            Value::Variable(var) => ty_as_range(&self.compiled[var].ty),
        }
    }
}

pub fn simplify_value(value: Value) -> Value {
    match value {
        Value::Binary(op, left, right) => {
            let left = simplify_value(*left);
            let right = simplify_value(*right);

            if let Some((left, right)) = option_pair(value_as_int(&left), value_as_int(&right)) {
                match op {
                    BinaryOp::Add => return Value::InstConstant(left + right),
                    BinaryOp::Sub => return Value::InstConstant(left - right),
                    BinaryOp::Mul => return Value::InstConstant(left * right),
                    _ => {}
                }
            }

            Value::Binary(op, Box::new(left), Box::new(right))
        }
        // TODO at least recursively call simplify
        value => value,
    }
}

// TODO return error if value is error?
pub fn value_as_int(value: &Value) -> Option<&BigInt> {
    match value {
        Value::InstConstant(value) => Some(value),
        _ => None,
    }
}

fn option_pair<A, B>(left: Option<A>, right: Option<B>) -> Option<(A, B)> {
    match (left, right) {
        (Some(left), Some(right)) => Some((left, right)),
        _ => None,
    }
}

pub enum SyncDomainsEqual {
    Equal,
    NotEqual(&'static str),
}

pub fn sync_domains_equal(left: &SyncDomain<Value>, right: &SyncDomain<Value>) -> Result<SyncDomainsEqual, ErrorGuaranteed> {
    let SyncDomain { clock: target_clock, reset: target_reset } = left;
    let SyncDomain { clock: source_clock, reset: source_reset } = right;

    let value_eq = |a: &Value, b: &Value| {
        match (a, b) {
            // optimistically assume they match
            (&Value::Error(e), _) | (_, &Value::Error(e)) => Err(e),
            // TODO equality is _probably_ the wrong operation for this
            (a, b) => Ok(a == b),
        }
    };

    // we're intentionally not emitting any error if _either_ of them is an error already, to prevent confusion
    match (value_eq(target_clock, source_clock)?, value_eq(target_reset, source_reset)?) {
        (false, false) => Ok(SyncDomainsEqual::NotEqual("different clock and reset")),
        (false, true) => Ok(SyncDomainsEqual::NotEqual("different clock")),
        (true, false) => Ok(SyncDomainsEqual::NotEqual("different reset")),
        (true, true) => Ok(SyncDomainsEqual::Equal),
    }
}
