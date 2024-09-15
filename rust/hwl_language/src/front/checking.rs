use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed, ResultOrGuaranteed};
use crate::front::driver::{CompileState, EvalTrueError};
use crate::front::types::{IntegerTypeInfo, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::BinaryOp;
use crate::syntax::pos::Span;
use crate::try_opt_result;
use num_bigint::BigInt;
use num_traits::One;

impl CompileState<'_, '_> {
    // TODO double-check that all of these type-checking functions do their error handling correctly
    //   and don't short-circuit unnecessarily, preventing multiple errors
    pub fn check_type_contains(&self, span_ty: Span, span_value: Span, ty: &Type, value: &Value) -> ResultOrGuaranteed<()> {
        match (ty, value) {
            (&Type::Error(e), _) | (_, &Value::Error(e)) => return Err(e),
            (Type::Range, Value::Range(_)) => return Ok(()),
            (Type::Integer(IntegerTypeInfo { range }), value) => {
                if let Value::Range(range) = range.as_ref() {
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
            _ => {}
        }

        let feature = format!("check_type_contains {:?} {:?}", ty, value);
        self.diag.report_todo(span_value, &feature);
        Ok(())
    }

    pub fn require_value_true_for_range(&self, span_range: Span, value: &Value) -> ResultOrGuaranteed<()> {
        self.try_eval_bool_true(span_range, value).map_err(|e| {
            let value_str = self.compiled.value_to_readable_str(self.source, value);
            let title = format!("range valid check failed: value {} {}", value_str, e.to_message());
            let err = Diagnostic::new(title)
                .add_error(span_range, "when checking that this range is non-decreasing")
                .finish();
            self.diag.report(err).into()
        })
    }

    pub fn require_value_true_for_type_check(&self, span_ty: Span, span_value: Span, value: &Value) -> ResultOrGuaranteed<()> {
        self.try_eval_bool_true(span_value, value).map_err(|e| {
            let value_str = self.compiled.value_to_readable_str(self.source, value);
            let title = format!("type check failed: value {} {}", value_str, e.to_message());
            let err = Diagnostic::new(title)
                .add_error(span_value, "when type checking this value")
                .add_info(span_ty, "against this type")
                .finish();
            self.diag.report(err).into()
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
    pub fn try_eval_bool(&self, origin: Span, value: &Value) -> ResultOrGuaranteed<Option<bool>> {
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

    pub fn try_eval_bool_inner(&self, origin: Span, value: &Value) -> ResultOrGuaranteed<Option<bool>> {
        // TODO this is wrong, we should be returning None a lot more, eg. if the ranges of the operands are not tight
        match *value {
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
    pub fn range_of_value(&self, origin: Span, value: &Value) -> ResultOrGuaranteed<Option<RangeInfo<Box<Value>>>> {
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

    pub fn range_of_value_inner(&self, origin: Span, value: &Value) -> ResultOrGuaranteed<Option<RangeInfo<Box<Value>>>> {
        // TODO if range ends are themselves params with ranges, assume the worst case
        //   although that misses things like (n < n+1)
        fn ty_as_range(ty: &Type) -> ResultOrGuaranteed<Option<RangeInfo<Box<Value>>>> {
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

        match *value {
            Value::Error(e) => Err(e),
            // params have types which we can use to extract a range
            Value::GenericParameter(param) => ty_as_range(&self.compiled[param].ty),
            Value::FunctionParameter(param) => ty_as_range(&self.compiled[param].ty),
            // a single integer corresponds to the range containing only that integer
            // TODO should we generate an inclusive or exclusive range here?
            //   this will become moot once we switch to +1 deltas
            Value::Int(ref value) => Ok(Some(RangeInfo {
                start: Some(Box::new(Value::Int(value.clone()))),
                end: Some(Box::new(Value::Int(value + 1u32))),
                end_inclusive: false,
            })),
            Value::ModulePort(_) => Ok(None),
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
                                    let right_end_inclusive = Box::new(Value::Binary(BinaryOp::Sub, right_end.clone(), Box::new(Value::Int(BigInt::one()))));
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
                            self.diag.report(err)
                        })?;
                        let cond = Value::Binary(BinaryOp::CmpLte, Box::new(Value::Int(BigInt::ZERO)), right_start.clone());
                        self.try_eval_bool_true(origin, &cond)
                            .map_err(|e| {
                                let cond_str = self.compiled.value_to_readable_str(self.source, &cond);
                                let title = format!("power exponent range check failed: value {} {}", cond_str, e.to_message());
                                let err = Diagnostic::new(title)
                                    .add_error(origin, "while checking this expression")
                                    .finish();
                                self.diag.report(err)
                            })?;

                        let left_start = try_opt_result!(left.start);

                        // if base is >0, then the result is >0 too
                        // TODO this range can be improved a lot: consider cases +,0,- separately
                        let base_positive = self.try_eval_bool(origin, &Value::Binary(BinaryOp::CmpLt, Box::new(Value::Int(BigInt::ZERO)), left_start))?;
                        if base_positive == Some(true) {
                            Ok(Some(RangeInfo {
                                start: Some(Box::new(Value::Int(BigInt::one()))),
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
            Value::Function(_) => panic!("function can't have a range type"),
            Value::Module(_) => panic!("module can't have a range type"),
            // TODO get their types
            Value::Wire => Ok(None),
            Value::Reg => Ok(None),
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
                    BinaryOp::Add => return Value::Int(left + right),
                    BinaryOp::Sub => return Value::Int(left - right),
                    BinaryOp::Mul => return Value::Int(left * right),
                    _ => {}
                }
            }

            Value::Binary(op, Box::new(left), Box::new(right))
        }
        // TODO at least recursively call simplify
        value => value,
    }
}

pub fn value_as_int(value: &Value) -> Option<&BigInt> {
    match value {
        Value::Int(value) => Some(value),
        _ => None,
    }
}

fn option_pair<A, B>(left: Option<A>, right: Option<B>) -> Option<(A, B)> {
    match (left, right) {
        (Some(left), Some(right)) => Some((left, right)),
        _ => None,
    }
}
