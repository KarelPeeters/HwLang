use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::common::ValueDomainKind;
use crate::front::driver::{CompileState, EvalTrueError};
use crate::front::types::{IntegerTypeInfo, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::{BinaryOp, PortKind, SyncDomain};
use crate::syntax::pos::Span;
use crate::util::option_pair;
use annotate_snippets::Level;
use num_bigint::BigInt;
use num_traits::One;

// TODO give this a better name
#[derive(Debug, Copy, Clone)]
pub enum DomainUserControlled {
    Target,
    Source,
    Both,
}

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
            // TODO careful about function arguments
            &Value::GenericParameter(_) => ValueDomainKind::Const,
            &Value::Never => ValueDomainKind::Const,
            &Value::Unit => ValueDomainKind::Const,
            &Value::BoolConstant(_) => ValueDomainKind::Const,
            &Value::IntConstant(_) => ValueDomainKind::Const,
            &Value::StringConstant(_) => ValueDomainKind::Const,
            Value::Range(info) => {
                let RangeInfo { start, end } = info;

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
            &Value::Wire(wire) =>
                ValueDomainKind::from_domain_kind(self.compiled[wire].domain.clone()),
            &Value::Register(reg) =>
                ValueDomainKind::Sync(self.compiled[reg].domain.clone()),
            // TODO this is a bit confusing, the origin of the variable matters!
            &Value::Variable(_) =>
                ValueDomainKind::Error(diags.report_todo(span, "domain of variable value")),
            &Value::Constant(_) =>
                ValueDomainKind::Const,
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
                            self.compiled.sync_kind_to_readable_string(&self.source, &self.parsed, &ValueDomainKind::Sync(left.clone())),
                            self.compiled.sync_kind_to_readable_string(&self.source, &self.parsed, &ValueDomainKind::Sync(right.clone())),
                        );
                        let e = self.diags.report_simple("cannot merge different sync domains", span, label);
                        ValueDomainKind::Error(e)
                    }
                    Err(e) => ValueDomainKind::Error(e),
                }
            }
            // function can marge if both domains match
            (ValueDomainKind::FunctionBody(left), ValueDomainKind::FunctionBody(right)) => {
                if left == right {
                    ValueDomainKind::FunctionBody(left.clone())
                } else {
                    let e = self.diags.report_simple("cannot merge different function body domains", span, "function body domains");
                    ValueDomainKind::Error(e)
                }
            }
            // failed merges
            (ValueDomainKind::FunctionBody(_), _) | (_, ValueDomainKind::FunctionBody(_)) => {
                ValueDomainKind::Error(self.diags.report_simple("cannot merge function domain with anything else", span, "sync domain"))
            }
            (ValueDomainKind::Clock, _) | (_, ValueDomainKind::Clock) => {
                ValueDomainKind::Error(self.diags.report_simple("cannot merge clock domain with anything else", span, "clock domain"))
            }
            (ValueDomainKind::Async, _) | (_, ValueDomainKind::Async) => {
                ValueDomainKind::Error(self.diags.report_simple("cannot merge async domain with anything else", span, "async domain"))
            }
        }
    }

    /// Checks whether the source sync domain can be assigned to the target sync domain.
    /// This is equivalent to checking whether source is more contained that target.
    pub fn check_domain_assign(
        &self,
        target_span: Span,
        target: &ValueDomainKind,
        source_span: Span,
        source: &ValueDomainKind,
        user_controlled: DomainUserControlled,
        hint: &str,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;

        let invalid_reason = match (target, source) {
            // propagate errors
            (&ValueDomainKind::Error(e), _) | (_, &ValueDomainKind::Error(e)) =>
                return Err(e),
            // clock assignments are not yet implemented
            (ValueDomainKind::Clock, _) | (_, ValueDomainKind::Clock) =>
                return Err(self.diags.report_todo(target_span.join(source_span), "clock assignment")),
            // const target must have const source
            (ValueDomainKind::Const, ValueDomainKind::Const) => None,
            (ValueDomainKind::Const, ValueDomainKind::Async) => Some("async to const"),
            (ValueDomainKind::Const, ValueDomainKind::Sync(_)) => Some("sync to const"),
            (ValueDomainKind::Const, ValueDomainKind::FunctionBody(_)) => Some("function body to const"),
            // const can be the source of everything
            (ValueDomainKind::Async, ValueDomainKind::Const) => None,
            (ValueDomainKind::Sync(_), ValueDomainKind::Const) => None,
            (ValueDomainKind::FunctionBody(_), ValueDomainKind::Const) => None,
            // async can be the target of everything
            (ValueDomainKind::Async, _) => None,
            // sync cannot be the target of async
            (ValueDomainKind::Sync(_), ValueDomainKind::Async) => Some("async to sync"),
            // sync pair is allowed if clock and reset match
            (ValueDomainKind::Sync(target), ValueDomainKind::Sync(source)) => {
                let SyncDomain { clock: target_clock, reset: target_reset } = target;
                let SyncDomain { clock: source_clock, reset: source_reset } = source;

                // TODO equality is _probably_ the wrong operation for this
                let value_eq = |a: &Value, b: &Value| {
                    match (a, b) {
                        // optimistically assume they match
                        (&Value::Error(_), _) | (_, &Value::Error(_)) => true,
                        (a, b) => a == b,
                    }
                };

                match (value_eq(target_clock, source_clock), value_eq(target_reset, source_reset)) {
                    (false, false) => Some("different clock and reset"),
                    (false, true) => Some("different clock"),
                    (true, false) => Some("different reset"),
                    (true, true) => None,
                }
            }
            // function can only be used in matching function body itself
            (ValueDomainKind::FunctionBody(target), ValueDomainKind::FunctionBody(source)) => {
                if target == source {
                    None
                } else {
                    Some("different function body")
                }
            }
            (ValueDomainKind::Sync(_), ValueDomainKind::FunctionBody(_)) => Some("function body to sync"),
            (ValueDomainKind::FunctionBody(_), ValueDomainKind::Sync(_)) => Some("sync to function body"),
            (ValueDomainKind::FunctionBody(_), ValueDomainKind::Async) => Some("async to function body"),
        };

        let (target_level, source_level) = match user_controlled {
            DomainUserControlled::Target => (Level::Error, Level::Info),
            DomainUserControlled::Source => (Level::Info, Level::Error),
            DomainUserControlled::Both => (Level::Error, Level::Error),
        };

        if let Some(invalid_reason) = invalid_reason {
            let err = Diagnostic::new(format!("unsafe domain crossing: {}", invalid_reason))
                .add(target_level, target_span, format!("target in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, target)))
                .add(source_level, source_span, format!("source in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, source)))
                .footer(Level::Help, hint)
                .finish();
            Err(diags.report(err))
        } else {
            Ok(())
        }
    }


    pub fn type_of_value(&self, span: Span, value: &Value) -> Type {
        let diags = self.diags;
        let value = simplify_value(value.clone());

        let result = match &value {
            &Value::Error(e) => Type::Error(e),
            &Value::GenericParameter(param) =>
                self.compiled[param].ty.clone(),
            &Value::ModulePort(port) => {
                match &self.compiled[port].kind {
                    PortKind::Clock => Type::Clock,
                    PortKind::Normal { domain: _, ty } => ty.clone(),
                }
            }
            Value::Never => Type::Never,
            Value::Unit => Type::Unit,
            Value::BoolConstant(_) => Type::Boolean,
            Value::IntConstant(value) => {
                let range = RangeInfo {
                    start: Some(Box::new(Value::IntConstant(value.clone()))),
                    end: Some(Box::new(Value::IntConstant(value + 1u32))),
                };
                Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
            }
            Value::StringConstant(_) => Type::String,
            Value::Range(_) => Type::Range,
            &Value::Binary(op, ref left, ref right) =>
                self.type_of_binary(span, op, &left, &right),
            Value::UnaryNot(inner) => {
                let inner = self.type_of_value(span, inner);
                match inner {
                    // valid types
                    Type::Boolean | Type::Bits(_) => inner,
                    // other types are not allowed
                    _ => {
                        let inner_ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &inner);
                        let title = format!("unary not only works for boolean or bits type, got {}", inner_ty_str);
                        Type::Error(diags.report_simple(title, span, "for this expression"))
                    }
                }
            }
            Value::FunctionReturn(func) => func.ret_ty.clone(),
            Value::Module(_) => Type::Error(diags.report_todo(span, "type of module")),
            &Value::Wire(wire) => self.compiled[wire].ty.clone(),
            &Value::Register(reg) => self.compiled[reg].ty.clone(),
            &Value::Variable(var) => self.compiled[var].ty.clone(),
            &Value::Constant(constant) => self.compiled[constant].ty.clone(),
        };

        if self.log_type_check {
            eprintln!(
                "type_of_value({}) -> {}",
                self.compiled.value_to_readable_str(self.source, self.parsed, &value),
                self.compiled.type_to_readable_str(self.source, self.parsed, &result)
            );
        }

        result
    }

    fn type_of_binary(&self, origin: Span, op: BinaryOp, left: &Value, right: &Value) -> Type {
        let left_ty = self.type_of_value(origin, left);
        let right_ty = self.type_of_value(origin, right);

        fn option_op(op: BinaryOp, a: &Option<Box<Value>>, b: &Option<Box<Value>>) -> Option<Box<Value>> {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let value = simplify_value(Value::Binary(op, a.clone(), b.clone()));
                    Some(Box::new(value))
                }
                _ => None,
            }
        }

        match op {
            BinaryOp::Add => {
                let left_range = self.require_int_range_ty(origin, &left_ty);
                let right_range = self.require_int_range_ty(origin, &right_ty);

                match (left_range, right_range) {
                    (Err(e), _) | (_, Err(e)) => Type::Error(e),
                    (Ok(left_range), Ok(right_range)) => {
                        let RangeInfo { start: left_start, end: left_end } = left_range;
                        let RangeInfo { start: right_start, end: right_end } = right_range;
                        let range = RangeInfo {
                            start: option_op(BinaryOp::Add, left_start, right_start),
                            end: option_op(BinaryOp::Add, left_end, right_end),
                        };
                        Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
                    }
                }
            }
            BinaryOp::Sub => {
                let left_range = self.require_int_range_ty(origin, &left_ty);
                let right_range = self.require_int_range_ty(origin, &right_ty);

                match (left_range, right_range) {
                    (Err(e), _) | (_, Err(e)) => Type::Error(e),
                    (Ok(left_range), Ok(right_range)) => {
                        let RangeInfo { start: left_start, end: left_end } = left_range;
                        let RangeInfo { start: right_start, end: right_end } = right_range;

                        let right_end_inclusive = right_end.as_ref()
                            .map(|right_end| Box::new(Value::Binary(BinaryOp::Sub, right_end.clone(), Box::new(Value::IntConstant(BigInt::one())))));

                        let range = RangeInfo {
                            start: option_op(BinaryOp::Sub, left_start, &right_end_inclusive),
                            end: option_op(BinaryOp::Sub, left_end, right_start),
                        };
                        Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
                    }
                }
            }
            BinaryOp::Pow => {
                let base_range = self.require_int_range_ty(origin, &left_ty);
                let exp_range = self.require_int_range_ty(origin, &right_ty);
                match (base_range, exp_range) {
                    (Err(e), _) | (_, Err(e)) => Type::Error(e),
                    (Ok(base_range), Ok(exp_range)) => {
                        self.type_of_binary_power(origin, base_range, exp_range)
                    }
                }
            }
            _ => Type::Error(self.diags.report_todo(origin, format!("type of binary operator `{:?}`", op.symbol()))),
        }
    }

    fn type_of_binary_power(&self, origin: Span, base_range: &RangeInfo<Box<Value>>, exp_range: &RangeInfo<Box<Value>>) -> Type {
        let diags = self.diags;

        let RangeInfo { start: base_start, end: _ } = base_range;
        let RangeInfo { start: exp_start, end: _ } = exp_range;

        // check that exponent is non-negative
        let exp_start = match exp_start {
            None => {
                let exp_str = self.compiled.range_to_readable_str(self.source, self.parsed, &exp_range);
                let title = format!("power exponent cannot be negative, got range without lower bound {:?}", exp_str);
                let e = diags.report_simple(title, origin, "while checking this expression");
                return Type::Error(e);
            }
            Some(exp_start) => exp_start
        };
        let cond = Value::Binary(BinaryOp::CmpLte, Box::new(Value::IntConstant(BigInt::ZERO)), exp_start.clone());
        match self.try_eval_bool_true(origin, &cond) {
            Ok(()) => {}
            Err(e) => {
                let cond_str = self.compiled.value_to_readable_str(self.source, self.parsed, &cond);
                let title = format!("power exponent cannot be negative, check failed: value {} {}", cond_str, e.to_message());
                let e = diags.report_simple(title, origin, "while checking this expression");
                return Type::Error(e);
            }
        }

        // compute facts about the ranges
        let eval_known_optional = |op: BinaryOp, left: Option<Box<Value>>, right: Option<Box<Value>>| {
            let result = match (left, right) {
                (Some(left), Some(right)) => {
                    let base_positive = Value::Binary(op, left, right);
                    self.try_eval_bool(origin, &base_positive)
                }
                (None, _) | (_, None) => Ok(None),
            };
            match result {
                Ok(Some(b)) => Ok(b),
                Ok(None) => Ok(false),
                Err(e) => Err(e),
            }
        };
        macro_rules! throw_type {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => return Type::Error(e),
                }
            };
        }

        let known_base_positive = throw_type!(
            eval_known_optional(BinaryOp::CmpLt, Some(Box::new(Value::IntConstant(BigInt::ZERO))), base_start.clone()));
        let known_base_negative = throw_type!(
            eval_known_optional(BinaryOp::CmpGt, Some(Box::new(Value::IntConstant(BigInt::ZERO))), base_start.clone()));
        let known_base_non_zero = known_base_positive | known_base_negative;
        let known_exponent_positive = throw_type!(
            eval_known_optional(BinaryOp::CmpLt, Some(Box::new(Value::IntConstant(BigInt::ZERO))), Some(exp_start.clone())));

        // check that `0 ** 0` can't happen
        // TODO at some point this should be lifted into the statement prover,
        //   users should be allowed to check for this themselves!
        if !(known_base_non_zero || known_exponent_positive) {
            let e = diags.report_simple(
                "power expression where base and exponent can both be zero is not allowed",
                origin,
                "while checking this expression",
            );
            return Type::Error(e);
        }

        // some simple range result logic
        // TODO this can be improved:
        // * is exp is 1, the result range is the base range
        // * if exp is 0, the result range is 1
        // * if exp is even, result is non-negative
        // * if exp is odd, result is the same sign as base
        let range = if known_base_positive {
            // (>0) ** (>=0) = (>0)
            RangeInfo { start: Some(BigInt::one()), end: None }
        } else {
            RangeInfo { start: None, end: None }
        };

        let range = range.map_inner(|x| Box::new(Value::IntConstant(x)));
        Type::Integer(IntegerTypeInfo { range: Box::new(Value::Range(range)) })
    }

    // TODO if range ends are themselves params with ranges, assume the worst case
    //   although that misses things like (n < n+1)
    fn require_int_range_ty<'a>(&self, span: Span, ty: &'a Type) -> Result<&'a RangeInfo<Box<Value>>, ErrorGuaranteed> {
        let diags = self.diags;

        match ty {
            &Type::Error(e) => Err(e),
            Type::Integer(IntegerTypeInfo { range }) => self.require_int_range_direct(span, range.as_ref()),
            _ => {
                let ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, ty);
                Err(diags.report_simple(format!("expected integer type, got {}", ty_str), span, "for this expression"))
            }
        }
    }

    pub fn require_int_range_direct<'a>(&self, span: Span, range: &'a Value) -> Result<&'a RangeInfo<Box<Value>>, ErrorGuaranteed> {
        match range {
            &Value::Error(e) => Err(e),
            Value::Range(range) => Ok(range),
            _ => Err(self.diags.report_todo(span, "indirect integer range")),
        }
    }

    // TODO double-check that all of these type-checking functions do their error handling correctly
    //   and don't short-circuit unnecessarily, preventing multiple errors
    // TODO move error formatting out of this function, it depends too much on the context and spans are not always available
    pub fn check_type_contains(&self, span_ty: Option<Span>, span_value: Span, ty: &Type, value: &Value) -> Result<(), ErrorGuaranteed> {
        let ty_value = self.type_of_value(span_value, value);

        match (ty, &ty_value) {
            // propagate errors, we can't just silently ignore them:
            //   downstream compiler code might actually depend on the type matching
            (&Type::Error(e), _) | (_, &Type::Error(e)) => return Err(e),

            // equal types are always fine
            _ if ty == &ty_value => return Ok(()),

            // any contains everything
            (&Type::Any, _) => return Ok(()),

            // basic types that only contain themselves
            (Type::Unit, Type::Unit) => return Ok(()),
            (Type::Never, Type::Never) => return Ok(()),
            // TODO differentiate between arbitrary open, half-open, ...
            (Type::Range, Type::Range) => return Ok(()),
            (Type::Boolean, Type::Boolean) => return Ok(()),

            // integer range check
            (Type::Integer(IntegerTypeInfo { range: range_ty }), Type::Integer(_)) => {
                let range_ty = self.require_int_range_direct(span_ty.unwrap_or(span_value), range_ty)?;

                let RangeInfo { start: start_ty, end: end_ty } = range_ty;

                // check that the value fits in the required range
                if let Some(start_ty) = start_ty {
                    let cond = Value::Binary(BinaryOp::CmpLte, start_ty.clone(), Box::new(value.clone()));
                    self.require_value_true_for_type_check(span_ty, span_value, &cond)?;
                }
                if let Some(end_ty) = end_ty {
                    let cond = Value::Binary(BinaryOp::CmpLt, Box::new(value.clone()), end_ty.clone());
                    self.require_value_true_for_type_check(span_ty, span_value, &cond)?;
                }
                return Ok(());
            }

            // fallthrough into error
            _ => {}
        };

        let ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, ty);
        let value_str = self.compiled.value_to_readable_str(self.source, self.parsed, value);
        let value_ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &ty_value);

        let title = format!("type mismatch: value {} with type {} does not match type {}", value_str, value_ty_str, ty_str);
        let err = Diagnostic::new(title)
            .add_error(span_value, "value used here")
            .add_info_maybe(span_ty, "type defined here")
            .finish();
        Err(self.diags.report(err))
    }

    pub fn require_value_true_for_range(&self, span_range: Span, value: &Value) -> Result<(), ErrorGuaranteed> {
        self.try_eval_bool_true(span_range, value).map_err(|e| {
            let value_str = self.compiled.value_to_readable_str(self.source, self.parsed, value);
            let title = format!("range valid check failed: value {} {}", value_str, e.to_message());
            let err = Diagnostic::new(title)
                .add_error(span_range, "when checking that this range is non-decreasing")
                .finish();
            self.diags.report(err).into()
        })
    }

    pub fn require_value_true_for_type_check(&self, span_ty: Option<Span>, span_value: Span, value: &Value) -> Result<(), ErrorGuaranteed> {
        self.try_eval_bool_true(span_value, value).map_err(|e| {
            let value_str = self.compiled.value_to_readable_str(self.source, self.parsed, value);
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
        if self.log_type_check {
            eprintln!(
                "try_eval_bool({}) -> {:?}",
                self.compiled.value_to_readable_str(self.source, self.parsed, value),
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
                let left_ty = self.type_of_value(origin, left);
                let left_range = self.require_int_range_ty(origin, &left_ty)?;
                let right_ty = self.type_of_value(origin, right);
                let right_range = self.require_int_range_ty(origin, &right_ty)?;

                if self.log_type_check {
                    eprintln!("comparing ranges with op {:?}", binary_op);
                    eprintln!("  {:?}", left_range);
                    eprintln!("  {:?}", right_range);
                }

                let compare_lt = |allow_eq: bool| {
                    let right_start = value_as_int(right_range.start.as_ref()?)?;
                    let left_end_inclusive = value_as_int(left_range.end.as_ref()?)? - 1;

                    if allow_eq {
                        Some(&left_end_inclusive <= right_start)
                    } else {
                        Some(&left_end_inclusive < right_start)
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
}

pub fn simplify_value(value: Value) -> Value {
    match value {
        Value::Binary(op, left, right) => {
            let left = simplify_value(*left);
            let right = simplify_value(*right);

            if let Some((left, right)) = option_pair(value_as_int(&left), value_as_int(&right)) {
                match op {
                    BinaryOp::Add => return Value::IntConstant(left + right),
                    BinaryOp::Sub => return Value::IntConstant(left - right),
                    BinaryOp::Mul => return Value::IntConstant(left * right),
                    _ => {}
                }
            }

            Value::Binary(op, Box::new(left), Box::new(right))
        }
        Value::UnaryNot(inner) => {
            let inner = simplify_value(*inner);
            Value::UnaryNot(Box::new(inner))
        }
        // TODO at least recursively call simplify
        value => value,
    }
}

// TODO return error if value is error?
pub fn value_as_int(value: &Value) -> Option<&BigInt> {
    match value {
        Value::IntConstant(value) => Some(value),
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
