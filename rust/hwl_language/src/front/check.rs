use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::ValueDomain;
use crate::front::implication::{HardwareValueWithImplications, ValueWithImplications};
use crate::front::item::ElaborationArenas;
use crate::front::types::{HardwareType, Type, TypeBool, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, HardwareInt, HardwareUInt, HardwareValue, MaybeCompile, MixedCompoundValue,
    MixedString, SimpleCompileValue, Value,
};
use crate::syntax::ast::SyncDomain;
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::token::TOKEN_STR_UNSAFE_VALUE_WITH_DOMAIN;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::iter::IterExt;
use crate::util::range::Range;
use crate::util::range_multi::MultiRange;
use std::fmt::Debug;
use std::sync::Arc;

impl CompileItemContext<'_, '_> {
    pub fn check_valid_domain_crossing(
        &self,
        crossing_span: Span,
        target: Spanned<ValueDomain>,
        source: Spanned<ValueDomain>,
        required_reason: &str,
    ) -> DiagResult {
        let diags = self.refs.diags;

        let valid = match (target.inner, source.inner) {
            (ValueDomain::Clock, ValueDomain::Clock) => Ok(()),
            (ValueDomain::Clock, _) => Err("non-clock to clock"),
            (ValueDomain::Sync(_), ValueDomain::Clock) => Err("clock to sync"),
            (ValueDomain::Sync(_), ValueDomain::Async) => Err("async to sync"),

            // compile-time from nothing else and to everything
            (_, ValueDomain::CompileTime) => Ok(()),
            (ValueDomain::CompileTime, _) => Err("non compile to compile"),

            // const only from const and to everything
            (_, ValueDomain::Const) => Ok(()),
            (ValueDomain::Const, _) => Err("non-const to const"),

            // async from everything
            (ValueDomain::Async, _) => Ok(()),

            // sync only if matching
            // TODO expand this to look though wires without logic
            (ValueDomain::Sync(target_inner), ValueDomain::Sync(source_inner)) => {
                let SyncDomain {
                    clock: target_clock,
                    reset: target_reset,
                } = target_inner;
                let SyncDomain {
                    clock: source_clock,
                    reset: source_reset,
                } = source_inner;

                // protect again clock domain crossings
                let clock_match = target_clock == source_clock;
                // protect against the source resetting while the target is not
                let reset_match = match (target_reset, source_reset) {
                    (_, None) => true,
                    (None, Some(_)) => false,
                    (Some(target_reset), Some(source_reset)) => target_reset == source_reset,
                };

                match (clock_match, reset_match) {
                    (true, true) => Ok(()),
                    (false, true) => Err("different clock"),
                    (true, false) => Err("different reset"),
                    (false, false) => Err("different clock and reset"),
                }
            }
        };

        valid.map_err(|invalid_reason| {
            let target_str = target.inner.diagnostic_string(self);
            let source_str = source.inner.diagnostic_string(self);
            DiagnosticError::new(
                format!("invalid domain crossing: {invalid_reason}"),
                crossing_span,
                "invalid domain crossing here",
            )
            .add_info(target.span, format!("target domain is {target_str}"))
            .add_info(source.span, format!("source domain is {source_str}"))
            .add_footer_info(format!("crossing due to {required_reason}"))
            .add_footer_hint(format!(
                "to intentionally cross domains, use `{}` or `unsafe_bool_to_clock`",
                TOKEN_STR_UNSAFE_VALUE_WITH_DOMAIN
            ))
            .report(diags)
        })
    }
}

// TODO turn this into a lambda
// TODO do we really need this many variants?
#[derive(Debug, Copy, Clone)]
pub enum TypeContainsReason {
    Assignment {
        span_target: Span,
        span_target_ty: Span,
    },
    Operator(Span),
    InstanceModule(Span),
    InstancePortInput {
        span_connection_port_id: Span,
        span_port_ty: Span,
    },
    InstancePortOutput {
        span_connection_signal_id: Span,
        span_signal_ty: Span,
    },
    Interface(Span),
    Return {
        span_keyword: Span,
        span_return_ty: Span,
    },
    ForIndexType(Span),
    IfCondition(Span),
    MatchPattern(Span),
    WhileCondition(Span),
    ArrayIndex {
        span_index: Span,
    },
    ArrayLen {
        span_len: Span,
    },
    Parameter {
        param_ty: Span,
    },
    Internal(Span),
}

impl TypeContainsReason {
    pub fn add_diag_info(self, elab: &ElaborationArenas, diag: DiagnosticError, target_ty: &Type) -> DiagnosticError {
        let target_ty_str = target_ty.value_string(elab);
        match self {
            // TODO improve assignment error message
            TypeContainsReason::Assignment {
                span_target,
                span_target_ty,
            } => diag
                .add_info(span_target, format!("target requires type `{target_ty_str}`"))
                .add_info(span_target_ty, "target type set here"),
            TypeContainsReason::Operator(span) => {
                diag.add_info(span, format!("operator requires type `{target_ty_str}`"))
            }
            TypeContainsReason::InstanceModule(span) => {
                diag.add_info(span, format!("module instance requires type `{target_ty_str}`"))
            }
            TypeContainsReason::Interface(span) => {
                diag.add_info(span, format!("interface requires type `{target_ty_str}`"))
            }
            TypeContainsReason::InstancePortInput {
                span_connection_port_id,
                span_port_ty,
            } => diag
                .add_info(
                    span_connection_port_id,
                    format!("input port has type `{target_ty_str}`"),
                )
                .add_info(span_port_ty, "port type set here"),
            TypeContainsReason::InstancePortOutput {
                span_connection_signal_id,
                span_signal_ty,
            } => diag
                .add_info(
                    span_connection_signal_id,
                    format!("target signal has type `{target_ty_str}`"),
                )
                .add_info(span_signal_ty, "target signal type set here"),
            TypeContainsReason::Return {
                span_keyword,
                span_return_ty,
            } => diag
                .add_info(span_keyword, format!("return requires type `{target_ty_str}`"))
                .add_info(span_return_ty, format!("return type `{target_ty_str}` set here")),
            TypeContainsReason::ForIndexType(span_ty) => diag.add_info(
                span_ty,
                format!("for loop iteration variable type `{target_ty_str}` set here"),
            ),
            TypeContainsReason::IfCondition(span) => {
                diag.add_info(span, format!("if condition requires type `{target_ty_str}`"))
            }
            TypeContainsReason::MatchPattern(span) => {
                diag.add_info(span, format!("match pattern against value with type `{target_ty_str}`"))
            }
            TypeContainsReason::WhileCondition(span) => {
                diag.add_info(span, format!("while condition requires type `{target_ty_str}`"))
            }
            TypeContainsReason::ArrayIndex { span_index } => {
                diag.add_info(span_index, format!("array index requires type `{target_ty_str}`"))
            }
            TypeContainsReason::ArrayLen { span_len } => {
                diag.add_info(span_len, format!("array length requires type `{target_ty_str}`"))
            }
            TypeContainsReason::Parameter { param_ty } => {
                diag.add_info(param_ty, format!("parameter requires type `{target_ty_str}`"))
            }
            TypeContainsReason::Internal(span) => DiagnosticError::new_internal_compiler_error(
                "type check failed, should have been checked already",
                span,
            ),
        }
    }
}

pub fn check_type_contains_value<V: Typed + Debug>(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    target_ty: &Type,
    value: Spanned<&V>,
) -> DiagResult {
    // TODO if constant value, use value in message?
    let value_ty = value.map_inner(|v| v.ty());
    check_type_contains_type(diags, elab, reason, target_ty, value_ty.as_ref())
}

pub fn check_type_contains_type(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    target_ty: &Type,
    value_ty: Spanned<&Type>,
) -> DiagResult {
    if target_ty.contains_type(value_ty.inner) {
        Ok(())
    } else {
        // TODO simpler error message for very simply values, eg. ints, bools, short tuples/arrays
        // TODO for assignments, error should point at the assignment operator, not the value
        let mut diag = DiagnosticError::new(
            "type mismatch",
            value_ty.span,
            format!(
                "source value with type `{}` does not fit",
                value_ty.inner.value_string(elab)
            ),
        );
        diag = reason.add_diag_info(elab, diag, target_ty);
        Err(diag.report(diags))
    }
}

pub fn check_type_is_int(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<Value>,
) -> DiagResult<MaybeCompile<BigInt, HardwareInt>> {
    check_type_contains_value(diags, elab, reason, &Type::Int(MultiRange::open()), value.as_ref())?;

    let err = || diags.report_error_internal(value.span, "unexpected value kind for int type");
    match value.inner {
        Value::Simple(v) => match v {
            SimpleCompileValue::Int(v) => Ok(MaybeCompile::Compile(v)),
            _ => Err(err()),
        },
        Value::Hardware(v) => match v.ty {
            HardwareType::Int(ty) => Ok(MaybeCompile::Hardware(HardwareValue {
                ty,
                domain: v.domain,
                expr: v.expr,
            })),
            _ => Err(err()),
        },
        Value::Compound(_) => Err(err()),
    }
}

pub fn check_type_is_uint(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<Value>,
) -> DiagResult<MaybeCompile<BigUint, HardwareUInt>> {
    let ty_uint = Type::Int(MultiRange::from(Range {
        start: Some(BigInt::ZERO),
        end: None,
    }));
    check_type_contains_value(diags, elab, reason, &ty_uint, value.as_ref())?;

    let err = || diags.report_error_internal(value.span, "unexpected value kind for uint type");
    match value.inner {
        Value::Simple(v) => match v {
            SimpleCompileValue::Int(v) => Ok(MaybeCompile::Compile(BigUint::try_from(v).unwrap())),
            _ => Err(err()),
        },
        Value::Hardware(v) => match v.ty {
            HardwareType::Int(ty) => Ok(MaybeCompile::Hardware(HardwareValue {
                ty: ty.map(|x| BigUint::try_from(x).unwrap()),
                domain: v.domain,
                expr: v.expr,
            })),
            _ => Err(err()),
        },
        Value::Compound(_) => Err(err()),
    }
}

pub fn check_type_is_int_hardware(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<HardwareValue>,
) -> DiagResult<HardwareInt> {
    let value_ty = value.as_ref().map_inner(|value| value.ty.as_type());
    check_type_contains_type(diags, elab, reason, &Type::Int(MultiRange::open()), value_ty.as_ref())?;

    match value.inner.ty {
        HardwareType::Int(ty) => Ok(HardwareValue {
            ty,
            domain: value.inner.domain,
            expr: value.inner.expr,
        }),
        _ => Err(diags.report_error_internal(value.span, "expected int type")),
    }
}

pub fn check_type_is_uint_compile(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> DiagResult<BigUint> {
    let ty_uint = Type::Int(MultiRange::from(Range {
        start: Some(BigInt::ZERO),
        end: None,
    }));
    check_type_contains_value(diags, elab, reason, &ty_uint, value.as_ref())?;

    let err = || diags.report_error_internal(value.span, "expected uint value");
    match value.inner {
        CompileValue::Simple(v) => match v {
            SimpleCompileValue::Int(v) => Ok(BigUint::try_from(v).unwrap()),
            _ => Err(err()),
        },
        CompileValue::Compound(_) => Err(err()),
        CompileValue::Hardware(never) => never.unreachable(),
    }
}

pub fn check_type_is_bool(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<ValueWithImplications>,
) -> DiagResult<MaybeCompile<bool, HardwareValueWithImplications<TypeBool>>> {
    check_type_contains_value(diags, elab, reason, &Type::Bool, value.as_ref())?;

    let err = || diags.report_error_internal(value.span, "unexpected value kind for bool type");
    match value.inner {
        Value::Simple(v) => match v {
            SimpleCompileValue::Bool(v) => Ok(MaybeCompile::Compile(v)),
            _ => Err(err()),
        },
        Value::Compound(_) => Err(err()),
        Value::Hardware(v) => match v.value.ty {
            HardwareType::Bool => Ok(MaybeCompile::Hardware(v.map_type(|_| TypeBool))),
            _ => Err(diags.report_error_internal(value.span, "expected bool type")),
        },
    }
}

pub fn check_type_is_bool_compile(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> DiagResult<bool> {
    check_type_contains_value(diags, elab, reason, &Type::Bool, value.as_ref())?;

    let err = || diags.report_error_internal(value.span, "expected bool value");
    match value.inner {
        CompileValue::Simple(v) => match v {
            SimpleCompileValue::Bool(v) => Ok(v),
            _ => Err(err()),
        },
        CompileValue::Compound(_) => Err(err()),
        CompileValue::Hardware(never) => never.unreachable(),
    }
}

pub fn check_type_is_bool_array(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<Value>,
    expected_len: Option<&BigUint>,
) -> DiagResult<MaybeCompile<Vec<bool>, HardwareValue<BigUint>>> {
    if let Type::Array(ty_inner, ty_len) = value.inner.ty()
        && expected_len.is_none_or(|expected_len| expected_len == &ty_len)
        && let Type::Bool = *ty_inner
    {
        let err = || diags.report_error_internal(value.span, "expected bool array");
        return match value.inner {
            Value::Simple(v) => match v {
                SimpleCompileValue::Array(v) => {
                    let result = v
                        .iter()
                        .map(|e| match e {
                            &Value::Simple(SimpleCompileValue::Bool(b)) => Ok(b),
                            _ => Err(err()),
                        })
                        .try_collect_vec()?;
                    Ok(MaybeCompile::Compile(result))
                }
                _ => return Err(err()),
            },
            Value::Compound(_) => return Err(err()),
            Value::Hardware(c) => Ok(MaybeCompile::Hardware(HardwareValue {
                ty: ty_len,
                domain: c.domain,
                expr: c.expr,
            })),
        };
    }

    let expected_ty_str = match expected_len {
        None => Type::Array(Arc::new(Type::Bool), BigUint::ZERO)
            .value_string(elab)
            .replace("0", "_"),
        Some(expected_len) => Type::Array(Arc::new(Type::Bool), expected_len.clone()).value_string(elab),
    };
    let value_ty_str = value.inner.ty().value_string(elab);
    let mut diag = DiagnosticError::new(
        "type mismatch",
        value.span,
        format!("expected `{expected_ty_str}`, got type `{value_ty_str}`"),
    );

    diag = reason.add_diag_info(elab, diag, &Type::Array(Arc::new(Type::Bool), BigUint::ZERO));
    Err(diag.report(diags))
}

pub fn check_type_is_string(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<Value>,
) -> DiagResult<Arc<MixedString>> {
    check_type_contains_value(diags, elab, reason, &Type::String, value.as_ref())?;

    match value.inner {
        Value::Compound(MixedCompoundValue::String(v)) => Ok(v),
        _ => Err(diags.report_error_internal(value.span, "expected string value")),
    }
}

pub fn check_type_is_string_compile(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> DiagResult<Arc<String>> {
    check_type_contains_value(diags, elab, reason, &Type::String, value.as_ref())?;

    match value.inner {
        Value::Compound(CompileCompoundValue::String(v)) => Ok(v),
        _ => Err(diags.report_error_internal(value.span, "expected string value")),
    }
}

pub fn check_hardware_type_for_bit_operation(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    ty: Spanned<&Type>,
) -> DiagResult<HardwareType> {
    ty.inner.as_hardware_type(elab).map_err(|_| {
        let ty_str = ty.inner.value_string(elab);
        DiagnosticError::new(
            "converting to/from bits is only possible for hardware types",
            ty.span,
            format!("actual type `{}`", ty_str),
        )
        .report(diags)
    })
}

pub fn check_type_is_range_compile(
    diags: &Diagnostics,
    elab: &ElaborationArenas,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> DiagResult<Range<BigInt>> {
    check_type_contains_value(diags, elab, reason, &Type::Range, value.as_ref())?;

    match value.inner {
        CompileValue::Compound(CompileCompoundValue::Range(range)) => Ok(range),
        CompileValue::Compound(_) | CompileValue::Simple(_) => {
            Err(diags.report_error_internal(value.span, "expected range value"))
        }
        CompileValue::Hardware(never) => never.unreachable(),
    }
}
