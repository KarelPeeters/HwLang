use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, DiagnosticBuilder, Diagnostics, ErrorGuaranteed};
use crate::front::domain::ValueDomain;
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::syntax::ast::{Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::util::big_int::{BigInt, BigUint};
use annotate_snippets::Level;
use itertools::Itertools;
use unwrap_match::unwrap_match;

impl CompileItemContext<'_, '_> {
    pub fn check_valid_domain_crossing(
        &self,
        crossing_span: Span,
        target: Spanned<ValueDomain>,
        source: Spanned<ValueDomain>,
        required_reason: &str,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.refs.diags;

        let valid = match (target.inner, source.inner) {
            // TODO is clock->clock actually okay?
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
            let target_str = target.inner.to_diagnostic_string(self);
            let source_str = source.inner.to_diagnostic_string(self);
            let diag = Diagnostic::new(format!("invalid domain crossing: {invalid_reason}"))
                .add_error(crossing_span, "invalid domain crossing here")
                .add_info(target.span, format!("target domain is {target_str}"))
                .add_info(source.span, format!("source domain is {source_str}"))
                .footer(Level::Info, required_reason)
                .finish();
            diags.report(diag)
        })
    }
}

// TODO turn this into a lambda
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
}

impl TypeContainsReason {
    pub fn add_diag_info(self, diag: DiagnosticBuilder, target_ty: &Type) -> DiagnosticBuilder {
        let target_ty_str = target_ty.to_diagnostic_string();
        match self {
            // TODO improve assignment error message
            TypeContainsReason::Assignment {
                span_target,
                span_target_ty,
            } => diag
                .add_info(span_target, format!("target requires type `{}`", target_ty_str))
                .add_info(span_target_ty, "target type set here"),
            TypeContainsReason::Operator(span) => {
                diag.add_info(span, format!("operator requires type `{}`", target_ty_str))
            }
            TypeContainsReason::InstanceModule(span) => {
                diag.add_info(span, format!("module instance requires type `{}`", target_ty_str))
            }
            TypeContainsReason::Interface(span) => {
                diag.add_info(span, format!("interface requires type `{}`", target_ty_str))
            }
            TypeContainsReason::InstancePortInput {
                span_connection_port_id,
                span_port_ty,
            } => diag
                .add_info(
                    span_connection_port_id,
                    format!("input port has type `{}`", target_ty_str),
                )
                .add_info(span_port_ty, "port type set here"),
            TypeContainsReason::InstancePortOutput {
                span_connection_signal_id,
                span_signal_ty,
            } => diag
                .add_info(
                    span_connection_signal_id,
                    format!("target signal has type `{}`", target_ty_str),
                )
                .add_info(span_signal_ty, "target signal type set here"),
            TypeContainsReason::Return {
                span_keyword,
                span_return_ty,
            } => diag
                .add_info(span_keyword, format!("return requires type `{}`", target_ty_str))
                .add_info(span_return_ty, format!("return type `{}` set here", target_ty_str)),
            TypeContainsReason::ForIndexType(span_ty) => diag.add_info(
                span_ty,
                format!("for loop iteration variable type `{}` set here", target_ty_str),
            ),
            TypeContainsReason::IfCondition(span) => {
                diag.add_info(span, format!("if condition requires type `{}`", target_ty_str))
            }
            TypeContainsReason::WhileCondition(span) => {
                diag.add_info(span, format!("while condition requires type `{}`", target_ty_str))
            }
            TypeContainsReason::ArrayIndex { span_index } => {
                diag.add_info(span_index, format!("array index requires type `{}`", target_ty_str))
            }
            TypeContainsReason::ArrayLen { span_len } => {
                diag.add_info(span_len, format!("array length requires type `{}`", target_ty_str))
            }
            TypeContainsReason::Parameter { param_ty } => {
                diag.add_info(param_ty, format!("parameter requires type `{}`", target_ty_str))
            }
        }
    }
}

// TODO go over this again and see if/how users are using `accept_undefined` and `allow_compound_subtype`
pub fn check_type_contains_value(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    target_ty: &Type,
    value: Spanned<&Value>,
    accept_undefined: bool,
    allow_compound_subtype: bool,
) -> Result<(), ErrorGuaranteed> {
    match value.inner {
        Value::Compile(value_inner) => {
            let value = Spanned {
                span: value.span,
                inner: value_inner,
            };
            check_type_contains_compile_value(diags, reason, target_ty, value, accept_undefined)
        }
        Value::Hardware(value_inner) => {
            let value_ty = Spanned {
                span: value.span,
                inner: &value_inner.ty.as_type(),
            };
            check_type_contains_type(diags, reason, target_ty, value_ty, allow_compound_subtype)
        }
    }
}

pub fn check_type_contains_compile_value(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    target_ty: &Type,
    value: Spanned<&CompileValue>,
    accept_undefined: bool,
) -> Result<(), ErrorGuaranteed> {
    let ty_contains_value = target_ty.contains_type(&value.inner.ty(), true);

    if ty_contains_value && (accept_undefined || !value.inner.contains_undefined()) {
        Ok(())
    } else {
        let mut diag = Diagnostic::new("value does not fit in type");
        diag = reason.add_diag_info(diag, target_ty);
        // TODO abbreviate source value if it gets too long
        let value_str = value.inner.to_diagnostic_string();
        let value_ty_str = value.inner.ty().to_diagnostic_string();
        let diag = diag
            .add_error(
                value.span,
                format!("source value `{value_str}` with type `{value_ty_str}` does not fit"),
            )
            .finish();
        Err(diags.report(diag))
    }
}

pub fn check_type_contains_type(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    target_ty: &Type,
    value_ty: Spanned<&Type>,
    allow_compound_subtype: bool,
) -> Result<(), ErrorGuaranteed> {
    if target_ty.contains_type(value_ty.inner, allow_compound_subtype) {
        Ok(())
    } else {
        let mut diag = Diagnostic::new("value does not fit in type");
        diag = reason.add_diag_info(diag, target_ty);

        if !allow_compound_subtype && target_ty.contains_type(value_ty.inner, true) {
            diag = diag.footer(
                Level::Info,
                "compound subtyping is not allowed for hardware values, if it was the value would have fit",
            );
        }

        let diag = diag
            .add_error(
                value_ty.span,
                format!(
                    "source value with type `{}` does not fit",
                    value_ty.inner.to_diagnostic_string()
                ),
            )
            .finish();
        Err(diags.report(diag))
    }
}

// TODO reduce boilerplate with a trait?
pub fn check_type_is_int(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<Value>,
) -> Result<Spanned<Value<BigInt, HardwareValue<ClosedIncRange<BigInt>>>>, ErrorGuaranteed> {
    check_type_contains_value(diags, reason, &Type::Int(IncRange::OPEN), value.as_ref(), false, true)?;

    match value.inner {
        Value::Compile(value_inner) => match value_inner {
            CompileValue::Int(value_inner) => Ok(Spanned {
                span: value.span,
                inner: Value::Compile(value_inner),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected int value, should have already been checked")),
        },
        Value::Hardware(value_inner) => match value_inner.ty {
            HardwareType::Int(ty) => Ok(Spanned {
                span: value.span,
                inner: Value::Hardware(HardwareValue {
                    ty,
                    domain: value_inner.domain,
                    expr: value_inner.expr,
                }),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected int type, should have already been checked")),
        },
    }
}

pub fn check_type_is_int_compile(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> Result<BigInt, ErrorGuaranteed> {
    check_type_contains_compile_value(diags, reason, &Type::Int(IncRange::OPEN), value.as_ref(), false)?;

    match value.inner {
        CompileValue::Int(value_inner) => Ok(value_inner),
        _ => Err(diags.report_internal_error(value.span, "expected int value, should have already been checked")),
    }
}

pub fn check_type_is_int_hardware(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<HardwareValue>,
) -> Result<Spanned<HardwareValue<ClosedIncRange<BigInt>>>, ErrorGuaranteed> {
    let value_ty = value.as_ref().map_inner(|value| value.ty.as_type());
    check_type_contains_type(diags, reason, &Type::Int(IncRange::OPEN), value_ty.as_ref(), false)?;

    match value.inner.ty {
        HardwareType::Int(ty) => Ok(Spanned {
            span: value.span,
            inner: HardwareValue {
                ty,
                domain: value.inner.domain,
                expr: value.inner.expr,
            },
        }),
        _ => Err(diags.report_internal_error(value.span, "expected int type, should have already been checked")),
    }
}

pub fn check_type_is_uint_compile(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> Result<BigUint, ErrorGuaranteed> {
    let range = IncRange {
        start_inc: Some(BigInt::ZERO),
        end_inc: None,
    };
    check_type_contains_compile_value(diags, reason, &Type::Int(range), value.as_ref(), false)?;

    match value.inner {
        CompileValue::Int(value_inner) => Ok(BigUint::try_from(value_inner).unwrap()),
        _ => Err(diags.report_internal_error(value.span, "expected int value, should have already been checked")),
    }
}

pub fn check_type_is_bool(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<Value>,
) -> Result<Spanned<Value<bool, HardwareValue<()>>>, ErrorGuaranteed> {
    check_type_contains_value(diags, reason, &Type::Bool, value.as_ref(), false, false)?;

    match value.inner {
        Value::Compile(value_inner) => match value_inner {
            CompileValue::Bool(value_inner) => Ok(Spanned {
                span: value.span,
                inner: Value::Compile(value_inner),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected bool value, should have already been checked")),
        },
        Value::Hardware(value_inner) => match value_inner.ty {
            HardwareType::Bool => Ok(Spanned {
                span: value.span,
                inner: Value::Hardware(HardwareValue {
                    ty: (),
                    domain: value_inner.domain,
                    expr: value_inner.expr,
                }),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected bool type, should have already been checked")),
        },
    }
}

pub fn check_type_is_bool_compile(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<CompileValue>,
) -> Result<bool, ErrorGuaranteed> {
    check_type_contains_compile_value(diags, reason, &Type::Bool, value.as_ref(), false)?;

    match value.inner {
        CompileValue::Bool(value_inner) => Ok(value_inner),
        _ => Err(diags.report_internal_error(value.span, "expected bool value, should have already been checked")),
    }
}

pub fn check_type_is_bool_array(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<Value>,
    expected_len: Option<&BigUint>,
) -> Result<Value<Vec<bool>, HardwareValue<BigUint>>, ErrorGuaranteed> {
    if let Type::Array(ty_inner, ty_len) = value.inner.ty() {
        if expected_len.is_none_or(|expected_len| expected_len == &ty_len) {
            if let Type::Bool = *ty_inner {
                return match value.inner {
                    Value::Compile(c) => {
                        let c = unwrap_match!(c, CompileValue::Array(c) => c);
                        let result = c
                            .into_iter()
                            .map(|c| unwrap_match!(c, CompileValue::Bool(c) => c))
                            .collect_vec();
                        Ok(Value::Compile(result))
                    }
                    Value::Hardware(c) => Ok(Value::Hardware(HardwareValue {
                        ty: ty_len,
                        domain: c.domain,
                        expr: c.expr,
                    })),
                };
            }
        }
    }

    let expected_ty_str = match expected_len {
        None => Type::Array(Box::new(Type::Bool), BigUint::ZERO)
            .to_diagnostic_string()
            .replace("0", "_"),
        Some(expected_len) => Type::Array(Box::new(Type::Bool), expected_len.clone()).to_diagnostic_string(),
    };
    let value_ty_str = value.inner.ty().to_diagnostic_string();
    let mut diag = Diagnostic::new("value does not fit in type").add_error(
        value.span,
        format!("expected `{}`, got type `{}`", expected_ty_str, value_ty_str),
    );

    diag = reason.add_diag_info(diag, &Type::Array(Box::new(Type::Bool), BigUint::ZERO));
    Err(diags.report(diag.finish()))
}

pub fn check_hardware_type_for_bit_operation(
    diags: &Diagnostics,
    ty: Spanned<&Type>,
) -> Result<HardwareType, ErrorGuaranteed> {
    if let Some(ty_hw) = ty.inner.as_hardware_type() {
        // TODO this feels strange, maybe clock should not actually be a hardware type, only a domain?
        if let HardwareType::Clock = ty_hw {
            return Err(diags.report_todo(ty.span, "interaction between to/from bits and clocks"));
        }

        return Ok(ty_hw);
    }

    let diag = Diagnostic::new("converting to/from bits is only possible for hardware types")
        .add_error(ty.span, format!("actual type `{}`", ty.inner.to_diagnostic_string()))
        .finish();
    Err(diags.report(diag))
}
