use crate::front::block::TypedIrExpression;
use crate::front::compile::CompileState;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, DiagnosticBuilder, Diagnostics, ErrorGuaranteed};
use crate::front::misc::ValueDomain;
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast::{Spanned, SyncDomain};
use crate::syntax::pos::Span;
use annotate_snippets::Level;
use num_bigint::BigInt;

impl CompileState<'_> {
    pub fn check_valid_domain_crossing(
        &self,
        crossing_span: Span,
        target: Spanned<&ValueDomain>,
        source: Spanned<&ValueDomain>,
        required_reason: &str,
    ) -> Result<(), ErrorGuaranteed> {
        let valid = match (target.inner, source.inner) {
            (ValueDomain::Clock, ValueDomain::Clock) => Ok(()),
            (ValueDomain::Clock, _) => Err("non-clock to clock"),
            (ValueDomain::Sync(_), ValueDomain::Clock) => Err("clock to sync"),
            (ValueDomain::Sync(_), ValueDomain::Async) => Err("async to sync"),

            // compile-time from nothing else and to everything
            (_, ValueDomain::CompileTime) => Ok(()),
            (ValueDomain::CompileTime, _) => Err("compile-time to non-compile-time"),

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

                match (target_clock == source_clock, target_reset == source_reset) {
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
            self.diags.report(diag)
        })
    }
}

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
    Return {
        span_keyword: Span,
        span_return_ty: Span,
    },
    ForIndexType {
        span_ty: Span,
    },
    WhileCondition {
        span_keyword: Span,
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
                diag.add_info(span, format!("module instance requires `{}`", target_ty_str))
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
            TypeContainsReason::ForIndexType { span_ty } => diag.add_info(
                span_ty,
                format!("for loop iteration variable type `{}` set here", target_ty_str),
            ),
            TypeContainsReason::WhileCondition { span_keyword } => diag.add_info(
                span_keyword,
                format!("while condition requires type `{}`", target_ty_str),
            ),
        }
    }
}

pub fn check_type_contains_value(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    target_ty: &Type,
    value: Spanned<&MaybeCompile<TypedIrExpression>>,
    accept_undefined: bool,
    allow_compound_subtype: bool,
) -> Result<(), ErrorGuaranteed> {
    match value.inner {
        MaybeCompile::Compile(value_inner) => {
            let value = Spanned {
                span: value.span,
                inner: value_inner,
            };
            check_type_contains_compile_value(diags, reason, target_ty, value, accept_undefined)
        }
        MaybeCompile::Other(value_inner) => {
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
    let value_is_undefined = value.inner.contains_undefined();

    if ty_contains_value && (accept_undefined || !value_is_undefined) {
        Ok(())
    } else {
        let mut diag = Diagnostic::new("value does not fit in type");
        diag = reason.add_diag_info(diag, target_ty);
        let diag = diag
            .add_error(
                value.span,
                format!("source value `{}` does not fit", value.inner.to_diagnostic_string()),
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
    value: Spanned<MaybeCompile<TypedIrExpression>>,
) -> Result<Spanned<MaybeCompile<TypedIrExpression<ClosedIncRange<BigInt>>, BigInt>>, ErrorGuaranteed> {
    check_type_contains_value(diags, reason, &Type::Int(IncRange::OPEN), value.as_ref(), false, true)?;

    match value.inner {
        MaybeCompile::Compile(value_inner) => match value_inner {
            CompileValue::Int(value_inner) => Ok(Spanned {
                span: value.span,
                inner: MaybeCompile::Compile(value_inner),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected int value, should have already been checked")),
        },
        MaybeCompile::Other(value_inner) => match value_inner.ty {
            HardwareType::Int(ty) => Ok(Spanned {
                span: value.span,
                inner: MaybeCompile::Other(TypedIrExpression {
                    ty,
                    domain: value_inner.domain,
                    expr: value_inner.expr,
                }),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected int type, should have already been checked")),
        },
    }
}

pub fn check_type_is_bool(
    diags: &Diagnostics,
    reason: TypeContainsReason,
    value: Spanned<MaybeCompile<TypedIrExpression>>,
) -> Result<Spanned<MaybeCompile<TypedIrExpression<()>, bool>>, ErrorGuaranteed> {
    check_type_contains_value(diags, reason, &Type::Bool, value.as_ref(), false, false)?;

    match value.inner {
        MaybeCompile::Compile(value_inner) => match value_inner {
            CompileValue::Bool(value_inner) => Ok(Spanned {
                span: value.span,
                inner: MaybeCompile::Compile(value_inner),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected bool value, should have already been checked")),
        },
        MaybeCompile::Other(value_inner) => match value_inner.ty {
            HardwareType::Bool => Ok(Spanned {
                span: value.span,
                inner: MaybeCompile::Other(TypedIrExpression {
                    ty: (),
                    domain: value_inner.domain,
                    expr: value_inner.expr,
                }),
            }),
            _ => Err(diags.report_internal_error(value.span, "expected bool type, should have already been checked")),
        },
    }
}
