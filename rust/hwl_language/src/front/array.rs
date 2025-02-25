use crate::front::assignment::AssignmentTargetStep;
use crate::front::block::TypedIrExpression;
use crate::front::check::{check_type_contains_value, TypeContainsReason};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrArrayLiteralElement, IrExpression};
use crate::front::misc::ValueDomain;
use crate::front::types::{ClosedIncRange, HardwareType, Type, Typed};
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;
use itertools::{enumerate, Itertools};
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, ToPrimitive};

#[derive(Debug, Clone)]
pub enum ArrayAccessStep {
    IndexOrSliceLen {
        start: MaybeCompile<TypedIrExpression<ClosedIncRange<BigInt>>, BigInt>,
        slice_len: Option<BigUint>,
    },
    SliceUntilEnd {
        start: BigInt,
    },
}

pub fn eval_array_index_expression_step(
    diags: &Diagnostics,
    curr: Spanned<MaybeCompile<TypedIrExpression>>,
    step: Spanned<ArrayAccessStep>,
) -> Result<Spanned<MaybeCompile<TypedIrExpression>>, ErrorGuaranteed> {
    let result = match curr.inner {
        MaybeCompile::Compile(CompileValue::Type(curr)) => match step.inner {
            ArrayAccessStep::IndexOrSliceLen {
                start: len,
                slice_len: None,
            } => {
                let len = match len {
                    MaybeCompile::Compile(len) => BigUint::try_from(len).map_err(|e| {
                        diags.report_simple(
                            "array type length cannot be negative",
                            step.span,
                            format!("got `{}`", e.into_original()),
                        )
                    })?,
                    MaybeCompile::Other(_) => {
                        return Err(diags.report_simple(
                            "array type length must be compile-time constant",
                            step.span,
                            "got non-compile-time value",
                        ))
                    }
                };
                let array_ty = Type::Array(Box::new(curr), len);
                MaybeCompile::Compile(CompileValue::Type(array_ty))
            }
            ArrayAccessStep::IndexOrSliceLen {
                start: _,
                slice_len: Some(_),
            }
            | ArrayAccessStep::SliceUntilEnd { .. } => {
                return Err(diags.report_simple("array dimension length must be an integer", step.span, "got slice"));
            }
        },
        MaybeCompile::Compile(CompileValue::Array(curr_inner)) => {
            let array_len = Spanned {
                span: curr.span,
                inner: curr_inner.len(),
            };
            match step.inner {
                ArrayAccessStep::IndexOrSliceLen { start, slice_len } => match start {
                    MaybeCompile::Compile(start) => match slice_len {
                        None => {
                            let start = check_range_compile_index(diags, &start, step.span, array_len)?;
                            MaybeCompile::Compile(curr_inner[start].clone())
                        }
                        Some(slice_len) => {
                            let SliceInfo { start, slice_len } =
                                check_range_compile_slice(diags, &start, &slice_len, step.span, array_len)?;
                            MaybeCompile::Compile(CompileValue::Array(curr_inner[start..start + slice_len].to_vec()))
                        }
                    },
                    MaybeCompile::Other(start) => {
                        let array = require_array_convert_to_ir(
                            diags,
                            Spanned {
                                span: curr.span,
                                inner: MaybeCompile::Compile(CompileValue::Array(curr_inner)),
                            },
                            step.span,
                        )?;
                        MaybeCompile::Other(build_index_or_slice_ir_expression(
                            diags, array, start, slice_len, step.span,
                        )?)
                    }
                },
                ArrayAccessStep::SliceUntilEnd { start } => {
                    let start = check_range_compile_slice_start(diags, &start, step.span, array_len)?;
                    MaybeCompile::Compile(CompileValue::Array(curr_inner[start..].to_vec()))
                }
            }
        }
        MaybeCompile::Compile(curr_inner) => {
            let diag = Diagnostic::new("tried to index into non-array value")
                .add_info(
                    curr.span,
                    format!("non-array value `{}` here", curr_inner.to_diagnostic_string()),
                )
                .add_error(step.span, "tried to index here")
                .finish();
            return Err(diags.report(diag));
        }
        MaybeCompile::Other(curr_inner) => match curr_inner.ty {
            HardwareType::Array(curr_ty_element, curr_len) => {
                let array = Spanned {
                    span: curr.span,
                    inner: TypedIrExpression {
                        ty: (*curr_ty_element, curr_len),
                        domain: curr_inner.domain,
                        expr: curr_inner.expr,
                    },
                };
                match step.inner {
                    ArrayAccessStep::IndexOrSliceLen { start, slice_len } => {
                        let start = match start {
                            MaybeCompile::Compile(start) => TypedIrExpression {
                                ty: ClosedIncRange::single(start.clone()),
                                domain: ValueDomain::CompileTime,
                                expr: IrExpression::Int(start),
                            },
                            MaybeCompile::Other(start) => start,
                        };
                        MaybeCompile::Other(build_index_or_slice_ir_expression(
                            diags, array, start, slice_len, step.span,
                        )?)
                    }
                    ArrayAccessStep::SliceUntilEnd { start } => {
                        let start = Spanned {
                            span: step.span,
                            inner: start,
                        };
                        MaybeCompile::Other(build_slice_until_end_ir_expression(diags, array, start)?)
                    }
                }
            }
            _ => {
                let diag = Diagnostic::new("tried to index into non-array value")
                    .add_info(
                        curr.span,
                        format!(
                            "non-array value with type `{}` here",
                            curr_inner.ty.to_diagnostic_string()
                        ),
                    )
                    .add_error(step.span, "tried to index here")
                    .finish();
                return Err(diags.report(diag));
            }
        },
    };

    Ok(Spanned {
        span: curr.span.join(step.span),
        inner: result,
    })
}

fn require_array_convert_to_ir(
    diags: &Diagnostics,
    array: Spanned<MaybeCompile<TypedIrExpression>>,
    step_span: Span,
) -> Result<Spanned<TypedIrExpression<(HardwareType, BigUint)>>, ErrorGuaranteed> {
    let array_span = array.span;

    let result = match array.inner {
        MaybeCompile::Compile(array) => match array {
            CompileValue::Array(_) => {
                let ty = array.ty();
                let ty_hw = ty.as_hardware_type().ok_or_else(|| {
                    diags.report_internal_error(
                        array_span,
                        "converting a compile array to hardware should result in an array",
                    )
                })?;
                let array_ir = array.as_ir_expression(diags, array_span, &ty_hw)?;

                match ty_hw {
                    HardwareType::Array(array_inner_ty, array_len) => TypedIrExpression {
                        ty: (*array_inner_ty, array_len),
                        domain: ValueDomain::CompileTime,
                        expr: array_ir.expr,
                    },
                    _ => unreachable!("array compile value should turn into hardware array"),
                }
            }
            _ => {
                return Err(diags.report(diag_expected_array_value(array, array_span, step_span)));
            }
        },
        MaybeCompile::Other(array) => match array.ty {
            HardwareType::Array(array_inner_ty, array_len) => TypedIrExpression {
                ty: (*array_inner_ty, array_len),
                domain: array.domain,
                expr: array.expr,
            },
            _ => {
                return Err(diags.report(diag_expected_array_type(&array.ty.as_type(), array_span, step_span)));
            }
        },
    };

    Ok(Spanned {
        span: array_span,
        inner: result,
    })
}

pub fn build_index_or_slice_ir_expression(
    diags: &Diagnostics,
    array: Spanned<TypedIrExpression<(HardwareType, BigUint)>>,
    start: TypedIrExpression<ClosedIncRange<BigInt>>,
    slice_len: Option<BigUint>,
    step_span: Span,
) -> Result<TypedIrExpression, ErrorGuaranteed> {
    let (array_elem_ty, array_len) = array.inner.ty;
    let array_len = Spanned {
        span: array.span,
        inner: array_len,
    };

    let domain = array.inner.domain.join(&start.domain);

    match slice_len {
        None => {
            check_range_hardware_index(diags, &start.ty, step_span, array_len.as_ref())?;
            Ok(TypedIrExpression {
                ty: array_elem_ty,
                domain,
                expr: IrExpression::ArrayIndex {
                    base: Box::new(array.inner.expr),
                    index: Box::new(start.expr),
                },
            })
        }
        Some(slice_len) => {
            check_range_hardware_slice(diags, &start.ty, &slice_len, step_span, array_len.as_ref())?;
            Ok(TypedIrExpression {
                ty: HardwareType::Array(Box::new(array_elem_ty), slice_len.clone()),
                domain,
                expr: IrExpression::ArraySlice {
                    base: Box::new(array.inner.expr),
                    start: Box::new(start.expr),
                    len: slice_len,
                },
            })
        }
    }
}

fn build_slice_until_end_ir_expression(
    diags: &Diagnostics,
    array: Spanned<TypedIrExpression<(HardwareType, BigUint)>>,
    start: Spanned<BigInt>,
) -> Result<TypedIrExpression, ErrorGuaranteed> {
    let (array_elem_ty, array_len) = array.inner.ty;
    let array_len = Spanned {
        span: array.span,
        inner: array_len,
    };
    let domain = array.inner.domain;

    let SliceLength(slice_len) = check_range_hardware_slice_start(diags, &start.inner, start.span, array_len.as_ref())?;

    Ok(TypedIrExpression {
        ty: HardwareType::Array(Box::new(array_elem_ty), slice_len.clone()),
        domain,
        expr: IrExpression::ArraySlice {
            base: Box::new(array.inner.expr),
            start: Box::new(IrExpression::Int(start.inner)),
            len: slice_len,
        },
    })
}

#[derive(Debug, Copy, Clone)]
pub struct VarAssignSpans {
    pub variable_declaration: Span,
    pub assigned_value: Span,
    pub assignment_operator: Span,
}

// TODO rename function and params (eg. old_value/curr_value)
// TODO split into more functions
pub fn handle_array_variable_assignment_steps(
    diags: &Diagnostics,
    value: Spanned<MaybeCompile<TypedIrExpression>>,
    expected_ty: Option<Spanned<&Type>>,
    steps: &[Spanned<AssignmentTargetStep>],
    spans: VarAssignSpans,
    eval_value: impl FnOnce(
        Spanned<MaybeCompile<TypedIrExpression>>,
        &Type,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed>,
) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
    let value_span = value.span;

    let (step, steps_remaining) = match steps.split_first() {
        None => {
            // all steps are done, do the actual evaluation and type-checking
            let expected_ty_eval = expected_ty.map_or(&Type::Any, |ty| ty.inner);
            let result = eval_value(value, expected_ty_eval)?;

            if let Some(expected_ty) = expected_ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: value_span,
                    span_target_ty: expected_ty.span,
                };
                let result_spanned = Spanned {
                    span: value_span,
                    inner: &result,
                };
                check_type_contains_value(diags, reason, expected_ty.inner, result_spanned, true, true)?;
            }

            return Ok(result);
        }
        Some(split) => split,
    };

    let next_span = value_span.join(step.span);

    match &step.inner {
        AssignmentTargetStep::ArrayAccess(access) => match access {
            ArrayAccessStep::IndexOrSliceLen { start, slice_len } => match (value.inner, start) {
                (MaybeCompile::Compile(value), MaybeCompile::Compile(start)) => match value {
                    CompileValue::Array(values) => {
                        let values_len = Spanned {
                            span: value_span,
                            inner: values.len(),
                        };
                        let ty_inner = expected_ty
                            .map(|ty| {
                                ty.map_inner(|ty_inner| match ty_inner {
                                    Type::Array(ty_inner, _) => Ok(&**ty_inner),
                                    _ => {
                                        let diag =
                                            Diagnostic::new("array index operation on variable with non-array type")
                                                .add_error(step.span, "this indexing step requires an array type")
                                                .add_info(ty.span, "variable type set here")
                                                .finish();
                                        Err(diags.report(diag))
                                    }
                                })
                                .transpose()
                            })
                            .transpose()?;

                        match slice_len {
                            None => {
                                let start = check_range_compile_index(diags, start, step.span, values_len)?;

                                let mut values = values;
                                let value_to_replace = std::mem::replace(&mut values[start], CompileValue::Undefined);

                                let expected_ty_replace =
                                    ty_inner.map(|ty_inner| ty_inner.map_inner(|ty_inner| ty_inner.clone()));
                                let value_replaced = handle_array_variable_assignment_steps(
                                    diags,
                                    Spanned {
                                        span: next_span,
                                        inner: MaybeCompile::Compile(value_to_replace),
                                    },
                                    expected_ty_replace.as_ref().map(Spanned::as_ref),
                                    steps_remaining,
                                    spans,
                                    eval_value,
                                )?;

                                match value_replaced {
                                    MaybeCompile::Compile(value_replaced) => {
                                        values[start] = value_replaced;
                                        Ok(MaybeCompile::Compile(CompileValue::Array(values)))
                                    }
                                    MaybeCompile::Other(value_replaced) => {
                                        Ok(MaybeCompile::Other(replace_compile_array_slice_with_hardware(
                                            diags,
                                            expected_ty,
                                            values,
                                            value_span,
                                            start,
                                            None,
                                            spans,
                                            value_replaced,
                                        )?))
                                    }
                                }
                            }
                            Some(slice_len) => {
                                let SliceInfo { start, slice_len } =
                                    check_range_compile_slice(diags, start, slice_len, step.span, values_len)?;

                                let mut values = values;

                                let mut values_to_replace = vec![];
                                for di in 0..slice_len {
                                    values_to_replace
                                        .push(std::mem::replace(&mut values[start + di], CompileValue::Undefined));
                                }
                                let values_to_replace = CompileValue::Array(values_to_replace);

                                let expected_ty_replace = ty_inner.map(|ty_inner| {
                                    ty_inner.map_inner(|ty_inner| {
                                        Type::Array(Box::new(ty_inner.clone()), BigUint::from(slice_len))
                                    })
                                });
                                let values_replaced = handle_array_variable_assignment_steps(
                                    diags,
                                    Spanned {
                                        span: next_span,
                                        inner: MaybeCompile::Compile(values_to_replace),
                                    },
                                    expected_ty_replace.as_ref().map(Spanned::as_ref),
                                    steps_remaining,
                                    spans,
                                    eval_value,
                                )?;

                                match values_replaced {
                                    MaybeCompile::Compile(values_replaced) => {
                                        let values_replaced = match values_replaced {
                                            CompileValue::Array(values) => values,
                                            _ => {
                                                return Err(diags.report_internal_error(
                                                    step.span,
                                                    "replaced value should be an array again",
                                                ))
                                            }
                                        };
                                        for (di, replaced) in enumerate(values_replaced) {
                                            values[start + di] = replaced;
                                        }
                                        Ok(MaybeCompile::Compile(CompileValue::Array(values)))
                                    }
                                    MaybeCompile::Other(values_replaced) => {
                                        Ok(MaybeCompile::Other(replace_compile_array_slice_with_hardware(
                                            diags,
                                            expected_ty,
                                            values,
                                            value_span,
                                            start,
                                            Some(slice_len),
                                            spans,
                                            values_replaced,
                                        )?))
                                    }
                                }
                            }
                        }
                    }
                    _ => Err(diags.report(diag_expected_array_value(value, value_span, step.span))),
                },
                (value, index) => {
                    let value_spanned = Spanned {
                        span: value_span,
                        inner: value,
                    };
                    let array = require_array_convert_to_ir(diags, value_spanned, step.span)?;
                    let (array_inner_ty, array_len) = array.inner.ty;
                    let array_len = Spanned {
                        span: array.span,
                        inner: array_len,
                    };

                    let (index_expr, index_domain) = match index {
                        MaybeCompile::Compile(index) => {
                            check_range_mixed_index(diags, index, step.span, array_len.as_ref())?;
                            (IrExpression::Int(index.clone()), ValueDomain::CompileTime)
                        }
                        MaybeCompile::Other(index) => {
                            check_range_hardware_index(diags, &index.ty, step.span, array_len.as_ref())?;
                            (index.expr.clone(), index.domain.clone())
                        }
                    };

                    Ok(MaybeCompile::Other(TypedIrExpression {
                        ty: array_inner_ty,
                        domain: array.inner.domain.join(&index_domain),
                        expr: IrExpression::ArrayIndex {
                            base: Box::new(array.inner.expr),
                            index: Box::new(index_expr),
                        },
                    }))
                }
            },
            ArrayAccessStep::SliceUntilEnd { start } => match value.inner {
                MaybeCompile::Compile(value) => match value {
                    CompileValue::Array(values) => {
                        let array_len = Spanned {
                            span: value_span,
                            inner: values.len(),
                        };
                        let start = check_range_compile_slice_start(diags, start, step.span, array_len)?;
                        Ok(MaybeCompile::Compile(CompileValue::Array(values[start..].to_vec())))
                    }
                    _ => Err(diags.report(diag_expected_array_value(value, value_span, step.span))),
                },
                MaybeCompile::Other(value) => match value.ty {
                    HardwareType::Array(array_ty_inner, array_len) => {
                        let array_len = Spanned {
                            span: value_span,
                            inner: array_len,
                        };
                        let SliceLength(slice_len) =
                            check_range_hardware_slice_start(diags, start, step.span, array_len.as_ref())?;

                        Ok(MaybeCompile::Other(TypedIrExpression {
                            ty: *array_ty_inner,
                            domain: value.domain,
                            expr: IrExpression::ArraySlice {
                                base: Box::new(value.expr),
                                start: Box::new(IrExpression::Int(start.clone())),
                                len: slice_len,
                            },
                        }))
                    }
                    _ => Err(diags.report(diag_expected_array_type(&value.ty.as_type(), value_span, step.span))),
                },
            },
        },
    }
}

fn replace_compile_array_slice_with_hardware(
    diags: &Diagnostics,
    expected_ty: Option<Spanned<&Type>>,
    values: Vec<CompileValue>,
    values_span: Span,
    start: usize,
    slice_len: Option<usize>,
    spans: VarAssignSpans,
    replacement: TypedIrExpression,
) -> Result<TypedIrExpression, ErrorGuaranteed> {
    // figure out the array type
    let expected_ty = expected_ty.ok_or_else(|| {
        let diag = Diagnostic::new("converting array to hardware needs type hint")
            .add_error(spans.assignment_operator, "necessary for this assignment")
            .add_info(values_span, "compile-time array here")
            .add_info(spans.assigned_value, "non-compile-time assigned value here")
            .add_info(spans.variable_declaration, "this variable needs a type hint")
            .finish();
        diags.report(diag)
    })?;
    let expected_ty = expected_ty.inner.as_hardware_type().ok_or_else(|| {
        let diag = Diagnostic::new("converting array to hardware needs array with hardware type hint")
            .add_error(spans.assignment_operator, "necessary for this assignment")
            .add_info(values_span, "compile-time array here")
            .add_info(spans.assigned_value, "non-compile-time assigned value here")
            .add_info(
                expected_ty.span,
                format!(
                    "type `{}` is not representable in hardware",
                    expected_ty.inner.to_diagnostic_string()
                ),
            )
            .finish();
        diags.report(diag)
    })?;

    let array_ty_inner = match expected_ty {
        HardwareType::Array(inner, _len) => *inner,
        _ => return Err(diags.report_internal_error(values_span, "expected type should be an array")),
    };

    // double-check that the replacement value has the right type
    // (this should have been checked with a proper error message earlier already)
    let expected_replacement_ty = match slice_len {
        None => array_ty_inner.clone(),
        Some(slice_len) => HardwareType::Array(Box::new(array_ty_inner.clone()), BigUint::from(slice_len)),
    };
    if expected_replacement_ty != replacement.ty {
        return Err(diags.report_internal_error(values_span, "incorrect replacement type"));
    };

    // convert necessary values into hardware
    let values_len = values.len();
    let mut values = values;
    let after = values.drain(start + slice_len.unwrap_or(1)..).collect_vec();
    let before = values.into_iter().take(start).collect();

    let before_expected_ty = HardwareType::Array(Box::new(array_ty_inner.clone()), BigUint::from(start));
    let before = CompileValue::Array(before).as_ir_expression(diags, values_span, &before_expected_ty)?;
    let after_expected_ty = HardwareType::Array(Box::new(array_ty_inner.clone()), BigUint::from(after.len()));
    let after = CompileValue::Array(after).as_ir_expression(diags, values_span, &after_expected_ty)?;

    // concatenate
    let element_inner = if slice_len.is_some() {
        IrArrayLiteralElement::Spread(replacement.expr)
    } else {
        IrArrayLiteralElement::Single(replacement.expr)
    };

    let literal_elements = vec![
        IrArrayLiteralElement::Spread(before.expr),
        element_inner,
        IrArrayLiteralElement::Spread(after.expr),
    ];
    let expr = IrExpression::ArrayLiteral(array_ty_inner.to_ir(), BigUint::from(values_len), literal_elements);

    Ok(TypedIrExpression {
        ty: HardwareType::Array(Box::new(array_ty_inner), BigUint::from(values_len)),
        domain: replacement.domain,
        expr,
    })
}

pub fn check_range_compile_index(
    diags: &Diagnostics,
    index: &BigInt,
    index_span: Span,
    array_len: Spanned<usize>,
) -> Result<usize, ErrorGuaranteed> {
    if index.is_negative() || index >= &BigInt::from(array_len.inner) {
        let diag = Diagnostic::new("array index out of bounds")
            .add_error(index_span, format!("this index `{index}` is out of bounds"))
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }

    Ok(index.to_usize().unwrap())
}

pub fn check_range_hardware_index(
    diags: &Diagnostics,
    index: &ClosedIncRange<BigInt>,
    index_span: Span,
    array_len: Spanned<&BigUint>,
) -> Result<(), ErrorGuaranteed> {
    if index.start_inc.is_negative() || index.end_inc >= BigInt::from(array_len.inner.clone()) {
        let diag = Diagnostic::new("array index out of bounds")
            .add_error(index_span, format!("this index with type `{index}` is out of bounds"))
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(())
}

pub fn check_range_mixed_index(
    diags: &Diagnostics,
    index: &BigInt,
    index_span: Span,
    array_len: Spanned<&BigUint>,
) -> Result<(), ErrorGuaranteed> {
    if index.is_negative() || index >= &BigInt::from(array_len.inner.clone()) {
        let diag = Diagnostic::new("array index out of bounds")
            .add_error(index_span, format!("this index with type `{index}` is out of bounds"))
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(())
}

pub struct SliceInfo {
    pub start: usize,
    pub slice_len: usize,
}

pub fn check_range_compile_slice(
    diags: &Diagnostics,
    start: &BigInt,
    slice_len: &BigUint,
    step_span: Span,
    array_len: Spanned<usize>,
) -> Result<SliceInfo, ErrorGuaranteed> {
    if start.is_negative() || (start + BigInt::from(slice_len.clone())) > BigInt::from(array_len.inner) {
        let diag = Diagnostic::new("array slice range out of bounds")
            .add_error(
                step_span,
                format!("this range `{start}..+{slice_len}` is out out of bounds"),
            )
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(SliceInfo {
        start: start.to_usize().unwrap(),
        slice_len: slice_len.to_usize().unwrap(),
    })
}

pub fn check_range_hardware_slice(
    diags: &Diagnostics,
    start: &ClosedIncRange<BigInt>,
    slice_len: &BigUint,
    step_span: Span,
    array_len: Spanned<&BigUint>,
) -> Result<(), ErrorGuaranteed> {
    if start.start_inc.is_negative()
        || (&start.end_inc + BigInt::from(slice_len.clone())) > BigInt::from(array_len.inner.clone())
    {
        let diag = Diagnostic::new("array slice range out of bounds")
            .add_error(
                step_span,
                format!("this range `({start})..+{slice_len}` is out out of bounds"),
            )
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(())
}

pub fn check_range_compile_slice_start(
    diags: &Diagnostics,
    start: &BigInt,
    step_span: Span,
    array_len: Spanned<usize>,
) -> Result<usize, ErrorGuaranteed> {
    if start.is_negative() || start > &BigInt::from(array_len.inner) {
        let diag = Diagnostic::new("array slice range out of bounds")
            .add_error(step_span, format!("this range `{start}..` is out out of bounds"))
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(start.to_usize().unwrap())
}

pub struct SliceLength(pub BigUint);

pub fn check_range_hardware_slice_start(
    diags: &Diagnostics,
    start: &BigInt,
    step_span: Span,
    array_len: Spanned<&BigUint>,
) -> Result<SliceLength, ErrorGuaranteed> {
    if start.is_negative() || start > &BigInt::from(array_len.inner.clone()) {
        let diag = Diagnostic::new("array slice range out of bounds")
            .add_error(step_span, format!("this range `{start}..` is out out of bounds"))
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(SliceLength(start.to_biguint().unwrap()))
}

pub fn diag_expected_array_value(value: CompileValue, value_span: Span, step_span: Span) -> Diagnostic {
    Diagnostic::new("array indexing on non-array type")
        .add_error(step_span, "array access operation here")
        .add_info(
            value_span,
            format!("on non-array value `{}` here", value.to_diagnostic_string()),
        )
        .finish()
}

pub fn diag_expected_array_type(value_ty: &Type, value_span: Span, step_span: Span) -> Diagnostic {
    Diagnostic::new("array indexing on non-array type")
        .add_error(step_span, "array access operation here")
        .add_info(
            value_span,
            format!(
                "on non-array value with type `{}` here",
                value_ty.to_diagnostic_string()
            ),
        )
        .finish()
}
