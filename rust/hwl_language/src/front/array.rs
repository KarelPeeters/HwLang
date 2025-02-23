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
use unwrap_match::unwrap_match;

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
        MaybeCompile::Compile(CompileValue::Array(curr_inner)) => match step.inner {
            ArrayAccessStep::IndexOrSliceLen { start, slice_len } => match start {
                MaybeCompile::Compile(start) => match slice_len {
                    None => {
                        let start = check_range_compile_index(&start, curr_inner.len())?;
                        MaybeCompile::Compile(curr_inner[start].clone())
                    }
                    Some(slice_len) => {
                        let (start, slice_len) = check_range_compile_slice(&start, &slice_len, curr_inner.len())?;
                        MaybeCompile::Compile(CompileValue::Array(curr_inner[start..start + slice_len].to_vec()))
                    }
                },
                MaybeCompile::Other(start) => {
                    // convert array to ir
                    // TODO extract all of this to a function
                    let curr_len = BigUint::from(curr_inner.len());
                    let curr_array = CompileValue::Array(curr_inner);
                    let curr_ty = curr_array.ty().as_hardware_type().unwrap_or_else(|| todo!("error"));
                    let curr_ir = curr_array.as_ir_expression(diags, curr.span, &curr_ty)?;
                    let curr_ty_element = unwrap_match!(curr_ty, HardwareType::Array(inner, _) => *inner);

                    let result_expr = match slice_len {
                        None => {
                            check_range_hardware_index(&start.ty, &curr_len)?;
                            IrExpression::ArrayIndex {
                                base: Box::new(curr_ir.expr),
                                index: Box::new(start.expr),
                            }
                        }
                        Some(slice_len) => {
                            check_range_hardware_slice(&start.ty, &slice_len, &curr_len)?;
                            IrExpression::ArraySlice {
                                base: Box::new(curr_ir.expr),
                                start: Box::new(start.expr),
                                len: slice_len,
                            }
                        }
                    };

                    MaybeCompile::Other(TypedIrExpression {
                        ty: curr_ty_element,
                        domain: start.domain,
                        expr: result_expr,
                    })
                }
            },
            ArrayAccessStep::SliceUntilEnd { start } => {
                let start = check_range_compile_slice_start(&start, curr_inner.len())?;
                MaybeCompile::Compile(CompileValue::Array(curr_inner[start..].to_vec()))
            }
        },
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
            HardwareType::Array(curr_ty_element, curr_len) => match step.inner {
                ArrayAccessStep::IndexOrSliceLen { start, slice_len } => {
                    let start = match start {
                        MaybeCompile::Compile(start) => TypedIrExpression {
                            ty: ClosedIncRange::single(start.clone()),
                            domain: ValueDomain::CompileTime,
                            expr: IrExpression::Int(start),
                        },
                        MaybeCompile::Other(start) => start,
                    };

                    let result_expr = match slice_len {
                        None => {
                            check_range_hardware_index(&start.ty, &curr_len)?;
                            IrExpression::ArrayIndex {
                                base: Box::new(curr_inner.expr),
                                index: Box::new(start.expr),
                            }
                        }
                        Some(slice_len) => {
                            check_range_hardware_slice(&start.ty, &slice_len, &curr_len)?;
                            IrExpression::ArraySlice {
                                base: Box::new(curr_inner.expr),
                                start: Box::new(start.expr),
                                len: slice_len,
                            }
                        }
                    };

                    MaybeCompile::Other(TypedIrExpression {
                        ty: *curr_ty_element,
                        domain: start.domain,
                        expr: result_expr,
                    })
                }
                ArrayAccessStep::SliceUntilEnd { start } => {
                    check_range_hardware_slice_start(&start, &curr_len)?;
                    let slice_len = curr_len - start.to_biguint().unwrap();

                    MaybeCompile::Other(TypedIrExpression {
                        ty: *curr_ty_element,
                        domain: curr_inner.domain,
                        expr: IrExpression::ArraySlice {
                            base: Box::new(curr_inner.expr),
                            start: Box::new(IrExpression::Int(start)),
                            len: slice_len,
                        },
                    })
                }
            },
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

    // let next_inner = match (curr.inner, step.inner) {
    //     (MaybeCompile::Compile(curr_inner), MaybeCompile::Compile(index_inner)) => {
    //         let index = Spanned {
    //             span: index.span,
    //             inner: index_inner,
    //         };
    //
    //         match curr_inner {
    //             CompileValue::Type(curr) => {
    //                 // declare new array type
    //                 let dim_len = match index.inner {
    //                     CompileValue::Int(dim_len) => BigUint::try_from(dim_len).map_err(|e| {
    //                         diags.report_simple(
    //                             "array dimension length cannot be negative",
    //                             index.span,
    //                             format!("got value `{}`", e.into_original()),
    //                         )
    //                     })?,
    //                     _ => {
    //                         return Err(diags.report_simple(
    //                             "array dimension length must be an integer",
    //                             index.span,
    //                             format!("got value `{}`", index.inner.to_diagnostic_string()),
    //                         ));
    //                     }
    //                 };
    //
    //                 let result_ty = Type::Array(Box::new(curr), dim_len);
    //                 MaybeCompile::Compile(CompileValue::Type(result_ty))
    //             }
    //             CompileValue::Array(curr_inner) => {
    //                 // simple compile-time array indexing
    //                 match check_array_index(diags, curr.span, BigUint::from(curr_inner.len()), index)? {
    //                     CheckedArrayIndex::Single { index } => {
    //                         let index = index.to_usize().unwrap();
    //                         MaybeCompile::Compile(curr_inner[index].clone())
    //                     }
    //                     CheckedArrayIndex::Slice { start, len } => {
    //                         let start = start.to_usize().unwrap();
    //                         let len = len.to_usize().unwrap();
    //                         MaybeCompile::Compile(CompileValue::Array(curr_inner[start..start + len].to_vec()))
    //                     }
    //                 }
    //             }
    //             _ => {
    //                 return Err(diags.report_simple(
    //                     "array index on invalid target, must be type or array",
    //                     index.span,
    //                     format!("target `{}` here", curr_inner.to_diagnostic_string()),
    //                 ));
    //             }
    //         }
    //     }
    //     (curr_inner, index_inner) => {
    //         let index = Spanned {
    //             span: index.span,
    //             inner: index_inner,
    //         };
    //
    //         let (curr_ty_inner, curr_len) = match curr_inner.ty() {
    //             Type::Array(curr_ty_inner, curr_len) => (*curr_ty_inner, curr_len),
    //             _ => {
    //                 let diag = Diagnostic::new("array index on invalid target, must be array")
    //                     .add_error(
    //                         curr.span,
    //                         format!(
    //                             "target with non-array type `{}` here",
    //                             curr_inner.ty().to_diagnostic_string()
    //                         ),
    //                     )
    //                     .add_info(index.span, "array indexed here")
    //                     .finish();
    //
    //                 return Err(diags.report(diag));
    //             }
    //         };
    //         let curr_ty_inner_hw = curr_ty_inner.as_hardware_type().ok_or_else(|| {
    //             let diag =
    //                 Diagnostic::new("non-compile-time array index on array that is not representable in hardware")
    //                     .add_error(
    //                         curr.span,
    //                         format!(
    //                             "target with non-hardware type `{}` here",
    //                             curr_inner.ty().to_diagnostic_string()
    //                         ),
    //                     )
    //                     .add_info(index.span, "array indexed with non-compile-time index here")
    //                     .finish();
    //
    //             diags.report(diag)
    //         })?;
    //         let curr_ir = curr_inner.as_ir_expression(
    //             diags,
    //             curr.span,
    //             &HardwareType::Array(Box::new(curr_ty_inner_hw.clone()), curr_len.clone()),
    //         )?;
    //         let base = Box::new(curr_ir.expr);
    //
    //         let result = match index.inner {
    //             MaybeCompile::Compile(index_inner) => {
    //                 let index = Spanned {
    //                     span: index.span,
    //                     inner: index_inner,
    //                 };
    //                 match check_array_index(diags, curr.span, curr_len, index)? {
    //                     CheckedArrayIndex::Single { index } => {
    //                         let expr = IrExpression::ArrayIndex {
    //                             base,
    //                             index: Box::new(IrExpression::Int(BigInt::from(index))),
    //                         };
    //                         TypedIrExpression {
    //                             ty: curr_ty_inner_hw,
    //                             domain: curr_ir.domain,
    //                             expr,
    //                         }
    //                     }
    //                     CheckedArrayIndex::Slice { start, len } => {
    //                         let expr = IrExpression::ArraySlice {
    //                             base,
    //                             start: Box::new(IrExpression::Int(BigInt::from(start))),
    //                             len: len.clone(),
    //                         };
    //                         TypedIrExpression {
    //                             ty: HardwareType::Array(Box::new(curr_ty_inner_hw), len),
    //                             domain: curr_ir.domain,
    //                             expr,
    //                         }
    //                     }
    //                 }
    //             }
    //             MaybeCompile::Other(index_inner) => {
    //                 match index_inner {
    //                     IrArrayIndexKind::IndexSingle(index_inner) => {
    //                         // check type and range
    //                         let reason = TypeContainsReason::ArrayIndex { span_index: index.span };
    //                         let valid_range = IncRange {
    //                             start_inc: Some(BigInt::ZERO),
    //                             end_inc: Some(BigInt::from(curr_len) - 1),
    //                         };
    //                         let value_ty = Spanned {
    //                             span: index.span,
    //                             inner: &index_inner.ty.as_type(),
    //                         };
    //                         check_type_contains_type(diags, reason, &Type::Int(valid_range), value_ty, false)?;
    //
    //                         // build expression
    //                         let expr = IrExpression::ArrayIndex {
    //                             base,
    //                             index: Box::new(index_inner.expr),
    //                         };
    //                         TypedIrExpression {
    //                             ty: curr_ty_inner_hw,
    //                             domain: curr_ir.domain.join(&index_inner.domain),
    //                             expr,
    //                         }
    //                     }
    //                     IrArrayIndexKind::SliceRange { start, len } => {
    //                         // check ranges
    //                         let min_start = &start.ty.start_inc;
    //                         let max_end_inc = &start.ty.end_inc + BigInt::from(len.clone()) - 1;
    //
    //                         let valid_start_range = ClosedIncRange {
    //                             start_inc: BigInt::ZERO,
    //                             end_inc: BigInt::from(curr_len.clone()),
    //                         };
    //                         let valid_end_inc_range = ClosedIncRange {
    //                             start_inc: BigInt::from(-1),
    //                             end_inc: BigInt::from(curr_len) - 1,
    //                         };
    //
    //                         let start_valid = valid_start_range.contains(&min_start);
    //                         let end_valid = valid_end_inc_range.contains(&max_end_inc);
    //
    //                         if !start_valid || !end_valid {
    //                             let invalid = match (start_valid, end_valid) {
    //                                 (false, false) => "start and end",
    //                                 (false, true) => "start",
    //                                 (true, false) => "end",
    //                                 (true, true) => unreachable!(),
    //                             };
    //
    //                             let msg_error = format!(
    //                                 "got slice range with start range `{}` and length `{}`,resulting in total range `{}`",
    //                                 start.ty,
    //                                 len,
    //                                 ClosedIncRange {
    //                                     start_inc: min_start,
    //                                     end_inc: &max_end_inc,
    //                                 }
    //                             );
    //                             let msg_info = format!(
    //                                 "for this array the valid start range is `{}` and end range is `{}`",
    //                                 valid_start_range, valid_end_inc_range
    //                             );
    //
    //                             let diag = Diagnostic::new(format!("array slice range {invalid} out of bounds"))
    //                                 .add_error(index.span, msg_error)
    //                                 .add_info(curr.span, msg_info)
    //                                 .finish();
    //                             return Err(diags.report(diag));
    //                         }
    //
    //                         // build result
    //                         let expr = IrExpression::ArraySlice {
    //                             base,
    //                             start: Box::new(start.expr),
    //                             len: len.clone(),
    //                         };
    //                         TypedIrExpression {
    //                             ty: HardwareType::Array(Box::new(curr_ty_inner_hw), len),
    //                             domain: curr_ir.domain.join(&start.domain),
    //                             expr,
    //                         }
    //                     }
    //                 }
    //             }
    //         };
    //
    //         MaybeCompile::Other(result)
    //     }
    // };

    // // TODO this is a bit of sketchy span, but maybe the best we can do
    // Ok(Spanned {
    //     span: curr.span.join(index.span),
    //     inner: next_inner,
    // })
}

// TODO rename
// TODO split into more functions
pub fn handle_array_variable_assignment_steps(
    diags: &Diagnostics,
    // TODO rename params, eg. old_value
    full_span: Span,
    value: MaybeCompile<TypedIrExpression>,
    expected_ty: Option<Spanned<&Type>>,
    steps: &[Spanned<AssignmentTargetStep>],
    eval_value: impl FnOnce(
        Spanned<MaybeCompile<TypedIrExpression>>,
        &Type,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed>,
) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
    if let Some((step, rest)) = steps.split_first() {
        match &step.inner {
            AssignmentTargetStep::ArrayAccess(access) => {
                match access {
                    ArrayAccessStep::IndexOrSliceLen { start, slice_len } => match (value, start) {
                        (MaybeCompile::Compile(value), MaybeCompile::Compile(start)) => match value {
                            CompileValue::Array(values) => {
                                let values_len = values.len();
                                let ty_inner = expected_ty.map(|ty| {
                                    ty.map_inner(|ty| match ty {
                                        Type::Array(ty_inner, _) => &**ty_inner,
                                        _ => todo!("error"),
                                    })
                                });

                                match slice_len {
                                    None => {
                                        let start = check_range_compile_index(start, values_len)?;

                                        let mut values = values;
                                        let value_to_replace =
                                            std::mem::replace(&mut values[start], CompileValue::Undefined);

                                        let expected_ty_replace =
                                            ty_inner.map(|ty_inner| ty_inner.map_inner(|ty_inner| ty_inner.clone()));
                                        let value_replaced = handle_array_variable_assignment_steps(
                                            diags,
                                            full_span,
                                            MaybeCompile::Compile(value_to_replace),
                                            expected_ty_replace.as_ref().map(Spanned::as_ref),
                                            rest,
                                            eval_value,
                                        )?;

                                        match value_replaced {
                                            MaybeCompile::Compile(value_replaced) => {
                                                values[start] = value_replaced;
                                                Ok(MaybeCompile::Compile(CompileValue::Array(values)))
                                            }
                                            MaybeCompile::Other(value_replaced) => {
                                                // convert existing array elements into hardware
                                                // TODO create function for this
                                                // TODO fix duplication
                                                let expected_ty = expected_ty
                                                    .unwrap_or_else(|| todo!("error"))
                                                    .inner
                                                    .as_hardware_type()
                                                    .unwrap_or_else(|| todo!("error"));
                                                let array_ty_inner = match expected_ty {
                                                    HardwareType::Array(inner, _len) => *inner,
                                                    _ => todo!("error"),
                                                };

                                                let after = values.drain(start + 1..).collect_vec();
                                                let before = values.into_iter().take(start).collect();

                                                let before_expected_ty = HardwareType::Array(
                                                    Box::new(array_ty_inner.clone()),
                                                    BigUint::from(start),
                                                );
                                                let before = CompileValue::Array(before).as_ir_expression(
                                                    diags,
                                                    full_span,
                                                    &before_expected_ty,
                                                )?;
                                                let after_expected_ty = HardwareType::Array(
                                                    Box::new(array_ty_inner.clone()),
                                                    BigUint::from(after.len()),
                                                );
                                                let after = CompileValue::Array(after).as_ir_expression(
                                                    diags,
                                                    full_span,
                                                    &after_expected_ty,
                                                )?;

                                                // concatenate
                                                let literal_elements = vec![
                                                    IrArrayLiteralElement::Spread(before.expr),
                                                    IrArrayLiteralElement::Single(value_replaced.expr),
                                                    IrArrayLiteralElement::Spread(after.expr),
                                                ];
                                                let expr = IrExpression::ArrayLiteral(
                                                    array_ty_inner.to_ir(),
                                                    literal_elements,
                                                );

                                                Ok(MaybeCompile::Other(TypedIrExpression {
                                                    ty: HardwareType::Array(
                                                        Box::new(array_ty_inner),
                                                        BigUint::from(values_len),
                                                    ),
                                                    domain: value_replaced.domain,
                                                    expr,
                                                }))
                                            }
                                        }
                                    }
                                    Some(slice_len) => {
                                        let (start, slice_len) =
                                            check_range_compile_slice(start, slice_len, values_len)?;

                                        let mut values = values;

                                        let mut values_to_replace = vec![];
                                        for di in 0..slice_len {
                                            values_to_replace.push(std::mem::replace(
                                                &mut values[start + di],
                                                CompileValue::Undefined,
                                            ));
                                        }
                                        let values_to_replace = CompileValue::Array(values_to_replace);

                                        let expected_ty_replace = ty_inner.map(|ty_inner| {
                                            ty_inner.map_inner(|ty_inner| {
                                                Type::Array(Box::new(ty_inner.clone()), BigUint::from(slice_len))
                                            })
                                        });
                                        let values_replaced = handle_array_variable_assignment_steps(
                                            diags,
                                            full_span,
                                            MaybeCompile::Compile(values_to_replace),
                                            expected_ty_replace.as_ref().map(Spanned::as_ref),
                                            rest,
                                            eval_value,
                                        )?;

                                        match values_replaced {
                                            MaybeCompile::Compile(values_replaced) => {
                                                let values_replaced = match values_replaced {
                                                    CompileValue::Array(values) => values,
                                                    _ => todo!("error"),
                                                };
                                                for (di, replaced) in enumerate(values_replaced) {
                                                    values[start + di] = replaced;
                                                }
                                                Ok(MaybeCompile::Compile(CompileValue::Array(values)))
                                            }
                                            MaybeCompile::Other(values_replaced) => {
                                                // convert existing array elements into hardware
                                                let expected_ty = expected_ty
                                                    .unwrap_or_else(|| todo!("error"))
                                                    .inner
                                                    .as_hardware_type()
                                                    .unwrap_or_else(|| todo!("error"));
                                                let array_ty_inner = match expected_ty {
                                                    HardwareType::Array(inner, _len) => *inner,
                                                    _ => todo!("error"),
                                                };

                                                let after = values.drain(start + slice_len..).collect_vec();
                                                let before = values.into_iter().take(start).collect();

                                                let before_expected_ty = HardwareType::Array(
                                                    Box::new(array_ty_inner.clone()),
                                                    BigUint::from(start),
                                                );
                                                let before = CompileValue::Array(before).as_ir_expression(
                                                    diags,
                                                    full_span,
                                                    &before_expected_ty,
                                                )?;
                                                let after_expected_ty = HardwareType::Array(
                                                    Box::new(array_ty_inner.clone()),
                                                    BigUint::from(after.len()),
                                                );
                                                let after = CompileValue::Array(after).as_ir_expression(
                                                    diags,
                                                    full_span,
                                                    &after_expected_ty,
                                                )?;

                                                // concatenate
                                                let literal_elements = vec![
                                                    IrArrayLiteralElement::Spread(before.expr),
                                                    IrArrayLiteralElement::Spread(values_replaced.expr),
                                                    IrArrayLiteralElement::Spread(after.expr),
                                                ];
                                                let expr = IrExpression::ArrayLiteral(
                                                    array_ty_inner.to_ir(),
                                                    literal_elements,
                                                );

                                                Ok(MaybeCompile::Other(TypedIrExpression {
                                                    ty: HardwareType::Array(
                                                        Box::new(array_ty_inner),
                                                        BigUint::from(values_len),
                                                    ),
                                                    domain: values_replaced.domain,
                                                    expr,
                                                }))
                                            }
                                        }
                                    }
                                }

                                // let mut before = values;
                                // let after = before.drain(start+len..).collect_vec();
                                // let value_to_replace = match slice_len {
                                //     None => before.pop().unwrap(),
                                //     Some(_) => CompileValue::Array(before.drain(start..).collect_vec()),
                                // };
                            }
                            _ => todo!("error"),
                        },
                        (value, index) => {
                            // at least one of them is non-compile, so convert both to hardware
                            // TODO common error message for hardware conversion fails
                            let value_ty = value.ty().as_hardware_type().unwrap_or_else(|| todo!("error"));
                            // TODO use better span
                            let value = value.as_ir_expression(diags, full_span, &value_ty)?;

                            let (array_ty_inner, array_len) = match value.ty {
                                HardwareType::Array(array_ty_inner, array_len) => (*array_ty_inner, array_len),
                                _ => todo!("error"),
                            };

                            let (index_expr, index_domain) = match index {
                                MaybeCompile::Compile(index) => {
                                    check_range_mixed_index(index, &array_len)?;
                                    (IrExpression::Int(index.clone()), ValueDomain::CompileTime)
                                }
                                MaybeCompile::Other(index) => {
                                    check_range_hardware_index(&index.ty, &array_len)?;
                                    (index.expr.clone(), index.domain.clone())
                                }
                            };

                            Ok(MaybeCompile::Other(TypedIrExpression {
                                ty: array_ty_inner,
                                domain: value.domain.join(&index_domain),
                                expr: IrExpression::ArrayIndex {
                                    base: Box::new(value.expr),
                                    index: Box::new(index_expr),
                                },
                            }))
                        }
                    },
                    ArrayAccessStep::SliceUntilEnd { start } => match value {
                        MaybeCompile::Compile(value) => match value {
                            CompileValue::Array(values) => {
                                let start = check_range_compile_slice_start(start, values.len())?;
                                Ok(MaybeCompile::Compile(CompileValue::Array(values[start..].to_vec())))
                            }
                            _ => todo!("error"),
                        },
                        MaybeCompile::Other(value) => match value.ty {
                            HardwareType::Array(array_ty_inner, array_len) => {
                                let SliceLength(slice_len) = check_range_mixed_slice_start(start, &array_len)?;

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
                            _ => todo!("error"),
                        },
                    },
                }
            }
        }

        // TODO continue recursing, building the expression back up
    } else {
        let expected_ty_eval = expected_ty.map_or(&Type::Any, |ty| ty.inner);

        // eval value
        let value_spanned = Spanned {
            span: full_span,
            inner: value,
        };
        let result = eval_value(value_spanned, expected_ty_eval)?;

        // check type
        if let Some(expected_ty) = expected_ty {
            let reason = TypeContainsReason::Assignment {
                span_target: full_span,
                span_target_ty: expected_ty.span,
            };
            let result_spanned = Spanned {
                span: full_span,
                inner: &result,
            };
            check_type_contains_value(diags, reason, expected_ty.inner, result_spanned, true, true)?;
        }

        Ok(result)
    }
}

fn check_range_compile_index(index: &BigInt, array_len: usize) -> Result<usize, ErrorGuaranteed> {
    if index.is_negative() || index >= &BigInt::from(array_len) {
        todo!("error")
    }
    Ok(index.to_usize().unwrap())
}

fn check_range_hardware_index(index: &ClosedIncRange<BigInt>, array_len: &BigUint) -> Result<(), ErrorGuaranteed> {
    if index.start_inc.is_negative() || index.end_inc >= BigInt::from(array_len.clone()) {
        todo!("error")
    }
    Ok(())
}

fn check_range_mixed_index(index: &BigInt, array_len: &BigUint) -> Result<(), ErrorGuaranteed> {
    if index.is_negative() || index >= &BigInt::from(array_len.clone()) {
        todo!("error")
    }
    Ok(())
}

fn check_range_compile_slice(
    start: &BigInt,
    slice_len: &BigUint,
    array_len: usize,
) -> Result<(usize, usize), ErrorGuaranteed> {
    if start.is_negative() || (start + BigInt::from(slice_len.clone())) > BigInt::from(array_len) {
        todo!("error")
    }
    Ok((start.to_usize().unwrap(), slice_len.to_usize().unwrap()))
}

fn check_range_hardware_slice(
    start: &ClosedIncRange<BigInt>,
    slice_len: &BigUint,
    array_len: &BigUint,
) -> Result<(), ErrorGuaranteed> {
    if start.start_inc.is_negative()
        || (&start.end_inc + BigInt::from(slice_len.clone())) > BigInt::from(array_len.clone())
    {
        todo!("error")
    }
    Ok(())
}

fn check_range_compile_slice_start(start: &BigInt, array_len: usize) -> Result<usize, ErrorGuaranteed> {
    if start.is_negative() || start > &BigInt::from(array_len) {
        todo!("error")
    }
    Ok(start.to_usize().unwrap())
}

struct SliceLength(BigUint);

fn check_range_mixed_slice_start(start: &BigInt, array_len: &BigUint) -> Result<SliceLength, ErrorGuaranteed> {
    if start.is_negative() || start > &BigInt::from(array_len.clone()) {
        todo!("error")
    }
    Ok(SliceLength(array_len - start.to_biguint().unwrap()))
}

fn check_range_hardware_slice_start(start: &BigInt, array_len: &BigUint) -> Result<(), ErrorGuaranteed> {
    if start.is_negative() || start > &BigInt::from(array_len.clone()) {
        todo!("error")
    }
    Ok(())
}

// fn check_array_index(
//     diags: &Diagnostics,
//     array_span: Span,
//     array_len: BigUint,
//     index: Spanned<CompileValue>,
// ) -> Result<CheckedArrayIndex, ErrorGuaranteed> {
//     match index.inner {
//         CompileValue::Int(index_inner) => {
//             let valid_range_index = BigInt::ZERO..BigInt::from(array_len);
//             if !valid_range_index.contains(&index_inner) {
//                 return Err(diags.report_simple(
//                     "array index out of bounds",
//                     index.span,
//                     format!("got index `{index_inner}`, valid index range for this array is `{valid_range_index:?}`"),
//                 ));
//             }
//             let index = index_inner.to_biguint().unwrap();
//
//             Ok(CheckedArrayIndex::Single { index })
//         }
//         CompileValue::IntRange(range) => {
//             // because we're using inclusive ranges (they are more convenient for math),
//             //   the slicing ranges become a bit weird
//             let valid_range_start_inc = BigInt::ZERO..=BigInt::from(array_len.clone());
//             let valid_range_end_inc = BigInt::from(-1i32)..BigInt::from(array_len.clone());
//
//             let IncRange { start_inc, end_inc } = &range;
//
//             let start_valid = start_inc
//                 .as_ref()
//                 .map_or(true, |start_inc| valid_range_start_inc.contains(start_inc));
//             let end_valid = end_inc
//                 .as_ref()
//                 .map_or(true, |end_inc| valid_range_end_inc.contains(end_inc));
//             if !start_valid || !end_valid {
//                 let invalid = match (start_valid, end_valid) {
//                     (false, false) => "start and end",
//                     (false, true) => "start",
//                     (true, false) => "end",
//                     (true, true) => unreachable!(),
//                 };
//
//                 let diag = Diagnostic::new(format!("array slice range {invalid} out of bounds"))
//                     .add_error(index.span, format!("got slice range `{range}`"))
//                     .add_info(array_span, format!("for this array the valid start range is `{valid_range_start_inc:?}` and end range is `{valid_range_end_inc:?}`"))
//                     .finish();
//                 return Err(diags.report(diag));
//             }
//
//             let start_inc = match &range.start_inc {
//                 None => BigUint::ZERO,
//                 Some(start_inc) => start_inc.to_biguint().unwrap(),
//             };
//             let end_ex = match &range.end_inc {
//                 None => array_len.clone(),
//                 Some(end_inc) => (end_inc + 1u32).to_biguint().unwrap(),
//             };
//
//             if start_inc > end_ex {
//                 return Err(diags.report_internal_error(
//                     index.span,
//                     format!("invalid decreasing range `{range:?}` turned into `{start_inc}..{end_ex}` "),
//                 ));
//             }
//
//             let len = end_ex - &start_inc;
//             Ok(CheckedArrayIndex::Slice { start: start_inc, len })
//         }
//         _ => Err(diags.report_simple(
//             "array index must be an integer or an integer range",
//             index.span,
//             format!(
//                 "got value `{}` with type `{}`",
//                 index.inner.to_diagnostic_string(),
//                 index.inner.ty().to_diagnostic_string()
//             ),
//         )),
//     }
// }
