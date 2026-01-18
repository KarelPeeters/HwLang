use crate::front::compile::CompileRefs;
use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::ValueDomain;
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{
    CompileValue, HardwareInt, HardwareValue, MaybeCompile, NotCompile, SimpleCompileValue, Value, ValueCommon,
};
use crate::mid::ir::{IrExpression, IrExpressionLarge, IrLargeArena, IrTargetStep};
use crate::syntax::pos::Span;
use crate::syntax::pos::Spanned;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::iter::IterExt;
use crate::util::range::ClosedNonEmptyRange;
use crate::util::range_multi::{AnyMultiRange, ClosedNonEmptyMultiRange};
use itertools::{Itertools, zip_eq};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ArraySteps<S = ArrayStep> {
    pub steps: Vec<Spanned<S>>,
}

pub type ArrayStep = MaybeCompile<ArrayStepCompile, ArrayStepHardware>;

#[derive(Debug, Clone)]
pub enum ArrayStepCompile {
    ArrayIndex(BigInt),
    ArraySlice { start: BigInt, length: Option<BigUint> },
}

#[derive(Debug, Clone)]
pub enum ArrayStepHardware {
    ArrayIndex(HardwareInt),
    ArraySlice { start: HardwareInt, length: BigUint },
}

impl<S> ArraySteps<S> {
    pub fn new(steps: Vec<Spanned<S>>) -> Self {
        Self { steps }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[derive(Debug, Copy, Clone)]
struct EncounteredAny;

impl ArraySteps<ArrayStep> {
    pub fn try_as_compile(&self) -> Result<ArraySteps<&ArrayStepCompile>, NotCompile> {
        let steps = self
            .steps
            .iter()
            .map(|s| match &s.inner {
                MaybeCompile::Compile(s_compile) => Ok(Spanned::new(s.span, s_compile)),
                MaybeCompile::Hardware(_) => Err(NotCompile),
            })
            .try_collect_vec()?;
        Ok(ArraySteps { steps })
    }

    pub fn apply_to_expected_type(&self, refs: CompileRefs, ty: Spanned<Type>) -> DiagResult<Type> {
        let (ty, _) = self.apply_to_type_impl(refs, ty, true)?;
        Ok(ty)
    }

    pub fn apply_to_hardware_type(
        &self,
        refs: CompileRefs,
        ty: Spanned<&HardwareType>,
    ) -> DiagResult<(HardwareType, Vec<IrTargetStep>)> {
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        let (result_ty, steps) = self.apply_to_type_impl(refs, ty.map_inner(HardwareType::as_type), false)?;
        let steps = steps.expect("any is not a hardware type, so we cannot have encountered it");

        let result_ty_hw = result_ty.as_hardware_type(elab).map_err(|_| {
            diags.report_error_internal(
                ty.span,
                "applying access steps to hardware type should result in hardware type again",
            )
        })?;

        Ok((result_ty_hw, steps))
    }

    pub fn for_each_domain(&self, mut f: impl FnMut(Spanned<ValueDomain>)) {
        for step in &self.steps {
            let &d = match &step.inner {
                ArrayStep::Compile(_) => &ValueDomain::CompileTime,
                ArrayStep::Hardware(step) => match step {
                    ArrayStepHardware::ArrayIndex(index) => &index.domain,
                    ArrayStepHardware::ArraySlice { start, length: _ } => &start.domain,
                },
            };
            f(Spanned::new(step.span, d));
        }
    }

    /// Return the result type after applying the steps to the given type.
    /// This can be used to get the type of step expressions but also the target type for an assignment with steps.
    ///
    /// This function also checks index and slice ranges for validity,
    /// reporting proper diagnostics if they are out of bounds.
    ///
    /// If `pass_any` is true, applying any step to `Type::Any` will return `Type::Any` and `EncounteredAny`. This is
    /// useful to get the inferred type for assignments, once we encounter Any the result type is also Any.
    fn apply_to_type_impl(
        &self,
        refs: CompileRefs,
        ty: Spanned<Type>,
        pass_any: bool,
    ) -> DiagResult<(Type, Result<Vec<IrTargetStep>, EncounteredAny>)> {
        let diags = refs.diags;
        let ArraySteps { steps } = self;

        let mut steps_ir = vec![];
        let mut curr_ty = ty;
        for step in steps {
            // for now we only have arrays steps, so we can always unwrap an array type
            let (ty_array_inner, ty_array_len) = match &curr_ty.inner {
                Type::Array(ty_inner, len) => (&**ty_inner, len),
                Type::Any if pass_any => return Ok((Type::Any, Err(EncounteredAny))),
                _ => return Err(err_expected_array(refs, curr_ty.as_ref(), step.span).report(diags)),
            };
            let ty_array_len = Spanned::new(curr_ty.span, ty_array_len);

            let (step_ir, slice_len) = match &step.inner {
                ArrayStep::Compile(step_inner) => match step_inner {
                    ArrayStepCompile::ArrayIndex(index) => {
                        check_range_index(
                            diags,
                            Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(index.clone())),
                            ty_array_len,
                        )?;
                        let step_ir = IrTargetStep::ArrayIndex(IrExpression::Int(index.clone()));
                        (step_ir, None)
                    }
                    ArrayStepCompile::ArraySlice { start, length: len } => {
                        let len = check_range_slice(
                            diags,
                            Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(start.clone())),
                            len.as_ref().map(|len| Spanned::new(step.span, len)),
                            ty_array_len,
                        )?;
                        let step_ir = IrTargetStep::ArraySlice {
                            start: IrExpression::Int(start.clone()),
                            len: len.clone(),
                        };
                        (step_ir, Some(len))
                    }
                },
                ArrayStep::Hardware(step_inner) => match step_inner {
                    ArrayStepHardware::ArrayIndex(index) => {
                        check_range_index(diags, Spanned::new(step.span, &index.ty), ty_array_len)?;
                        let step_ir = IrTargetStep::ArrayIndex(index.expr.clone());
                        (step_ir, None)
                    }
                    ArrayStepHardware::ArraySlice { start, length: len } => {
                        let len = check_range_slice(
                            diags,
                            Spanned::new(step.span, &start.ty),
                            Some(Spanned::new(step.span, len)),
                            ty_array_len,
                        )?;
                        let step_ir = IrTargetStep::ArraySlice {
                            start: start.expr.clone(),
                            len: len.clone(),
                        };
                        (step_ir, Some(len))
                    }
                },
            };

            steps_ir.push(step_ir);

            let next_ty = match slice_len {
                None => ty_array_inner.clone(),
                Some(slice_len) => Type::Array(Arc::new(ty_array_inner.clone()), slice_len),
            };
            curr_ty = Spanned::new(curr_ty.span.join(step.span), next_ty);
        }

        Ok((curr_ty.inner, Ok(steps_ir)))
    }

    pub fn apply_to_value(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        value: Spanned<Value>,
    ) -> DiagResult<Value> {
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        let ArraySteps { steps } = self;

        let mut curr = value;

        for step in steps {
            let next_inner: Value = match (&step.inner, curr.inner) {
                (ArrayStep::Compile(step_inner), Value::Simple(curr_inner)) => match curr_inner {
                    SimpleCompileValue::Array(curr_inner) => {
                        // index/slice into array
                        let value_len = Spanned::new(curr.span, curr_inner.len());
                        match step_inner {
                            ArrayStepCompile::ArrayIndex(index) => {
                                let index =
                                    check_range_index_compile(diags, Spanned::new(step.span, index), value_len)?;

                                let result = match Arc::try_unwrap(curr_inner) {
                                    Ok(mut curr_inner) => curr_inner.swap_remove(index),
                                    Err(curr_inner) => curr_inner[index].clone(),
                                };

                                Value::from(result)
                            }
                            ArrayStepCompile::ArraySlice { start, length: len } => {
                                let SliceInfo { start, len } = check_range_slice_compile(
                                    diags,
                                    Spanned::new(step.span, start),
                                    len.as_ref().map(|len| Spanned::new(step.span, len)),
                                    value_len,
                                )?;

                                let result = match Arc::try_unwrap(curr_inner) {
                                    Ok(mut curr_inner) => {
                                        // TODO maybe re-use the existing allocation instead by draining before/after?
                                        curr_inner.drain(start..start + len).collect()
                                    }
                                    Err(curr_inner) => curr_inner[start..start + len].to_vec(),
                                };

                                Value::Simple(SimpleCompileValue::Array(Arc::new(result)))
                            }
                        }
                    }
                    _ => {
                        return Err(
                            err_expected_array(refs, Spanned::new(curr.span, &curr_inner.ty()), step.span)
                                .report(diags),
                        );
                    }
                },
                (step_inner, curr_inner) => {
                    // convert curr to hardware
                    let ty = curr_inner.ty();
                    let ty = ty.as_hardware_type(elab).map_err(|_| {
                        DiagnosticError::new(
                            "hardware array indexing target needs to have hardware type",
                            curr.span,
                            format!(
                                "this array target needs to have a hardware type, has type `{}`",
                                ty.value_string(elab)
                            ),
                        )
                        .add_info(step.span, "for this hardware array access operation")
                        .report(diags)
                    })?;
                    let curr_inner = curr_inner.as_hardware_value_unchecked(refs, large, curr.span, ty.clone())?;
                    let (curr_array_inner_ty, curr_array_len) = match curr_inner.ty {
                        HardwareType::Array(curr_array_inner_ty, curr_array_len) => {
                            (curr_array_inner_ty, curr_array_len)
                        }
                        _ => {
                            return Err(
                                err_expected_array(refs, Spanned::new(curr.span, &ty.as_type()), step.span)
                                    .report(diags),
                            );
                        }
                    };
                    let curr_array_len = Spanned::new(curr.span, curr_array_len);

                    // convert step to hardware
                    let (result_expr, step_domain, slice_len) = match step_inner {
                        ArrayStep::Compile(ArrayStepCompile::ArrayIndex(index)) => {
                            check_range_index(
                                diags,
                                Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(index.clone())),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpressionLarge::ArrayIndex {
                                    base: curr_inner.expr,
                                    index: IrExpression::Int(index.clone()),
                                },
                                ValueDomain::CompileTime,
                                None,
                            )
                        }
                        ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, length: len }) => {
                            let len = check_range_slice(
                                diags,
                                Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(start.clone())),
                                len.as_ref().map(|len| Spanned::new(step.span, len)),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpressionLarge::ArraySlice {
                                    base: curr_inner.expr,
                                    start: IrExpression::Int(start.clone()),
                                    len: len.clone(),
                                },
                                ValueDomain::CompileTime,
                                Some(len),
                            )
                        }
                        ArrayStep::Hardware(ArrayStepHardware::ArrayIndex(index)) => {
                            check_range_index(diags, Spanned::new(step.span, &index.ty), curr_array_len.as_ref())?;
                            (
                                IrExpressionLarge::ArrayIndex {
                                    base: curr_inner.expr,
                                    index: index.expr.clone(),
                                },
                                index.domain,
                                None,
                            )
                        }
                        ArrayStep::Hardware(ArrayStepHardware::ArraySlice { start, length: len }) => {
                            let len = check_range_slice(
                                diags,
                                Spanned::new(step.span, &start.ty),
                                Some(Spanned::new(step.span, len)),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpressionLarge::ArraySlice {
                                    base: curr_inner.expr,
                                    start: start.expr.clone(),
                                    len: len.clone(),
                                },
                                start.domain,
                                Some(len),
                            )
                        }
                    };

                    // build final value
                    let next_ty = match slice_len {
                        None => Arc::unwrap_or_clone(curr_array_inner_ty),
                        Some(slice_len) => HardwareType::Array(curr_array_inner_ty.clone(), slice_len),
                    };
                    Value::Hardware(HardwareValue {
                        ty: next_ty,
                        domain: curr_inner.domain.join(step_domain),
                        expr: large.push_expr(result_expr),
                    })
                }
            };

            curr = Spanned::new(curr.span.join(step.span), next_inner);
        }

        Ok(curr.inner)
    }
}

impl ArraySteps<&ArrayStepCompile> {
    /// Evaluate the operation `target[steps] = value`, where all operands are compile-time constants.
    /// This does not mutate `target` in-place, it returns the new value after assignment.
    pub fn set_compile_value(
        &self,
        refs: CompileRefs,
        target: Spanned<CompileValue>,
        assign_op_span: Span,
        value: Spanned<CompileValue>,
    ) -> DiagResult<CompileValue> {
        let target = target.map_inner(SetCompileTarget::Scalar);
        set_compile_value_impl(refs, target, &self.steps, assign_op_span, value)
    }

    pub fn get_compile_value(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        value: Spanned<CompileValue>,
    ) -> DiagResult<CompileValue> {
        let diags = refs.diags;

        // TODO avoid clones
        let self_mapped = ArraySteps {
            steps: self
                .steps
                .iter()
                .map(|s| s.map_inner(|s| ArrayStep::Compile(s.clone())))
                .collect_vec(),
        };
        let value_span = value.span;

        let result = self_mapped.apply_to_value(refs, large, value.map_inner(Value::from))?;

        CompileValue::try_from(&result).map_err(|_: NotCompile| {
            diags.report_error_internal(value_span, "applying compile-time steps to compile-time value should result in compile-time value again, got hardware")
        })
    }
}

enum SetCompileTarget {
    Scalar(CompileValue),
    Slice {
        array: Vec<CompileValue>,
        start: usize,
        len: usize,
    },
}

fn set_compile_value_impl(
    refs: CompileRefs,
    target: Spanned<SetCompileTarget>,
    steps: &[Spanned<&ArrayStepCompile>],
    assign_op_span: Span,
    value: Spanned<CompileValue>,
) -> DiagResult<CompileValue> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    // if done, actually assign the value and return
    let (step, rest) = match steps.split_first() {
        None => {
            let result = match target.inner {
                SetCompileTarget::Scalar(target) => {
                    // scalar assignment target, just replace the entire value
                    let _ = target;
                    value.inner
                }
                SetCompileTarget::Slice {
                    array: mut target_array,
                    start: target_start,
                    len: target_len,
                } => {
                    // slice assignment target, assign all elements in the selected subrange
                    match value.inner {
                        CompileValue::Simple(SimpleCompileValue::Array(value_inner)) => {
                            if value_inner.len() != target_len {
                                return Err(DiagnosticError::new(
                                    "slice assignment length mismatch",
                                    assign_op_span,
                                    "length mismatch on this assignment",
                                )
                                .add_info(target.span, format!("target slice has length `{target_len}`"))
                                .add_info(value.span, format!("source array has length `{}`", value_inner.len()))
                                .report(diags));
                            }
                            let target_slice = &mut target_array[target_start..target_start + target_len];

                            match Arc::try_unwrap(value_inner) {
                                Ok(value_inner) => {
                                    for (t, v) in zip_eq(target_slice, value_inner) {
                                        *t = v;
                                    }
                                }
                                Err(value_inner) => {
                                    for (t, v) in zip_eq(target_slice, value_inner.as_ref()) {
                                        *t = v.clone();
                                    }
                                }
                            }

                            CompileValue::Simple(SimpleCompileValue::Array(Arc::new(target_array)))
                        }
                        _ => {
                            return Err(DiagnosticError::new(
                                "expected array value for slice assignment",
                                assign_op_span,
                                "value assigned to slice here",
                            )
                            .add_info(
                                value.span,
                                format!(
                                    "non-array value with type `{}` here",
                                    value.inner.ty().value_string(elab)
                                ),
                            )
                            .add_info(target.span, "target is a slice assignment")
                            .report(diags));
                        }
                    }
                }
            };
            return Ok(result);
        }
        Some(pair) => pair,
    };

    // check that the current target is an array
    // (for now all steps are array steps, so we can unwrap here)
    let (mut target_array, target_start, target_len) = match target.inner {
        SetCompileTarget::Scalar(target_inner) => match target_inner {
            CompileValue::Simple(SimpleCompileValue::Array(target_inner)) => {
                let len = target_inner.len();
                (Arc::unwrap_or_clone(target_inner), 0, len)
            }
            _ => {
                return Err(DiagnosticError::new(
                    "expected array value for array access",
                    step.span,
                    "this array access needs an array",
                )
                .add_info(
                    target.span,
                    format!(
                        "non-array value with type `{}` here",
                        target_inner.ty().value_string(elab)
                    ),
                )
                .report(diags));
            }
        },
        SetCompileTarget::Slice { array, start, len } => (array, start, len),
    };
    let target_len = Spanned::new(target.span, target_len);

    let new_span = target.span.join(step.span);

    match &step.inner {
        ArrayStepCompile::ArrayIndex(index) => {
            let index = check_range_index_compile(diags, Spanned::new(step.span, index), target_len)?;

            let new_target = SetCompileTarget::Scalar(target_array[target_start + index].clone());
            let new_target = Spanned::new(new_span, new_target);
            target_array[index] = set_compile_value_impl(refs, new_target, rest, assign_op_span, value)?;

            Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(target_array))))
        }
        ArrayStepCompile::ArraySlice {
            start: slice_start,
            length: slice_len,
        } => {
            let SliceInfo {
                start: slice_start,
                len: slice_len,
            } = check_range_slice_compile(
                diags,
                Spanned::new(step.span, slice_start),
                slice_len.as_ref().map(|len| Spanned::new(step.span, len)),
                target_len,
            )?;

            let new_target = SetCompileTarget::Slice {
                array: target_array,
                start: target_start + slice_start,
                len: slice_len,
            };
            let new_target = Spanned::new(new_span, new_target);

            let new_value = set_compile_value_impl(refs, new_target, rest, assign_op_span, value)?;
            Ok(new_value)
        }
    }
}

pub fn check_range_index(
    diags: &Diagnostics,
    index: Spanned<&ClosedNonEmptyMultiRange<BigInt>>,
    array_len: Spanned<&BigUint>,
) -> DiagResult {
    let ClosedNonEmptyRange {
        start: index_start,
        end: index_end,
    } = index.inner.enclosing_range();

    if &BigInt::ZERO <= index_start && index_end <= &BigInt::from(array_len.inner.clone()) {
        Ok(())
    } else {
        let index_str = if let Some(index) = index.inner.as_single() {
            format!("`{index}`")
        } else {
            format!("with range `{}`", index.inner)
        };

        Err(DiagnosticError::new(
            "array index out of bounds",
            index.span,
            format!("index {index_str} is out of bounds"),
        )
        .add_info(
            array_len.span,
            format!("for this array with length `{}`", array_len.inner),
        )
        .report(diags))
    }
}

pub fn check_range_index_compile(
    diags: &Diagnostics,
    index: Spanned<&BigInt>,
    array_len: Spanned<usize>,
) -> DiagResult<usize> {
    let index_range = index.cloned().map_inner(ClosedNonEmptyMultiRange::single);
    check_range_index(diags, index_range.as_ref(), array_len.map_inner(BigUint::from).as_ref())?;
    Ok(usize::try_from(index.inner).unwrap())
}

pub fn check_range_slice(
    diags: &Diagnostics,
    slice_start: Spanned<&ClosedNonEmptyMultiRange<BigInt>>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<&BigUint>,
) -> DiagResult<BigUint> {
    if slice_start.inner.as_single().is_none() && slice_len.is_none() {
        return Err(diags.report_error_internal(
            slice_start.span,
            "start with non-single range and no length doesn't make sense",
        ));
    }

    let slice_str = || {
        let start = if let Some(slice_start_single) = slice_start.inner.as_single() {
            format!("{slice_start_single}")
        } else {
            format!("({})", slice_start.inner)
        };

        match slice_len {
            None => format!("{start}.."),
            Some(slice_len) => format!("{}..+{}", start, slice_len.inner),
        }
    };

    // check start
    // TODO check start and end at once and report a single unified diagnostic
    let ClosedNonEmptyRange {
        start: slice_start_start,
        end: slice_start_end,
    } = slice_start.inner.enclosing_range();

    if !(&BigInt::ZERO <= slice_start_start && slice_start_end - 1 <= BigInt::from(array_len.inner.clone())) {
        return Err(DiagnosticError::new(
            "array slice start out of bounds",
            slice_start.span,
            format!("slice start of `{}` is out out of bounds", slice_str()),
        )
        .add_info(
            array_len.span,
            format!("for this array with length `{}`", array_len.inner),
        )
        .report(diags));
    }

    // check start + len if length is provided
    // if len is not provided, knowing that the start is valid is enough
    if let Some(slice_len) = slice_len {
        let slice_end_max = slice_start_end + BigInt::from(slice_len.inner.clone());

        #[allow(clippy::nonminimal_bool)]
        if !(slice_end_max - 1 <= BigInt::from(array_len.inner.clone())) {
            return Err(DiagnosticError::new(
                "array slice end out of bounds",
                slice_len.span,
                format!("slice end of `{}` is out out of bounds", slice_str()),
            )
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .report(diags));
        }

        Ok(slice_len.inner.clone())
    } else {
        let slice_start = BigUint::try_from(slice_start.inner.as_single().unwrap()).unwrap();
        Ok(BigUint::try_from(array_len.inner - slice_start).unwrap())
    }
}

pub struct SliceInfo {
    pub start: usize,
    pub len: usize,
}

pub fn check_range_slice_compile(
    diags: &Diagnostics,
    slice_start: Spanned<&BigInt>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<usize>,
) -> DiagResult<SliceInfo> {
    let start_range = slice_start.cloned().map_inner(ClosedNonEmptyMultiRange::single);
    check_range_slice(
        diags,
        start_range.as_ref(),
        slice_len,
        array_len.map_inner(BigUint::from).as_ref(),
    )?;
    let start = usize::try_from(slice_start.inner).unwrap();
    let len = slice_len
        .map(|len| usize::try_from(len.inner).unwrap())
        .unwrap_or(array_len.inner - start);
    Ok(SliceInfo { start, len })
}

pub fn err_expected_array(refs: CompileRefs, ty: Spanned<&Type>, step_span: Span) -> DiagnosticError {
    DiagnosticError::new(
        "array indexing on non-array type",
        step_span,
        "array indexing operation here",
    )
    .add_info(
        ty.span,
        format!(
            "on non-array value with type `{}` here",
            ty.inner.value_string(&refs.shared.elaboration_arenas)
        ),
    )
}
