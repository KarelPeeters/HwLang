use crate::front::compile::CompileRefs;
use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::ValueDomain;
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, HardwareInt, HardwareValue, MaybeCompile, NotCompile, SimpleCompileValue,
    Value, ValueCommon,
};
use crate::mid::ir::{IrExpression, IrExpressionLarge, IrLargeArena, IrTargetStep};
use crate::syntax::pos::{Span, Spanned};
use crate::util::big_int::{BigInt, BigUint};
use crate::util::data::VecExt;
use crate::util::iter::IterExt;
use crate::util::range::ClosedNonEmptyRange;
use crate::util::range_multi::{AnyMultiRange, ClosedNonEmptyMultiRange};
use itertools::{Either, Itertools, zip_eq};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AssignmentSteps<S = AssignmentStep> {
    pub steps: Vec<Spanned<S>>,
}

pub type AssignmentStep = MaybeCompile<AssignmentStepCompile, AssignmentStepHardware>;

#[derive(Debug, Clone)]
pub enum AssignmentStepCompile {
    ArrayIndex(BigInt),
    ArraySlice { start: BigInt, length: Option<BigUint> },
    TupleIndex(BigUint),
    StructField(Arc<String>),
}

#[derive(Debug, Clone)]
pub enum AssignmentStepHardware {
    ArrayIndex(HardwareInt),
    ArraySlice { start: HardwareInt, length: BigUint },
}

impl<S> AssignmentSteps<S> {
    pub fn new(steps: Vec<Spanned<S>>) -> Self {
        Self { steps }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[derive(Debug, Copy, Clone)]
struct EncounteredAny;

#[derive(Debug, Copy, Clone)]
struct ArrayUnknownLength;

impl AssignmentSteps<AssignmentStep> {
    pub fn try_as_compile(&self) -> Result<AssignmentSteps<&AssignmentStepCompile>, NotCompile> {
        let steps = self
            .steps
            .iter()
            .map(|s| match &s.inner {
                MaybeCompile::Compile(s_compile) => Ok(Spanned::new(s.span, s_compile)),
                MaybeCompile::Hardware(_) => Err(NotCompile),
            })
            .try_collect_vec()?;
        Ok(AssignmentSteps { steps })
    }

    pub fn apply_to_expected_type(&self, refs: CompileRefs, ty: Spanned<Type>) -> DiagResult<Type> {
        let (ty, _) = self.apply_to_type_impl(refs, ty)?;
        Ok(ty)
    }

    pub fn apply_to_hardware_type(
        &self,
        refs: CompileRefs,
        ty: Spanned<&HardwareType>,
    ) -> DiagResult<(HardwareType, Vec<IrTargetStep>)> {
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        let (result_ty, steps) = self.apply_to_type_impl(refs, ty.map_inner(HardwareType::as_type))?;

        let steps = steps.map_err(|e: Either<EncounteredAny, ArrayUnknownLength>| {
            diags.report_error_internal(ty.span, format!("applying steps to hardware type failed: {e:?}"))
        })?;

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
                AssignmentStep::Compile(_) => &ValueDomain::CompileTime,
                AssignmentStep::Hardware(step) => match step {
                    AssignmentStepHardware::ArrayIndex(index) => &index.domain,
                    AssignmentStepHardware::ArraySlice { start, length: _ } => &start.domain,
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
    ///
    /// If `pass_any` is true, applying any step to `Type::Any` will return `Type::Any` and `EncounteredAny`. This is
    /// useful to get the inferred type for assignments, once we encounter Any the result type is also Any.
    fn apply_to_type_impl(
        &self,
        refs: CompileRefs,
        ty: Spanned<Type>,
    ) -> DiagResult<(
        Type,
        Result<Vec<IrTargetStep>, Either<EncounteredAny, ArrayUnknownLength>>,
    )> {
        let diags = refs.diags;
        let AssignmentSteps { steps } = self;

        let mut steps_ir = Ok(vec![]);
        let mut curr_ty = ty;

        for step in steps {
            // TODO rework this
            let (array_inner, array_len) = match &curr_ty.inner {
                Type::Array(ty_inner, len) => (&**ty_inner, len),
                Type::Any => return Ok((Type::Any, Err(Either::Left(EncounteredAny)))),
                _ => return Err(err_expected_array(refs, curr_ty.as_ref(), step.span, todo!())),
            };
            let array_len = Spanned::new(curr_ty.span, array_len.as_ref());

            enum StepKind {
                Single,
                Slice(Option<BigUint>),
            }

            let (step_ir, step_kind) = match &step.inner {
                AssignmentStep::Compile(step_inner) => match step_inner {
                    AssignmentStepCompile::ArrayIndex(index) => {
                        check_range_index(
                            diags,
                            Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(index.clone())),
                            array_len,
                        )?;

                        match array_len.inner {
                            None => (Err(ArrayUnknownLength), StepKind::Single),
                            Some(_) => {
                                let step_ir = IrTargetStep::ArrayIndex(IrExpression::Int(index.clone()));
                                (Ok(step_ir), StepKind::Single)
                            }
                        }
                    }
                    AssignmentStepCompile::ArraySlice {
                        start,
                        length: slice_len,
                    } => {
                        let slice_len = check_range_slice(
                            diags,
                            Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(start.clone())),
                            slice_len.as_ref().map(|len| Spanned::new(step.span, len)),
                            array_len,
                        )?;

                        let step_ir = match &slice_len {
                            Some(slice_len) => {
                                let step_ir = IrTargetStep::ArraySlice {
                                    start: IrExpression::Int(start.clone()),
                                    len: slice_len.clone(),
                                };
                                Ok(step_ir)
                            }
                            None => Err(ArrayUnknownLength),
                        };

                        (step_ir, StepKind::Slice(slice_len))
                    }
                    AssignmentStepCompile::TupleIndex(_) => todo!(),
                    AssignmentStepCompile::StructField(_) => todo!(),
                },
                AssignmentStep::Hardware(step_inner) => match step_inner {
                    AssignmentStepHardware::ArrayIndex(index) => {
                        check_range_index(diags, Spanned::new(step.span, &index.ty), array_len)?;

                        match array_len.inner {
                            None => (Err(ArrayUnknownLength), StepKind::Single),
                            Some(_) => {
                                let step_ir = IrTargetStep::ArrayIndex(index.expr.clone());
                                (Ok(step_ir), StepKind::Single)
                            }
                        }
                    }
                    AssignmentStepHardware::ArraySlice {
                        start,
                        length: slice_len,
                    } => {
                        let slice_len = check_range_slice(
                            diags,
                            Spanned::new(step.span, &start.ty),
                            Some(Spanned::new(step.span, slice_len)),
                            array_len,
                        )?;

                        let step_ir = match &slice_len {
                            Some(slice_len) => {
                                let step_ir = IrTargetStep::ArraySlice {
                                    start: start.expr.clone(),
                                    len: slice_len.clone(),
                                };
                                Ok(step_ir)
                            }
                            None => Err(ArrayUnknownLength),
                        };
                        (step_ir, StepKind::Slice(slice_len))
                    }
                },
            };

            if array_len.inner.is_some() {
                match step_ir {
                    Ok(step_ir) => {
                        if let Ok(steps_ir) = &mut steps_ir {
                            steps_ir.push(step_ir);
                        }
                    }
                    Err(e) => {
                        steps_ir = Err(e);
                    }
                }
            } else {
                steps_ir = Err(ArrayUnknownLength);
            }

            let next_ty = match step_kind {
                StepKind::Single => array_inner.clone(),
                StepKind::Slice(slice_len) => Type::Array(Arc::new(array_inner.clone()), slice_len),
            };

            curr_ty = Spanned::new(curr_ty.span.join(step.span), next_ty);
        }

        let steps_ir = steps_ir.map_err(Either::Right);
        Ok((curr_ty.inner, steps_ir))
    }

    pub fn apply_to_value(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        value: Spanned<Value>,
    ) -> DiagResult<Value> {
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        let AssignmentSteps { steps } = self;

        let mut curr = value;

        for step in steps {
            let next_inner: Value = match (&step.inner, curr.inner) {
                (AssignmentStep::Compile(step_inner), Value::Simple(curr_inner)) => match curr_inner {
                    SimpleCompileValue::Array(curr_inner) => {
                        // index/slice into array
                        let value_len = Spanned::new(curr.span, curr_inner.len());
                        match step_inner {
                            AssignmentStepCompile::ArrayIndex(index) => {
                                let index =
                                    check_range_index_compile(diags, Spanned::new(step.span, index), value_len)?;

                                let result = match Arc::try_unwrap(curr_inner) {
                                    Ok(curr_inner) => curr_inner.get_owned(index),
                                    Err(curr_inner) => curr_inner[index].clone(),
                                };

                                Value::from(result)
                            }
                            AssignmentStepCompile::ArraySlice { start, length: len } => {
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
                            AssignmentStepCompile::TupleIndex(_) => {
                                // let err_not_tuple = |ty: &str| {
                                //     DiagnosticError::new(
                                //         "indexing into non-tuple type",
                                //         index_span,
                                //         "attempt to index into non-tuple type here",
                                //     )
                                //         .add_info(base.span, format!("base has type `{ty}`"))
                                //         .report(diags)
                                // };
                                // let err_index_out_of_bounds = |len: usize| {
                                //     DiagnosticError::new(
                                //         "tuple index out of bounds",
                                //         index_span,
                                //         format!("index `{index}` is out of bounds"),
                                //     )
                                //         .add_info(base.span, format!("base is tuple with length `{len}`"))
                                //         .report(diags)
                                // };
                                //
                                // // TODO use common step logic once that supports tuple indexing
                                // match base.inner {
                                //     Value::Compound(MixedCompoundValue::Tuple(inner)) => {
                                //         let index = index
                                //             .as_usize_if_lt(inner.len())
                                //             .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                                //         inner[index].clone()
                                //     }
                                //     Value::Simple(SimpleCompileValue::Type(Type::Tuple(Some(inner)))) => {
                                //         let index = index
                                //             .as_usize_if_lt(inner.len())
                                //             .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                                //         Value::new_ty(inner[index].clone())
                                //     }
                                //     Value::Hardware(value) => match value.ty {
                                //         HardwareType::Tuple(inner_tys) => {
                                //             let index = index
                                //                 .as_usize_if_lt(inner_tys.len())
                                //                 .ok_or_else(|| err_index_out_of_bounds(inner_tys.len()))?;
                                //
                                //             let expr = IrExpressionLarge::TupleIndex {
                                //                 base: value.expr,
                                //                 index,
                                //             };
                                //             Value::Hardware(HardwareValue {
                                //                 ty: inner_tys[index].clone(),
                                //                 domain: value.domain,
                                //                 expr: self.large.push_expr(expr),
                                //             })
                                //         }
                                //         _ => return Err(err_not_tuple(&value.ty.value_string(elab))),
                                //     },
                                //     v => return Err(err_not_tuple(&v.ty().value_string(elab))),
                                // }
                                todo!()
                            }
                            AssignmentStepCompile::StructField(_) => todo!(),
                        }
                    }
                    _ => {
                        return Err(err_expected_array(
                            refs,
                            Spanned::new(curr.span, &curr_inner.ty()),
                            step.span,
                            todo!(),
                        ));
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
                            return Err(err_expected_array(
                                refs,
                                Spanned::new(curr.span, &ty.as_type()),
                                step.span,
                                todo!(),
                            ));
                        }
                    };
                    let curr_array_len = Spanned::new(curr.span, &curr_array_len);

                    // convert step to hardware
                    let (result_expr, step_domain, slice_len) = match step_inner {
                        AssignmentStep::Compile(AssignmentStepCompile::ArrayIndex(index)) => {
                            check_range_index(
                                diags,
                                Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(index.clone())),
                                curr_array_len.map_inner(Some),
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
                        AssignmentStep::Compile(AssignmentStepCompile::ArraySlice { start, length: len }) => {
                            let len = check_range_slice_known(
                                diags,
                                Spanned::new(step.span, &ClosedNonEmptyMultiRange::single(start.clone())),
                                len.as_ref().map(|len| Spanned::new(step.span, len)),
                                curr_array_len,
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
                        AssignmentStep::Hardware(AssignmentStepHardware::ArrayIndex(index)) => {
                            check_range_index(
                                diags,
                                Spanned::new(step.span, &index.ty),
                                curr_array_len.map_inner(Some),
                            )?;
                            (
                                IrExpressionLarge::ArrayIndex {
                                    base: curr_inner.expr,
                                    index: index.expr.clone(),
                                },
                                index.domain,
                                None,
                            )
                        }
                        AssignmentStep::Hardware(AssignmentStepHardware::ArraySlice { start, length: len }) => {
                            let len = check_range_slice_known(
                                diags,
                                Spanned::new(step.span, &start.ty),
                                Some(Spanned::new(step.span, len)),
                                curr_array_len,
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
                        AssignmentStep::Compile(AssignmentStepCompile::TupleIndex(_)) => todo!(),
                        AssignmentStep::Compile(AssignmentStepCompile::StructField(_)) => todo!(),
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

impl AssignmentSteps<&AssignmentStepCompile> {
    /// Evaluate the operation `target[steps] = value`, where all operands are compile-time constants.
    pub fn set_compile_value(
        &self,
        refs: CompileRefs,
        target: Spanned<&mut CompileValue>,
        assign_op_span: Span,
        source: Spanned<CompileValue>,
    ) -> DiagResult {
        let target_span = target.span;
        let target = Spanned::new(target_span, SetCompileTarget::Scalar(target.inner));
        set_compile_value_impl(refs, target, &self.steps, assign_op_span, source)
    }

    /// Evaluate the expression `value[steps]`
    pub fn get_compile_value(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        value: Spanned<CompileValue>,
    ) -> DiagResult<CompileValue> {
        let diags = refs.diags;

        // TODO avoid clones
        let self_mapped = AssignmentSteps {
            steps: self
                .steps
                .iter()
                .map(|s| s.map_inner(|s| AssignmentStep::Compile(s.clone())))
                .collect_vec(),
        };
        let value_span = value.span;

        let result = self_mapped.apply_to_value(refs, large, value.map_inner(Value::from))?;

        CompileValue::try_from(&result).map_err(|_: NotCompile| {
            diags.report_error_internal(value_span, "applying compile-time steps to compile-time value should result in compile-time value again, got hardware")
        })
    }
}

enum SetCompileTarget<'a> {
    Scalar(&'a mut CompileValue),
    Slice(&'a mut [CompileValue]),
}

fn set_compile_value_impl(
    refs: CompileRefs,
    target: Spanned<SetCompileTarget<'_>>,
    steps: &[Spanned<&AssignmentStepCompile>],
    assign_op_span: Span,
    source: Spanned<CompileValue>,
) -> DiagResult {
    let diags = refs.diags;

    // if done just do the final assignment, otherwise get the next step
    let Some((step, steps)) = steps.split_first() else {
        match target.inner {
            SetCompileTarget::Scalar(target) => {
                // scalar assignment target, just replace the entire value
                *target = source.inner;
            }
            SetCompileTarget::Slice(target_slice) => {
                // slice assignment target, assign all elements in the selected subrange
                let source_span = source.span;
                match source.inner {
                    CompileValue::Simple(SimpleCompileValue::Array(source)) => {
                        if source.len() != target_slice.len() {
                            return Err(DiagnosticError::new(
                                "slice assignment length mismatch",
                                assign_op_span,
                                "length mismatch on this assignment",
                            )
                            .add_info(target.span, format!("target slice has length `{}`", target_slice.len()))
                            .add_info(source_span, format!("source array has length `{}`", source.len()))
                            .report(diags));
                        }

                        match Arc::try_unwrap(source) {
                            Ok(source) => {
                                for (t, v) in zip_eq(target_slice, source) {
                                    *t = v;
                                }
                            }
                            Err(source) => {
                                for (t, v) in zip_eq(target_slice, source.as_ref()) {
                                    *t = v.clone();
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(DiagnosticError::new(
                            "expected array value for slice assignment",
                            assign_op_span,
                            "value assigned to slice here",
                        )
                        .add_info(
                            source.span,
                            format!(
                                "non-array value with type `{}` here",
                                source.inner.ty().value_string(&refs.shared.elaboration_arenas)
                            ),
                        )
                        .add_info(target.span, "target is a slice assignment")
                        .report(diags));
                    }
                }
            }
        }
        return Ok(());
    };

    let target_span = target.span;
    let new_target = match &step.inner {
        AssignmentStepCompile::ArrayIndex(index) => {
            let target_inner = check_target_is_array(refs, target, step.span, false)?;

            let index = check_range_index_compile(
                diags,
                Spanned::new(step.span, index),
                Spanned::new(target_span, target_inner.len()),
            )?;

            SetCompileTarget::Scalar(&mut target_inner[index])
        }
        AssignmentStepCompile::ArraySlice {
            start: slice_start,
            length: slice_len,
        } => {
            let target_inner = check_target_is_array(refs, target, step.span, true)?;

            let SliceInfo {
                start: slice_start,
                len: slice_len,
            } = check_range_slice_compile(
                diags,
                Spanned::new(step.span, slice_start),
                slice_len.as_ref().map(|len| Spanned::new(step.span, len)),
                Spanned::new(target_span, target_inner.len()),
            )?;

            SetCompileTarget::Slice(&mut target_inner[slice_start..][..slice_len])
        }
        AssignmentStepCompile::TupleIndex(index) => {
            // check target is tuple
            let target_inner = match target.inner {
                SetCompileTarget::Scalar(target_inner) => match target_inner {
                    CompileValue::Compound(CompileCompoundValue::Tuple(target_inner)) => target_inner,
                    CompileValue::Hardware(never) => never.unreachable(),
                    _ => todo!("err expected tuple, got {target_inner:?}"),
                },
                SetCompileTarget::Slice(_) => todo!("err expected tuple, got slice"),
            };

            // check index in bounds
            let index = index
                .as_usize_if_lt(target_inner.len())
                .ok_or_else(|| todo!("err index out of bounds"))?;

            // build new target
            SetCompileTarget::Scalar(&mut target_inner[index])
        }
        AssignmentStepCompile::StructField(_) => todo!(),
    };

    let new_target = Spanned::new(target_span.join(step.span), new_target);
    set_compile_value_impl(refs, new_target, steps, assign_op_span, source)
}

fn check_target_is_array<'a>(
    refs: CompileRefs,
    target: Spanned<SetCompileTarget<'a>>,
    step_span: Span,
    op_is_slice: bool,
) -> DiagResult<&'a mut [CompileValue]> {
    match target.inner {
        SetCompileTarget::Scalar(target_inner) => match target_inner {
            CompileValue::Simple(SimpleCompileValue::Array(target_inner)) => {
                Ok(Arc::make_mut(target_inner).as_mut_slice())
            }
            _ => Err(err_expected_array(
                refs,
                Spanned::new(target.span, &target_inner.ty()),
                step_span,
                op_is_slice,
            )),
        },
        SetCompileTarget::Slice(array) => Ok(array),
    }
}

pub fn check_range_index(
    diags: &Diagnostics,
    index: Spanned<&ClosedNonEmptyMultiRange<BigInt>>,
    array_len: Spanned<Option<&BigUint>>,
) -> DiagResult {
    let ClosedNonEmptyRange {
        start: index_start,
        end: index_end,
    } = index.inner.enclosing_range();

    if &BigInt::ZERO <= index_start
        && array_len
            .inner
            .is_none_or(|array_len| index_end <= &BigInt::from(array_len.clone()))
    {
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
        .add_info(array_len.span, info_array_length(array_len.inner))
        .report(diags))
    }
}

pub fn check_range_index_compile(
    diags: &Diagnostics,
    index: Spanned<&BigInt>,
    array_len: Spanned<usize>,
) -> DiagResult<usize> {
    let index_range = index.cloned().map_inner(ClosedNonEmptyMultiRange::single);

    let array_len = array_len.map_inner(BigUint::from);
    let array_len = array_len.as_ref().map_inner(Some);
    check_range_index(diags, index_range.as_ref(), array_len)?;

    Ok(usize::try_from(index.inner).unwrap())
}

pub fn check_range_slice_known(
    diags: &Diagnostics,
    slice_start: Spanned<&ClosedNonEmptyMultiRange<BigInt>>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<&BigUint>,
) -> DiagResult<BigUint> {
    Ok(check_range_slice(diags, slice_start, slice_len, array_len.map_inner(Some))?.unwrap())
}

pub fn check_range_slice(
    diags: &Diagnostics,
    slice_start: Spanned<&ClosedNonEmptyMultiRange<BigInt>>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<Option<&BigUint>>,
) -> DiagResult<Option<BigUint>> {
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

    if !(&BigInt::ZERO <= slice_start_start
        && array_len
            .inner
            .is_none_or(|array_len| slice_start_end - 1 <= BigInt::from(array_len.clone())))
    {
        return Err(DiagnosticError::new(
            "array slice start out of bounds",
            slice_start.span,
            format!("slice start of `{}` is out of bounds", slice_str()),
        )
        .add_info(array_len.span, info_array_length(array_len.inner))
        .report(diags));
    }

    // check start + len if length is provided
    //   if len is not provided, knowing that the start is valid is enough
    if let Some(slice_len) = slice_len {
        // if we don't know the array length, we can't check the slice end yet
        if let Some(array_len_inner) = array_len.inner {
            let slice_end_max = slice_start_end + BigInt::from(slice_len.inner.clone());

            #[allow(clippy::nonminimal_bool)]
            if !(slice_end_max - 1 <= BigInt::from(array_len_inner)) {
                return Err(DiagnosticError::new(
                    "array slice end out of bounds",
                    slice_len.span,
                    format!("slice end of `{}` is out of bounds", slice_str()),
                )
                .add_info(array_len.span, info_array_length(Some(array_len_inner)))
                .report(diags));
            }
            Ok(Some(slice_len.inner.clone()))
        } else {
            Ok(None)
        }
    } else {
        Ok(array_len.inner.map(|array_len| {
            let slice_start = BigUint::try_from(slice_start.inner.as_single().unwrap()).unwrap();
            BigUint::try_from(array_len - slice_start).unwrap()
        }))
    }
}

fn info_array_length(array_len: Option<&BigUint>) -> String {
    match array_len {
        Some(len) => format!("for this array with length `{}`", len),
        None => "for this array".to_string(),
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
    check_range_slice_known(
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

pub fn err_expected_array(refs: CompileRefs, ty: Spanned<&Type>, step_span: Span, op_is_slice: bool) -> DiagError {
    let op_name = if op_is_slice { "slicing" } else { "indexing" };

    DiagnosticError::new(
        format!("array {op_name} on non-array type"),
        step_span,
        format!("array {op_name} operation here"),
    )
    .add_info(
        ty.span,
        format!(
            "on non-array value with type `{}` here",
            ty.inner.value_string(&refs.shared.elaboration_arenas)
        ),
    )
    .report(refs.diags)
}
