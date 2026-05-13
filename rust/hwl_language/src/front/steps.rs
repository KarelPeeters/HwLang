use crate::front::compile::CompileRefs;
use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::ValueDomain;
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, HardwareUInt, HardwareValue, MaybeCompile, MixedCompoundValue, NotCompile,
    SimpleCompileValue, Value, ValueCommon,
};
use crate::mid::builder::{IrTargetStepBuild, IrTargetStepBuildSlice, IrTargetStepsBuilder};
use crate::mid::ir::{IrExpression, IrExpressionLarge, IrLargeArena, IrTargetSteps};
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
    ArrayIndex(BigUint),
    ArraySlice { start: BigUint, length: Option<BigUint> },
    TupleIndex(BigUint),
    StructField(Arc<String>),
}

#[derive(Debug, Clone)]
pub enum AssignmentStepHardware {
    ArrayIndex(HardwareUInt),
    ArraySlice { start: HardwareUInt, length: BigUint },
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
struct EncounteredUnknown;

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
        // we don't care about the resulting steps, so we don't need a real arena
        let mut dummy_large = IrLargeArena::new();
        let (ty, _) = self.apply_to_type_impl(refs, &mut dummy_large, ty)?;
        Ok(ty)
    }

    pub fn apply_to_hardware_type(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        ty: Spanned<&HardwareType>,
    ) -> DiagResult<(HardwareType, IrTargetSteps)> {
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        let (result_ty, steps) = self.apply_to_type_impl(refs, large, ty.map_inner(HardwareType::as_type))?;

        let steps = steps.map_err(|e: Either<EncounteredAny, EncounteredUnknown>| {
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
    /// Applying any step to [Type::Any] will return [Type::Any] and [EncounteredAny]. This is
    /// useful to get the inferred type for assignments, once we encounter Any the result type is also Any.
    fn apply_to_type_impl(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        ty: Spanned<Type>,
    ) -> DiagResult<(Type, Result<IrTargetSteps, Either<EncounteredAny, EncounteredUnknown>>)> {
        let diags = refs.diags;
        let AssignmentSteps { steps } = self;

        let mut steps_builder = Ok(IrTargetStepsBuilder::new());
        let mut curr_ty = ty;

        for step in steps {
            // stop once we encounter Type::Any
            if matches!(curr_ty.inner, Type::Any) {
                return Ok((Type::Any, Err(Either::Left(EncounteredAny))));
            }

            // map step to IR and get the next type
            let step_span = step.span;
            let check_type_is_array = |step_is_slice: bool| {
                let (inner, len) = check_type_is_array(refs, curr_ty.as_ref(), step_span, step_is_slice)?;
                let len = Spanned::new(curr_ty.span, len);
                Ok((inner, len))
            };

            let (step_ir, next_ty) = match &step.inner {
                AssignmentStep::Compile(step) => match step {
                    AssignmentStepCompile::ArrayIndex(index) => {
                        let (array_inner, array_len) = check_type_is_array(false)?;
                        check_range_index(
                            diags,
                            Spanned::new(step_span, &ClosedNonEmptyMultiRange::single(index.clone())),
                            array_len,
                        )?;

                        let step_ir = array_len
                            .inner
                            .map(|_| IrTargetStepBuild::ArrayIndex {
                                index: IrExpression::Int(index.into()),
                                index_range: ClosedNonEmptyRange::single(index.clone()),
                            })
                            .ok_or(EncounteredUnknown);

                        (step_ir, array_inner.as_ref().clone())
                    }
                    AssignmentStepCompile::ArraySlice {
                        start: slice_start,
                        length: slice_length,
                    } => {
                        let (array_inner, array_len) = check_type_is_array(true)?;

                        let slice_len = check_range_slice(
                            diags,
                            Spanned::new(step_span, &ClosedNonEmptyMultiRange::single(slice_start.clone())),
                            slice_length
                                .as_ref()
                                .map(|slice_length| Spanned::new(step_span, slice_length)),
                            array_len,
                        )?;

                        match slice_len {
                            None => (Err(EncounteredUnknown), Type::Array(Arc::clone(array_inner), None)),
                            Some(slice_len) => {
                                let step_ir = IrTargetStepBuild::ArraySlice(IrTargetStepBuildSlice {
                                    start: IrExpression::Int(slice_start.into()),
                                    start_range: ClosedNonEmptyRange::single(slice_start.clone()),
                                    len: slice_len.clone(),
                                });
                                let next_ty = Type::Array(Arc::clone(array_inner), Some(slice_len));
                                (Ok(step_ir), next_ty)
                            }
                        }
                    }
                    AssignmentStepCompile::TupleIndex(index) => {
                        let fields = match &curr_ty.inner {
                            Type::Tuple(fields) => fields,
                            _ => return Err(err_expected_tuple(refs, curr_ty.as_ref(), step_span)),
                        };

                        match fields {
                            None => (Err(EncounteredUnknown), Type::Any),
                            Some(fields) => {
                                let index = check_tuple_index(diags, fields.len(), index, curr_ty.span, step_span)?;
                                (Ok(IrTargetStepBuild::TupleIndex(index)), fields[index].clone())
                            }
                        }
                    }
                    AssignmentStepCompile::StructField(field) => {
                        let ty = match curr_ty.inner {
                            Type::Struct(ty) => ty,
                            _ => return Err(err_expected_struct(refs, curr_ty.as_ref(), step_span)),
                        };
                        let ty_info = refs.shared.elaboration_arenas.struct_info(ty);

                        let field_str = Spanned::new(step_span, field.as_str());
                        let field_index = ty_info.field_index(diags, curr_ty.span, field_str)?;

                        let step_ir = IrTargetStepBuild::StructField(field_index);
                        let field_ty = ty_info.fields[field_index].1.inner.clone();
                        (Ok(step_ir), field_ty)
                    }
                },
                AssignmentStep::Hardware(step) => match step {
                    AssignmentStepHardware::ArrayIndex(index) => {
                        let (array_inner, array_len) = check_type_is_array(false)?;
                        check_range_index(diags, Spanned::new(step_span, &index.ty), array_len)?;

                        let step_ir = array_len
                            .inner
                            .map(|_| IrTargetStepBuild::ArrayIndex {
                                index: index.expr.clone(),
                                index_range: index.ty.enclosing_range().cloned(),
                            })
                            .ok_or(EncounteredUnknown);

                        (step_ir, array_inner.as_ref().clone())
                    }
                    AssignmentStepHardware::ArraySlice {
                        start: slice_start,
                        length: slice_len,
                    } => {
                        let (array_inner, array_len) = check_type_is_array(true)?;
                        check_range_slice(
                            diags,
                            Spanned::new(step_span, &slice_start.ty),
                            Some(Spanned::new(step_span, slice_len)),
                            array_len,
                        )?;

                        let step_ir = IrTargetStepBuildSlice {
                            start: slice_start.expr.clone(),
                            start_range: slice_start.ty.enclosing_range().cloned(),
                            len: slice_len.clone(),
                        };
                        let step_ir = IrTargetStepBuild::ArraySlice(step_ir);

                        let next_ty = Type::Array(Arc::clone(array_inner), Some(slice_len.clone()));

                        (Ok(step_ir), next_ty)
                    }
                },
            };

            // append IR step
            match step_ir {
                Ok(step_ir) => {
                    if let Ok(steps_builder) = &mut steps_builder {
                        steps_builder
                            .push(large, step_ir)
                            .map_err(|_| diags.report_error_internal(step_span, "invalid step sequence"))?;
                    }
                }
                Err(e) => {
                    steps_builder = Err(e);
                }
            }

            // move on to the next type
            curr_ty = Spanned::new(curr_ty.span.join(step.span), next_ty);
        }

        let steps_ir = steps_builder.map(IrTargetStepsBuilder::finish).map_err(Either::Right);
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

        let mut curr_value = value;

        for step in steps {
            let curr_span = curr_value.span;
            let step_span = step.span;

            let next: Value = match &step.inner {
                AssignmentStep::Compile(step) => match step {
                    AssignmentStepCompile::ArrayIndex(index) => {
                        let index_range = ClosedNonEmptyMultiRange::single(index.clone());
                        let index_spanned = Spanned::new(step_span, &index_range);

                        match curr_value.inner {
                            Value::Simple(SimpleCompileValue::Array(curr_value)) => {
                                let index = check_range_index_compile(
                                    diags,
                                    Spanned::new(step_span, index),
                                    Spanned::new(curr_span, curr_value.len()),
                                )?;

                                match Arc::try_unwrap(curr_value) {
                                    Ok(curr) => Value::from(curr.get_owned(index)),
                                    Err(curr) => Value::from(curr[index].clone()),
                                }
                            }
                            Value::Hardware(ref curr_value)
                                if let HardwareType::Array(curr_inner, curr_len) = &curr_value.ty =>
                            {
                                check_range_index(diags, index_spanned, Spanned::new(curr_span, Some(curr_len)))?;

                                let expr = IrExpressionLarge::ArrayIndex {
                                    base: curr_value.expr.clone(),
                                    index: IrExpression::Int(index.into()),
                                };
                                Value::Hardware(HardwareValue {
                                    ty: (**curr_inner).clone(),
                                    domain: curr_value.domain,
                                    expr: large.push_expr(expr),
                                })
                            }
                            _ => {
                                let err = err_expected_array(
                                    refs,
                                    curr_value.as_ref().map_inner(Value::ty).as_ref(),
                                    step_span,
                                    false,
                                );
                                return Err(err);
                            }
                        }
                    }
                    AssignmentStepCompile::ArraySlice {
                        start: slice_start,
                        length: slice_length,
                    } => {
                        let slice_length = slice_length
                            .as_ref()
                            .map(|slice_length| Spanned::new(step_span, slice_length));

                        match curr_value.inner {
                            Value::Simple(SimpleCompileValue::Array(curr_value)) => {
                                let slice_start = Spanned::new(step_span, slice_start);
                                let slice_range = check_range_slice_compile(
                                    diags,
                                    slice_start,
                                    slice_length,
                                    Spanned::new(curr_span, curr_value.len()),
                                )?
                                .as_range();

                                let result = match Arc::try_unwrap(curr_value) {
                                    Ok(curr_value) => curr_value.slice_owned(slice_range),
                                    Err(curr_value) => curr_value[slice_range].to_vec(),
                                };

                                Value::Simple(SimpleCompileValue::Array(Arc::new(result)))
                            }
                            Value::Hardware(ref curr_value)
                                if let HardwareType::Array(array_inner, array_len) = &curr_value.ty =>
                            {
                                let slice_len = check_range_slice_known(
                                    diags,
                                    Spanned::new(step_span, &ClosedNonEmptyMultiRange::single(slice_start.clone())),
                                    slice_length,
                                    Spanned::new(curr_span, array_len),
                                )?;

                                let expr = IrExpressionLarge::ArraySlice {
                                    base: curr_value.expr.clone(),
                                    start: IrExpression::Int(slice_start.into()),
                                    len: slice_len.clone(),
                                };
                                Value::Hardware(HardwareValue {
                                    ty: HardwareType::Array(Arc::clone(array_inner), slice_len),
                                    domain: curr_value.domain,
                                    expr: large.push_expr(expr),
                                })
                            }
                            _ => {
                                let curr_ty = curr_value.as_ref().map_inner(Value::ty);
                                return Err(err_expected_array(refs, curr_ty.as_ref(), step_span, true));
                            }
                        }
                    }
                    AssignmentStepCompile::TupleIndex(index) => match curr_value.inner {
                        Value::Simple(SimpleCompileValue::Type(Type::Tuple(fields))) => {
                            let fields = fields.ok_or_else(|| {
                                DiagnosticError::new(
                                    "cannot index into tuple type with unknown field types",
                                    step_span,
                                    "trying to index here",
                                )
                                .add_info(
                                    curr_span,
                                    format!("target resolved to type `{}`", Type::Tuple(None).value_string(elab)),
                                )
                                .report(diags)
                            })?;
                            let index = check_tuple_index(diags, fields.len(), index, curr_span, step_span)?;

                            let result = match Arc::try_unwrap(fields) {
                                Ok(fields) => fields.get_owned(index),
                                Err(fields) => fields[index].clone(),
                            };
                            Value::new_ty(result)
                        }
                        Value::Compound(MixedCompoundValue::Tuple(fields)) => {
                            let index = check_tuple_index(diags, fields.len(), index, curr_span, step_span)?;
                            fields.get_owned(index)
                        }
                        Value::Hardware(ref curr_value) if let HardwareType::Tuple(fields) = &curr_value.ty => {
                            let index = check_tuple_index(diags, fields.len(), index, curr_span, step_span)?;

                            let expr = IrExpressionLarge::TupleIndex {
                                base: curr_value.expr.clone(),
                                index,
                            };
                            Value::Hardware(HardwareValue {
                                ty: fields[index].clone(),
                                domain: curr_value.domain,
                                expr: large.push_expr(expr),
                            })
                        }
                        _ => {
                            let curr_ty = curr_value.as_ref().map_inner(Value::ty);
                            return Err(err_expected_tuple(refs, curr_ty.as_ref(), step_span));
                        }
                    },
                    AssignmentStepCompile::StructField(field_str) => {
                        let field_str = Spanned::new(step_span, field_str.as_str());

                        match curr_value.inner {
                            Value::Compound(MixedCompoundValue::Struct(curr_value)) => {
                                let ty_info = elab.struct_info(curr_value.ty);
                                let field_index = ty_info.field_index(diags, curr_span, field_str)?;

                                curr_value.fields.get_owned(field_index)
                            }
                            Value::Hardware(ref curr_value) if let HardwareType::Struct(curr_ty) = &curr_value.ty => {
                                let ty_info = elab.struct_info(curr_ty.inner());
                                let field_index = ty_info.field_index(diags, curr_span, field_str)?;

                                let field_type = ty_info.hw.as_ref().unwrap().fields[field_index].clone();

                                let expr = IrExpressionLarge::StructField {
                                    base: curr_value.expr.clone(),
                                    field: field_index,
                                };
                                Value::Hardware(HardwareValue {
                                    ty: field_type,
                                    domain: curr_value.domain,
                                    expr: large.push_expr(expr),
                                })
                            }
                            _ => {
                                let curr_ty = curr_value.as_ref().map_inner(Value::ty);
                                return Err(err_expected_struct(refs, curr_ty.as_ref(), step_span));
                            }
                        }
                    }
                },
                AssignmentStep::Hardware(step) => {
                    // check type
                    let curr_ty = curr_value.inner.ty();
                    let step_is_slice = match step {
                        AssignmentStepHardware::ArrayIndex(_) => false,
                        AssignmentStepHardware::ArraySlice { .. } => true,
                    };
                    check_type_is_array(refs, Spanned::new(curr_value.span, &curr_ty), step_span, step_is_slice)?;

                    // convert value to hardware
                    let curr_ty = curr_ty.as_hardware_type(elab).map_err(|_| {
                        DiagnosticError::new(
                            "hardware array indexing target needs to have hardware type",
                            curr_span,
                            format!(
                                "this array target needs to have a hardware type, has type `{}`",
                                curr_ty.value_string(elab)
                            ),
                        )
                        .add_info(step_span, "for this hardware array access operation")
                        .report(diags)
                    })?;

                    let curr_value =
                        curr_value
                            .inner
                            .as_hardware_value_unchecked(refs, large, curr_value.span, curr_ty.clone())?;
                    let (array_inner, array_len) = match curr_value.ty {
                        HardwareType::Array(array_inner, array_len) => (array_inner, array_len),
                        _ => {
                            let diag = diags.report_error_internal(step_span, "should still be array after conversion");
                            return Err(diag);
                        }
                    };
                    let array_len = Spanned::new(curr_span, &array_len);

                    // apply step
                    match step {
                        AssignmentStepHardware::ArrayIndex(index) => {
                            check_range_index(diags, Spanned::new(step_span, &index.ty), array_len.map_inner(Some))?;

                            let expr = IrExpressionLarge::ArrayIndex {
                                base: curr_value.expr,
                                index: index.expr.clone(),
                            };
                            Value::Hardware(HardwareValue {
                                ty: Arc::unwrap_or_clone(array_inner),
                                domain: curr_value.domain.join(index.domain),
                                expr: large.push_expr(expr),
                            })
                        }
                        AssignmentStepHardware::ArraySlice {
                            start: slice_start,
                            length: slice_length,
                        } => {
                            check_range_slice(
                                diags,
                                Spanned::new(step_span, &slice_start.ty),
                                Some(Spanned::new(step_span, slice_length)),
                                array_len.map_inner(Some),
                            )?;

                            let expr = IrExpressionLarge::ArraySlice {
                                base: curr_value.expr,
                                start: slice_start.expr.clone(),
                                len: slice_length.clone(),
                            };
                            Value::Hardware(HardwareValue {
                                ty: HardwareType::Array(array_inner, slice_length.clone()),
                                domain: curr_value.domain.join(slice_start.domain),
                                expr: large.push_expr(expr),
                            })
                        }
                    }
                }
            };

            curr_value = Spanned::new(curr_value.span.join(step_span), next);
        }

        Ok(curr_value.inner)
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

impl SetCompileTarget<'_> {
    pub fn ty(&self) -> Type {
        match self {
            SetCompileTarget::Scalar(v) => v.ty(),
            SetCompileTarget::Slice(slice) => {
                let v = SimpleCompileValue::Array(Arc::new(slice.to_vec()));
                v.ty()
            }
        }
    }
}

fn set_compile_value_impl(
    refs: CompileRefs,
    target: Spanned<SetCompileTarget<'_>>,
    steps: &[Spanned<&AssignmentStepCompile>],
    assign_op_span: Span,
    source: Spanned<CompileValue>,
) -> DiagResult {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

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
                                source.inner.ty().value_string(elab)
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
                length: slice_len,
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
                    _ => {
                        let curr_ty = Spanned::new(target_span, target_inner.ty());
                        return Err(err_expected_tuple(refs, curr_ty.as_ref(), step.span));
                    }
                },
                SetCompileTarget::Slice(_) => {
                    let curr_ty = Spanned::new(target_span, target.inner.ty());
                    return Err(err_expected_tuple(refs, curr_ty.as_ref(), step.span));
                }
            };

            // check index in bounds
            let index = check_tuple_index(diags, target_inner.len(), index, target_span, step.span)?;

            // build new target
            SetCompileTarget::Scalar(&mut target_inner[index])
        }
        AssignmentStepCompile::StructField(field) => {
            // check target is struct
            let target_inner = match target.inner {
                SetCompileTarget::Scalar(target_inner) => match target_inner {
                    CompileValue::Compound(CompileCompoundValue::Struct(target_inner)) => target_inner,
                    CompileValue::Hardware(never) => never.unreachable(),
                    _ => {
                        let curr_ty = Spanned::new(target_span, target_inner.ty());
                        return Err(err_expected_struct(refs, curr_ty.as_ref(), step.span));
                    }
                },
                SetCompileTarget::Slice(_) => {
                    let curr_ty = Spanned::new(target_span, target.inner.ty());
                    return Err(err_expected_struct(refs, curr_ty.as_ref(), step.span));
                }
            };

            // get field index
            let ty_info = elab.struct_info(target_inner.ty);
            let field_str = Spanned::new(step.span, field.as_str());
            let field_index = ty_info.field_index(diags, target_span, field_str)?;

            // build new target
            SetCompileTarget::Scalar(&mut target_inner.fields[field_index])
        }
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
    index: Spanned<&ClosedNonEmptyMultiRange<BigUint>>,
    array_len: Spanned<Option<&BigUint>>,
) -> DiagResult {
    let ClosedNonEmptyRange {
        start: _,
        end: index_end,
    } = index.inner.enclosing_range();

    if array_len.inner.is_none_or(|array_len| index_end <= array_len) {
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
    index: Spanned<&BigUint>,
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
    slice_start: Spanned<&ClosedNonEmptyMultiRange<BigUint>>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<&BigUint>,
) -> DiagResult<BigUint> {
    Ok(check_range_slice(diags, slice_start, slice_len, array_len.map_inner(Some))?.unwrap())
}

pub fn check_range_slice(
    diags: &Diagnostics,
    slice_start: Spanned<&ClosedNonEmptyMultiRange<BigUint>>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<Option<&BigUint>>,
) -> DiagResult<Option<BigUint>> {
    #[derive(Copy, Clone)]
    enum SliceLen<T, U> {
        NoneButStartSingle(U),
        Some(T),
    }
    let slice_len = match slice_len {
        Some(slice_len) => SliceLen::Some(slice_len),
        None => match slice_start.inner.as_single() {
            None => {
                return Err(diags.report_error_internal(
                    slice_start.span,
                    "start with non-single range and no length doesn't make sense",
                ));
            }
            Some(slice_start_single) => SliceLen::NoneButStartSingle(slice_start_single),
        },
    };

    let slice_str = || {
        let start = if let Some(slice_start_single) = slice_start.inner.as_single() {
            format!("{slice_start_single}")
        } else {
            format!("({})", slice_start.inner)
        };

        match slice_len {
            SliceLen::NoneButStartSingle(_) => format!("{start}.."),
            SliceLen::Some(slice_len) => format!("{}..+{}", start, slice_len.inner),
        }
    };

    // check start
    let ClosedNonEmptyRange {
        start: _,
        end: slice_start_end,
    } = slice_start.inner.enclosing_range();

    if array_len
        .inner
        .is_some_and(|array_len| slice_start_end - 1 > BigInt::from(array_len.clone()))
    {
        return Err(DiagnosticError::new(
            "array slice start out of bounds",
            slice_start.span,
            format!("slice start of `{}` is out of bounds", slice_str()),
        )
        .add_info(array_len.span, info_array_length(array_len.inner))
        .report(diags));
    }

    // check array length compared to slice length (if we have enough info)
    match slice_len {
        SliceLen::Some(slice_len) => {
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
            }

            Ok(Some(slice_len.inner.clone()))
        }
        SliceLen::NoneButStartSingle(slice_start) => Ok(array_len
            .inner
            .map(|array_len| BigUint::try_from(array_len - slice_start).unwrap())),
    }
}

fn info_array_length(array_len: Option<&BigUint>) -> String {
    match array_len {
        Some(len) => format!("for this array with length `{}`", len),
        None => "for this array".to_string(),
    }
}

#[derive(Copy, Clone)]
pub struct SliceInfo {
    pub start: usize,
    pub length: usize,
}

impl SliceInfo {
    fn as_range(self) -> std::ops::Range<usize> {
        let SliceInfo { start, length } = self;
        start..start + length
    }
}

pub fn check_range_slice_compile(
    diags: &Diagnostics,
    slice_start: Spanned<&BigUint>,
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
    Ok(SliceInfo { start, length: len })
}

fn check_tuple_index(
    diags: &Diagnostics,
    tuple_len: usize,
    index: &BigUint,
    base_span: Span,
    step_span: Span,
) -> DiagResult<usize> {
    index.as_usize_if_lt(tuple_len).ok_or_else(|| {
        DiagnosticError::new(
            "tuple index out of bounds",
            step_span,
            format!("index `{index}` is out of bounds"),
        )
        .add_info(base_span, format!("base is tuple with length `{}`", tuple_len))
        .report(diags)
    })
}

fn check_type_is_array<'t>(
    refs: CompileRefs,
    ty: Spanned<&'t Type>,
    step_span: Span,
    step_is_slice: bool,
) -> DiagResult<(&'t Arc<Type>, Option<&'t BigUint>)> {
    match &ty.inner {
        Type::Array(ty_inner, len) => Ok((ty_inner, len.as_ref())),
        _ => Err(err_expected_array(refs, ty, step_span, step_is_slice)),
    }
}

fn err_expected_array(refs: CompileRefs, ty: Spanned<&Type>, step_span: Span, step_is_slice: bool) -> DiagError {
    let op_name = if step_is_slice { "slice" } else { "index" };

    DiagnosticError::new(
        format!("cannot {op_name} non-array type"),
        step_span,
        format!("array {op_name} here"),
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

pub fn err_expected_tuple(refs: CompileRefs, ty: Spanned<&Type>, step_span: Span) -> DiagError {
    DiagnosticError::new("cannot tuple index non-tuple type", step_span, "tuple index here")
        .add_info(
            ty.span,
            format!(
                "on non-tuple value with type `{}` here",
                ty.inner.value_string(&refs.shared.elaboration_arenas)
            ),
        )
        .report(refs.diags)
}

pub fn err_expected_struct(refs: CompileRefs, ty: Spanned<&Type>, step_span: Span) -> DiagError {
    DiagnosticError::new("cannot struct index non-struct type", step_span, "struct index here")
        .add_info(
            ty.span,
            format!(
                "on non-struct value with type `{}` here",
                ty.inner.value_string(&refs.shared.elaboration_arenas)
            ),
        )
        .report(refs.diags)
}
