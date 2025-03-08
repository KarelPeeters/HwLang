use crate::front::block::TypedIrExpression;
use crate::front::check::{check_type_is_uint_compile, TypeContainsReason};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrExpression, IrTargetStep};
use crate::front::misc::ValueDomain;
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_traits::{Signed, ToPrimitive};

#[derive(Debug, Clone)]
pub struct ArraySteps<S = ArrayStep> {
    steps: Vec<Spanned<S>>,
}

pub type ArrayStep = MaybeCompile<ArrayStepHardware, ArrayStepCompile>;

#[derive(Debug, Clone)]
pub enum ArrayStepCompile {
    ArrayIndex(BigInt),
    ArraySlice { start: BigInt, len: Option<BigUint> },
}

#[derive(Debug, Clone)]
pub enum ArrayStepHardware {
    ArrayIndex(TypedIrExpression<ClosedIncRange<BigInt>>),
    ArraySlice {
        start: TypedIrExpression<ClosedIncRange<BigInt>>,
        len: BigUint,
    },
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
    pub fn any_hardware(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step.inner, MaybeCompile::Other(_)))
    }

    pub fn unwrap_compile(&self) -> ArraySteps<&ArrayStepCompile> {
        ArraySteps {
            steps: self
                .steps
                .iter()
                .map(|step| step.as_ref().map_inner(|m| m.as_ref().unwrap_compile()))
                .collect(),
        }
    }

    pub fn apply_to_expected_type(&self, diags: &Diagnostics, ty: Spanned<Type>) -> Result<Type, ErrorGuaranteed> {
        let (ty, _) = self.apply_to_type_impl(diags, ty, true)?;
        Ok(ty)
    }

    pub fn apply_to_hardware_type(
        &self,
        diags: &Diagnostics,
        ty: Spanned<&HardwareType>,
    ) -> Result<(HardwareType, Vec<IrTargetStep>), ErrorGuaranteed> {
        let (result_ty, steps) = self.apply_to_type_impl(diags, ty.map_inner(HardwareType::as_type), false)?;
        let steps = steps.unwrap();

        let result_ty_hw = result_ty.as_hardware_type().ok_or_else(|| {
            diags.report_internal_error(
                ty.span,
                "applying access steps to hardware type should result in hardware type again",
            )
        })?;

        Ok((result_ty_hw, steps))
    }

    pub fn for_each_domain(&self, mut f: impl FnMut(Spanned<&ValueDomain>)) {
        for step in &self.steps {
            let d = match &step.inner {
                ArrayStep::Compile(_) => &ValueDomain::CompileTime,
                ArrayStep::Other(step) => match step {
                    ArrayStepHardware::ArrayIndex(index) => &index.domain,
                    ArrayStepHardware::ArraySlice { start, len: _ } => &start.domain,
                },
            };
            f(Spanned::new(step.span, d));
        }
    }

    fn apply_to_type_impl(
        &self,
        diags: &Diagnostics,
        ty: Spanned<Type>,
        pass_any: bool,
    ) -> Result<(Type, Result<Vec<IrTargetStep>, EncounteredAny>), ErrorGuaranteed> {
        let ArraySteps { steps } = self;

        // forward
        let mut steps_ir = vec![];
        let mut slice_lens = vec![];
        let mut curr_ty = ty;
        for step in steps {
            let (ty_array_inner, ty_array_len) = match &curr_ty.inner {
                Type::Array(ty_inner, len) => (&**ty_inner, len),
                Type::Any if pass_any => return Ok((Type::Any, Err(EncounteredAny))),
                _ => return Err(diags.report(diag_expected_array_type(curr_ty.as_ref(), step.span))),
            };
            let ty_array_len = Spanned::new(curr_ty.span, ty_array_len);

            let (step_ir, slice_len) = match &step.inner {
                ArrayStep::Compile(step_inner) => match step_inner {
                    ArrayStepCompile::ArrayIndex(index) => {
                        check_range_index(
                            diags,
                            Spanned::new(step.span, ClosedIncRange::single(index)),
                            ty_array_len,
                        )?;
                        (IrTargetStep::ArrayIndex(IrExpression::Int(index.clone())), None)
                    }
                    ArrayStepCompile::ArraySlice { start, len } => {
                        let len = check_range_slice(
                            diags,
                            Spanned::new(step.span, ClosedIncRange::single(start)),
                            len.as_ref().map(|len| Spanned::new(step.span, len)),
                            ty_array_len,
                        )?;
                        (
                            IrTargetStep::ArraySlice(IrExpression::Int(start.clone()), len.clone()),
                            Some(len),
                        )
                    }
                },
                ArrayStep::Other(step_inner) => match step_inner {
                    ArrayStepHardware::ArrayIndex(index) => {
                        check_range_index(diags, Spanned::new(step.span, index.ty.as_ref()), ty_array_len)?;
                        (IrTargetStep::ArrayIndex(index.expr.clone()), None)
                    }
                    ArrayStepHardware::ArraySlice { start, len } => {
                        let len = check_range_slice(
                            diags,
                            Spanned::new(step.span, start.ty.as_ref()),
                            Some(Spanned::new(step.span, len)),
                            ty_array_len,
                        )?;
                        (IrTargetStep::ArraySlice(start.expr.clone(), len.clone()), Some(len))
                    }
                },
            };

            curr_ty = Spanned::new(curr_ty.span.join(step.span), ty_array_inner.clone());
            steps_ir.push(step_ir);
            if let Some(slice_len) = slice_len {
                slice_lens.push(slice_len);
            }
        }

        // backward
        let result_ty = slice_lens
            .into_iter()
            .rev()
            .fold(curr_ty.inner, |acc, len| Type::Array(Box::new(acc), len));

        Ok((result_ty, Ok(steps_ir)))
    }

    pub fn apply_to_value(
        &self,
        diags: &Diagnostics,
        value: Spanned<MaybeCompile<TypedIrExpression>>,
    ) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let ArraySteps { steps } = self;

        let mut curr = value;

        for step in steps {
            let next_inner = match (&step.inner, curr.inner) {
                (MaybeCompile::Compile(step_inner), MaybeCompile::Compile(curr_inner)) => match curr_inner {
                    CompileValue::Type(inner) => {
                        // create new array type
                        // TODO it's a bit awkward that indexing and array type def shares the same syntax
                        let reason = TypeContainsReason::ArrayLen { span_len: step.span };
                        match step_inner {
                            ArrayStepCompile::ArrayIndex(len) => {
                                let len_as_value = Spanned::new(step.span, CompileValue::Int(len.clone()));
                                let len = check_type_is_uint_compile(diags, reason, len_as_value)?;
                                MaybeCompile::Compile(CompileValue::Type(Type::Array(Box::new(inner), len)))
                            }
                            ArrayStepCompile::ArraySlice { start, len } => {
                                let slice_as_range = IncRange {
                                    start_inc: Some(start.clone()),
                                    end_inc: len.as_ref().map(|len| start + BigInt::from(len.clone())),
                                };
                                let slice_as_value = Spanned::new(step.span, CompileValue::IntRange(slice_as_range));
                                check_type_is_uint_compile(diags, reason, slice_as_value)?;
                                unreachable!()
                            }
                        }
                    }
                    CompileValue::Array(curr_inner) => {
                        // index into array
                        let value_len = Spanned::new(curr.span, curr_inner.len());
                        match step_inner {
                            ArrayStepCompile::ArrayIndex(index) => {
                                let index =
                                    check_range_index_compile(diags, Spanned::new(step.span, index), value_len)?;
                                let mut curr_inner = curr_inner;
                                MaybeCompile::Compile(curr_inner.swap_remove(index))
                            }
                            ArrayStepCompile::ArraySlice { start, len } => {
                                let SliceInfo { start, len } = check_range_slice_compile(
                                    diags,
                                    Spanned::new(step.span, start),
                                    len.as_ref().map(|len| Spanned::new(step.span, len)),
                                    value_len,
                                )?;
                                let mut curr_inner = curr_inner;
                                let slice = curr_inner.drain(start..start + len).collect();
                                MaybeCompile::Compile(CompileValue::Array(slice))
                            }
                        }
                    }
                    _ => {
                        return Err(diags.report(diag_expected_array_value(
                            Spanned::new(curr.span, &curr_inner),
                            step.span,
                        )))
                    }
                },
                (step_inner, curr_inner) => {
                    // convert curr to hardware
                    let ty = curr_inner.ty();
                    let ty = ty.as_hardware_type().ok_or_else(|| {
                        let diag = Diagnostic::new("hardware array indexing target needs to have hardware type")
                            .add_error(
                                curr.span,
                                format!(
                                    "this array target needs to have a hardware type, has type `{}`",
                                    ty.to_diagnostic_string()
                                ),
                            )
                            .add_info(step.span, "for this hardware array access operation")
                            .finish();
                        diags.report(diag)
                    })?;
                    let curr_inner = curr_inner.as_ir_expression(diags, curr.span, &ty)?;
                    let (curr_array_inner_ty, curr_array_len) = match curr_inner.ty {
                        HardwareType::Array(curr_array_inner_ty, curr_array_len) => {
                            (curr_array_inner_ty, curr_array_len)
                        }
                        _ => {
                            return Err(diags.report(diag_expected_array_type(
                                Spanned::new(curr.span, &ty.as_type()),
                                step.span,
                            )))
                        }
                    };
                    let curr_array_len = Spanned::new(curr.span, curr_array_len);

                    // convert step to hardware
                    let (result_expr, step_domain) = match step_inner {
                        MaybeCompile::Compile(ArrayStepCompile::ArrayIndex(index)) => {
                            check_range_index(
                                diags,
                                Spanned::new(step.span, ClosedIncRange::single(index)),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpression::ArrayIndex {
                                    base: Box::new(curr_inner.expr),
                                    index: Box::new(IrExpression::Int(index.clone())),
                                },
                                ValueDomain::CompileTime,
                            )
                        }
                        MaybeCompile::Compile(ArrayStepCompile::ArraySlice { start, len }) => {
                            let len = check_range_slice(
                                diags,
                                Spanned::new(step.span, ClosedIncRange::single(start)),
                                len.as_ref().map(|len| Spanned::new(step.span, len)),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpression::ArraySlice {
                                    base: Box::new(curr_inner.expr),
                                    start: Box::new(IrExpression::Int(start.clone())),
                                    len: len.clone(),
                                },
                                ValueDomain::CompileTime,
                            )
                        }
                        MaybeCompile::Other(ArrayStepHardware::ArrayIndex(index)) => {
                            check_range_index(
                                diags,
                                Spanned::new(step.span, index.ty.as_ref()),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpression::ArrayIndex {
                                    base: Box::new(curr_inner.expr),
                                    index: Box::new(index.expr.clone()),
                                },
                                index.domain.clone(),
                            )
                        }
                        MaybeCompile::Other(ArrayStepHardware::ArraySlice { start, len }) => {
                            let len = check_range_slice(
                                diags,
                                Spanned::new(step.span, start.ty.as_ref()),
                                Some(Spanned::new(step.span, len)),
                                curr_array_len.as_ref(),
                            )?;
                            (
                                IrExpression::ArraySlice {
                                    base: Box::new(curr_inner.expr),
                                    start: Box::new(start.expr.clone()),
                                    len: len.clone(),
                                },
                                start.domain.clone(),
                            )
                        }
                    };

                    // build final value
                    MaybeCompile::Other(TypedIrExpression {
                        ty: *curr_array_inner_ty,
                        domain: curr_inner.domain.join(&step_domain),
                        expr: result_expr,
                    })
                }
            };

            curr = Spanned::new(curr.span.join(step.span), next_inner);
        }

        Ok(curr.inner)
    }
}

impl ArraySteps<&ArrayStepCompile> {
    pub fn set_compile_value(
        &self,
        diags: &Diagnostics,
        old: Spanned<CompileValue>,
        op_span: Span,
        new: Spanned<CompileValue>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        fn set_compile_value_impl(
            diags: &Diagnostics,
            old_curr: Spanned<CompileValue>,
            steps: &[Spanned<&ArrayStepCompile>],
            op_span: Span,
            new_curr: Spanned<CompileValue>,
        ) -> Result<CompileValue, ErrorGuaranteed> {
            let (step, rest) = match steps.split_first() {
                None => return Ok(new_curr.inner),
                Some(pair) => pair,
            };

            let mut old_curr_inner = match old_curr.inner {
                CompileValue::Array(curr) => curr,
                _ => {
                    let diag = Diagnostic::new("expected array value for array access")
                        .add_info(
                            old_curr.span,
                            format!("non-array value `{}` here", old_curr.inner.to_diagnostic_string()),
                        )
                        .add_error(step.span, "this array access needs an array")
                        .finish();
                    return Err(diags.report(diag));
                }
            };
            let array_len = Spanned::new(old_curr.span, old_curr_inner.len());

            match &step.inner {
                ArrayStepCompile::ArrayIndex(index) => {
                    let index = check_range_index_compile(diags, Spanned::new(step.span, index), array_len)?;
                    let old_next = Spanned::new(old_curr.span.join(step.span), old_curr_inner[index].clone());
                    old_curr_inner[index] = set_compile_value_impl(diags, old_next, rest, op_span, new_curr)?;
                    Ok(CompileValue::Array(old_curr_inner))
                }
                ArrayStepCompile::ArraySlice { start, len } => {
                    let new_curr_inner = match new_curr.inner {
                        CompileValue::Array(new_curr_inner) => new_curr_inner,
                        _ => return Err(diags.report(diag_expected_array_value(new_curr.as_ref(), step.span))),
                    };

                    let SliceInfo { start, len: slice_len } = check_range_slice_compile(
                        diags,
                        Spanned::new(step.span, start),
                        len.as_ref().map(|len| Spanned::new(step.span, len)),
                        array_len,
                    )?;

                    if slice_len != new_curr_inner.len() {
                        let diag = Diagnostic::new("slice assignment length mismatch")
                            .add_error(op_span, "length mismatch on this assignment")
                            .add_info(
                                old_curr.span.join(step.span),
                                format!("target slice has length {}", slice_len),
                            )
                            .add_info(
                                new_curr.span,
                                format!("source array has length {}", new_curr_inner.len()),
                            )
                            .finish();
                        return Err(diags.report(diag));
                    }

                    for i in 0..slice_len {
                        let old_next = Spanned::new(old_curr.span.join(step.span), old_curr_inner[start + i].clone());
                        let new_next = Spanned::new(new_curr.span.join(step.span), new_curr_inner[i].clone());
                        old_curr_inner[start + i] = set_compile_value_impl(diags, old_next, rest, op_span, new_next)?;
                    }

                    Ok(CompileValue::Array(old_curr_inner))
                }
            }
        }

        let ArraySteps { steps } = self;
        set_compile_value_impl(diags, old, steps, op_span, new)
    }

    pub fn get_compile_value(
        &self,
        diags: &Diagnostics,
        value: Spanned<CompileValue>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        // TODO avoid clones
        let self_mapped = ArraySteps {
            steps: self
                .steps
                .iter()
                .map(|s| s.map_inner(|s| ArrayStep::Compile(s.clone())))
                .collect_vec(),
        };
        let value_span = value.span;

        let result = self_mapped.apply_to_value(diags, value.map_inner(MaybeCompile::Compile))?;
        match result {
            MaybeCompile::Compile(result) => Ok(result),
            MaybeCompile::Other(_) => Err(diags.report_internal_error(value_span, "applying compile-time steps to compile-time value should result in compile-time value again, got hardware")),
        }
    }
}

pub fn check_range_index(
    diags: &Diagnostics,
    index: Spanned<ClosedIncRange<&BigInt>>,
    array_len: Spanned<&BigUint>,
) -> Result<(), ErrorGuaranteed> {
    if index.inner.start_inc.is_negative() || index.inner.end_inc >= &BigInt::from(array_len.inner.clone()) {
        let index_str = if index.inner.as_single().is_some() {
            format!("`{}`", index.inner)
        } else {
            format!("with range `{}`", index.inner)
        };

        let diag = Diagnostic::new("array index out of bounds")
            .add_error(index.span, format!("index {index_str} is out of bounds"))
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }
    Ok(())
}

pub fn check_range_index_compile(
    diags: &Diagnostics,
    index: Spanned<&BigInt>,
    array_len: Spanned<usize>,
) -> Result<usize, ErrorGuaranteed> {
    check_range_index(
        diags,
        index.map_inner(ClosedIncRange::single),
        array_len.map_inner(BigUint::from).as_ref(),
    )?;
    Ok(index.inner.to_usize().unwrap())
}

pub fn check_range_slice(
    diags: &Diagnostics,
    slice_start: Spanned<ClosedIncRange<&BigInt>>,
    slice_len: Option<Spanned<&BigUint>>,
    array_len: Spanned<&BigUint>,
) -> Result<BigUint, ErrorGuaranteed> {
    if slice_start.inner.as_single().is_none() && slice_len.is_none() {
        return Err(diags.report_internal_error(
            slice_start.span,
            "start with non-single range and no length doesn't make sense",
        ));
    }

    let slice_str = || {
        let start = if let Some(slice_start_single) = slice_start.inner.as_single() {
            format!("{}", slice_start_single)
        } else {
            format!("({})", slice_start.inner)
        };

        match slice_len {
            None => format!("{}..", start),
            Some(slice_len) => format!("{}..+{}", start, slice_len.inner),
        }
    };

    // check start
    if slice_start.inner.start_inc.is_negative() || slice_start.inner.end_inc > &BigInt::from(array_len.inner.clone()) {
        let diag = Diagnostic::new("array slice start out of bounds")
            .add_error(
                slice_start.span,
                format!("slice start of `{}` is out out of bounds", slice_str()),
            )
            .add_info(
                array_len.span,
                format!("for this array with length `{}`", array_len.inner),
            )
            .finish();
        return Err(diags.report(diag));
    }

    // check start + len if length is provided
    // if len is not provided, knowing that the start is valid is enough
    if let Some(slice_len) = slice_len {
        let slice_end_inc = slice_start.inner.end_inc + BigInt::from(slice_len.inner.clone()) - 1;
        if slice_end_inc >= BigInt::from(array_len.inner.clone()) {
            let diag = Diagnostic::new("array slice end out of bounds")
                .add_error(
                    slice_len.span,
                    format!("slice end of `{}` is out out of bounds", slice_str()),
                )
                .add_info(
                    array_len.span,
                    format!("for this array with length `{}`", array_len.inner),
                )
                .finish();
            return Err(diags.report(diag));
        }

        Ok(slice_len.inner.clone())
    } else {
        let slice_start = slice_start.inner.as_single().unwrap();
        Ok(array_len.inner - slice_start.to_biguint().unwrap())
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
) -> Result<SliceInfo, ErrorGuaranteed> {
    check_range_slice(
        diags,
        slice_start.map_inner(ClosedIncRange::single),
        slice_len,
        array_len.map_inner(BigUint::from).as_ref(),
    )?;
    let start = slice_start.inner.to_usize().unwrap();
    let len = slice_len
        .map(|len| len.inner.to_usize().unwrap())
        .unwrap_or(array_len.inner - start);
    Ok(SliceInfo { start, len })
}

pub fn diag_expected_array_value(value: Spanned<&CompileValue>, step_span: Span) -> Diagnostic {
    Diagnostic::new("array indexing on non-array value")
        .add_error(step_span, "array access operation here")
        .add_info(
            value.span,
            format!("on non-array value `{}` here", value.inner.to_diagnostic_string()),
        )
        .finish()
}

pub fn diag_expected_array_type(ty: Spanned<&Type>, step_span: Span) -> Diagnostic {
    Diagnostic::new("array indexing on non-array type")
        .add_error(step_span, "array access operation here")
        .add_info(
            ty.span,
            format!(
                "on non-array value with type `{}` here",
                ty.inner.to_diagnostic_string()
            ),
        )
        .finish()
}
