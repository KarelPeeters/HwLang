use crate::front::range_arithmetic::range_binary_add;
use crate::mid::ir::{
    IrExpression, IrExpressionLarge, IrIntArithmeticOp, IrLargeArena, IrSignals, IrType, IrVariables,
};
use crate::util::big_int::BigUint;
use crate::util::data::{IndexMapExt, VecExt};

#[derive(Debug, Clone)]
pub struct IrTargetSteps {
    pub steps_scalar: Vec<IrTargetStepScalar>,
    pub step_slice: Option<IrTargetStepSlice>,
}

#[derive(Debug, Clone)]
pub enum IrTargetStep {
    Scalar(IrTargetStepScalar),
    Slice(IrTargetStepSlice),
}

#[derive(Debug, Clone)]
pub enum IrTargetStepScalar {
    ArrayIndex(IrExpression),
    TupleIndex(usize),
    StructField(usize),
}

#[derive(Debug, Clone)]
pub struct IrTargetStepSlice {
    pub start: IrExpression,
    pub len: BigUint,
}

#[derive(Debug)]
pub struct InvalidTypeForStep;

#[derive(Debug)]
pub struct InvalidScalarAfterSlice;

impl IrTargetSteps {
    pub fn new() -> Self {
        IrTargetSteps {
            steps_scalar: vec![],
            step_slice: None,
        }
    }

    pub fn single(step: IrTargetStep) -> Self {
        let mut result = IrTargetSteps {
            steps_scalar: vec![],
            step_slice: None,
        };
        match step {
            IrTargetStep::Scalar(step) => {
                result.steps_scalar.push(step);
            }
            IrTargetStep::Slice(step) => {
                result.step_slice = Some(step);
            }
        }
        result
    }

    pub fn apply_to_type(&self, base: IrType) -> Result<IrType, InvalidTypeForStep> {
        let IrTargetSteps {
            steps_scalar,
            step_slice,
        } = self;

        let mut curr = base;
        for step in steps_scalar {
            curr = match step {
                IrTargetStepScalar::ArrayIndex(_) => match curr {
                    IrType::Array(inner, _) => *inner,
                    _ => return Err(InvalidTypeForStep),
                },
                &IrTargetStepScalar::TupleIndex(index) => match curr {
                    IrType::Tuple(fields) => fields.get_owned(index),
                    _ => return Err(InvalidTypeForStep),
                },
                &IrTargetStepScalar::StructField(index) => match curr {
                    IrType::Struct(fields) => fields.fields.get_index_owned(index).unwrap().1,
                    _ => return Err(InvalidTypeForStep),
                },
            };
        }
        if let Some(step) = step_slice {
            let IrTargetStepSlice { start: _, len } = step;
            curr = match curr {
                IrType::Array(inner, _) => IrType::Array(inner, len.clone()),
                _ => return Err(InvalidTypeForStep),
            }
        }
        Ok(curr)
    }

    pub fn is_empty(&self) -> bool {
        self.steps_scalar.is_empty() && self.step_slice.is_none()
    }

    pub fn push(
        &mut self,
        large: &mut IrLargeArena,
        signals: &IrSignals,
        vars: &IrVariables,
        step: IrTargetStep,
    ) -> Result<(), InvalidScalarAfterSlice> {
        match step {
            IrTargetStep::Scalar(step) => match step {
                IrTargetStepScalar::ArrayIndex(index) => {
                    let combined_index = if let Some(prev_step) = Option::take(&mut self.step_slice) {
                        // index after slice: add slice start to index, ignore slice len
                        let IrTargetStepSlice {
                            start: prev_start,
                            len: _,
                        } = prev_step;

                        let prev_start_range = prev_start.ty(large, signals, vars).unwrap_int();
                        let index_range = index.ty(large, signals, vars).unwrap_int();
                        let range_combined = range_binary_add(prev_start_range.as_ref(), index_range.as_ref());

                        large.push_expr(IrExpressionLarge::IntArithmetic(
                            IrIntArithmeticOp::Add,
                            range_combined,
                            prev_start,
                            index,
                        ))
                    } else {
                        // index after scalar (or nothing): just do index
                        index
                    };

                    self.steps_scalar.push(IrTargetStepScalar::ArrayIndex(combined_index));
                }
                IrTargetStepScalar::TupleIndex(_) | IrTargetStepScalar::StructField(_) => {
                    if self.step_slice.is_some() {
                        // field after slice: not allowed
                        return Err(InvalidScalarAfterSlice);
                    } else {
                        // field after scalar (or nothing): just do field
                        self.steps_scalar.push(step);
                    }
                }
            },
            IrTargetStep::Slice(step) => {
                let combined = if let Some(prev_step) = Option::take(&mut self.step_slice) {
                    // slice after slice: add slice starts, ignore previous len
                    let IrTargetStepSlice {
                        start: prev_start,
                        len: _,
                    } = prev_step;
                    let IrTargetStepSlice {
                        start: curr_start,
                        len: curr_len,
                    } = step;

                    let prev_start_range = prev_start.ty(large, signals, vars).unwrap_int();
                    let curr_start_range = curr_start.ty(large, signals, vars).unwrap_int();
                    let combined_start_range = range_binary_add(prev_start_range.as_ref(), curr_start_range.as_ref());

                    let combined_start = large.push_expr(IrExpressionLarge::IntArithmetic(
                        IrIntArithmeticOp::Add,
                        combined_start_range,
                        prev_start,
                        curr_start,
                    ));

                    IrTargetStepSlice {
                        start: combined_start,
                        len: curr_len,
                    }
                } else {
                    // slice after scalar (or nothing): just do slice
                    step
                };

                self.step_slice = Some(combined);
            }
        }

        Ok(())
    }
}
