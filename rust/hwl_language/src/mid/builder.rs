pub use crate::front::range_arithmetic::range_binary_add_uint;
use crate::mid::ir::{
    IrExpression, IrExpressionLarge, IrIntArithmeticOp, IrLargeArena, IrTargetStepScalar, IrTargetStepSlice,
    IrTargetSteps,
};
use crate::util::big_int::{BigInt, BigUint};
use crate::util::range::ClosedNonEmptyRange;

/// Utility to build [IrTargetSteps] instances.
/// This automatically combines sequences of slice and index operations into the normalized form required by the IR.
pub struct IrTargetStepsBuilder {
    steps_scalar: Vec<IrTargetStepScalar>,
    step_slice: Option<IrTargetStepBuildSlice>,
}

#[derive(Debug)]
pub enum IrTargetStepBuild {
    ArrayIndex {
        index: IrExpression,
        index_range: ClosedNonEmptyRange<BigUint>,
    },
    ArraySlice(IrTargetStepBuildSlice),
    TupleIndex(usize),
    StructField(usize),
}

#[derive(Debug)]
pub struct IrTargetStepBuildSlice {
    pub start: IrExpression,
    pub start_range: ClosedNonEmptyRange<BigUint>,
    pub len: BigUint,
}

#[derive(Debug)]
pub struct InvalidScalarAfterSlice;

impl IrTargetStepsBuilder {
    pub fn new() -> Self {
        IrTargetStepsBuilder {
            steps_scalar: vec![],
            step_slice: None,
        }
    }

    pub fn push(&mut self, large: &mut IrLargeArena, step: IrTargetStepBuild) -> Result<(), InvalidScalarAfterSlice> {
        match step {
            IrTargetStepBuild::ArrayIndex { index, index_range } => {
                let combined_index = if let Some(prev_step) = Option::take(&mut self.step_slice) {
                    // add slice start to index, ignore len
                    let IrTargetStepBuildSlice {
                        start: prev_start,
                        start_range: prev_start_range,
                        len: _,
                    } = prev_step;

                    let range_combined = range_binary_add_uint(prev_start_range.as_ref(), index_range.as_ref());
                    large.push_expr(IrExpressionLarge::IntArithmetic(
                        IrIntArithmeticOp::Add,
                        range_combined.map(BigInt::from),
                        prev_start,
                        index,
                    ))
                } else {
                    // just index
                    index
                };

                self.steps_scalar.push(IrTargetStepScalar::ArrayIndex(combined_index));
            }
            IrTargetStepBuild::ArraySlice(IrTargetStepBuildSlice {
                start: curr_start,
                start_range: curr_start_range,
                len: curr_len,
            }) => {
                let (combined_start, combined_start_range) = if let Some(prev_step) = Option::take(&mut self.step_slice)
                {
                    // add slice starts, ignore previous len
                    let IrTargetStepBuildSlice {
                        start: prev_start,
                        start_range: prev_start_range,
                        len: _,
                    } = prev_step;

                    let combined_start_range =
                        range_binary_add_uint(prev_start_range.as_ref(), curr_start_range.as_ref());
                    let combined_start = large.push_expr(IrExpressionLarge::IntArithmetic(
                        IrIntArithmeticOp::Add,
                        combined_start_range.as_ref().map(BigInt::from),
                        prev_start,
                        curr_start,
                    ));

                    (combined_start, combined_start_range)
                } else {
                    // just use current start
                    (curr_start, curr_start_range)
                };

                self.step_slice = Some(IrTargetStepBuildSlice {
                    start: combined_start,
                    start_range: combined_start_range,
                    len: curr_len,
                })
            }
            IrTargetStepBuild::TupleIndex(index) => {
                if self.step_slice.is_some() {
                    return Err(InvalidScalarAfterSlice);
                }
                self.steps_scalar.push(IrTargetStepScalar::TupleIndex(index));
            }
            IrTargetStepBuild::StructField(index) => {
                if self.step_slice.is_some() {
                    return Err(InvalidScalarAfterSlice);
                }
                self.steps_scalar.push(IrTargetStepScalar::StructField(index));
            }
        }

        Ok(())
    }

    pub fn finish(self) -> IrTargetSteps {
        let IrTargetStepsBuilder {
            steps_scalar,
            step_slice,
        } = self;
        IrTargetSteps {
            steps_scalar,
            step_slice: step_slice.map(|step| {
                let IrTargetStepBuildSlice {
                    start,
                    start_range: _,
                    len,
                } = step;
                IrTargetStepSlice { start, len }
            }),
        }
    }
}
