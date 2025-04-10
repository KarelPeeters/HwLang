use crate::front::assignment::ValueVersioned;
use crate::front::types::ClosedIncRange;
use crate::util::big_int::BigInt;
use itertools::Itertools;

#[derive(Debug, Default)]
pub struct Implications {
    pub if_true: Vec<Implication>,
    pub if_false: Vec<Implication>,
}

impl Implications {
    pub fn invert(self) -> Self {
        Self {
            if_true: self.if_false,
            if_false: self.if_true,
        }
    }
}

#[derive(Debug)]
pub struct Implication {
    pub value: ValueVersioned,
    pub op: ImplicationOp,
    pub right: BigInt,
}

impl Implication {
    pub fn new(value: ValueVersioned, op: ImplicationOp, right: BigInt) -> Self {
        Self { value, op, right }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ImplicationOp {
    Neq,
    Lt,
    Gt,
}

#[derive(Debug)]
pub struct ClosedIncRangeMulti {
    ranges: Vec<ClosedIncRange<BigInt>>,
}

impl ClosedIncRangeMulti {
    pub fn from_range(range: ClosedIncRange<BigInt>) -> Self {
        Self { ranges: vec![range] }
    }

    pub fn to_range(self) -> Option<ClosedIncRange<BigInt>> {
        Some(ClosedIncRange {
            start_inc: self.ranges.first()?.start_inc.clone(),
            end_inc: self.ranges.last()?.end_inc.clone(),
        })
    }

    pub fn apply_implication(&mut self, op: ImplicationOp, right: &BigInt) {
        match op {
            ImplicationOp::Lt => {
                for i in 0..self.ranges.len() {
                    let range = &mut self.ranges[i];
                    if &range.start_inc >= right {
                        drop(self.ranges.drain(i..));
                        break;
                    }
                    if &range.end_inc >= right {
                        range.end_inc = right - 1;
                    }
                }
            }
            ImplicationOp::Gt => {
                for i in (0..self.ranges.len()).rev() {
                    let range = &mut self.ranges[i];
                    if &range.end_inc <= right {
                        drop(self.ranges.drain(..=i));
                        break;
                    }
                    if &range.start_inc <= right {
                        range.start_inc = right + 1;
                    }
                }
            }
            ImplicationOp::Neq => {
                for i in 0..self.ranges.len() {
                    let range = &self.ranges[i];
                    if range.contains(&right) {
                        let mut new_ranges = vec![];

                        let first = ClosedIncRange {
                            start_inc: range.start_inc.clone(),
                            end_inc: right - 1,
                        };
                        if &first.start_inc <= &first.end_inc {
                            new_ranges.push(first);
                        }
                        let second = ClosedIncRange {
                            start_inc: right + 1,
                            end_inc: range.end_inc.clone(),
                        };
                        if &second.start_inc <= &second.end_inc {
                            new_ranges.push(second);
                        }

                        drop(self.ranges.splice(i..=i, new_ranges));
                        break;
                    }
                }
            }
        }

        self.validate();
    }

    fn validate(&self) {
        // ranges non-empty
        for range in &self.ranges {
            assert!(range.end_inc >= range.start_inc);
        }
        // ranges non-overlapping
        for (a, b) in self.ranges.iter().tuple_windows() {
            assert!(a.end_inc < b.start_inc);
        }
    }
}
