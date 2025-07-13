use crate::front::flow::ValueVersion;
use crate::front::types::{ClosedIncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::util::big_int::BigInt;
use itertools::Itertools;

#[derive(Default, Debug, Clone)]
pub struct BoolImplications {
    pub if_true: Vec<Implication>,
    pub if_false: Vec<Implication>,
}

#[derive(Debug, Clone)]
pub struct Implication {
    pub version: ValueVersion,
    pub op: ImplicationOp,
    pub right: BigInt,
}

#[derive(Debug, Copy, Clone)]
pub enum ImplicationOp {
    Neq,
    Lt,
    Gt,
}

pub type ValueWithVersion = Value<CompileValue, HardwareValueWithVersion>;
pub type ValueWithImplications = Value<CompileValue, HardwareValueWithImplications>;
pub type HardwareValueWithMaybeVersion = HardwareValueWithVersion<Option<ValueVersion>>;

#[derive(Debug, Clone)]
pub struct HardwareValueWithVersion<V = ValueVersion> {
    pub value: HardwareValue,
    pub version: V,
}

#[derive(Debug, Clone)]
pub struct HardwareValueWithImplications {
    pub value: HardwareValue,
    pub version: Option<ValueVersion>,
    pub implications: BoolImplications,
}

impl<V> HardwareValueWithVersion<V> {
    pub fn map_version<F: FnOnce(V) -> U, U>(self, f: F) -> HardwareValueWithVersion<U> {
        HardwareValueWithVersion {
            value: self.value,
            version: f(self.version),
        }
    }
}

impl BoolImplications {
    pub fn invert(self) -> Self {
        Self {
            if_true: self.if_false,
            if_false: self.if_true,
        }
    }
}

pub fn join_implications(branch_implications: &[Vec<Implication>]) -> Vec<Implication> {
    // TODO do something more interesting here, this placeholder implementation is correct but a missed opportunity
    let _ = branch_implications;
    vec![]
}

impl Implication {
    pub fn new(value: ValueVersion, op: ImplicationOp, right: BigInt) -> Self {
        Self {
            version: value,
            op,
            right,
        }
    }
}

impl ValueWithVersion {
    pub fn into_value(self) -> Value {
        self.map_hardware(|v| v.value)
    }
}

impl ValueWithImplications {
    pub fn simple(value: Value) -> Self {
        value.map_hardware(HardwareValueWithImplications::simple)
    }

    pub fn simple_version(value: ValueWithVersion) -> Self {
        value.map_hardware(HardwareValueWithImplications::simple_version)
    }

    pub fn into_value(self) -> Value {
        self.map_hardware(|v| v.value)
    }
}

impl HardwareValueWithImplications {
    pub fn simple(value: HardwareValue) -> Self {
        Self {
            value,
            version: None,
            implications: BoolImplications::default(),
        }
    }

    pub fn simple_version(value: HardwareValueWithVersion) -> Self {
        Self {
            value: value.value,
            version: Some(value.version),
            implications: BoolImplications::default(),
        }
    }
}

impl Typed for ValueWithImplications {
    fn ty(&self) -> Type {
        match self {
            Value::Compile(v) => v.ty(),
            Value::Hardware(v) => v.value.ty(),
        }
    }
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
                    if range.contains(right) {
                        let mut new_ranges = vec![];

                        let first = ClosedIncRange {
                            start_inc: range.start_inc.clone(),
                            end_inc: right - 1,
                        };
                        if first.start_inc <= first.end_inc {
                            new_ranges.push(first);
                        }
                        let second = ClosedIncRange {
                            start_inc: right + 1,
                            end_inc: range.end_inc.clone(),
                        };
                        if second.start_inc <= second.end_inc {
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
