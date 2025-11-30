use crate::front::flow::ValueVersion;
use crate::front::types::{ClosedIncRange, HardwareType, Type, Typed};
use crate::front::value::{HardwareValue, MixedCompoundValue, SimpleCompileValue, Value};
use crate::util::big_int::BigInt;
use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct BoolImplications<V = ValueVersion> {
    pub if_true: Vec<Implication<V>>,
    pub if_false: Vec<Implication<V>>,
}

// TODO rename the concept "implication" to "type narrowing" everywhere
#[derive(Debug, Clone)]
pub struct Implication<V = ValueVersion> {
    pub version: V,
    pub kind: ImplicationKind,
}

#[derive(Debug, Clone)]
pub enum ImplicationKind {
    BoolEq(bool),
    IntOp(ImplicationIntOp, BigInt),
}

#[derive(Debug, Copy, Clone)]
pub enum ImplicationIntOp {
    Neq,
    Lt,
    Gt,
}

pub type ValueWithVersion<S = SimpleCompileValue, C = MixedCompoundValue, T = HardwareType> =
    Value<S, C, HardwareValueWithVersion<ValueVersion, T>>;
pub type ValueWithImplications<S = SimpleCompileValue, C = MixedCompoundValue, T = HardwareType> =
    Value<S, C, HardwareValueWithImplications<T>>;
pub type HardwareValueWithMaybeVersion = HardwareValueWithVersion<Option<ValueVersion>>;

#[derive(Debug, Clone)]
pub struct HardwareValueWithVersion<V = ValueVersion, T = HardwareType> {
    pub value: HardwareValue<T>,
    pub version: V,
}

#[derive(Debug, Clone)]
pub struct HardwareValueWithImplications<T = HardwareType> {
    pub value: HardwareValue<T>,
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
    pub fn new(version: Option<ValueVersion>) -> Self {
        let mut result = BoolImplications {
            if_true: vec![],
            if_false: vec![],
        };
        if let Some(version) = version {
            result.if_true.push(Implication::new_bool(version, true));
            result.if_false.push(Implication::new_bool(version, false));
        }
        result
    }

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
    pub fn new_int(value: ValueVersion, op: ImplicationIntOp, right: BigInt) -> Self {
        Self {
            version: value,
            kind: ImplicationKind::IntOp(op, right),
        }
    }

    pub fn new_bool(value: ValueVersion, equal: bool) -> Self {
        Self {
            version: value,
            kind: ImplicationKind::BoolEq(equal),
        }
    }
}

impl ValueWithVersion {
    pub fn into_value(self) -> Value {
        self.map_hardware(|v| v.value)
    }
}

impl<S, C, T> ValueWithImplications<S, C, T> {
    pub fn simple(value: Value<S, C, HardwareValue<T>>) -> Self {
        value.map_hardware(HardwareValueWithImplications::simple)
    }

    pub fn simple_version(value: ValueWithVersion<S, C, T>) -> Self {
        value.map_hardware(HardwareValueWithImplications::simple_version)
    }

    pub fn into_value(self) -> Value<S, C, HardwareValue<T>> {
        self.map_hardware(|v| v.value)
    }
}

impl<T> HardwareValueWithImplications<T> {
    pub fn simple(value: HardwareValue<T>) -> Self {
        Self {
            value,
            version: None,
            implications: BoolImplications::new(None),
        }
    }

    pub fn simple_version(value: HardwareValueWithVersion<ValueVersion, T>) -> Self {
        Self {
            value: value.value,
            version: Some(value.version),
            implications: BoolImplications::new(Some(value.version)),
        }
    }

    pub fn map_type<U>(self, f: impl FnOnce(T) -> U) -> HardwareValueWithImplications<U> {
        HardwareValueWithImplications {
            value: self.value.map_type(f),
            version: self.version,
            implications: self.implications,
        }
    }
}

impl Typed for ValueWithImplications {
    fn ty(&self) -> Type {
        match self {
            Value::Simple(v) => v.ty(),
            Value::Compound(v) => v.ty(),
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

    pub fn apply_implication(&mut self, op: ImplicationIntOp, right: &BigInt) {
        match op {
            ImplicationIntOp::Lt => {
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
            ImplicationIntOp::Gt => {
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
            ImplicationIntOp::Neq => {
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
