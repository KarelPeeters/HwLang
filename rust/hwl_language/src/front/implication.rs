use crate::front::flow::ValueVersion;
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{HardwareValue, MixedCompoundValue, SimpleCompileValue, Value};
use crate::mid::ir::IrExpression;
use crate::util::big_int::BigInt;
use crate::util::range_multi::MultiRange;
use std::fmt::Debug;

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
    IntIn(MultiRange<BigInt>),
}

#[derive(Debug, Copy, Clone)]
pub enum ImplicationIntOp {
    Neq,
    Lt,
    Gt,
}

pub type ValueWithVersion<S = SimpleCompileValue, C = MixedCompoundValue, T = HardwareType, E = IrExpression> =
    Value<S, C, HardwareValueWithVersion<ValueVersion, T, E>>;
pub type ValueWithImplications<S = SimpleCompileValue, C = MixedCompoundValue, T = HardwareType, E = IrExpression> =
    Value<S, C, HardwareValueWithImplications<T, E>>;
pub type HardwareValueWithMaybeVersion = HardwareValueWithVersion<Option<ValueVersion>>;

#[derive(Debug, Clone)]
pub struct HardwareValueWithVersion<V = ValueVersion, T = HardwareType, E = IrExpression> {
    pub value: HardwareValue<T, E>,
    pub version: V,
}

#[derive(Debug, Clone)]
pub struct HardwareValueWithImplications<T = HardwareType, E = IrExpression> {
    pub value: HardwareValue<T, E>,
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

impl Implication {
    pub fn new_bool(value: ValueVersion, equal: bool) -> Self {
        Self {
            version: value,
            kind: ImplicationKind::BoolEq(equal),
        }
    }

    pub fn new_int(value: ValueVersion, range: MultiRange<BigInt>) -> Self {
        Self {
            version: value,
            kind: ImplicationKind::IntIn(range),
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

impl<V, E> Typed for HardwareValueWithVersion<V, HardwareType, E> {
    fn ty(&self) -> Type {
        self.value.ty.as_type()
    }
}
