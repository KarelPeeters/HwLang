use crate::swrite;
use crate::util::int::IntRepresentation;
use num_bigint::{BigInt, BigUint};
use num_traits::One;
use std::fmt::Formatter;

// TODO add an arena for types?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    // Higher order type, containing other types (including type itself!).
    Type,
    // Lattice top type (including type!)
    Any,
    // Lattice bottom type
    Undefined,
    Clock,
    Bool,
    String,
    Int(IntRange),
    Array(Box<Type>, BigUint),
    Range,
    Module, // TODO maybe maybe this (optionally) more specific, with ports and implemented interfaces? 
    Function, // TODO make this (optionally) more specific, with arg and return types
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum HardwareType {
    // TODO should this just be booL?
    Clock,
    Bool,
    Int(ClosedIntRange),
    Array(Box<HardwareType>, BigUint),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct IntRange {
    pub start_inc: Option<BigInt>,
    pub end_inc: Option<BigInt>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ClosedIntRange {
    pub start_inc: BigInt,
    pub end_inc: BigInt,
}

impl Type {
    pub fn union(&self, other: &Type) -> Type {
        match (self, other) {
            // top and bottom
            (Type::Any, _) | (_, Type::Any) => Type::Any,
            (Type::Undefined, other) | (other, Type::Undefined) => other.clone(),

            // simple matches
            (Type::Type, Type::Type) => Type::Type,
            (Type::Clock, Type::Clock) => Type::Clock,
            (Type::Bool, Type::Bool) => Type::Bool,
            (Type::String, Type::String) => Type::String,
            (Type::Range, Type::Range) => Type::Range,
            (Type::Module, Type::Module) => Type::Module,
            (Type::Function, Type::Function) => Type::Function,

            // integer
            (Type::Int(a), Type::Int(b)) => {
                let IntRange { start_inc: a_start, end_inc: a_end } = a;
                let IntRange { start_inc: b_start, end_inc: b_end } = b;

                let start = match (a_start, b_start) {
                    (Some(a_start), Some(b_start)) => Some(a_start.min(b_start).clone()),
                    (None, _) | (_, None) => None,
                };
                let end = match (a_end, b_end) {
                    (Some(a_end), Some(b_end)) => Some(a_end.max(b_end).clone()),
                    (None, _) | (_, None) => None,
                };

                Type::Int(IntRange { start_inc: start, end_inc: end })
            }

            // array
            (Type::Array(a_inner, a_len), Type::Array(b_inner, b_len)) => {
                if a_len == b_len {
                    let inner = a_inner.union(b_inner);
                    Type::Array(Box::new(inner), a_len.clone())
                } else {
                    // TODO into list?
                    Type::Any
                }
            }

            // simple mismatches
            (
                Type::Type | Type::Clock | Type::Bool | Type::String | Type::Range | Type::Module | Type::Function | Type::Int(_) | Type::Array(_, _),
                Type::Type | Type::Clock | Type::Bool | Type::String | Type::Range | Type::Module | Type::Function | Type::Int(_) | Type::Array(_, _),
            ) => Type::Any,
        }
    }

    pub fn contains_type(&self, ty: &Type) -> bool {
        self == &self.union(ty)
    }

    pub fn as_hardware_type(&self) -> Option<HardwareType> {
        match self {
            Type::Clock =>
                Some(HardwareType::Clock),
            Type::Bool =>
                Some(HardwareType::Bool),
            Type::Int(range) =>
                range.clone().try_into_closed().map(HardwareType::Int),
            Type::Array(inner, len) =>
                inner.as_hardware_type().map(|inner| HardwareType::Array(Box::new(inner), len.clone())),
            Type::Type | Type::Any | Type::Undefined => None,
            Type::String | Type::Range | Type::Module | Type::Function =>
                None,
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        match self {
            Type::Type => "type".to_string(),
            Type::Any => "any".to_string(),
            Type::Undefined => "undefined".to_string(),

            Type::Clock => "clock".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Int(range) => format!("int({})", range),
            Type::Array(first_inner, first_len) => {
                let mut dims = String::new();

                swrite!(&mut dims, "{}", first_len);
                let mut inner = first_inner;
                while let Type::Array(curr_inner, curr_len) = &**inner {
                    swrite!(&mut dims, ", {}", curr_len);
                    inner = curr_inner;
                }

                let inner_str = inner.to_diagnostic_string();
                format!("{inner_str}[{dims}]")
            }
            Type::Range => "range".to_string(),
            Type::Module => "module".to_string(),
            Type::Function => "function".to_string(),
        }
    }
}

impl HardwareType {
    pub fn as_type(&self) -> Type {
        match self {
            HardwareType::Clock => Type::Clock,
            HardwareType::Bool => Type::Bool,
            HardwareType::Int(range) => Type::Int(range.clone().into_range()),
            HardwareType::Array(inner, len) => Type::Array(Box::new(inner.as_type()), len.clone()),
        }
    }

    pub fn bit_width(&self) -> BigUint {
        match self {
            HardwareType::Clock => BigUint::one(),
            HardwareType::Bool => BigUint::one(),
            HardwareType::Int(range) => {
                let ClosedIntRange { start_inc, end_inc } = range;
                IntRepresentation::for_range(start_inc.clone()..=end_inc.clone()).bits
            }
            HardwareType::Array(inner, len) => inner.bit_width() * len,
        }
    }
}

impl IntRange {
    pub fn try_into_closed(self) -> Option<ClosedIntRange> {
        let IntRange { start_inc, end_inc } = self;
        Some(ClosedIntRange {
            start_inc: start_inc?,
            end_inc: end_inc?,
        })
    }
}

impl ClosedIntRange {
    pub fn into_range(self) -> IntRange {
        let ClosedIntRange { start_inc, end_inc } = self;
        IntRange {
            start_inc: Some(start_inc),
            end_inc: Some(end_inc),
        }
    }
}

impl std::fmt::Display for IntRange {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let IntRange { start_inc, end_inc } = self;
        match (start_inc, end_inc) {
            (None, None) => write!(f, ".."),
            (Some(start_inc), None) => write!(f, "{}..", start_inc),
            (None, Some(end_inc)) => write!(f, "..={}", end_inc),
            (Some(start_inc), Some(end_inc)) => write!(f, "{}..={}", start_inc, end_inc),
        }
    }
}