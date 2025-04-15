use crate::mid::ir::IrType;
use crate::swrite;
use crate::util::big_int::{BigInt, BigUint};
use itertools::{zip_eq, Itertools};
use std::collections::Bound;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, RangeBounds};

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
    Int(IncRange<BigInt>),
    Tuple(Vec<Type>),
    Array(Box<Type>, BigUint),
    Range,
    // TODO maybe maybe this (optionally) more specific, with ports and implemented interfaces?
    Module,
    // TODO make this (optionally) more specific, with arg and return types
    Function,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum HardwareType {
    Clock,
    Bool,
    Int(ClosedIncRange<BigInt>),
    Tuple(Vec<HardwareType>),
    Array(Box<HardwareType>, BigUint),
}

// TODO rename to min/max? more intuitive than start/end, min and max are clearly inclusive
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct IncRange<T> {
    pub start_inc: Option<T>,
    pub end_inc: Option<T>,
}

// TODO can this represent the empty range? maybe exclusive is better after all...
//   we don't really want empty ranges for int types, but for for loops and slices we do
// TODO switch to exclusive ranges, much more intuitive to program with, especially for arrays and loops
//   match code becomes harder, but that's fine
// TODO transition this to multi-range as the int type
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ClosedIncRange<T> {
    pub start_inc: T,
    pub end_inc: T,
}

pub trait Typed {
    fn ty(&self) -> Type;
}

impl Type {
    pub const UNIT: Type = Type::Tuple(Vec::new());

    pub fn union(&self, other: &Type, allow_compound_subtype: bool) -> Type {
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
                let IncRange {
                    start_inc: a_start,
                    end_inc: a_end,
                } = a;
                let IncRange {
                    start_inc: b_start,
                    end_inc: b_end,
                } = b;

                let start = match (a_start, b_start) {
                    (Some(a_start), Some(b_start)) => Some(a_start.min(b_start).clone()),
                    (None, _) | (_, None) => None,
                };
                let end = match (a_end, b_end) {
                    (Some(a_end), Some(b_end)) => Some(a_end.max(b_end).clone()),
                    (None, _) | (_, None) => None,
                };

                Type::Int(IncRange {
                    start_inc: start,
                    end_inc: end,
                })
            }

            // tuple
            (Type::Tuple(a), Type::Tuple(b)) => {
                if a.len() == b.len() {
                    Type::Tuple(
                        zip_eq(a, b)
                            .map(|(a, b)| {
                                if allow_compound_subtype {
                                    a.union(b, allow_compound_subtype)
                                } else if a == b {
                                    a.clone()
                                } else {
                                    Type::Any
                                }
                            })
                            .collect_vec(),
                    )
                } else {
                    Type::Any
                }
            }
            // array
            // TODO cache this
            (Type::Array(a_inner, a_len), Type::Array(b_inner, b_len)) => {
                if a_len == b_len {
                    let inner = if allow_compound_subtype {
                        a_inner.union(b_inner, allow_compound_subtype)
                    } else if a_inner == b_inner {
                        *a_inner.clone()
                    } else {
                        Type::Any
                    };
                    Type::Array(Box::new(inner), a_len.clone())
                } else {
                    // TODO into list?
                    Type::Any
                }
            }

            // simple mismatches
            (
                Type::Type
                | Type::Clock
                | Type::Bool
                | Type::String
                | Type::Range
                | Type::Module
                | Type::Function
                | Type::Int(_)
                | Type::Tuple(_)
                | Type::Array(_, _),
                Type::Type
                | Type::Clock
                | Type::Bool
                | Type::String
                | Type::Range
                | Type::Module
                | Type::Function
                | Type::Int(_)
                | Type::Tuple(_)
                | Type::Array(_, _),
            ) => Type::Any,
        }
    }

    pub fn contains_type(&self, ty: &Type, allow_compound_subtype: bool) -> bool {
        self == &self.union(ty, allow_compound_subtype)
    }

    // TODO centralize error messages for this, everyone is just doing them manually for now
    pub fn as_hardware_type(&self) -> Option<HardwareType> {
        match self {
            Type::Clock => Some(HardwareType::Clock),
            Type::Bool => Some(HardwareType::Bool),
            Type::Int(range) => range.clone().try_into_closed().map(HardwareType::Int).ok(),
            Type::Tuple(inner) => inner
                .iter()
                .map(Type::as_hardware_type)
                .collect::<Option<_>>()
                .map(HardwareType::Tuple),
            Type::Array(inner, len) => inner
                .as_hardware_type()
                .map(|inner| HardwareType::Array(Box::new(inner), len.clone())),
            Type::Type | Type::Any | Type::Undefined => None,
            Type::String | Type::Range | Type::Module | Type::Function => None,
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
            Type::Tuple(inner) => {
                let inner_str = inner.iter().map(Type::to_diagnostic_string).join(", ");
                format!("({})", inner_str)
            }
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
            HardwareType::Tuple(inner) => Type::Tuple(inner.iter().map(HardwareType::as_type).collect_vec()),
            HardwareType::Array(inner, len) => Type::Array(Box::new(inner.as_type()), len.clone()),
        }
    }

    pub fn as_ir(&self) -> IrType {
        match self {
            HardwareType::Clock => IrType::Bool,
            HardwareType::Bool => IrType::Bool,
            HardwareType::Int(range) => IrType::Int(range.clone()),
            HardwareType::Tuple(inner) => IrType::Tuple(inner.iter().map(HardwareType::as_ir).collect_vec()),
            HardwareType::Array(inner, len) => IrType::Array(Box::new(inner.as_ir()), len.clone()),
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        self.as_type().to_diagnostic_string()
    }
}

impl<T> IncRange<T> {
    pub const OPEN: IncRange<T> = IncRange {
        start_inc: None,
        end_inc: None,
    };

    pub fn try_into_closed(self) -> Result<ClosedIncRange<T>, Self> {
        let IncRange { start_inc, end_inc } = self;

        let start_inc = match start_inc {
            Some(start_inc) => start_inc,
            None => {
                return Err(IncRange {
                    start_inc: None,
                    end_inc,
                })
            }
        };
        let end_inc = match end_inc {
            Some(end_inc) => end_inc,
            None => {
                return Err(IncRange {
                    start_inc: Some(start_inc),
                    end_inc: None,
                })
            }
        };

        Ok(ClosedIncRange { start_inc, end_inc })
    }
}

impl<T> ClosedIncRange<T> {
    pub fn single(value: T) -> ClosedIncRange<T>
    where
        T: Clone,
    {
        ClosedIncRange {
            start_inc: value.clone(),
            end_inc: value,
        }
    }

    pub fn into_range(self) -> IncRange<T> {
        let ClosedIncRange { start_inc, end_inc } = self;
        IncRange {
            start_inc: Some(start_inc),
            end_inc: Some(end_inc),
        }
    }

    pub fn as_ref(&self) -> ClosedIncRange<&T> {
        ClosedIncRange {
            start_inc: &self.start_inc,
            end_inc: &self.end_inc,
        }
    }

    pub fn map(self, mut f: impl FnMut(T) -> T) -> Self {
        let ClosedIncRange { start_inc, end_inc } = self;
        ClosedIncRange {
            start_inc: f(start_inc),
            end_inc: f(end_inc),
        }
    }

    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialOrd,
    {
        let ClosedIncRange { start_inc, end_inc } = self;
        start_inc <= value && value <= end_inc
    }

    pub fn contains_range(&self, other: &ClosedIncRange<T>) -> bool
    where
        T: PartialOrd,
    {
        let ClosedIncRange { start_inc, end_inc } = self;
        let ClosedIncRange {
            start_inc: other_start_inc,
            end_inc: other_end_inc,
        } = other;
        start_inc <= other_start_inc && other_end_inc <= end_inc
    }

    pub fn as_single(&self) -> Option<&T>
    where
        T: Eq,
    {
        if self.start_inc == self.end_inc {
            Some(&self.start_inc)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_
    where
        T: Clone + AddAssign<u32> + Ord,
    {
        let mut next = self.start_inc.clone();
        std::iter::from_fn(move || {
            if next <= self.end_inc {
                let curr = next.clone();
                next += 1;
                Some(curr)
            } else {
                None
            }
        })
    }
}

impl<T: Display> Display for IncRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let IncRange { start_inc, end_inc } = self;
        match (start_inc, end_inc) {
            (None, None) => write!(f, ".."),
            (Some(start_inc), None) => write!(f, "{}..", start_inc),
            (None, Some(end_inc)) => write!(f, "..={}", end_inc),
            (Some(start_inc), Some(end_inc)) => write!(f, "{}..={}", start_inc, end_inc),
        }
    }
}

impl<T: Display> Display for ClosedIncRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ClosedIncRange { start_inc, end_inc } = self;
        write!(f, "{}..={}", start_inc, end_inc)
    }
}

impl<T> RangeBounds<T> for IncRange<T> {
    fn start_bound(&self) -> Bound<&T> {
        match &self.start_inc {
            None => Bound::Unbounded,
            Some(start_inc) => Bound::Included(start_inc),
        }
    }

    fn end_bound(&self) -> Bound<&T> {
        match &self.end_inc {
            None => Bound::Unbounded,
            Some(end_inc) => Bound::Included(end_inc),
        }
    }
}

impl<T> RangeBounds<T> for ClosedIncRange<T> {
    fn start_bound(&self) -> Bound<&T> {
        Bound::Included(&self.start_inc)
    }

    fn end_bound(&self) -> Bound<&T> {
        Bound::Included(&self.end_inc)
    }
}
