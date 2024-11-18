use crate::new::misc::{MaybeUnchecked, Unchecked};
use crate::util::Never;
use num_bigint::{BigInt, BigUint};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type<U = Unchecked> {
    Unchecked(U),
    Bool,
    String,
    Int(RangeInfo<U>),
    Array(Box<Type<U>>, MaybeUnchecked<BigUint, U>),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RangeInfo<U> {
    pub start_inc: Option<MaybeUnchecked<BigInt, U>>,
    pub end_inc: Option<MaybeUnchecked<BigInt, U>>,
}

impl Type<Unchecked> {
    pub fn require_checked(self) -> Result<Type<Never>, Unchecked> {
        let r = match self {
            Type::Unchecked(u) => return Err(u),
            Type::Bool => Type::Bool,
            Type::String => Type::String,
            Type::Int(range) =>
                Type::Int(RangeInfo {
                    start_inc: range.start_inc.map(|v| v.require_checked()).transpose()?,
                    end_inc: range.end_inc.map(|v| v.require_checked()).transpose()?,
                }),
            Type::Array(t, len) =>
                Type::Array(Box::new(t.require_checked()?), len.require_checked()?),
        };
        Ok(r)
    }
}