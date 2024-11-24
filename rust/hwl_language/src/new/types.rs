use crate::data::diagnostic::ErrorGuaranteed;
use crate::new::value::CompileValue;
use crate::swrite;
use crate::util::int::IntRepresentation;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    Error(ErrorGuaranteed),
    Clock,
    Bool,
    String,
    Int(IntRange),
    Array(Box<Type>, BigUint),
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
    // TODO return detailed justification on failure
    pub fn contains_value(&self, value: &CompileValue) -> Result<bool, ErrorGuaranteed> {
        let result = match (self, value) {
            (&Type::Error(e), _) => return Err(e),
            (Type::Bool, CompileValue::Bool(_)) => true,
            (Type::String, CompileValue::String(_)) => true,
            (Type::Int(range), CompileValue::Int(value)) => {
                let IntRange { start_inc, end_inc } = range;

                let start_ok = start_inc.as_ref()
                    .map_or(true, |start_inc| start_inc <= value);
                let end_ok = end_inc.as_ref()
                    .map_or(true, |end_inc| value <= end_inc);

                start_ok && end_ok
            }
            (Type::Array(inner, len), CompileValue::Array(values)) => {
                let mut r = len == &BigUint::from(values.len());
                for v in values {
                    r &= inner.contains_value(v)?;
                }
                r
            }

            // TODO maybe clock should accept bool values?
            (Type::Clock, _) => false,

            (
                Type::Bool | Type::String | Type::Int(_) | Type::Array(_, _),
                CompileValue::Bool(_) | CompileValue::String(_) | CompileValue::Int(_) | CompileValue::Array(_)
            ) => false,
        };
        Ok(result)
    }

    pub fn hardware_bit_width(&self) -> Result<Option<BigUint>, ErrorGuaranteed> {
        match self {
            &Type::Error(e) => Err(e),
            Type::Clock => Ok(Some(BigUint::one())),
            Type::Bool => Ok(Some(BigUint::one())),
            Type::String => Ok(None),
            Type::Int(range) => {
                let IntRange { start_inc, end_inc } = range;
                match (start_inc, end_inc) {
                    (Some(start_inc), Some(end_inc)) =>
                        Ok(Some(IntRepresentation::for_range(start_inc.clone()..=end_inc.clone()).bits)),
                    _ =>
                        Ok(None),
                }
            }
            Type::Array(inner, len) =>
                inner.hardware_bit_width()
                    .map(|inner_width| {
                        inner_width.map(|inner_width| inner_width * len)
                    }),
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        match self {
            Type::Error(_) => "error".to_string(),
            Type::Clock => "clock".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Int(range) => {
                let IntRange { start_inc, end_inc } = range;

                // TODO is this back-lifting a good idea or not?
                match (start_inc, end_inc) {
                    (None, None) => "int".to_string(),
                    (Some(start_inc), None) => {
                        if start_inc.is_zero() {
                            "uint".to_string()
                        } else {
                            format!("int_range({}..)", start_inc)
                        }
                    },
                    (None, Some(end_inc)) => format!("int_range(..={})", end_inc),
                    (Some(start_inc), Some(end_inc)) => format!("int_range({}..={})", start_inc, end_inc),
                }
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