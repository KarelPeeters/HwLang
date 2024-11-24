use num_bigint::{BigInt, BigUint};
use num_traits::Signed as _;
use std::cmp::max;
use std::ops::RangeInclusive;

pub enum Signed {
    Signed,
    Unsigned,
}

pub struct IntRepresentation {
    pub signed: Signed,
    pub bits: BigUint,
}

impl IntRepresentation {
    pub fn for_range(range: RangeInclusive<BigInt>) -> Self {
        if range.is_empty() {
            return IntRepresentation {
                signed: Signed::Unsigned,
                bits: BigUint::ZERO,
            };
        }

        let (start, end) = range.into_inner();

        let (signed, bits) = if start < BigInt::ZERO {
            // signed
            // prevent max value underflow
            let max_value = if end.is_negative() {
                BigInt::ZERO
            } else {
                end
            };
            let max_bits = max(
                1 + (start + 1u32).bits(),
                1 + max_value.bits(),
            );

            (Signed::Signed, max_bits)
        } else {
            // unsigned
            (Signed::Unsigned, end.bits())
        };

        IntRepresentation {
            signed,
            bits: BigUint::from(bits),
        }
    }
}
