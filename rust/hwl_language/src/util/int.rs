use crate::front::types::ClosedIncRange;
use crate::util::big_int::{BigInt, BigUint};
use std::cmp::max;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Signed {
    Signed,
    Unsigned,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IntRepresentation {
    Unsigned { width: u64 },
    Signed { width_1: u64 },
}

#[derive(Debug)]
pub struct InvalidRange;

impl IntRepresentation {
    pub fn for_range(range: &ClosedIncRange<BigInt>) -> Self {
        let ClosedIncRange { start_inc, end_inc } = range;
        Self::for_range_impl(start_inc, end_inc)
    }

    pub fn for_single(value: &BigInt) -> Self {
        Self::for_range_impl(value, value)
    }

    pub fn signed(self) -> Signed {
        match self {
            IntRepresentation::Unsigned { .. } => Signed::Unsigned,
            IntRepresentation::Signed { .. } => Signed::Signed,
        }
    }

    pub fn size_bits(self) -> u64 {
        match self {
            IntRepresentation::Unsigned { width } => width,
            IntRepresentation::Signed { width_1 } => width_1 + 1,
        }
    }

    pub fn range(self) -> ClosedIncRange<BigInt> {
        match self {
            IntRepresentation::Unsigned { width } => ClosedIncRange {
                start_inc: BigInt::ZERO,
                end_inc: BigUint::pow_2_to(&BigUint::from(width)) - 1u8,
            },
            IntRepresentation::Signed { width_1 } => ClosedIncRange {
                start_inc: -BigUint::pow_2_to(&BigUint::from(width_1)),
                end_inc: BigUint::pow_2_to(&BigUint::from(width_1)) - 1u8,
            },
        }
    }

    pub fn value_to_bits(self, value: &BigInt, bits: &mut Vec<bool>) -> Result<(), InvalidRange> {
        let range = self.range();
        if !range.contains(value) {
            return Err(InvalidRange);
        }
        let len_start = bits.len();

        match self {
            IntRepresentation::Unsigned { width } => {
                let value = BigUint::try_from(value).unwrap();
                for i in 0..width {
                    bits.push(value.get_bit_zero_padded(i));
                }
            }
            IntRepresentation::Signed { width_1 } => {
                for i in 0..=width_1 {
                    bits.push(value.get_bit_sign_padded(i));
                }
            }
        }

        assert_eq!(BigUint::from(bits.len()), BigUint::from(len_start) + self.size_bits());
        Ok(())
    }

    pub fn value_from_bits(self, bits: &[bool]) -> Result<BigInt, InvalidRange> {
        if bits.len() as u64 != self.size_bits() {
            return Err(InvalidRange);
        }

        // TODO this is really inefficient, just construct from bits directly
        match self {
            IntRepresentation::Unsigned { width } => {
                let mut result = BigUint::ZERO;

                let width = usize::try_from(width).unwrap();
                for i in 0..width {
                    let i_u64 = u64::try_from(i).unwrap();
                    if bits[i] {
                        result = result.set_bit(i_u64, true);
                    }
                }

                let result = BigInt::from(result);
                assert!(self.range().contains(&result));
                Ok(result)
            }
            IntRepresentation::Signed { width_1 } => {
                let mut result = BigUint::ZERO;

                let width_1 = usize::try_from(width_1).unwrap();
                for i in 0..width_1 {
                    let i_u64 = u64::try_from(i).unwrap();
                    if bits[i] {
                        result = result.set_bit(i_u64, true);
                    }
                }

                let mut result = BigInt::from(result);
                if bits[width_1] {
                    result -= BigUint::pow_2_to(&BigUint::from(width_1));
                }

                assert!(self.range().contains(&result));
                Ok(result)
            }
        }
    }

    fn for_range_impl(start: &BigInt, end: &BigInt) -> Self {
        assert!(start <= end, "Range must be valid, got {start}..={end}");

        // TODO consider switching to zero-width range when there is only a single possible value
        //    this is a bit tricky for signed, and also only a partial optimization
        //    (we could minimize bits even for non-single-value ranges too)

        match BigUint::try_from(start) {
            Ok(_) => {
                // non-negative start => unsigned
                let end = BigUint::try_from(end).unwrap();
                IntRepresentation::Unsigned { width: end.size_bits() }
            }
            Err(_) => {
                // negative start => signed
                let end_non_neg = BigUint::try_from(end).unwrap_or(BigUint::ZERO);
                let width_1 = max(
                    BigUint::try_from(-start - 1u8).unwrap().size_bits(),
                    end_non_neg.size_bits(),
                );
                IntRepresentation::Signed { width_1 }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::front::types::ClosedIncRange;
    use crate::util::big_int::BigInt;
    use crate::util::int::{IntRepresentation, Signed};
    use std::ops::Range;

    #[track_caller]
    fn test_case(range: Range<i64>, signed: Signed, width: u32) {
        let width = width as u64;
        let expected = match signed {
            Signed::Signed => {
                assert!(width > 0);
                IntRepresentation::Signed { width_1: width - 1 }
            }
            Signed::Unsigned => IntRepresentation::Unsigned { width },
        };
        let range = ClosedIncRange {
            start_inc: BigInt::from(range.start),
            end_inc: BigInt::from(range.end) - 1,
        };
        let result = IntRepresentation::for_range(&range);
        println!("range {:?} => {:?}", range, result);
        assert_eq!(expected, result, "mismatch for range {range:?}");
    }

    #[test]
    fn int_range_type_manual() {
        // positive
        test_case(0..1, Signed::Unsigned, 0);
        test_case(0..2, Signed::Unsigned, 1);
        test_case(0..6, Signed::Unsigned, 3);
        test_case(0..7, Signed::Unsigned, 3);
        test_case(0..8, Signed::Unsigned, 3);
        test_case(0..9, Signed::Unsigned, 4);

        // negative
        test_case(-1..0, Signed::Signed, 1);
        test_case(-2..0, Signed::Signed, 2);
        test_case(-6..0, Signed::Signed, 4);
        test_case(-7..0, Signed::Signed, 4);
        test_case(-8..0, Signed::Signed, 4);
        test_case(-9..0, Signed::Signed, 5);

        // mixed
        test_case(-1..1, Signed::Signed, 1);
        test_case(-2..1, Signed::Signed, 2);
        test_case(-1..2, Signed::Signed, 2);
        test_case(-7..8, Signed::Signed, 4);
        test_case(-8..7, Signed::Signed, 4);
        test_case(-8..8, Signed::Signed, 4);
        test_case(-9..8, Signed::Signed, 5);
        test_case(-8..9, Signed::Signed, 5);
    }

    #[test]
    fn int_range_type_automatic() {
        // test that the typical 2s complement ranges behave as expected
        for w in 0u32..32u32 {
            println!("testing w={w}");
            // unsigned
            if w > 1 {
                // for w=0 this case doesn't make any sense
                // for w=1 the bit width should actually get smaller
                test_case(0..2i64.pow(w) - 1, Signed::Unsigned, w);
            }
            test_case(0..2i64.pow(w), Signed::Unsigned, w);
            test_case(0..2i64.pow(w) + 1, Signed::Unsigned, w + 1);

            // singed (only possible if there is room for a sign bit)
            if w > 0 {
                if w > 1 {
                    test_case((-2i64.pow(w - 1) + 1)..2i64.pow(w - 1), Signed::Signed, w);
                }
                test_case(-2i64.pow(w - 1)..(2i64.pow(w - 1) - 1), Signed::Signed, w);
                test_case(-2i64.pow(w - 1)..2i64.pow(w - 1), Signed::Signed, w);
                test_case((-2i64.pow(w - 1) - 1)..2i64.pow(w - 1), Signed::Signed, w + 1);
                test_case(-2i64.pow(w - 1)..(2i64.pow(w - 1) + 1), Signed::Signed, w + 1);
            }
        }
    }
}
