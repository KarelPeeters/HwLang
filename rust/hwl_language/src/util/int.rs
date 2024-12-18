use crate::new::types::ClosedIncRange;
use num_bigint::{BigInt, BigUint};
use num_traits::Signed as _;
use std::cmp::max;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Signed {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IntRepresentation {
    /// Signedness, whether the number can be negative.
    /// If [Signed::Signed], [width] is at least 1.
    pub signed: Signed,
    /// Total width including the signed bit if any.
    pub width: BigUint,
}

impl IntRepresentation {
    pub fn for_range(range: &ClosedIncRange<BigInt>) -> Self {
        let ClosedIncRange {
            start_inc: start,
            end_inc: end,
        } = range;
        assert!(start <= end, "Range must be valid, got {start}..={end}");

        if start == end {
            return IntRepresentation {
                signed: Signed::Unsigned,
                width: BigUint::ZERO,
            };
        }

        let (signed, bits) = if start.is_negative() {
            // signed
            // prevent max value underflow
            let max_value = if end.is_negative() { &BigInt::ZERO } else { end };
            let max_bits = max(1 + (start + 1u32).bits(), 1 + max_value.bits());

            (Signed::Signed, max_bits)
        } else {
            // unsigned
            (Signed::Unsigned, end.bits())
        };

        IntRepresentation {
            signed,
            width: BigUint::from(bits),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::new::types::ClosedIncRange;
    use crate::util::int::{IntRepresentation, Signed};
    use num_bigint::BigInt;
    use std::ops::Range;

    #[track_caller]
    fn test_case(range: Range<i64>, signed: Signed, width: u32) {
        let expected = IntRepresentation {
            signed,
            width: width.into(),
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
