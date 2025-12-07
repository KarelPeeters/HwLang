//! Module to compute output ranges of expressions.
//!
//! Proofs validness of these range bounds integer ranges can be found in `int_range_proofs.py`.

use crate::front::range::ClosedIncRange;
use crate::util::big_int::{BigInt, BigUint};
use std::cmp::{max, min};

pub fn range_unary_neg(a: ClosedIncRange<&BigInt>) -> ClosedIncRange<BigInt> {
    ClosedIncRange {
        start_inc: -a.end_inc,
        end_inc: -a.start_inc,
    }
}

pub fn range_unary_abs(a: ClosedIncRange<&BigInt>) -> ClosedIncRange<BigInt> {
    if a.end_inc < &BigInt::ZERO {
        ClosedIncRange {
            start_inc: -a.end_inc,
            end_inc: -a.start_inc,
        }
    } else {
        ClosedIncRange {
            start_inc: max(a.start_inc.clone(), BigInt::ZERO),
            end_inc: max(-a.start_inc, a.end_inc.clone()),
        }
    }
}

pub fn range_binary_add(a: ClosedIncRange<&BigInt>, b: ClosedIncRange<&BigInt>) -> ClosedIncRange<BigInt> {
    ClosedIncRange {
        start_inc: a.start_inc + b.start_inc,
        end_inc: a.end_inc + b.end_inc,
    }
}

pub fn range_binary_sub(a: ClosedIncRange<&BigInt>, b: ClosedIncRange<&BigInt>) -> ClosedIncRange<BigInt> {
    ClosedIncRange {
        start_inc: a.start_inc - b.end_inc,
        end_inc: a.end_inc - b.start_inc,
    }
}

pub fn range_binary_mul(a: ClosedIncRange<&BigInt>, b: ClosedIncRange<&BigInt>) -> ClosedIncRange<BigInt> {
    let extremes = [
        a.start_inc * b.start_inc,
        a.start_inc * b.end_inc,
        a.end_inc * b.start_inc,
        a.end_inc * b.end_inc,
    ];
    ClosedIncRange {
        start_inc: extremes.iter().min().unwrap().clone(),
        end_inc: extremes.iter().max().unwrap().clone(),
    }
}

pub fn range_binary_div(a: ClosedIncRange<&BigInt>, b: ClosedIncRange<&BigInt>) -> Option<ClosedIncRange<BigInt>> {
    // check for potential division by zero
    if b.contains(&&BigInt::ZERO) {
        return None;
    }

    // TODO these ranges can probably be tightened
    let right_positive = b.start_inc > &BigInt::ZERO;
    if right_positive {
        Some(ClosedIncRange {
            start_inc: min(
                a.start_inc.div_floor(b.end_inc).unwrap(),
                a.start_inc.div_floor(b.start_inc).unwrap(),
            ),
            end_inc: max(
                a.end_inc.div_floor(b.end_inc).unwrap(),
                a.end_inc.div_floor(b.start_inc).unwrap(),
            ),
        })
    } else {
        Some(ClosedIncRange {
            start_inc: min(
                a.end_inc.div_floor(b.end_inc).unwrap(),
                a.end_inc.div_floor(b.start_inc).unwrap(),
            ),
            end_inc: max(
                a.start_inc.div_floor(b.end_inc).unwrap(),
                a.start_inc.div_floor(b.start_inc).unwrap(),
            ),
        })
    }
}

pub fn range_binary_mod(_a: ClosedIncRange<&BigInt>, b: ClosedIncRange<&BigInt>) -> Option<ClosedIncRange<BigInt>> {
    // check for potential division by zero
    if b.contains(&&BigInt::ZERO) {
        return None;
    }

    // Note: Python modulo's range only depends on the divisor `b`, not the dividend `a`
    // The result has the same sign as `b` (or is zero)
    let right_positive = b.start_inc > &BigInt::ZERO;
    if right_positive {
        Some(ClosedIncRange {
            start_inc: BigInt::ZERO,
            end_inc: b.end_inc - 1,
        })
    } else {
        Some(ClosedIncRange {
            start_inc: b.start_inc + 1,
            end_inc: BigInt::ZERO,
        })
    }
}

pub fn range_binary_pow(a: ClosedIncRange<&BigInt>, b: ClosedIncRange<&BigUint>) -> Option<ClosedIncRange<BigInt>> {
    // check for potential 0**0
    if a.contains(&&BigInt::ZERO) && b.contains(&&BigUint::ZERO) {
        return None;
    }

    let mut result_min = min(a.start_inc.clone().pow(b.start_inc), a.start_inc.clone().pow(b.end_inc));
    let mut result_max = max(a.start_inc.clone().pow(b.end_inc), a.end_inc.clone().pow(b.end_inc));

    // If base is negative, even/odd powers can cause extremes.
    // To guard this, try the second highest exponent too if it exists.
    if let Ok(b_end_inc_1) = BigUint::try_from(b.end_inc - 1) {
        result_min = min(result_min, a.start_inc.clone().pow(&b_end_inc_1));
        result_max = max(result_max, a.start_inc.clone().pow(&b_end_inc_1));
    }

    Some(ClosedIncRange {
        start_inc: result_min,
        end_inc: result_max,
    })
}
