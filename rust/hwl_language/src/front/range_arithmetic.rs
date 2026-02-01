//! Module to compute output ranges of expressions.
//!
//! Proofs of the validity of these range bounds can be found in `int_range_proofs.py`.

use crate::util::big_int::{BigInt, BigUint};
use crate::util::range::{ClosedNonEmptyRange, Range};
use crate::util::range_multi::{AnyMultiRange, ClosedNonEmptyMultiRange, MultiRange};
use itertools::{Itertools, chain};
use std::cmp::{max, min};

pub fn range_unary_neg(a: ClosedNonEmptyRange<&BigInt>) -> ClosedNonEmptyRange<BigInt> {
    let (a_min, a_max) = range_to_min_max(a);
    range_from_min_max(-a_max, -a_min)
}

pub fn multi_range_unary_neg(a: &ClosedNonEmptyMultiRange<BigInt>) -> ClosedNonEmptyMultiRange<BigInt> {
    wrap_multi_unary(a, range_unary_neg)
}

pub fn range_unary_abs(a: ClosedNonEmptyRange<&BigInt>) -> ClosedNonEmptyRange<BigInt> {
    let (a_min, a_max) = range_to_min_max(a);
    if a_max < BigInt::ZERO {
        range_from_min_max(-a_max, -a_min)
    } else {
        range_from_min_max(max(a_min.clone(), BigInt::ZERO), max(-a_min, a_max.clone()))
    }
}

pub fn multi_range_unary_abs(a: &ClosedNonEmptyMultiRange<BigInt>) -> ClosedNonEmptyMultiRange<BigInt> {
    wrap_multi_unary(a, range_unary_abs)
}

pub fn range_binary_add(
    a: ClosedNonEmptyRange<&BigInt>,
    b: ClosedNonEmptyRange<&BigInt>,
) -> ClosedNonEmptyRange<BigInt> {
    let (a_min, a_max) = range_to_min_max(a);
    let (b_min, b_max) = range_to_min_max(b);
    range_from_min_max(a_min + b_min, a_max + b_max)
}

pub fn multi_range_binary_add(
    a: &ClosedNonEmptyMultiRange<BigInt>,
    b: &ClosedNonEmptyMultiRange<BigInt>,
) -> ClosedNonEmptyMultiRange<BigInt> {
    wrap_multi_binary(a, b, range_binary_add)
}

pub fn range_binary_sub(
    a: ClosedNonEmptyRange<&BigInt>,
    b: ClosedNonEmptyRange<&BigInt>,
) -> ClosedNonEmptyRange<BigInt> {
    let (a_min, a_max) = range_to_min_max(a);
    let (b_min, b_max) = range_to_min_max(b);
    range_from_min_max(a_min - b_max, a_max - b_min)
}

pub fn multi_range_binary_sub(
    a: &ClosedNonEmptyMultiRange<BigInt>,
    b: &ClosedNonEmptyMultiRange<BigInt>,
) -> ClosedNonEmptyMultiRange<BigInt> {
    wrap_multi_binary(a, b, range_binary_sub)
}

pub fn range_binary_mul(
    a: ClosedNonEmptyRange<&BigInt>,
    b: ClosedNonEmptyRange<&BigInt>,
) -> ClosedNonEmptyRange<BigInt> {
    let (a_min, a_max) = range_to_min_max(a);
    let (b_min, b_max) = range_to_min_max(b);
    let extremes = [a_min * b_min, a_min * &b_max, &a_max * b_min, a_max * b_max];
    range_from_min_max(
        extremes.iter().min().unwrap().clone(),
        extremes.iter().max().unwrap().clone(),
    )
}

pub fn multi_range_binary_mul(
    a: &ClosedNonEmptyMultiRange<BigInt>,
    b: &ClosedNonEmptyMultiRange<BigInt>,
) -> ClosedNonEmptyMultiRange<BigInt> {
    wrap_multi_binary(a, b, range_binary_mul)
}

pub fn range_binary_div(
    a: ClosedNonEmptyRange<&BigInt>,
    b: ClosedNonEmptyRange<&BigInt>,
) -> Option<ClosedNonEmptyRange<BigInt>> {
    // check for potential division by zero
    if b.contains(&&BigInt::ZERO) {
        return None;
    }

    let (a_min, a_max) = range_to_min_max(a);
    let (b_min, b_max) = range_to_min_max(b);

    // TODO these ranges can probably be tightened
    let right_positive = b_min > &BigInt::ZERO;
    if right_positive {
        Some(range_from_min_max(
            min(a_min.div_floor(&b_max).unwrap(), a_min.div_floor(b_min).unwrap()),
            max(a_max.div_floor(&b_max).unwrap(), a_max.div_floor(b_min).unwrap()),
        ))
    } else {
        Some(range_from_min_max(
            min(a_max.div_floor(&b_max).unwrap(), a_max.div_floor(b_min).unwrap()),
            max(a_min.div_floor(&b_max).unwrap(), a_min.div_floor(b_min).unwrap()),
        ))
    }
}

pub fn multi_range_binary_div(
    a: &ClosedNonEmptyMultiRange<BigInt>,
    b: &ClosedNonEmptyMultiRange<BigInt>,
) -> Option<ClosedNonEmptyMultiRange<BigInt>> {
    if b.contains(&BigInt::ZERO) {
        return None;
    }
    Some(wrap_multi_binary(a, b, |r_a, r_b| {
        range_binary_div(r_a, r_b).expect("already checked for division by zero")
    }))
}

pub fn range_binary_mod(
    _a: ClosedNonEmptyRange<&BigInt>,
    b: ClosedNonEmptyRange<&BigInt>,
) -> Option<ClosedNonEmptyRange<BigInt>> {
    // check for potential division by zero
    if b.contains(&&BigInt::ZERO) {
        return None;
    }

    let (b_min, b_max) = range_to_min_max(b);

    // TODO this could be tightened depending on the range of `a`
    let right_positive = b_min > &BigInt::ZERO;
    if right_positive {
        Some(range_from_min_max(BigInt::ZERO, b_max - 1))
    } else {
        Some(range_from_min_max(b_min + 1, BigInt::ZERO))
    }
}

pub fn multi_range_binary_mod(
    a: &ClosedNonEmptyMultiRange<BigInt>,
    b: &ClosedNonEmptyMultiRange<BigInt>,
) -> Option<ClosedNonEmptyMultiRange<BigInt>> {
    if b.contains(&BigInt::ZERO) {
        return None;
    }
    Some(wrap_multi_binary(a, b, |r_a, r_b| {
        range_binary_mod(r_a, r_b).expect("already checked for division by zero")
    }))
}

pub fn range_binary_pow(
    a: ClosedNonEmptyRange<&BigInt>,
    b: ClosedNonEmptyRange<&BigUint>,
) -> Option<ClosedNonEmptyRange<BigInt>> {
    let (a_min, a_max) = range_to_min_max(a);
    let (b_min, b_max) = uint_range_to_min_max(b);

    // check for potential 0**0
    if a.contains(&&BigInt::ZERO) && b.contains(&&BigUint::ZERO) {
        return None;
    }

    // calculate and combine different cases
    let cases_basic = [a_min.pow(b_min), a_min.pow(&b_max), a_max.pow(&b_max)];
    let case_even_odd = if b_min < &b_max {
        let b_max_m1 = BigUint::try_from(b_max - 1).expect("start < end and 0 <= start, so 0 < end");
        Some(a_min.pow(&b_max_m1))
    } else {
        None
    };
    let case_zero = if a_min <= &BigInt::ZERO && BigInt::ZERO < a_max {
        Some(BigInt::ZERO)
    } else {
        None
    };

    let cases = chain!(cases_basic, case_even_odd, case_zero);
    let (result_min, result_max) = cases.minmax().into_option().unwrap();

    Some(range_from_min_max(result_min, result_max))
}

pub fn multi_range_binary_pow(
    a: &ClosedNonEmptyMultiRange<BigInt>,
    b: &ClosedNonEmptyMultiRange<BigUint>,
) -> Option<ClosedNonEmptyMultiRange<BigInt>> {
    // check for potential 0**0
    if a.contains(&BigInt::ZERO) && b.contains(&BigUint::ZERO) {
        return None;
    }
    Some(wrap_multi_binary(a, b, |r_a, r_b| {
        range_binary_pow(r_a, r_b).expect("already checked for 0**0")
    }))
}

// multi-range versions just apply the single-range versions to each possible combination
// TODO is there an easy way to optimize that?
// TODO many of these should be more sparse, eg. power operations and multiplications often create gaps
fn wrap_multi_unary<A: Ord, R: Ord + Clone>(
    a: &ClosedNonEmptyMultiRange<A>,
    f: impl Fn(ClosedNonEmptyRange<&A>) -> ClosedNonEmptyRange<R>,
) -> ClosedNonEmptyMultiRange<R> {
    let mut result = MultiRange::EMPTY;
    for r in a.ranges() {
        result = result.union(&MultiRange::from(Range::from(f(r))));
    }
    ClosedNonEmptyMultiRange::try_from(result).unwrap()
}

fn wrap_multi_binary<A: Ord, B: Ord, R: Ord + Clone>(
    a: &ClosedNonEmptyMultiRange<A>,
    b: &ClosedNonEmptyMultiRange<B>,
    f: impl Fn(ClosedNonEmptyRange<&A>, ClosedNonEmptyRange<&B>) -> ClosedNonEmptyRange<R>,
) -> ClosedNonEmptyMultiRange<R> {
    let mut result = MultiRange::EMPTY;
    for r_a in a.ranges() {
        for r_b in b.ranges() {
            result = result.union(&MultiRange::from(Range::from(f(r_a, r_b))));
        }
    }
    ClosedNonEmptyMultiRange::try_from(result).unwrap()
}

// for arithmetic, reasoning about min/max(inclusive) is easier than start/end(exclusive)
fn range_to_min_max(a: ClosedNonEmptyRange<&BigInt>) -> (&BigInt, BigInt) {
    let ClosedNonEmptyRange { start, end } = a;
    (start, end - 1)
}

fn uint_range_to_min_max(a: ClosedNonEmptyRange<&BigUint>) -> (&BigUint, BigUint) {
    let ClosedNonEmptyRange { start, end } = a;
    (
        start,
        BigUint::try_from(end - 1).expect("non-empty, so max will not be negative"),
    )
}

fn range_from_min_max(min: BigInt, max: BigInt) -> ClosedNonEmptyRange<BigInt> {
    ClosedNonEmptyRange {
        start: min,
        end: max + 1,
    }
}
