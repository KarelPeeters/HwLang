use crate::front::flow::ValueVersion;
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{HardwareValue, MixedCompoundValue, SimpleCompileValue, Value};
use crate::util::big_int::BigInt;
use itertools::Itertools;
use std::fmt::{Debug, Formatter};

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
    IntIn(IncRangeMulti),
}

#[derive(Debug, Copy, Clone)]
pub enum ImplicationIntOp {
    Neq,
    Lt,
    Gt,
}

pub type ValueWithVersion<S = SimpleCompileValue, C = MixedCompoundValue, T = HardwareType> =
    Value<S, C, HardwareValueWithVersion<ValueVersion, T>>;
pub type ValueWithImplications<S = SimpleCompileValue, C = MixedCompoundValue, T = HardwareType> =
    Value<S, C, HardwareValueWithImplications<T>>;
pub type HardwareValueWithMaybeVersion = HardwareValueWithVersion<Option<ValueVersion>>;

#[derive(Debug, Clone)]
pub struct HardwareValueWithVersion<V = ValueVersion, T = HardwareType> {
    pub value: HardwareValue<T>,
    pub version: V,
}

#[derive(Debug, Clone)]
pub struct HardwareValueWithImplications<T = HardwareType> {
    pub value: HardwareValue<T>,
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

pub fn join_implications(branch_implications: &[Vec<Implication>]) -> Vec<Implication> {
    // TODO do something more interesting here, this placeholder implementation is correct but a missed opportunity
    let _ = branch_implications;
    vec![]
}

impl Implication {
    pub fn new_bool(value: ValueVersion, equal: bool) -> Self {
        Self {
            version: value,
            kind: ImplicationKind::BoolEq(equal),
        }
    }

    pub fn new_int(value: ValueVersion, range: IncRangeMulti) -> Self {
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

// TODO move somewhere else, together with the rest of the range code
// TODO change back to closed range, that's the only use case for now
#[derive(Eq, PartialEq, Clone)]
pub struct IncRangeMulti {
    // TODO use SmallVec
    ranges: Vec<IncRange<BigInt>>,
}

#[derive(Eq, PartialEq, Clone)]
pub struct ClosedIncRangeMulti {
    inner: IncRangeMulti,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
enum Bound<'a> {
    OpenBottom,
    Mid(&'a BigInt),
    OpenTop,
}

impl Debug for IncRangeMulti {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let IncRangeMulti { ranges } = self;
        write!(f, "IncRangeMulti({})", ranges.iter().format(", "))
    }
}

impl Debug for ClosedIncRangeMulti {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl IncRangeMulti {
    pub const EMPTY: IncRangeMulti = IncRangeMulti { ranges: vec![] };

    #[must_use]
    pub fn single(value: BigInt) -> Self {
        IncRangeMulti::from_range(IncRange::single(value))
    }

    #[must_use]
    pub fn from_range(range: IncRange<BigInt>) -> Self {
        // TODO assert range value
        IncRangeMulti { ranges: vec![range] }
    }

    #[must_use]
    pub fn contains(&self, x: &BigInt) -> bool {
        // TODO speed up with binary search?
        self.ranges.iter().any(|r| r.contains(x))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    #[must_use]
    pub fn enclosing_range(&self) -> Option<IncRange<BigInt>> {
        Some(IncRange {
            start_inc: self.ranges.first()?.start_inc.clone(),
            end_inc: self.ranges.last()?.end_inc.clone(),
        })
    }

    #[must_use]
    pub fn complement(&self) -> IncRangeMulti {
        let mut ranges = Vec::new();

        if self.ranges.is_empty() {
            ranges.push(IncRange {
                start_inc: None,
                end_inc: None,
            });
        } else {
            if let Some(first_start) = &self.ranges[0].start_inc {
                ranges.push(IncRange {
                    start_inc: None,
                    end_inc: Some(first_start - 1),
                });
            }

            for (curr, next) in self.ranges.iter().tuple_windows() {
                let start = curr.end_inc.as_ref().unwrap() + 1;
                let end = next.start_inc.as_ref().unwrap() - 1;
                if start <= end {
                    ranges.push(IncRange {
                        start_inc: Some(start),
                        end_inc: Some(end),
                    });
                }
            }

            if let Some(last_end) = &self.ranges.last().unwrap().end_inc {
                ranges.push(IncRange {
                    start_inc: Some(last_end + 1),
                    end_inc: None,
                });
            }
        }

        let result = IncRangeMulti { ranges };
        if cfg!(debug_assertions) {
            result.assert_valid();
        }
        result
    }

    #[must_use]
    pub fn union(&self, other: &IncRangeMulti) -> IncRangeMulti {
        let mut iter_self = self.ranges.iter().peekable();
        let mut iter_other = other.ranges.iter().peekable();

        let mut result = Vec::new();
        let mut curr: Option<IncRange<BigInt>> = None;

        loop {
            let next = match (iter_self.peek(), iter_other.peek()) {
                (Some(l), Some(r)) => {
                    let l_start = l.start_inc.as_ref().map_or(Bound::OpenBottom, Bound::Mid);
                    let r_start = r.start_inc.as_ref().map_or(Bound::OpenBottom, Bound::Mid);
                    if l_start <= r_start {
                        iter_self.next().unwrap()
                    } else {
                        iter_other.next().unwrap()
                    }
                }
                (Some(_), None) => iter_self.next().unwrap(),
                (None, Some(_)) => iter_other.next().unwrap(),
                (None, None) => break,
            };

            match &mut curr {
                None => {
                    curr = Some(next.clone());
                }
                Some(curr) => {
                    let curr_end = curr.end_inc.as_ref();
                    let next_start = next.start_inc.as_ref();

                    let merge = match (curr_end, next_start) {
                        (None, _) | (_, None) => true,
                        (Some(curr_end), Some(next_start)) => curr_end >= &(next_start - 1),
                    };

                    if merge {
                        let curr_end = curr.end_inc.as_ref().map_or(Bound::OpenTop, Bound::Mid);
                        let next_end = next.end_inc.as_ref().map_or(Bound::OpenTop, Bound::Mid);

                        if next_end > curr_end {
                            curr.end_inc = next.end_inc.clone();
                        }
                    } else {
                        result.push(std::mem::replace(curr, next.clone()));
                    }
                }
            }
        }

        if let Some(curr) = curr {
            result.push(curr);
        }

        let result = IncRangeMulti { ranges: result };
        if cfg!(debug_assertions) {
            result.assert_valid();
        }
        result
    }

    #[must_use]
    pub fn intersect(&self, other: &IncRangeMulti) -> IncRangeMulti {
        self.complement().union(&other.complement()).complement()
    }

    #[must_use]
    pub fn subtract(&self, range: &IncRangeMulti) -> IncRangeMulti {
        self.complement().union(range).complement()
    }

    fn assert_valid(&self) {
        for i in 0..self.ranges.len() {
            let curr = &self.ranges[i];
            let next = self.ranges.get(i + 1);

            if let (Some(start_inc), Some(end_inc)) = (&curr.start_inc, &curr.end_inc) {
                assert!(start_inc <= end_inc);
            }

            if let Some(next) = next {
                let curr_end = curr.end_inc.as_ref().unwrap();
                let next_start = next.start_inc.as_ref().unwrap();
                assert!(curr_end < next_start);
            }
        }
    }
}

impl ClosedIncRangeMulti {
    pub const EMPTY: ClosedIncRangeMulti = ClosedIncRangeMulti {
        inner: IncRangeMulti::EMPTY,
    };

    pub fn from_range(range: ClosedIncRange<BigInt>) -> Self {
        ClosedIncRangeMulti {
            inner: IncRangeMulti::from_range(range.into_range()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter_ranges(&self) -> impl Iterator<Item = ClosedIncRange<BigInt>> {
        self.inner.ranges.iter().map(|r| {
            r.clone()
                .try_into_closed()
                .expect("started from closed, all ranges should be closed")
        })
    }

    pub fn enclosing_range(&self) -> Option<ClosedIncRange<BigInt>> {
        Some(
            self.inner
                .enclosing_range()?
                .try_into_closed()
                .expect("started from closed, result should be closed again"),
        )
    }

    #[must_use]
    pub fn subtract(&mut self, range: &IncRangeMulti) -> ClosedIncRangeMulti {
        // subtracting from a closed range will always yield a closed (or empty) range
        ClosedIncRangeMulti {
            inner: self.inner.subtract(range),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::front::implication::IncRangeMulti;
    use crate::front::types::IncRange;
    use crate::util::big_int::BigInt;
    use itertools::Itertools;
    use std::ops::RangeInclusive;

    #[test]
    fn test_remove_nothing() {
        #[allow(clippy::reversed_empty_ranges)]
        test_remove_util(&[Some(0)..=Some(7)], Some(10)..=Some(9), &[Some(0)..=Some(7)]);
        test_remove_util(&[Some(0)..=Some(7)], Some(10)..=Some(10), &[Some(0)..=Some(7)]);
        test_remove_util(&[Some(0)..=Some(7)], Some(8)..=Some(10), &[Some(0)..=Some(7)]);
        test_remove_util(&[Some(0)..=Some(7)], Some(-2)..=Some(-1), &[Some(0)..=Some(7)]);
    }

    #[test]
    fn test_remove_full() {
        #![allow(clippy::reversed_empty_ranges)]
        test_remove_util::<_, _, i8>(&[0..=7], 0..=7, &[]);
        test_remove_util::<_, _, i8>(&[0..=7], -2..=10, &[]);

        test_remove_util(&[2..=4, 6..=8, 10..=12], 1..=9, &[10..=12]);
    }

    #[test]
    fn test_remove_split_edge() {
        test_remove_util(&[0..=7], 4..=10, &[0..=3]);
        test_remove_util(&[0..=7], -2..=3, &[4..=7]);
        test_remove_util(&[0..=3, 6..=9], 2..=8, &[0..=1, 9..=9]);
    }

    #[test]
    fn test_remove_split_inside() {
        test_remove_util(&[0..=7], 2..=4, &[0..=1, 5..=7]);
        test_remove_util(&[0..=3, 6..=9], 7..=8, &[0..=3, 6..=6, 9..=9]);
    }

    #[test]
    fn test_remove_split_edge_and_full() {
        test_remove_util(&[0..=3, 5..=7, 9..=11], 2..=10, &[0..=1, 11..=11]);
    }

    #[test]
    fn test_complement() {
        test_complement_util::<i8, Option<i8>>(&[], &[None..=None]);

        test_complement_util(&[None..=Some(8)], &[Some(9)..=None]);
        test_complement_util(&[Some(4)..=Some(8)], &[None..=Some(3), Some(9)..=None]);
        test_complement_util(
            &[Some(4)..=Some(8), Some(10)..=None],
            &[None..=Some(3), Some(9)..=Some(9)],
        );
        test_complement_util(&[None..=Some(8), Some(10)..=None], &[Some(9)..=Some(9)]);
    }

    #[test]
    fn test_union_empty() {
        test_union_util::<i8, i8>(&[], &[], &[]);
        test_union_util::<i8, i8>(&[], &[0..=5], &[0..=5]);
    }

    #[test]
    fn test_union_non_overlapping() {
        test_union_util(&[0..=3], &[5..=7], &[0..=3, 5..=7]);
        test_union_util(&[0..=3, 10..=12], &[5..=7], &[0..=3, 5..=7, 10..=12]);
        test_union_util(&[0..=3], &[5..=7, 10..=12], &[0..=3, 5..=7, 10..=12]);
    }

    #[test]
    fn test_union_overlapping() {
        test_union_util(&[0..=5], &[3..=8], &[0..=8]);
        test_union_util(&[0..=5], &[5..=8], &[0..=8]);
        test_union_util(&[0..=5, 10..=15], &[3..=12], &[0..=15]);
    }

    #[test]
    fn test_union_adjacent() {
        test_union_util(&[0..=5], &[6..=10], &[0..=10]);
        test_union_util(&[0..=5, 10..=15], &[6..=9], &[0..=15]);
    }

    #[test]
    fn test_union_contained() {
        test_union_util(&[0..=10], &[3..=7], &[0..=10]);
        test_union_util(&[3..=7], &[0..=10], &[0..=10]);
    }

    #[test]
    fn test_union_open_ranges() {
        test_union_util(
            &[None..=Some(5)],
            &[Some(10)..=None],
            &[None..=Some(5), Some(10)..=None],
        );
        test_union_util(&[None..=Some(5)], &[Some(3)..=None], &[None..=None]);
        test_union_util(&[Some(0)..=Some(5)], &[None..=Some(3)], &[None..=Some(5)]);
    }

    fn build_range<I: Into<Option<i8>> + Clone>(range: RangeInclusive<I>) -> IncRange<BigInt> {
        let (s, e) = range.into_inner();
        IncRange {
            start_inc: s.into().map(BigInt::from),
            end_inc: e.into().map(BigInt::from),
        }
    }

    fn build_multi_range<I: Into<Option<i8>> + Clone>(ranges: &[RangeInclusive<I>]) -> IncRangeMulti {
        let initial = ranges.iter().map(|r| build_range(r.clone())).collect_vec();
        let result = IncRangeMulti { ranges: initial };
        result.assert_valid();
        result
    }

    fn test_remove_util<I: Into<Option<i8>> + Clone, R: Into<Option<i8>> + Clone, E: Into<Option<i8>> + Clone>(
        initial: &[RangeInclusive<I>],
        remove: RangeInclusive<R>,
        expected: &[RangeInclusive<E>],
    ) {
        let initial = build_multi_range(initial);

        let remove_range = IncRangeMulti::from_range(build_range(remove));
        let result = initial.subtract(&remove_range);

        let expected = build_multi_range(expected);
        assert_eq!(result, expected);
    }

    fn test_complement_util<A: Into<Option<i8>> + Clone, B: Into<Option<i8>> + Clone>(
        a: &[RangeInclusive<A>],
        b: &[RangeInclusive<B>],
    ) {
        let a = build_multi_range(a);
        let b = build_multi_range(b);

        assert_eq!(a, b.complement());
        assert_eq!(b, a.complement());
    }

    fn test_union_util<A: Into<Option<i8>> + Clone, B: Into<Option<i8>> + Clone>(
        a: &[RangeInclusive<A>],
        b: &[RangeInclusive<B>],
        expected: &[RangeInclusive<A>],
    ) {
        let a = build_multi_range(a);
        let b = build_multi_range(b);
        let expected = build_multi_range(expected);

        let result = a.union(&b);
        result.assert_valid();
        assert_eq!(result, expected);

        let result_reverse = b.union(&a);
        result_reverse.assert_valid();
        assert_eq!(result_reverse, expected);
    }
}
