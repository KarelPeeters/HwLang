use crate::util::big_int::{BigInt, BigUint};
use itertools::Itertools;
use std::fmt::{Debug, Display, Formatter};
use std::ops::AddAssign;

// TODO rename to min/max? more intuitive than start/end, min and max are clearly inclusive
// TODO clarify and enforce invariants (ie. start <= end)
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct IncRange<T> {
    pub start_inc: Option<T>,
    pub end_inc: Option<T>,
}

// TODO can this represent the empty range? maybe exclusive is better after all...
//   we don't really want empty ranges for int types, but for for loops and slices we do
// TODO switch to exclusive ranges, much more intuitive to program with, especially for arrays and loops
//   match code becomes harder, but that's fine
// TODO transition this to multi-range as the int type
// TODO make sure that people can only construct non-decreasing ranges,
//   there are still some panics in the compiler because of this
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClosedIncRange<T> {
    pub start_inc: T,
    pub end_inc: T,
}

#[derive(Eq, PartialEq, Clone)]
pub struct IncRangeMulti {
    // TODO use SmallVec?
    ranges: Vec<IncRange<BigInt>>,
}

#[derive(Eq, PartialEq, Clone)]
pub struct ClosedIncRangeMulti {
    inner: IncRangeMulti,
}

impl<T> IncRange<T> {
    pub const OPEN: IncRange<T> = IncRange {
        start_inc: None,
        end_inc: None,
    };

    pub fn single(value: T) -> Self
    where
        T: Clone,
    {
        IncRange {
            start_inc: Some(value.clone()),
            end_inc: Some(value),
        }
    }

    pub fn as_ref(&self) -> IncRange<&T> {
        IncRange {
            start_inc: self.start_inc.as_ref(),
            end_inc: self.end_inc.as_ref(),
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> IncRange<U> {
        IncRange {
            start_inc: self.start_inc.map(&mut f),
            end_inc: self.end_inc.map(&mut f),
        }
    }

    pub fn try_into_closed(self) -> Result<ClosedIncRange<T>, Self> {
        let IncRange { start_inc, end_inc } = self;

        let start_inc = match start_inc {
            Some(start_inc) => start_inc,
            None => {
                return Err(IncRange {
                    start_inc: None,
                    end_inc,
                });
            }
        };
        let end_inc = match end_inc {
            Some(end_inc) => end_inc,
            None => {
                return Err(IncRange {
                    start_inc: Some(start_inc),
                    end_inc: None,
                });
            }
        };

        Ok(ClosedIncRange { start_inc, end_inc })
    }

    pub fn contains(&self, other: &T) -> bool
    where
        T: Ord,
    {
        let IncRange { start_inc, end_inc } = self;
        match (start_inc, end_inc) {
            (None, None) => true,
            (Some(start_inc), None) => start_inc <= other,
            (None, Some(end_inc)) => other <= end_inc,
            (Some(start_inc), Some(end_inc)) => start_inc <= other && other <= end_inc,
        }
    }

    pub fn contains_range(&self, other: &IncRange<T>) -> bool
    where
        T: Ord,
    {
        let IncRange {
            start_inc: self_start_inc,
            end_inc: self_end_inc,
        } = self;
        let IncRange {
            start_inc: other_start_inc,
            end_inc: other_end_inc,
        } = other;

        let start_contains = match (self_start_inc, other_start_inc) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(self_start_inc), Some(other_start_inc)) => self_start_inc <= other_start_inc,
        };
        let end_contains = match (self_end_inc, other_end_inc) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(self_end_inc), Some(other_end_inc)) => self_end_inc >= other_end_inc,
        };

        start_contains && end_contains
    }

    pub fn union<'a>(&'a self, other: IncRange<&'a T>) -> IncRange<&'a T>
    where
        T: Ord + Clone,
    {
        let IncRange {
            start_inc: a_start,
            end_inc: a_end,
        } = self;
        let IncRange {
            start_inc: b_start,
            end_inc: b_end,
        } = other;

        let start = match (a_start, b_start) {
            (Some(a_start), Some(b_start)) => Some(a_start.min(b_start)),
            (None, _) | (_, None) => None,
        };
        let end = match (a_end, b_end) {
            (Some(a_end), Some(b_end)) => Some(a_end.max(b_end)),
            (None, _) | (_, None) => None,
        };

        IncRange {
            start_inc: start,
            end_inc: end,
        }
    }

    pub fn intersect<'a>(&'a self, other: IncRange<&'a T>) -> Option<IncRange<&'a T>>
    where
        T: Ord + Clone,
    {
        let IncRange {
            start_inc: a_start,
            end_inc: a_end,
        } = self;
        let IncRange {
            start_inc: b_start,
            end_inc: b_end,
        } = other;

        let start = match (a_start, b_start) {
            (Some(a_start), Some(b_start)) => Some(a_start.max(b_start)),
            (Some(a_start), None) => Some(a_start),
            (None, Some(b_start)) => Some(b_start),
            (None, None) => None,
        };
        let end = match (a_end, b_end) {
            (Some(a_end), Some(b_end)) => Some(a_end.min(b_end)),
            (Some(a_end), None) => Some(a_end),
            (None, Some(b_end)) => Some(b_end),
            (None, None) => None,
        };

        if let (Some(start), Some(end)) = (start, end)
            && start > end
        {
            None
        } else {
            Some(IncRange {
                start_inc: start,
                end_inc: end,
            })
        }
    }
}

impl<T: Clone> IncRange<&T> {
    pub fn cloned(self) -> IncRange<T> {
        IncRange {
            start_inc: self.start_inc.cloned(),
            end_inc: self.end_inc.cloned(),
        }
    }
}

impl<T> ClosedIncRange<T> {
    pub fn single(value: T) -> Self
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

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> ClosedIncRange<U> {
        let ClosedIncRange { start_inc, end_inc } = self;
        ClosedIncRange {
            start_inc: f(start_inc),
            end_inc: f(end_inc),
        }
    }

    pub fn contains(&self, value: &T) -> bool
    where
        T: Ord,
    {
        let ClosedIncRange { start_inc, end_inc } = self;
        start_inc <= value && value <= end_inc
    }

    pub fn contains_range(&self, other: ClosedIncRange<&T>) -> bool
    where
        T: Ord,
    {
        // TODO always accept less-than-empty `other` ranges?
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

    pub fn union(self, other: ClosedIncRange<T>) -> ClosedIncRange<T>
    where
        T: Ord + Clone,
    {
        let ClosedIncRange {
            start_inc: a_start,
            end_inc: a_end,
        } = self;
        let ClosedIncRange {
            start_inc: b_start,
            end_inc: b_end,
        } = other;

        ClosedIncRange {
            start_inc: a_start.min(b_start),
            end_inc: a_end.max(b_end),
        }
    }
}

impl<T: Display> Display for IncRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let IncRange { start_inc, end_inc } = self;
        match (start_inc, end_inc) {
            (None, None) => write!(f, ".."),
            (Some(start_inc), None) => write!(f, "{start_inc}.."),
            (None, Some(end_inc)) => write!(f, "..={end_inc}"),
            (Some(start_inc), Some(end_inc)) => write!(f, "{start_inc}..={end_inc}"),
        }
    }
}

impl<T: Display> Display for ClosedIncRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ClosedIncRange { start_inc, end_inc } = self;
        write!(f, "{start_inc}..={end_inc}")
    }
}

pub struct ClosedIncRangeIterator<T> {
    next: T,
    end_inc: T,
}

impl IntoIterator for ClosedIncRange<BigUint> {
    type Item = BigUint;
    type IntoIter = ClosedIncRangeIterator<BigUint>;

    fn into_iter(self) -> Self::IntoIter {
        ClosedIncRangeIterator {
            next: self.start_inc,
            end_inc: self.end_inc,
        }
    }
}

impl Iterator for ClosedIncRangeIterator<BigUint> {
    type Item = BigUint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next <= self.end_inc {
            let curr = self.next.clone();
            self.next += 1u8;
            Some(curr)
        } else {
            None
        }
    }
}

impl IntoIterator for ClosedIncRange<BigInt> {
    type Item = BigInt;
    type IntoIter = ClosedIncRangeIterator<BigInt>;

    fn into_iter(self) -> Self::IntoIter {
        ClosedIncRangeIterator {
            next: self.start_inc,
            end_inc: self.end_inc,
        }
    }
}

impl Iterator for ClosedIncRangeIterator<BigInt> {
    type Item = BigInt;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next <= self.end_inc {
            let curr = self.next.clone();
            self.next += 1u8;
            Some(curr)
        } else {
            None
        }
    }
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

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
enum CompareBound<'a> {
    OpenBottom,
    Mid(&'a BigInt),
    OpenTop,
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
                    let l_start = l.start_inc.as_ref().map_or(CompareBound::OpenBottom, CompareBound::Mid);
                    let r_start = r.start_inc.as_ref().map_or(CompareBound::OpenBottom, CompareBound::Mid);
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
                        let curr_end = curr.end_inc.as_ref().map_or(CompareBound::OpenTop, CompareBound::Mid);
                        let next_end = next.end_inc.as_ref().map_or(CompareBound::OpenTop, CompareBound::Mid);

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
    use crate::front::range::IncRange;
    use crate::front::range::IncRangeMulti;
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
