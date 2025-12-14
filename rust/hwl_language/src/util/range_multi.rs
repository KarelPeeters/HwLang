use crate::util::range::{ClosedNonEmptyRange, ClosedRange, NonEmptyRange, Range, RangeEmpty};
use itertools::Itertools;
use std::fmt::{Debug, Display, Formatter};

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct MultiRange<T> {
    // TODO use SmallVec?
    ranges: Vec<NonEmptyRange<T>>,
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct ClosedMultiRange<T> {
    inner: MultiRange<T>,
}

impl<T> MultiRange<T> {
    pub const EMPTY: MultiRange<T> = MultiRange { ranges: vec![] };

    #[must_use]
    pub fn contains(&self, x: &T) -> bool
    where
        T: Ord,
    {
        // TODO speed up with binary search?
        self.ranges.iter().any(|r| r.contains(x))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn assert_valid(&self)
    where
        T: Ord,
    {
        for i in 0..self.ranges.len() {
            let curr = &self.ranges[i];
            let next = self.ranges.get(i + 1);

            curr.assert_valid();

            if let Some(next) = next {
                let curr_end = curr.end.as_ref().unwrap();
                let next_start = next.start.as_ref().unwrap();
                assert!(
                    curr_end <= next_start,
                    "multi ranges must be non-overlapping and non-adjacent"
                );
            }
        }
    }

    #[must_use]
    pub fn enclosing_range(&self) -> Option<NonEmptyRange<&T>>
    where
        T: Ord,
    {
        let range = NonEmptyRange {
            start: self.ranges.first()?.start.as_ref(),
            end: self.ranges.last()?.end.as_ref(),
        };
        if cfg!(debug_assertions) {
            range.assert_valid();
        }
        Some(range)
    }

    #[must_use]
    pub fn complement(&self) -> MultiRange<T>
    where
        T: Ord + Clone,
    {
        let mut ranges = Vec::new();

        if self.ranges.is_empty() {
            ranges.push(NonEmptyRange::OPEN);
        } else {
            if let Some(first_start) = &self.ranges[0].start {
                ranges.push(NonEmptyRange {
                    start: None,
                    end: Some(first_start.clone()),
                });
            }

            for (curr, next) in self.ranges.iter().tuple_windows() {
                let curr_end = curr.end.as_ref().unwrap();
                let next_start = next.start.as_ref().unwrap();

                debug_assert!(curr_end < next_start);
                ranges.push(NonEmptyRange {
                    start: Some(curr_end.clone()),
                    end: Some(next_start.clone()),
                });
            }

            if let Some(last_end) = &self.ranges.last().unwrap().end {
                ranges.push(NonEmptyRange {
                    start: Some(last_end.clone()),
                    end: None,
                });
            }
        }

        let result = MultiRange { ranges };
        if cfg!(debug_assertions) {
            result.assert_valid();
        }
        result
    }

    #[must_use]
    pub fn union(&self, other: &MultiRange<T>) -> MultiRange<T>
    where
        T: Ord + Clone,
    {
        let mut iter_self = self.ranges.iter().peekable();
        let mut iter_other = other.ranges.iter().peekable();

        let mut result = Vec::new();
        let mut curr: Option<NonEmptyRange<T>> = None;

        loop {
            let next = match (iter_self.peek(), iter_other.peek()) {
                (Some(l), Some(r)) => {
                    let (l_start, _) = l.as_ref().bounds();
                    let (r_start, _) = r.as_ref().bounds();
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
                    let (_, curr_end) = curr.as_ref().bounds();
                    let (next_start, next_end) = next.as_ref().bounds();

                    if next_start <= curr_end {
                        if next_end > curr_end {
                            curr.end = next.end.clone();
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

        let result = MultiRange { ranges: result };
        if cfg!(debug_assertions) {
            result.assert_valid();
        }
        result
    }

    #[must_use]
    pub fn intersect(&self, other: &MultiRange<T>) -> MultiRange<T>
    where
        T: Ord + Clone,
    {
        self.complement().union(&other.complement()).complement()
    }

    #[must_use]
    pub fn subtract(&self, range: &MultiRange<T>) -> MultiRange<T>
    where
        T: Ord + Clone,
    {
        self.complement().union(range).complement()
    }
}

impl<T> ClosedMultiRange<T> {
    pub const EMPTY: ClosedMultiRange<T> = ClosedMultiRange {
        inner: MultiRange::EMPTY,
    };

    fn assert_valid(&self)
    where
        T: Ord,
    {
        self.inner.assert_valid();
        if let Some(range) = self.enclosing_range() {
            range.assert_valid();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter_ranges(&self) -> impl Iterator<Item = ClosedNonEmptyRange<T>>
    where
        T: Ord + Clone,
    {
        self.inner.ranges.iter().map(|r| {
            ClosedNonEmptyRange::try_from(r.clone()).expect("started from closed, all ranges should be closed")
        })
    }

    pub fn enclosing_range(&self) -> Option<ClosedNonEmptyRange<&T>>
    where
        T: Ord,
    {
        Some(
            ClosedNonEmptyRange::try_from(self.inner.enclosing_range()?)
                .expect("started from closed, result should be closed again"),
        )
    }

    #[must_use]
    pub fn subtract(&mut self, range: &MultiRange<T>) -> ClosedMultiRange<T>
    where
        T: Ord + Clone,
    {
        // subtracting from a closed range will always yield a closed (or empty) range
        ClosedMultiRange {
            inner: self.inner.subtract(range),
        }
    }
}

impl<T: Ord> From<Range<T>> for MultiRange<T> {
    fn from(value: Range<T>) -> Self {
        let ranges = match NonEmptyRange::try_from(value) {
            Ok(non_empty) => vec![non_empty],
            Err(RangeEmpty) => vec![],
        };
        let result = MultiRange { ranges };
        if cfg!(debug_assertions) {
            result.assert_valid();
        }
        result
    }
}

impl<T: Ord> From<ClosedRange<T>> for ClosedMultiRange<T> {
    fn from(value: ClosedRange<T>) -> Self {
        let inner = MultiRange::from(Range::from(value));
        let result = ClosedMultiRange { inner };
        if cfg!(debug_assertions) {
            result.assert_valid();
        }
        result
    }
}

impl<T: Ord> From<ClosedNonEmptyRange<T>> for ClosedMultiRange<T> {
    fn from(value: ClosedNonEmptyRange<T>) -> Self {
        ClosedMultiRange::from(ClosedRange::from(value))
    }
}

impl<T: Display> Debug for MultiRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let MultiRange { ranges } = self;
        write!(f, "MultiRange({})", ranges.iter().format(", "))
    }
}

impl<T: Display> Debug for ClosedMultiRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ClosedMultiRange { inner } = self;
        write!(f, "ClosedMultiRange({})", inner.ranges.iter().format(", "))
    }
}

#[cfg(test)]
mod test {
    use crate::util::range::{NonEmptyRange, Range};
    use crate::util::range_multi::MultiRange;
    use itertools::Itertools;

    #[test]
    fn test_remove_nothing() {
        test_remove_util(&[Some(0)..Some(8)], Some(10)..Some(11), &[Some(0)..Some(8)]);
        test_remove_util(&[Some(0)..Some(8)], Some(8)..Some(11), &[Some(0)..Some(8)]);
        test_remove_util(&[Some(0)..Some(8)], Some(-2)..Some(0), &[Some(0)..Some(8)]);
    }

    #[test]
    fn test_remove_full() {
        test_remove_util::<_, _, i8>(&[0..8], 0..8, &[]);
        test_remove_util::<_, _, i8>(&[0..8], -2..11, &[]);

        test_remove_util(&[2..5, 6..9, 10..13], 1..10, &[10..13]);
    }

    #[test]
    fn test_remove_split_edge() {
        test_remove_util(&[0..8], 4..11, &[0..4]);
        test_remove_util(&[0..8], -2..4, &[4..8]);
        test_remove_util(&[0..4, 6..10], 2..9, &[0..2, 9..10]);
    }

    #[test]
    fn test_remove_split_inside() {
        test_remove_util(&[0..8], 2..5, &[0..2, 5..8]);
        test_remove_util(&[0..4, 6..10], 7..9, &[0..4, 6..7, 9..10]);
    }

    #[test]
    fn test_remove_split_edge_and_full() {
        test_remove_util(&[0..4, 5..8, 9..12], 2..11, &[0..2, 11..12]);
    }

    #[test]
    fn test_complement() {
        test_complement_util::<i8, Option<i8>>(&[], &[None..None]);

        test_complement_util(&[None..Some(9)], &[Some(9)..None]);
        test_complement_util(&[Some(4)..Some(9)], &[None..Some(4), Some(9)..None]);
        test_complement_util(&[Some(4)..Some(9), Some(10)..None], &[None..Some(4), Some(9)..Some(10)]);
        test_complement_util(&[None..Some(9), Some(10)..None], &[Some(9)..Some(10)]);
    }

    #[test]
    fn test_union_empty() {
        test_union_util::<i8, i8>(&[], &[], &[]);
        test_union_util::<i8, i8>(&[], &[0..6], &[0..6]);
    }

    #[test]
    fn test_union_non_overlapping() {
        test_union_util(&[0..4], &[5..8], &[0..4, 5..8]);
        test_union_util(&[0..4, 10..13], &[5..8], &[0..4, 5..8, 10..13]);
        test_union_util(&[0..4], &[5..8, 10..13], &[0..4, 5..8, 10..13]);
    }

    #[test]
    fn test_union_overlapping() {
        test_union_util(&[0..6], &[3..9], &[0..9]);
        test_union_util(&[0..6], &[5..9], &[0..9]);
        test_union_util(&[0..6, 10..16], &[3..13], &[0..16]);
    }

    #[test]
    fn test_union_adjacent() {
        test_union_util(&[0..6], &[6..11], &[0..11]);
        test_union_util(&[0..6, 10..16], &[6..10], &[0..16]);
    }

    #[test]
    fn test_union_contained() {
        test_union_util(&[0..11], &[3..8], &[0..11]);
        test_union_util(&[3..8], &[0..11], &[0..11]);
    }

    #[test]
    fn test_union_open_ranges() {
        test_union_util(&[None..Some(6)], &[Some(10)..None], &[None..Some(6), Some(10)..None]);
        test_union_util(&[None..Some(6)], &[Some(3)..None], &[None..None]);
        test_union_util(&[Some(0)..Some(6)], &[None..Some(4)], &[None..Some(6)]);
    }

    fn build_range<I: Into<Option<i8>>>(range: std::ops::Range<I>) -> NonEmptyRange<i8> {
        let std::ops::Range { start, end } = range;
        let result = NonEmptyRange {
            start: start.into(),
            end: end.into(),
        };
        result.assert_valid();
        result
    }

    fn build_multi_range<I: Into<Option<i8>> + Clone>(ranges: &[std::ops::Range<I>]) -> MultiRange<i8> {
        let initial = ranges.iter().map(|r| build_range(r.clone())).collect_vec();
        let result = MultiRange { ranges: initial };
        result.assert_valid();
        result
    }

    fn test_remove_util<I: Into<Option<i8>> + Clone, R: Into<Option<i8>> + Clone, E: Into<Option<i8>> + Clone>(
        initial: &[std::ops::Range<I>],
        remove: std::ops::Range<R>,
        expected: &[std::ops::Range<E>],
    ) {
        let initial = build_multi_range(initial);

        let remove_range = MultiRange::from(Range::from(build_range(remove)));
        let result = initial.subtract(&remove_range);

        let expected = build_multi_range(expected);
        assert_eq!(result, expected);
    }

    fn test_complement_util<A: Into<Option<i8>> + Clone, B: Into<Option<i8>> + Clone>(
        a: &[std::ops::Range<A>],
        b: &[std::ops::Range<B>],
    ) {
        let a = build_multi_range(a);
        let b = build_multi_range(b);

        assert_eq!(a, b.complement());
        assert_eq!(b, a.complement());
    }

    fn test_union_util<A: Into<Option<i8>> + Clone, B: Into<Option<i8>> + Clone>(
        a: &[std::ops::Range<A>],
        b: &[std::ops::Range<B>],
        expected: &[std::ops::Range<A>],
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
