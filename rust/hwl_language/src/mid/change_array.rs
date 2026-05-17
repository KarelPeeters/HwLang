use crate::util::big_int::BigUint;
use crate::util::iter::IterExt;
use crate::util::range::ClosedRange;
use std::fmt::Debug;

// TODO make variants private?
#[derive(Debug, Clone)]
pub struct ChangeArray<T> {
    len: BigUint,
    start: Option<T>,
    changes: Vec<(BigUint, T)>,
}

impl<T> ChangeArray<T> {
    pub fn new(len: BigUint, init: T) -> ChangeArray<T> {
        ChangeArray {
            len,
            start: Some(init),
            changes: vec![],
        }
    }

    pub fn new_empty() -> ChangeArray<T> {
        ChangeArray {
            len: BigUint::ZERO,
            start: None,
            changes: vec![],
        }
    }

    pub fn assert_valid(&self) {
        if self.len.is_zero() {
            assert!(self.start.is_none());
            assert!(self.changes.is_empty());
        } else {
            assert!(self.start.is_some());
        }

        let mut prev_index = &BigUint::ZERO;
        for (index, _) in &self.changes {
            assert!(prev_index < index);
            assert!(index < &self.len);
            prev_index = index;
        }
    }

    pub fn get(&self, index: &BigUint) -> &T {
        assert!(index < &self.len);
        match self.changes.binary_search_by_key(&index, pair_first) {
            Ok(found) => &self.changes[found].1,
            Err(insert) => {
                if insert == 0 {
                    self.start.as_ref().unwrap()
                } else {
                    &self.changes[insert - 1].1
                }
            }
        }
    }

    /// Ensure the given `indices` are present in this array. The past indices must be non-decreasing.
    /// This can be useful to prepare for a mutation that needs particular indices to exist
    ///   to avoid editing wrong parts of the array.
    fn ensure_indices<'a>(&mut self, indices: impl IntoIterator<Item = &'a BigUint>)
    where
        T: Clone,
    {
        let indices = indices.into_iter();

        let Some(start) = &self.start else {
            assert!(indices.is_empty());
            return;
        };

        // TODO faster implementation without constant array reshuffling
        let mut prev_index = None;
        for new_index in indices {
            // correctness checks
            assert!(new_index <= &self.len);
            if let Some(prev_index) = prev_index {
                assert!(prev_index <= new_index);
            }
            prev_index = Some(new_index);

            // skip index zero (which always exists as `start`)
            //   and index len (which should never exist)
            if new_index.is_zero() {
                continue;
            }
            if new_index == &self.len {
                continue;
            }

            // find new index
            match self.changes.binary_search_by_key(&new_index, pair_first) {
                Ok(_) => {
                    // index already exists, nothing to do
                }
                Err(insert) => {
                    // index does not yet exist, create it as a copy of the previous value
                    let prev_value = if insert == 0 {
                        start.clone()
                    } else {
                        self.changes[insert - 1].1.clone()
                    };

                    self.changes.insert(insert, (new_index.clone(), prev_value));
                }
            }
        }

        if cfg!(debug_assertions) {
            self.assert_valid();
        }
    }

    /// Visit each element in the given range.
    ///
    /// The number of times the callback is called is not necessarily the same as the length of the range:
    ///   sub-ranges that don't contain any change points are only visited once.
    pub fn for_each_in_range(&self, range: ClosedRange<&BigUint>, mut f: impl FnMut(&T)) {
        // unwrap and handle edge cases
        let Some(start) = &self.start else {
            assert!(range.start.is_zero());
            assert!(range.is_empty());
            return;
        };

        let ClosedRange {
            start: range_start,
            end: range_end,
        } = range;
        range.assert_valid();
        assert!(range_end <= &self.len);

        if range.is_empty() {
            return;
        }

        // get indices
        let change_start = match self.changes.binary_search_by_key(&range_start, pair_first) {
            Ok(found) => Some(found),
            Err(insert) => insert.checked_sub(1),
        };
        let change_end = self
            .changes
            .binary_search_by_key(&range_end, pair_first)
            .unwrap_or_else(|insert| insert);

        // actually visit
        if change_start.is_none() {
            f(start);
        }
        for (_, value) in &self.changes[change_start.unwrap_or(0)..change_end] {
            f(value);
        }
    }

    /// Visit each element in the given range.
    /// The number of times the callback is called is not necessarily the same as the length of the range:
    ///   sub-ranges that don't contain any change points are only visited once.
    ///
    /// The start and end points of the range are added to this array if they don't yet exist,
    ///   to ensure modifications to the inner element don't incorrectly affect elements outside the range.
    pub fn for_each_in_range_mut(&mut self, range: ClosedRange<&BigUint>, mut f: impl FnMut(&mut T))
    where
        T: Clone,
    {
        // unwrap and handle edge cases
        if self.start.is_none() {
            assert!(range.start.is_zero());
            assert!(range.is_empty());
            return;
        };

        let ClosedRange {
            start: range_start,
            end: range_end,
        } = range;
        range.assert_valid();
        assert!(range_end <= &self.len);

        if range.is_empty() {
            return;
        }

        // ensure the necessary indices exist
        let range_end_sub_1 = range_end.sub_1().unwrap();
        self.ensure_indices([range_start, &range_end_sub_1, range_end]);

        // find those indices
        let change_start = if range_start.is_zero() {
            None
        } else {
            Some(self.changes.binary_search_by_key(&range_start, pair_first).unwrap())
        };
        let change_end = if range_end == &self.len {
            self.changes.len()
        } else {
            self.changes.binary_search_by_key(&range_end, pair_first).unwrap()
        };

        // actually visit
        let start = self.start.as_mut().unwrap();
        if change_start.is_none() {
            f(start);
        }
        for (_, value) in &mut self.changes[change_start.unwrap_or(0)..change_end] {
            f(value);
        }
    }
}

fn pair_first<A, B>(x: &(A, B)) -> &A {
    &x.0
}

#[cfg(test)]
mod tests {
    use crate::mid::change_array::ChangeArray;
    use crate::util::big_int::BigUint;
    use crate::util::exhaust::exhaust;
    use crate::util::range::ClosedRange;
    use itertools::{Itertools, enumerate};

    #[test]
    fn test_exhaust() {
        const LEN_LIMIT: u64 = 6;
        const VALUE_LIMIT: u64 = 3;

        exhaust(|ex| {
            let len = ex.choose(LEN_LIMIT);
            let array_dense = (0..len).map(|_| ex.choose(VALUE_LIMIT)).collect_vec();
            let array_change = build_from_array(&array_dense);

            println!("generated array");
            println!("  dense:  {:?}", array_dense);
            println!("  change: {:?}", array_change);

            for i in 0..len {
                assert_eq!(&array_dense[i as usize], array_change.get(&BigUint::from(i)));
            }

            exhaust(|ex_inner| {
                let range_start = ex_inner.choose(len + 1);
                let range_end = ex_inner.choose_range(range_start, len + 1);
                let range = ClosedRange {
                    start: range_start,
                    end: range_end,
                }
                .map(BigUint::from);

                println!("  visiting range {range_start}..{range_end}");
                println!("  immutable");
                let expected_visit = {
                    let mut r = vec![];
                    if len != 0 {
                        let mut prev = None;
                        for &x in &array_dense[range_start as usize..range_end as usize] {
                            if Some(x) != prev {
                                r.push(x);
                                prev = Some(x);
                            }
                        }
                    }
                    r
                };
                let actual_visit = {
                    let mut r = vec![];
                    array_change.for_each_in_range(range.as_ref(), |&x| r.push(x));
                    r
                };
                assert_eq!(expected_visit, actual_visit);

                println!("  mutable");
                let mut array_dense_mut = array_dense.clone();
                let mut array_change_mut = array_change.clone();

                array_dense_mut[range_start as usize..range_end as usize].fill(VALUE_LIMIT);
                array_change_mut.for_each_in_range_mut(range.as_ref(), |x| *x = VALUE_LIMIT);

                for i in 0..len {
                    assert_eq!(&array_dense_mut[i as usize], array_change_mut.get(&BigUint::from(i)));
                }
            });
        });
    }

    fn build_from_array<T: Clone + Eq>(array_dense: &[T]) -> ChangeArray<T> {
        let result = match array_dense.split_first() {
            None => ChangeArray::new_empty(),
            Some((first, rest)) => {
                let mut changes = vec![];
                let mut prev = first;
                for (i, curr) in enumerate(rest) {
                    if curr != prev {
                        changes.push((BigUint::from(i + 1), curr.clone()));
                        prev = curr;
                    }
                }

                ChangeArray {
                    len: BigUint::from(array_dense.len()),
                    start: Some(first.clone()),
                    changes,
                }
            }
        };
        result.assert_valid();
        result
    }
}
