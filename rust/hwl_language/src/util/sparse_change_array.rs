use crate::util::Never;
use crate::util::big_int::BigUint;
use crate::util::iter::IterExt;
use crate::util::range::ClosedRange;
use std::fmt::Debug;
use std::ops::ControlFlow;

/// Data structure for arrays that are sparse, not in their values, but in _changes_ to their values.
///
/// For all visitor methods:
///   The number of times the callback is called is not necessarily the same as the length of the array or range,
///   sub-ranges that don't contain any change points are only visited once.
///
/// For mutating visitor methods:
///   Range boundry indices are added to the array,
///   to ensure modifications to the inner element don't incorrectly affect elements outside the range.
#[derive(Debug, Clone)]
pub struct SparseChangeArray<T> {
    len: BigUint,
    start: Option<T>,
    changes: Vec<(BigUint, T)>,
}

impl<T> SparseChangeArray<T> {
    pub fn new(len: BigUint, init: T) -> SparseChangeArray<T> {
        SparseChangeArray {
            len,
            start: Some(init),
            changes: vec![],
        }
    }

    pub fn new_empty() -> SparseChangeArray<T> {
        SparseChangeArray {
            len: BigUint::ZERO,
            start: None,
            changes: vec![],
        }
    }

    pub fn len(&self) -> &BigUint {
        &self.len
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

    #[allow(clippy::needless_lifetimes)]
    pub fn for_each<'a, B>(&'a self, mut f: impl FnMut(&'a T) -> ControlFlow<B>) -> ControlFlow<B> {
        if let Some(start) = &self.start {
            f(start)?;
            self.changes.iter().try_for_each(|(_, x)| f(x))
        } else {
            ControlFlow::Continue(())
        }
    }

    #[allow(clippy::needless_lifetimes)]
    pub fn for_each_mut<'a, B>(&'a mut self, mut f: impl FnMut(&'a mut T) -> ControlFlow<B>) -> ControlFlow<B> {
        if let Some(start) = &mut self.start {
            f(start)?;
            self.changes.iter_mut().try_for_each(|(_, x)| f(x))
        } else {
            ControlFlow::Continue(())
        }
    }

    #[allow(clippy::needless_lifetimes)]
    pub fn for_each_in_range<'a, B>(
        &'a self,
        range: ClosedRange<&BigUint>,
        mut f: impl FnMut(&'a T) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        // unwrap and handle edge cases
        let Some(start) = &self.start else {
            assert!(range.start.is_zero());
            assert!(range.is_empty());
            return ControlFlow::Continue(());
        };

        let ClosedRange {
            start: range_start,
            end: range_end,
        } = range;
        range.assert_valid();
        assert!(range_end <= &self.len);

        if range.is_empty() {
            return ControlFlow::Continue(());
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
            f(start)?;
        }
        self.changes[change_start.unwrap_or(0)..change_end]
            .iter()
            .map(|(_, x)| x)
            .try_for_each(f)
    }

    #[allow(clippy::needless_lifetimes)]
    pub fn for_each_in_range_mut<'a, B>(
        &'a mut self,
        range: ClosedRange<&BigUint>,
        mut f: impl FnMut(&'a mut T) -> ControlFlow<B>,
    ) -> ControlFlow<B>
    where
        T: Clone,
    {
        // unwrap and handle edge cases
        if self.start.is_none() {
            assert!(range.start.is_zero());
            assert!(range.is_empty());
            return ControlFlow::Continue(());
        };

        let ClosedRange {
            start: range_start,
            end: range_end,
        } = range;
        range.assert_valid();
        assert!(range_end <= &self.len);

        if range.is_empty() {
            return ControlFlow::Continue(());
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
            f(start)?;
        }
        self.changes[change_start.unwrap_or(0)..change_end]
            .iter_mut()
            .map(|(_, x)| x)
            .try_for_each(f)
    }

    /// Ensure the given `indices` are present in this array. The past indices must be non-decreasing.
    /// This is useful to prepare for a mutation that needs particular indices to exist,
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

    pub fn zip2_for_each<U, R>(
        a: &SparseChangeArray<T>,
        b: &SparseChangeArray<U>,
        mut f: impl FnMut(&T, &U) -> ControlFlow<R>,
    ) -> ControlFlow<R> {
        zip_impl::<Never, _, _, _>(None, Some(a), Some(b), |_, a, b| f(a.unwrap(), b.unwrap()))
    }

    pub fn zip3_mut_for_each<U, V, R>(
        a: &mut SparseChangeArray<T>,
        b: Option<&SparseChangeArray<U>>,
        c: Option<&SparseChangeArray<V>>,
        mut f: impl FnMut(&mut T, Option<&U>, Option<&V>) -> ControlFlow<R>,
    ) -> ControlFlow<R>
    where
        T: Clone,
    {
        zip_impl(Some(a), b, c, |a, b, c| f(a.unwrap(), b, c))
    }
}

fn zip_impl<A: Clone, B, C, R>(
    a: Option<&mut SparseChangeArray<A>>,
    b: Option<&SparseChangeArray<B>>,
    c: Option<&SparseChangeArray<C>>,
    mut f: impl FnMut(Option<&mut A>, Option<&B>, Option<&C>) -> ControlFlow<R>,
) -> ControlFlow<R> {
    // check array lengths match and handle no args or empty array cases
    let mut len = a.as_ref().map(|a| a.len.clone());
    if let Some(b) = b {
        assert!(len.is_none_or(|len| len == b.len));
        len = Some(b.len.clone());
    }
    if let Some(c) = c {
        assert!(len.is_none_or(|len| len == c.len));
        len = Some(c.len.clone());
    }

    let Some(len) = len else {
        return ControlFlow::Continue(());
    };
    if len.is_zero() {
        return ControlFlow::Continue(());
    }

    if let Some(a) = a {
        // ensure we have the necessary indices
        // TODO do this in a single combined pass?
        if let Some(b) = b {
            a.ensure_indices(b.changes.iter().map(pair_first));
        }
        if let Some(c) = c {
            a.ensure_indices(c.changes.iter().map(pair_first));
        }

        // build state
        let mut state_b = b.map(|b| (b.start.as_ref().unwrap(), b.changes.as_slice()));
        let mut state_c = c.map(|c| (c.start.as_ref().unwrap(), c.changes.as_slice()));

        // visit first element
        f(
            Some(a.start.as_mut().unwrap()),
            state_b.map(|(s, _)| s),
            state_c.map(|(s, _)| s),
        )?;

        // visit change elements
        // just walk over all indices in the base, and step the others when relevant
        for (index_a, curr_a) in &mut a.changes {
            let index_a = &*index_a;

            // pop b/c if necessary
            if let Some((curr_b, changes_b)) = &mut state_b {
                if let Some(((index_b, next_b), rest_b)) = changes_b.split_first() {
                    debug_assert!(index_a <= index_b);
                    if index_a == index_b {
                        *curr_b = next_b;
                        *changes_b = rest_b;
                    }
                }
            }
            if let Some((curr_c, changes_c)) = &mut state_c {
                if let Some(((index_c, next_c), rest_c)) = changes_c.split_first() {
                    debug_assert!(index_a <= index_c);
                    if index_a == index_c {
                        *curr_c = next_c;
                        *changes_c = rest_c;
                    }
                }
            }

            // visit change element
            f(Some(curr_a), state_b.map(|(s, _)| s), state_c.map(|(s, _)| s))?;
        }

        if let Some((_, changes_b)) = state_b {
            debug_assert!(changes_b.is_empty());
        }
        if let Some((_, changes_c)) = state_c {
            debug_assert!(changes_c.is_empty());
        }
    } else {
        // handle simple cases first
        let (b, c) = match (b, c) {
            (None, None) => return ControlFlow::Continue(()),
            (Some(b), None) => return b.for_each(|b| f(None, Some(b), None)),
            (None, Some(c)) => return c.for_each(|c| f(None, None, Some(c))),
            (Some(b), Some(c)) => (b, c),
        };

        // build state
        let mut curr_b = b.start.as_ref().unwrap();
        let mut changes_b = b.changes.as_slice();
        let mut curr_c = c.start.as_ref().unwrap();
        let mut changes_c = c.changes.as_slice();

        // visit first element
        f(None, Some(curr_b), Some(curr_c))?;

        loop {
            // pop the lowest index (or both)
            match (changes_b.split_first(), changes_c.split_first()) {
                (None, None) => break,
                (Some(((_, next_b), rest_b)), None) => {
                    curr_b = next_b;
                    changes_b = rest_b;
                }
                (None, Some(((_, next_c), rest_c))) => {
                    curr_c = next_c;
                    changes_c = rest_c;
                }
                (Some(((index_b, next_b), rest_b)), Some(((index_c, next_c), rest_c))) => {
                    let cmp = index_b.cmp(index_c);
                    if cmp.is_le() {
                        curr_b = next_b;
                        changes_b = rest_b;
                    }
                    if cmp.is_ge() {
                        curr_c = next_c;
                        changes_c = rest_c;
                    }
                }
            }

            // visit change element
            f(None, Some(curr_b), Some(curr_c))?;
        }
    }

    ControlFlow::Continue(())
}

fn pair_first<A, B>(x: &(A, B)) -> &A {
    &x.0
}

#[cfg(test)]
mod tests {
    use super::zip_impl;
    use crate::util::big_int::BigUint;
    use crate::util::exhaust::exhaust;
    use crate::util::range::ClosedRange;
    use crate::util::sparse_change_array::SparseChangeArray;
    use crate::util::{Never, ResultNeverExt};
    use indexmap::IndexSet;
    use itertools::{Itertools, enumerate};
    use std::ops::ControlFlow;

    #[test]
    fn test_exhaust() {
        const LEN_LIMIT: usize = 6;
        const VALUE_LIMIT: u64 = 3;

        exhaust(|ex| {
            let len = ex.choose(LEN_LIMIT as u64) as usize;
            let array_dense = (0..len).map(|_| ex.choose(VALUE_LIMIT)).collect_vec();
            let array_change = build_from_array(&array_dense);

            println!("generated array");
            println!("  dense:  {:?}", array_dense);
            println!("  change: {:?}", array_change);

            for i in 0..len {
                assert_eq!(&array_dense[i], array_change.get(&BigUint::from(i)));
            }

            exhaust(|ex_inner| {
                let range_start = ex_inner.choose((len + 1) as u64);
                let range_end = ex_inner.choose_range(range_start, (len + 1) as u64);
                let range = ClosedRange {
                    start: range_start,
                    end: range_end,
                }
                .map(BigUint::from);

                println!("  visiting range {range_start}..{range_end}");

                // check immutable visit
                //   verify that we get the exact callback sequence we expect
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
                    array_change
                        .for_each_in_range::<Never>(range.as_ref(), |&x| {
                            r.push(x);
                            ControlFlow::Continue(())
                        })
                        .remove_never();
                    r
                };
                assert_eq!(expected_visit, actual_visit);

                // check mutable visit
                //   verify that filling a range works the same for dense and change arrays
                println!("  mutable");
                let mut array_dense_mut = array_dense.clone();
                let mut array_change_mut = array_change.clone();

                array_dense_mut[range_start as usize..range_end as usize].fill(VALUE_LIMIT);
                array_change_mut
                    .for_each_in_range_mut::<Never>(range.as_ref(), |x| {
                        *x = VALUE_LIMIT;
                        ControlFlow::Continue(())
                    })
                    .remove_never();

                assert_array_match(&array_dense_mut, &array_change_mut);
            });
        });
    }

    #[test]
    fn test_zip_impl_exhaust() {
        const LEN_LIMIT: usize = 5;
        const VALUE_LIMIT: u64 = 2;

        exhaust(|ex| {
            let len = ex.choose(LEN_LIMIT as u64) as usize;

            let mut create_array_pair = || {
                ex.choose_bool().then(|| {
                    let dense = (0..len).map(|_| ex.choose(VALUE_LIMIT)).collect_vec();
                    let change = build_from_array(&dense);
                    (dense, change)
                })
            };
            let mut a_pair = create_array_pair();
            let b_pair = create_array_pair();
            let c_pair = create_array_pair();

            println!("generated arrays");
            println!("  a: {:?}", a_pair);
            println!("  b: {:?}", b_pair);
            println!("  c: {:?}", c_pair);

            // we do two checks:
            //   * ensure we've seen all tuples we expected to see (including the order)
            //   * mutate using both dense and change arrays using the same operation, and check that results match
            let op = |seen: &mut IndexSet<(Option<u64>, Option<u64>, Option<u64>)>,
                      a: Option<&mut u64>,
                      b: Option<&u64>,
                      c: Option<&u64>| {
                seen.insert((a.as_ref().map(|a| **a), b.copied(), c.copied()));
                if let Some(a) = a {
                    let b = b.copied().unwrap_or(VALUE_LIMIT);
                    let c = c.copied().unwrap_or(VALUE_LIMIT);
                    if b < c {
                        *a = b + c;
                    }
                }
            };

            // run on dense
            let mut seen_dense = IndexSet::new();
            if a_pair.is_some() || b_pair.is_some() || c_pair.is_some() {
                for i in 0..len {
                    op(
                        &mut seen_dense,
                        a_pair.as_mut().map(|(a, _)| &mut a[i]),
                        b_pair.as_ref().map(|(b, _)| &b[i]),
                        c_pair.as_ref().map(|(c, _)| &c[i]),
                    );
                }
            }

            // run on change
            let mut seen_change = IndexSet::new();
            zip_impl(
                a_pair.as_mut().map(|(_, a)| a),
                b_pair.as_ref().map(|(_, b)| b),
                c_pair.as_ref().map(|(_, c)| c),
                |a, b, c| {
                    op(&mut seen_change, a, b, c);
                    ControlFlow::Continue(())
                },
            )
            .remove_never();

            // check match (including set order)
            assert_eq!(
                seen_dense.into_iter().collect_vec(),
                seen_change.into_iter().collect_vec()
            );
            if let Some((a_dense, a_change)) = a_pair {
                assert_array_match(&a_dense, &a_change);
            }
        });
    }

    fn build_from_array<T: Clone + Eq>(array_dense: &[T]) -> SparseChangeArray<T> {
        let result = match array_dense.split_first() {
            None => SparseChangeArray::new_empty(),
            Some((first, rest)) => {
                let mut changes = vec![];
                let mut prev = first;
                for (i, curr) in enumerate(rest) {
                    if curr != prev {
                        changes.push((BigUint::from(i + 1), curr.clone()));
                        prev = curr;
                    }
                }

                SparseChangeArray {
                    len: BigUint::from(array_dense.len()),
                    start: Some(first.clone()),
                    changes,
                }
            }
        };
        result.assert_valid();
        result
    }

    fn assert_array_match(dense: &[u64], change: &SparseChangeArray<u64>) {
        assert_eq!(&BigUint::from(dense.len()), change.len());
        for i in 0..dense.len() {
            assert_eq!(&dense[i], change.get(&BigUint::from(i)));
        }
    }
}
