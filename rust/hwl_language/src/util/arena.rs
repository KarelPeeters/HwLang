use crate::util::{Never, ResultNeverExt};
use indexmap::map::IndexMap;
use itertools::Itertools;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
// TODO use refcell for all of these data structures?
//   that would allow users to push new values without worrying about mutability
//   the trickier functions (that actually allow mutating existing values) would still be behind &mut.
//   !! but what about allowing internal iteration? that will conflict with the refcell!

#[macro_export]
macro_rules! new_index_type {
    ($vis:vis $name:ident, Ord) => {
        $crate::new_index_type!($vis $name);

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.inner().index().cmp(&other.inner().index())
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
    };
    ($vis:vis $name:ident) => {
        #[derive(Copy, Clone, Eq, PartialEq, Hash)]
        $vis struct $name($crate::util::arena::Idx);

        // trick to make the imports not leak outside of the macro
        const _: () = {
            use $crate::util::arena::IndexType;
            use $crate::util::arena::Idx;

            impl IndexType for $name {
                fn new(idx: Idx) -> Self {
                    Self(idx)
                }
                fn inner(&self) -> Idx {
                    self.0
                }
            }

            impl std::fmt::Debug for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "<{} {}>", stringify!($name), self.0.index())
                }
            }
        };
    };
}
pub trait IndexType: Sized + Debug + Copy + Eq + Hash {
    fn new(idx: Idx) -> Self;
    fn inner(&self) -> Idx;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Idx {
    index: usize,
    check: u64,
}

impl Idx {
    pub fn index(&self) -> usize {
        self.index
    }
}

pub struct Arena<K: IndexType, T> {
    values: Vec<T>,
    check: u64,
    ph: PhantomData<K>,
}

#[allow(dead_code)]
impl<K: IndexType, T> Arena<K, T> {
    pub fn push(&mut self, value: T) -> K {
        self.push_with_index(|_| value)
    }

    // TODO consider passing &mut self as an argument:
    // * keep a separate next_index, so new allocations get higher indices
    // * the current value is not yet inserted, that breaks the arena guarantee a bit, so check for this condition
    pub fn push_with_index(&mut self, value: impl FnOnce(K) -> T) -> K {
        let key = K::new(Idx {
            index: self.values.len(),
            check: self.check,
        });
        let value = value(key);
        self.values.push(value);
        key
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &T)> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (K, &mut T)> {
        self.into_iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = K> + Clone + '_ {
        self.into_iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }

    pub fn map_values<U>(self, mut f: impl FnMut(K, T) -> U) -> Arena<K, U> {
        self.try_map_values::<_, Never>(|k, v| Ok(f(k, v))).remove_never()
    }

    pub fn try_map_values<U, E>(self, mut f: impl FnMut(K, T) -> Result<U, E>) -> Result<Arena<K, U>, E> {
        Ok(Arena {
            values: self
                .values
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let k = K::new(Idx {
                        index,
                        check: self.check,
                    });
                    f(k, value)
                })
                .try_collect()?,
            check: self.check,
            ph: PhantomData,
        })
    }

    pub fn get_by_index(&self, index: usize) -> Option<(K, &T)> {
        if index < self.values.len() {
            let k = K::new(Idx {
                index,
                check: self.check,
            });
            Some((k, &self[k]))
        } else {
            None
        }
    }
}

impl<K: IndexType, T> Index<K> for Arena<K, T> {
    type Output = T;
    fn index(&self, index: K) -> &Self::Output {
        assert_eq!(
            self.check,
            index.inner().check,
            "Arena index {:?} used in arena which did not create it",
            index
        );
        &self.values[index.inner().index]
    }
}

impl<K: IndexType, T> IndexMut<K> for Arena<K, T> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        assert_eq!(
            self.check,
            index.inner().check,
            "Arena index {:?} used in arena which did not create it",
            index
        );
        &mut self.values[index.inner().index]
    }
}

impl<K: IndexType, T> Default for Arena<K, T> {
    fn default() -> Self {
        Self {
            values: vec![],
            check: rand::random(),
            ph: PhantomData,
        }
    }
}

impl<K: IndexType, T: Debug> Debug for Arena<K, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let map: IndexMap<_, _> = self.iter().collect();
        map.fmt(f)
    }
}

pub struct ArenaIteratorRef<'s, K, T> {
    inner: std::iter::Enumerate<std::slice::Iter<'s, T>>,
    check: u64,
    ph: PhantomData<K>,
}

pub struct ArenaIteratorMut<'s, K, T> {
    inner: std::iter::Enumerate<std::slice::IterMut<'s, T>>,
    check: u64,
    ph: PhantomData<K>,
}

impl<'s, K: IndexType, T> IntoIterator for &'s Arena<K, T> {
    type Item = (K, &'s T);
    type IntoIter = ArenaIteratorRef<'s, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArenaIteratorRef {
            inner: self.values.iter().enumerate(),
            check: self.check,
            ph: PhantomData,
        }
    }
}

impl<'s, K: IndexType, T> IntoIterator for &'s mut Arena<K, T> {
    type Item = (K, &'s mut T);
    type IntoIter = ArenaIteratorMut<'s, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArenaIteratorMut {
            inner: self.values.iter_mut().enumerate(),
            check: self.check,
            ph: PhantomData,
        }
    }
}

impl<'s, K: IndexType, T: 's> Iterator for ArenaIteratorRef<'s, K, T> {
    type Item = (K, &'s T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(key_wrapper(self.check))
    }
}

impl<K, T> Clone for ArenaIteratorRef<'_, K, T> {
    fn clone(&self) -> Self {
        ArenaIteratorRef {
            inner: self.inner.clone(),
            check: self.check,
            ph: PhantomData,
        }
    }
}

impl<'s, K: IndexType, T: 's> Iterator for ArenaIteratorMut<'s, K, T> {
    type Item = (K, &'s mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(key_wrapper(self.check))
    }
}

fn key_wrapper<K: IndexType, V>(check: u64) -> impl Fn((usize, V)) -> (K, V) {
    move |(index, value)| (K::new(Idx { index, check }), value)
}

#[cfg(test)]
mod test {
    use crate::util::arena::Arena;

    new_index_type!(TestIdx);

    #[test]
    fn basic() {
        let mut arena: Arena<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');
        assert_eq!(arena[ai], 'a');
        assert_eq!(arena[bi], 'b');
    }

    #[test]
    fn duplicate() {
        let mut arena: Arena<TestIdx, char> = Default::default();
        let ai0 = arena.push('a');
        let ai1 = arena.push('a');
        assert_eq!(arena[ai0], 'a');
        assert_eq!(arena[ai1], 'a');
        assert_ne!(ai0, ai1)
    }

    #[test]
    fn iter() {
        let mut arena: Arena<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');

        let expected = vec![(ai, &'a'), (bi, &'b')];
        let mut actual: Vec<(TestIdx, &char)> = arena.iter().collect();
        actual.sort_by_key(|(i, _)| i.0.index);

        assert_eq!(actual, expected);
    }
}
