use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use indexmap::map::IndexMap;

// TODO use refcell for all of these data structures?
//   that would allow users to push new values without worrying about mutability
//   the trickier functions (that actually allow mutating existing values) would still be behind &mut.
//   !! but what about allowing internal iteration? that will conflict with the refcell!

#[macro_export]
macro_rules! new_index_type {
    ($vis:vis $name:ident) => {
        #[derive(Copy, Clone, Eq, PartialEq, Hash)]
        $vis struct $name(crate::util::arena::Idx);

        //trick to make the imports not leak outside of the macro
        const _: () = {
            use crate::util::arena::IndexType;
            use crate::util::arena::Idx;

            impl IndexType for $name {
                fn idx(&self) -> Idx {
                    self.0
                }
                fn new(idx: Idx) -> Self {
                    Self(idx)
                }
            }

            impl std::fmt::Debug for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "{:?} {}", self.0, stringify!($name))
                }
            }
        };
    }
}

pub trait IndexType: Sized + Debug {
    fn idx(&self) -> Idx;
    fn new(idx: Idx) -> Self;

    fn index(&self) -> usize {
        self.idx().i
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Idx {
    i: usize,
}

impl Idx {
    fn new(i: usize) -> Self {
        Self { i }
    }
}

impl Debug for Idx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{}>", self.i)
    }
}

// TODO include randomly generated checker index value to avoid accidental mixing?
//   on clone or value map, switch up the future ID, and keep a list of valid old IDs as well
pub struct Arena<K: IndexType, T> {
    //TODO for now this is implemented as a map, but this can be improved
    //  to just be a vec using generational indices
    map: IndexMap<usize, T>,
    next_i: usize,
    ph: PhantomData<K>,
}

#[allow(dead_code)]
impl<K: IndexType, T> Arena<K, T> {
    pub fn push(&mut self, value: T) -> K {
        let i = self.next_i;
        self.next_i += 1;
        assert!(self.map.insert(i, value).is_none());
        K::new(Idx::new(i))
    }

    pub fn replace(&mut self, index: K, new_value: T) -> T {
        self.map.insert(index.idx().i, new_value)
            .unwrap_or_else(|| panic!("Value {:?} not found", index))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=(K, &T)> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=(K, &mut T)> {
        self.into_iter()
    }

    pub fn keys(&self) -> impl Iterator<Item=K> + '_ {
        self.into_iter().map(|(k, _)| k)
    }

    pub fn retain<F: FnMut(K, &T) -> bool>(&mut self, mut keep: F) {
        self.map.retain(|&i, v| keep(K::new(Idx::new(i)), v))
    }

    pub fn map_values<U>(&self, mut f: impl FnMut(&T) -> U) -> Arena<K, U> {
        let new_map = self.map.iter()
            .map(|(&i, v)| (i, f(v)))
            .collect();

        Arena {
            map: new_map,
            next_i: self.next_i,
            ph: Default::default(),
        }
    }
}

impl<K: IndexType, T> Index<K> for Arena<K, T> {
    type Output = T;
    fn index(&self, index: K) -> &Self::Output {
        self.map.get(&index.idx().i)
            .unwrap_or_else(|| panic!("Value {:?} not found", index))
    }
}

impl<K: IndexType, T> IndexMut<K> for Arena<K, T> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        self.map.get_mut(&index.idx().i)
            .unwrap_or_else(|| panic!("Value {:?} not found", index))
    }
}

impl<K: IndexType, T> Default for Arena<K, T> {
    fn default() -> Self {
        Self { map: Default::default(), next_i: 0, ph: PhantomData }
    }
}

impl<K: IndexType, T: Debug> Debug for Arena<K, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.map.fmt(f)
    }
}

pub struct ArenaIterator<'s, K, T> {
    inner: indexmap::map::Iter<'s, usize, T>,
    ph: PhantomData<K>,
}

pub struct ArenaIteratorMut<'s, K, T> {
    inner: indexmap::map::IterMut<'s, usize, T>,
    ph: PhantomData<K>,
}

impl<'s, K: IndexType, T> IntoIterator for &'s Arena<K, T> {
    type Item = (K, &'s T);
    type IntoIter = ArenaIterator<'s, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArenaIterator { inner: self.map.iter(), ph: PhantomData }
    }
}

impl<'s, K: IndexType, T> IntoIterator for &'s mut Arena<K, T> {
    type Item = (K, &'s mut T);
    type IntoIter = ArenaIteratorMut<'s, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArenaIteratorMut { inner: self.map.iter_mut(), ph: PhantomData }
    }
}

impl<'s, K: IndexType, T: 's> Iterator for ArenaIterator<'s, K, T> {
    type Item = (K, &'s T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
            .map(|(&i, v)| (K::new(Idx::new(i)), v))
    }
}

impl<'s, K: IndexType, T: 's> Iterator for ArenaIteratorMut<'s, K, T> {
    type Item = (K, &'s mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
            .map(|(&i, v)| (K::new(Idx::new(i)), v))
    }
}

#[derive(Clone)]
pub struct ArenaSet<K: IndexType, T: Eq + Hash + Clone> {
    //TODO this implementation should also be optimized
    //TODO this clone bound should be removed
    map_fwd: IndexMap<usize, T>,
    map_back: IndexMap<T, usize>,
    next_i: usize,
    ph: PhantomData<K>,
}

impl<K: IndexType, T: Eq + Hash + Clone + Debug> ArenaSet<K, T> {
    pub fn lookup(&self, value: &T) -> Option<K> {
        self.map_back.get(value).map(|&i| K::new(Idx::new(i)))
    }

    pub fn push(&mut self, value: T) -> K {
        if let Some(&i) = self.map_back.get(&value) {
            K::new(Idx::new(i))
        } else {
            let i = self.next_i;
            self.next_i += 1;
            self.map_fwd.insert(i, value.clone());
            self.map_back.insert(value, i);
            K::new(Idx::new(i))
        }
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.map_fwd.len(), self.map_back.len());
        self.map_fwd.len()
    }

    pub fn iter(&self) -> impl Iterator<Item=(K, &T)> {
        self.into_iter()
    }
}

impl<K: IndexType, T: Eq + Hash + Clone> Index<K> for ArenaSet<K, T> {
    type Output = T;
    fn index(&self, index: K) -> &Self::Output {
        self.map_fwd.get(&index.idx().i)
            .unwrap_or_else(|| panic!("Value {:?} not found", index))
    }
}

impl<K: IndexType, T: Eq + Hash + Clone> Default for ArenaSet<K, T> {
    fn default() -> Self {
        Self {
            map_fwd: Default::default(),
            map_back: Default::default(),
            next_i: 0,
            ph: PhantomData,
        }
    }
}

impl<K: IndexType, T: Debug + Eq + Hash + Clone> Debug for ArenaSet<K, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.map_fwd.fmt(f)
    }
}

pub struct ArenaSetIterator<'s, K, T> {
    inner: indexmap::map::Iter<'s, usize, T>,
    ph: PhantomData<K>,
}

impl<'s, K: IndexType, T: Eq + Hash + Clone> IntoIterator for &'s ArenaSet<K, T> {
    type Item = (K, &'s T);
    type IntoIter = ArenaSetIterator<'s, K, T>;

    fn into_iter(self) -> Self::IntoIter {
        ArenaSetIterator { inner: self.map_fwd.iter(), ph: PhantomData }
    }
}

impl<'s, K: IndexType, T: 's> Iterator for ArenaSetIterator<'s, K, T> {
    type Item = (K, &'s T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
            .map(|(&i, v)| (K::new(Idx::new(i)), v))
    }
}

#[cfg(test)]
mod test {
    use crate::util::arena::{Arena, ArenaSet};

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
    fn pop() {
        let mut arena: Arena<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');
        arena.pop(ai);
        assert_eq!(arena[bi], 'b');
    }

    #[test]
    #[should_panic]
    fn pop_twice() {
        let mut arena: Arena<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        arena.push('b');
        arena.pop(ai);
        arena.pop(ai);
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
    fn basic_set() {
        let mut arena: ArenaSet<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');
        assert_eq!(arena[ai], 'a');
        assert_eq!(arena[bi], 'b');
    }

    #[test]
    fn duplicate_set() {
        let mut arena: ArenaSet<TestIdx, char> = Default::default();
        let ai0 = arena.push('a');
        let ai1 = arena.push('a');
        assert_eq!(arena[ai0], 'a');
        assert_eq!(ai0, ai1)
    }

    #[test]
    fn pop_set() {
        let mut arena: ArenaSet<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');
        arena.pop(ai);
        assert_eq!(arena[bi], 'b');
    }

    #[test]
    #[should_panic]
    fn pop_twice_set() {
        let mut arena: ArenaSet<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        arena.push('b');
        assert_eq!(arena.pop(ai), 'a');
        arena.pop(ai); //panics
    }

    #[test]
    fn iter() {
        let mut arena: Arena<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');

        let expected = vec![
            (ai, &'a'),
            (bi, &'b'),
        ];
        let mut actual: Vec<(TestIdx, &char)> = arena.iter().collect();
        actual.sort_by_key(|x| (x.0).0.i);

        assert_eq!(actual, expected);
    }

    #[test]
    fn iter_set() {
        let mut arena: ArenaSet<TestIdx, char> = Default::default();
        let ai = arena.push('a');
        let bi = arena.push('b');

        let expected = vec![
            (ai, &'a'),
            (bi, &'b'),
        ];
        let mut actual: Vec<(TestIdx, &char)> = arena.iter().collect();
        actual.sort_by_key(|x| (x.0).0.i);

        assert_eq!(actual, expected);
    }
}