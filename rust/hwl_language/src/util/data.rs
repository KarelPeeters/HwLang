use indexmap::map::Entry;
use indexmap::IndexMap;
use std::hash::Hash;

pub trait IndexMapExt<K, V> {
    // The same as [IndexMap::insert], but asserts that the key is not already present.
    fn insert_first(&mut self, key: K, value: V) -> &mut V;

    fn sort_by_key<T: Ord>(&mut self, f: impl FnMut(&K, &V) -> T);
}

impl<K, V> IndexMapExt<K, V> for IndexMap<K, V>
where
    K: Eq + Hash,
{
    fn insert_first(&mut self, key: K, value: V) -> &mut V {
        match self.entry(key) {
            Entry::Occupied(_) => panic!("entry already exists"),
            Entry::Vacant(entry) => entry.insert(value),
        }
    }

    fn sort_by_key<T: Ord>(&mut self, mut f: impl FnMut(&K, &V) -> T) {
        self.sort_by(|k0, v0, k1, v1| f(k0, v0).cmp(&f(k1, v1)));
    }
}

pub trait VecExt<T> {
    fn single(self) -> Option<T>;

    fn with_pushed<R>(&mut self, v: T, f: impl FnOnce(&mut Self) -> R) -> R;
}

impl<T> VecExt<T> for Vec<T> {
    fn single(self) -> Option<T> {
        if self.len() == 1 {
            Some(self.into_iter().next().unwrap())
        } else {
            None
        }
    }

    fn with_pushed<R>(&mut self, v: T, f: impl FnOnce(&mut Self) -> R) -> R {
        self.push(v);
        let result = f(self);
        assert!(self.pop().is_some());
        result
    }
}

pub fn vec_concat<const N: usize, T>(vecs: [Vec<T>; N]) -> Vec<T> {
    let mut result = vec![];
    for vec in vecs {
        result.extend(vec);
    }
    result
}
