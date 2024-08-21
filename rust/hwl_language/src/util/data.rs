use indexmap::IndexMap;
use std::hash::Hash;

pub trait IndexMapExt<K, V> {
    // The same as [IndexMap::insert], but asserts that the key is not already present.
    fn insert_first(&mut self, key: K, value: V);

    fn sort_by_key<T: Ord>(&mut self, f: impl FnMut(&K, &V) -> T);
}

impl<K, V> IndexMapExt<K, V> for IndexMap<K, V>
where
    K: Eq + Hash,
{
    fn insert_first(&mut self, key: K, value: V) {
        let prev = self.insert(key, value);
        assert!(prev.is_none());
    }

    fn sort_by_key<T: Ord>(&mut self, mut f: impl FnMut(&K, &V) -> T) {
        self.sort_by(|k0, v0, k1, v1| f(k0, v0).cmp(&f(k1, v1)));
    }
}
