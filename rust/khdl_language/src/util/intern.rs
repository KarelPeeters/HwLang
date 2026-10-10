use crate::syntax::ast::MaybeIdentifier;
use crate::syntax::pos::Spanned;
use crate::util::arena::RandomCheck;
use crate::util::sync::dashmap_shard_count;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use fnv::FnvBuildHasher;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU32, Ordering};

/// A fully-evaluated identifier.
/// Identifiers are interned through [Interner], so they're very cheap to store and compare.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Id {
    check: RandomCheck,
    index: u32,
}

pub type MaybeId = MaybeIdentifier<Id>;

pub struct Interner {
    check: RandomCheck,
    string_to_id: DashMap<String, u32, FnvBuildHasher>,
    id_to_string: DashMap<u32, String, FnvBuildHasher>,
    next_id: AtomicU32,
}

impl Interner {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        let shard_count = dashmap_shard_count(thread_count).get();
        Self {
            check: RandomCheck::new(),
            string_to_id: DashMap::with_hasher_and_shard_amount(FnvBuildHasher::default(), shard_count),
            id_to_string: DashMap::with_hasher_and_shard_amount(FnvBuildHasher::default(), shard_count),
            next_id: AtomicU32::new(0),
        }
    }

    pub fn push(&self, s: &str) -> Id {
        if let Some(index) = self.string_to_id.get(s) {
            return self.id(*index);
        }

        match self.string_to_id.entry(s.to_owned()) {
            Entry::Occupied(entry) => self.id(*entry.get()),
            Entry::Vacant(entry) => {
                let index = self.next_id.fetch_add(1, Ordering::Relaxed);
                self.id_to_string.insert(index, s.to_owned());
                entry.insert(index);
                self.id(index)
            }
        }
    }

    pub fn get(&self, id: Id) -> &str {
        assert_eq!(self.check, id.check);

        let entry = self.id_to_string.get(&id.index).unwrap();
        let s_ref = entry.value().as_str();
        let s_ptr = s_ref as *const str;
        drop(entry);

        // Safety:
        //   * the string is stored in a `String`, whose heap buffer does not move when the map grows,
        //   * entries are never removed or mutated,
        //   so the pointer stays valid for as long as `self` is borrowed, even after the reference guard is dropped.
        unsafe { &*s_ptr }
    }

    fn id(&self, index: u32) -> Id {
        Id {
            check: self.check,
            index,
        }
    }
}

impl Id {
    pub fn str(self, interner: &Interner) -> &str {
        interner.get(self)
    }
}

impl MaybeId {
    pub fn str(self, interner: &Interner) -> Option<&str> {
        match self {
            MaybeIdentifier::Dummy { .. } => None,
            MaybeIdentifier::Identifier(id) => Some(id.str(interner)),
        }
    }

    pub fn diagnostic_str(self, interner: &Interner) -> &str {
        self.str(interner).unwrap_or("_")
    }
}

impl MaybeIdentifier<Spanned<Id>> {
    pub fn diagnostic_str(self, interner: &Interner) -> &str {
        self.map_id(|id| id.inner).diagnostic_str(interner)
    }

    pub fn spanned_string(self, interner: &Interner) -> Spanned<Option<String>> {
        match self {
            MaybeIdentifier::Dummy { span } => Spanned::new(span, None),
            MaybeIdentifier::Identifier(id) => Spanned::new(id.span, Some(id.inner.str(interner).to_owned())),
        }
    }
}
