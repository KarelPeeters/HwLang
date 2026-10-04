use crate::syntax::ast::MaybeIdentifier;
use crate::syntax::pos::Spanned;
use crate::util::arena::RandomCheck;
use indexmap::IndexSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

/// A fully-evaluated identifier.
/// Identifiers are interned through [Interner], so they're very cheap to store and compare.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Id {
    check: RandomCheck,
    index: usize,
}

pub type MaybeId = MaybeIdentifier<Id>;

// TODO replace with actual fast implementation:
//   * look at lasso
//   * dashmap? large shared string buffers with offsets?
pub struct Interner {
    check: RandomCheck,
    map: RwLock<IndexSet<String>>,
}

impl Interner {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        let _ = thread_count;
        Self {
            check: RandomCheck::new(),
            map: RwLock::new(IndexSet::new()),
        }
    }

    pub fn push(&self, s: &str) -> Id {
        // try read-only lookup first
        let index = if let Some(index) = self.map.read().unwrap().get_index_of(s) {
            index
        } else {
            // get write lock, then try cheap read again
            let mut map = self.map.write().unwrap();
            if let Some(index) = map.get_index_of(s) {
                index
            } else {
                // actually insert owned value
                let (index, new) = map.insert_full(s.to_owned());
                assert!(new);
                index
            }
        };

        Id {
            check: self.check,
            index,
        }
    }

    pub fn push_owned(&self, s: String) -> Id {
        // TODO avoid clone if possible
        self.push(&s)
    }

    pub fn push_arc(&self, s: Arc<String>) -> Id {
        // TODO avoid clone if possible
        self.push(&s)
    }

    pub fn get(&self, id: Id) -> &str {
        assert_eq!(self.check, id.check);

        let map = self.map.read().unwrap();
        let s = map.get_index(id.index).unwrap().as_str() as *const str;

        // Safety:
        //   We never delete anything from the map,
        //   and strings are heap allocated and so stay stable when the map resizes.
        //   This means the returned reference can get the same lifetime as self.
        unsafe { &*s }
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
