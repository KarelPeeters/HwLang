use std::marker::PhantomData;
use crate::front::driver::ItemReference;
use crate::front::values::Value;

/// Used to deduplicate [nominative types](https://en.wikipedia.org/wiki/Nominal_type_system) like structs or enums.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeUnique {
    pub item_reference: ItemReference,
    ph: PhantomData<Private>,
    // pub params: Vec<Value>,
}

// TODO remove this once/if we fix TypeUnique
struct Private;

#[macro_export]
macro_rules! impl_eq_hash_unique {
    ($name:ident) => {
        impl Eq for $name {}
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.unique == other.unique
            }
        }
        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.unique.hash(state)
            }
        }
    };
}
