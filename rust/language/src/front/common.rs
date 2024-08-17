use crate::data::compiled::{GenericTypeParameter, GenericValueParameter};
use crate::front::driver::Item;
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::syntax::pos::FileId;
use indexmap::IndexMap;

/// Utility type to refer to a specific item in a specific file.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ItemReference {
    pub file: FileId,
    pub item_index: usize,
}

// TODO pick a better name for this
// TODO is this still necessary? can't items also be sorted into types or values immediately? 
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(Item),
    Direct(ScopedEntryDirect),
}

// TODO transpose or not?
pub type ScopedEntryDirect = MaybeConstructor<TypeOrValue>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValue<T = Type, V = Value> {
    Type(T),
    Value(V),
}

impl<T, V> TypeOrValue<T, V> {
    pub fn as_ref(&self) -> TypeOrValue<&T, &V> {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t),
            TypeOrValue::Value(v) => TypeOrValue::Value(v),
        }
    }

    pub fn unwrap_type(self) -> T {
        match self {
            TypeOrValue::Type(t) => t,
            TypeOrValue::Value(_) => panic!("Expected type, got value"),
        }
    }

    pub fn unwrap_value(self) -> V {
        match self {
            TypeOrValue::Type(_) => panic!("Expected value, got type"),
            TypeOrValue::Value(v) => v,
        }
    }

    pub fn unit(&self) -> TypeOrValue<(), ()> {
        match self {
            TypeOrValue::Type(_) => TypeOrValue::Type(()),
            TypeOrValue::Value(_) => TypeOrValue::Value(()),
        }
    }
}

/// A Monad trait, specifically for replacing generic parameters in a type or value with more concrete arguments.
// TODO replace this a more general "map" trait, where the user can supply their own closure
// TODO no need to mix type and value parameters into a single map
pub trait GenericContainer {
    /// The implementation can assume the replacement has already been kind- and type-checked.
    fn replace_generic_params(
        &self,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Self;
}

impl GenericContainer for TypeOrValue {
    fn replace_generic_params(
        &self,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Self {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t.replace_generic_params(map_ty, map_value)),
            TypeOrValue::Value(v) => TypeOrValue::Value(v.replace_generic_params(map_ty, map_value)),
        }
    }
}
