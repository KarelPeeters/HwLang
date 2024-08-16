use crate::front::driver::Item;
use crate::front::param::{GenericContainer, GenericParameterUniqueId};
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use indexmap::IndexMap;

// TODO pick a better name for this
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(Item),
    Direct(ScopedEntryDirect),
}

// TODO transpose or not?
pub type ScopedEntryDirect = MaybeConstructor<TypeOrValue>;

#[derive(Debug, Clone)]
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
}

impl GenericContainer for TypeOrValue {
    fn replace_generic_params(&self, map: &IndexMap<GenericParameterUniqueId, TypeOrValue>) -> Self {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t.replace_generic_params(map)),
            TypeOrValue::Value(v) => TypeOrValue::Value(v.replace_generic_params(map)),
        }
    }
}
