use crate::data::diagnostic::ErrorGuaranteed;
use crate::new::types::Type;
use crate::new::value::Value;
use crate::new_index_type;

// TODO move everything in this file to a better place
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValue<V> {
    Type(Type<V>),
    Value(V),
    Error(ErrorGuaranteed),
}

new_index_type!(pub Item);

#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(Item),
    Direct(TypeOrValue<Value>),
}

