use crate::resolve::compile::ItemReference;
use crate::resolve::types::Type;

#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(ItemReference),
    TypeParam(TypeParameter),
    ValueParam(ValueParameter),
    Variable(Variable),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeParameter {
    item: ItemReference,
    name: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ValueParameter {
    item: ItemReference,
    name: String,
    ty: Type,
}

#[derive(Debug, Clone)]
pub struct Variable {
    ty: Type,
    // TODO
}
