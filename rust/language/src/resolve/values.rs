use std::hash::{Hash, Hasher};
use num_bigint::BigInt;
use crate::new_index_type;
use crate::resolve::compile::{FunctionBody, Item, ItemReference};
use crate::resolve::scoped_entry::ValueParameter;
use crate::resolve::types::{Type, TypeUnique};
use crate::syntax::ast::Identifier;
use crate::util::arena::ArenaSet;

// TODO should all values have types? or can eg. ints just be free abstract objects?
// TODO during compilation, have a "value" wrapper that lazily computes the content and type to break up cycles
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Value {
    Parameter(ValueParameter),
    
    Int(BigInt),
    Function(FunctionValue),
    Module(ModuleValue),
    // Struct(StructValue),
    // Tuple(TupleValue),
    // Enum(EnumValue),

    // TODO should this be a dedicated type or just an instance of the normal range struct?
    Range { start: Option<BigInt>, end: Option<BigInt> },
}

#[derive(Debug, Clone)]
pub struct FunctionValue {
    // only this field is used in hash and eq
    // TODO also include captured values once those exist
    pub item_reference: ItemReference,
    
    pub ty: Type,
    pub params: Vec<Identifier>,
    pub body: FunctionBody,
}

#[derive(Debug, Clone)]
pub struct ModuleValue {
    // only this field is used in hash and eq
    pub unique: TypeUnique,
    // TODO include real content
    pub ty: Type,
}
