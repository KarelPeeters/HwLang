use std::hash::{Hash, Hasher};
use num_bigint::BigInt;
use crate::new_index_type;
use crate::resolve::compile::{FunctionBody, Item, ItemReference};
use crate::resolve::types::Type;
use crate::syntax::ast::Identifier;
use crate::util::arena::ArenaSet;

// TODO find a better name for this, eg. InterpreterValue, CompileValue, just Value, ...
new_index_type!(pub Value);

// TODO this should probably have either garbage collection or no arena at all:
//   we'll be running bytecode which can generate a large number of intermediate eg. integers
//   alternatively we can keep the arena for fixed things (like types and signatures)
//   but not for values (ints, arrays, ...)
pub type Values = ArenaSet<Value, ValueInfo>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ValueInfo {
    Type(Type),
    Item(Item),
    
    Int(ValueIntInfo),
    Function(ValueFunctionInfo),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ValueIntInfo {
    pub ty: Type,
    pub value: BigInt,
}

#[derive(Debug, Clone)]
pub struct ValueFunctionInfo {
    // only this field is used in hash and eq
    pub item_reference: ItemReference,
    
    pub ty: Type,
    pub params: Vec<Identifier>,
    pub body: FunctionBody,
}

impl Eq for ValueFunctionInfo {}

impl PartialEq for ValueFunctionInfo {
    fn eq(&self, other: &Self) -> bool {
        self.item_reference == other.item_reference
    }
}

impl Hash for ValueFunctionInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.item_reference.hash(state)
    }
}