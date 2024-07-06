use std::hash::{Hash, Hasher};
use num_bigint::BigInt;
use crate::new_index_type;
use crate::resolve::compile::{FunctionBody, Item, ItemReference};
use crate::resolve::types::{Type, TypeUnique};
use crate::syntax::ast::Identifier;
use crate::util::arena::ArenaSet;

// TODO find a better name for this, eg. InterpreterValue, CompileValue, just Value, ...
new_index_type!(pub Value);

// TODO this should probably have either garbage collection or no arena at all:
//   we'll be running bytecode which can generate a large number of intermediate eg. integers
//   alternatively we can keep the arena for fixed things (like types and signatures)
//   but not for values (ints, arrays, ...)
// TODO why is this an arena again?
pub type Values = ArenaSet<Value, ValueInfo>;

// TODO should all values have types? or can eg. ints just be free abstract objects?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ValueInfo {
    Type(Type),
    Item(Item),

    Int(BigInt),
    Function(ValueFunctionInfo),
    Module(ValueModuleInfo),

    // TODO should this be a dedicated type or just an instance of the normal range struct?
    Range { start: Option<BigInt>, end: Option<BigInt> },
}

#[derive(Debug, Clone)]
pub struct ValueFunctionInfo {
    // only this field is used in hash and eq
    // TODO also include captured values once those exist
    pub item_reference: ItemReference,
    
    pub ty: Type,
    pub params: Vec<Identifier>,
    pub body: FunctionBody,
}

#[derive(Debug, Clone)]
pub struct ValueModuleInfo {
    // only this field is used in hash and eq
    pub unique: TypeUnique,
    // TODO include real content
    pub ty: Type,
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

impl Eq for ValueModuleInfo {}

impl PartialEq for ValueModuleInfo {
    fn eq(&self, other: &Self) -> bool {
        self.unique == other.unique
    }
}

impl Hash for ValueModuleInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique.hash(state)
    }
}