use num_bigint::BigInt;
use crate::new_index_type;
use crate::resolve::types::Type;
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
    Int(ValueIntInfo),
    Function(ValueFunctionInfo),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ValueIntInfo {
    pub ty: Type,
    pub value: BigInt,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ValueFunctionInfo {
    pub ty: Type,
    // TODO value is the parsed and fully resolved body IR/AST
}
