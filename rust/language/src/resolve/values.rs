use crate::new_index_type;
use crate::util::arena::ArenaSet;

// TODO find a better name for this, eg. InterpreterValue, CompileValue, just Value, ...
new_index_type!(pub ResolvedValue);

// TODO this should probably have either garbage collection or no arena at all:
//   we'll be running bytecode which can generate a large number of intermediate eg. integers
//   alternatively we can keep the arena for fixed things (like types and signatures)
//   but not for values (ints, arrays, ...)
pub type ResolvedValues = ArenaSet<ResolvedValue, ResolvedValueInfo>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ResolvedValueInfo {
    SignatureType,
}