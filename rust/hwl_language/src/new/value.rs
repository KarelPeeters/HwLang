use crate::new::misc::{MaybeUnchecked, Unchecked};
use num_bigint::BigInt;

// don't implement eq for this type
#[derive(Debug, Clone)]
pub enum Value<U = Unchecked> {
    Compile(MaybeUnchecked<KnownCompileValue, U>),
    // TODO add ports, wires, regs, variables
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum KnownCompileValue {
    Bool(bool),
    Int(BigInt),
    String(String),
    Array(Vec<KnownCompileValue>),
    // TODO list, tuple, struct, function, module (once we allow passing modules as generics)
}
