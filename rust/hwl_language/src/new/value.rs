use crate::data::diagnostic::ErrorGuaranteed;
use num_bigint::BigInt;

// don't implement eq for this type
#[derive(Debug, Clone)]
pub enum Value {
    Compile(CompileValue),
    // TODO add ports, wires, regs, variables
}

#[derive(Debug, Clone)]
pub enum CompileValue {
    Known(KnownCompileValue),
    Unchecked(UncheckedValue),
    Error(ErrorGuaranteed),
}

#[derive(Debug, Clone)]
pub struct UncheckedValue(());

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum KnownCompileValue {
    Bool(bool),
    Int(BigInt),
    String(String),
    Array(Vec<KnownCompileValue>),
    // TODO list, tuple, struct, function, module (once we allow passing modules as generics)
}

impl From<ErrorGuaranteed> for CompileValue {
    fn from(e: ErrorGuaranteed) -> Self {
        CompileValue::Error(e)
    }
}
