use crate::data::compiled::{Item, Register, Wire};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::types::{GenericArguments, PortConnections};
use crate::front::values::Value;
use crate::syntax::ast::{Spanned, SyncDomain};
use crate::syntax::pos::Span;

// TODO include body comments for eg. the values that values were resolved to
#[derive(Debug, Clone)]
pub struct ModuleChecked {
    pub statements: Vec<ModuleStatement>,
    pub regs: Vec<(Register, Value)>,
    pub wires: Vec<(Wire, Option<Value>)>,
}

#[derive(Debug, Clone)]
pub enum ModuleStatement {
    Instance(ModuleInstance),
    Combinatorial(ModuleBlockCombinatorial),
    Clocked(ModuleBlockClocked),
    Err(ErrorGuaranteed),
}

#[derive(Debug, Clone)]
pub struct ModuleBlockCombinatorial {
    pub span: Span,

    // TODO include necessary temporary variables and other metadata
    pub statements: Vec<LowerStatement>,
}

#[derive(Debug, Clone)]
pub struct ModuleBlockClocked {
    pub span: Span,
    pub domain: SyncDomain<Value>,
    pub statements_reset: Vec<LowerStatement>,
    pub statements: Vec<LowerStatement>,
}

#[derive(Debug, Clone)]
pub struct ModuleInstance {
    pub module: Item,
    pub name: Option<String>,
    pub generic_arguments: Option<GenericArguments>,
    pub port_connections: PortConnections,
}

#[derive(Debug, Clone)]
pub enum LowerStatement {
    Assignment { target: Spanned<Value>, value: Spanned<Value> },
    Error(ErrorGuaranteed),
}
