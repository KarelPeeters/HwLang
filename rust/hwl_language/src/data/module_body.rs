use crate::data::compiled::{Item, ModulePort, Register};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::types::{GenericArguments, PortConnections};
use crate::front::values::Value;
use crate::syntax::ast::SyncDomain;
use crate::syntax::pos::Span;

// TODO include body comments for eg. the values that values were resolved to
#[derive(Debug)]
pub struct ModuleChecked {
    pub statements: Vec<ModuleStatement>,
    pub regs: Vec<(Register, Value)>,
}

#[derive(Debug)]
pub enum ModuleStatement {
    Instance(ModuleInstance),
    Combinatorial(ModuleBlockCombinatorial),
    Clocked(ModuleBlockClocked),
    Err(ErrorGuaranteed),
}

#[derive(Debug)]
pub struct ModuleBlockCombinatorial {
    pub span: Span,

    // TODO include necessary temporary variables and other metadata
    pub statements: Vec<LowerStatement>,
}

#[derive(Debug)]
pub struct ModuleBlockClocked {
    pub span: Span,
    pub domain: SyncDomain<Value>,

    // TODO how to implement ports that are just registers?
    pub on_reset: Vec<LowerStatement>, // TODO IR
    pub on_block: Vec<LowerStatement>, // TODO IR
}

#[derive(Debug)]
pub struct ModuleInstance {
    pub module: Item,
    pub name: Option<String>,
    pub generic_arguments: Option<GenericArguments>,
    pub port_connections: PortConnections,
}

#[derive(Debug)]
pub enum LowerStatement {
    PortPortAssignment(ModulePort, ModulePort),
    Error(ErrorGuaranteed),
}
