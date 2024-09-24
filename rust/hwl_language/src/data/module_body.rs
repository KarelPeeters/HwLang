use crate::data::compiled::{Item, ModulePort};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::types::Type;
use crate::front::values::Value;
use crate::syntax::ast::SyncDomain;
use crate::syntax::pos::Span;

// TODO include body comments for eg. the values that values were resolved to
#[derive(Debug)]
pub struct ModuleBody {
    pub blocks: Vec<ModuleBlockInfo>,
    pub regs: Vec<ModuleRegInfo>,
}

#[derive(Debug)]
pub enum ModuleBlockInfo {
    Combinatorial(ModuleBlockCombinatorial),
    Clocked(ModuleBlockClocked),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ModuleReg {
    pub module_item: Item,
    pub index: usize,
}

#[derive(Debug)]
pub struct ModuleRegInfo {
    pub sync: SyncDomain<Value>,
    pub ty: Type,
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

    pub clock: Value,
    pub reset: Value,

    // TODO how to implement ports that are just registers?
    pub on_reset: Vec<LowerStatement>, // TODO IR
    pub on_block: Vec<LowerStatement>, // TODO IR
}

#[derive(Debug)]
pub enum LowerStatement {
    PortPortAssignment(ModulePort, ModulePort),
    Error(ErrorGuaranteed),
}
