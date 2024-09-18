use crate::data::compiled::{Item, ModulePort};
use crate::front::types::Type;
use crate::front::values::Value;
use crate::syntax::ast::SyncDomain;

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
    // TODO include necessary temporary variables and other metadata
    pub statements: Vec<CombinatorialStatement>,
}

#[derive(Debug)]
pub enum CombinatorialStatement {
    PortPortAssignment(ModulePort, ModulePort),
}

#[derive(Debug)]
pub struct ModuleBlockClocked {
    regs: Vec<BlockReg>,

    // TODO how to implement ports that are just registers?
    on_reset: (), // TODO IR 
    on_block: (), // TODO IR
}

#[derive(Debug)]
pub struct BlockReg {
    ty: Type,
}
