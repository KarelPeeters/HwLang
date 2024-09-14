use crate::data::compiled::ModulePort;

#[derive(Debug)]
pub struct ModuleBody {
    pub blocks: Vec<ModuleBlock>,
}

#[derive(Debug)]
pub enum ModuleBlock {
    Combinatorial(ModuleBlockCombinatorial),
    Clocked(ModuleBlockClocked),
}

#[derive(Debug)]
pub struct ModuleBlockCombinatorial {
    // TODO include necessary temporary variables and other metadata
    pub statements: Vec<CombinatorialStatement>,
}

#[derive(Debug)]
pub struct ModuleBlockClocked {
    // TODO
}

#[derive(Debug)]
pub enum CombinatorialStatement {
    PortPortAssignment(ModulePort, ModulePort),
}
