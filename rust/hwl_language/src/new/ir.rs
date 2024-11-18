use crate::data::diagnostic::ErrorGuaranteed;
use crate::new_index_type;
use crate::util::arena::Arena;

#[derive(Debug)]
pub struct IrDesign {
    pub top_module: Result<IrModule, ErrorGuaranteed>,
    pub modules: Arena<IrModule, IrModuleContent>,
}

new_index_type!(pub IrModule);

#[derive(Debug)]
pub struct IrModuleContent {}
