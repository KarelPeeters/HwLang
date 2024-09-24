use crate::data::compiled::{FunctionChecked, Item};
use crate::front::driver::{CompileState, ResolveResult};
use crate::syntax::ast;

impl CompileState<'_, '_> {
    pub fn check_function_body(&mut self, func_item: Item, funct_ast: &ast::ItemDefFunction) -> ResolveResult<FunctionChecked> {
        // TODO
        let _ = func_item;
        let _ = funct_ast;

        Ok(FunctionChecked {})
    }
}
