use crate::data::compiled::{FunctionChecked, FunctionSignatureInfo, Item};
use crate::front::common::ExpressionContext;
use crate::front::driver::{CompileState, ResolveResult};
use crate::syntax::ast::ItemDefFunction;

impl CompileState<'_, '_> {
    pub fn check_function_body(&mut self, func_item: Item, funct_ast: &ItemDefFunction) -> ResolveResult<FunctionChecked> {
        let ItemDefFunction { span: _, vis: _, id: _, params: _, ret_ty: _, body } = funct_ast;
        let &FunctionSignatureInfo { scope_inner, ref ret_ty } = self.compiled.function_info.get(&func_item)
            .expect("signature and info should be resolved by now");
        let ret_ty = ret_ty.clone();

        // TODO check return type (and control flow)
        // TODO check control flow
        let _ = ret_ty;

        let ctx_func = ExpressionContext::FunctionBody(func_item);
        self.visit_block(ctx_func, scope_inner, body)?;

        Ok(FunctionChecked {})
    }
}
