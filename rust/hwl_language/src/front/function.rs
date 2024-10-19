use crate::data::compiled::{FunctionChecked, FunctionSignatureInfo, Item};
use crate::front::common::ExpressionContext;
use crate::front::driver::CompileState;
use crate::syntax::ast::ItemDefFunction;

impl CompileState<'_, '_> {
    pub fn check_function_body(&mut self, func_item: Item, funct_ast: &ItemDefFunction) -> FunctionChecked {
        let &ItemDefFunction { span: _, vis: _, id: ref func_id, params: _, ret_ty: ref ret_ty_ast, ref body } = funct_ast;
        let &FunctionSignatureInfo { scope_inner, ref ret_ty } = self.compiled.function_info.get(&func_item)
            .expect("signature and info should be resolved by now");
        let ret_ty = ret_ty.clone();

        // TODO check control flow (eg. require return on final block, warn on dead code)
        let ret_ty_span = ret_ty_ast.as_ref().map_or(func_id.span, |ty| ty.span);
        let mut ctx_func = ExpressionContext::FunctionBody { func_item, ret_ty_span, ret_ty };

        self.visit_block(&mut ctx_func, scope_inner, body);

        FunctionChecked {}
    }
}
