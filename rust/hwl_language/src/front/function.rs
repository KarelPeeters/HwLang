use crate::data::compiled::{FunctionChecked, FunctionSignatureInfo, Item};
use crate::front::common::{ContextDomain, ExpressionContext, ValueDomainKind};
use crate::front::driver::CompileState;
use crate::front::module::MaybeDriverCollector;
use crate::syntax::ast::{ItemDefFunction, Spanned};

impl CompileState<'_, '_> {
    pub fn check_function_body(&mut self, func_item: Item, funct_ast: &ItemDefFunction) -> FunctionChecked {
        let &ItemDefFunction { span: _, vis: _, id: _, params: _, ret_ty: ref ret_ty_ast, ref body } = funct_ast;
        let &FunctionSignatureInfo { scope_inner, ref ret_ty } = self.compiled.function_info.get(&func_item)
            .expect("signature and info should be resolved by now");

        let domain = Spanned {
            span: funct_ast.body.span,
            inner: &ValueDomainKind::FunctionBody(func_item),
        };
        let function_return_ty = Spanned {
            span: ret_ty_ast.as_ref().map_or(funct_ast.span, |t| t.span),
            inner: &ret_ty.clone(),
        };
        let ctx_func = ExpressionContext {
            scope: scope_inner,
            domain: ContextDomain::Specific(domain),
            function_return_ty: Some(function_return_ty),
        };

        // TODO check control flow (eg. require return on final block, warn on dead code)
        let lower_block = self.visit_block(&ctx_func, &mut MaybeDriverCollector::None, body);

        FunctionChecked {
            block: lower_block,
        }
    }
}
