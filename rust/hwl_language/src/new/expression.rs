use crate::new::compile::CompileState;
use crate::new::misc::TypeOrValue;
use crate::new::types::Type;
use crate::new::value::Value;
use crate::syntax::ast::Expression;

impl CompileState<'_> {
    pub fn eval_expression_as_ty_or_value(&mut self, expr: &Expression) -> TypeOrValue<Value> {
        todo!()
    }

    pub fn eval_expression_as_ty(&mut self, expr: &Expression) -> Type {
        todo!()
    }

    pub fn eval_expression_as_value(&mut self, expr: &Expression) -> Value {
        todo!()
    }
}