use crate::new::compile::CompileState;
use crate::new::misc::{DomainSignal, TypeOrValue};
use crate::new::types::Type;
use crate::new::value::ExpressionValue;
use crate::syntax::ast::{DomainKind, Expression};

impl CompileState<'_> {
    pub fn eval_expression_as_ty_or_value(&mut self, _: &Expression) -> TypeOrValue<ExpressionValue> {
        todo!()
    }

    pub fn eval_expression_as_ty(&mut self, _: &Expression) -> Type {
        todo!()
    }

    pub fn eval_expression_as_value(&mut self, _: &Expression) -> ExpressionValue {
        todo!()
    }

    pub fn eval_domain(&mut self, domain: &DomainKind<Box<Expression>>) -> DomainKind<DomainSignal> {
        todo!()
    }
}