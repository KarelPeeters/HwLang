use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::new::compile::CompileState;
use crate::new::types::Type;
use crate::new::value::{CompileValue, ExpressionValue};
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;

impl CompileState<'_> {
    pub fn type_of_expression_value(&self, value: &ExpressionValue) -> Type {
        match value {
            ExpressionValue::Undefined => Type::Undefined,
            ExpressionValue::Compile(c) => c.ty(),
            &ExpressionValue::Port(port) => self.ports[port].ty.inner.as_type(),
            &ExpressionValue::Wire(wire) => self.wires[wire].ty.inner.as_type(),
            &ExpressionValue::Register(reg) => self.registers[reg].ty.inner.as_type(),
            &ExpressionValue::Variable(var) => self.variables[var].ty.clone(),
            ExpressionValue::Expression { ty, domain: _ } => ty.clone()
        }
    }

    pub fn check_type_contains_value(&self, assignment_span: Span, target_ty: Spanned<&Type>, value: Spanned<&ExpressionValue>) -> Result<(), ErrorGuaranteed> {
        let value_ty = self.type_of_expression_value(&value.inner);
        if target_ty.inner.contains_type(&value_ty) {
            Ok(())
        } else {
            let diag = Diagnostic::new("value does not fit in type")
                .add_error(assignment_span, "invalid assignment here")
                .add_info(target_ty.span, format!("target type `{}` defined here", target_ty.inner.to_diagnostic_string()))
                .add_info(value.span, format!("source value with type `{}` defined here", value_ty.to_diagnostic_string()))
                .finish();
            Err(self.diags.report(diag))
        }
    }

    pub fn check_type_contains_compile_value(&self, assignment_span: Span, target_ty: Spanned<&Type>, value: Spanned<&CompileValue>) -> Result<(), ErrorGuaranteed> {
        if target_ty.inner.contains_type(&value.inner.ty()) {
            Ok(())
        } else {
            let diag = Diagnostic::new("value does not fit in type")
                .add_error(assignment_span, "invalid assignment here")
                .add_info(target_ty.span, format!("target type `{}` defined here", target_ty.inner.to_diagnostic_string()))
                .add_info(value.span, format!("source value `{}` defined here", value.inner.to_diagnostic_string()))
                .finish();
            Err(self.diags.report(diag))
        }
    }

    pub fn check_type_contains_type(&self, assignment_span: Span, target_ty: Spanned<&Type>, source_ty: Spanned<&Type>) -> Result<(), ErrorGuaranteed> {
        if target_ty.inner.contains_type(&source_ty.inner) {
            Ok(())
        } else {
            let diag = Diagnostic::new("type does not fit in type")
                .add_error(assignment_span, "invalid assignment here")
                .add_info(target_ty.span, format!("target type `{}` defined here", target_ty.inner.to_diagnostic_string()))
                .add_info(source_ty.span, format!("source type `{}` defined here", source_ty.inner.to_diagnostic_string()))
                .finish();
            Err(self.diags.report(diag))
        }
    }
}
