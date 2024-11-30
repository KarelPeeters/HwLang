use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::new::compile::CompileState;
use crate::new::types::Type;
use crate::new::value::{CompileValue, ExpressionValue, NamedValue};
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;

impl CompileState<'_> {
    pub fn type_of_expression_value(&self, value: &ExpressionValue<NamedValue>) -> Type {
        match value {
            ExpressionValue::Compile(c) => c.ty(),
            &ExpressionValue::Other(n) => self.type_of_named_value(n),
        }
    }

    // TODO this function is a bit weird, usually you'd just ask the type _after_ context evaluation
    pub fn type_of_named_value(&self, value: NamedValue) -> Type {
        match value {
            NamedValue::Constant(cst) => self.constants[cst].value.ty(),
            NamedValue::Parameter(param) => self.parameters[param].value.ty(),
            // TODO for compile-time variables, just look at the value itself
            NamedValue::Variable(var) => self.variables[var].ty.clone(),
            NamedValue::Port(port) => self.ports[port].ty.inner.as_type(),
            NamedValue::Wire(wire) => self.wires[wire].ty.inner.as_type(),
            NamedValue::Register(reg) => self.registers[reg].ty.inner.as_type(),
        }
    }

    pub fn check_type_contains_value(&self, assignment_span: Span, target_ty: Spanned<&Type>, value: Spanned<&ExpressionValue<NamedValue>>) -> Result<(), ErrorGuaranteed> {
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
