use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::new::compile::CompileState;
use crate::new::misc::{DomainSignal, ValueDomain};
use crate::new::types::Type;
use crate::new::value::{CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::throw;
use annotate_snippets::Level;

impl CompileState<'_> {
    pub fn type_of_expression_value(&self, value: &MaybeCompile<NamedValue>) -> Type {
        match value {
            MaybeCompile::Compile(c) => c.ty(),
            &MaybeCompile::Other(n) => self.type_of_named_value(n),
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

    pub fn check_type_contains_value(&self, assignment_span: Span, target_ty: Spanned<&Type>, value: Spanned<&MaybeCompile<NamedValue>>) -> Result<(), ErrorGuaranteed> {
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

    pub fn check_type_contains_compile_value(&self, assignment_span: Span, target_ty: Spanned<&Type>, value: Spanned<&CompileValue>, accept_undefined: bool) -> Result<(), ErrorGuaranteed> {
        let ty_contains_value = target_ty.inner.contains_type(&value.inner.ty());
        let value_is_undefined = matches!(value.inner, CompileValue::Undefined);

        if ty_contains_value && (accept_undefined || !value_is_undefined) {
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

    pub fn check_valid_domain_crossing(&self, assignment_span: Span, target: Spanned<&ValueDomain>, source: Spanned<&ValueDomain>, required_reason: &str) -> Result<(), ErrorGuaranteed> {
        let valid = match (target.inner, source.inner) {
            // clock only clock, and even that is not yet supported
            (ValueDomain::Clock, ValueDomain::Clock) =>
                throw!(self.diags.report_todo(assignment_span, "clock assignments")),
            (ValueDomain::Clock, _) => Err("non-clock to clock"),
            (ValueDomain::Sync(_), ValueDomain::Clock) => Err("clock to sync"),
            (ValueDomain::Sync(_), ValueDomain::Async) => Err("async to sync"),

            // compile-time from nothing else and to everything
            (_, ValueDomain::CompileTime) => Ok(()),
            (ValueDomain::CompileTime, _) => Err("compile-time to non-compile-time"),

            // async from everything
            (ValueDomain::Async, _) => Ok(()),

            // sync only if matching
            // TODO expand this to look though wires without logic
            (ValueDomain::Sync(target_inner), ValueDomain::Sync(source_inner)) => {
                let SyncDomain { clock: target_clock, reset: target_reset } = target_inner;
                let SyncDomain { clock: source_clock, reset: source_reset } = source_inner;

                match (target_clock == source_clock, target_reset == source_reset) {
                    (true, true) => Ok(()),
                    (false, true) => Err("different clock"),
                    (true, false) => Err("different reset"),
                    (false, false) => Err("different clock and reset"),
                }
            }
        };

        valid.map_err(|invalid_reason| {
            let target_str = self.value_domain_to_diagnostic_string(&target.inner);
            let source_str = self.value_domain_to_diagnostic_string(source.inner);
            let diag = Diagnostic::new(format!("invalid domain crossing: {invalid_reason}"))
                .add_error(assignment_span, "invalid domain crossing here")
                .add_info(target.span, format!("target domain is {target_str}"))
                .add_info(source.span, format!("source domain is {source_str}"))
                .footer(Level::Info, required_reason)
                .finish();
            self.diags.report(diag)
        })
    }

    fn value_domain_to_diagnostic_string(&self, domain: &ValueDomain) -> String {
        match domain {
            ValueDomain::CompileTime => "compile-time".to_string(),
            ValueDomain::Clock => "clock".to_string(),
            ValueDomain::Async => "async".to_string(),
            ValueDomain::Sync(sync) => {
                let SyncDomain { clock, reset } = sync;
                format!(
                    "sync({}, {})",
                    self.domain_signal_to_diagnostic_string(clock),
                    self.domain_signal_to_diagnostic_string(reset),
                )
            }
        }
    }

    fn domain_signal_to_diagnostic_string(&self, signal: &DomainSignal) -> String {
        match signal {
            &DomainSignal::Port(port) => self.ports[port].id.string.clone(),
            &DomainSignal::Wire(wire) => self.wires[wire].id.string().to_string(),
            &DomainSignal::Register(reg) => self.registers[reg].id.string().to_string(),
            DomainSignal::BoolNot(signal) => format!("!{}", self.domain_signal_to_diagnostic_string(signal)),
        }
    }
}
