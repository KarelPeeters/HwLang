use crate::front::compile::{CompileItemContext, Port, Register, Variable, Wire};
use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::domain::ValueDomain;
use crate::front::types::HardwareType;
use crate::front::value::HardwareValue;
use crate::mid::ir::IrAssignmentTargetBase;
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;

// TODO expand to all possible values again?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Polarized<V> {
    pub inverted: bool,
    pub signal: V,
}

impl<V> Polarized<V> {
    pub fn new(signal: V) -> Self {
        Polarized {
            signal,
            inverted: false,
        }
    }

    pub fn invert(self) -> Self {
        Polarized {
            signal: self.signal,
            inverted: !self.inverted,
        }
    }

    pub fn map_inner<F, U>(self, f: F) -> Polarized<U>
    where
        F: FnOnce(V) -> U,
    {
        Polarized {
            signal: f(self.signal),
            inverted: self.inverted,
        }
    }

    pub fn try_map_inner<F, U, E>(self, f: F) -> Result<Polarized<U>, E>
    where
        F: FnOnce(V) -> Result<U, E>,
    {
        Ok(Polarized {
            signal: f(self.signal)?,
            inverted: self.inverted,
        })
    }
}

impl Polarized<Signal> {
    pub fn diagnostic_string(self, s: &CompileItemContext) -> String {
        let Polarized { inverted, signal } = self;
        let signal_str = signal.diagnostic_string(s);
        match inverted {
            false => signal_str.to_owned(),
            true => format!("(!{})", signal_str),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Signal {
    Port(Port),
    Wire(Wire),
    Register(Register),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum SignalOrVariable {
    Signal(Signal),
    Variable(Variable),
}

impl Signal {
    pub fn diagnostic_string<'c>(self, s: &'c CompileItemContext) -> &'c str {
        match self {
            Signal::Port(port) => &s.ports[port].name,
            Signal::Wire(wire) => s.wires[wire].id.diagnostic_str(),
            Signal::Register(reg) => s.registers[reg].id.diagnostic_str(),
        }
    }

    pub fn ty<'s>(
        self,
        diags: &Diagnostics,
        state: &'s mut CompileItemContext,
        use_span: Span,
    ) -> Result<Spanned<&'s HardwareType>, ErrorGuaranteed> {
        match self {
            Signal::Port(port) => Ok(state.ports[port].ty.as_ref()),
            Signal::Wire(wire) => state.wires[wire].typed(diags, use_span).map(|typed| typed.ty.as_ref()),
            Signal::Register(reg) => Ok(state.registers[reg].ty.as_ref()),
        }
    }

    pub fn suggest_domain(
        self,
        state: &mut CompileItemContext,
        suggest_domain: Spanned<ValueDomain>,
    ) -> Result<Spanned<ValueDomain>, ErrorGuaranteed> {
        let diags = state.refs.diags;
        match self {
            Signal::Port(port) => Ok(state.ports[port].domain.map_inner(ValueDomain::from_port_domain)),
            Signal::Wire(wire) => state.wires[wire].suggest_domain(suggest_domain),
            Signal::Register(reg) => {
                let reg_info = &mut state.registers[reg];
                if let Some(domain) = reg_info.domain? {
                    Ok(domain.map_inner(ValueDomain::Sync))
                } else if let ValueDomain::Sync(suggest_domain_inner) = suggest_domain.inner {
                    let reg_domain = reg_info.suggest_domain(Spanned::new(suggest_domain.span, suggest_domain_inner));
                    Ok(reg_domain.map_inner(ValueDomain::Sync))
                } else {
                    Err(diags.report_internal_error(suggest_domain.span, "suggesting non-sync domain for register"))
                }
            }
        }
    }

    pub fn domain(self, state: &mut CompileItemContext, span: Span) -> Result<Spanned<ValueDomain>, ErrorGuaranteed> {
        let diags = state.refs.diags;
        match self {
            Signal::Port(port) => Ok(state.ports[port].domain.map_inner(ValueDomain::from_port_domain)),
            Signal::Wire(wire) => state.wires[wire].domain(diags, span),
            Signal::Register(reg) => state.registers[reg]
                .domain(diags, span)
                .map(|d| d.map_inner(ValueDomain::Sync)),
        }
    }

    pub fn as_ir_target_base(
        self,
        diags: &Diagnostics,
        state: &mut CompileItemContext,
        use_span: Span,
    ) -> Result<IrAssignmentTargetBase, ErrorGuaranteed> {
        match self {
            Signal::Port(port) => Ok(IrAssignmentTargetBase::Port(state.ports[port].ir)),
            Signal::Wire(wire) => Ok(IrAssignmentTargetBase::Wire(
                state.wires[wire].typed(diags, use_span)?.ir,
            )),
            Signal::Register(reg) => Ok(IrAssignmentTargetBase::Register(state.registers[reg].ir)),
        }
    }

    pub fn as_hardware_value(
        self,
        state: &mut CompileItemContext,
        span: Span,
    ) -> Result<HardwareValue, ErrorGuaranteed> {
        let diags = state.refs.diags;
        match self {
            Signal::Port(port) => Ok(state.ports[port].as_hardware_value()),
            Signal::Wire(wire) => state.wires[wire].as_hardware_value(diags, span),
            Signal::Register(reg) => state.registers[reg].as_hardware_value(diags, span),
        }
    }
}
