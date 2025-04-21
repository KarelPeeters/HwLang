use crate::front::compile::{CompileItemContext, Port, Register, Variable, Wire};
use crate::front::domain::ValueDomain;
use crate::front::types::HardwareType;
use crate::front::value::HardwareValue;
use crate::mid::ir::{IrAssignmentTargetBase, IrExpression};
use crate::syntax::ast::Spanned;

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
    pub fn domain(&self, s: &CompileItemContext) -> Spanned<ValueDomain> {
        let &Polarized { inverted, signal } = self;
        let signal_domain = signal.domain(s);
        match inverted {
            false => signal_domain,
            true => signal_domain.as_ref().map_inner(ValueDomain::invert),
        }
    }

    pub fn to_diagnostic_string(self, s: &CompileItemContext) -> String {
        let Polarized { inverted, signal } = self;
        let signal_str = signal.to_diagnostic_string(s);
        match inverted {
            false => signal_str,
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
    pub fn to_diagnostic_string(self, s: &CompileItemContext) -> String {
        match self {
            Signal::Port(port) => s.ports[port].name.clone(),
            Signal::Wire(wire) => s.wires[wire].id.string().to_owned(),
            Signal::Register(reg) => s.registers[reg].id.string().to_owned(),
        }
    }

    pub fn ty<'s>(self, state: &'s CompileItemContext) -> Spanned<&'s HardwareType> {
        match self {
            Signal::Port(port) => state.ports[port].ty.as_ref(),
            Signal::Wire(wire) => state.wires[wire].ty.as_ref(),
            Signal::Register(reg) => state.registers[reg].ty.as_ref(),
        }
    }

    pub fn domain(self, state: &CompileItemContext) -> Spanned<ValueDomain> {
        match self {
            Signal::Port(port) => state.ports[port].domain.map_inner(ValueDomain::from_port_domain),
            Signal::Wire(wire) => state.wires[wire].domain.clone(),
            Signal::Register(reg) => state.registers[reg].domain.map_inner(ValueDomain::Sync),
        }
    }

    pub fn as_ir_target_base(self, state: &CompileItemContext) -> IrAssignmentTargetBase {
        match self {
            Signal::Port(port) => IrAssignmentTargetBase::Port(state.ports[port].ir),
            Signal::Wire(wire) => IrAssignmentTargetBase::Wire(state.wires[wire].ir),
            Signal::Register(reg) => IrAssignmentTargetBase::Register(state.registers[reg].ir),
        }
    }

    pub fn as_ir_expression(self, state: &CompileItemContext) -> HardwareValue {
        match self {
            Signal::Port(port) => {
                let port_info = &state.ports[port];
                HardwareValue {
                    ty: port_info.ty.inner.clone(),
                    domain: ValueDomain::from_port_domain(port_info.domain.inner),
                    expr: IrExpression::Port(port_info.ir),
                }
            }
            Signal::Wire(wire) => {
                let wire_info = &state.wires[wire];
                HardwareValue {
                    ty: wire_info.ty.inner.clone(),
                    domain: wire_info.domain.inner.clone(),
                    expr: IrExpression::Wire(wire_info.ir),
                }
            }
            Signal::Register(reg) => {
                let reg_info = &state.registers[reg];
                HardwareValue {
                    ty: reg_info.ty.inner.clone(),
                    domain: ValueDomain::Sync(reg_info.domain.inner),
                    expr: IrExpression::Register(reg_info.ir),
                }
            }
        }
    }
}
