use crate::front::block::TypedIrExpression;
use crate::front::compile::{Port, Register, Variable, Wire};
use crate::front::ir::{IrAssignmentTargetBase, IrExpression};
use crate::front::types::HardwareType;
use crate::front::value::NamedValue;
use crate::syntax::ast::{DomainKind, Spanned, SyncDomain};
use crate::syntax::parsed::AstRefItem;

use super::compile::CompileStateLong;

// TODO move this to a better place
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(AstRefItem),
    Direct(NamedValue),
}

pub type DomainSignal = Polarized<Signal>;

// TODO expand to all possible values again?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Polarized<V> {
    pub inverted: bool,
    pub signal: V,
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

#[derive(Debug, Copy, Clone)]
pub enum PortDomain {
    Clock,
    Kind(DomainKind<Polarized<Port>>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ValueDomain<V = Polarized<Signal>> {
    CompileTime,
    Clock,
    // TODO allow separate sync/async per edge, necessary for "async" reset
    Async,
    Sync(SyncDomain<V>),
}

impl ValueDomain {
    pub fn join(&self, other: &Self) -> Self {
        // TODO expand signal equality check, eg. make it look through wire assignments
        match (self, other) {
            (ValueDomain::CompileTime, other) | (other, ValueDomain::CompileTime) => other.clone(),
            (&ValueDomain::Sync(left), &ValueDomain::Sync(right)) if left == right => ValueDomain::Sync(left),
            _ => ValueDomain::Async,
        }
    }

    pub fn from_port_domain(domain: PortDomain) -> Self {
        match domain {
            PortDomain::Clock => ValueDomain::Clock,
            PortDomain::Kind(kind) => match kind {
                DomainKind::Async => ValueDomain::Async,
                DomainKind::Sync(sync) => ValueDomain::Sync(SyncDomain {
                    clock: sync.clock.map_inner(Signal::Port),
                    reset: sync.reset.map_inner(Signal::Port),
                }),
            },
        }
    }

    pub fn to_diagnostic_string(&self, s: &CompileStateLong) -> String {
        match self {
            ValueDomain::CompileTime => "compile-time".to_owned(),
            ValueDomain::Clock => "clock".to_owned(),
            ValueDomain::Async => "async".to_owned(),
            ValueDomain::Sync(sync) => sync.to_diagnostic_string(s),
        }
    }
}

impl<V> ValueDomain<V> {
    pub fn from_domain_kind(domain: DomainKind<V>) -> Self {
        match domain {
            DomainKind::Async => ValueDomain::Async,
            DomainKind::Sync(sync) => ValueDomain::Sync(SyncDomain {
                clock: sync.clock,
                reset: sync.reset,
            }),
        }
    }
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
    pub fn to_diagnostic_string(self, s: &CompileStateLong) -> String {
        let Polarized { inverted, signal } = self;
        let signal_str = signal.to_diagnostic_string(s);
        match inverted {
            false => signal_str,
            true => format!("(!{})", signal_str),
        }
    }
}

impl Signal {
    pub fn to_diagnostic_string(self, s: &CompileStateLong) -> String {
        match self {
            Signal::Port(port) => s.ports[port].id.string.clone(),
            Signal::Wire(wire) => s.wires[wire].id.string().to_owned(),
            Signal::Register(reg) => s.registers[reg].id.string().to_owned(),
        }
    }

    pub fn ty<'s>(self, state: &'s CompileStateLong) -> Spanned<&'s HardwareType> {
        match self {
            Signal::Port(port) => state.ports[port].ty.as_ref(),
            Signal::Wire(wire) => state.wires[wire].ty.as_ref(),
            Signal::Register(reg) => state.registers[reg].ty.as_ref(),
        }
    }

    pub fn domain(self, state: &CompileStateLong) -> Spanned<ValueDomain> {
        match self {
            Signal::Port(port) => state.ports[port]
                .domain
                .clone()
                .map_inner(ValueDomain::from_port_domain),
            Signal::Wire(wire) => state.wires[wire].domain.clone(),
            Signal::Register(reg) => state.registers[reg].domain.clone().map_inner(ValueDomain::Sync),
        }
    }

    pub fn as_ir_target_base(self, state: &CompileStateLong) -> IrAssignmentTargetBase {
        match self {
            Signal::Port(port) => IrAssignmentTargetBase::Port(state.ports[port].ir),
            Signal::Wire(wire) => IrAssignmentTargetBase::Wire(state.wires[wire].ir),
            Signal::Register(reg) => IrAssignmentTargetBase::Register(state.registers[reg].ir),
        }
    }

    pub fn as_ir_expression(self, state: &CompileStateLong) -> TypedIrExpression {
        match self {
            Signal::Port(port) => {
                let port_info = &state.ports[port];
                TypedIrExpression {
                    ty: port_info.ty.inner.clone(),
                    domain: ValueDomain::from_port_domain(port_info.domain.inner.clone()),
                    expr: IrExpression::Port(port_info.ir),
                }
            }
            Signal::Wire(wire) => {
                let wire_info = &state.wires[wire];
                TypedIrExpression {
                    ty: wire_info.ty.inner.clone(),
                    domain: wire_info.domain.inner.clone(),
                    expr: IrExpression::Wire(wire_info.ir),
                }
            }
            Signal::Register(reg) => {
                let reg_info = &state.registers[reg];
                TypedIrExpression {
                    ty: reg_info.ty.inner.clone(),
                    domain: ValueDomain::Sync(reg_info.domain.inner.clone()),
                    expr: IrExpression::Register(reg_info.ir),
                }
            }
        }
    }
}

impl PortDomain {
    pub fn to_diagnostic_string(self, s: &CompileStateLong) -> String {
        ValueDomain::from_port_domain(self).to_diagnostic_string(s)
    }
}

impl DomainKind<Polarized<Signal>> {
    pub fn to_diagnostic_string(&self, s: &CompileStateLong) -> String {
        match self {
            DomainKind::Async => "async".to_owned(),
            DomainKind::Sync(sync) => sync.to_diagnostic_string(s),
        }
    }
}

impl SyncDomain<Polarized<Signal>> {
    pub fn to_diagnostic_string(&self, s: &CompileStateLong) -> String {
        format!(
            "sync({}, {})",
            self.clock.to_diagnostic_string(s),
            self.reset.to_diagnostic_string(s)
        )
    }
}

impl SyncDomain<Polarized<Port>> {
    pub fn to_diagnostic_string(&self, s: &CompileStateLong) -> String {
        self.map_inner(|p| p.map_inner(Signal::Port)).to_diagnostic_string(s)
    }
}
