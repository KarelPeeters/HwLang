use crate::front::block::TypedIrExpression;
use crate::front::compile::{CompileItemContext, Port, Register, Variable, Wire};
use crate::front::ir::{IrAssignmentTargetBase, IrExpression};
use crate::front::types::HardwareType;
use crate::front::value::NamedValue;
use crate::syntax::ast::{DomainKind, Spanned, SyncDomain};
use crate::syntax::parsed::AstRefItem;

// TODO move this to a better place
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    /// Indirection though an item, the item should be evaluated.
    Item(AstRefItem),
    /// A named value: port, register, wire, variable.
    /// These are not fully evaluated immediately, they might be used symbolically
    ///   as assignment targets or in domain expressions.
    Named(NamedValue),
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
pub enum PortDomain<P> {
    Clock,
    Kind(DomainKind<Polarized<P>>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ValueDomain<V = Polarized<Signal>> {
    CompileTime,
    Clock,
    Const,
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

    pub fn from_port_domain(domain: PortDomain<Port>) -> Self {
        match domain {
            PortDomain::Clock => ValueDomain::Clock,
            PortDomain::Kind(kind) => match kind {
                DomainKind::Const => ValueDomain::Const,
                DomainKind::Async => ValueDomain::Async,
                DomainKind::Sync(sync) => ValueDomain::Sync(SyncDomain {
                    clock: sync.clock.map_inner(Signal::Port),
                    reset: sync.reset.map(|p| p.map_inner(Signal::Port)),
                }),
            },
        }
    }

    pub fn invert(&self) -> ValueDomain {
        // TODO once we properly track partial domains (one for the posedge, one for the negedge),
        //   we need to actually do something here
        // TODO when we do that make sure to only do this for bools and bool arrays,
        //   and once a non-trivial operation happens it needs to delay to the worst side.
        //   Maybe implement that by changing the current join to assume the worst,
        //   and adding a separate join_simple that propagates the edges more carefully.
        self.clone()
    }

    pub fn to_diagnostic_string(&self, s: &CompileItemContext) -> String {
        match self {
            ValueDomain::CompileTime => "compile-time".to_owned(),
            ValueDomain::Clock => "clock".to_owned(),
            ValueDomain::Const => "const".to_owned(),
            ValueDomain::Async => "async".to_owned(),
            ValueDomain::Sync(sync) => sync.to_diagnostic_string(s),
        }
    }
}

impl<V> ValueDomain<V> {
    pub fn from_domain_kind(domain: DomainKind<V>) -> Self {
        match domain {
            DomainKind::Const => ValueDomain::Const,
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

impl Signal {
    pub fn to_diagnostic_string(self, s: &CompileItemContext) -> String {
        match self {
            Signal::Port(port) => s.ports[port].id.string.clone(),
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

    pub fn as_ir_expression(self, state: &CompileItemContext) -> TypedIrExpression {
        match self {
            Signal::Port(port) => {
                let port_info = &state.ports[port];
                TypedIrExpression {
                    ty: port_info.ty.inner.clone(),
                    domain: ValueDomain::from_port_domain(port_info.domain.inner),
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
                    domain: ValueDomain::Sync(reg_info.domain.inner),
                    expr: IrExpression::Register(reg_info.ir),
                }
            }
        }
    }
}

impl<P> PortDomain<P> {
    pub fn map_inner<Q>(self, mut f: impl FnMut(P) -> Q) -> PortDomain<Q> {
        match self {
            PortDomain::Clock => PortDomain::Clock,
            PortDomain::Kind(kind) => PortDomain::Kind(kind.map_signal(|p: Polarized<P>| p.map_inner(&mut f))),
        }
    }
}

impl PortDomain<Port> {
    pub fn to_diagnostic_string(self, s: &CompileItemContext) -> String {
        ValueDomain::from_port_domain(self).to_diagnostic_string(s)
    }
}

impl DomainKind<Polarized<Signal>> {
    pub fn to_diagnostic_string(&self, s: &CompileItemContext) -> String {
        match self {
            DomainKind::Const => "const".to_owned(),
            DomainKind::Async => "async".to_owned(),
            DomainKind::Sync(sync) => sync.to_diagnostic_string(s),
        }
    }
}

impl SyncDomain<Polarized<Signal>> {
    pub fn to_diagnostic_string(&self, s: &CompileItemContext) -> String {
        let SyncDomain { clock, reset } = self;

        match reset {
            None => format!("sync({})", clock.to_diagnostic_string(s)),
            Some(reset) => format!(
                "sync({}, {})",
                clock.to_diagnostic_string(s),
                reset.to_diagnostic_string(s)
            ),
        }
    }
}

impl SyncDomain<Polarized<Port>> {
    pub fn to_diagnostic_string(&self, s: &CompileItemContext) -> String {
        self.map_signal(|p| p.map_inner(Signal::Port)).to_diagnostic_string(s)
    }
}
