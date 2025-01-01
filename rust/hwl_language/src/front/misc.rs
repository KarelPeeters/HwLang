use crate::front::compile::{CompileState, Port, Register, Wire};
use crate::front::value::NamedValue;
use crate::syntax::ast::{DomainKind, SyncDomain};
use crate::syntax::parsed::AstRefItem;

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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Signal {
    Port(Port),
    Wire(Wire),
    Register(Register),
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

    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
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
    pub fn to_diagnostic_string(self, s: &CompileState) -> String {
        let Polarized { inverted, signal } = self;

        let signal_str = signal.to_diagnostic_string(s);
        match inverted {
            false => signal_str,
            true => format!("(!{})", signal_str),
        }
    }
}

impl Signal {
    pub fn to_diagnostic_string(self, s: &CompileState) -> String {
        match self {
            Signal::Port(port) => s.ports[port].id.string.clone(),
            Signal::Wire(wire) => s.wires[wire].id.string().to_owned(),
            Signal::Register(reg) => s.registers[reg].id.string().to_owned(),
        }
    }
}

impl PortDomain {
    pub fn to_diagnostic_string(self, s: &CompileState) -> String {
        ValueDomain::from_port_domain(self).to_diagnostic_string(s)
    }
}

impl DomainKind<Polarized<Signal>> {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        match self {
            DomainKind::Async => "async".to_owned(),
            DomainKind::Sync(sync) => sync.to_diagnostic_string(s),
        }
    }
}

impl SyncDomain<Polarized<Signal>> {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        format!(
            "sync({}, {})",
            self.clock.to_diagnostic_string(s),
            self.reset.to_diagnostic_string(s)
        )
    }
}

impl SyncDomain<Polarized<Port>> {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        self.map_inner(|p| p.map_inner(Signal::Port)).to_diagnostic_string(s)
    }
}
