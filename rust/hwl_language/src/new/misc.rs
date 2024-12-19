use crate::data::parsed::AstRefItem;
use crate::new::compile::{CompileState, Port, Register, Wire};
use crate::new::value::NamedValue;
use crate::syntax::ast::{DomainKind, SyncDomain};

// TODO move this to a better place
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(AstRefItem),
    Direct(NamedValue),
}

// TODO expand to all possible values again
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DomainSignal {
    Port(Port),
    Wire(Wire),
    Register(Register),
    // TODO make invert a common struct field instead of a boxed variant?
    BoolNot(Box<DomainSignal>),
}

#[derive(Debug, Clone)]
pub enum PortDomain<V> {
    Clock,
    Kind(DomainKind<V>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ValueDomain<V = DomainSignal> {
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
            (ValueDomain::Sync(left), ValueDomain::Sync(right)) if left == right => ValueDomain::Sync(left.clone()),
            _ => ValueDomain::Async,
        }
    }

    pub fn from_domain_kind(domain: DomainKind<DomainSignal>) -> Self {
        match domain {
            DomainKind::Async => ValueDomain::Async,
            DomainKind::Sync(sync) => ValueDomain::Sync(SyncDomain {
                clock: sync.clock,
                reset: sync.reset,
            }),
        }
    }

    pub fn from_port_domain(domain: PortDomain<DomainSignal>) -> Self {
        match domain {
            PortDomain::Clock => ValueDomain::Clock,
            PortDomain::Kind(kind) => ValueDomain::from_domain_kind(kind),
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

impl DomainSignal {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        match self {
            &DomainSignal::Port(port) => s.ports[port].id.string.clone(),
            &DomainSignal::Wire(wire) => s.wires[wire].id.string().to_owned(),
            &DomainSignal::Register(reg) => s.registers[reg].id.string().to_owned(),
            DomainSignal::BoolNot(x) => format!("!({})", x.to_diagnostic_string(s)),
        }
    }
}

impl PortDomain<DomainSignal> {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        match self {
            PortDomain::Clock => "clock".to_owned(),
            PortDomain::Kind(kind) => kind.to_diagnostic_string(s),
        }
    }
}

impl DomainKind<DomainSignal> {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        match self {
            DomainKind::Async => "async".to_owned(),
            DomainKind::Sync(sync) => sync.to_diagnostic_string(s),
        }
    }
}

impl SyncDomain<DomainSignal> {
    pub fn to_diagnostic_string(&self, s: &CompileState) -> String {
        format!(
            "sync({}, {})",
            self.clock.to_diagnostic_string(s),
            self.reset.to_diagnostic_string(s)
        )
    }
}
