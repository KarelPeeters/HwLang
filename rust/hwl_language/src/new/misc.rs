use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::parsed::AstRefItem;
use crate::new::compile::{Port, Register, Wire};
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
    Invert(Box<DomainSignal>),
}

#[derive(Debug, Clone)]
pub enum PortDomain<V> {
    Clock,
    Kind(DomainKind<V>),
}

#[derive(Debug, Clone)]
pub enum ValueDomain<V = DomainSignal> {
    // TODO extract error case out?
    Error(ErrorGuaranteed),
    CompileTime,
    Clock,
    // TODO allow separate sync/async per edge, necessary for "async" reset
    Async,
    Sync(SyncDomain<V>),
}

impl ValueDomain {
    pub fn from_domain_kind(domain: DomainKind<DomainSignal>) -> Self {
        match domain {
            DomainKind::Async => ValueDomain::Async,
            DomainKind::Sync(sync) => ValueDomain::Sync(SyncDomain {
                clock: sync.clock,
                reset: sync.reset,
            }),
        }
    }
}
