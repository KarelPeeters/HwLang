use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::parsed::AstRefItem;
use crate::new::value::ScopedValue;
use crate::syntax::ast::{DomainKind, SyncDomain};

// TODO move this to a better place
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(AstRefItem),
    Direct(ScopedValue),
}

// TODO expand to all possible values again
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DomainSignal {
    Error(ErrorGuaranteed),
    Compile(bool),
    Port(/*TODO*/),
    Wire(/*TODO*/),
    Register(/*TODO*/),
    Invert(Box<DomainSignal>),
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
