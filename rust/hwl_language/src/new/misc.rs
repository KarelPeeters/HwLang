use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::parsed::AstRefItem;
use crate::new::types::Type;
use crate::new::value::ScopedValue;
use crate::syntax::ast::{DomainKind, SyncDomain};

// TODO move everything in this file to a better place
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValue<V> {
    Type(Type),
    Value(V),
    Error(ErrorGuaranteed),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValueNoError<V> {
    Type(Type),
    Value(V),
}

impl<V> From<ErrorGuaranteed> for TypeOrValue<V> {
    fn from(e: ErrorGuaranteed) -> Self {
        TypeOrValue::Error(e)
    }
}

#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(AstRefItem),
    Direct(TypeOrValue<ScopedValue>),
}

impl<V> TypeOrValue<V> {
    pub fn map_value<W>(self, mut f: impl FnMut(V) -> W) -> TypeOrValue<W> {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t),
            TypeOrValue::Value(v) => TypeOrValue::Value(f(v)),
            TypeOrValue::Error(e) => TypeOrValue::Error(e),
        }
    }
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
