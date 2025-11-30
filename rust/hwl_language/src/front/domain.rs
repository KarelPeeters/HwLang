use crate::front::compile::CompileItemContext;
use crate::front::signal::Signal;
use crate::front::signal::{Polarized, Port};
use crate::syntax::ast::{DomainKind, SyncDomain};
use crate::syntax::pos::Spanned;

#[derive(Debug)]
pub enum BlockDomain {
    CompileTime,
    Combinatorial,
    Clocked(Spanned<SyncDomain<DomainSignal>>),
}

pub type DomainSignal = Polarized<Signal>;

#[derive(Debug, Copy, Clone)]
pub enum PortDomain<P> {
    Clock,
    Kind(DomainKind<Polarized<P>>),
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
    pub fn diagnostic_string(self, s: &CompileItemContext) -> String {
        ValueDomain::from_port_domain(self).diagnostic_string(s)
    }
}

impl DomainKind<Polarized<Signal>> {
    pub fn diagnostic_string(&self, s: &CompileItemContext) -> String {
        match self {
            DomainKind::Const => "const".to_owned(),
            DomainKind::Async => "async".to_owned(),
            DomainKind::Sync(sync) => sync.diagnostic_string(s),
        }
    }
}

impl DomainKind<Polarized<Port>> {
    pub fn diagnostic_string(&self, s: &CompileItemContext) -> String {
        self.map_signal(|s| s.map_inner(Signal::Port)).diagnostic_string(s)
    }
}

impl SyncDomain<Polarized<Signal>> {
    pub fn diagnostic_string(&self, s: &CompileItemContext) -> String {
        let SyncDomain { clock, reset } = self;

        match reset {
            None => format!("sync({})", clock.diagnostic_string(s)),
            Some(reset) => format!("sync({}, {})", clock.diagnostic_string(s), reset.diagnostic_string(s)),
        }
    }
}

impl SyncDomain<Polarized<Port>> {
    pub fn diagnostic_string(&self, s: &CompileItemContext) -> String {
        self.map_signal(|p| p.map_inner(Signal::Port)).diagnostic_string(s)
    }
}

// TODO create separate HardwareDomain enum?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ValueDomain<V = Polarized<Signal>> {
    CompileTime,
    // TODO should clock domains refer to the "original" clock signal so multiple aliases are allowed?
    Clock,
    Const,
    // TODO allow separate sync/async per edge, necessary for "async" reset
    Async,
    Sync(SyncDomain<V>),
}

impl ValueDomain {
    pub fn join(self, other: Self) -> Self {
        // TODO expand signal equality check, eg. make it look through wire assignments
        match (self, other) {
            (ValueDomain::CompileTime, other) | (other, ValueDomain::CompileTime) => other,
            (ValueDomain::Const, other) | (other, ValueDomain::Const) => other,
            (ValueDomain::Sync(left), ValueDomain::Sync(right)) => {
                if left == right {
                    ValueDomain::Sync(left)
                } else {
                    ValueDomain::Async
                }
            }
            (ValueDomain::Async | ValueDomain::Clock, _) | (_, ValueDomain::Async | ValueDomain::Clock) => {
                ValueDomain::Async
            }
        }
    }

    pub fn fold(domains: impl IntoIterator<Item = Self>) -> Self {
        domains.into_iter().fold(ValueDomain::CompileTime, |acc, d| acc.join(d))
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
        *self
    }

    pub fn diagnostic_string(&self, s: &CompileItemContext) -> String {
        match self {
            ValueDomain::CompileTime => "compile-time".to_owned(),
            ValueDomain::Clock => "clock".to_owned(),
            ValueDomain::Const => "const".to_owned(),
            ValueDomain::Async => "async".to_owned(),
            ValueDomain::Sync(sync) => sync.diagnostic_string(s),
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
