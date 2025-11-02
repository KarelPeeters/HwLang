use crate::syntax::pos::Span;

// TODO move this somewhere else
pub enum SignalMask<S> {
    Scalar(S),
    Compound(Vec<S>),
}

// TODO this is going to be tricky (for reads), since those are just expressions
//   and we want to track the full indexing that happens after?
#[derive(Debug, Copy, Clone)]
pub struct SignalAccess {
    /// First time this signal was read when it had not yet been (certainly) written.
    read_before_write: Option<Span>,
    /// First time this signal was written.
    write: Option<Span>,
    // TODO include "maybe" as an option since it has different behavior
}
