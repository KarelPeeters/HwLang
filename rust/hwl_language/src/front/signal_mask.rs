// TODO move this somewhere else
#[derive(Debug)]
pub enum SignalMask<S> {
    Scalar(S),
    Compound(Vec<SignalMask<S>>),
}

#[derive(Debug, Copy, Clone)]
pub struct MaskLengthMismatch;

impl<S> SignalMask<S> {
    pub fn merge_ref<T>(
        &mut self,
        other: &SignalMask<T>,
        mut f: impl FnMut(&mut S, &T),
    ) -> Result<(), MaskLengthMismatch>
    where
        S: Eq + Clone,
    {
        match (&mut *self, other) {
            (SignalMask::Scalar(s), SignalMask::Scalar(o)) => f(s, o),
            _ => todo!(),
        }

        // compact if all children are the scame scalar
        if let SignalMask::Compound(v) = self {
            let mut all_equal_scalar = true;
            let mut value = None;

            for x in v {
                let success = if let SignalMask::Scalar(x) = x {
                    match &value {
                        None => {
                            value = Some(x);
                            true
                        }
                        Some(y) => x == *y,
                    }
                } else {
                    false
                };
                if !success {
                    all_equal_scalar = false;
                    break;
                }
            }

            if all_equal_scalar && let Some(value) = value {
                *self = SignalMask::Scalar(value.clone());
            }
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SignalWrite {
    No,
    Certain,
    Maybe,
}

// // TODO this is going to be tricky (for reads), since those are just expressions
// //   and we want to track the full indexing that happens after?
// #[derive(Debug, Copy, Clone)]
// pub struct SignalAccess {
//     /// First time this signal was read when it had not yet been (certainly) written.
//     read_before_write: Option<Span>,
//     /// First time this signal was written.
//     write: Option<Span>,
//     // TODO include "maybe" as an option since it has different behavior
// }
