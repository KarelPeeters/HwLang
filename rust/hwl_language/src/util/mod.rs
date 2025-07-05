use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;

pub mod arena;
pub mod big_int;
pub mod data;
pub mod int;
pub mod iter;
pub mod store;
pub mod sync;

pub const NON_ZERO_USIZE_ONE: NonZeroUsize = NonZeroUsize::new(1).unwrap();
pub const NON_ZERO_USIZE_TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();

// TODO maybe "!" is stable enough by now
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Never {}

impl Never {
    pub fn unreachable(self) -> ! {
        match self {}
    }
}

pub trait ResultNeverExt<T> {
    fn remove_never(self) -> T;
}

impl<T> ResultNeverExt<T> for Result<T, Never> {
    fn remove_never(self) -> T {
        match self {
            Ok(v) => v,
            Err(e) => e.unreachable(),
        }
    }
}

#[macro_export]
macro_rules! throw {
    ($e:expr) => {
        return Err($e.into())
    };
}

#[macro_export]
macro_rules! try_opt_result {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return Ok(None),
        }
    };
}

#[macro_export]
macro_rules! try_inner {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Ok(Err(e.into())),
        }
    };
}

pub trait ResultExt<T, E> {
    fn as_ref_ok(&self) -> Result<&T, E>;
    fn as_ref_mut_ok(&mut self) -> Result<&mut T, E>;
}

pub trait ResultDoubleExt<T, E> {
    fn flatten_err(self) -> Result<T, E>;
}

pub trait ResultSplitExt {
    type A;
    type B;
    type E;
    fn result_split(self) -> (Result<Self::A, Self::E>, Result<Self::B, Self::E>);
}

impl<T, E: Copy> ResultExt<T, E> for Result<T, E> {
    fn as_ref_ok(&self) -> Result<&T, E> {
        match self {
            Ok(v) => Ok(v),
            &Err(e) => Err(e),
        }
    }

    fn as_ref_mut_ok(&mut self) -> Result<&mut T, E> {
        match self {
            Ok(v) => Ok(v),
            &mut Err(e) => Err(e),
        }
    }
}

impl<T, E> ResultDoubleExt<T, E> for Result<Result<T, E>, E> {
    fn flatten_err(self) -> Result<T, E> {
        self?
    }
}

pub fn option_pair<A, B>(left: Option<A>, right: Option<B>) -> Option<(A, B)> {
    let left = left?;
    let right = right?;
    Some((left, right))
}

pub fn result_pair<A, B, E>(left: Result<A, E>, right: Result<B, E>) -> Result<(A, B), E> {
    let left = left?;
    let right = right?;
    Ok((left, right))
}

pub fn result_triple<A, B, C, E>(a: Result<A, E>, b: Result<B, E>, c: Result<C, E>) -> Result<(A, B, C), E> {
    let a = a?;
    let b = b?;
    let c = c?;
    Ok((a, b, c))
}

pub fn result_pair_split<A, B, E: Copy>(r: Result<(A, B), E>) -> (Result<A, E>, Result<B, E>) {
    match r {
        Ok((a, b)) => (Ok(a), Ok(b)),
        Err(e) => (Err(e), Err(e)),
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Indent {
    depth: usize,
}

impl Indent {
    pub const I: &'static str = "    ";

    pub fn new(depth: usize) -> Indent {
        Indent { depth }
    }

    pub fn nest(self) -> Indent {
        Indent { depth: self.depth + 1 }
    }

    pub fn nest_n(self, n: usize) -> Indent {
        Indent { depth: self.depth + n }
    }
}

impl Display for Indent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.depth {
            f.write_str(Self::I)?;
        }
        Ok(())
    }
}

pub fn separator_non_trailing(s: &str, i: usize, len: usize) -> &str {
    if i == len - 1 {
        ""
    } else {
        s
    }
}
