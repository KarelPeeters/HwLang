use std::fmt::{Display, Formatter};

pub mod arena;
pub mod data;
pub mod int;
pub mod io;
pub mod iter;

// TODO maybe "!" is stable enough by now
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Never {}

impl Never {
    pub fn unreachable(self) -> ! {
        match self {}
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
        match self {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(e),
        }
    }
}

pub fn option_pair<A, B>(left: Option<A>, right: Option<B>) -> Option<(A, B)> {
    match (left, right) {
        (Some(left), Some(right)) => Some((left, right)),
        (None, _) | (_, None) => None,
    }
}

pub fn result_pair<A, B, E>(left: Result<A, E>, right: Result<B, E>) -> Result<(A, B), E> {
    match (left, right) {
        (Ok(left), Ok(right)) => Ok((left, right)),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}

pub fn result_triple<A, B, C, E>(a: Result<A, E>, b: Result<B, E>, c: Result<C, E>) -> Result<(A, B, C), E> {
    match (a, b, c) {
        (Ok(a), Ok(b), Ok(c)) => Ok((a, b, c)),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => Err(e),
    }
}

pub fn result_pair_split<A, B, E: Copy>(r: Result<(A, B), E>) -> (Result<A, E>, Result<B, E>) {
    match r {
        Ok((a, b)) => (Ok(a), Ok(b)),
        Err(e) => (Err(e), Err(e)),
    }
}

pub trait StringMut {
    fn as_mut_string(&mut self) -> &mut String;
}

impl StringMut for String {
    fn as_mut_string(&mut self) -> &mut String {
        self
    }
}

impl StringMut for &mut String {
    fn as_mut_string(&mut self) -> &mut String {
        self
    }
}

/// Variant of write! that only works for strings, and doesn't return a spurious error.
#[macro_export]
macro_rules! swrite {
    ($dst:expr, $($arg:tt)*) => {{
        use std::fmt::Write;
        use $crate::util::StringMut;
        let dst = $dst.as_mut_string();
        write!(dst, $($arg)*).unwrap();
    }};
}

/// Variant of writeln! that only works for strings, and doesn't return a spurious error.
#[macro_export]
macro_rules! swriteln {
    ($dst:expr $(,)?) => {{
        use std::fmt::Write;
        use $crate::util::StringMut;
        let dst = $dst.as_mut_string();
        writeln!(dst).unwrap();
    }};
    ($dst:expr, $($arg:tt)*) => {{
        use std::fmt::Write;
        use $crate::util::StringMut;
        let dst = $dst.as_mut_string();
        writeln!(dst, $($arg)*).unwrap();
    }};
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
