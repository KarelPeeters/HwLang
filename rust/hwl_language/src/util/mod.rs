pub mod arena;
pub mod io;
pub mod iter;
pub mod data;
pub mod int;

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
    ($e:expr) => { return Err($e.into()) };
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
}

pub trait ResultDoubleExt<T, E> {
    fn flatten_err(self) -> Result<T, E>;
}

impl<T, E: Copy> ResultExt<T, E> for Result<T, E> {
    fn as_ref_ok(&self) -> Result<&T, E> {
        match *self {
            Ok(ref v) => Ok(v),
            Err(e) => Err(e),
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

/// Variant of write! that only works for strings, and doesn't return a spurious error.
#[macro_export]
macro_rules! swrite {
    ($dst:expr, $($arg:tt)*) => {{
        use std::fmt::Write;
        let dst: &mut String = $dst;
        write!(dst, $($arg)*).unwrap();
    }};
}

/// Variant of writeln! that only works for strings, and doesn't return a spurious error.
#[macro_export]
macro_rules! swriteln {
    ($dst:expr $(,)?) => {{
        use std::fmt::Write;
        let dst: &mut String = $dst;
        writeln!(dst).unwrap();
    }};
    ($dst:expr, $($arg:tt)*) => {{
        use std::fmt::Write;
        let dst: &mut String = $dst;
        writeln!(dst, $($arg)*).unwrap();
    }};
}
