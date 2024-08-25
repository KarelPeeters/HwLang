pub mod arena;
pub mod io;
pub mod iter;
pub mod data;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Never {}

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