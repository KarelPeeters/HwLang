pub mod arena;
pub mod io;
pub mod iter;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Never {}

#[macro_export]
macro_rules! throw {
    ($e:expr) => { return Err($e.into()) };
}