// Printing to stdout breaks the LSP. For temporary logging, print to stderr instead.
#![deny(clippy::print_stdout)]

pub mod syntax;
pub mod front;
pub mod back;
pub mod data;

pub mod util;
pub mod error;
pub mod constants;