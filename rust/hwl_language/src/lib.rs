// Printing to stdout breaks the LSP. For temporary logging, print to stderr instead.
#![deny(clippy::print_stdout)]

pub mod constants;

pub mod front;
pub mod syntax;
pub mod util;
