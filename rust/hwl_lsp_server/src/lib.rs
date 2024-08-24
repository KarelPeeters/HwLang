// Printing to stdout breaks the LSP. For temporary logging, print to stderr instead.
#![deny(clippy::print_stdout)]

pub mod server;
pub mod engine;
