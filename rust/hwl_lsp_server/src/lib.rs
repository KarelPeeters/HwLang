// Printing to stdout breaks the LSP. For temporary logging, print to stderr instead.
#![deny(clippy::print_stdout)]

pub mod handlers;
pub mod server;
pub mod util;
