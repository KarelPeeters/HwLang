// Printing to stdout breaks the LSP. For temporary logging, print to stderr instead.
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]

pub mod handlers;
pub mod server;
pub mod support;
pub mod util;
