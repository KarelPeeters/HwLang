use clap::Parser;
use hwl_bin::args::{Args, ArgsCommand};
use hwl_bin::build::main_build;
use hwl_bin::fmt::main_fmt;
use std::process::ExitCode;

// TODO automatically disable this when miri is used
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> ExitCode {
    // TODO add a way to print all elaborated items and the instantiation tree
    let Args { command } = Args::parse();
    match command {
        ArgsCommand::Build(args) => main_build(args),
        ArgsCommand::Fmt(args) => main_fmt(args),
    }
}

fn foo(
    // the x value
    x: usize,
    y: bool, // the y value
             // no z value
) {
}
