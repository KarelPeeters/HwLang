use clap::Parser;
use khdl_bin::args::{Args, ArgsCommand};
use khdl_bin::main_build::main_build;
use khdl_bin::main_fmt::main_fmt;
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
