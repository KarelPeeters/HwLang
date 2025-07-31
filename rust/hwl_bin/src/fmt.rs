use crate::args::ArgsFormat;
use std::process::ExitCode;

pub fn main_fmt(_: ArgsFormat) -> ExitCode {
    eprintln!("Formatting is not yet implemented.");
    ExitCode::FAILURE
}
