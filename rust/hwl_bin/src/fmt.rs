use crate::args::ArgsFormat;
use crate::util::find_and_read_manifest;
use hwl_language::syntax::format::{format, FormatSettings};
use std::process::ExitCode;

pub fn main_fmt(args: ArgsFormat) -> ExitCode {
    let ArgsFormat { manifest, file } = args;

    // we always need a manifest to get formatting settings
    // let manifest = find_and_read_manifest(manifest);

    // let file = file.unwrap();

    let src = std::fs::read_to_string("/home/karel/Documents/hwlang/design/project/top.kh").unwrap();

    let settings = FormatSettings::default();
    let result = format(&src, &settings).unwrap();

    println!("{}", result);

    ExitCode::SUCCESS
}
