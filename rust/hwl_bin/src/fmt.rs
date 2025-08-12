use crate::args::ArgsFormat;
use crate::util::print_diagnostics;
use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::syntax::format::{FormatSettings, format};
use hwl_language::syntax::source::SourceDatabase;
use std::process::ExitCode;

pub fn main_fmt(args: ArgsFormat) -> ExitCode {
    let ArgsFormat { manifest, file } = args;

    // we always need a manifest to get formatting settings
    // let manifest = find_and_read_manifest(manifest);

    // let file = file.unwrap();

    let path = "/home/karel/Documents/hwlang/design/project/top.kh";
    let source_str = std::fs::read_to_string(path).unwrap();
    let mut source = SourceDatabase::new();
    let file = source.add_file(path.to_owned(), source_str);

    let settings = FormatSettings::default();

    let diags = Diagnostics::new();
    let result = format(&diags, &source, file, &settings);

    match result {
        Ok(result) => {
            // println!("Formatting result:");
            // println!("{result}");

            std::fs::write("output.kh", result).unwrap();

            ExitCode::SUCCESS
        }
        Err(_) => {
            print_diagnostics(&source, diags);
            ExitCode::FAILURE
        }
    }
}
