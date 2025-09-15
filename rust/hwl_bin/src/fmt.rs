use crate::args::ArgsFormat;
use crate::util::print_diagnostics;
use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::syntax::format::FormatSettings;
use hwl_language::syntax::format_new::{check_format_output_matches, format};

use hwl_language::syntax::source::SourceDatabase;
use std::process::ExitCode;

pub fn main_fmt(args: ArgsFormat) -> ExitCode {
    let _ = args;
    // let ArgsFormat { manifest, file } = args;

    // we always need a manifest to get formatting settings
    // let manifest = find_and_read_manifest(manifest);

    // let file = file.unwrap();

    let path = "/home/karel/Documents/hwlang/design/project/top.kh";
    let source_str = std::fs::read_to_string(path).unwrap();
    let mut source = SourceDatabase::new();
    let file = source.add_file(path.to_owned(), source_str);

    let settings = FormatSettings::default();

    let diags = Diagnostics::new();
    let Ok(result) = format(&diags, &source, &settings, file) else {
        print_diagnostics(&source, diags);
        return ExitCode::FAILURE;
    };

    std::fs::write("output.kh", result.debug_str()).unwrap();

    let Ok(()) = check_format_output_matches(
        &diags,
        &source,
        file,
        &result.old_tokens,
        &result.old_ast,
        &result.new_content,
    ) else {
        print_diagnostics(&source, diags);
        return ExitCode::FAILURE;
    };

    let file2 = source.add_file("dummy2.kh".to_owned(), result.new_content);
    let result2 = format(&diags, &source, &settings, file2).unwrap();

    std::fs::write("output2.kh", result2.debug_str()).unwrap();
    println!("{:?}", result2.stats);

    ExitCode::SUCCESS
}
