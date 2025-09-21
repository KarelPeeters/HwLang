use crate::args::ArgsFormat;
use crate::util::print_diagnostics;
use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::syntax::format::FormatSettings;
use hwl_language::syntax::format::format;

use hwl_language::syntax::source::SourceDatabase;
use std::process::ExitCode;

// TODO integrations:
//   * commandline tool
//      * no args: project manifest, arg: file/dir, no formatting config yet so no need to worry about that interaction
//      * --check mode: don't edit files, just print diff
//   * webdemo: button on left to format current file, tab on right to show output and internal nodes
//   * LSP: implement code format action, maybe also format range?
//   * add to python as a standalone function

// TODO project:
//   * format all existing hwl code
//   * add CI check that everything is indeed formatted?

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

    let result_content = result.new_content;
    let file2 = source.add_file("dummy2.kh".to_owned(), result_content.clone());
    let result2 = format(&diags, &source, &settings, file2).unwrap();

    std::fs::write("output2.kh", result2.debug_str()).unwrap();
    println!("{:?}", result2.stats);

    if result_content != result2.new_content {
        eprintln!("Error: formatting is not stable!");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}
