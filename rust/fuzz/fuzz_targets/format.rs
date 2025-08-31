#![no_main]

use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::syntax::format::FormatSettings;
use hwl_language::syntax::format_new::format;
use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::source::SourceDatabase;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| target(data));

fn target(data: &str) {
    let mut source = SourceDatabase::new();
    let file = source.add_file(format!("dummy.kh"), data.to_owned());

    let ast = parse_file_content(file, &source[file].content);

    if ast.is_ok() {
        let diags = Diagnostics::new();
        let _ = format(&diags, &mut source, &FormatSettings::default(), file);
        // TODO check that tokens match
    }
}
