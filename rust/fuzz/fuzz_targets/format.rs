#![no_main]

use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::syntax::format::FormatSettings;
use hwl_language::syntax::format_new::format;
use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::source::SourceDatabase;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| target(data));

fn target(data: &str) {
    if !is_simple_str(data) {
        return;
    }

    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy.kh".to_owned(), data.to_owned());

    let ast = parse_file_content(file, &source[file].content);

    if ast.is_ok() {
        // check that formatting works
        let diags = Diagnostics::new();
        let result =
            format(&diags, &mut source, &FormatSettings::default(), file).expect("internal error during formatting");

        // check that formatting is idempotent
        let file2 = source.add_file("dummy2.kh".to_owned(), result.clone());
        let result2 = format(&diags, &mut source, &FormatSettings::default(), file2)
            .expect("internal error during second formatting");
        assert_eq!(result, result2, "formatting is not idempotent");
    }
}

fn is_simple_str(s: &str) -> bool {
    s.chars().all(|c| {
        match c {
            // ascii whitespace
            '\t' | '\n' | '\r' | ' ' => true,
            // ascii printable
            ' '..='~' => true,
            _ => false,
        }
    })
}
