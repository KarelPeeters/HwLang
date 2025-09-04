#![no_main]

use hwl_language::front::diagnostic::{Diagnostics, diags_to_debug_string};
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

    if std::env::var("FUZZ_SAVE").is_ok() {
        std::fs::write("fuzz.kh", data).unwrap();
    }

    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy.kh".to_owned(), data.to_owned());

    let Ok(ast) = parse_file_content(file, &source[file].content) else {
        return;
    };

    // check that formatting works
    let diags = Diagnostics::new();
    let Ok(result) = format(&diags, &mut source, &FormatSettings::default(), file) else {
        eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
        panic!("internal error during formatting");
    };

    // check that formatting is idempotent
    let file2 = source.add_file("dummy2.kh".to_owned(), result.clone());
    let Ok(result2) = format(&diags, &mut source, &FormatSettings::default(), file2) else {
        eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
        panic!("internal error during second format");
    };
    assert_eq!(result, result2, "formatting is not idempotent");
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
