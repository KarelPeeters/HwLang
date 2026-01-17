#![no_main]

use hwl_language::front::diagnostic::{Diagnostics, diags_to_string};
use hwl_language::syntax::format::FormatSettings;
use hwl_language::syntax::format::format_file;
use hwl_language::syntax::parse_file_content_without_recovery;
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::token::{TokenCategory, tokenize};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| target(data));

fn target(data: &str) {
    let env_fuzz_debug = std::env::var_os("FUZZ_DEBUG").is_some();
    let env_ban_comments = std::env::var_os("BAN_COMMENTS").is_some();

    if env_fuzz_debug {
        std::fs::write("ignored/fuzz.kh", data).unwrap();
    }

    // discard invalid or non-parsable inputs
    if !is_simple_str(data) {
        return;
    }
    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy.kh".to_owned(), data.to_owned());
    if parse_file_content_without_recovery(file, &source[file].content).is_err() {
        return;
    };

    // discard inputs with any comments
    if env_ban_comments
        && tokenize(file, &source[file].content, false)
            .unwrap()
            .iter()
            .any(|t| t.ty.category() == TokenCategory::Comment)
    {
        return;
    }

    let diags = Diagnostics::new();
    let settings = FormatSettings::default();

    // check that formatting works
    let Ok(result) = format_file(&diags, &source, &settings, file) else {
        eprintln!("{}", diags_to_string(&source, diags.finish(), true));
        panic!("internal error during formatting");
    };
    let new_content = result.new_content;

    // check that formatting is stable
    let file2 = source.add_file("dummy2.kh".to_owned(), new_content.clone());
    let Ok(result2) = format_file(&diags, &source, &settings, file2) else {
        eprintln!("{}", diags_to_string(&source, diags.finish(), true));
        panic!("internal error during second format");
    };
    assert_eq!(new_content, result2.new_content, "formatting is not idempotent");
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
