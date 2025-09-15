use crate::front::diagnostic::{Diagnostics, diags_to_debug_string};
use crate::syntax::format::FormatSettings;
use crate::syntax::format::{check_format_output_matches, format};
use crate::syntax::source::SourceDatabase;

pub fn assert_formatted(src: &str) {
    assert_formats_to(src, src);
}

pub fn assert_formats_to(src: &str, expected: &str) {
    let diags = Diagnostics::new();
    let mut source = SourceDatabase::new();
    let settings = FormatSettings::default();

    // first format
    let file = source.add_file("dummy.kh".to_owned(), src.to_owned());
    let Ok(result) = format(&diags, &source, &settings, file) else {
        eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
        panic!("formatting failed");
    };

    let Ok(()) = check_format_output_matches(
        &diags,
        &source,
        file,
        &result.old_tokens,
        &result.old_ast,
        &result.new_content,
    ) else {
        eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
        panic!("formatting output does not match the original tokens/ast");
    };

    let result_new_content = result.new_content;
    assert_eq!(result_new_content, expected, "output differs from expected");

    // if relevant, check for idempotence
    if src != expected {
        let file2 = source.add_file("dummy2.kh".to_owned(), result_new_content.clone());
        let Ok(result2) = format(&diags, &source, &settings, file2) else {
            eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
            panic!("formatting failed the second time");
        };
        assert_eq!(result2.new_content, result_new_content, "output is not stable");
    }
}
