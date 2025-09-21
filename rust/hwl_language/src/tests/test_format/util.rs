use crate::front::diagnostic::{Diagnostics, diags_to_debug_string};
use crate::syntax::format::FormatSettings;
use crate::syntax::format::format;
use crate::syntax::source::SourceDatabase;

pub fn assert_formatted(src: &str) {
    assert_formats_to(src, src);
}

pub fn assert_formats_to(src: &str, expected: &str) {
    let settings = FormatSettings::default();

    // first format
    let result = assert_format_valid("dummy.kh", src, &settings);
    assert_eq!(result, expected, "output differs from expected");

    // second format, if necessary, to check for stability
    if src != result {
        let result2 = assert_format_valid("dummy2.kh", &result, &settings);
        assert_eq!(result2, result, "formatting output is not stable");
    }
}

pub fn assert_format_valid(name: &str, src: &str, settings: &FormatSettings) -> String {
    let diags = Diagnostics::new();
    let mut source = SourceDatabase::new();

    let file = source.add_file(name.to_owned(), src.to_owned());

    let Ok(result) = format(&diags, &source, &settings, file) else {
        eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
        panic!("formatting failed");
    };

    result.new_content
}
