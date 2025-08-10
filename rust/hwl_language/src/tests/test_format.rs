use crate::front::diagnostic::Diagnostics;
use crate::syntax::format::{FormatSettings, format};
use crate::syntax::source::SourceDatabase;
use itertools::Itertools;

#[test]
fn test_empty() {
    assert_stable("");
}

#[test]
fn test_items_separated() {
    let src = "const a = 0;\nconst b = 1;\n\nconst c = 2;\n";
    assert_stable(src);
}

#[test]
fn test_everything() {
    let src = std::fs::read_to_string("src/tests/test_format.kh").unwrap();
    assert_stable(&src);
}

#[test]
fn test_empty_comment() {
    let src = "// test\n";
    assert_stable(src);
}

#[test]
fn test_item_comments() {
    let src = "// a\nconst a = 0;\n// b\nconst b = 1;\n\n// c\nconst c = 2;\n// d\n";
    assert_stable(src);
}

#[test]
fn test_long_const() {
    let src = "const output_width\n    = very_long_identifier_very_long_identifier_very_long_identifier_very_long_identifier_very_long_identifier;\n";
    assert_stable(src);
}

fn assert_stable(src: &str) {
    let diags = Diagnostics::new();
    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy.kh".to_owned(), src.to_owned());
    let settings = FormatSettings::default();
    let after = format(&diags, &source, file, &settings).unwrap();
    assert_eq!(after, src);
}
