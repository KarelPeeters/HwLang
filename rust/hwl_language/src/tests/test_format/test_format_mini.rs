use crate::tests::test_format::util::{assert_format_stable, assert_formats_to};

#[test]
fn test_empty() {
    assert_format_stable("");
}

#[test]
fn test_items_separated() {
    let src = "const a = 0;\nconst b = 1;\n\nconst c = 2;\n";
    assert_format_stable(src);
}

#[test]
fn test_empty_comment() {
    let src = "// test\n";
    assert_format_stable(src);
}

#[test]
fn test_item_comments() {
    let src = "// a\nconst a = 0;\n// b\nconst b = 1;\n\n// c\nconst c = 2;\n// d\n";
    assert_format_stable(src);
}

#[test]
fn test_long_const() {
    let src = "const output_width\n    = very_long_identifier_very_long_identifier_very_long_identifier_very_long_identifier_very_long_identifier;\n";
    assert_format_stable(src);
}

#[test]
fn test_comment_after_curly() {
    let src = "interface foo {}\n";
    assert_format_stable(src);
    let src = "interface foo {} // test\n";
    assert_format_stable(src);
}

#[test]
fn preserve_at_most_one_blank() {
    let src = "\ninterface a{}interface b{}\ninterface c{}\n\ninterface d{}\n\n\ninterface e{}\n\n\n";
    let expected = "interface a {}\ninterface b {}\ninterface c {}\n\ninterface d {}\n\ninterface e {}\n";
    assert_formats_to(src, expected);
}

#[test]
fn stable_initial_comments() {
    let src = "//a\n//b\n\n//c\n";
    assert_format_stable(src);
}

#[test]
fn preserve_mixed_comments() {
    let src = "// a\n//b\n/**/\n//c\n\n/*d*/\n\n//e\n";
    assert_format_stable(src);
}

#[test]
fn correct_dot_order() {
    let src = "const c = a.2.b;\n";
    assert_format_stable(src);
}

#[test]
fn pub_const() {
    let src = "pub const c = false;\n";
    assert_format_stable(src);
}

#[test]
fn string_end_on_new_line_indented() {
    let src = "const {\n    \"test\n\";\n}\n";
    assert_format_stable(src);
}
