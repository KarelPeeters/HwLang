use crate::tests::test_format::util::{assert_formats_to, assert_formatted};

#[test]
fn test_empty() {
    assert_formatted("");
}

#[test]
fn test_items_separated() {
    let src = "const a = 0;\nconst b = 1;\n\nconst c = 2;\n";
    assert_formatted(src);
}

#[test]
fn test_empty_comment() {
    let src = "// test\n";
    assert_formatted(src);
}

#[test]
fn test_item_comments() {
    let src = "// a\nconst a = 0;\n// b\nconst b = 1;\n\n// c\nconst c = 2;\n// d\n";
    assert_formatted(src);
}

#[test]
fn test_long_const() {
    let src = "const output_width\n    = very_long_identifier_very_long_identifier_very_long_identifier_very_long_identifier_very_long_identifier;\n";
    assert_formatted(src);
}

#[test]
fn test_comment_after_curly() {
    let src = "interface foo {}\n";
    assert_formatted(src);
    let src = "interface foo {} // test\n";
    assert_formatted(src);
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
    assert_formatted(src);
}

#[test]
fn preserve_mixed_comments() {
    let src = "// a\n//b\n/**/\n//c\n\n/*d*/\n\n//e\n";
    assert_formatted(src);
}

#[test]
fn correct_dot_order() {
    let src = "const c = a.2.b;\n";
    assert_formatted(src);
}

#[test]
fn pub_const() {
    let src = "pub const c = false;\n";
    assert_formatted(src);
}

#[test]
fn string_end_on_new_line_indented() {
    let src = "const {\n    \"test\n\";\n}\n";
    assert_formatted(src);
}

#[test]
fn mix_comma_comment() {
    let src = "enum Foo {\n    A,\n    B, // b\n    C // c\n    ,\n    /*d0*/D /*d1*/, // d2\n    E,\n    // end\n}\n";
    assert_formatted(src);
}

#[test]
fn mix_comma_comment_end() {
    let src = "enum Foo {\n    C // c\n    ,\n}\n";
    assert_formatted(src);
}

#[test]
fn mix_comma_comment_multiple() {
    let src = "enum Foo {\n    A // a\n    // b\n    ,\n}\n";
    assert_formatted(src);
}

#[test]
fn enum_content_wrap() {
    let long_identifier = "a".repeat(120);
    let src = format!("enum Foo {{\n    A(\n        {long_identifier}\n    ),\n}}\n");
    println!("{}", src);
    assert_formatted(&src);
}

#[test]
fn port_block() {
    let src = "module foo ports(\n    async {}\n) {}\n";
    assert_formatted(src);
}

#[test]
fn idempotent_comments_in_import() {
    let src = "import a.\n//\n\n//\nb;\n";
    assert_formatted(src);
}
