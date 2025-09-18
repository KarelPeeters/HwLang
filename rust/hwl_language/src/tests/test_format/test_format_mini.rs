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

#[test]
fn idempotent_block_comment_newlines() {
    let src = "const {\n/* \n*/\n\n/*\n*/\n}\n";
    let expected = "const {\n/* \n*/\n\n/*\n*/\n}\n";
    assert_formats_to(src, expected);
}

#[test]
fn idempotent_block_comment_before_semi() {
    let src = "const {\n    a = b + c /* test */;\n}\n";
    assert_formatted(src);
}

#[test]
fn block_comment_in_binary_should_not_force_wrap() {
    let src = "const a = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb /*j*/ / bbbb;\n";
    let expected = "const a = bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n    * bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb /*j*/\n    / bbbb;\n";
    assert_formats_to(src, expected);
}

#[test]
fn line_comment_in_binary_should_not_force_wrap() {
    let src = "const a = b + c // test\n;\n";
    let expected = "const a = b + c // test\n;\n";
    assert_formats_to(src, expected);
}

const LONG: &str = "long_long_long_long_long_long_long_long_long_long_long_long_long_long";

#[test]
fn combine_binary_ops_per_level() {
    let src = format!("const c = a * {LONG} / {LONG};");
    let expected = format!("const c = a\n    * {LONG}\n    / {LONG};\n");
    assert_formats_to(&src, &expected);
}

#[test]
fn format_binaries_mixed() {
    let src = format!("const c = {LONG}\n    * {LONG}\n    + {LONG};\n");
    assert_formatted(&src);
}

#[test]
fn format_binary_correct_level() {
    let src = format!("const c = [\n    {LONG}\n        + {LONG},\n];\n");
    assert_formatted(&src);
}

#[test]
fn format_binary_mix_cases() {
    let src = "const c = long_long_long_long_long_long_long_long_long_long_long_long_long_long
    * long_long_long_long_long_long_long_long_long_long_long_long_long_long
    + long_long_long_long_long_long_long_long_long_long_long_long_long_long;

const c = long_long_long_long_long_long_long_long_long_long_long_long_long_long
    + long_long_long_long_long_long_long_long_long_long_long_long_long_long
        * long_long_long_long_long_long_long_long_long_long_long_long_long_long;

const c = [
    long_long_long_long_long_long_long_long_long_long_long_long_long_long
        + long_long_long_long_long_long_long_long_long_long_long_long_long_long,
];

const c = [
    long_long_long_long_long_long_long_long_long_long_long_long_long_long
        * long_long_long_long_long_long_long_long_long_long_long_long_long_long
        + long_long_long_long_long_long_long_long_long_long_long_long_long_long,
];

const c = [
    long_long_long_long_long_long_long_long_long_long_long_long_long_long
        + long_long_long_long_long_long_long_long_long_long_long_long_long_long
            * long_long_long_long_long_long_long_long_long_long_long_long_long_long,
];
";
    assert_formatted(&src);
}

#[test]
fn pub_wire_reg() {
    let src_wire = "module foo ports() {\n    pub wire a = false;\n}\n";
    assert_formatted(src_wire);
    let src_reg = "module foo ports() {\n    pub reg a: bool = false;\n}\n";
    assert_formatted(src_reg);
}

#[test]
fn comment_comma_interaction_present_same() {
    let src = "const c = [\n    a + b, // test\n];\n";
    let expected = "const c = [\n    a + b, // test\n];\n";
    assert_formats_to(src, expected);
}

#[test]
fn comment_comma_interaction_missing_same() {
    let src = "const c = [\n    a + b // test\n];\n";
    let expected = "const c = [\n    a + b, // test\n];\n";
    assert_formats_to(src, expected);
}

#[test]
fn comment_comma_interaction_present_next() {
    let src = "const c = [\n    a + b,\n    // test\n];\n";
    let expected = "const c = [\n    a + b,\n    // test\n];\n";
    assert_formats_to(src, expected);
}

#[test]
fn comment_comma_interaction_missing_next() {
    let src = "const c = [\n    a + b\n    // test\n];\n";
    let expected = "const c = [\n    a + b,\n    // test\n];\n";
    assert_formats_to(src, expected);
}

#[test]
fn comment_comma_interaction_unstable() {
    let src = "const c = [a + b\n/*test*//*\n*/];\n";
    let expected = "const c = [\n    a + b,\n/*test*/ /*\n*/\n];\n";
    assert_formats_to(src, expected);
}

#[test]
fn block_comment_line_chan() {
    let src = "struct u(\n    if (false) {/*\n    *//*\n    *//**/})\n{}\n";
    let expected = "struct u(\n    if (false) { /*\n    */ /*\n    */ /**/\n    }\n) {}\n";
    assert_formats_to(src, expected);
}

#[test]
fn long_block_comment_before_assign() {
    let src = "const c /*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*/ = a;";
    let expected = "const c /*aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*/\n    = a;\n";
    assert_formats_to(src, expected);
}

#[test]
fn block_comment_after_assign() {
    let src = "const a =\n/***/\nb;\n";
    let expected = "const a =\n/***/\nb;\n";
    assert_formats_to(src, expected);
}

#[test]
fn block_comment_in_fill() {
    let src = "import a.[b/*\n*/];\n";
    let expected = "import a.[\n    b, /*\n*/\n];\n";
    assert_formats_to(src, expected);
}

#[test]
fn block_comment_in_string_same_line() {
    let src = "const a = \"{ /* long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment */ a}\";";
    let expected = "const a\n    = \"{ /* long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment */\n    a\n}\";\n";
    assert_formats_to(src, expected);
}

#[test]
fn block_comment_in_string_next_line() {
    let src = "const a = \"{\n    /* long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment */ a}\";";
    let expected = "const a = \"{\n    /* long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment */a\n}\";\n";
    assert_formats_to(src, expected);
}

#[test]
fn comment_after_dot() {
    let src = "const\nn=l.g/*\np*//l;";
    let expected = "const n = l.g /*\np*/\n    / l;\n";
    assert_formats_to(src, expected);
}

#[test]
fn long_comment_with_newline_causing_wrapping() {
    let long = "long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment_long_comment";
    let src = format!("const a = (b * c/* {long}\n*/,);");
    let expected = format!("const a = (\n    b\n        * c /* {long}\n*/,\n);\n");
    assert_formats_to(&src, &expected);
}
