use crate::tests::test_format::util::{assert_formats_to, assert_formatted};

#[test]
fn skip_item() {
    let src = "// khdl-fmt skip\nconst table = [\n    0, 1, 2, 3,\n    4, 5, 6, 7,\n];\nconst b   = 1;\n";
    let expected = "// khdl-fmt skip\nconst table = [\n    0, 1, 2, 3,\n    4, 5, 6, 7,\n];\nconst b = 1;\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_item_block_comment() {
    let src = "/* khdl-fmt skip */const x = [1,2,\n  3];\nconst b   = 1;\n";
    let expected = "/* khdl-fmt skip */const x = [1,2,\n  3];\nconst b = 1;\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_statement() {
    let src = "fn f() {\n    //   khdl-fmt   skip  \n    val x  =  [1,2,\n        3];\n    val y  =  2;\n}\n";
    let expected = "fn f() {\n    //   khdl-fmt   skip  \n    val x  =  [1,2,\n        3];\n    val y = 2;\n}\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_expression() {
    let src = "const x = foo( /* khdl-fmt skip */a+b, c+d);\n";
    let expected = "const x = foo( /* khdl-fmt skip */a+b, c + d);\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_list_items() {
    let src = "struct S {\n    // khdl-fmt skip\n    a:   int,\n    b:   int,\n}\nenum E {\n    /* khdl-fmt skip */A(  int ),\n    B(  int ),\n}\n";
    let expected = "struct S {\n    // khdl-fmt skip\n    a:   int,\n    b: int,\n}\nenum E {\n    /* khdl-fmt skip */A(  int ),\n    B(int),\n}\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_module_statement() {
    let src = "module foo ports() {\n    // khdl-fmt skip\n    wire  a=1;\n    wire  b=2;\n}\n";
    let expected = "module foo ports() {\n    // khdl-fmt skip\n    wire  a=1;\n    wire b = 2;\n}\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_trailing_comment() {
    let src = "const a=1; // khdl-fmt skip\nconst  b=2;\nconst  c=3;\n";
    let expected = "const a = 1; // khdl-fmt skip\nconst  b=2;\nconst c = 3;\n";
    assert_formats_to(src, expected);

    let src = "const x = foo(a, /* khdl-fmt skip */ b  +  c, d  +  e);\n";
    let expected = "const x = foo(a, /* khdl-fmt skip */ b  +  c, d + e);\n";
    assert_formats_to(src, expected);
}

#[test]
fn skip_without_target() {
    let src = "// khdl-fmt skip\n";
    assert_formatted(src);
    let src = "const a=1;\n// khdl-fmt skip\n";
    assert_formats_to(src, "const a = 1;\n// khdl-fmt skip\n");
}

#[test]
fn off_on_items() {
    let src = "const a=1;\n// khdl-fmt off\nconst  b  =  2;\nconst   c = [1,\n  2];\n// khdl-fmt on\nconst d=3;\n";
    let expected =
        "const a = 1;\n// khdl-fmt off\nconst  b  =  2;\nconst   c = [1,\n  2];\n// khdl-fmt on\nconst d = 3;\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_until_end_of_file() {
    let src = "const a=1;\n// khdl-fmt off\nconst  b  =  2;\nconst   c = [1,\n  2];\n";
    let expected = "const a = 1;\n// khdl-fmt off\nconst  b  =  2;\nconst   c = [1,\n  2];\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_on_array_elements() {
    let src = "const x = [\n    // khdl-fmt off\n    0, 1, 2, 3,\n    4, 5, 6, 7,\n    // khdl-fmt on\n];\n";
    assert_formatted(src);
}

#[test]
fn off_on_array_elements_no_trailing_comma() {
    let src = "const x = [\n    // khdl-fmt off\n    0, 1, 2, 3,\n    4, 5, 6, 7\n    // khdl-fmt on\n];\n";
    assert_formatted(src);
}

#[test]
fn off_on_module_statements() {
    let src = "module foo ports() {\n    wire a=1;\n    // khdl-fmt off\n    wire  b=2;\n    wire   c=3;\n    // khdl-fmt on\n    wire d=4;\n}\n";
    let expected = "module foo ports() {\n    wire a = 1;\n    // khdl-fmt off\n    wire  b=2;\n    wire   c=3;\n    // khdl-fmt on\n    wire d = 4;\n}\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_on_inside_expression() {
    let src = "const x = foo(a, /* khdl-fmt off */ b  +  c, d /* khdl-fmt on */, e  +  f);\n";
    let expected = "const x = foo(a, /* khdl-fmt off */ b  +  c, d /* khdl-fmt on */, e + f);\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_trailing_comment() {
    let src = "const a=1; // khdl-fmt off\nconst  b=2;\n// khdl-fmt on\nconst  c=3;\n";
    let expected = "const a = 1; // khdl-fmt off\nconst  b=2;\n// khdl-fmt on\nconst c = 3;\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_blank_lines() {
    let src = "\n\n// khdl-fmt off\n\n\nconst  b=2;\n\n\n// khdl-fmt on\n\n\nconst  c=3;\n\n";
    let expected = "// khdl-fmt off\n\n\nconst  b=2;\n\n\n// khdl-fmt on\n\nconst c = 3;\n";
    assert_formats_to(src, expected);
}

#[test]
fn directives_nested_and_repeated() {
    let src = "// khdl-fmt off\nconst a=1;\n// khdl-fmt skip\nconst  b=2;\n// khdl-fmt on\n// khdl-fmt skip\nconst  c=[1,\n2];\n// khdl-fmt off\nconst  d=2;\n// khdl-fmt on\nconst   e=2;\n";
    let expected = "// khdl-fmt off\nconst a=1;\n// khdl-fmt skip\nconst  b=2;\n// khdl-fmt on\n// khdl-fmt skip\nconst  c=[1,\n2];\n// khdl-fmt off\nconst  d=2;\n// khdl-fmt on\nconst e = 2;\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_on_nested() {
    let src = "// khdl-fmt off\nconst  a=1;\n// khdl-fmt off\nconst  b=2;\n// khdl-fmt on\nconst  c=3;\n// khdl-fmt on\nconst  d=4;\n";
    let expected = "// khdl-fmt off\nconst  a=1;\n// khdl-fmt off\nconst  b=2;\n// khdl-fmt on\nconst  c=3;\n// khdl-fmt on\nconst d = 4;\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_on_nested_inside_expression() {
    let src = "const x = f(a, /* khdl-fmt off */ b  +  c, /* khdl-fmt off */ d /* khdl-fmt on */, e  +  f /* khdl-fmt on */, g  +  h);\n";
    let expected = "const x = f(a, /* khdl-fmt off */ b  +  c, /* khdl-fmt off */ d /* khdl-fmt on */, e  +  f /* khdl-fmt on */, g + h);\n";
    assert_formats_to(src, expected);
}

#[test]
fn off_directly_before_on() {
    let src =
        "// khdl-fmt off\nconst  a=1;\n// khdl-fmt off\n// khdl-fmt on\nconst  b=2;\n// khdl-fmt on\nconst  c=3;\n";
    let expected =
        "// khdl-fmt off\nconst  a=1;\n// khdl-fmt off\n// khdl-fmt on\nconst  b=2;\n// khdl-fmt on\nconst c = 3;\n";
    assert_formats_to(src, expected);
}

#[test]
fn on_without_off() {
    let src = "const  a=1;\n// khdl-fmt on\nconst  b=2;\n";
    let expected = "const a = 1;\n// khdl-fmt on\nconst b = 2;\n";
    assert_formats_to(src, expected);
}

#[test]
fn non_directive_comments() {
    let src = "// khdl-fmt\nconst a=1;\n// khdl-fmt skipx\nconst b=2;\n// khdl-fmt on\nconst c=3;\n";
    let expected = "// khdl-fmt\nconst a = 1;\n// khdl-fmt skipx\nconst b = 2;\n// khdl-fmt on\nconst c = 3;\n";
    assert_formats_to(src, expected);
}
