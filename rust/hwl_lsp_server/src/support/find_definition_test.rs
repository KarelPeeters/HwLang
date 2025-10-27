use crate::support::PosNotOnIdentifier;
use crate::support::find_definition::find_definition;
use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::pos::{Pos, Span};
use hwl_language::syntax::source::SourceDatabase;
use itertools::Itertools;
use std::ops::Range;

#[track_caller]
fn test_resolve(src: &str, pos: usize, expected: Result<&[Range<usize>], PosNotOnIdentifier>) {
    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy".to_owned(), src.to_owned());
    let ast = parse_file_content(file, src).unwrap();

    let pos = Pos { file, byte: pos };
    let actual_spans = find_definition(&source, &ast, pos);

    let expected_typed =
        expected.map(|expected| expected.iter().map(|r| Span::new(file, r.start, r.end)).collect_vec());
    assert_eq!(expected_typed, actual_spans);
}

#[test]
fn resolve_file_const() {
    let src = "const a = 1; const b = a;";
    test_resolve(src, 23, Ok(&[6..7]));
}

#[test]
fn resolve_file_const_uncertain() {
    let src = "const a = 1; const a = 2; const b = a;";
    test_resolve(src, 36, Ok(&[6..7, 19..20]));
}

#[test]
fn resolve_param_certain() {
    let src = "const a = false; fn f(b: bool) { val _ = b; }";
    test_resolve(src, 41, Ok(&[22..23]));
}

#[test]
fn resolve_param_if_uncertain() {
    let src = "const b = false; fn f(a: bool, if(a) { b: bool }) { val _ = b; }";
    test_resolve(src, 60, Ok(&[39..40, 6..7]));
}

#[test]
fn resolve_param_if_certain_earlier() {
    let src = "const b = false; fn f(a: bool, if(a) { b: bool, c: g(b) }) {}";
    test_resolve(src, 53, Ok(&[39..40]));
}

// TODO get this working
// #[test]
// fn resolve_param_if_certain_branch() {
//     let src = "const b = false; fn f(a: bool, if(a) { b: bool } else { b: uint }) { val _ = b; }";
//     test_resolve(src, 77, Ok(&[39..40, 56..57]));
// }

#[test]
fn resolve_in_const_clock() {
    let src = "const a = 1; const { val _ = a; }";
    test_resolve(src, 29, Ok(&[6..7]));
}

#[test]
fn resolve_second_instance() {
    let src = "module parent ports(x: in async bool) { instance foo ports(x); instance foo ports(x); }";
    test_resolve(src, 59, Ok(&[20..21]));
    test_resolve(src, 82, Ok(&[20..21]));
}

#[test]
fn resolve_wire_process_after() {
    let src = "module foo ports() { comb { val _ = x; } wire x = false; }";
    test_resolve(src, 36, Ok(&[46..47]));
}

#[test]
fn resolve_pub_wire_top() {
    let src = "module foo ports() { pub wire x = false; wire y = x; }";
    test_resolve(src, 50, Ok(&[30..31]));
}

#[test]
fn resolve_pub_wire_if() {
    let src = "module foo ports() { if (true) { pub wire x = false; } wire y = x; }";
    test_resolve(src, 64, Ok(&[42..43]));
}

#[test]
fn resolve_pub_wire_for() {
    let src = "module foo ports() { for (i in 0..1) { pub wire x = false; } wire y = x; }";
    test_resolve(src, 70, Ok(&[48..49]));
}

#[test]
fn resolve_pub_wire_if_after() {
    let src = "module foo ports() { wire y = x; if (true) { pub wire x = false; } }";
    test_resolve(src, 30, Ok(&[54..55]));
}

#[test]
fn resolve_general_simple() {
    let src = "module foo ports() { wire id_from_str(\"x\") = false; comb { x; } }";
    test_resolve(src, 59, Ok(&[26..42]));
}

#[test]
fn resolve_simple_general() {
    let src = "module foo ports() { wire x = false; comb { id_from_str(\"x\"); } }";
    test_resolve(src, 44, Ok(&[26..27]));
}

#[test]
fn resolve_general_general() {
    let src = "module foo ports() { wire id_from_str(\"x\") = false; comb { id_from_str(\"x\"); } }";
    test_resolve(src, 59, Ok(&[26..42]));
}
