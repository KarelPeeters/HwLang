use crate::syntax::parse_file_content;
use crate::syntax::pos::{Pos, Span};
use crate::syntax::resolve::{FindDefinition, find_definition};
use crate::syntax::source::SourceDatabase;
use itertools::Itertools;
use std::ops::Range;

#[track_caller]
fn test_resolve(src: &str, pos: usize, expected: FindDefinition<&[Range<usize>]>) {
    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy".to_owned(), src.to_owned());
    let ast = parse_file_content(file, src).unwrap();

    let pos = Pos { file, byte: pos };
    let actual_spans = find_definition(&source, &ast, pos);

    let expected = match expected {
        FindDefinition::Found(expected_spans) => {
            let spans = expected_spans
                .iter()
                .map(|r| Span::new(file, r.start, r.end))
                .collect_vec();
            FindDefinition::Found(spans)
        }
        FindDefinition::PosNotOnIdentifier => FindDefinition::PosNotOnIdentifier,
        FindDefinition::DefinitionNotFound => FindDefinition::DefinitionNotFound,
    };

    assert_eq!(expected, actual_spans);
}

#[test]
fn resolve_file_const() {
    let src = "const a = 1; const b = a;";
    test_resolve(src, 23, FindDefinition::Found(&[6..7]));
}

#[test]
fn resolve_file_const_uncertain() {
    let src = "const a = 1; const a = 2; const b = a;";
    test_resolve(src, 36, FindDefinition::Found(&[6..7, 19..20]));
}

#[test]
fn resolve_param_certain() {
    let src = "const a = false; fn f(b: bool) { val _ = b; }";
    test_resolve(src, 41, FindDefinition::Found(&[22..23]));
}

#[test]
fn resolve_param_if_uncertain() {
    let src = "const b = false; fn f(a: bool, if(a) { b: bool }) { val _ = b; }";
    test_resolve(src, 60, FindDefinition::Found(&[39..40, 6..7]));
}

#[test]
fn resolve_param_if_certain_earlier() {
    let src = "const b = false; fn f(a: bool, if(a) { b: bool, c: g(b) }) {}";
    test_resolve(src, 53, FindDefinition::Found(&[39..40]));
}

// TODO get this working
// #[test]
// fn resolve_param_if_certain_branch() {
//     let src = "const b = false; fn f(a: bool, if(a) { b: bool } else { b: uint }) { val _ = b; }";
//     test_resolve(src, 77, FindDefinition::Found(&[39..40, 56..57]));
// }

#[test]
fn resolve_in_const_clock() {
    let src = "const a = 1; const { val _ = a; }";
    test_resolve(src, 29, FindDefinition::Found(&[6..7]));
}

#[test]
fn resolve_second_instance() {
    let src = "module parent ports(x: in async bool) { instance foo ports(x); instance foo ports(x); }";
    println!("first");
    test_resolve(src, 59, FindDefinition::Found(&[20..21]));
    println!("second");
    test_resolve(src, 82, FindDefinition::Found(&[20..21]));
}

#[test]
fn resolve_wire_process_after() {
    let src = "module foo ports() { comb { val _ = x; } wire x = false; }";
    test_resolve(src, 36, FindDefinition::Found(&[46..47]));
}

#[test]
fn resolve_pub_wire_top() {
    let src = "module foo ports() { pub wire x = false; wire y = x; }";
    test_resolve(src, 50, FindDefinition::Found(&[30..31]));
}

#[test]
fn resolve_pub_wire_if() {
    let src = "module foo ports() { if (true) { pub wire x = false; } wire y = x; }";
    test_resolve(src, 64, FindDefinition::Found(&[42..43]));
}

#[test]
fn resolve_pub_wire_for() {
    let src = "module foo ports() { for (i in 0..1) { pub wire x = false; } wire y = x; }";
    test_resolve(src, 70, FindDefinition::Found(&[48..49]));
}

#[test]
fn resolve_pub_wire_if_after() {
    let src = "module foo ports() { wire y = x; if (true) { pub wire x = false; } }";
    test_resolve(src, 30, FindDefinition::Found(&[54..55]));
}

#[test]
fn resolve_general_simple() {
    let src = "module foo ports() { wire id_from_str(\"x\") = false; comb { x; } }";
    test_resolve(src, 59, FindDefinition::Found(&[26..42]));
}

#[test]
fn resolve_simple_general() {
    let src = "module foo ports() { wire x = false; comb { id_from_str(\"x\"); } }";
    test_resolve(src, 44, FindDefinition::Found(&[26..27]));
}

#[test]
fn resolve_general_general() {
    let src = "module foo ports() { wire id_from_str(\"x\") = false; comb { id_from_str(\"x\"); } }";
    test_resolve(src, 59, FindDefinition::Found(&[26..42]));
}
