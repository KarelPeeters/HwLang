use crate::syntax::parse_file_content;
use crate::syntax::pos::{Pos, Span};
use crate::syntax::resolve::{find_definition, FindDefinition};
use crate::syntax::source::FileId;
use itertools::Itertools;
use std::ops::Range;

#[track_caller]
fn test_resolve(src: &str, pos: usize, expected: FindDefinition<&[Range<usize>]>) {
    let file = FileId::dummy();
    let ast = parse_file_content(file, src).unwrap();

    let pos = Pos { file, byte: pos };
    let actual_spans = find_definition(&ast, pos);

    let expected = match expected {
        FindDefinition::Found(expected_spans) => {
            let spans = expected_spans
                .iter()
                .map(|r| Span::new(Pos { file, byte: r.start }, Pos { file, byte: r.end }))
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
