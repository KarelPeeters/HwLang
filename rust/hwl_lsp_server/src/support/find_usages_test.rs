use crate::support::PosNotOnIdentifier;
use crate::support::find_usages::find_usages;
use hwl_language::syntax::parser::parse_file_content_without_recovery;
use hwl_language::syntax::pos::{Pos, Span};
use hwl_language::syntax::source::SourceDatabase;
use itertools::Itertools;
use std::ops::Range;

#[track_caller]
fn test_usages(src: &str, pos: usize, expected: Result<&[Range<usize>], PosNotOnIdentifier>) {
    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy".to_owned(), src.to_owned());
    let ast = parse_file_content_without_recovery(file, src).unwrap();

    let pos = Pos { file, byte: pos };
    let actual_spans = find_usages(&source, &ast, pos);

    let expected_typed =
        expected.map(|expected| expected.iter().map(|r| Span::new(file, r.start, r.end)).collect_vec());
    assert_eq!(expected_typed, actual_spans);
}

#[test]
fn find_basic_on_decl() {
    let src = "const a = 5; const b = a; const c = a;";
    test_usages(src, 0, Err(PosNotOnIdentifier));
    test_usages(src, 6, Ok(&[23..24, 36..37]));
}

#[test]
fn find_basic_on_use() {
    let src = "const a = 5; const b = a; const c = a;";
    let expected = &[23..24, 36..37];
    test_usages(src, 23, Ok(expected));
    test_usages(src, 36, Ok(expected));
}

#[test]
fn find_general_id() {
    let src = "const a = \"c\"; module m ports() { wire ident(a) = false; wire w0 = c; wire w1 = a; }";
    // usages of ident itself
    test_usages(src, 39, Ok(&[67..68, 80..81]));
    // usages of identifiers inside ident
    test_usages(src, 51 - 6, Ok(&[45..46, 80..81]));
}
