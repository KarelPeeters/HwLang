use hwlang::parse::parser::parse_file;
use hwlang::parse::pos::FileId;

#[test]
fn parse() {
    let src = include_str!("parse.kh");

    let result = parse_file(FileId(0), src).expect("Failed to parse");

    println!("{:#?}", result);
}