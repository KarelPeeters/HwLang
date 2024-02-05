use std::path::Path;
use hwlang::parse::parser::parse_file;
use hwlang::parse::pos::FileId;
use hwlang::util::visit_dirs;

#[test]
fn parse_std() {
    visit_dirs(Path::new("std"), &mut |entry| {
        let src = std::fs::read_to_string(entry.path()).unwrap();
        let result = parse_file(FileId(0), &src).expect("Failed to parse");
        println!("{:#?}", result);
    }).unwrap();
}
