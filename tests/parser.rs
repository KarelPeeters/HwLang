use std::path::Path;

use hwlang::parse::pos::{byte_offset_to_pos, FileId};
use hwlang::util::visit_dirs;

fn test_parse(path: impl AsRef<Path>) {
    let src = std::fs::read_to_string(path).unwrap();

    // let result = parse_file(FileId(0), &src).expect("Failed to parse");
    // println!("{:#?}", result);

    let result = hwlang::grammar::PackageContentParser::new().parse(&src);

    match result {
        Ok(package) => {
            println!("{:#?}", package);
        },
        Err(e) => {
            let e = e.map_location(|loc| (loc, byte_offset_to_pos(&src, loc, FileId(0))));
            println!("{:?}", e);
            panic!();
        }
    };
}

#[test]
fn parse_types() {
    test_parse("std/types.kh")
}

#[test]
fn parse_memory() {
    test_parse("std/memory.kh")
}

#[test]
fn parse_synchronize() {
    test_parse("std/synchronize.kh")
}

#[test]
fn parse_new() {
    test_parse("std/new.kh")
}

#[test]
fn parse_std_all() {
    visit_dirs(Path::new("std"), &mut |entry| {
        test_parse(entry.path())
    }).unwrap();
}
