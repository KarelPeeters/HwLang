use std::path::Path;

use language::syntax::parse_file_content;
use language::util::visit_dirs;

fn test_parse(path: impl AsRef<Path>) {
    let src = std::fs::read_to_string(path).unwrap();

    // let result = parse_file(FileId(0), &src).expect("Failed to parse");
    // println!("{:#?}", result);

    let result = parse_file_content(&src);

    match result {
        Ok(package) => {
            println!("{:#?}", package);
        }
        Err(e) => {
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
fn parse_util() {
    test_parse("std/util.kh")
}

#[test]
fn parse_new() {
    test_parse("design/sketches/new.kh")
}

#[test]
fn parse_std_all() {
    visit_dirs(Path::new("std"), &mut |entry| test_parse(entry.path())).unwrap();
}
