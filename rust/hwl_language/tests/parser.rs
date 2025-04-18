use std::path::Path;

use hwl_language::syntax::source::FileId;
use hwl_language::{
    syntax::parse_file_content,
    util::io::{recurse_for_each_file, IoErrorWithPath},
};

fn test_parse(path: impl AsRef<Path>) {
    let src = std::fs::read_to_string(path).unwrap();

    let result = parse_file_content(FileId::dummy(), &src);

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
    test_parse("../../design/project/std/types.kh")
}

#[test]
fn parse_util() {
    test_parse("../../design/project/util/test.kh")
}

#[test]
fn parse_top() {
    test_parse("../../design/project/top.kh")
}

#[test]
fn parse_std_all() {
    recurse_for_each_file::<IoErrorWithPath>(Path::new("../../design/project/std"), |_, entry| {
        test_parse(entry.path());
        Ok(())
    })
    .unwrap();
}
