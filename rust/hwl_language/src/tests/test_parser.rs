use crate::syntax::parse_file_content_without_recovery;
use crate::syntax::source::FileId;
use hwl_util::io::{IoErrorWithPath, recurse_for_each_file};
use std::path::Path;

fn test_parse(path: impl AsRef<Path>) {
    let src = std::fs::read_to_string(path).unwrap();

    let result = parse_file_content_without_recovery(FileId::dummy(), &src);

    match result {
        Ok(package) => {
            println!("{package:#?}");
        }
        Err(e) => {
            println!("{e:?}");
            panic!();
        }
    };
}

#[test]
fn parse_std_all() {
    let mut count = 0;
    recurse_for_each_file::<IoErrorWithPath>(Path::new("../hwl_std/src/std"), |_, entry| {
        test_parse(entry);
        count += 1;
        Ok(())
    })
    .unwrap();

    assert!(count > 0);
}
