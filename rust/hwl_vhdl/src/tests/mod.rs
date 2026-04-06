use crate::syntax::parser::{parse_error_to_diagnostic, parse_file_content_without_recovery};
use crate::syntax::token::tokenize;
use hwl_common::diagnostic::{diags_to_string, Diagnostics};
use hwl_common::source::SourceDatabase;
use std::path::Path;

// TODO test standard packages
// TODO copy over all LRM examples
// TODO use test cases from nvc and ghdl
pub mod test_cases_ghdl;
pub mod test_cases_nvc;
pub mod test_syntax_custom;
pub mod test_syntax_lrm;

pub fn test_parse(src: &str) {
    println!("{}", src);

    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy.vhd".to_owned(), src.to_owned());

    let diags = Diagnostics::new();
    let mut any_fail = false;

    match tokenize(file, src, false) {
        Ok(_) => {}
        Err(e) => {
            any_fail = true;
            let _ = e.to_diagnostic().report(&diags);
        }
    }

    let result = parse_file_content_without_recovery(file, src);

    match result {
        Ok(result) => {
            println!("{:#?}", result);
        }
        Err(e) => {
            any_fail = true;
            let _ = parse_error_to_diagnostic(e).report(&diags);
        }
    }

    let diags = diags.finish();
    let any_diags = !diags.is_empty();
    println!("{}", diags_to_string(&source, diags, true));
    assert!(!any_diags && !any_fail);
}

pub fn test_parse_files(paths: &[&str]) {
    let mut source = SourceDatabase::new();
    let diags = Diagnostics::new();

    for &path_rel in paths {
        let path = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(path_rel);
        println!("{:?}", path);

        let src = std::fs::read_to_string(&path).unwrap();

        let file = source.add_file(path_rel.to_owned(), src.to_owned());
        let result = parse_file_content_without_recovery(file, &src);

        match result {
            Ok(result) => {
                println!("{:#?}", result);
            }
            Err(e) => {
                let _ = parse_error_to_diagnostic(e).report(&diags);
            }
        }
    }

    let diags = diags.finish();
    let any_diag = !diags.is_empty();
    println!("{}", diags_to_string(&source, diags, true));

    if any_diag {
        panic!("failed");
    }
}
