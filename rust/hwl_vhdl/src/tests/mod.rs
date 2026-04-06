use crate::syntax::parser::{parse_error_to_diagnostic, parse_file_content_without_recovery};
use hwl_common::diagnostic::{diags_to_string, Diagnostics};
use hwl_common::source::SourceDatabase;

// TODO test standard packages
// TODO copy over all LRM examples
// TODO use test cases from nvc and ghdl
pub mod test_syntax_custom;
pub mod test_syntax_lrm;

pub fn test_parse(src: &str) {
    println!("{}", src);

    let mut source = SourceDatabase::new();
    let file = source.add_file("dummy.vhd".to_owned(), src.to_owned());
    let result = parse_file_content_without_recovery(file, src);

    match result {
        Ok(result) => {
            println!("{:#?}", result);
        }
        Err(e) => {
            let diags = Diagnostics::new();
            let _ = parse_error_to_diagnostic(e).report(&diags);
            println!("{}", diags_to_string(&source, diags.finish(), true));
            panic!();
        }
    }
}
