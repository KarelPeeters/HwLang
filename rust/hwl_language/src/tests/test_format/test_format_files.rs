use crate::syntax::format::FormatSettings;
use crate::tests::test_format::util::{assert_format_valid, assert_formatted};

#[test]
fn test_declaration() {
    run_file_tests("declaration.kh");
}

#[test]
fn test_expression() {
    run_file_tests("expression.kh");
}

#[test]
fn test_module() {
    run_file_tests("module.kh");
}

#[test]
fn test_statement() {
    run_file_tests("statement.kh");
}

#[test]
fn test_preserve() {
    run_file_tests("preserve.kh");
}

pub fn run_file_tests(file_name: &str) {
    // check that the file itself is properly formatted
    let src = std::fs::read_to_string(format!("src/tests/test_format/{file_name}")).unwrap();
    assert_formatted(&src);

    // format the file with different max line lengths to trigger wrapping edge cases
    // TODO speed this up with bisect instead of this brute forcing
    let max_line_len = src.lines().map(str::len).max().unwrap_or(0);
    for len in (0..max_line_len).rev() {
        let settings = FormatSettings {
            max_line_length: len,
            ..Default::default()
        };
        assert_format_valid(&format!("dummy_{len}"), &src, &settings);
    }
}
