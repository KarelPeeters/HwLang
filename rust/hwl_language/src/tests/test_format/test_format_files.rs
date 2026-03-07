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

    // initial format with unlimited line length, to determine the max line length the formatter wants to use
    let settings_inf = FormatSettings {
        max_line_length: usize::MAX,
        ..Default::default()
    };
    let result_inf = assert_format_valid("dummy_inf.kh", &src, &settings_inf);
    let max_line_length = result_inf.lines().map(str::len).max().unwrap_or(0);

    // format with different line lengths to trigger wrapping edge cases
    for len in (0..max_line_length).rev() {
        let settings = FormatSettings {
            max_line_length: len,
            ..Default::default()
        };
        assert_format_valid(&format!("dummy_{len}.kh"), &src, &settings);
    }
}
