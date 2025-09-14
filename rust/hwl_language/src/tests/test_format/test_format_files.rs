use crate::tests::test_format::util::assert_formatted;

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
    let src = std::fs::read_to_string(format!("src/tests/test_format/{file_name}")).unwrap();
    assert_formatted(&src);

    // TODO try gradually reducing/increasing the width to find extra edge cases
}
