use crate::tests::test_format::util::assert_formatted;

#[test]
fn test_declaration() {
    assert_file_stable("declaration.kh");
}

#[test]
fn test_expression() {
    assert_file_stable("expression.kh");
}

#[test]
fn test_module() {
    assert_file_stable("module.kh");
}

#[test]
fn test_statement() {
    assert_file_stable("statement.kh");
}

#[test]
fn test_preserve() {
    assert_file_stable("preserve.kh");
}

pub fn assert_file_stable(file_name: &str) {
    let src = std::fs::read_to_string(format!("src/tests/test_format/{file_name}")).unwrap();
    assert_formatted(&src);
}
