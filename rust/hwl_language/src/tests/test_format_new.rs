use crate::front::diagnostic::{Diagnostics, diags_to_debug_string};
use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::format;
use crate::syntax::source::SourceDatabase;

#[test]
fn test_declaration() {
    test_stable("declaration.kh");
}

#[test]
fn test_expression() {
    test_stable("expression.kh");
}

#[test]
fn test_module() {
    test_stable("module.kh");
}

#[test]
fn test_statement() {
    test_stable("statement.kh");
}

#[test]
fn test_preserve() {
    test_stable("preserve.kh");
}

fn test_stable(file_name: &str) {
    let content = std::fs::read_to_string(format!("src/tests/format/{file_name}")).unwrap();

    let diags = Diagnostics::new();
    let mut source = SourceDatabase::new();
    let file = source.add_file(file_name.to_owned(), content.clone());

    let content_new = match format(&diags, &mut source, &FormatSettings::default(), file) {
        Ok(content_new) => content_new,
        Err(_) => {
            eprintln!("{}", diags_to_debug_string(&source, diags.finish()));
            panic!("formatting failed");
        }
    };

    assert_eq!(content_new, content);
}
