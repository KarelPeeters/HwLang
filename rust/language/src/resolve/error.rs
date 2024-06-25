use crate::syntax::ast;

pub type ResolveResult<T> = Result<T, ResolveError>;

#[derive(Debug)]
pub enum ResolveError {
    IdentifierDeclaredTwice(ast::Identifier),
    UndeclaredIdentifier(ast::Identifier),
    CannotAcess(ast::Identifier),
}
