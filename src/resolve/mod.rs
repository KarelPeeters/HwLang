use crate::syntax::ast;

pub mod compile_set;
pub mod scope;

pub type ResolveResult<T> = Result<T, ResolveError>;

#[derive(Debug)]
pub enum ResolveError {
    IdentifierDeclaredTwice(ast::Identifier),
    UndeclaredIdentifier(ast::Identifier),
}
