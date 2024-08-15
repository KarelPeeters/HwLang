use crate::front::driver::CompileSetError;
use crate::front::scope::ScopeError;
use crate::syntax::ParseError;

#[must_use]
#[derive(Debug)]
pub enum CompileError {
    SnippetError(DiagnosticError),
    CompileSetError(CompileSetError),
    ParseError(ParseError),
    ScopeError(ScopeError),
}

#[must_use]
#[derive(Debug)]
pub struct DiagnosticError {
    pub string: String,
}

impl From<CompileSetError> for CompileError {
    fn from(error: CompileSetError) -> Self {
        CompileError::CompileSetError(error)
    }
}

impl From<ParseError> for CompileError {
    fn from(error: ParseError) -> Self {
        CompileError::ParseError(error)
    }
}

impl From<ScopeError> for CompileError {
    fn from(error: ScopeError) -> Self {
        CompileError::ScopeError(error)
    }
}

impl From<DiagnosticError> for CompileError {
    fn from(error: DiagnosticError) -> Self {
        CompileError::SnippetError(error)
    }
}
