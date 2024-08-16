use crate::front::source::CompileSetError;

#[must_use]
#[derive(Debug)]
pub enum CompileError {
    SnippetError(DiagnosticError),
    CompileSetError(CompileSetError),
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

impl From<DiagnosticError> for CompileError {
    fn from(error: DiagnosticError) -> Self {
        CompileError::SnippetError(error)
    }
}
