use crate::back::LowerError;
use crate::data::source::CompileSetError;

#[must_use]
#[derive(Debug)]
pub enum CompileError {
    // TODO this is not really a catagory, maybe this should be pushed to different sub-error types?
    //   maybe even with some fancy additional generics for the builder
    SnippetError(DiagnosticError),
    CompileSetError(CompileSetError),
    LowerError(LowerError)
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

impl From<LowerError> for CompileError {
    fn from(error: LowerError) -> Self {
        CompileError::LowerError(error)
    }
}