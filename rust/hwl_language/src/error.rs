use crate::back::LowerError;
use crate::data::diagnostic::Diagnostic;
use crate::data::source::CompileSetError;

#[must_use]
#[derive(Debug)]
pub enum CompileError {
    CompileSetError(CompileSetError),
    // TODO this is not really a category, maybe this should be pushed to different sub-error types?
    //   maybe even with some fancy additional generics for the builder
    SnippetError(Diagnostic),
    LowerError(LowerError),
}

pub type CompileResult<T> = Result<T, CompileError>;

impl From<CompileSetError> for CompileError {
    fn from(error: CompileSetError) -> Self {
        CompileError::CompileSetError(error)
    }
}

impl From<Diagnostic> for CompileError {
    fn from(error: Diagnostic) -> Self {
        CompileError::SnippetError(error)
    }
}

impl From<LowerError> for CompileError {
    fn from(error: LowerError) -> Self {
        CompileError::LowerError(error)
    }
}