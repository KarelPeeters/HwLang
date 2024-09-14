use crate::back::LowerError;
use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::source::CompileSetError;

#[must_use]
#[derive(Debug)]
pub enum CompileError {
    CompileSetError(CompileSetError),
    Diagnostic(ErrorGuaranteed),
    LowerError(LowerError),
}

pub type CompileResult<T> = Result<T, CompileError>;

impl From<CompileSetError> for CompileError {
    fn from(value: CompileSetError) -> Self {
        CompileError::CompileSetError(value)
    }
}

impl From<ErrorGuaranteed> for CompileError {
    fn from(value: ErrorGuaranteed) -> Self {
        CompileError::Diagnostic(value)
    }
}

impl From<LowerError> for CompileError {
    fn from(value: LowerError) -> Self {
        CompileError::LowerError(value)
    }
}