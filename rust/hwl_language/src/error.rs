use crate::back::LowerError;
use crate::data::diagnostic::DiagnosticError;
use crate::data::source::CompileSetError;

#[must_use]
#[derive(Debug)]
pub enum CompileError {
    CompileSetError(CompileSetError),
    Diagnostic(DiagnosticError),
    LowerError(LowerError),
}

pub type CompileResult<T> = Result<T, CompileError>;

impl From<CompileSetError> for CompileError {
    fn from(value: CompileSetError) -> Self {
        CompileError::CompileSetError(value)
    }
}

impl From<DiagnosticError> for CompileError {
    fn from(value: DiagnosticError) -> Self {
        CompileError::Diagnostic(value)
    }
}

impl From<LowerError> for CompileError {
    fn from(value: LowerError) -> Self {
        CompileError::LowerError(value)
    }
}