use crate::resolve::compile::{CompileSetError, FrontError};
use crate::resolve::error::ResolveError;
use crate::syntax::ParseError;

#[derive(Debug)]
pub enum CompileError {
    CompileSetError(CompileSetError),
    ParseError(ParseError),
    ResolveError(ResolveError),
    FrontError(FrontError),
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

impl From<ResolveError> for CompileError {
    fn from(error: ResolveError) -> Self {
        CompileError::ResolveError(error)
    }
}

impl From<FrontError> for CompileError {
    fn from(error: FrontError) -> Self {
        CompileError::FrontError(error)
    }
}
