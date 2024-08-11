use crate::front::driver::CompileSetError;
use crate::front::error::FrontError;
use crate::front::scope::ScopeError;
use crate::syntax::ParseError;

#[derive(Debug)]
pub enum CompileError {
    CompileSetError(CompileSetError),
    ParseError(ParseError),
    ScopeError(ScopeError),
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

impl From<ScopeError> for CompileError {
    fn from(error: ScopeError) -> Self {
        CompileError::ScopeError(error)
    }
}

impl From<FrontError> for CompileError {
    fn from(error: FrontError) -> Self {
        CompileError::FrontError(error)
    }
}
