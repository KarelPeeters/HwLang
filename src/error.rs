use lalrpop_util::lexer::Token;
use lalrpop_util::ParseError;

use crate::resolve::ResolveError;
use crate::syntax::pos::Pos;

#[derive(Debug)]
pub enum Error {
    ParseError(ParseError<Pos, Token<'static>, String>),
    ResolveError(ResolveError),
}

impl From<ParseError<Pos, Token<'static>, String>> for Error {
    fn from(error: ParseError<Pos, Token<'static>, String>) -> Self {
        Error::ParseError(error)
    }
}

impl From<ResolveError> for Error {
    fn from(error: ResolveError) -> Self {
        Error::ResolveError(error)
    }
}
