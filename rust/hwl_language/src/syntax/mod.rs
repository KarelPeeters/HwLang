use lalrpop_util::lalrpop_mod;

use pos::Pos;

use crate::syntax::pos::{FileId, Span};
use crate::syntax::token::{TokenCategory, TokenType, Tokenizer};
use crate::util::Never;

pub mod ast;
pub mod pos;
pub mod token;

lalrpop_mod!(grammar, "/syntax/grammar.rs");

// TODO convert to diagnostic
pub type ParseError = lalrpop_util::ParseError<Pos, TokenType<String>, Never>;

/// Utility struct for the grammer file.
pub struct LocationBuilder {
    file: FileId,
}

impl LocationBuilder {
    pub fn span(&self, start: usize, end: usize) -> Span {
        Span {
            start: Pos { file: self.file, byte: start },
            end: Pos { file: self.file, byte: end },
        }
    }
}

pub fn parse_file_content(file: FileId, src: &str) -> Result<ast::FileContent, ParseError> {
    // construct a tokenizer to match the format lalrpop is expecting
    let tokenizer = Tokenizer::new(file, src)
        .filter(|token| match token {
            Ok(token) => !matches!(token.ty.category(), TokenCategory::WhiteSpace | TokenCategory::Comment),
            Err(_) => true,
        })
        .map(|token| token.map(|token| (token.span.start.byte, token.ty, token.span.end.byte)));

    // utility converter to include the file in positions and spans
    let location_builder = LocationBuilder { file };

    // actual parsing
    let result = grammar::FileContentParser::new()
        .parse(&location_builder, &src, tokenizer);

    // convert the error back to our own formats
    result.map_err(|e| {
        e.map_location(|byte| Pos { file, byte })
            .map_token(|token| token.map(str::to_owned))
            .map_error(|_| unreachable!("no custom errors used in the grammer"))
    })
}
