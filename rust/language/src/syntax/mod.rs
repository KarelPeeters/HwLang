use lalrpop_util::lalrpop_mod;

use pos::Pos;

use crate::syntax::pos::{FileId, FileOffsets, Span};
use crate::syntax::token::tokenize;
use crate::util::Never;

pub mod ast;
pub mod pos;
pub mod token;

lalrpop_mod!(grammar, "/syntax/grammar.rs");

pub type ParseError = lalrpop_util::ParseError<Pos, String, Never>;

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

pub fn parse_file_content(src: &str, offsets: &FileOffsets) -> Result<ast::FileContent, ParseError> {
    let file = offsets.file();
    
    // test the tokenizer
    // TODO this will go away once we're actually using the tokenizer for parsing
    match tokenize(file, src) {
        Ok(_) => {},
        Err(e) => {
            return Err(ParseError::InvalidToken { location: e.pos });
        },
    }

    // actual parsing
    let location_builder = LocationBuilder { file };
    grammar::FileContentParser::new()
        .parse(&location_builder, &src)
        .map_err(|e| {
            e.map_location(|byte| Pos { file, byte })
                .map_token(|token| token.1.to_owned())
                .map_error(|_| unreachable!("no custom errors used in the grammer"))
        })
}
