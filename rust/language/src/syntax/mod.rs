use lalrpop_util::lalrpop_mod;

use pos::{FileId, Pos};
use crate::syntax::pos::FileOffsets;

use crate::syntax::token::tokenize;
use crate::util::Never;

pub mod ast;
pub mod pos;
pub mod token;

lalrpop_mod!(grammar, "/syntax/grammar.rs");

pub type ParseError = lalrpop_util::ParseError<Pos, String, Never>;

pub fn parse_file_content(file_id: FileId, src: &str) -> Result<ast::FileContent, ParseError> {
    // test the tokenizer
    match tokenize(file_id, src) {
        Ok(_) => {},
        Err(e) => {
            return Err(ParseError::InvalidToken { location: e.pos });
        },
    }
    
    let offsets = FileOffsets::new(file_id, src);
    grammar::FileContentParser::new()
        .parse(&offsets, &src)
        .map_err(|e| {
            e.map_location(|offset| offsets.byte_to_pos(offset))
                .map_token(|token| token.1.to_owned())
                .map_error(|_| unreachable!("no custom errors used in the grammer"))
        })
}
