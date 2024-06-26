use std::panic::Location;
use lalrpop_util::lalrpop_mod;

use pos::{byte_offset_to_pos, FileId, Pos};
use crate::syntax::pos::LocationBuilder;

use crate::util::Never;

pub mod ast;
pub mod pos;

lalrpop_mod!(grammar, "/syntax/grammar.rs");

pub type ParseError = lalrpop_util::ParseError<Pos, String, Never>;

pub fn parse_file_content(file_id: FileId, src: &str) -> Result<ast::FileContent, ParseError> {
    let loc = LocationBuilder::new(file_id, src);
    grammar::FileContentParser::new()
        .parse(&loc, &src)
        .map_err(|e| {
            e.map_location(|offset| byte_offset_to_pos(&src, offset, file_id).unwrap())
                .map_token(|token| token.1.to_owned())
                .map_error(|_| unreachable!("no custom errors used in the grammer"))
        })
}
