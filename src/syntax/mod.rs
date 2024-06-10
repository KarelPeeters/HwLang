use lalrpop_util::lalrpop_mod;

use pos::{byte_offset_to_pos, FileId, Pos};

use crate::util::Never;

pub mod ast;
pub mod pos;

lalrpop_mod!(grammar, "/syntax/grammar.rs");

pub type ParseError = lalrpop_util::ParseError<Pos, String, Never>;

pub fn parse_file_content(src: &str, file_id: FileId) -> Result<ast::FileContent, ParseError> {
    grammar::FileContentParser::new()
        .parse(&src)
        .map_err(|e| {
            e.map_location(|loc| byte_offset_to_pos(&src, loc, file_id).unwrap())
                .map_token(|token| token.1.to_owned())
                .map_error(|_| unreachable!("no custom errors used in the grammer"))
        })
}
