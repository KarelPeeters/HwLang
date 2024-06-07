use lalrpop_util::{lexer::Token, ParseError};
use pos::{byte_offset_to_pos, FileId, Pos};

use crate::grammar::PackageContentParser;

pub mod ast;
pub mod pos;
// pub mod parser;

pub fn parse_package_content(
    src: &str,
) -> Result<ast::PackageContent, ParseError<Pos, Token, &str>> {
    PackageContentParser::new()
        .parse(&src)
        .map_err(|e| e.map_location(|loc| byte_offset_to_pos(&src, loc, FileId(0)).unwrap()))
}
