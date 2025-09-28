use crate::front::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::syntax::ast::FileContent;
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::syntax::token::{TokenCategory, TokenError, TokenType, Tokenizer};
use crate::util::arena::Arena;
use crate::util::iter::IterExt;
use annotate_snippets::Level;
use grammar_wrapper::grammar;
use itertools::enumerate;
use pos::Pos;

pub mod ast;
pub mod collect;
pub mod format;
pub mod hierarchy;
pub mod manifest;
pub mod parsed;
pub mod pos;
pub mod resolve;
pub mod source;
pub mod token;

#[allow(clippy::all)]
mod grammar_wrapper {
    use lalrpop_util::lalrpop_mod;
    lalrpop_mod!(pub grammar, "/syntax/grammar.rs");
}

pub type ParseError = lalrpop_util::ParseError<Pos, TokenType, TokenError>;

/// Utility struct for the grammer file.
pub struct LocationBuilder {
    file: FileId,
}

impl LocationBuilder {
    pub fn span(&self, start_byte: usize, end_byte: usize) -> Span {
        Span {
            file: self.file,
            start_byte,
            end_byte,
        }
    }
}

pub fn parse_file_content(file: FileId, src: &str) -> Result<FileContent, ParseError> {
    // construct a tokenizer to match the format lalrpop is expecting
    let tokenizer = Tokenizer::new(file, src, false)
        .into_iter()
        .filter(|token| match token {
            Ok(token) => !matches!(token.ty.category(), TokenCategory::Comment),
            Err(_) => true,
        })
        .map(|token| token.map(|token| (token.span.start_byte, token.ty, token.span.end_byte)));

    // utility converter to include the file in positions and spans
    let location_builder = LocationBuilder { file };

    // actual parsing
    // TODO create and pass arena
    // TODO wrap in content
    let mut arena_expressions = Arena::new();
    let result = grammar::FileItemsParser::new().parse(&location_builder, &mut arena_expressions, tokenizer);

    match result {
        Ok(file_items) => {
            let span_full = Span::new(file, 0, src.len());
            Ok(FileContent {
                span: span_full,
                items: file_items,
                arena_expressions,
            })
        }
        Err(e) => Err(e.map_location(|byte| Pos { file, byte })),
    }
}

pub fn parse_error_to_diagnostic(error: ParseError) -> Diagnostic {
    match error {
        ParseError::InvalidToken { location } => {
            let span = Span::empty_at(location);
            Diagnostic::new("invalid token")
                .add_error(span, "invalid token")
                .finish()
        }
        ParseError::UnrecognizedEof { location, expected } => {
            let span = Span::empty_at(location);

            Diagnostic::new("unexpected eof")
                .add_error(span, "invalid token")
                .footer(Level::Info, format_expected(&expected))
                .finish()
        }
        ParseError::UnrecognizedToken { token, expected } => {
            let (start, ty, end) = token;
            let span = Span::new(start.file, start.byte, end.byte);

            // TODO use token string instead of name for keywords and symbols
            Diagnostic::new("unexpected token")
                .add_error(span, format!("unexpected token {}", ty.diagnostic_string()))
                .footer(Level::Info, format_expected(&expected))
                .finish()
        }
        ParseError::ExtraToken { token } => {
            let (start, _, end) = token;
            let span = Span::new(start.file, start.byte, end.byte);

            Diagnostic::new("unexpected extra token")
                .add_error(span, "extra token")
                .finish()
        }
        ParseError::User { error } => error.to_diagnostic(),
    }
}

// Workaround for lalrpop using strings instead of a real token type
fn format_expected(expected: &[String]) -> String {
    if let Some(expected) = expected.iter().single() {
        return format!("expected {}", format_expected_single(expected));
    }

    let mut result = String::new();
    result.push_str("expected one of [");
    for (i, e) in enumerate(expected) {
        if i > 0 {
            result.push_str(", ");
        }
        result.push_str(format_expected_single(e));
    }
    result.push(']');
    result
}

fn format_expected_single(expected: &str) -> &str {
    expected.strip_prefix("Token").unwrap_or(expected)
}
