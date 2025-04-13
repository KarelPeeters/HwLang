use crate::front::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::syntax::token::{TokenCategory, TokenError, TokenType, Tokenizer};
use crate::util::iter::IterExt;
use annotate_snippets::Level;
use grammar_wrapper::grammar;
use itertools::enumerate;
use pos::Pos;
pub mod ast;
pub mod parsed;
pub mod pos;
pub mod source;
pub mod token;

#[allow(clippy::all)]
mod grammar_wrapper {
    use lalrpop_util::lalrpop_mod;
    lalrpop_mod!(pub grammar, "/syntax/grammar.rs");
}

pub type ParseError = lalrpop_util::ParseError<Pos, TokenType<String>, TokenError>;

/// Utility struct for the grammer file.
pub struct LocationBuilder {
    file: FileId,
}

impl LocationBuilder {
    pub fn span(&self, start: usize, end: usize) -> Span {
        Span {
            start: Pos {
                file: self.file,
                byte: start,
            },
            end: Pos {
                file: self.file,
                byte: end,
            },
        }
    }
}

pub fn parse_file_content(file: FileId, src: &str) -> Result<ast::FileContent, ParseError> {
    // construct a tokenizer to match the format lalrpop is expecting
    let tokenizer = Tokenizer::new(file, src)
        .into_iter()
        .filter(|token| match token {
            Ok(token) => !matches!(token.ty.category(), TokenCategory::WhiteSpace | TokenCategory::Comment),
            Err(_) => true,
        })
        .map(|token| token.map(|token| (token.span.start.byte, token.ty, token.span.end.byte)));

    // utility converter to include the file in positions and spans
    let location_builder = LocationBuilder { file };

    // actual parsing
    let result = grammar::FileContentParser::new().parse(&location_builder, src, tokenizer);

    // convert the error back to our own formats
    result.map_err(|e| {
        e.map_location(|byte| Pos { file, byte })
            .map_token(|token| token.map(str::to_owned))
    })
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
            let span = Span::new(start, end);

            // TODO use token string instead of name for keywords and symbols
            Diagnostic::new("unexpected token")
                .add_error(span, format!("unexpected token {}", ty.diagnostic_str()))
                .footer(Level::Info, format_expected(&expected))
                .finish()
        }
        ParseError::ExtraToken { token } => {
            let (start, _, end) = token;
            let span = Span::new(start, end);

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
