use annotate_snippets::Level;
use itertools::enumerate;
use lalrpop_util::lalrpop_mod;

use crate::front::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::syntax::pos::{FileId, Span};
use crate::syntax::token::{TokenCategory, TokenError, TokenType, Tokenizer};
use pos::Pos;

pub mod ast;
pub mod parsed;
pub mod pos;
pub mod source;
pub mod token;

lalrpop_mod!(grammar, "/syntax/grammar.rs");

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
    let result = grammar::FileContentParser::new().parse(&location_builder, &src, tokenizer);

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
                .footer(Level::Info, format!("expected one of {}", format_expected(&expected)))
                .finish()
        }
        ParseError::UnrecognizedToken { token, expected } => {
            let (start, ty, end) = token;
            let span = Span::new(start, end);

            let ty_formatted = format!("{:?}", ty.map(|_| ())).replace("(())", "");

            Diagnostic::new("unexpected token")
                .add_error(span, format!("unexpected token {:?}", ty_formatted))
                .footer(Level::Info, format!("expected one of {}", format_expected(&expected)))
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
    let mut result = String::new();
    result.push_str("[");
    for (i, e) in enumerate(expected) {
        if i > 0 {
            result.push_str(", ");
        }
        if let Some(suffix) = e.strip_prefix("Token") {
            result.push_str(suffix);
        } else {
            result.push_str(e);
        }
    }
    result.push_str("]");
    result
}
