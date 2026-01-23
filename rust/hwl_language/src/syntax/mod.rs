use crate::front::diagnostic::DiagnosticError;
use crate::syntax::ast::FileContent;
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::syntax::token::{TokenCategory, TokenError, TokenType, Tokenizer};
use crate::util::arena::Arena;
use crate::util::iter::IterExt;
use grammar_wrapper::grammar;
use itertools::enumerate;
use pos::Pos;

pub mod ast;
pub mod collect;
pub mod external;
pub mod format;
pub mod hierarchy;
pub mod manifest;
pub mod parsed;
pub mod pos;
pub mod source;
pub mod token;
pub mod visitor;

#[allow(clippy::all)]
mod grammar_wrapper {
    use lalrpop_util::lalrpop_mod;
    lalrpop_mod!(pub grammar, "/syntax/grammar.rs");
}

pub type ParseError = lalrpop_util::ParseError<Pos, TokenType, TokenError>;

/// Utility struct for the grammar file.
pub struct ParseContext {
    file: FileId,
    errors: Vec<ParseError>,
}

/// Token that proves that a parse error has been reported, intentionally not constructible outside this module.
/// Similar to [crate::front::diagnostic::DiagError].
#[derive(Debug, Copy, Clone)]
pub struct ReportedParseError(());

impl ParseContext {
    pub fn span(&self, start_byte: usize, end_byte: usize) -> Span {
        Span {
            file: self.file,
            start_byte,
            end_byte,
        }
    }

    pub fn pos(&self, byte: usize) -> Pos {
        Pos { file: self.file, byte }
    }

    pub fn push_error(
        &mut self,
        recovery: lalrpop_util::ErrorRecovery<usize, TokenType, TokenError>,
    ) -> ReportedParseError {
        // we don't need the dropped tokens for now, so we drop them here
        let lalrpop_util::ErrorRecovery {
            error,
            dropped_tokens: _,
        } = recovery;
        self.errors.push(error.map_location(|x| self.pos(x)));
        ReportedParseError(())
    }
}

/// [FileContent], possibly with some recovered errors.
pub struct FileContentRecovery {
    pub recovered_content: FileContent,
    pub errors: Vec<ParseError>,
}

pub fn parse_file_content_without_recovery(file: FileId, src: &str) -> Result<FileContent, ParseError> {
    let content = parse_file_content_with_recovery(file, src)?;

    let FileContentRecovery {
        recovered_content,
        errors,
    } = content;

    if let Some(err) = errors.into_iter().next() {
        Err(err)
    } else {
        Ok(recovered_content)
    }
}

pub fn parse_file_content_with_recovery(file: FileId, src: &str) -> Result<FileContentRecovery, ParseError> {
    // construct a tokenizer to match the format lalrpop is expecting
    let tokenizer = Tokenizer::new(file, src, false)
        .into_iter()
        .filter(|token| match token {
            Ok(token) => !matches!(token.ty.category(), TokenCategory::Comment),
            Err(_) => true,
        })
        .map(|token| token.map(|token| (token.span.start_byte, token.ty, token.span.end_byte)));

    // utility converter to include the file in positions and spans
    let mut ctx = ParseContext {
        file,
        errors: Vec::new(),
    };

    // actual parsing
    let mut arena_expressions = Arena::new();
    let result = grammar::FileItemsParser::new().parse(&mut ctx, &mut arena_expressions, tokenizer);

    match result {
        Ok(file_items) => {
            let span_full = Span::new(file, 0, src.len());
            let content = FileContent {
                span: span_full,
                items: file_items,
                arena_expressions,
            };
            Ok(FileContentRecovery {
                recovered_content: content,
                errors: ctx.errors,
            })
        }
        Err(e) => Err(e.map_location(|byte| Pos { file, byte })),
    }
}

pub fn parse_error_to_diagnostic(error: ParseError) -> DiagnosticError {
    match error {
        ParseError::InvalidToken { location } => {
            let span = Span::empty_at(location);
            DiagnosticError::new("invalid token", span, "invalid token")
        }
        ParseError::UnrecognizedEof { location, expected } => {
            let span = Span::empty_at(location);

            DiagnosticError::new("unexpected eof", span, "invalid token").add_footer_info(format_expected(&expected))
        }
        ParseError::UnrecognizedToken { token, expected } => {
            let (start, ty, end) = token;
            let span = Span::new(start.file, start.byte, end.byte);

            // TODO use token string instead of name for keywords and symbols
            DiagnosticError::new(
                "unexpected token",
                span,
                format!("unexpected token `{}`", ty.diagnostic_string()),
            )
            .add_footer_info(format_expected(&expected))
        }
        ParseError::ExtraToken { token } => {
            let (start, _, end) = token;
            let span = Span::new(start.file, start.byte, end.byte);

            DiagnosticError::new("unexpected extra token", span, "extra token")
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
