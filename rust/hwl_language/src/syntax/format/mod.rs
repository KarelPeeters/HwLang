//! Formatter for the language.
//!
//! Formatting turns out to be a surprisingly hard problem, especially if we want to do line wrapping
//! and preserve comments and (some) newlines.
//!
//! The current implementation works as follows:
//! * Parse the source code to tokens and an AST.
//! * Convert the AST to an [HNode] tree. The leaf nodes of this tree are token types, not the actual tokens.
//!   This step reduces the formatting problem from a heterogeneous AST into a small set of formatting primitives.
//! * Convert the [HNode] tree to an [LNode] tree.
//!   This steps maps token types in the [HNode] tree to the actual token strings,
//!   and re-inserts comments and newlines where applicable.
//! * Simplify the [LNode] tree to an [LNodeSimple] tree.
//!   This flattens nested sequences and pulls escaping newlines out of their groups.
//!   "Simplify" is a bit of a misnomer, this step is necessary for the correctness of the output.
//! * Finally the [LNodeSimple] tree is formatted to a string, respecting the given [FormatSettings].
//!   At this point we actually make line-breaking decisions.
//!
//! Some resources that were helpful during development:
//! * https://prettier.io/docs/technical-details, https://github.com/prettier/prettier/blob/main/commands.md
//! * https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/
//! * https://yorickpeterse.com/articles/how-to-write-a-code-formatter/
//! * https://github.com/rust-lang/rustfmt/blob/master/Contributing.md
//! * [A prettier printer - Philip Wadler](https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf)
//! * [A New Approach to Optimal Code Formatting - Phillip M. Yelland](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44667.pdf)
//! * [PRETTY PRINTING - Derek C. Oppen - 1979](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/770/CS-TR-79-770.pdf)
//! * [Strictly Pretty - Christian Lindig - 2000](https://lindig.github.io/papers/strictly-pretty-2000.pdf)
//!
//! The current [LNode]s and the low-level line breaking implementation are very similar to prettier.

use crate::front::diagnostic::{DiagError, DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::syntax::ast::FileContent;
use crate::syntax::format::flatten::ast_to_node;
use crate::syntax::format::high::{HNode, lower_nodes};
use crate::syntax::format::low::{LNode, LNodeSimple, StringsStats, node_to_string};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{Token, TokenType, tokenize};
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};

mod common;
mod flatten;
mod high;
mod low;

#[derive(Debug)]
pub struct FormatSettings {
    pub indent_str: String,
    // TODO tab size is such a weird setting, it only matters if `indent_str` or literals contain tabs,
    //   do we really want to support this?
    // TODO assert settings validness somewhere, eg. indent_str should parse as a single whitespace token
    pub tab_size: usize,
    pub max_line_length: usize,
    // TODO add option to sort imports, tricky because tokens won't match and we might lose even more comments
    // pub sort_imports: bool,
    // TODO use \r\n on Windows?
    pub newline_str: String,
}

impl Default for FormatSettings {
    fn default() -> Self {
        Self {
            indent_str: "    ".to_string(),
            tab_size: 4,
            max_line_length: 120,
            newline_str: "\n".to_string(),
        }
    }
}

pub struct FormatOutput<'s> {
    pub old_tokens: Vec<Token>,
    pub old_ast: FileContent,
    pub node_high: HNode,
    pub node_low: LNode<'s>,
    pub node_simple: LNodeSimple<'s>,
    pub stats: StringsStats,
    pub new_content: String,
}

#[derive(Debug, Copy, Clone)]
pub enum FormatError {
    Syntax(DiagError),
    Internal(DiagError),
}

pub fn format<'s>(
    diags: &Diagnostics,
    source: &'s SourceDatabase,
    settings: &FormatSettings,
    file: FileId,
) -> Result<FormatOutput<'s>, FormatError> {
    // tokenize and parse
    let old_info = &source[file];
    let old_content = &old_info.content;
    let old_offsets = &old_info.offsets;
    let old_tokens =
        tokenize(file, old_content, false).map_err(|e| FormatError::Syntax(diags.report(e.to_diagnostic())))?;
    let old_ast = parse_file_content(file, old_content)
        .map_err(|e| FormatError::Syntax(diags.report(parse_error_to_diagnostic(e))))?;

    // flatten the ast to high-level nodes
    let node_high = ast_to_node(&old_ast);

    // lower the high-level nodes to low-level nodes
    let node_low = lower_nodes(old_content, old_offsets, &old_tokens, &node_high).map_err(|e| {
        let msg_slot;
        let (span, msg_source) = match e.index {
            None => (Span::empty_at(source.full_span(file).end()), "reached end end of file"),
            Some(index) => {
                let source_token = &old_tokens[index.0];
                msg_slot = format!("has `{}`", source_token.ty.diagnostic_string());
                (source_token.span, msg_slot.as_str())
            }
        };
        let expected_str = e.expected.diagnostic_string();
        let reason = format!("formatter token mismatch: source {msg_source} but formatter emitted `{expected_str}`");
        FormatError::Internal(diags.report_internal_error(span, reason))
    })?;

    // simplify
    let node_simple = node_low.simplify();

    // format the low-level nodes to a string
    let string_output = node_to_string(settings, old_content, &node_simple);

    // check that the output matches the input, as an extra precaution against formatter bugs
    check_format_output_matches(diags, source, file, &old_tokens, &old_ast, &string_output.string)
        .map_err(FormatError::Internal)?;

    Ok(FormatOutput {
        old_tokens,
        old_ast,
        node_high,
        node_low,
        node_simple,
        stats: string_output.stats,
        new_content: string_output.string,
    })
}

/// Check that formatting the given file yields the same token sequence and parsed AST,
/// to ensure that the semantics haven't changed.
///
/// This is best-effort only, the checks that actually happen are:
/// * the sequence of token types is the same, ignoring commas that might have been added or removed
/// * the content of each token is the same
/// * the number of expressions in the AST is the same
fn check_format_output_matches(
    diags: &Diagnostics,
    source: &SourceDatabase,
    old_file: FileId,
    old_tokens: &[Token],
    old_ast: &FileContent,
    new_content: &str,
) -> DiagResult {
    let old_content = &source[old_file].content;
    let old_span = source.full_span(old_file);

    // re-tokenize the output
    // TODO don't allocate a full vector for tokens, just use the internal iterator
    let dummy_file = FileId::dummy();
    let new_tokens = tokenize(dummy_file, new_content, false)
        .map_err(|e| diags.report_internal_error(old_span, format!("failed to re-tokenize formatter output: {e:?}")))?;

    // check that each old token has a matching new token (ignored removed or added commas)
    let mut new_tokens_iter = new_tokens.iter().peekable();
    for old_token in old_tokens {
        loop {
            let new_token = new_tokens_iter.peek();
            match new_token {
                None => {
                    let diag =
                        Diagnostic::new_internal_error("formatting missing output token, reached end of new file")
                            .add_error(
                                old_token.span,
                                format!("expected `{}`", old_token.ty.diagnostic_string()),
                            )
                            .finish();
                    return Err(diags.report(diag));
                }
                Some(new_token) => {
                    let old_token_str = &old_content[old_token.span.range_bytes()];
                    let new_token_str = &new_content[new_token.span.range_bytes()];
                    if old_token.ty == new_token.ty {
                        if old_token_str == new_token_str {
                            // success
                            new_tokens_iter.next().unwrap();
                            break;
                        } else {
                            let reason = format!(
                                "formatting output token content mismatch for token type `{}`, got `{new_token_str}`",
                                old_token.ty.diagnostic_string()
                            );
                            let diag = Diagnostic::new_internal_error(reason)
                                .add_error(old_token.span, format!("expected `{old_token_str}`"))
                                .finish();
                            return Err(diags.report(diag));
                        }
                    } else if old_token.ty == TokenType::Comma {
                        // comma that was removed
                        break;
                    } else if new_token.ty == TokenType::Comma {
                        // comma that was added
                        new_tokens_iter.next().unwrap();
                        continue;
                    } else {
                        let reason = format!(
                            "formatting output token type mismatch, got `{}` with content `{}`",
                            new_token.ty.diagnostic_string(),
                            new_token_str
                        );
                        let diag = Diagnostic::new_internal_error(reason)
                            .add_error(
                                old_token.span,
                                format!("expected `{}`", old_token.ty.diagnostic_string()),
                            )
                            .finish();
                        return Err(diags.report(diag));
                    }
                }
            }
        }
    }

    // check that there is no leftover new tokens (except commas)
    for new_token in new_tokens_iter {
        if new_token.ty == TokenType::Comma {
            continue;
        }

        let new_token_str = &new_content[new_token.span.range_bytes()];
        let reason = format!(
            "formatting added unexpected output token of type `{}` with content `{}`",
            new_token.ty.diagnostic_string(),
            new_token_str
        );
        let diag = Diagnostic::new_internal_error(reason)
            .add_error(Span::empty_at(old_span.end()), "old file ended here")
            .finish();
        return Err(diags.report(diag));
    }

    // check that parsing still works and yields at least plausible results
    let new_ast =
        parse_file_content(dummy_file, new_content).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;
    if new_ast.arena_expressions.len() != old_ast.arena_expressions.len() {
        return Err(diags.report_internal_error(
            old_span,
            format!(
                "number of expressions changed during formatting, before: {}, after: {}",
                old_ast.arena_expressions.len(),
                new_ast.arena_expressions.len()
            ),
        ));
    }

    Ok(())
}

impl FormatOutput<'_> {
    pub fn debug_str(&self) -> String {
        format!(
            "{}\n\nnode_high:\n{}\n\nnode_low:\n{}\n\nnode_simple:\n{}",
            self.new_content,
            self.node_high.debug_str(),
            self.node_low.debug_str(),
            self.node_simple.debug_str()
        )
    }
}
