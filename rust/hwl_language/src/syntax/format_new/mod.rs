//! Sketch of the formatter implementation:
//! * convert the AST to a tree of FNodes, which are just tokens with some extra structure
//! * cross-reference emitted formatting tokens with the original tokens to get span info
//! * bottom-up traverse the format tree to figure out which branches need to wrap due to line comments
//! * top-down traverse the format tree, printing the tokens to the final buffer
//!    * if any line overflows and we can still make additional wrapping choices, roll back and try
//!    * if blank lines between items: insert matching blank line
//! TODO add debug mode that slowly decreases the line width and saves every time the output changes
//! TODO document all of this

use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::syntax::ast::FileContent;
use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::flatten::ast_to_node;
use crate::syntax::format_new::high::{HNode, lower_nodes};
use crate::syntax::format_new::low::{LNode, LNodeSimple, StringsStats, node_to_string};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{Token, TokenType, tokenize};
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};

mod common;
mod flatten;
mod high;
mod low;

pub struct FormatOutput<'s> {
    pub old_tokens: Vec<Token>,
    pub old_ast: FileContent,
    pub node_high: HNode,
    pub node_low: LNode<'s>,
    pub node_simple: LNodeSimple<'s>,
    pub stats: StringsStats,
    pub new_content: String,
}

pub fn format<'s>(
    diags: &Diagnostics,
    source: &'s SourceDatabase,
    settings: &FormatSettings,
    file: FileId,
) -> DiagResult<FormatOutput<'s>> {
    // tokenize and parse
    let old_info = &source[file];
    let old_content = &old_info.content;
    let old_offsets = &old_info.offsets;
    let mut old_tokens = tokenize(file, old_content, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    // TODO remove whitespace tokens, everyone just filters them out anyway, we can always recover whitespace as "stuff between other tokens" if we really want to
    old_tokens.retain(|t| t.ty != TokenType::WhiteSpace);
    let old_ast = parse_file_content(file, old_content).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;

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
        diags.report_internal_error(span, reason)
    })?;

    // simplify
    let node_simple = node_low.simplify();

    // format the low-level nodes to a string
    let string_output = node_to_string(settings, old_content, &node_simple);
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
pub fn check_format_output_matches(
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
    let dummy_file = FileId::dummy();
    let mut new_tokens = tokenize(dummy_file, new_content, false)
        .map_err(|e| diags.report_internal_error(old_span, format!("failed to re-tokenize formatter output: {e:?}")))?;
    new_tokens.retain(|t| t.ty != TokenType::WhiteSpace);

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
                    if old_token.ty == new_token.ty {
                        let old_token_str = &old_content[old_token.span.range_bytes()];
                        let new_token_str = &new_content[new_token.span.range_bytes()];
                        if old_token_str == new_token_str {
                            // success
                            new_tokens_iter.next().unwrap();
                            break;
                        } else {
                            let reason = format!(
                                "formatting output token content mismatch for token type `{}`, got {new_token_str}",
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
                        let diag = Diagnostic::new_internal_error(format!(
                            "formatting output token type mismatch, got `{}`",
                            new_token.ty.diagnostic_string()
                        ))
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
