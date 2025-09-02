use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::syntax::ast::FileContent;
use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::flatten::ast_to_node;
use crate::syntax::format_new::high::lower_nodes;
use crate::syntax::format_new::low::node_to_string;
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{Token, TokenType, tokenize};
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};

mod common;
mod flatten;
mod high;
mod low;

// Sketch of the formatter implementation:
// * convert the AST to a tree of FNodes, which are just tokens with some extra structure
// * cross-reference emitted formatting tokens with the original tokens to get span info
// * bottom-up traverse the format tree to figure out which branches need to wrap due to line comments
// * top-down traverse the format tree, printing the tokens to the final buffer
//    * if any line overflows and we can still make additional wrapping choices, roll back and try
//    * if blank lines between items: insert matching blank line
// TODO add debug mode that slowly decreases the line width and saves every time the output changes
pub fn format(
    diags: &Diagnostics,
    source: &mut SourceDatabase,
    settings: &FormatSettings,
    file: FileId,
) -> DiagResult<String> {
    let result_first = format_single(diags, source, settings, file)?;

    // check that output parses to the same tokens and ast
    let new_file = source.add_file(
        format!("{} (after formatting)", source[file].debug_info_path),
        result_first.new_string.clone(),
    );
    check_format_output_matches(
        diags,
        source,
        file,
        &result_first.old_tokens,
        &result_first.old_ast,
        new_file,
    )?;

    Ok(result_first.new_string)
}

struct FormatResult {
    old_tokens: Vec<Token>,
    old_ast: FileContent,
    new_string: String,
}

fn format_single(
    diags: &Diagnostics,
    source: &SourceDatabase,
    settings: &FormatSettings,
    file: FileId,
) -> DiagResult<FormatResult> {
    // tokenize and parse
    let old_info = &source[file];
    let old_string = &old_info.content;
    let old_offsets = &old_info.offsets;
    let mut old_tokens = tokenize(file, old_string, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    // TODO remove whitespace tokens, everyone just filters them out anyway, we can always recover whitespace as "stuff between other tokens" if we really want to
    old_tokens.retain(|t| t.ty != TokenType::WhiteSpace);
    let old_ast = parse_file_content(file, old_string).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;

    let root_node = ast_to_node(&old_ast);

    println!("HNode tree:");
    println!("{}", root_node.tree_string());

    let root_node = lower_nodes(old_string, old_offsets, &old_tokens, root_node).map_err(|e| {
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

    // println!("LNode tree:");
    // println!("{}", root_node.debug_str());

    println!("LNode tree simplified:");
    // TODO fuzz test whether this ever changes anything
    let root_node = root_node.simplify();
    println!("{}", root_node.debug_str());

    let new_string = node_to_string(settings, old_string, &old_tokens, &root_node);
    Ok(FormatResult {
        old_tokens,
        old_ast,
        new_string,
    })
}

fn check_format_output_matches(
    diags: &Diagnostics,
    source: &SourceDatabase,
    old_file: FileId,
    old_tokens: &[Token],
    old_ast: &FileContent,
    new_file: FileId,
) -> DiagResult {
    // check that tokens match approximately (commas might have changed)
    let old_content = &source[old_file].content;
    let new_content = &source[new_file].content;
    let mut new_tokens = tokenize(new_file, new_content, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    new_tokens.retain(|t| t.ty != TokenType::WhiteSpace);

    let mut new_tokens_iter = new_tokens.iter().peekable();
    for old_token in old_tokens {
        loop {
            let new_token = new_tokens_iter.peek();
            match new_token {
                None => {
                    let diag = Diagnostic::new_internal_error("formatting missing output token")
                        .add_error(
                            old_token.span,
                            format!("expected `{}`", old_token.ty.diagnostic_string()),
                        )
                        .add_error(Span::empty_at(source.full_span(new_file).end()), "reached end of file")
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
                                "formatting output token content mismatch for token type `{}`",
                                old_token.ty.diagnostic_string()
                            );
                            let diag = Diagnostic::new_internal_error(reason)
                                .add_error(old_token.span, format!("expected `{old_token_str}`"))
                                .add_error(new_token.span, format!("got `{new_token_str}`"))
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
                        let diag = Diagnostic::new_internal_error("formatting output token type mismatch")
                            .add_error(
                                old_token.span,
                                format!("expected `{}`", old_token.ty.diagnostic_string()),
                            )
                            .add_error(new_token.span, format!("got `{}`", new_token.ty.diagnostic_string()))
                            .finish();
                        return Err(diags.report(diag));
                    }
                }
            }
        }
    }

    // check that parsing still works and yields at least plausible results
    let new_ast = parse_file_content(new_file, new_content).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;
    if new_ast.arena_expressions.len() != old_ast.arena_expressions.len() {
        return Err(diags.report_internal_error(
            source.full_span(new_file),
            format!(
                "number of expressions changed during formatting, before: {}, after: {}",
                old_ast.arena_expressions.len(),
                new_ast.arena_expressions.len()
            ),
        ));
    }

    Ok(())
}
