use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::flatten::ast_to_node;
use crate::syntax::format_new::high::lower_nodes;
use crate::syntax::format_new::low::node_to_string;
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::tokenize;
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
    source: &SourceDatabase,
    file: FileId,
    settings: &FormatSettings,
) -> DiagResult<String> {
    // tokenize and parse
    let source_info = &source[file];
    let source_str = &source_info.content;
    let source_offsets = &source_info.offsets;
    let mut source_tokens = tokenize(file, source_str, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    // TODO remove whitespace tokens, everyone just filters them out anyway, we can always recover whitespace as "stuff between other tokens" if we really want to
    source_tokens.retain(|t| t.ty != crate::syntax::token::TokenType::WhiteSpace);
    let source_ast = parse_file_content(file, source_str).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;

    println!("Source tokens:");
    for token in &source_tokens {
        println!("    {token:?}");
    }

    let root_node = ast_to_node(&source_ast);

    println!("Initial tree:");
    println!("{}", root_node.tree_string());

    let root_node = lower_nodes(source_offsets, &source_tokens, root_node).map_err(|e| {
        let (span, got) = match e.index {
            None => (Span::empty_at(source.full_span(file).end()), "end of file"),
            Some(index) => {
                let source_token = &source_tokens[index.0];
                (source_token.span, source_token.ty.diagnostic_string())
            }
        };
        let reason = format!(
            "token mismatch, expected `{}` got `{got}`",
            e.expected.diagnostic_string()
        );
        diags.report_internal_error(span, reason)
    })?;

    println!("Mapped tree:");
    println!("{}", root_node.debug_str());

    let result = node_to_string(settings, source_str, source_offsets, &source_tokens, &root_node);

    Ok(result)
}
