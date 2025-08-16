use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::core::simplify_and_connect_nodes;
use crate::syntax::format_new::flatten::ast_to_format_tree;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::tokenize;
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};

mod core;
mod flatten;

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

    let root_node = ast_to_format_tree(&source_ast);

    println!("Initial tree:");
    println!("{}", root_node.tree_string());

    let root_node = simplify_and_connect_nodes(source_offsets, &source_tokens, root_node).map_err(|_| todo!())?;

    println!("Mapped tree:");
    println!("{}", root_node.tree_string());

    todo!()

    // println!("Tree tokens:");
    // node_root
    //     .try_for_each_token(&mut |node_token_ty, node_token_fixed| {
    //         println!("    {node_token_ty:?} (fixed: {node_token_fixed})");
    //         Ok::<(), Never>(())
    //     })
    //     .remove_never();

    // cross-reference tokens and figure out which nodes contain newlines
    // let source_end_pos = source.full_span(file).end();
    // let fixed_token_map = crate::syntax::format_new::core::match_tokens(diags, source_end_pos, &source_tokens, &node_root)?;
    //
    // // convert to output string
    // let mut result_ctx = crate::syntax::format_new::core::StringBuilderContext {
    //     source_str,
    //     source_tokens: &source_tokens,
    //     node_token_to_source_token: &fixed_token_map,
    //
    //     settings,
    //
    //     result: String::with_capacity(source_str.len() * 2),
    //     state: crate::syntax::format_new::core::StringState {
    //         next_node_token_index: 0,
    //         curr_line_index: 0,
    //         curr_line_start: 0,
    //         indent: 0,
    //     },
    // };
    // result_ctx.write_node(&node_root, true).expect(crate::syntax::format_new::core::EXPECT_WRAP);
    //
    // if result_ctx.state.next_node_token_index != fixed_token_map.len() {
    //     return Err(todo!("err"));
    // }
    //
    // Ok(result_ctx.result)
}
