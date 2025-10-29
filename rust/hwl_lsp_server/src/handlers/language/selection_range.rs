use crate::handlers::dispatch::RequestHandler;
use crate::server::state::{RequestResult, ServerState};
use crate::util::encode::{lsp_to_pos, span_to_lsp};
use crate::util::uri::uri_to_path;
use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::pos::{Pos, Span};
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{FoldRangeKind, SyntaxVisitor, syntax_visit};
use hwl_language::util::{Never, ResultNeverExt};
use itertools::Itertools;
use lsp_types::request::SelectionRangeRequest;
use lsp_types::{SelectionRange, SelectionRangeParams, TextDocumentIdentifier};

impl RequestHandler<SelectionRangeRequest> for ServerState {
    fn handle_request(&mut self, params: SelectionRangeParams) -> RequestResult<Option<Vec<SelectionRange>>> {
        let SelectionRangeParams {
            text_document,
            positions,
            work_done_progress_params: _,
            partial_result_params: _,
        } = params;
        let TextDocumentIdentifier { uri } = text_document;

        // TODO integrate all of this better with the main compiler, eg. cache the parsed ast, reason about imports, ...
        let path = uri_to_path(&uri)?;
        let src = self.vfs.read_str_maybe_from_disk(&path)?;

        let mut source = SourceDatabase::new();
        let file = source.add_file("dummy".to_owned(), src.to_owned());
        let offsets = &source[file].offsets;

        // parse source to ast
        let ast = match parse_file_content(file, src) {
            Ok(ast) => ast,
            Err(_) => return Ok(None),
        };

        // collect selection ranges
        let result_ranges = positions
            .into_iter()
            .map(|pos_lsp| {
                let pos = lsp_to_pos(self.settings.position_encoding, offsets, src, file, pos_lsp);
                CollectedSelectionRanges {
                    pos_lsp,
                    pos,
                    ranges: vec![],
                }
            })
            .collect_vec();
        let mut visitor = SelectionRangeVisitor { result_ranges };
        syntax_visit(&source, &ast, &mut visitor).remove_never();

        // convert to lsp types
        let result = visitor
            .result_ranges
            .into_iter()
            .map(|collected_range| {
                // TODO check that parents always contain the child
                let mut ranges = collected_range.ranges;
                ranges.sort_by_key(|span| span.len_bytes());

                let mut curr = None;
                for span in ranges.into_iter().rev() {
                    curr = Some(SelectionRange {
                        range: span_to_lsp(self.settings.position_encoding, offsets, src, span),
                        parent: curr.map(Box::new),
                    });
                }

                // default to empty range at position, following the spec
                curr.unwrap_or_else(|| SelectionRange {
                    range: lsp_types::Range::new(collected_range.pos_lsp, collected_range.pos_lsp),
                    parent: None,
                })
            })
            .collect_vec();

        Ok(Some(result))
    }
}

struct CollectedSelectionRanges {
    pos_lsp: lsp_types::Position,
    pos: Pos,
    ranges: Vec<Span>,
}

struct SelectionRangeVisitor {
    result_ranges: Vec<CollectedSelectionRanges>,
}

impl SyntaxVisitor for SelectionRangeVisitor {
    type Break = Never;
    const SCOPE_DECLARE: bool = false;

    fn should_visit_span(&self, span: Span) -> bool {
        self.result_ranges.iter().any(|r| span.touches_pos(r.pos))
    }

    fn report_range(&mut self, range: Span, _: Option<FoldRangeKind>) {
        for r in &mut self.result_ranges {
            if range.touches_pos(r.pos) {
                r.ranges.push(range);
            }
        }
    }
}
