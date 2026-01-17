use crate::handlers::dispatch::RequestHandler;
use crate::server::state::{RequestResult, ServerState};
use crate::util::encode::span_to_lsp;
use crate::util::uri::uri_to_path;
use hwl_language::syntax::pos::Span;
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::visitor::{FoldRangeKind, SyntaxVisitor, syntax_visit};
use hwl_language::syntax::{FileContentRecovery, parse_file_content_with_recovery};
use hwl_language::util::{Never, ResultNeverExt};
use itertools::Itertools;
use lsp_types::request::FoldingRangeRequest;
use lsp_types::{FoldingRange, FoldingRangeParams, TextDocumentIdentifier};

impl RequestHandler<FoldingRangeRequest> for ServerState {
    fn handle_request(&mut self, params: FoldingRangeParams) -> RequestResult<Option<Vec<FoldingRange>>> {
        let FoldingRangeParams {
            text_document,
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
        let FileContentRecovery {
            recovered_content,
            errors: _,
        } = match parse_file_content_with_recovery(file, src) {
            Ok(ast) => ast,
            Err(_) => return Ok(None),
        };

        // collect folding ranges
        let mut visitor = FoldingRangeVisitor { result_ranges: vec![] };
        syntax_visit(&source, &recovered_content, &mut visitor).remove_never();

        let result = visitor
            .result_ranges
            .into_iter()
            .map(|(span, kind)| {
                let span_lsp = span_to_lsp(self.settings.position_encoding, offsets, src, span);
                let kind = match kind {
                    FoldRangeKind::Comment => lsp_types::FoldingRangeKind::Comment,
                    FoldRangeKind::Imports => lsp_types::FoldingRangeKind::Imports,
                    FoldRangeKind::Region => lsp_types::FoldingRangeKind::Region,
                };

                // TODO think about populating start/end characters and collapsed text
                FoldingRange {
                    start_line: span_lsp.start.line,
                    start_character: None,
                    end_line: span_lsp.end.line,
                    end_character: None,
                    kind: Some(kind),
                    collapsed_text: None,
                }
            })
            .collect_vec();
        Ok(Some(result))
    }
}

struct FoldingRangeVisitor {
    result_ranges: Vec<(Span, FoldRangeKind)>,
}

impl SyntaxVisitor for FoldingRangeVisitor {
    type Break = Never;
    const SCOPE_DECLARE: bool = false;

    fn should_visit_span(&self, _: Span) -> bool {
        true
    }

    fn report_range(&mut self, range: Span, fold: Option<FoldRangeKind>) {
        if let Some(fold) = fold {
            self.result_ranges.push((range, fold));
        }
    }
}
