use crate::handlers::dispatch::RequestHandler;
use crate::server::state::{RequestResult, ServerState};
use crate::support::PosNotOnIdentifier;
use crate::support::find_usages::find_usages;
use crate::util::encode::{lsp_to_pos, span_to_lsp};
use crate::util::uri::uri_to_path;
use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::source::SourceDatabase;
use itertools::Itertools;
use lsp_types::request::References;
use lsp_types::{Location, ReferenceContext, ReferenceParams, TextDocumentIdentifier, TextDocumentPositionParams};

impl RequestHandler<References> for ServerState {
    fn handle_request(&mut self, params: ReferenceParams) -> RequestResult<Option<Vec<Location>>> {
        let ReferenceParams {
            text_document_position,
            work_done_progress_params: _,
            partial_result_params: _,
            context,
        } = params;
        let TextDocumentPositionParams {
            text_document,
            position,
        } = text_document_position;
        let TextDocumentIdentifier { uri } = text_document;

        // TODO what does this mean?
        let ReferenceContext { include_declaration: _ } = context;

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

        // find usages
        let pos = lsp_to_pos(self.settings.position_encoding, offsets, src, file, position);
        let result = match find_usages(&source, &ast, pos) {
            Ok(spans) => {
                let spans_lsp = spans
                    .into_iter()
                    .map(|span| {
                        let span = span_to_lsp(self.settings.position_encoding, offsets, src, span);
                        Location {
                            uri: uri.clone(),
                            range: span,
                        }
                    })
                    .collect_vec();
                Some(spans_lsp)
            }
            Err(PosNotOnIdentifier) => None,
        };

        Ok(result)
    }
}
