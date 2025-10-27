use crate::handlers::dispatch::RequestHandler;
use crate::server::state::{RequestResult, ServerState};
use crate::util::encode::{lsp_to_pos, span_to_lsp};
use crate::util::uri::uri_to_path;
use hwl_language::syntax::parse_file_content;
use hwl_language::syntax::resolve::{FindDefinition, find_definition};
use hwl_language::syntax::source::SourceDatabase;
use itertools::Itertools;
use lsp_types::request::GotoDefinition;
use lsp_types::{
    GotoDefinitionParams, GotoDefinitionResponse, Location, TextDocumentIdentifier, TextDocumentPositionParams,
};

impl RequestHandler<GotoDefinition> for ServerState {
    fn handle_request(&mut self, params: GotoDefinitionParams) -> RequestResult<Option<GotoDefinitionResponse>> {
        let GotoDefinitionParams {
            text_document_position_params,
            work_done_progress_params: _,
            partial_result_params: _,
        } = params;
        let TextDocumentPositionParams {
            text_document,
            position,
        } = text_document_position_params;
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
            Err(_) => return Ok(Some(GotoDefinitionResponse::Array(vec![]))),
        };

        // find declarations
        let pos = lsp_to_pos(self.settings.position_encoding, offsets, src, file, position);
        let result = match find_definition(&source, &ast, pos) {
            FindDefinition::Found(spans) => {
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
                Some(GotoDefinitionResponse::Array(spans_lsp))
            }
            FindDefinition::PosNotOnIdentifier => None,
        };

        Ok(result)
    }
}
