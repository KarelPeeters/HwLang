use crate::server::state::{RequestHandler, ServerState};
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::{SemanticTokensParams, SemanticTokensResult};

impl RequestHandler<SemanticTokensFullRequest> for ServerState {
    fn handle_request(&mut self, params: SemanticTokensParams) -> Result<Option<SemanticTokensResult>, String> {
        Ok(None)
    }
}
