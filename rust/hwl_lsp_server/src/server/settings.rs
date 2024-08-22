use crate::server::language::semantic_token_legend;
use lsp_types::{InitializeParams, PositionEncodingKind, SemanticTokensFullOptions, SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, WorkDoneProgressOptions};

pub struct Settings {
    pub initialize_params: InitializeParams,
    pub server_capabilities: ServerCapabilities,
}

impl Settings {
    pub fn new(initialize_params: InitializeParams) -> Self {
        const NO_WORK_DONE: WorkDoneProgressOptions = WorkDoneProgressOptions { work_done_progress: None };

        // TODO for all of these, first check that the server supports them

        let server_capabilities = ServerCapabilities {
            // TODO use utf-8 if the client supports it, should be faster on both sides
            position_encoding: Some(PositionEncodingKind::UTF16),
            // TODO support incremental (and maybe even will_save-type notifications)
            text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
            semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                work_done_progress_options: NO_WORK_DONE,
                legend: semantic_token_legend(),
                // TODO support ranges
                range: None,
                // TODO support delta
                full: Some(SemanticTokensFullOptions::Bool(true)),
            })),
            ..Default::default()
        };

        Self { initialize_params, server_capabilities }
    }

    pub fn server_capabilities(&self) -> &ServerCapabilities {
        &self.server_capabilities
    }
}