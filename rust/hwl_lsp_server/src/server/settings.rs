use crate::server::language::semantic_token_legend;
use lsp_types::{InitializeParams, PositionEncodingKind, SemanticTokensFullOptions, SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind, WorkDoneProgressOptions};

pub struct Settings {
    pub initialize_params: InitializeParams,
    pub server_capabilities: ServerCapabilities,

    pub position_encoding: PositionEncoding,
}

#[derive(Debug, Copy, Clone)]
pub enum PositionEncoding {
    Utf8,
    Utf16,
}

const NO_WORK_DONE: WorkDoneProgressOptions = WorkDoneProgressOptions { work_done_progress: None };

impl Settings {
    pub fn new(initialize_params: InitializeParams) -> Self {
        // default to utf-16 (as required by the spec), but prefer utf-8 if the client supports it
        let mut position_encoding = PositionEncoding::Utf16;
        if let Some(general) = &initialize_params.capabilities.general {
            if let Some(position_encodings) = &general.position_encodings {
                if position_encodings.contains(&PositionEncodingKind::UTF8) {
                    position_encoding = PositionEncoding::Utf8;
                }
            }
        }

        // TODO for all of these, first check that the server supports them
        let server_capabilities = ServerCapabilities {
            position_encoding: Some(position_encoding.to_lsp()),
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

        Self { initialize_params, server_capabilities, position_encoding }
    }

    pub fn server_capabilities(&self) -> &ServerCapabilities {
        &self.server_capabilities
    }
}

impl PositionEncoding {
    pub fn to_lsp(self) -> PositionEncodingKind {
        match self {
            PositionEncoding::Utf8 => PositionEncodingKind::UTF8,
            PositionEncoding::Utf16 => PositionEncodingKind::UTF16,
        }
    }
}
