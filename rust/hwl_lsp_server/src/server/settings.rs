use crate::server::language::semantic_token_legend;
use lsp_types::{
    InitializeParams, OneOf, PositionEncodingKind, SemanticTokensFullOptions, SemanticTokensOptions,
    SemanticTokensServerCapabilities, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    WorkDoneProgressOptions,
};

pub struct Settings {
    pub initialize_params: InitializeParams,
    pub server_capabilities: ServerCapabilities,

    pub position_encoding: PositionEncoding,
    pub supports_multi_line_semantic_tokens: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum PositionEncoding {
    Utf8,
    Utf16,
}

const NO_WORK_DONE: WorkDoneProgressOptions = WorkDoneProgressOptions {
    work_done_progress: None,
};

#[derive(Debug)]
pub struct SettingsError(pub String);

impl Settings {
    pub fn new(initialize_params: InitializeParams) -> Result<Self, SettingsError> {
        // figure out position encoding
        //   default to utf-16 (as required by the spec), but prefer utf-8 if the client supports it
        let mut position_encoding = PositionEncoding::Utf16;
        if let Some(general) = &initialize_params.capabilities.general {
            if let Some(position_encodings) = &general.position_encodings {
                if position_encodings.contains(&PositionEncodingKind::UTF8) {
                    position_encoding = PositionEncoding::Utf8;
                }
            }
        }

        // make sure the client can watch files for us
        let mut watch_dynamic = false;
        if let Some(workspace) = &initialize_params.capabilities.workspace {
            if let Some(did_change_watched_files) = workspace.did_change_watched_files {
                watch_dynamic = did_change_watched_files.dynamic_registration.unwrap_or(false);
            }
        }
        // TODO support watching file changes ourself? but then how does synchronization work?
        if !watch_dynamic {
            return Err(SettingsError(
                "this server requires dynamic watched files registration".to_string(),
            ));
        }

        // check if the client supports multi-line tokens
        let mut supports_multi_line_semantic_tokens = false;
        if let Some(text_document) = &initialize_params.capabilities.text_document {
            if let Some(semantic_tokens) = &text_document.semantic_tokens {
                supports_multi_line_semantic_tokens = semantic_tokens.multiline_token_support.unwrap_or(false);
            }
        }

        // TODO for all of these, first check that the client supports them?
        let server_capabilities = ServerCapabilities {
            position_encoding: Some(position_encoding.to_lsp()),
            // TODO support incremental (and maybe even will_save-type notifications)
            text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
            semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
                SemanticTokensOptions {
                    work_done_progress_options: NO_WORK_DONE,
                    legend: semantic_token_legend(),
                    // TODO support ranges
                    range: None,
                    // TODO support delta
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                },
            )),
            definition_provider: Some(OneOf::Left(true)),
            document_formatting_provider: Some(OneOf::Left(true)),
            ..Default::default()
        };

        Ok(Self {
            initialize_params,
            server_capabilities,
            position_encoding,
            supports_multi_line_semantic_tokens,
        })
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
