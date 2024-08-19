use tower_lsp::jsonrpc;
use tower_lsp::lsp_types::{
    CompletionOptions, DefinitionOptions, DocumentHighlightOptions, InitializeParams, InitializeResult,
    InitializedParams, MessageType, OneOf, SemanticTokensFullOptions, SemanticTokensLegend,
    SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities, ServerInfo,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions, WorkDoneProgressOptions,
};

use crate::server::core::ServerCore;
use crate::server::language::semantic_token_legend;

impl ServerCore {
    pub async fn initialize(&self, params: InitializeParams) -> jsonrpc::Result<InitializeResult> {
        // TODO see if there's anything useful in params

        let server_info = ServerInfo {
            name: "hwlang_lsp".to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        };

        self.log_info(format!("client params: {:#?}", params)).await;

        Ok(InitializeResult {
            capabilities: capabilities(),
            server_info: Some(server_info),
        })
    }

    pub async fn initialized(&self, params: InitializedParams) {
        // assert that there are no params
        let InitializedParams {} = params;

        self.client
            .log_message(MessageType::INFO, "HWLang LSP server initialized!")
            .await;
    }

    pub async fn shutdown(&self) -> jsonrpc::Result<()> {
        Ok(())
    }
}

const NO_PROGRESS: WorkDoneProgressOptions = WorkDoneProgressOptions {
    work_done_progress: None,
};

fn capabilities() -> ServerCapabilities {
    // TODO spread this out through the different wrapper implementations as well
    ServerCapabilities {
        position_encoding: None,
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            // TODO support incremental file changes
            change: Some(TextDocumentSyncKind::FULL),
            will_save: Some(false),
            will_save_wait_until: Some(false),
            save: None,
        })),
        selection_range_provider: None,
        hover_provider: None,
        completion_provider: Some(CompletionOptions {
            resolve_provider: Some(true),
            trigger_characters: None,
            all_commit_characters: None,
            work_done_progress_options: NO_PROGRESS,
            completion_item: None,
        }),
        signature_help_provider: None,
        definition_provider: Some(OneOf::Right(DefinitionOptions {
            work_done_progress_options: NO_PROGRESS,
        })),
        type_definition_provider: None,
        implementation_provider: None,
        references_provider: None,
        document_highlight_provider: Some(OneOf::Right(DocumentHighlightOptions {
            work_done_progress_options: NO_PROGRESS,
        })),
        document_symbol_provider: None,
        workspace_symbol_provider: None,
        code_action_provider: None,
        code_lens_provider: None,
        document_formatting_provider: None,
        document_range_formatting_provider: None,
        document_on_type_formatting_provider: None,
        rename_provider: None,
        document_link_provider: None,
        color_provider: None,
        folding_range_provider: None,
        declaration_provider: None,
        execute_command_provider: None,
        workspace: None,
        call_hierarchy_provider: None,
        semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
            SemanticTokensOptions {
                work_done_progress_options: NO_PROGRESS,
                legend: SemanticTokensLegend {
                    token_types: semantic_token_legend(),
                    token_modifiers: vec![],
                },
                range: Some(false),
                full: Some(SemanticTokensFullOptions::Delta { delta: Some(false) }),
            },
        )),
        moniker_provider: None,
        linked_editing_range_provider: None,
        inline_value_provider: None,
        inlay_hint_provider: None,
        diagnostic_provider: None,
        experimental: None,
    }
}

