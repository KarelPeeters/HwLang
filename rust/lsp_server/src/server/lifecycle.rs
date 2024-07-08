use tower_lsp::jsonrpc;
use tower_lsp::lsp_types::{CompletionOptions, DefinitionOptions, InitializedParams, InitializeParams, InitializeResult, MessageType, OneOf, ServerCapabilities, ServerInfo, TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions, WorkDoneProgressOptions};

use crate::server::core::ServerCore;

impl ServerCore {
    pub async fn initialize(&self, _: InitializeParams) -> jsonrpc::Result<InitializeResult> {
        // TODO see if there's anything useful in params

        let server_info = ServerInfo {
            name: "hwlang_lsp".to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        };
        Ok(InitializeResult {
            capabilities: CAPABILITIES,
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

const CAPABILITIES: ServerCapabilities = ServerCapabilities {
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
        work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
        completion_item: None,
    }),
    signature_help_provider: None,
    definition_provider: Some(OneOf::Right(DefinitionOptions {
        work_done_progress_options: WorkDoneProgressOptions { work_done_progress: None },
    })),
    type_definition_provider: None,
    implementation_provider: None,
    references_provider: None,
    document_highlight_provider: None,
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
    semantic_tokens_provider: None,
    moniker_provider: None,
    linked_editing_range_provider: None,
    inline_value_provider: None,
    inlay_hint_provider: None,
    diagnostic_provider: None,
    experimental: None,
};
