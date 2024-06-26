use tower_lsp::{Client, LanguageServer, LspService, Server};
use tower_lsp::jsonrpc::{Error, Result};
use tower_lsp::lsp_types::*;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend { client });
    Server::new(stdin, stdout, socket).serve(service).await;
}

#[derive(Debug)]
struct Backend {
    client: Client,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
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

    async fn initialized(&self, params: InitializedParams) {
        // assert that there are no params
        let InitializedParams {} = params;

        self.client
            .log_message(MessageType::INFO, "HWLang LSP server initialized!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, format!("did_open({:?})", params))
            .await;
    }

    // TODO register for https://microsoft.github.io/language-server-protocol/specifications/specification-3-15/#workspace_didChangeWatchedFiles
    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, format!("did_change({:?})", params))
            .await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let _ = params;
        self.client
            .log_message(MessageType::INFO, format!("completion({:?})", params))
            .await;

        // TODO actually populate, and check if there are more useful fields to populate
        let items = vec![
            CompletionItem {
                label: "custom_completion_text".to_string(),
                label_details: None,
                kind: Some(CompletionItemKind::TEXT),
                ..CompletionItem::default()
            },
            CompletionItem {
                label: "custom_completion_text_method".to_string(),
                label_details: None,
                kind: Some(CompletionItemKind::METHOD),
                ..CompletionItem::default()
            }
        ];

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn completion_resolve(&self, params: CompletionItem) -> Result<CompletionItem> {
        // TODO add in extra information
        Ok(params)
    }
}

const CAPABILITIES: ServerCapabilities = ServerCapabilities {
    position_encoding: None,
    text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
        open_close: Some(true),
        change: Some(TextDocumentSyncKind::INCREMENTAL),
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
    definition_provider: None,
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
