use log::{error, warn};
use tower_lsp::jsonrpc;
use tower_lsp::jsonrpc::Error;
use tower_lsp::lsp_types::{DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, MessageType, TextEdit, WillSaveTextDocumentParams};

use crate::server::core::ServerCore;

impl ServerCore {
    pub async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, format!("did_open({:?})", params))
            .await;
    }

    // TODO register for https://microsoft.github.io/language-server-protocol/specifications/specification-3-15/#workspace_didChangeWatchedFiles
    pub async fn did_change(&self, params: DidChangeTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, format!("did_change({:?})", params))
            .await;
    }

    pub async fn will_save(&self, _params: WillSaveTextDocumentParams) {
        warn!("Got a textDocument/willSave notification, but it is not implemented");
    }

    pub async fn will_save_wait_until(&self, _params: WillSaveTextDocumentParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> {
        error!("Got a textDocument/willSaveWaitUntil request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn did_save(&self, _params: DidSaveTextDocumentParams) {
        warn!("Got a textDocument/didSave notification, but it is not implemented");
    }

    pub async fn did_close(&self, _params: DidCloseTextDocumentParams) {
        warn!("Got a textDocument/didClose notification, but it is not implemented");
    }
}