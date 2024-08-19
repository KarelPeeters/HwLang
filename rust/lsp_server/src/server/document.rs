use log::{error, warn};
use tower_lsp::jsonrpc;
use tower_lsp::jsonrpc::Error;
use tower_lsp::lsp_types::{
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, TextEdit,
    VersionedTextDocumentIdentifier, WillSaveTextDocumentParams,
};

use crate::server::core::ServerCore;

// TODO register for https://microsoft.github.io/language-server-protocol/specifications/specification-3-15/#workspace_didChangeWatchedFiles?
//    so we don't have to implement file watching ourself
// TODO should we care about versions?
impl ServerCore {
    pub async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem {
            uri,
            language_id,
            version: _,
            text,
        } = text_document;

        self.log_info(format!("Opening {:?} with language {:?}", uri, language_id))
            .await;

        let prev = self.state.lock().await.documents.insert(uri.clone(), text);
        if prev.is_some() {
            self.log_error(format!("did_open({:?}) which is already in database", uri))
                .await;
        }
    }

    pub async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let DidChangeTextDocumentParams {
            text_document,
            content_changes,
        } = params;
        let VersionedTextDocumentIdentifier { ref uri, version: _ } = text_document;

        let mut state = self.state.lock().await;

        let entry = match state.documents.get_mut(&text_document.uri) {
            Some(content) => content,
            None => {
                drop(state);
                self.log_error(format!("did_change({:?}) which is not in the database", uri))
                    .await;
                return;
            }
        };

        for mut change in content_changes {
            let TextDocumentContentChangeEvent {
                range,
                range_length,
                text,
            } = change;
            if range.is_some() || range_length.is_some() {
                change.text = text;
                self.log_error(format!("did_change({:?}) with range not supported yet", change))
                    .await;
                return;
            }
            *entry = text;
        }
    }

    pub async fn will_save(&self, _params: WillSaveTextDocumentParams) {
        warn!("Got a textDocument/willSave notification, but it is not implemented");
    }

    pub async fn will_save_wait_until(
        &self,
        _params: WillSaveTextDocumentParams,
    ) -> jsonrpc::Result<Option<Vec<TextEdit>>> {
        error!("Got a textDocument/willSaveWaitUntil request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn did_save(&self, _params: DidSaveTextDocumentParams) {
        warn!("Got a textDocument/didSave notification, but it is not implemented");
    }

    pub async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        let prev = self.state.lock().await.documents.remove(&uri);
        if prev.is_none() {
            self.log_error(format!("did_close({:?}) which was not in database", uri))
                .await;
        }
    }
}
