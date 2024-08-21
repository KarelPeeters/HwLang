use crate::server::state::{NotificationHandler, ServerState};
use lsp_types::notification::{DidCloseTextDocument, DidOpenTextDocument};
use lsp_types::{DidCloseTextDocumentParams, DidOpenTextDocumentParams, TextDocumentIdentifier, TextDocumentItem};

impl NotificationHandler<DidOpenTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidOpenTextDocumentParams) {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem { uri, text, language_id: _, version: _ } = text_document;

        let prev = self.open_files.insert(uri, text);
        // TODO proper error handling for this?
        assert!(prev.is_none());
    }
}

impl NotificationHandler<DidCloseTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidCloseTextDocumentParams) {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        let prev = self.open_files.swap_remove(&uri);
        // TODO proper error handling for this?
        assert!(prev.is_some());
    }
}