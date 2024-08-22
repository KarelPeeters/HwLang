use crate::server::state::{NotificationHandler, ServerState};
use hwl_language::syntax::pos::FileLineOffsets;
use lsp_types::notification::{DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument};
use lsp_types::{DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, VersionedTextDocumentIdentifier};
use std::collections::HashSet;

pub struct OpenFileInfo {
    text: String,
    line_offsets: Option<FileLineOffsets>,
    lines_0_containing_non_ascii: Option<HashSet<u32>>,
}

pub struct OpenFileInfoFull<'i> {
    pub text: &'i str,
    pub offsets: &'i FileLineOffsets,
}

impl OpenFileInfo {
    pub fn new(text: String) -> Self {
        Self {
            text,
            line_offsets: None,
            lines_0_containing_non_ascii: None,
        }
    }

    pub fn set_text(&mut self, text: String) {
        self.text = text;
        self.line_offsets = None;
        self.lines_0_containing_non_ascii = None;
    }
    
    pub fn get_full(&mut self) -> OpenFileInfoFull {
        let offsets = self.line_offsets.get_or_insert_with(|| FileLineOffsets::new(&self.text));
        OpenFileInfoFull {
            text: &self.text,
            offsets,
        }
    }
}

impl NotificationHandler<DidOpenTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidOpenTextDocumentParams) {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem { uri, text, language_id: _, version: _ } = text_document;

        let prev = self.open_files.insert(uri, OpenFileInfo::new(text));
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

impl NotificationHandler<DidChangeTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidChangeTextDocumentParams) {
        let DidChangeTextDocumentParams { text_document, content_changes } = params;
        let VersionedTextDocumentIdentifier { uri, version: _ } = text_document;

        let file = self.open_files.get_mut(&uri).unwrap();
        for change in content_changes {
            let TextDocumentContentChangeEvent { range, range_length, text } = change;
            assert!(range.is_none() && range_length.is_none());
            file.set_text(text);
        }
    }
}
