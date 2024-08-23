use crate::server::state::{NotificationHandler, ServerState};
use hwl_language::syntax::pos::FileLineOffsets;
use indexmap::IndexMap;
use lsp_types::notification::{DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument};
use lsp_types::{DidChangeTextDocumentParams, DidChangeWatchedFilesParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, FileChangeType, FileEvent, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Uri, VersionedTextDocumentIdentifier};

// Name and the general principle from the VFS of rust-analyzer.
pub struct VirtualFileSystem {
    map: IndexMap<Uri, VirtualFileContent>,
}

pub struct VirtualFileContent {
    // TODO support binary content?
    text: String,
    cached_line_offsets: Option<FileLineOffsets>,
}

pub struct VirtualFileContentInit<'i> {
    pub text: &'i str,
    pub offsets: &'i FileLineOffsets,
}

impl VirtualFileSystem {
    pub fn new() -> Self {
        Self { map: IndexMap::default() }
    }

    // TODO are there ordering guarantees between create and open? can we add some extra assertions here?
    pub fn set_text_maybe_create(&mut self, uri: &Uri, text: String) {
        // TODO check if text is the same before invaliding?
        self.map.insert(uri.clone(), VirtualFileContent {
            text,
            cached_line_offsets: None,
        });
    }

    pub fn delete(&mut self, uri: &Uri) {
        assert!(self.map.swap_remove(uri).is_some());
    }

    pub fn get_full(&mut self, uri: &Uri) -> Option<VirtualFileContentInit> {
        let content = self.map.get_mut(uri)?;
        let line_offsets = content.cached_line_offsets.get_or_insert_with(|| FileLineOffsets::new(&content.text));
        Some(VirtualFileContentInit {
            text: &content.text,
            offsets: line_offsets,
        })
    }
}

impl VirtualFileContent {
    // pub fn new(text: String) -> Self {
    //     Self {
    //         content: text,
    //         is
    //         cached_line_offsets: None,
    //     }
    // }

    pub fn set_text(&mut self, text: String) {
        self.text = text;
        self.cached_line_offsets = None;
    }

    pub fn get_full(&mut self) -> VirtualFileContentInit {
        let offsets = self.cached_line_offsets.get_or_insert_with(|| FileLineOffsets::new(&self.text));
        VirtualFileContentInit {
            text: &self.text,
            offsets,
        }
    }
}

impl NotificationHandler<DidOpenTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidOpenTextDocumentParams) {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem { uri, text, language_id: _, version: _ } = text_document;

        assert!(self.open_files.insert(uri.clone()));
        self.virtual_file_system.set_text_maybe_create(&uri, text);
    }
}

impl NotificationHandler<DidCloseTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidCloseTextDocumentParams) {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        assert!(self.open_files.remove(&uri));
    }
}

impl NotificationHandler<DidChangeTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidChangeTextDocumentParams) {
        let DidChangeTextDocumentParams { text_document, content_changes } = params;
        let VersionedTextDocumentIdentifier { uri, version: _ } = text_document;

        for change in content_changes {
            let TextDocumentContentChangeEvent { range, range_length, text } = change;
            assert!(range.is_none() && range_length.is_none());
            self.virtual_file_system.set_text_maybe_create(&uri, text);
        }
    }
}

impl NotificationHandler<DidChangeWatchedFiles> for ServerState {
    fn handle_notification(&mut self, params: DidChangeWatchedFilesParams) {
        let DidChangeWatchedFilesParams { changes } = params;
        for change in changes {
            let FileEvent { uri, typ } = change;
            match typ {
                // TODO assert that file does not yet exist for created
                FileChangeType::CREATED | FileChangeType::CHANGED => {
                    // let text = std::fs::read_to_string(uri.path())
                    eprintln!("file at {:?} changed, we have to read it", uri);
                },
                FileChangeType::DELETED => {
                    self.virtual_file_system.delete(&uri);
                }
                _ => panic!("unknown file change type {typ:?} for uri {uri:?}")
            };
        }
    }
}