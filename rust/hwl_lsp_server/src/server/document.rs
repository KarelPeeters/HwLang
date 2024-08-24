use crate::server::dispatch::NotificationHandler;
use crate::server::state::{NotificationError, ServerState};
use hwl_language::throw;
use indexmap::IndexMap;
use lsp_types::notification::{DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument};
use lsp_types::{DidChangeTextDocumentParams, DidChangeWatchedFilesParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, FileChangeType, FileEvent, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Uri, VersionedTextDocumentIdentifier};

/// The name and the general principle come from the VFS of rust-analyzer.
pub struct VirtualFileSystem {
    map: IndexMap<Uri, Content>,
}

pub enum Content {
    Unknown(Vec<u8>),
    Text(String),
}

#[derive(Debug)]
pub struct FileAlreadyExists;
#[derive(Debug)]
pub struct FileDoesNotExist;

impl VirtualFileSystem {
    pub fn new() -> Self {
        Self { map: IndexMap::default() }
    }

    pub fn create(&mut self, uri: &Uri, content: Content) -> Result<(), FileAlreadyExists> {
        let prev = self.map.insert(uri.clone(), content);
        match prev {
            None => Ok(()),
            Some(_) => Err(FileAlreadyExists),
        }
    }

    // TODO support incremental updates (and even incremental derived data updates?)
    pub fn update(&mut self, uri: &Uri, content: Content) -> Result<(), FileDoesNotExist> {
        let slot = self.map.get_mut(uri).ok_or(FileDoesNotExist)?;
        *slot = content;
        Ok(())
    }

    pub fn delete(&mut self, uri: &Uri) -> Result<(), FileDoesNotExist> {
        match self.map.swap_remove(uri) {
            Some(_) => Ok(()),
            None => Err(FileDoesNotExist),
        }
    }

    pub fn exists(&self, uri: &Uri) -> bool {
        self.map.contains_key(uri)
    }

    pub fn get(&self, uri: &Uri) -> Result<&Content, FileDoesNotExist> {
        self.map.get(uri).ok_or(FileDoesNotExist)
    }
}

impl NotificationHandler<DidOpenTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidOpenTextDocumentParams) -> Result<(), NotificationError> {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem { uri, text, language_id: _, version: _ } = text_document;

        if !self.open_files.insert(uri.clone()) {
            throw!(NotificationError::Invalid(format!("trying to open file {uri:?} which is already open")))
        }

        if self.virtual_file_system.exists(&uri) {
            self.virtual_file_system.update(&uri, Content::Text(text)).unwrap();
        } else {
            self.virtual_file_system.create(&uri, Content::Text(text)).unwrap();
        }

        Ok(())
    }
}

impl NotificationHandler<DidCloseTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidCloseTextDocumentParams) -> Result<(), NotificationError> {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        if !self.open_files.remove(&uri) {
            throw!(NotificationError::Invalid(format!("trying to close file {uri:?} which is not open")))
        }

        // leave the file content in the VFS

        Ok(())
    }
}

impl NotificationHandler<DidChangeTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidChangeTextDocumentParams) -> Result<(), NotificationError> {
        let DidChangeTextDocumentParams { text_document, content_changes } = params;
        let VersionedTextDocumentIdentifier { uri, version: _ } = text_document;

        if !self.open_files.contains(&uri) {
            throw!(NotificationError::Invalid(format!("trying to change file {uri:?} which is not open")))
        }

        for change in content_changes {
            let TextDocumentContentChangeEvent { range, range_length, text } = change;
            assert!(range.is_none() && range_length.is_none());

            self.virtual_file_system.update(&uri, Content::Text(text))
                .expect("file is open so it must exist in the VFS too");
        }

        Ok(())
    }
}

impl NotificationHandler<DidChangeWatchedFiles> for ServerState {
    fn handle_notification(&mut self, params: DidChangeWatchedFilesParams) -> Result<(), NotificationError> {
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
                    match self.virtual_file_system.delete(&uri) {
                        Ok(()) => {}
                        Err(e) => {
                            let _: FileDoesNotExist = e;
                            throw!(NotificationError::Invalid(format!("deleted file that does not yet exist file change type {typ:?} for uri {uri:?}")))
                        }
                    }
                }
                _ => {
                    throw!(NotificationError::Invalid(format!("unknown file change type {typ:?} for uri {uri:?}")))
                }
            };
        }

        Ok(())
    }
}