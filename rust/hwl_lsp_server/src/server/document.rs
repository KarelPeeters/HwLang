use crate::engine::vfs::VfsError;
use crate::server::dispatch::NotificationHandler;
use crate::server::state::{RequestError, RequestResult, ServerState};
use crate::server::util::uri_to_path;
use hwl_language::throw;
use hwl_util::io::IoErrorExt;
use lsp_types::notification::{
    DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument,
};
use lsp_types::{
    DidChangeTextDocumentParams, DidChangeWatchedFilesParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    FileChangeType, FileEvent, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem,
    VersionedTextDocumentIdentifier,
};

impl NotificationHandler<DidOpenTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidOpenTextDocumentParams) -> RequestResult<()> {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem {
            uri,
            text,
            language_id: _,
            version: _,
        } = text_document;

        if !self.open_files.insert(uri.clone()) {
            throw!(RequestError::Invalid(format!(
                "trying to open file {uri:?} which is already open"
            )))
        }

        self.vfs.create_or_update(uri, text.into_bytes());

        Ok(())
    }
}

impl NotificationHandler<DidCloseTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidCloseTextDocumentParams) -> RequestResult<()> {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        if !self.open_files.swap_remove(&uri) {
            throw!(RequestError::Invalid(format!(
                "trying to close file {uri:?} which is not open"
            )))
        }

        // leave the file content in the VFS
        Ok(())
    }
}

impl NotificationHandler<DidChangeTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidChangeTextDocumentParams) -> RequestResult<()> {
        let DidChangeTextDocumentParams {
            text_document,
            content_changes,
        } = params;
        let VersionedTextDocumentIdentifier { uri, version: _ } = text_document;

        if !self.open_files.contains(&uri) {
            throw!(RequestError::Invalid(format!(
                "trying to change file {uri:?} which is not open"
            )))
        }

        // TODO support incremental changes
        for change in content_changes {
            let TextDocumentContentChangeEvent {
                range,
                range_length,
                text,
            } = change;
            assert!(range.is_none() && range_length.is_none());
            self.vfs.create_or_update(uri.clone(), text.into_bytes());
        }

        Ok(())
    }
}

impl NotificationHandler<DidChangeWatchedFiles> for ServerState {
    fn handle_notification(&mut self, params: DidChangeWatchedFilesParams) -> RequestResult<()> {
        let DidChangeWatchedFilesParams { changes } = params;

        for change in changes {
            let FileEvent { uri, typ } = change;
            match typ {
                FileChangeType::CREATED => {
                    let path = uri_to_path(&uri)?;
                    let content = std::fs::read(&path).map_err(|e| VfsError::Io(e.with_path(path)))?;
                    self.vfs.create_or_update(uri, content);
                }
                FileChangeType::CHANGED => {
                    let path = uri_to_path(&uri)?;
                    let content = std::fs::read(&path).map_err(|e| VfsError::Io(e.with_path(path)))?;
                    self.vfs.create_or_update(uri, content);
                }
                FileChangeType::DELETED => {
                    self.vfs.delete(&uri)?;
                }
                _ => throw!(RequestError::Invalid(format!(
                    "unknown file change type {typ:?} for uri {uri:?}"
                ))),
            };
        }

        Ok(())
    }
}
