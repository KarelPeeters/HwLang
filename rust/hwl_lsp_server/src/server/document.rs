use crate::engine::vfs::{Content, FileDoesNotExist, VfsError};
use crate::server::dispatch::NotificationHandler;
use crate::server::state::{RequestError, RequestResult, ServerState};
use fluent_uri::enc::EStr;
use fluent_uri::HostData;
use hwl_language::throw;
use lsp_types::notification::{DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument};
use lsp_types::{DidChangeTextDocumentParams, DidChangeWatchedFilesParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, FileChangeType, FileEvent, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Uri, VersionedTextDocumentIdentifier};
use std::path::PathBuf;

impl NotificationHandler<DidOpenTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidOpenTextDocumentParams) -> RequestResult<()> {
        let DidOpenTextDocumentParams { text_document } = params;
        let TextDocumentItem { uri, text, language_id: _, version: _ } = text_document;

        if !self.open_files.insert(uri.clone()) {
            throw!(RequestError::Invalid(format!("trying to open file {uri:?} which is already open")))
        }

        let vfs = self.vfs.inner()?;
        if vfs.exists(&uri) {
            vfs.update(&uri, Content::Text(text)).unwrap();
        } else {
            vfs.create(&uri, Content::Text(text)).unwrap();
        }

        Ok(())
    }
}

impl NotificationHandler<DidCloseTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidCloseTextDocumentParams) -> RequestResult<()> {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        if !self.open_files.remove(&uri) {
            throw!(RequestError::Invalid(format!("trying to close file {uri:?} which is not open")))
        }

        // leave the file content in the VFS

        Ok(())
    }
}

impl NotificationHandler<DidChangeTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidChangeTextDocumentParams) -> RequestResult<()> {
        let DidChangeTextDocumentParams { text_document, content_changes } = params;
        let VersionedTextDocumentIdentifier { uri, version: _ } = text_document;

        if !self.open_files.contains(&uri) {
            throw!(RequestError::Invalid(format!("trying to change file {uri:?} which is not open")))
        }

        for change in content_changes {
            let TextDocumentContentChangeEvent { range, range_length, text } = change;
            assert!(range.is_none() && range_length.is_none());

            self.vfs.inner()?.update(&uri, Content::Text(text))
                .expect("file is open so it must exist in the VFS too");
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
                // TODO assert that file does not yet exist for created
                FileChangeType::CREATED | FileChangeType::CHANGED => {
                    // let text = std::fs::read_to_string(uri.path())
                    eprintln!("file at {:?} {:?} changed, we have to read it", uri, println!("{}", uri.path()));
                },
                FileChangeType::DELETED => {
                    match self.vfs.inner()?.delete(&uri) {
                        Ok(()) => {}
                        Err(e) => {
                            let _: FileDoesNotExist = e;
                            throw!(RequestError::Invalid(format!("deleted file that does not yet exist file change type {typ:?} for uri {uri:?}")))
                        }
                    }
                }
                _ => {
                    throw!(RequestError::Invalid(format!("unknown file change type {typ:?} for uri {uri:?}")))
                }
            };
        }

        Ok(())
    }
}

// TODO maybe we want to support non-path files too, as temporary files that haven't been saved yet?
//  or should those just get their own virtual file system each?
pub fn uri_to_path(uri: &Uri) -> Result<PathBuf, VfsError> {
    // check that the URI is just a path
    // TODO check that this works on Windows, Linux and with different LSP clients
    let auth_ok = uri.authority().is_some_and(|a| {
        a.userinfo().is_none() && a.host().data() == HostData::RegName(EStr::new("")) && a.port().is_none()
    });
    let uri_ok = uri.scheme().map(|s| s.as_str()) == Some("file")
        && auth_ok
        && uri.query().is_none()
        && uri.fragment().is_none();

    if !uri_ok {
        return Err(VfsError::InvalidPathUri(uri.clone()));
    }
    // TODO always do decoding or only for some LSP clients? does the protocol really not specify this?
    let path = uri.path().as_estr().decode().into_string()
        .map_err(|_| VfsError::NonUtf8Path(uri.clone()))?
        .into_owned();
    Ok(PathBuf::from(path))
}

