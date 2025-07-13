use crate::engine::vfs::{Content, VfsError, VfsResult};
use crate::server::dispatch::NotificationHandler;
use crate::server::state::{RequestError, RequestResult, ServerState};
use fluent_uri::enc::EStr;
use fluent_uri::HostData;
use hwl_language::throw;
use hwl_util::io::IoErrorExt;
use lsp_types::notification::{
    DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument,
};
use lsp_types::{
    DidChangeTextDocumentParams, DidChangeWatchedFilesParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    FileChangeType, FileEvent, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Uri,
    VersionedTextDocumentIdentifier,
};
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

        let vfs = self.vfs.inner()?;
        if vfs.exists(&uri)? {
            vfs.update(&uri, Content::Text(text)).expect("cannot fail");
        } else {
            vfs.create(&uri, Content::Text(text)).expect("cannot fail");
        }

        Ok(())
    }
}

impl NotificationHandler<DidCloseTextDocument> for ServerState {
    fn handle_notification(&mut self, params: DidCloseTextDocumentParams) -> RequestResult<()> {
        let DidCloseTextDocumentParams { text_document } = params;
        let TextDocumentIdentifier { uri } = text_document;

        if !self.open_files.remove(&uri) {
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

        for change in content_changes {
            let TextDocumentContentChangeEvent {
                range,
                range_length,
                text,
            } = change;
            assert!(range.is_none() && range_length.is_none());

            self.vfs
                .inner()?
                .update(&uri, Content::Text(text))
                .expect("file is open so it must exist in the VFS too");
        }

        Ok(())
    }
}

impl NotificationHandler<DidChangeWatchedFiles> for ServerState {
    fn handle_notification(&mut self, params: DidChangeWatchedFilesParams) -> RequestResult<()> {
        let DidChangeWatchedFilesParams { changes } = params;

        let vfs = self.vfs.inner()?;

        for change in changes {
            let FileEvent { uri, typ } = change;
            match typ {
                FileChangeType::CREATED => {
                    let path = uri_to_path(&uri)?;
                    let content = std::fs::read(&path).map_err(|e| VfsError::Io(e.with_path(path)))?;
                    vfs.create(&uri, Content::Unknown(content))?;
                }
                FileChangeType::CHANGED => {
                    let path = uri_to_path(&uri)?;
                    let content = std::fs::read(&path).map_err(|e| VfsError::Io(e.with_path(path)))?;
                    vfs.update(&uri, Content::Unknown(content))?;
                }
                FileChangeType::DELETED => match vfs.delete(&uri) {
                    Ok(()) => {}
                    Err(e) => throw!(e),
                },
                _ => throw!(RequestError::Invalid(format!(
                    "unknown file change type {typ:?} for uri {uri:?}"
                ))),
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
        throw!(VfsError::InvalidPathUri(uri.clone()));
    }
    // TODO always do decoding or only for some LSP clients? does the protocol really not specify this?
    let path = uri.path().as_estr().decode().into_string().unwrap();

    // TODO this is sketchy
    let path = if cfg!(windows) {
        match path.strip_prefix('/') {
            Some(path) => path,
            None => throw!(VfsError::InvalidPathUri(uri.clone())),
        }
    } else {
        &*path
    };

    Ok(PathBuf::from(path))
}

// TODO steal all of this from rust-analyzer
pub fn abs_path_to_uri(path: &Path) -> VfsResult<Uri> {
    if !path.is_absolute() {
        throw!(VfsError::ExpectedAbsolutePath(path.to_owned()));
    }

    let path_str = path.to_str().unwrap();
    let uri_str = if cfg!(windows) {
        format!("file:///{path_str}").replace('\\', "/")
    } else {
        format!("file://{path_str}")
    };

    Uri::from_str(&uri_str).map_err(|e| VfsError::FailedToConvertPathToUri(path.to_owned(), uri_str, e))
}
