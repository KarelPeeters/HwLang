use crate::server::util::uri_to_path;
use hwl_util::io::{IoErrorExt, IoErrorWithPath};
use indexmap::IndexMap;
use lsp_types::Uri;
use std::path::PathBuf;
use std::str::Utf8Error;

// TODO make sure we only track manifest files and files with the right extension,
// TODO clarify what this is: this is just an overlay over the underlying file system, and is not guaranteed to contain every file
// TODO somehow collect all files referenced by the manifest file and add listeners for them
pub struct Vfs {
    changed: bool,
    files: IndexMap<Uri, Vec<u8>>,
}

pub type VfsResult<T> = Result<T, VfsError>;

// TODO remove unused variants
#[derive(Debug)]
pub enum VfsError {
    InvalidPathUri(Uri),
    Io(IoErrorWithPath),
    ExpectedAbsolutePath(PathBuf),
    FailedToConvertPathToUri(PathBuf, String, fluent_uri::ParseError),
    AlreadyExists(Uri),
    DoesNotExist(Uri),
    // TODO this should not really be a VfsError
    ContentInvalidUtf8(Utf8Error),
    PathInvalidUtf8(PathBuf),
}

// TODO log VFS operations
impl Vfs {
    pub fn new() -> Self {
        Vfs {
            changed: true,
            files: IndexMap::new(),
        }
    }

    pub fn create_or_update(&mut self, uri: Uri, content: Vec<u8>) {
        self.files.insert(uri, content);
        self.changed = true;
    }

    pub fn delete(&mut self, uri: &Uri) -> Result<(), VfsError> {
        match self.files.swap_remove(uri) {
            None => Err(VfsError::DoesNotExist(uri.clone())),
            Some(_) => {
                self.changed = true;
                Ok(())
            }
        }
    }

    pub fn read_maybe_from_disk(&mut self, uri: &Uri) -> Result<&[u8], VfsError> {
        // https://github.com/rust-lang/rust/issues/54663
        if self.files.contains_key(uri) {
            return Ok(self.files.get(uri).unwrap());
        }

        let path = uri_to_path(uri)?;
        let content = std::fs::read(&path).map_err(|e| e.with_path(path))?;

        let slot = self.files.entry(uri.clone()).or_default();
        *slot = content;
        Ok(slot)
    }

    pub fn read_str_maybe_from_disk(&mut self, uri: &Uri) -> Result<&str, VfsError> {
        let content = self.read_maybe_from_disk(uri)?;
        std::str::from_utf8(content).map_err(VfsError::ContentInvalidUtf8)
    }

    pub fn get_and_clear_changed(&mut self) -> bool {
        let changed = self.changed;
        self.changed = false;
        changed
    }
}

impl From<IoErrorWithPath> for VfsError {
    fn from(e: IoErrorWithPath) -> Self {
        VfsError::Io(e)
    }
}
