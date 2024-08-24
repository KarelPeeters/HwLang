use crate::server::document;
use crate::server::document::uri_to_path;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::util::io::{recurse_for_each_file, IoErrorWithPath};
use indexmap::IndexMap;
use lsp_types::Uri;
use std::ffi::OsStr;
use std::path::PathBuf;

/// The name and the general principle come from the VFS of rust-analyzer.
pub struct VirtualFileSystem {
    root: PathBuf,
    map: IndexMap<PathBuf, Content>,
    has_changed: bool,
}

pub enum Content {
    Unknown(Vec<u8>),
    Text(String),
}

#[derive(Debug)]
pub enum VfsError {
    InvalidPathUri(Uri),
    NonUtf8Path(Uri),
    FileAlreadyExists(Uri, PathBuf),
    FileDoesNotExist(Uri, PathBuf),
    Io(IoErrorWithPath),
}

pub type VfsResult<T> = Result<T, VfsError>;

impl VirtualFileSystem {
    pub fn new(root: Uri) -> VfsResult<Self> {
        eprintln!("new VFS with root {:?} {:?} {:?}", root.as_str(), root, root.path());

        let root_path = document::uri_to_path(&root)?;
        let vfs = VirtualFileSystem {
            map: IndexMap::default(),
            has_changed: true,
            root: root_path.clone(),
        };

        // initialize files from disk
        recurse_for_each_file(&root_path, &mut |steps, f| {
            let path = f.path();
            if path.extension() != Some(OsStr::new(LANGUAGE_FILE_EXTENSION)) {
                return;
            }

            eprintln!("would read file at {:?}", path);
        })?;

        Ok(vfs)
    }

    pub fn create(&mut self, uri: &Uri, content: Content) -> VfsResult<()> {
        self.has_changed = true;

        let path = uri_to_path(uri)?;
        let prev = self.map.insert(path.clone(), content);

        match prev {
            None => Ok(()),
            Some(_) => Err(VfsError::FileAlreadyExists(uri.clone(), path)),
        }
    }

    // TODO support incremental updates (and even incremental derived data updates?)
    pub fn update(&mut self, uri: &Uri, content: Content) -> VfsResult<()> {
        self.has_changed = true;

        let path = uri_to_path(uri)?;
        let slot = self.map.get_mut(&path)
            .ok_or_else(|| VfsError::FileDoesNotExist(uri.clone(), path))?;
        *slot = content;

        Ok(())
    }

    pub fn delete(&mut self, uri: &Uri) -> VfsResult<()> {
        self.has_changed = true;

        let path = uri_to_path(uri)?;
        let prev = self.map.swap_remove(&path);

        match prev {
            Some(_) => Ok(()),
            None => Err(VfsError::FileDoesNotExist(uri.clone(), path)),
        }
    }

    pub fn exists(&self, uri: &Uri) -> VfsResult<bool> {
        let path = uri_to_path(uri)?;
        Ok(self.map.contains_key(&path))
    }

    pub fn get(&self, uri: &Uri) -> VfsResult<&Content> {
        let path = uri_to_path(uri)?;
        self.map.get(&path)
            .ok_or_else(|| VfsError::FileDoesNotExist(uri.clone(), path))
    }

    pub fn get_and_clear_changed(&mut self) -> bool {
        std::mem::take(&mut self.has_changed)
    }
}

impl From<IoErrorWithPath> for VfsError {
    fn from(value: IoErrorWithPath) -> Self {
        VfsError::Io(value)
    }
}
