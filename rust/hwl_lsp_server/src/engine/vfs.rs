use crate::server::document::uri_to_path;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::throw;
use hwl_language::util::data::IndexMapExt;
use hwl_language::util::io::{recurse_for_each_file, IoErrorExt, IoErrorWithPath};
use indexmap::IndexMap;
use lsp_types::Uri;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;

/// The name and the general principle come from the VFS of rust-analyzer.
pub struct VirtualFileSystem {
    root: PathBuf,
    map_rel: IndexMap<PathBuf, Content>,
    has_changed: bool,
}

pub enum Content {
    Unknown(Vec<u8>),

    Text(String),
    NonUtf8(Utf8Error),
}

#[derive(Debug)]
pub enum VfsError {
    InvalidPathUri(Uri),
    NonUtf8Path(Uri),
    FileAlreadyExists(Uri, PathBuf),
    FileDoesNotExist(Uri, PathBuf),
    PathDoesNotStartWithRoot(Uri, PathBuf, PathBuf),
    Io(IoErrorWithPath),
    NonUtf8Content(PathBuf, Utf8Error),
}

pub type VfsResult<T> = Result<T, VfsError>;

impl VirtualFileSystem {
    pub fn new(root: Uri) -> VfsResult<Self> {
        eprintln!("new VFS with root {:?} {:?} {:?}", root.as_str(), root, root.path());

        let root_path = uri_to_path(&root)?;
        let mut vfs = VirtualFileSystem {
            map_rel: IndexMap::default(),
            has_changed: true,
            root: root_path.clone(),
        };

        // initialize files from disk
        recurse_for_each_file(&root_path, &mut |_, f| {
            let path = f.path();
            if path.extension() != Some(OsStr::new(LANGUAGE_FILE_EXTENSION)) {
                return Ok(());
            }

            let path_rel = path.strip_prefix(&root_path).unwrap();
            let source = std::fs::read(&path)
                .map_err(|e| e.with_path(path.clone()))?;

            vfs.map_rel.insert_first(path_rel.to_owned(), Content::Unknown(source));

            eprintln!("read file {:?}", path_rel);
            Ok(())
        })?;

        Ok(vfs)
    }

    pub fn create(&mut self, uri: &Uri, content: Content) -> VfsResult<()> {
        self.has_changed = true;

        let path = self.uri_to_relative_path(uri)?;
        let prev = self.map_rel.insert(path.clone(), content);

        match prev {
            None => Ok(()),
            Some(_) => Err(VfsError::FileAlreadyExists(uri.clone(), path)),
        }
    }

    // TODO support incremental updates (and even incremental derived data updates?)
    pub fn update(&mut self, uri: &Uri, content: Content) -> VfsResult<()> {
        self.has_changed = true;

        let path = self.uri_to_relative_path(uri)?;
        let slot = self.map_rel.get_mut(&path)
            .ok_or_else(|| VfsError::FileDoesNotExist(uri.clone(), path))?;
        *slot = content;

        Ok(())
    }

    pub fn delete(&mut self, uri: &Uri) -> VfsResult<()> {
        self.has_changed = true;

        let path = self.uri_to_relative_path(uri)?;
        let prev = self.map_rel.swap_remove(&path);

        match prev {
            Some(_) => Ok(()),
            None => Err(VfsError::FileDoesNotExist(uri.clone(), path)),
        }
    }

    pub fn exists(&self, uri: &Uri) -> VfsResult<bool> {
        let path = self.uri_to_relative_path(uri)?;
        Ok(self.map_rel.contains_key(&path))
    }

    pub fn get_text(&mut self, uri: &Uri) -> VfsResult<&str> {
        let path = self.uri_to_relative_path(uri)?;
        let content = self.map_rel.get_mut(&path)
            .ok_or_else(|| VfsError::FileDoesNotExist(uri.clone(), path.clone()))?;
        content.get_text(&path)
    }

    pub fn get_and_clear_changed(&mut self) -> bool {
        std::mem::take(&mut self.has_changed)
    }

    fn uri_to_relative_path(&self, uri: &Uri) -> VfsResult<PathBuf> {
        let path = uri_to_path(uri)?;
        match path.strip_prefix(&self.root) {
            Ok(rel) => Ok(rel.to_owned()),
            Err(_) => Err(VfsError::PathDoesNotStartWithRoot(uri.clone(), path, self.root.clone())),
        }
    }

    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    pub fn iter(&mut self) -> impl Iterator<Item=(&PathBuf, &mut Content)> {
        self.map_rel.iter_mut()
    }
}

impl Content {
    pub fn get_text(&mut self, path: &Path) -> VfsResult<&str> {
        if let Content::Unknown(bytes) = self {
            let bytes = std::mem::take(bytes);
            *self = match String::from_utf8(bytes) {
                Ok(text) => Content::Text(text),
                Err(e) => Content::NonUtf8(e.utf8_error()),
            };
        }

        match *self {
            Content::Text(ref text) => Ok(text),
            // TODO proper diagnostic that points to the actual place in the file
            //  (shared with the normal compiler)
            Content::NonUtf8(e) => throw!(VfsError::NonUtf8Content(path.to_owned(), e)),
            Content::Unknown(_) => unreachable!(),
        }
    }
}

impl From<IoErrorWithPath> for VfsError {
    fn from(value: IoErrorWithPath) -> Self {
        VfsError::Io(value)
    }
}
