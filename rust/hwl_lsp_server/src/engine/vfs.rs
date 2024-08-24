use crate::server::document;
use indexmap::IndexMap;
use lsp_types::Uri;
use std::path::PathBuf;

/// The name and the general principle come from the VFS of rust-analyzer.
pub struct VirtualFileSystem {
    map: IndexMap<Uri, Content>,
    has_changed: bool,
    root: PathBuf,
}

pub enum Content {
    Unknown(Vec<u8>),
    Text(String),
}

#[derive(Debug)]
pub enum VfsError {
    InvalidPathUri(Uri),
    NonUtf8Path(Uri),
    FileAlreadyExists(Uri),
    FileDoesNotExist(Uri),
    Io(std::io::Error),
}

#[derive(Debug)]
pub struct FileAlreadyExists;
#[derive(Debug)]
pub struct FileDoesNotExist;

impl VirtualFileSystem {
    pub fn new(root: Uri) -> Result<Self, VfsError> {
        eprintln!("new VFS with root {:?} {:?} {:?}", root.as_str(), root, root.path());

        let root_path = document::uri_to_path(&root)?;
        let vfs = VirtualFileSystem {
            map: IndexMap::default(),
            has_changed: true,
            root: root_path.clone(),
        };

        // initialize files from disk
        // recurse_for_each_file(&root_path, &mut |steps, f| {
        //     let path = f.path();
        //     if path.extension() != Some(OsStr::new(LANGUAGE_FILE_EXTENSION)) {
        //         return;
        //     }
        //
        //     eprintln!("would read file at {:?}", path);
        // })?;

        Ok(vfs)
    }

    pub fn create(&mut self, uri: &Uri, content: Content) -> Result<(), FileAlreadyExists> {
        self.has_changed = true;
        let prev = self.map.insert(uri.clone(), content);
        match prev {
            None => Ok(()),
            Some(_) => Err(FileAlreadyExists),
        }
    }

    // TODO support incremental updates (and even incremental derived data updates?)
    pub fn update(&mut self, uri: &Uri, content: Content) -> Result<(), FileDoesNotExist> {
        self.has_changed = true;
        let slot = self.map.get_mut(uri).ok_or(FileDoesNotExist)?;
        *slot = content;
        Ok(())
    }

    pub fn delete(&mut self, uri: &Uri) -> Result<(), FileDoesNotExist> {
        self.has_changed = true;
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

    pub fn get_and_clear_changed(&mut self) -> bool {
        std::mem::take(&mut self.has_changed)
    }
}

impl From<std::io::Error> for VfsError {
    fn from(value: std::io::Error) -> Self {
        VfsError::Io(value)
    }
}
