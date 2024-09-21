use crate::syntax::pos::{FileId, LineOffsets, Pos, PosFull, Span, SpanFull};
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::{new_index_type, throw};
use indexmap::IndexMap;
use itertools::enumerate;

/// The full set of source files that are part of this compilation.
/// Immutable once all files have been added.
#[derive(Clone)]
pub struct SourceDatabase {
    // TODO use arena for this too?
    pub files: IndexMap<FileId, FileInfo>,
    pub directories: Arena<Directory, DirectoryInfo>,
    pub root_directory: Directory,
}

// TODO rename
#[derive(Debug, Clone)]
pub enum CompileSetError {
    EmptyPath,
    DuplicatePath(FilePath),
}

/// Path relative to the root of the source database, without the trailing extension.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FilePath(pub Vec<String>);

#[derive(Clone)]
pub struct FileInfo {
    #[allow(dead_code)]
    pub id: FileId,
    #[allow(dead_code)]
    pub directory: Directory,

    /// only intended for use in user-visible diagnostic messages
    pub path_raw: String,

    pub source: String,
    pub offsets: LineOffsets,
}

// TODO rename to "FilePath" or "SourcePath"
new_index_type!(pub Directory);

#[derive(Clone)]
pub struct DirectoryInfo {
    #[allow(dead_code)]
    pub path: FilePath,
    pub file: Option<FileId>,
    pub children: IndexMap<String, Directory>,
}

impl SourceDatabase {
    pub fn new() -> Self {
        let mut directories = Arena::default();
        let root_directory = directories.push(DirectoryInfo {
            path: FilePath(vec![]),
            file: None,
            children: Default::default(),
        });

        SourceDatabase {
            files: IndexMap::default(),
            directories,
            root_directory,
        }
    }

    // TODO parse immediately?
    pub fn add_file(&mut self, path: FilePath, path_raw: String, source: String) -> Result<FileId, CompileSetError> {
        if path.0.is_empty() {
            throw!(CompileSetError::EmptyPath);
        }

        let file_id = FileId(self.files.len());
        eprintln!("adding {:?} => {:?}", file_id, path);
        let directory = self.get_directory(&path);
        let info = FileInfo {
            id: file_id,
            directory,
            path_raw,
            offsets: LineOffsets::new(&source),
            source,
        };

        let slot = &mut self.directories[directory].file;
        if slot.is_some() {
            throw!(CompileSetError::DuplicatePath(path));
        }
        assert_eq!(*slot, None);
        *slot = Some(file_id);

        self.files.insert_first(file_id, info);
        Ok(file_id)
    }

    pub fn add_external_vhdl(&mut self, library: String, source: String) {
        todo!("compile VHDL library={:?}, source.len={}", library, source.len())
    }

    pub fn add_external_verilog(&mut self, source: String) {
        todo!("compile verilog source.len={}", source.len())
    }

    fn get_directory(&mut self, path: &FilePath) -> Directory {
        let mut curr_dir = self.root_directory;
        for (i, path_item) in enumerate(&path.0) {
            curr_dir = match self.directories[curr_dir].children.get(path_item) {
                Some(&child) => child,
                None => {
                    let curr_path = &path.0[..=i];
                    let child = self.directories.push(DirectoryInfo {
                        path: FilePath(curr_path.to_vec()),
                        file: None,
                        children: Default::default(),
                    });
                    self.directories[curr_dir].children.insert(path_item.clone(), child);
                    child
                }
            };
        }
        curr_dir
    }

    pub fn expand_pos(&self, pos: Pos) -> PosFull {
        self[pos.file].offsets.expand_pos(pos)
    }

    pub fn expand_span(&self, span: Span) -> SpanFull {
        self[span.start.file].offsets.expand_span(span)
    }
}

impl std::ops::Index<FileId> for SourceDatabase {
    type Output = FileInfo;
    fn index(&self, index: FileId) -> &Self::Output {
        self.files.get(&index).unwrap()
    }
}

impl std::ops::Index<Directory> for SourceDatabase {
    type Output = DirectoryInfo;
    fn index(&self, index: Directory) -> &Self::Output {
        &self.directories[index]
    }
}