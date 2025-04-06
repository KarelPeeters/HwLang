use crate::constants::LANGUAGE_FILE_EXTENSION;
use crate::syntax::pos::{LineOffsets, Pos, PosFull, Span, SpanFull};
use crate::util::arena::{Arena, IndexType};
use crate::util::data::vec_concat;
use crate::util::io::{recurse_for_each_file, IoErrorExt, IoErrorWithPath};
use crate::{new_index_type, throw};
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Mutable builder for [SourceDataBase].
pub struct SourceDatabaseBuilder {
    root_directory: BuilderDirectory,
    directories: Arena<BuilderDirectory, DirectoryInfo<BuilderFileId, BuilderDirectory>>,
    files: Arena<BuilderFileId, FileSourceInfo<BuilderFileId, BuilderDirectory>>,
}

/// The full set of source files that are part of this compilation. This type is immutable.
pub struct SourceDatabase {
    root_directory: Directory,
    directories: Arena<Directory, DirectoryInfo>,
    files: Arena<FileId, FileSourceInfo>,
}

// TODO rename to file
new_index_type!(pub FileId, Ord);
new_index_type!(pub BuilderFileId);
// TODO rename to "FilePath" or "SourcePath"
new_index_type!(pub Directory, Ord);
new_index_type!(pub BuilderDirectory);

// TODO rename
#[derive(Debug, Clone)]
pub enum SourceSetError {
    EmptyPath,
    DuplicatePath(FilePath),
    NonUtf8Path(PathBuf),
    MissingFileName(PathBuf),
}

#[derive(Debug)]
pub enum SourceSetOrIoError {
    SourceSet(SourceSetError),
    Io(IoErrorWithPath),
}

/// Path relative to the root of the source database, without the trailing extension.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FilePath(pub Vec<String>);

#[derive(Clone)]
pub struct FileSourceInfo<F = FileId, D = Directory> {
    pub file_id: F,
    pub directory: D,
    pub source: String,
    pub offsets: LineOffsets,

    /// only intended for use in user-visible diagnostic messages
    pub path_raw: String,
}

#[derive(Clone)]
pub struct DirectoryInfo<F = FileId, D = Directory> {
    #[allow(dead_code)]
    pub path: FilePath,
    pub file: Option<F>,
    pub children: IndexMap<String, D>,
}

impl SourceDatabase {
    pub fn root_directory(&self) -> Directory {
        self.root_directory
    }

    /// Get the list of files, in a platform-independent sorted order.
    pub fn files(&self) -> impl Iterator<Item = FileId> + Clone + '_ {
        self.files.keys()
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn total_lines_of_code(&self) -> u64 {
        self.files
            .values()
            .map(|file_info| file_info.offsets.line_count() as u64)
            .sum()
    }

    pub fn expand_pos(&self, pos: Pos) -> PosFull {
        self[pos.file].offsets.expand_pos(pos)
    }

    pub fn expand_span(&self, span: Span) -> SpanFull {
        self[span.start.file].offsets.expand_span(span)
    }
}

impl SourceDatabaseBuilder {
    pub fn new() -> Self {
        let mut dirs = Arena::default();
        let root_dir = dirs.push(DirectoryInfo {
            path: FilePath(vec![]),
            file: None,
            children: Default::default(),
        });

        SourceDatabaseBuilder {
            root_directory: root_dir,
            directories: dirs,
            files: Arena::default(),
        }
    }

    pub fn finish(self) -> SourceDatabase {
        let (source, _, _) = self.finish_with_mapping();
        source
    }

    pub fn finish_with_mapping(
        self,
    ) -> (
        SourceDatabase,
        IndexMap<BuilderDirectory, Directory>,
        IndexMap<BuilderFileId, FileId>,
    ) {
        // sort to ensure deterministic order and keys,
        //   this forms the base of the cross-platform and multithreaded determinism in the rest of the compiler
        let mut dirs_ordered = self.directories.keys().collect_vec();
        dirs_ordered.sort_by_key(|&dir| &self.directories[dir].path);
        let mut files_ordered = self.files.keys().collect_vec();
        files_ordered.sort_by_key(|&file| &self.directories[self.files[file].directory].path);

        // TODO this is a mess, we're redundantly allocating arenas and hashmaps
        //   but it's all O(files), so not _that_ bad
        let mut dirs_dummy: Arena<Directory, ()> = Arena::default();
        let mut files_dummy: Arena<FileId, ()> = Arena::default();

        let mut dirs_build_to_final = IndexMap::new();
        let mut dirs_final_to_build = IndexMap::new();
        let mut files_build_to_final = IndexMap::new();
        let mut files_final_to_build = IndexMap::new();

        for build_file in files_ordered {
            let file = files_dummy.push(());
            files_build_to_final.insert(build_file, file);
            files_final_to_build.insert(file, build_file);
        }
        for build_dir in dirs_ordered {
            let dir = dirs_dummy.push(());
            dirs_build_to_final.insert(build_dir, dir);
            dirs_final_to_build.insert(dir, build_dir);
        }

        let dirs: Arena<Directory, DirectoryInfo> = dirs_dummy.map_values(|dir, ()| {
            let build_dir = &self.directories[*dirs_final_to_build.get(&dir).unwrap()];
            DirectoryInfo {
                path: build_dir.path.clone(),
                file: build_dir.file.map(|file| *files_build_to_final.get(&file).unwrap()),
                children: build_dir
                    .children
                    .iter()
                    .map(|(k, v)| {
                        let child = *dirs_build_to_final.get(v).unwrap();
                        (k.clone(), child)
                    })
                    .collect(),
            }
        });
        let files: Arena<FileId, FileSourceInfo> = files_dummy.map_values(|file, ()| {
            let build_file = &self.files[*files_final_to_build.get(&file).unwrap()];
            FileSourceInfo {
                file_id: file,
                directory: *dirs_build_to_final.get(&build_file.directory).unwrap(),
                path_raw: build_file.path_raw.clone(),
                offsets: build_file.offsets.clone(),
                source: build_file.source.clone(),
            }
        });

        let root_dir = *dirs_build_to_final.get(&self.root_directory).unwrap();

        let source = SourceDatabase {
            root_directory: root_dir,
            directories: dirs,
            files,
        };
        (source, dirs_build_to_final, files_build_to_final)
    }

    pub fn add_file(
        &mut self,
        path: FilePath,
        path_raw: String,
        source: String,
    ) -> Result<BuilderFileId, SourceSetError> {
        if path.0.is_empty() {
            throw!(SourceSetError::EmptyPath);
        }

        let directory = self.get_directory(&path);
        let slot = &mut self.directories[directory].file;
        if slot.is_some() {
            throw!(SourceSetError::DuplicatePath(path));
        }

        let file_id = self.files.push_with_index(|file_id| FileSourceInfo {
            file_id,
            directory,
            path_raw,
            offsets: LineOffsets::new(&source),
            source,
        });
        *slot = Some(file_id);
        Ok(file_id)
    }

    pub fn add_tree(&mut self, prefix: Vec<String>, root_path: &Path) -> Result<(), SourceSetOrIoError> {
        if !root_path.exists() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotFound)
                .with_path(root_path.to_owned())
                .into());
        }
        if !root_path.is_dir() {
            return Err(std::io::Error::from(std::io::ErrorKind::NotADirectory)
                .with_path(root_path.to_owned())
                .into());
        }

        recurse_for_each_file(root_path, |stack, f| -> Result<(), SourceSetOrIoError> {
            let path = f.path();
            if path.extension() != Some(OsStr::new(LANGUAGE_FILE_EXTENSION)) {
                return Ok(());
            }

            let mut stack: Vec<String> = stack
                .iter()
                .map(|s| to_str_or_err(&path, s).map(str::to_owned))
                .try_collect()?;

            let file_stem = path
                .file_stem()
                .ok_or_else(|| SourceSetError::MissingFileName(path.clone()))?;
            stack.push(to_str_or_err(&path, file_stem)?.to_owned());

            let filepath = FilePath(vec_concat([prefix.clone(), stack]));
            let source = std::fs::read_to_string(&path).map_err(|e| e.with_path(path.clone()))?;
            self.add_file(filepath, path.to_str().unwrap().to_owned(), source)
                .unwrap();

            Ok(())
        })?;

        Ok(())
    }

    fn get_directory(&mut self, path: &FilePath) -> BuilderDirectory {
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
}

impl std::ops::Index<FileId> for SourceDatabase {
    type Output = FileSourceInfo;
    fn index(&self, index: FileId) -> &Self::Output {
        &self.files[index]
    }
}

impl std::ops::Index<Directory> for SourceDatabase {
    type Output = DirectoryInfo;
    fn index(&self, index: Directory) -> &Self::Output {
        &self.directories[index]
    }
}

impl From<SourceSetError> for SourceSetOrIoError {
    fn from(e: SourceSetError) -> Self {
        SourceSetOrIoError::SourceSet(e)
    }
}

impl From<IoErrorWithPath> for SourceSetOrIoError {
    fn from(e: IoErrorWithPath) -> Self {
        SourceSetOrIoError::Io(e)
    }
}

fn to_str_or_err<'s>(path: &Path, s: &'s OsStr) -> Result<&'s str, SourceSetError> {
    s.to_str().ok_or_else(|| SourceSetError::NonUtf8Path(path.to_owned()))
}

impl FileId {
    pub fn dummy() -> FileId {
        let mut arena = Arena::default();
        arena.push(())
    }
}
