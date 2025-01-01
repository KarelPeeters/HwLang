use std::ffi::OsString;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::{fs, io};

pub fn recurse_for_each_file(
    dir: &Path,
    f: &mut impl FnMut(&[OsString], &DirEntry) -> Result<(), IoErrorWithPath>,
) -> Result<(), IoErrorWithPath> {
    let mut stack = vec![];
    recurse_for_each_file_impl(dir, &mut stack, f)
}

pub fn recurse_for_each_file_impl(
    root: &Path,
    stack: &mut Vec<OsString>,
    f: &mut impl FnMut(&[OsString], &DirEntry) -> Result<(), IoErrorWithPath>,
) -> Result<(), IoErrorWithPath> {
    if root.is_dir() {
        let read_dir = fs::read_dir(root).map_err(|e| IoErrorWithPath {
            path: root.to_owned(),
            error: e,
        })?;
        for entry in read_dir {
            let entry = entry.map_err(|e| IoErrorWithPath {
                path: root.to_owned(),
                error: e,
            })?;
            let next = entry.path();
            if next.is_dir() {
                stack.push(entry.file_name());
                recurse_for_each_file_impl(&next, stack, f)?;
                stack.pop();
            } else {
                f(stack, &entry)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct IoErrorWithPath {
    pub error: io::Error,
    pub path: PathBuf,
}

pub trait IoErrorExt {
    fn with_path(self, path: PathBuf) -> IoErrorWithPath;
}

impl IoErrorExt for io::Error {
    fn with_path(self, path: PathBuf) -> IoErrorWithPath {
        IoErrorWithPath { error: self, path }
    }
}
