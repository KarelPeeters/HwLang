use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::{fs, io};

// TODO make this only accept strings, nothing else will accept non-utf8 paths anyway
pub fn recurse_for_each_file<E: From<IoErrorWithPath>>(
    dir: &Path,
    mut f: impl FnMut(&[OsString], &Path) -> Result<(), E>,
) -> Result<(), E> {
    let mut stack = vec![];
    recurse_for_each_file_impl(dir, &mut stack, &mut f)
}

fn recurse_for_each_file_impl<E: From<IoErrorWithPath>>(
    root: &Path,
    stack: &mut Vec<OsString>,
    f: &mut impl FnMut(&[OsString], &Path) -> Result<(), E>,
) -> Result<(), E> {
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
            f(stack, &entry.path())?;
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
