use std::ffi::OsString;
use std::fs::DirEntry;
use std::{fs, io};
use std::path::Path;

pub fn recurse_for_each_file(dir: &Path, f: &mut impl FnMut(&[OsString], &DirEntry)) -> io::Result<()> {
    let mut stack = vec![];
    recurse_for_each_file_impl(dir, &mut stack, f)
}

pub fn recurse_for_each_file_impl(
    root: &Path,
    stack: &mut Vec<OsString>,
    f: &mut impl FnMut(&[OsString], &DirEntry)
) -> io::Result<()> {
    if root.is_dir() {
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let next = entry.path();
            if next.is_dir() {
                stack.push(entry.file_name());
                recurse_for_each_file_impl(&next, stack, f)?;
                stack.pop();
            } else {
                f(&stack, &entry);
            }
        }
    }
    Ok(())
}