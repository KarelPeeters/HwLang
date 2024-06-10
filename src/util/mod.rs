use std::{fs, io};
use std::fs::DirEntry;
use std::path::Path;

pub fn recurse_for_each_file(dir: &Path, f: &mut impl FnMut(&DirEntry)) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                recurse_for_each_file(&path, f)?;
            } else {
                f(&entry);
            }
        }
    }
    Ok(())
}