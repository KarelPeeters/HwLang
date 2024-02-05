use std::{fs, io};
use std::fs::DirEntry;
use std::path::Path;

pub fn visit_dirs(dir: &Path, f: &mut impl FnMut(&DirEntry)) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, f)?;
            } else {
                f(&entry);
            }
        }
    }
    Ok(())
}