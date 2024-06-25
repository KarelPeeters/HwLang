use std::ffi::OsStr;
use std::path::PathBuf;

use clap::Parser;
use itertools::Itertools;

use hwlang::resolve::compile::{CompileSet, FilePath};
use hwlang::util::io::recurse_for_each_file;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
}

fn main() {
    let args = Args::parse();

    let mut set = CompileSet::new();

    recurse_for_each_file(&args.root, &mut |stack, f| {
        let path = f.path();
        if path.extension() != Some(OsStr::new("kh")) {
            return;
        }
        
        let mut stack = stack.iter().map(|s| s.to_str().unwrap().to_owned()).collect_vec();
        stack.push(path.file_stem().unwrap().to_str().unwrap().to_owned());
        
        let source = std::fs::read_to_string(&path).unwrap();
        set.add_file(FilePath(stack), source).unwrap();
    }).unwrap();

    set.compile().unwrap();
}
