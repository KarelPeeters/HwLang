use std::ffi::OsStr;
use std::path::PathBuf;

use clap::Parser;
use itertools::Itertools;
use language::error::CompileError;
use language::front::driver::compile;
use language::front::source::{FilePath, SourceDatabase};
use language::util::io::recurse_for_each_file;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
}

fn main() {
    let Args { root } = Args::parse();

    let mut database = SourceDatabase::new();

    // TODO proper error handling for IO and string conversion errors
    recurse_for_each_file(&root, &mut |stack, f| {
        let path = f.path();
        if path.extension() != Some(OsStr::new("kh")) {
            return;
        }
        
        let mut stack = stack.iter().map(|s| s.to_str().unwrap().to_owned()).collect_vec();
        stack.push(path.file_stem().unwrap().to_str().unwrap().to_owned());
        
        let source = std::fs::read_to_string(&path).unwrap();
        database.add_file(FilePath(stack), path.to_str().unwrap().to_owned(), source).unwrap();
    }).unwrap();

    if database.files.len() == 0 {
        println!("Warning: no input files found");
    }

    match compile(&database) {
        Ok(()) => println!("Compilation finished successfully"),
        Err(e) => {
            match e {
                CompileError::SnippetError(e) => eprintln!("{}", e.string),
                _ => eprintln!("{:?}", e),
            }
            eprintln!("Compilation failed");
            std::process::exit(1);
        }
    }
}
