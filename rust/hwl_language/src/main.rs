use std::ffi::OsStr;
use std::path::PathBuf;

use clap::Parser;
use hwl_language::back::lower;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::data::lowered::LoweredDatabase;
use hwl_language::data::source::{FilePath, SourceDatabase};
use hwl_language::error::CompileError;
use hwl_language::front::driver::compile;
use hwl_language::util::io::{recurse_for_each_file, IoErrorExt};
use itertools::Itertools;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
}

fn main() {
    let args = Args::parse();

    match main_inner(&args) {
        Ok(result) => {
            println!("Compilation finished successfully");
            println!();
            println!("top module name: {}", result.top_module_name);
            println!("verilog source:");
            println!("----------------------------------------");
            print!("{}", result.verilog_source);
            if !result.verilog_source.ends_with("\n") {
                println!();
            }
            println!("----------------------------------------");
        }
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

fn main_inner(args: &Args) -> Result<LoweredDatabase, CompileError> {
    let Args { root } = args;

    let mut source_database = SourceDatabase::new();

    // TODO proper error handling for IO and string conversion errors
    // TODO make parsing a separate step?
    recurse_for_each_file(&root, &mut |stack, f| {
        let path = f.path();
        if path.extension() != Some(OsStr::new(LANGUAGE_FILE_EXTENSION)) {
            return Ok(());
        }

        let mut stack = stack.iter().map(|s| s.to_str().unwrap().to_owned()).collect_vec();
        stack.push(path.file_stem().unwrap().to_str().unwrap().to_owned());

        let source = std::fs::read_to_string(&path).map_err(|e| e.with_path(path.clone()))?;
        source_database.add_file(FilePath(stack), path.to_str().unwrap().to_owned(), source).unwrap();

        Ok(())
    }).unwrap();

    if source_database.files.len() == 0 {
        println!("Warning: no input files found");
    }

    let compiled_database = compile(&source_database)?;
    let lowered_database = lower(&source_database, &compiled_database)?;

    Ok(lowered_database)
}
