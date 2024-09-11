use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use clap::Parser;
use hwl_language::back::lower;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::data::diagnostic::{DiagnosticStringSettings, Diagnostics};
use hwl_language::data::lowered::LoweredDatabase;
use hwl_language::data::source::{CompileSetError, FilePath, SourceDatabase};
use hwl_language::error::CompileError;
use hwl_language::front::driver::compile;
use hwl_language::util::io::{recurse_for_each_file, IoErrorExt};
use itertools::Itertools;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
}

fn main() {
    let Args { root } = Args::parse();

    // collect source
    let source_database = match build_source_database(&root) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("building source database failed: {e:?}");
            std::process::exit(1);
        }
    };

    // run compilation
    let diag = Diagnostics::new();
    let main_result = main_inner(&diag, &source_database);

    // print diagnostics
    for diag in diag.finish() {
        let s = diag.to_string(&source_database, DiagnosticStringSettings::default());
        eprintln!("{}", s);
    }

    // print result
    match main_result {
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
            eprintln!("{:?}", e);
            eprintln!("Compilation failed");
            std::process::exit(1);
        }
    }
}

fn main_inner(diag: &Diagnostics, source_database: &SourceDatabase) -> Result<LoweredDatabase, CompileError> {
    let compiled_database = compile(diag, &source_database);
    let lowered_database = lower(diag, &source_database, &compiled_database)?;
    Ok(lowered_database)
}

fn build_source_database(root: &Path) -> Result<SourceDatabase, CompileSetError> {
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

    Ok(source_database)
}
