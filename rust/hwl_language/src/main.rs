use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use clap::Parser;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::data::diagnostic::{Diagnostic, DiagnosticStringSettings, Diagnostics};
use hwl_language::data::parsed::ParsedDatabase;
use hwl_language::data::source::{FilePath, SourceDatabase, SourceSetError};
use hwl_language::new::compile::compile;
use hwl_language::new::lower_verilog::lower;
use hwl_language::util::io::{recurse_for_each_file, IoErrorExt};
use itertools::Itertools;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
    #[arg(long)]
    print_diagnostics_immediately: bool,
}

fn main() {
    let Args {
        root,
        print_diagnostics_immediately,
    } = Args::parse();

    // collect source
    let source = match build_source_database(&root) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("building source database failed: {e:?}");
            std::process::exit(1);
        }
    };

    // build diagnostic handler
    let handler: Option<Box<dyn Fn(&Diagnostic)>> = if print_diagnostics_immediately {
        let source_database = source.clone();
        let handler = move |diag: &Diagnostic| {
            let s = diag
                .clone()
                .to_string(&source_database, DiagnosticStringSettings::default());
            eprintln!("{}\n", s);
        };
        Some(Box::new(handler))
    } else {
        None
    };

    // run compilation
    let diags = Diagnostics::new_with_handler(handler);
    let parsed = ParsedDatabase::new(&diags, &source);
    let compiled = compile(&diags, &source, &parsed);
    let lowered = lower(&diags, &source, &parsed, &compiled);

    // print diagnostics
    let diagnostics = diags.finish();
    let any_error = !diagnostics.is_empty();
    if !print_diagnostics_immediately {
        for diag in diagnostics {
            let s = diag.to_string(&source, DiagnosticStringSettings::default());
            eprintln!("{}\n", s);
        }
    }

    // print result
    if let Ok(lowered) = lowered {
        println!("top module name: {:?}", lowered.top_module_name);
        println!("verilog source:");
        println!("----------------------------------------");
        print!("{}", lowered.verilog_source);
        if !lowered.verilog_source.ends_with("\n") {
            println!();
        }
        println!("----------------------------------------");
    }

    if any_error {
        std::process::exit(1);
    }
}

fn build_source_database(root: &Path) -> Result<SourceDatabase, SourceSetError> {
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
        source_database
            .add_file(FilePath(stack), path.to_str().unwrap().to_owned(), source)
            .unwrap();

        Ok(())
    })
    .unwrap();

    if source_database.len() == 0 {
        println!("Warning: no input files found");
    }

    Ok(source_database)
}
