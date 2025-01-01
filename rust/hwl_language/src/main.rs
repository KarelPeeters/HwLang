use clap::Parser;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::front::compile::compile;
use hwl_language::front::diagnostic::{Diagnostic, DiagnosticStringSettings, Diagnostics};
use hwl_language::front::lower_verilog::lower;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FilePath, SourceDatabase, SourceSetError};
use hwl_language::syntax::token::Tokenizer;
use hwl_language::util::io::{recurse_for_each_file, IoErrorExt};
use itertools::Itertools;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    root: PathBuf,
    #[arg(long)]
    print_diagnostics_immediately: bool,
    #[arg(long)]
    profile: bool,
}

fn main() {
    // spawn a new thread with a larger stack size
    // TODO should we do this or switch to a heap stack everywhere (mostly item visiting and verilog lowering)
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(main_inner)
        .unwrap()
        .join()
        .unwrap();
}

fn main_inner() {
    let Args {
        root,
        print_diagnostics_immediately,
        profile,
    } = Args::parse();

    // collect source
    let start_all = Instant::now();
    let start_source = Instant::now();
    let source = match build_source_database(&root) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("building source database failed: {e:?}");
            std::process::exit(1);
        }
    };
    let source = Rc::new(source);
    let time_source = start_source.elapsed();

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
    let diags = Diagnostics::new_with_handler(handler);

    // run compilation
    let start_parse = Instant::now();
    let parsed = ParsedDatabase::new(&diags, &source);
    let time_parse = start_parse.elapsed();

    let start_compile = Instant::now();
    let compiled = compile(&diags, &source, &parsed);
    let time_compile = start_compile.elapsed();

    let start_lower = Instant::now();
    let lowered = lower(&diags, &source, &parsed, &compiled);
    let time_lower = start_lower.elapsed();

    let time_all = start_all.elapsed();

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
        if !profile {
            println!("verilog source:");
            println!("----------------------------------------");
            print!("{}", lowered.verilog_source);
            if !lowered.verilog_source.ends_with("\n") {
                println!();
            }
            println!("----------------------------------------");
        }
    }

    // print profiling info
    if profile {
        // profile tokenization separately
        let start_tokenize = Instant::now();
        let mut total_tokens = 0;
        for file in source.files() {
            total_tokens += Tokenizer::new(file, &source[file].source).into_iter().count();
        }
        let time_tokenize = start_tokenize.elapsed();

        println!();
        println!("profiling info:");
        println!("-----------------------------------------------");
        println!("input files:      {}", source.file_count());
        println!("input lines:      {}", source.total_lines_of_code());
        println!("input tokens:     {}", total_tokens);
        println!("----------------------------------------");
        println!("read source:      {:?}", time_source);
        println!("tokenize:         {:?}", time_tokenize);
        println!("parse + tokenize: {:?}", time_parse);
        println!("compile:          {:?}", time_compile);
        println!("lower:            {:?}", time_lower);
        println!("-----------------------------------------------");
        println!("total:            {:?}", time_all);
        println!();
    }

    // proper exit code
    if any_error {
        std::process::exit(1);
    }
}

fn build_source_database(root: &Path) -> Result<SourceDatabase, SourceSetError> {
    let mut source_database = SourceDatabase::new();

    // TODO proper error handling for IO and string conversion errors
    // TODO make parsing a separate step?
    recurse_for_each_file(root, &mut |stack, f| {
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

    if source_database.file_count() == 0 {
        println!("Warning: no input files found");
    }

    Ok(source_database)
}
