use clap::Parser;
use hwl_language::front::compile::{compile, StdoutPrintHandler};
use hwl_language::front::diagnostic::{Diagnostic, DiagnosticStringSettings, Diagnostics};
use hwl_language::front::lower_verilog::lower;
use hwl_language::simulator::simulator_codegen;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::token::Tokenizer;
use hwl_language::util::ResultExt;
use std::path::PathBuf;
use std::process::ExitCode;
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

fn main() -> ExitCode {
    // spawn a new thread with a larger stack size
    // TODO should we do this or switch to a heap stack everywhere (mostly item visiting and verilog lowering)
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(main_inner)
        .unwrap()
        .join()
        .unwrap()
}

fn main_inner() -> ExitCode {
    let Args {
        root,
        print_diagnostics_immediately,
        profile,
    } = Args::parse();

    // collect source
    let start_all = Instant::now();
    let start_source = Instant::now();

    let mut source = SourceDatabase::new();
    match source.add_tree(vec![], &root) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("building source database failed: {e:?}");
            return ExitCode::FAILURE;
        }
    }

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
    let compiled = compile(&diags, &source, &parsed, &mut StdoutPrintHandler);
    let time_compile = start_compile.elapsed();

    let start_lower = Instant::now();
    let lowered = compiled
        .as_ref_ok()
        .and_then(|c| lower(&diags, &source, &parsed, &c.modules, c.top_module));
    let time_lower = start_lower.elapsed();

    let start_simulator = Instant::now();
    let simulator_code = compiled
        .as_ref_ok()
        .and_then(|c| simulator_codegen(&diags, &c.modules, c.top_module));
    let time_simulator = start_simulator.elapsed();

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

    // save lowered verilog
    std::fs::create_dir_all("ignored").unwrap();
    std::fs::write(
        "ignored/lowered.v",
        lowered.as_ref().map_or(&String::new(), |s| &s.verilog_source),
    )
    .unwrap();

    // save simulator code
    std::fs::write(
        "ignored/simulator.cpp",
        simulator_code.as_ref().unwrap_or(&String::new()),
    )
    .unwrap();

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
        println!("lower verilog:    {:?}", time_lower);
        println!("lower c++:        {:?}", time_simulator);
        println!("-----------------------------------------------");
        println!("total:            {:?}", time_all);
        println!();
    }

    // proper exit code
    if any_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
