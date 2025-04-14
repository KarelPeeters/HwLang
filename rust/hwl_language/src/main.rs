use clap::Parser;
use hwl_language::constants::THREAD_STACK_SIZE;
use hwl_language::front::compile::{compile, ElaborationSet, StdoutPrintHandler};
use hwl_language::front::diagnostic::{DiagnosticStringSettings, Diagnostics};
use hwl_language::front::lower_verilog::lower;
use hwl_language::simulator::simulator_codegen;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::SourceDatabaseBuilder;
use hwl_language::syntax::token::Tokenizer;
use hwl_language::util::arena::IndexType;
use hwl_language::util::{ResultExt, NON_ZERO_USIZE_ONE};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    // input data
    root: PathBuf,

    // performance options
    #[arg(long, short = 'j')]
    thread_count: Option<NonZeroUsize>,

    // debug options
    #[arg(long)]
    profile: bool,
    #[arg(long)]
    print_files: bool,
    #[arg(long)]
    print_ir: bool,
    #[arg(long)]
    only_top: bool,
    #[arg(long)]
    skip_lower: bool,
    #[arg(long)]
    keep_main_stack: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.keep_main_stack {
        main_inner(args)
    } else {
        // spawn a new thread with a larger stack size
        // TODO should we do this or switch to a heap stack everywhere (mostly item visiting and verilog lowering)
        std::thread::Builder::new()
            .stack_size(THREAD_STACK_SIZE)
            .spawn(|| main_inner(args))
            .unwrap()
            .join()
            .unwrap()
    }
}

fn main_inner(args: Args) -> ExitCode {
    let Args {
        root,
        thread_count,
        profile,
        print_files,
        print_ir,
        only_top,
        skip_lower,
        keep_main_stack: _,
    } = args;

    let thread_count = thread_count.unwrap_or_else(|| NonZeroUsize::new(num_cpus::get()).unwrap_or(NON_ZERO_USIZE_ONE));
    let elaboration_set = if only_top {
        ElaborationSet::TopOnly
    } else {
        ElaborationSet::AsMuchAsPossible
    };

    let start_all = Instant::now();

    // collect source
    let start_source = Instant::now();
    let mut source_builder = SourceDatabaseBuilder::new();
    match source_builder.add_tree(vec![], &root) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("building source database failed: {e:?}");
            return ExitCode::FAILURE;
        }
    }
    let source = source_builder.finish();
    let time_source = start_source.elapsed();

    // print source info
    if print_files {
        eprintln!("Collected sources:");
        for file in source.files() {
            eprintln!("  [{}]: {:?}", file.inner().index(), &source[file].path_raw);
        }
    }

    // build diagnostics
    let diags = Diagnostics::new();

    // run compilation
    // TODO parallelize parsing
    let start_parse = Instant::now();
    let parsed = ParsedDatabase::new(&diags, &source);
    let time_parse = start_parse.elapsed();

    let start_compile = Instant::now();
    let compiled = compile(
        &diags,
        &source,
        &parsed,
        elaboration_set,
        &mut StdoutPrintHandler,
        &|| false,
        thread_count,
    );
    let time_compile = start_compile.elapsed();

    // TODO parallelize lowering?
    let lower_results = if skip_lower {
        None
    } else {
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

        Some((time_lower, time_simulator, lowered, simulator_code))
    };

    let time_all = start_all.elapsed();

    // print diagnostics
    let diagnostics = diags.finish();
    let any_error = !diagnostics.is_empty();
    for diag in diagnostics {
        let s = diag.to_string(&source, DiagnosticStringSettings::default());
        eprintln!("{}\n", s);
    }

    if print_ir {
        println!("{:#?}", compiled);
    }

    // TODO don't hardcode paths here
    if let Some((_, _, lowered, simulator_code)) = &lower_results {
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
        if let Some((time_lower, time_simulator, _, _)) = lower_results {
            println!("lower verilog:    {:?}", time_lower);
            println!("lower c++:        {:?}", time_simulator);
        }
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
