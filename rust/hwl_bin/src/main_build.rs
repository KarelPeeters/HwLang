use crate::args::ArgsBuild;
use crate::util::{ErrorExit, manifest_find_read_parse, print_diagnostics};
use hwl_language::back::lower_cpp::lower_to_cpp;
use hwl_language::back::lower_verilog::lower_to_verilog;
use hwl_language::front::compile::{COMPILE_THREAD_STACK_SIZE, ElaborationSet, compile};
use hwl_language::front::diagnostic::Diagnostics;
use hwl_language::front::print::StdoutPrintHandler;
use hwl_language::mid::cleanup::cleanup;
use hwl_language::syntax::collect::collect_source_from_manifest;
use hwl_language::syntax::hierarchy::HierarchyNode;
use hwl_language::syntax::manifest::Manifest;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::SourceDatabase;
use hwl_language::syntax::token::Tokenizer;
use hwl_language::util::arena::IndexType;
use hwl_language::util::{NON_ZERO_USIZE_ONE, ResultExt};
use std::num::NonZeroUsize;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub fn main_build(args: ArgsBuild) -> ExitCode {
    if args.keep_main_stack {
        main_build_inner(args)
    } else {
        // spawn a new thread with a larger stack size
        // TODO is this still needed? was it ever?
        std::thread::Builder::new()
            .stack_size(COMPILE_THREAD_STACK_SIZE)
            .spawn(|| main_build_inner(args))
            .unwrap()
            .join()
            .unwrap()
    }
}

fn main_build_inner(args: ArgsBuild) -> ExitCode {
    // interpret args
    let ArgsBuild {
        manifest,
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
    let start_source = Instant::now();

    // find and parse manifest
    let mut source = SourceDatabase::new();
    let manifest = match manifest_find_read_parse(&mut source, manifest) {
        Ok(m) => m,
        Err(ErrorExit) => return ExitCode::FAILURE,
    };
    let Manifest {
        source: manifest_source,
    } = manifest.parsed;

    // collect source
    let diags = Diagnostics::new();
    let (hierarchy, _) = match collect_source_from_manifest(
        &diags,
        &mut source,
        manifest.file,
        &manifest.path_parent,
        &manifest_source,
    ) {
        Ok(s) => s,
        Err(_) => {
            print_diagnostics(&source, diags);
            return ExitCode::FAILURE;
        }
    };
    let time_source = start_source.elapsed();

    // print source info
    if print_files {
        eprintln!("Collected sources:");
        for file in source.files() {
            let file_info = &source[file];
            eprintln!("  [{}]: {:?}", file.inner().index(), &file_info.debug_info_path,);
        }
        eprintln!("Collected hierarchy:");
        fn print_node(prefix: &str, node: &HierarchyNode) {
            if let Some(file) = node.file {
                eprintln!("{prefix}: [{:?}]", file.inner().index());
            }
            for (key, child) in &node.children {
                print_node(&format!("{prefix}/{key}"), child);
            }
        }
        print_node("  ", hierarchy.root_node());
    }

    // run compilation
    // TODO parallelize parsing
    let start_parse = Instant::now();
    let parsed = ParsedDatabase::new(&diags, &source, &hierarchy);
    let time_parse = start_parse.elapsed();

    let should_stop = Arc::new(AtomicBool::new(false));
    {
        let should_stop = should_stop.clone();
        ctrlc::set_handler(move || should_stop.store(true, Ordering::Relaxed)).expect("Failed to set Ctrl+C handler");
    }

    let start_compile = Instant::now();
    let mut compiled = compile(
        &diags,
        &source,
        &hierarchy,
        &parsed,
        elaboration_set,
        &mut StdoutPrintHandler,
        &|| should_stop.load(Ordering::Relaxed),
        thread_count,
        source.full_span(manifest.file),
    );
    let time_compile = start_compile.elapsed();

    let start_cleanup = Instant::now();
    if let Ok(compiled) = &mut compiled {
        cleanup(compiled);
    }
    let time_cleanup = start_cleanup.elapsed();

    // TODO don't hardcode paths here
    // TODO make this configurable
    // std::fs::write(
    //     "../ignored/lowered.ir",
    //     compiled
    //         .as_ref()
    //         .map_or("// failed".to_owned(), |s| format!("{:#?}", s)),
    // )
    // .unwrap();

    // TODO parallelize lowering?
    let lower_results = if skip_lower {
        None
    } else {
        let start_lower = Instant::now();
        let lowered = compiled
            .as_ref_ok()
            .and_then(|c| lower_to_verilog(&diags, &c.modules, &c.external_modules, c.top_module));
        let time_lower = start_lower.elapsed();

        let start_simulator = Instant::now();
        let simulator_code = compiled
            .as_ref_ok()
            .and_then(|c| lower_to_cpp(&diags, &c.modules, c.top_module));
        let time_simulator = start_simulator.elapsed();

        Some((time_lower, time_simulator, lowered, simulator_code))
    };

    let time_all = start_all.elapsed();

    // print results
    let any_error = print_diagnostics(&source, diags);

    if print_ir {
        eprintln!("{compiled:#?}");
    }

    // TODO don't hardcode paths here
    // TODO make this configurable
    if let Some((_, _, lowered, simulator_code)) = &lower_results {
        // save lowered verilog
        std::fs::create_dir_all("../ignored").unwrap();
        std::fs::write(
            "../ignored/lowered.v",
            lowered.as_ref().map_or("// failed", |s| &s.source),
        )
        .unwrap();

        // save simulator code
        std::fs::write(
            "../ignored/lowered.cpp",
            simulator_code.as_ref().map_or("// failed", String::as_str),
        )
        .unwrap();
    }

    // print profiling info
    // TODO expand to include separate profiling info for each elaborated item
    if profile {
        // profile tokenization separately
        let start_tokenize = Instant::now();
        let mut total_tokens = 0;
        for file in hierarchy.files() {
            total_tokens += Tokenizer::new(file, &source[file].content, false).into_iter().count();
        }
        let time_tokenize = start_tokenize.elapsed();

        eprintln!();
        eprintln!("profiling info:");
        eprintln!("-----------------------------------------------");
        eprintln!("input files:      {}", source.file_count());
        eprintln!("input lines:      {}", source.total_lines_of_code());
        eprintln!("input tokens:     {total_tokens}");
        eprintln!("----------------------------------------");
        eprintln!("read source:      {time_source:?}");
        eprintln!("tokenize:         {time_tokenize:?}");
        eprintln!("parse + tokenize: {time_parse:?}");
        eprintln!("compile:          {time_compile:?}");
        eprintln!("cleanup:          {time_cleanup:?}");
        if let Some((time_lower, time_simulator, _, _)) = lower_results {
            eprintln!("lower verilog:    {time_lower:?}");
            eprintln!("lower c++:        {time_simulator:?}");
        } else {
            eprintln!("lower verilog:    (skipped)");
            eprintln!("lower c++:        (skipped)");
        }
        eprintln!("-----------------------------------------------");
        eprintln!("total:            {time_all:?}");
        eprintln!();
    }

    // proper exit code
    if any_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
