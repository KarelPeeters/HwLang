use clap::Parser;
use hwl_language::back::lower_cpp::lower_to_cpp;
use hwl_language::back::lower_verilog::lower_to_verilog;
use hwl_language::front::compile::{compile, ElaborationSet, COMPILE_THREAD_STACK_SIZE};
use hwl_language::front::diagnostic::{DiagResult, DiagnosticStringSettings, Diagnostics};
use hwl_language::front::print::StdoutPrintHandler;
use hwl_language::syntax::hierarchy::{HierarchyNode, SourceHierarchy};
use hwl_language::syntax::manifest::{Manifest, ManifestSource, SourceEntry};
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FileId, SourceDatabase};
use hwl_language::syntax::token::Tokenizer;
use hwl_language::util::arena::IndexType;
use hwl_language::util::data::SliceExt;
use hwl_language::util::{ResultExt, NON_ZERO_USIZE_ONE};
use hwl_util::constants::{HWL_FILE_EXTENSION, HWL_MANIFEST_FILE_NAME};
use hwl_util::io::{recurse_for_each_file, IoErrorExt, IoErrorWithPath};
use itertools::chain;
use path_clean::PathClean;
use std::ffi::{OsStr, OsString};
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// TODO automatically disable this when miri is used
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
struct Args {
    // input data
    #[arg(long)]
    manifest: Option<PathBuf>,

    // performance options
    #[arg(long, short = 'j')]
    thread_count: Option<NonZeroUsize>,

    // debug options
    // TODO some of these have major effects and are not really debug options
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
    // TODO add a way to print all elaborated items and the instantiation tree
    let args = Args::parse();

    if args.keep_main_stack {
        main_inner(args)
    } else {
        // spawn a new thread with a larger stack size
        // TODO should we do this or switch to a heap stack everywhere (mostly item visiting and verilog lowering)
        std::thread::Builder::new()
            .stack_size(COMPILE_THREAD_STACK_SIZE)
            .spawn(|| main_inner(args))
            .unwrap()
            .join()
            .unwrap()
    }
}

fn main_inner(args: Args) -> ExitCode {
    // interpret args
    let Args {
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

    // find and parse manifest file
    let mut source = SourceDatabase::new();
    let found_manifest = match find_and_read_manifest(manifest) {
        Ok(m) => m,
        Err(FindManifestError(msg)) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    let manifest_file = source.add_file(
        found_manifest.manifest_path.to_string_lossy().into_owned(),
        found_manifest.manifest_source,
    );
    let manifest = match Manifest::from_toml(&source[manifest_file].source) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to parse manifest file: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Manifest {
        source: manifest_source,
    } = manifest;

    // collect source
    let diags = Diagnostics::new();
    let hierarchy = match find_and_collect_source(
        &diags,
        &mut source,
        manifest_file,
        &found_manifest.manifest_parent,
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
        print_node("  ", &hierarchy.root);
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
    let compiled = compile(
        &diags,
        &source,
        &hierarchy,
        &parsed,
        elaboration_set,
        &mut StdoutPrintHandler,
        &|| should_stop.load(Ordering::Relaxed),
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
            total_tokens += Tokenizer::new(file, &source[file].source, false).into_iter().count();
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

struct FoundManifest {
    manifest_path: PathBuf,
    manifest_parent: PathBuf,
    manifest_source: String,
}

struct FindManifestError(String);

fn find_and_read_manifest(manifest_path: Option<PathBuf>) -> Result<FoundManifest, FindManifestError> {
    let cwd = std::env::current_dir()
        .map_err(|e| FindManifestError(format!("Failed to get current working directory: {e:?}")))?;
    let cwd = std::path::absolute(&cwd).map_err(|e| {
        FindManifestError(format!(
            "Failed to convert working dir to absolute path: {:?}",
            e.with_path(cwd)
        ))
    })?;

    match manifest_path {
        Some(manifest_path) => {
            // directly read the manifest file
            let manifest_path = cwd.join(manifest_path).clean();
            match std::fs::read_to_string(&manifest_path) {
                Ok(s) => {
                    let manifest_parent = manifest_path
                        .parent()
                        .ok_or_else(|| {
                            FindManifestError(format!(
                                "Manifest path {manifest_path:?} does not have a parent directory"
                            ))
                        })?
                        .to_owned();
                    Ok(FoundManifest {
                        manifest_path,
                        manifest_parent,
                        manifest_source: s,
                    })
                }
                Err(e) => Err(FindManifestError(format!(
                    "Failed to read manifest file: {:?}",
                    e.with_path(manifest_path)
                ))),
            }
        }
        None => {
            // walk up the path until we find a folder containing a manifest file
            for ancestor in cwd.ancestors() {
                let cand_manifest_path = ancestor.join(HWL_MANIFEST_FILE_NAME);
                match std::fs::read_to_string(&cand_manifest_path) {
                    Ok(s) => {
                        return Ok(FoundManifest {
                            manifest_path: cand_manifest_path,
                            manifest_parent: ancestor.to_owned(),
                            manifest_source: s,
                        })
                    }
                    Err(e) => match e.kind() {
                        ErrorKind::NotFound => continue,
                        _ => {
                            return Err(FindManifestError(format!(
                                "Failed to read manifest file: {:?}",
                                e.with_path(cand_manifest_path)
                            )));
                        }
                    },
                }
            }

            Err(FindManifestError(format!(
                "No manifest file {HWL_MANIFEST_FILE_NAME} found in any parent directory of the current working directory {cwd:?}"
            )))
        }
    }
}

// TODO check that keys are valid identifiers
fn find_and_collect_source(
    diags: &Diagnostics,
    source: &mut SourceDatabase,
    // TODO get more detailed spans
    manifest_file: FileId,
    manifest_path: &Path,
    manifest: &ManifestSource,
) -> DiagResult<SourceHierarchy> {
    let mut hierarchy = SourceHierarchy::new();

    let manifest_span = source.full_span(manifest_file);
    let report_error = |m: String| diags.report_simple(m, manifest_span, "while collecting source here");

    for entry in manifest.entries() {
        let SourceEntry { steps, path_relative } = entry;
        let entry_path = manifest_path.join(path_relative).clean();

        let entry_meta = match entry_path.metadata() {
            Ok(m) => m,
            Err(e) => return Err(report_error(io_error_message(e.with_path(entry_path)))),
        };

        // collect the set of relevant files
        let mut files = vec![];
        if entry_meta.is_file() {
            // for single files we don't include the filename itself in the steps
            files.push((vec![], entry_path));
        } else {
            let mut step_err = Ok(());
            recurse_for_each_file(&entry_path, |relative_steps, path_file| {
                // short-circuit on error
                if step_err.is_err() {
                    return Ok(());
                }

                // filter by extension
                if path_file.extension() != Some(OsStr::new(HWL_FILE_EXTENSION)) {
                    return Ok(());
                }

                // add filename to steps
                let Some(file_stem) = path_file.file_stem() else {
                    return Ok(());
                };
                let all_steps = chain(
                    relative_steps.iter().map(OsString::as_os_str),
                    std::iter::once(file_stem),
                );

                // convert steps to strings
                let mut relative_steps_str = vec![];
                for step in all_steps {
                    match step.to_str() {
                        Some(step) => relative_steps_str.push(step.to_owned()),
                        None => {
                            step_err = Err(report_error(format!("Encountered non-UTF8 path {path_file:?}")));
                            return Ok(());
                        }
                    }
                }

                files.push((relative_steps_str, path_file.to_owned()));
                Ok(())
            })
            .map_err(|e| report_error(io_error_message(e)))?;

            step_err?;
        }

        // sort to ensure cross-platform determinism
        files.sort_by_key_ref(|(steps, _)| steps);

        // actually read in the files and add them to source and hierarchy
        // TODO maybe allow storing errors for non-UTF8 files into source?
        for (relative_steps, path) in files {
            let mut all_steps = steps.clone();
            all_steps.extend(relative_steps);

            let content = std::fs::read_to_string(&path)
                .map_err(|e| report_error(io_error_message(e.with_path(path.clone()))))?;
            let file = source.add_file(path.to_string_lossy().into_owned(), content);

            hierarchy.add_file(diags, source, manifest_span, &all_steps, file)?;
        }
    }

    Ok(hierarchy)
}

fn io_error_message(e: IoErrorWithPath) -> String {
    let IoErrorWithPath { error, path } = e;
    format!("IO error: {error:?} at {path:?}")
}

fn print_diagnostics(source: &SourceDatabase, diags: Diagnostics) -> bool {
    let diags = diags.finish();
    let any_error = !diags.is_empty();

    for diag in diags {
        let s = diag.to_string(&source, DiagnosticStringSettings::default());
        eprintln!("{s}\n");
    }

    any_error
}
