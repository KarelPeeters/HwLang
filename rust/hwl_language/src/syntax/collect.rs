use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::syntax::hierarchy::SourceHierarchy;
use crate::syntax::manifest::{ManifestSource, SourceEntry};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::util::data::SliceExt;
use hwl_util::constants::HWL_FILE_EXTENSION;
use hwl_util::io::{recurse_for_each_file, IoErrorExt, IoErrorWithPath};
use itertools::chain;
use path_clean::PathClean;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

pub fn collect_source_from_manifest(
    diags: &Diagnostics,
    source: &mut SourceDatabase,
    // TODO get more detailed spans
    manifest_file: FileId,
    manifest_path: &Path,
    manifest: &ManifestSource,
) -> DiagResult<SourceHierarchy> {
    let mut hierarchy = SourceHierarchy::new();

    let manifest_span = source.full_span(manifest_file);

    for entry in manifest.entries() {
        let SourceEntry { steps, path_relative } = entry;
        let entry_path = manifest_path.join(path_relative).clean();
        collect_source_from_tree(diags, source, &mut hierarchy, manifest_span, steps, entry_path)?;
    }

    Ok(hierarchy)
}

pub fn collect_source_from_tree(
    diags: &Diagnostics,
    source: &mut SourceDatabase,
    hierarchy: &mut SourceHierarchy,
    span: Span,
    steps: Vec<String>,
    entry_path: PathBuf,
) -> DiagResult<()> {
    let report_error = |m: String| diags.report_simple(m, span, "while collecting source here");

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

        // sort to ensure cross-platform determinism
        files.sort_by_key_ref(|(steps, _)| steps);
    }

    // actually read in the files and add them to source and hierarchy
    // TODO maybe allow storing errors for non-UTF8 files into source?
    for (relative_steps, path) in files {
        let mut all_steps = steps.clone();
        all_steps.extend(relative_steps);

        let content =
            std::fs::read_to_string(&path).map_err(|e| report_error(io_error_message(e.with_path(path.clone()))))?;
        let file = source.add_file(path.to_string_lossy().into_owned(), content);

        hierarchy.add_file(diags, source, span, &all_steps, file)?;
    }

    Ok(())
}

fn io_error_message(e: IoErrorWithPath) -> String {
    let IoErrorWithPath { error, path } = e;
    format!("IO error: {error:?} at {path:?}")
}
