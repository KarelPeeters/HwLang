use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::syntax::hierarchy::SourceHierarchy;
use crate::syntax::manifest::{ManifestSource, SourceEntry};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use hwl_util::constants::HWL_FILE_EXTENSION;
use hwl_util::io::{IoErrorExt, IoErrorWithPath, recurse_for_each_file};
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

        let files = collect_source_files_from_tree(diags, manifest_span, entry_path)?;

        add_source_files_to_tree(
            diags,
            source,
            &mut hierarchy,
            manifest_span,
            steps.clone(),
            files,
            |path| std::fs::read_to_string(path).map_err(|e| io_error_message(e.with_path(path.to_owned()))),
        )?;
    }

    Ok(hierarchy)
}

pub fn add_source_files_to_tree(
    diags: &Diagnostics,
    source: &mut SourceDatabase,
    hierarchy: &mut SourceHierarchy,
    manifest_span: Span,
    steps: Vec<String>,
    files: Vec<(Vec<String>, PathBuf)>,
    mut read: impl FnMut(&Path) -> Result<String, String>,
) -> DiagResult<Vec<FileId>> {
    let mut file_ids = vec![];

    // TODO maybe allow storing errors for non-UTF8 files into source?
    for (relative_steps, path) in files {
        let mut all_steps = steps.clone();
        all_steps.extend(relative_steps);

        let content =
            read(&path).map_err(|e| diags.report_simple(e, manifest_span, "while reading here files here"))?;

        let file = source.add_file(path.to_string_lossy().into_owned(), content);
        hierarchy.add_file(diags, source, manifest_span, &all_steps, file)?;

        file_ids.push(file);
    }

    Ok(file_ids)
}

pub fn collect_source_files_from_tree(
    diags: &Diagnostics,
    span: Span,
    entry_path: PathBuf,
) -> DiagResult<Vec<(Vec<String>, PathBuf)>> {
    let report_error = |m: String| diags.report_simple(m, span, "while collecting source here");

    let meta =
        std::fs::metadata(&entry_path).map_err(|e| report_error(io_error_message(e.with_path(entry_path.clone()))))?;
    if meta.is_file() {
        return Ok(vec![(vec![], entry_path)]);
    }

    let mut files = vec![];
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

    Ok(files)
}

pub fn io_error_message(e: IoErrorWithPath) -> String {
    let IoErrorWithPath { error, path } = e;
    format!("IO error: {error:?} at {path:?}")
}
