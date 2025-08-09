use crate::engine::encode::span_to_lsp;
use crate::engine::vfs::{Vfs, VfsResult};
use crate::server::sender::SendErrorOr;
use crate::server::settings::{PositionEncoding, Settings};
use crate::server::state::{RequestError, ServerState};
use crate::server::util::{abs_path_to_uri, uri_to_path, watcher_any_file_with_name};
use annotate_snippets::Level;
use hwl_language::back::lower_verilog::lower_to_verilog;
use hwl_language::front::compile::{ElaborationSet, compile};
use hwl_language::front::diagnostic::{Annotation, DiagResult, Diagnostic, Diagnostics};
use hwl_language::front::print::NoPrintHandler;
use hwl_language::syntax::collect::{add_source_files_to_tree, collect_source_files_from_tree};
use hwl_language::syntax::hierarchy::SourceHierarchy;
use hwl_language::syntax::manifest::{Manifest, SourceEntry};
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FileId, SourceDatabase};
use hwl_language::util::NON_ZERO_USIZE_ONE;
use hwl_util::constants::{HWL_FILE_EXTENSION, HWL_LSP_NAME, HWL_MANIFEST_FILE_NAME};
use hwl_util::io::recurse_for_each_file;
use indexmap::{IndexMap, IndexSet};
use itertools::{Itertools, zip_eq};
use lsp_types::{
    DiagnosticRelatedInformation, DiagnosticSeverity, Location, PublishDiagnosticsParams, Uri, notification,
};
use std::ffi::OsStr;
use std::fmt::Write;
use std::path::PathBuf;

struct ManifestCommon {
    diags: Diagnostics,
    source: SourceDatabase,
    manifest_folder: PathBuf,
    manifest_file: FileId,
    file_to_path: IndexMap<FileId, PathBuf>,
}

struct ManifestPartial {
    common: ManifestCommon,
    manifest: DiagResult<Manifest>,
}

struct ManifestCollected {
    common: ManifestCommon,
    inner: DiagResult<ManifestCollectedInner>,
}

struct ManifestCollectedInner {
    #[allow(dead_code)]
    manifest: Manifest,
    hierarchy: SourceHierarchy,
}

impl ServerState {
    pub fn compile_project_and_send_diagnostics(&mut self) -> Result<(), SendErrorOr<RequestError>> {
        // if nothing has changed we don't need to do any work
        if !self.vfs.get_and_clear_changed() {
            return Ok(());
        }

        // TODO parallel loop over everything?
        self.log("compile: building source database");
        let manifests = self.collect_manifests()?;

        // TODO multithread this on a shared thread pool
        let mut grouped_diags = IndexMap::new();

        for manifest in manifests {
            let diags = manifest.common.diags;
            let source = &manifest.common.source;

            if let Ok(inner) = &manifest.inner {
                let hierarchy = &inner.hierarchy;
                let parsed = ParsedDatabase::new(&diags, source, hierarchy);

                // TODO enable multithreading
                // TODO optionally also check C++ generation
                //   or maybe remove verilog instead, hopefully the backends get complete enough
                // TODO compile all items in the open files first, then later the rest for increased interactivity
                // TODO propagate prints to a separate output channel or log file
                // TODO enable multithreading
                let compiled = compile(
                    &diags,
                    source,
                    hierarchy,
                    &parsed,
                    ElaborationSet::AsMuchAsPossible,
                    &mut NoPrintHandler,
                    &|| false,
                    NON_ZERO_USIZE_ONE,
                    source.full_span(manifest.common.manifest_file),
                );
                let _ = compiled.and_then(|c| lower_to_verilog(&diags, &c.modules, &c.external_modules, c.top_module));
            }

            let file_to_uri = manifest
                .common
                .file_to_path
                .into_iter()
                .map(|(k, b)| Ok((k, abs_path_to_uri(&b)?)))
                .try_collect()
                .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

            group_diagnostics(&mut grouped_diags, &self.settings, source, &file_to_uri, diags.finish())?;
        }

        self.log(format!(
            "sending diagnostics: {}",
            grouped_diags.values().map(|v| v.len()).sum::<usize>()
        ));
        self.send_diagnostics(grouped_diags)
    }

    fn collect_manifests(&mut self) -> Result<Vec<ManifestCollected>, SendErrorOr<RequestError>> {
        // TODO support multiple workspaces, root_path, no workspace
        //   also register change listeners for workspaces when we add support for them
        #[allow(deprecated)]
        let root_uri = self
            .settings
            .initialize_params
            .root_uri
            .as_ref()
            .ok_or_else(|| {
                SendErrorOr::Other(RequestError::Internal(
                    "no root URI set in initialize params".to_string(),
                ))
            })?
            .clone();
        let root_path = uri_to_path(&root_uri).map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

        // start collecting watchers, they need to be set before reading the corresponding files
        // TODO this is_empty stuff is a bit weird
        let mut watchers = vec![watcher_any_file_with_name(root_uri.clone(), HWL_MANIFEST_FILE_NAME)];
        if self.curr_watchers.is_empty() {
            self.set_watchers(watchers.clone())?;
        }

        // find manifests
        let mut manifest_folders = vec![];
        recurse_for_each_file(&root_path, |_, entry_path| {
            if entry_path.file_name() != Some(OsStr::new(HWL_MANIFEST_FILE_NAME)) {
                return Ok(());
            }
            manifest_folders.push(entry_path.parent().unwrap().to_owned());
            Ok(())
        })
        .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

        // parse manifests and add watchers to their files
        let mut manifests_partial: Vec<ManifestPartial> = vec![];
        for manifest_folder in manifest_folders {
            let manifest_path = manifest_folder.join(HWL_MANIFEST_FILE_NAME);

            let manifest_source = self
                .vfs
                .read_str_maybe_from_disk(&manifest_path)
                .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

            let diags = Diagnostics::new();
            let mut source = SourceDatabase::new();
            let mut file_to_path = IndexMap::new();

            let manifest_file =
                source.add_file(manifest_path.to_string_lossy().into_owned(), manifest_source.to_owned());
            file_to_path.insert(manifest_file, manifest_path);

            let manifest = Manifest::parse_toml(&diags, &source, manifest_file);

            if let Ok(manifest) = &manifest {
                for entry in manifest.source.entries() {
                    let SourceEntry {
                        steps: _,
                        path_relative,
                    } = entry;
                    let entry_path = manifest_folder.join(path_relative);
                    watchers.push(watcher_any_file_with_name(
                        abs_path_to_uri(&entry_path).map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?,
                        &format!("*.{HWL_FILE_EXTENSION}"),
                    ));
                }
            }

            let common = ManifestCommon {
                diags,
                source,
                manifest_folder,
                manifest_file,
                file_to_path,
            };
            manifests_partial.push(ManifestPartial { common, manifest });
        }
        self.set_watchers(watchers)?;

        // collect sources
        // TODO fully share as much code as possible with the main binary
        let mut manifests_collected = vec![];
        for manifest in manifests_partial {
            let ManifestPartial { mut common, manifest } = manifest;
            let inner = manifest.and_then(|manifest| collect_partial_manifest(&mut self.vfs, &mut common, manifest));
            manifests_collected.push(ManifestCollected { common, inner });
        }

        Ok(manifests_collected)
    }

    fn send_diagnostics(
        &mut self,
        grouped: IndexMap<Uri, Vec<lsp_types::Diagnostic>>,
    ) -> Result<(), SendErrorOr<RequestError>> {
        let mut prev_files_leftover = std::mem::take(&mut self.curr_files_with_diagnostics);
        let mut next_files = IndexSet::new();

        // send new diagnostics
        for (uri, diags) in grouped {
            if diags.is_empty() {
                continue;
            }

            prev_files_leftover.swap_remove(&uri);
            next_files.insert(uri.clone());

            self.sender
                .send_notification::<notification::PublishDiagnostics>(PublishDiagnosticsParams {
                    uri: uri.clone(),
                    diagnostics: diags,
                    // TODO version
                    version: None,
                })?;
        }

        // clear files that no longer have diagnostics
        for uri in prev_files_leftover {
            self.sender
                .send_notification::<notification::PublishDiagnostics>(PublishDiagnosticsParams {
                    uri: uri.clone(),
                    diagnostics: vec![],
                    // TODO version
                    version: None,
                })?;
        }

        // store current files with diagnostics
        self.curr_files_with_diagnostics = next_files;

        Ok(())
    }
}

fn collect_partial_manifest(
    vfs: &mut Vfs,
    common: &mut ManifestCommon,
    manifest: Manifest,
) -> DiagResult<ManifestCollectedInner> {
    let diags = &common.diags;
    let source = &mut common.source;
    let mut hierarchy = SourceHierarchy::new();

    // TODO get more precise span
    let manifest_span = source.full_span(common.manifest_file);

    for entry in manifest.source.entries() {
        let SourceEntry { steps, path_relative } = entry;
        let entry_path = common.manifest_folder.join(&path_relative);

        // TODO don't early exit, visit all entries before returning error
        // TODO insert extra files from Vfs that don't yet exist on disk
        let files = collect_source_files_from_tree(diags, manifest_span, entry_path)?;

        let file_paths = files.iter().map(|(_, p)| p.to_owned()).collect_vec();

        let file_ids = add_source_files_to_tree(
            diags,
            source,
            &mut hierarchy,
            manifest_span,
            steps,
            files,
            // TODO this is sketchy
            |file_path| match vfs.read_str_maybe_from_disk(file_path) {
                Ok(s) => Ok(s.to_owned()),
                Err(e) => Err(format!("{e:?}")),
            },
        )?;

        for (file_id, file_path) in zip_eq(file_ids, file_paths) {
            common.file_to_path.insert(file_id, file_path);
        }
    }

    Ok(ManifestCollectedInner { manifest, hierarchy })
}

fn group_diagnostics(
    grouped: &mut IndexMap<Uri, Vec<lsp_types::Diagnostic>>,
    settings: &Settings,
    source: &SourceDatabase,
    file_to_uri: &IndexMap<FileId, Uri>,
    diags: Vec<Diagnostic>,
) -> Result<(), SendErrorOr<RequestError>> {
    for diagnostic in diags {
        let (file, diag) = diagnostic_to_lsp(settings.position_encoding, source, file_to_uri, diagnostic)
            .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;
        let uri = file_to_uri.get(&file).unwrap().clone();
        grouped.entry(uri).or_default().push(diag);
    }
    Ok(())
}

fn diagnostic_to_lsp(
    encoding: PositionEncoding,
    source: &SourceDatabase,
    file_to_uri: &IndexMap<FileId, Uri>,
    diagnostic: Diagnostic,
) -> VfsResult<(FileId, lsp_types::Diagnostic)> {
    let Diagnostic {
        title,
        snippets,
        footers,
        backtrace,
    } = &diagnostic;

    // don't show backtrace in LSP, it's not intended for end-users
    let _ = backtrace;
    let top_annotation = diagnostic.main_annotation().unwrap();

    // do the actual conversion
    // TODO check client capabilities
    let mut related_information = vec![];
    for (_, annotations) in snippets {
        for annotation in annotations {
            if annotation == top_annotation {
                continue;
            }
            let &Annotation { level, span, ref label } = annotation;

            let file = span.file;
            let file_info = &source[file];
            related_information.push(DiagnosticRelatedInformation {
                location: Location {
                    uri: file_to_uri.get(&file).unwrap().clone(),
                    range: span_to_lsp(encoding, &file_info.offsets, &file_info.content, span),
                },
                message: format!("{}: {}", level_to_str(level), label),
            });
        }
    }

    let top_file = top_annotation.span.file;
    let top_file_info = &source[top_file];

    let mut top_message = format!("{}\n{}", title, top_annotation.label);
    for &(footer_level, ref footer_message) in footers {
        write!(&mut top_message, "\n{}: {}", level_to_str(footer_level), footer_message).unwrap();
    }

    let diag = lsp_types::Diagnostic {
        range: span_to_lsp(
            encoding,
            &top_file_info.offsets,
            &top_file_info.content,
            top_annotation.span,
        ),
        severity: Some(level_to_severity(top_annotation.level)),
        code: None,
        code_description: None,
        source: Some(HWL_LSP_NAME.to_owned()),
        message: top_message,
        related_information: Some(related_information),
        // TODO set tags once we support those
        tags: None,
        // TODO data for auto-fixes
        data: None,
    };

    // return
    // TODO maybe get the path name from the source map?
    Ok((top_file, diag))
}

fn level_to_severity(level: Level) -> DiagnosticSeverity {
    match level {
        Level::Error => DiagnosticSeverity::ERROR,
        Level::Warning => DiagnosticSeverity::WARNING,
        Level::Info => DiagnosticSeverity::INFORMATION,
        Level::Note => DiagnosticSeverity::INFORMATION,
        Level::Help => DiagnosticSeverity::HINT,
    }
}

fn level_to_str(level: Level) -> &'static str {
    match level {
        Level::Error => "error",
        Level::Warning => "warning",
        Level::Info => "info",
        Level::Note => "note",
        Level::Help => "help",
    }
}
