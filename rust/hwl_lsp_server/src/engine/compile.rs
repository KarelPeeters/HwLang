use crate::engine::encode::span_to_lsp;
use crate::engine::vfs::{VfsError, VfsResult};
use crate::server::sender::SendErrorOr;
use crate::server::settings::{PositionEncoding, Settings};
use crate::server::state::{RequestError, ServerState};
use crate::server::util::{uri_join, uri_to_path, watcher_any_file_with_name};
use annotate_snippets::Level;
use hwl_language::back::lower_verilog::lower_to_verilog;
use hwl_language::front::compile::{compile, ElaborationSet};
use hwl_language::front::diagnostic::{Annotation, Diagnostic, Diagnostics};
use hwl_language::front::print::NoPrintHandler;
use hwl_language::syntax::manifest::{Manifest, SourceEntry};
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FileId, SourceDatabase, SourceDatabaseBuilder, SourceSetOrIoError};
use hwl_language::util::NON_ZERO_USIZE_ONE;
use hwl_util::constants::{HWL_FILE_EXTENSION, HWL_LSP_NAME, HWL_MANIFEST_FILE_NAME};
use hwl_util::io::recurse_for_each_file;
use indexmap::{IndexMap, IndexSet};
use itertools::zip_eq;
use lsp_types::{
    notification, DiagnosticRelatedInformation, DiagnosticSeverity, Location, PublishDiagnosticsParams, Uri,
};
use std::ffi::OsStr;
use std::fmt::Write;

#[allow(dead_code)]
struct CollectedManifest {
    manifest_folder_uri: Uri,
    manifest: Manifest,
    source: SourceDatabase,
    file_to_uri: IndexMap<FileId, Uri>,
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
            self.log(format!(
                "compiling manifest {:?}",
                uri_join(&manifest.manifest_folder_uri, HWL_MANIFEST_FILE_NAME)
                    .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?
                    .to_string()
            ));

            let diags = Diagnostics::new();
            let source = &manifest.source;
            let parsed = ParsedDatabase::new(&diags, source);

            for file in source.files() {
                self.log(format!("  file {}", source[file].path_raw));
            }

            // TODO enable multithreading
            // TODO optionally also check C++ generation
            //   or maybe remove verilog instead, hopefully the backends get complete enough
            // TODO compile all items in the open files first, then later the rest for increased interactivity
            // TODO propagate prints to a separate output channel or log file
            // TODO enable multithreading
            let compiled = compile(
                &diags,
                source,
                &parsed,
                ElaborationSet::AsMuchAsPossible,
                &mut NoPrintHandler,
                &|| false,
                NON_ZERO_USIZE_ONE,
            );
            let _ = compiled.and_then(|c| lower_to_verilog(&diags, &c.modules, &c.external_modules, c.top_module));

            group_diagnostics(
                &mut grouped_diags,
                &self.settings,
                source,
                &manifest.file_to_uri,
                diags.finish(),
            )?;
        }

        self.log(format!(
            "sending diagnostics: {}",
            grouped_diags.values().map(|v| v.len()).sum::<usize>()
        ));
        self.send_diagnostics(grouped_diags)
    }

    fn collect_manifests(&mut self) -> Result<Vec<CollectedManifest>, SendErrorOr<RequestError>> {
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
            // skip non-manifest files
            if entry_path.file_name() != Some(OsStr::new(HWL_MANIFEST_FILE_NAME)) {
                return Ok(());
            }

            // record manifest for later
            let entry_parent_rel = entry_path.strip_prefix(&root_path).unwrap().parent().unwrap();
            let entry_parent_uri = uri_join(&root_uri, entry_parent_rel)?;
            manifest_folders.push(entry_parent_uri);
            Ok(())
        })
        .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

        // parse manifests and add watchers to their files
        let mut manifests = vec![];
        for manifest_folder_uri in &manifest_folders {
            let manifest_uri = uri_join(manifest_folder_uri, HWL_MANIFEST_FILE_NAME)
                .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;
            self.log(format!("reading manifest {:?}", manifest_uri.to_string()));

            let manifest_source = self
                .vfs
                .read_str_maybe_from_disk(&manifest_uri)
                .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

            // TODO record parsing error as diagnostic once SourceDatabase has been split
            //   and we can convert toml errors to diagnostics
            let manifest = match Manifest::from_toml(manifest_source) {
                Ok(manifest) => manifest,
                Err(e) => {
                    return Err(SendErrorOr::Other(RequestError::Internal(format!(
                        "failed to parse manifest: {e}"
                    ))));
                }
            };

            for entry in manifest.source.entries() {
                let SourceEntry {
                    steps: _,
                    path_relative,
                } = entry;

                let entry_uri = uri_join(manifest_folder_uri, path_relative)
                    .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;
                watchers.push(watcher_any_file_with_name(
                    entry_uri,
                    &format!("*.{HWL_FILE_EXTENSION}"),
                ));
            }

            manifests.push(manifest);
        }
        self.set_watchers(watchers)?;

        // collect sources
        // TODO fully share code with main binary
        let mut collected_manifests = vec![];
        for (manifest_folder_uri, manifest) in zip_eq(manifest_folders, manifests) {
            let mut source = SourceDatabaseBuilder::new();
            let mut file_to_uri = IndexMap::new();

            for entry in manifest.source.entries() {
                let SourceEntry {
                    steps: entry_steps,
                    path_relative: entry_path_relative,
                } = entry;
                let entry_uri = uri_join(&manifest_folder_uri, entry_path_relative)
                    .map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;
                let entry_path = uri_to_path(&entry_uri).map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;

                // TODO source set errors should become diagnostics, not LSP errors
                let file_to_path = source.add_tree(entry_steps, &entry_path).map_err(|e| match e {
                    SourceSetOrIoError::SourceSet(e) => SendErrorOr::Other(RequestError::SourceSet(e)),
                    SourceSetOrIoError::Io(io) => SendErrorOr::Other(RequestError::Vfs(VfsError::Io(io))),
                })?;

                // TODO rethink fileid/path/uri mapping, this is sketchy
                for (file_id, file_path) in file_to_path {
                    let file_path_rel = file_path.strip_prefix(&entry_path).unwrap();
                    let file_uri =
                        uri_join(&entry_uri, file_path_rel).map_err(|e| SendErrorOr::Other(RequestError::Vfs(e)))?;
                    self.log(format!(
                        "  file {file_id:?} -> {file_path:?} {file_path_rel:?} {}",
                        file_uri.as_str()
                    ));
                    file_to_uri.insert(file_id, file_uri);
                }
            }

            let (source, _, file_to_file) = source.finish_with_mapping();
            let file_to_uri = file_to_uri
                .into_iter()
                .map(|(k, v)| (*file_to_file.get(&k).unwrap(), v))
                .collect();

            let collected = CollectedManifest {
                manifest_folder_uri,
                manifest,
                source,
                file_to_uri,
            };
            collected_manifests.push(collected);
        }

        Ok(collected_manifests)
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
                    range: span_to_lsp(encoding, &file_info.offsets, &file_info.source, span),
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
            &top_file_info.source,
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
