use crate::engine::encode::encode_span_to_lsp;
use crate::engine::vfs::VfsResult;
use crate::server::document::abs_path_to_uri;
use crate::server::settings::PositionEncoding;
use crate::server::state::{OrSendError, RequestError, RequestResult, ServerState};
use annotate_snippets::Level;
use hwl_language::constants::{LANGUAGE_FILE_EXTENSION, LSP_SERVER_NAME};
use hwl_language::front::compile::{compile, NoPrintHandler, PrintHandler};
use hwl_language::front::diagnostic::{Annotation, Diagnostic, Diagnostics};
use hwl_language::front::lower_verilog::lower;
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::pos::FileId;
use hwl_language::syntax::source::{FilePath, SourceDatabase, SourceSetError};
use hwl_language::{throw, try_inner};
use indexmap::IndexMap;
use lsp_types::{
    notification, DiagnosticRelatedInformation, DiagnosticSeverity, Location, PublishDiagnosticsParams, Uri,
};
use std::cmp::Ordering;
use std::fmt::Write;
use std::path::{Component, PathBuf};

impl ServerState {
    pub fn compile_project_and_send_diagnostics(&mut self) -> Result<(), OrSendError<RequestError>> {
        if self.vfs.inner().map_err(RequestError::from)?.get_and_clear_changed() {
            self.log("file system changed, updating diagnostics");

            let (source, abs_path_map) = match self.build_source_database()? {
                Ok(v) => v,
                Err(e) => throw!(RequestError::Internal(format!(
                    "compile set error, should not be possible through VFS: {e:?}"
                ))),
            };

            self.log("source database built, compiling");
            let diags = Diagnostics::new();
            let parsed = ParsedDatabase::new(&diags, &source);
            let compiled = compile(&diags, &source, &parsed, &mut NoPrintHandler);
            let _ = lower(&diags, &source, &parsed, &compiled);
            self.log("compilation finished");

            // build new diagnostic set, combined per URI
            let mut diagnostics_per_uri: IndexMap<Uri, Vec<lsp_types::Diagnostic>> = IndexMap::new();
            for diagnostic in diags.finish() {
                let (uri, diag) =
                    diagnostic_to_lsp(self.settings.position_encoding, &source, &abs_path_map, diagnostic)
                        .map_err(|e| OrSendError::Error(e.into()))?;

                diagnostics_per_uri.entry(uri).or_default().push(diag);
            }

            // iterate over all files to ensure old diagnostics get cleared
            // TODO optimize this, only send diff
            for path in abs_path_map.values() {
                let uri = abs_path_to_uri(path).map_err(|e| OrSendError::Error(e.into()))?;
                let diagnostics = diagnostics_per_uri.swap_remove(&uri).unwrap_or_default();

                self.sender
                    .send_notification::<notification::PublishDiagnostics>(PublishDiagnosticsParams {
                        uri,
                        diagnostics,
                        // TODO version
                        version: None,
                    })
                    .map_err(OrSendError::SendError)?;
            }
        }

        Ok(())
    }

    pub fn build_source_database(
        &mut self,
    ) -> RequestResult<Result<(SourceDatabase, IndexMap<FileId, PathBuf>), SourceSetError>> {
        let vfs = self.vfs.inner()?;
        let vfs_root = vfs.root().clone();

        let mut source = SourceDatabase::new();
        let mut abs_path_map: IndexMap<FileId, PathBuf> = IndexMap::new();

        for (path, content) in vfs.iter() {
            let mut steps = vec![];
            for c in path.components() {
                match c {
                    Component::Normal(c) => steps.push(
                        c.to_str()
                            .expect("the vfs layer should already check for non-utf8 paths ")
                            .to_owned(),
                    ),
                    _ => unreachable!("the vfs path should be a simple relative path, got {:?}", path),
                }
            }

            // remove final extension
            let last = steps.last_mut().unwrap();
            *last = last
                .strip_suffix(&format!(".{LANGUAGE_FILE_EXTENSION}"))
                .expect("only language source files should be in the VFS")
                .to_owned();

            let text = content.get_text(path)?;
            let file_id =
                try_inner!(source.add_file(FilePath(steps), path.to_str().unwrap().to_owned(), text.to_owned()));
            abs_path_map.insert(file_id, vfs_root.join(path));
        }

        Ok(Ok((source, abs_path_map)))
    }
}

fn diagnostic_to_lsp(
    encoding: PositionEncoding,
    source: &SourceDatabase,
    abs_path_map: &IndexMap<FileId, PathBuf>,
    diagnostic: Diagnostic,
) -> VfsResult<(Uri, lsp_types::Diagnostic)> {
    let Diagnostic {
        title,
        snippets,
        footers,
        backtrace,
    } = diagnostic;

    // don't show backtrace in LSP, it's not intended for end-users
    let _ = backtrace;

    // find the file with the highest level annotation, that's probably the main one
    let mut top_annotation: Option<&Annotation> = None;
    for (_, annotations) in &snippets {
        for annotation in annotations {
            let is_better = match &top_annotation {
                None => true,
                // TODO better level comparison function
                Some(prev) => compare_level(annotation.level, prev.level).is_gt(),
            };
            if is_better {
                top_annotation = Some(annotation);
            }
        }
    }
    let top_annotation = top_annotation.unwrap();

    // do the actual conversion
    // TODO check client capabilities
    let mut related_information = vec![];
    for (_, annotations) in &snippets {
        for annotation in annotations {
            if annotation == top_annotation {
                continue;
            }
            let &Annotation { level, span, ref label } = annotation;

            let file = span.start.file;
            let file_info = &source[file];
            related_information.push(DiagnosticRelatedInformation {
                location: Location {
                    uri: abs_path_to_uri(abs_path_map.get(&file).unwrap())?,
                    range: encode_span_to_lsp(encoding, &file_info.offsets, &file_info.source, span),
                },
                message: format!("{}: {}", level_to_str(level), label),
            });
        }
    }

    let top_file = top_annotation.span.start.file;
    let top_file_info = &source[top_file];
    let top_uri = abs_path_to_uri(abs_path_map.get(&top_file).unwrap())?;

    let mut top_message = format!("{}\n{}", title, top_annotation.label);
    for (footer_level, footer_message) in footers {
        write!(&mut top_message, "\n{}: {}", level_to_str(footer_level), footer_message).unwrap();
    }

    let diag = lsp_types::Diagnostic {
        range: encode_span_to_lsp(
            encoding,
            &top_file_info.offsets,
            &top_file_info.source,
            top_annotation.span,
        ),
        severity: Some(level_to_severity(top_annotation.level)),
        code: None,
        code_description: None,
        source: Some(LSP_SERVER_NAME.to_owned()),
        message: top_message,
        related_information: Some(related_information),
        // TODO set tags once we support those
        tags: None,
        // TODO data for auto-fixes
        data: None,
    };

    // return
    // TODO maybe get the path name from the source map?
    Ok((top_uri, diag))
}

fn compare_level(left: Level, right: Level) -> Ordering {
    (left as u8).cmp(&(right as u8)).reverse()
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
