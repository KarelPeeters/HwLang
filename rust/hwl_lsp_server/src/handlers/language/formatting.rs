use crate::handlers::dispatch::RequestHandler;
use crate::server::state::{RequestError, RequestResult, ServerState};
use crate::util::encode::span_to_lsp;
use crate::util::uri::uri_to_path;
use hwl_language::front::diagnostic::{DiagError, Diagnostics, diags_to_string};
use hwl_language::syntax::format::{FormatError, FormatSettings, format_file};
use hwl_language::syntax::source::SourceDatabase;
use lsp_types::request::Formatting;
use lsp_types::{DocumentFormattingParams, TextDocumentIdentifier, TextEdit};

impl RequestHandler<Formatting> for ServerState {
    fn handle_request(&mut self, params: DocumentFormattingParams) -> RequestResult<Option<Vec<TextEdit>>> {
        let DocumentFormattingParams {
            text_document,
            options,
            work_done_progress_params: _,
        } = params;

        // we ignore options coming from the client,
        //   we might add formatting configurability later but that will be through the manifest file
        let _ = options;

        let TextDocumentIdentifier { uri } = text_document;
        let path = uri_to_path(&uri)?;
        let src = self.vfs.read_str_maybe_from_disk(&path)?;

        let diags = Diagnostics::new();
        let mut source = SourceDatabase::new();
        let file = source.add_file("dummy.kh".to_owned(), src.to_owned());
        let result = format_file(&diags, &source, &FormatSettings::default(), file);

        match result {
            Ok(result) => {
                // TODO compress diff into minimal set of edits?
                let full_range = span_to_lsp(
                    self.settings.position_encoding,
                    &source[file].offsets,
                    &source[file].content,
                    source.full_span(file),
                );
                let edit = TextEdit {
                    range: full_range,
                    new_text: result.new_content,
                };
                Ok(Some(vec![edit]))
            }
            Err(FormatError::Syntax(_)) => {
                // syntax errors are not atypical during formatting, just silently ignore them
                Ok(None)
            }
            Err(FormatError::Internal(e)) => {
                let _: DiagError = e;
                let diags = diags_to_string(&source, diags.finish(), false);
                Err(RequestError::Internal(format!("formatter internal error: {diags}")))
            }
        }
    }
}
