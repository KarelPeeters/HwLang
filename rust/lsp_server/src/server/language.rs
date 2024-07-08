use language::syntax::parse_file_content;
use language::syntax::pos::FileId;
use log::error;
use tower_lsp::jsonrpc;
use tower_lsp::jsonrpc::Error;
use tower_lsp::lsp_types::{CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem, CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams, CodeAction, CodeActionParams, CodeActionResponse, CodeLens, CodeLensParams, ColorInformation, ColorPresentation, ColorPresentationParams, CompletionItem, CompletionItemKind, CompletionParams, CompletionResponse, DocumentColorParams, DocumentDiagnosticParams, DocumentDiagnosticReportResult, DocumentFormattingParams, DocumentHighlight, DocumentHighlightKind, DocumentHighlightParams, DocumentLink, DocumentLinkParams, DocumentOnTypeFormattingParams, DocumentRangeFormattingParams, DocumentSymbolParams, DocumentSymbolResponse, FoldingRange, FoldingRangeParams, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverParams, InlayHint, InlayHintParams, InlineValue, InlineValueParams, LinkedEditingRangeParams, LinkedEditingRanges, Location, MessageType, Moniker, MonikerParams, Position, PrepareRenameResponse, Range, ReferenceParams, RenameParams, SelectionRange, SelectionRangeParams, SemanticTokensDeltaParams, SemanticTokensFullDeltaResult, SemanticTokensParams, SemanticTokensRangeParams, SemanticTokensRangeResult, SemanticTokensResult, SignatureHelp, SignatureHelpParams, TextDocumentIdentifier, TextDocumentPositionParams, TextEdit, TypeHierarchyItem, TypeHierarchyPrepareParams, TypeHierarchySubtypesParams, TypeHierarchySupertypesParams, WorkspaceDiagnosticParams, WorkspaceDiagnosticReportResult, WorkspaceEdit};
use tower_lsp::lsp_types::request::{GotoDeclarationParams, GotoDeclarationResponse, GotoImplementationParams, GotoImplementationResponse, GotoTypeDefinitionParams, GotoTypeDefinitionResponse};

use crate::server::core::ServerCore;
use crate::server::util::ToLsp;

impl ServerCore {
    pub async fn goto_declaration(&self, _params: GotoDeclarationParams) -> jsonrpc::Result<Option<GotoDeclarationResponse>> {
        error!("Got a textDocument/declaration request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn goto_definition(&self, params: GotoDefinitionParams) -> jsonrpc::Result<Option<GotoDefinitionResponse>> {
        Ok(Some(GotoDefinitionResponse::Scalar(Location {
            uri: params.text_document_position_params.text_document.uri,
            range: Range::new(Position::new(0, 0), Position::new(0, 8)),
        })))
        
        // error!("Got a textDocument/definition request, but it is not implemented");
        // Err(Error::method_not_found())
    }

    pub async fn goto_type_definition(&self, _params: GotoTypeDefinitionParams) -> jsonrpc::Result<Option<GotoTypeDefinitionResponse>> {
        error!("Got a textDocument/typeDefinition request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn goto_implementation(&self, _params: GotoImplementationParams) -> jsonrpc::Result<Option<GotoImplementationResponse>> {
        error!("Got a textDocument/implementation request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn references(&self, _params: ReferenceParams) -> jsonrpc::Result<Option<Vec<Location>>> {
        error!("Got a textDocument/references request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn prepare_call_hierarchy(&self, _params: CallHierarchyPrepareParams) -> jsonrpc::Result<Option<Vec<CallHierarchyItem>>> {
        error!("Got a textDocument/prepareCallHierarchy request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn incoming_calls(&self, _params: CallHierarchyIncomingCallsParams) -> jsonrpc::Result<Option<Vec<CallHierarchyIncomingCall>>> {
        error!("Got a callHierarchy/incomingCalls request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn outgoing_calls(&self, _params: CallHierarchyOutgoingCallsParams) -> jsonrpc::Result<Option<Vec<CallHierarchyOutgoingCall>>> {
        error!("Got a callHierarchy/outgoingCalls request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn prepare_type_hierarchy(&self, _params: TypeHierarchyPrepareParams) -> jsonrpc::Result<Option<Vec<TypeHierarchyItem>>> {
        error!("Got a textDocument/prepareTypeHierarchy request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn supertypes(&self, _params: TypeHierarchySupertypesParams) -> jsonrpc::Result<Option<Vec<TypeHierarchyItem>>> {
        error!("Got a typeHierarchy/supertypes request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn subtypes(&self, _params: TypeHierarchySubtypesParams) -> jsonrpc::Result<Option<Vec<TypeHierarchyItem>>> {
        error!("Got a typeHierarchy/subtypes request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn document_highlight(&self, params: DocumentHighlightParams) -> jsonrpc::Result<Option<Vec<DocumentHighlight>>> {
        self.log_params("document_highlight", &params).await;
        error!("Got a textDocument/documentHighlight request, but it is not implemented");

        let DocumentHighlightParams { text_document_position_params, work_done_progress_params:  _, partial_result_params: _ } = params;
        let TextDocumentPositionParams { text_document, position: _ } = text_document_position_params;
        let TextDocumentIdentifier { uri } = text_document;

        let state = self.state.lock().await;
        let source = match state.documents.get(&uri) {
            Some(source) => source,
            None => return Ok(None),
        };

        let ast = match parse_file_content(FileId(0), source) {
            Ok(ast) => ast,
            Err(_) => return Ok(None),
        };

        let mut result = vec![];
        for item in ast.items {
            let range = item.span().to_lsp();
            self.log_info(format!("sending range {range:?}")).await;
            result.push(DocumentHighlight { range, kind: Some(DocumentHighlightKind::TEXT) });
        }

        Ok(Some(result))
    }

    pub async fn document_link(&self, _params: DocumentLinkParams) -> jsonrpc::Result<Option<Vec<DocumentLink>>> {
        error!("Got a textDocument/documentLink request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn document_link_resolve(&self, _params: DocumentLink) -> jsonrpc::Result<DocumentLink> {
        error!("Got a documentLink/resolve request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn hover(&self, _params: HoverParams) -> jsonrpc::Result<Option<Hover>> {
        error!("Got a textDocument/hover request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn code_lens(&self, _params: CodeLensParams) -> jsonrpc::Result<Option<Vec<CodeLens>>> {
        error!("Got a textDocument/codeLens request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn code_lens_resolve(&self, _params: CodeLens) -> jsonrpc::Result<CodeLens> {
        error!("Got a codeLens/resolve request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn folding_range(&self, _params: FoldingRangeParams) -> jsonrpc::Result<Option<Vec<FoldingRange>>> {
        error!("Got a textDocument/foldingRange request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn selection_range(&self, _params: SelectionRangeParams) -> jsonrpc::Result<Option<Vec<SelectionRange>>> {
        error!("Got a textDocument/selectionRange request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn document_symbol(&self, _params: DocumentSymbolParams) -> jsonrpc::Result<Option<DocumentSymbolResponse>> {
        error!("Got a textDocument/documentSymbol request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn semantic_tokens_full(&self, _params: SemanticTokensParams) -> jsonrpc::Result<Option<SemanticTokensResult>> {
        error!("Got a textDocument/semanticTokens/full request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn semantic_tokens_full_delta(&self, _params: SemanticTokensDeltaParams) -> jsonrpc::Result<Option<SemanticTokensFullDeltaResult>> {
        error!("Got a textDocument/semanticTokens/full/delta request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn semantic_tokens_range(&self, _params: SemanticTokensRangeParams) -> jsonrpc::Result<Option<SemanticTokensRangeResult>> {
        error!("Got a textDocument/semanticTokens/range request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn inline_value(&self, _params: InlineValueParams) -> jsonrpc::Result<Option<Vec<InlineValue>>> {
        error!("Got a textDocument/inlineValue request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn inlay_hint(&self, _params: InlayHintParams) -> jsonrpc::Result<Option<Vec<InlayHint>>> {
        error!("Got a textDocument/inlayHint request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn inlay_hint_resolve(&self, _params: InlayHint) -> jsonrpc::Result<InlayHint> {
        error!("Got a inlayHint/resolve request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn moniker(&self, _params: MonikerParams) -> jsonrpc::Result<Option<Vec<Moniker>>> {
        error!("Got a textDocument/moniker request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn completion(&self, params: CompletionParams) -> jsonrpc::Result<Option<CompletionResponse>> {
        let _ = params;
        self.client
            .log_message(MessageType::INFO, format!("completion({:?})", params))
            .await;

        // TODO actually populate, and check if there are more useful fields to populate
        let items = vec![
            CompletionItem {
                label: "custom_completion_text".to_string(),
                label_details: None,
                kind: Some(CompletionItemKind::TEXT),
                ..CompletionItem::default()
            },
            CompletionItem {
                label: "custom_completion_text_method".to_string(),
                label_details: None,
                kind: Some(CompletionItemKind::METHOD),
                ..CompletionItem::default()
            },
        ];

        Ok(Some(CompletionResponse::Array(items)))
    }

    pub async fn completion_resolve(&self, params: CompletionItem) -> jsonrpc::Result<CompletionItem> {
        // TODO fill in more details
        Ok(params)
    }

    pub async fn diagnostic(&self, _params: DocumentDiagnosticParams) -> jsonrpc::Result<DocumentDiagnosticReportResult> {
        error!("Got a textDocument/diagnostic request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn workspace_diagnostic(&self, _params: WorkspaceDiagnosticParams) -> jsonrpc::Result<WorkspaceDiagnosticReportResult> {
        error!("Got a workspace/diagnostic request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn signature_help(&self, _params: SignatureHelpParams) -> jsonrpc::Result<Option<SignatureHelp>> {
        error!("Got a textDocument/signatureHelp request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn code_action(&self, _params: CodeActionParams) -> jsonrpc::Result<Option<CodeActionResponse>> {
        error!("Got a textDocument/codeAction request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn code_action_resolve(&self, _params: CodeAction) -> jsonrpc::Result<CodeAction> {
        error!("Got a codeAction/resolve request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn document_color(&self, _params: DocumentColorParams) -> jsonrpc::Result<Vec<ColorInformation>> {
        error!("Got a textDocument/documentColor request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn color_presentation(&self, _params: ColorPresentationParams) -> jsonrpc::Result<Vec<ColorPresentation>> {
        error!("Got a textDocument/colorPresentation request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn formatting(&self, _params: DocumentFormattingParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> {
        error!("Got a textDocument/formatting request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn range_formatting(&self, _params: DocumentRangeFormattingParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> {
        error!("Got a textDocument/rangeFormatting request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn on_type_formatting(&self, _params: DocumentOnTypeFormattingParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> {
        error!("Got a textDocument/onTypeFormatting request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn rename(&self, _params: RenameParams) -> jsonrpc::Result<Option<WorkspaceEdit>> {
        error!("Got a textDocument/rename request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn prepare_rename(&self, _params: TextDocumentPositionParams) -> jsonrpc::Result<Option<PrepareRenameResponse>> {
        error!("Got a textDocument/prepareRename request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn linked_editing_range(&self, _params: LinkedEditingRangeParams) -> jsonrpc::Result<Option<LinkedEditingRanges>> {
        error!("Got a textDocument/linkedEditingRange request, but it is not implemented");
        Err(Error::method_not_found())
    }
}