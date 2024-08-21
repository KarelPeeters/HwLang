use serde_json::Value;
use tower_lsp::lsp_types::request::{GotoDeclarationParams, GotoDeclarationResponse, GotoImplementationParams, GotoImplementationResponse, GotoTypeDefinitionParams, GotoTypeDefinitionResponse};
use tower_lsp::lsp_types::{CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem, CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams, CodeAction, CodeActionParams, CodeActionResponse, CodeLens, CodeLensParams, ColorInformation, ColorPresentation, ColorPresentationParams, CompletionItem, CompletionParams, CompletionResponse, CreateFilesParams, DeleteFilesParams, DidChangeConfigurationParams, DidChangeTextDocumentParams, DidChangeWatchedFilesParams, DidChangeWorkspaceFoldersParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, DocumentColorParams, DocumentDiagnosticParams, DocumentDiagnosticReportResult, DocumentFormattingParams, DocumentHighlight, DocumentHighlightParams, DocumentLink, DocumentLinkParams, DocumentOnTypeFormattingParams, DocumentRangeFormattingParams, DocumentSymbolParams, DocumentSymbolResponse, ExecuteCommandParams, FoldingRange, FoldingRangeParams, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverParams, InitializeParams, InitializeResult, InitializedParams, InlayHint, InlayHintParams, InlineValue, InlineValueParams, LinkedEditingRangeParams, LinkedEditingRanges, Location, Moniker, MonikerParams, PrepareRenameResponse, ReferenceParams, RenameFilesParams, RenameParams, SelectionRange, SelectionRangeParams, SemanticTokensDeltaParams, SemanticTokensFullDeltaResult, SemanticTokensParams, SemanticTokensRangeParams, SemanticTokensRangeResult, SemanticTokensResult, SignatureHelp, SignatureHelpParams, SymbolInformation, TextDocumentPositionParams, TextEdit, TypeHierarchyItem, TypeHierarchyPrepareParams, TypeHierarchySubtypesParams, TypeHierarchySupertypesParams, WillSaveTextDocumentParams, WorkspaceDiagnosticParams, WorkspaceDiagnosticReportResult, WorkspaceEdit, WorkspaceSymbol, WorkspaceSymbolParams};
use tower_lsp::{jsonrpc, LanguageServer};

use crate::server::core::ServerCore;

// TODO stop using async (and tower-LSP, it mostly just complicates core with not much possible upside)

/// Wrapper type to allow for splitting the [LanguageServer] trait implementation
/// across multiple `impl` blocks in different files.
#[derive(Debug)]
pub struct ServerWrapper {
    pub core: ServerCore,
}

#[tower_lsp::async_trait]
impl LanguageServer for ServerWrapper {
    async fn initialize(&self, params: InitializeParams) -> jsonrpc::Result<InitializeResult> { self.core.initialize(params).await }
    async fn initialized(&self, params: InitializedParams) { self.core.initialized(params).await }
    async fn shutdown(&self) -> jsonrpc::Result<()> { self.core.shutdown().await }
    async fn did_open(&self, params: DidOpenTextDocumentParams) { self.core.did_open(params).await }
    async fn did_change(&self, params: DidChangeTextDocumentParams) { self.core.did_change(params).await }
    async fn will_save(&self, params: WillSaveTextDocumentParams) { self.core.will_save(params).await }
    async fn will_save_wait_until(&self, params: WillSaveTextDocumentParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> { self.core.will_save_wait_until(params).await }
    async fn did_save(&self, params: DidSaveTextDocumentParams) { self.core.did_save(params).await }
    async fn did_close(&self, params: DidCloseTextDocumentParams) { self.core.did_close(params).await }
    async fn goto_declaration(&self, params: GotoDeclarationParams) -> jsonrpc::Result<Option<GotoDeclarationResponse>> { self.core.goto_declaration(params).await }
    async fn goto_definition(&self, params: GotoDefinitionParams) -> jsonrpc::Result<Option<GotoDefinitionResponse>> { self.core.goto_definition(params).await }
    async fn goto_type_definition(&self, params: GotoTypeDefinitionParams) -> jsonrpc::Result<Option<GotoTypeDefinitionResponse>> { self.core.goto_type_definition(params).await }
    async fn goto_implementation(&self, params: GotoImplementationParams) -> jsonrpc::Result<Option<GotoImplementationResponse>> { self.core.goto_implementation(params).await }
    async fn references(&self, params: ReferenceParams) -> jsonrpc::Result<Option<Vec<Location>>> { self.core.references(params).await }
    async fn prepare_call_hierarchy(&self, params: CallHierarchyPrepareParams) -> jsonrpc::Result<Option<Vec<CallHierarchyItem>>> { self.core.prepare_call_hierarchy(params).await }
    async fn incoming_calls(&self, params: CallHierarchyIncomingCallsParams) -> jsonrpc::Result<Option<Vec<CallHierarchyIncomingCall>>> { self.core.incoming_calls(params).await }
    async fn outgoing_calls(&self, params: CallHierarchyOutgoingCallsParams) -> jsonrpc::Result<Option<Vec<CallHierarchyOutgoingCall>>> { self.core.outgoing_calls(params).await }
    async fn prepare_type_hierarchy(&self, params: TypeHierarchyPrepareParams) -> jsonrpc::Result<Option<Vec<TypeHierarchyItem>>> { self.core.prepare_type_hierarchy(params).await }
    async fn supertypes(&self, params: TypeHierarchySupertypesParams) -> jsonrpc::Result<Option<Vec<TypeHierarchyItem>>> { self.core.supertypes(params).await }
    async fn subtypes(&self, params: TypeHierarchySubtypesParams) -> jsonrpc::Result<Option<Vec<TypeHierarchyItem>>> { self.core.subtypes(params).await }
    async fn document_highlight(&self, params: DocumentHighlightParams) -> jsonrpc::Result<Option<Vec<DocumentHighlight>>> { self.core.document_highlight(params).await }
    async fn document_link(&self, params: DocumentLinkParams) -> jsonrpc::Result<Option<Vec<DocumentLink>>> { self.core.document_link(params).await }
    async fn document_link_resolve(&self, params: DocumentLink) -> jsonrpc::Result<DocumentLink> { self.core.document_link_resolve(params).await }
    async fn hover(&self, params: HoverParams) -> jsonrpc::Result<Option<Hover>> { self.core.hover(params).await }
    async fn code_lens(&self, params: CodeLensParams) -> jsonrpc::Result<Option<Vec<CodeLens>>> { self.core.code_lens(params).await }
    async fn code_lens_resolve(&self, params: CodeLens) -> jsonrpc::Result<CodeLens> { self.core.code_lens_resolve(params).await }
    async fn folding_range(&self, params: FoldingRangeParams) -> jsonrpc::Result<Option<Vec<FoldingRange>>> { self.core.folding_range(params).await }
    async fn selection_range(&self, params: SelectionRangeParams) -> jsonrpc::Result<Option<Vec<SelectionRange>>> { self.core.selection_range(params).await }
    async fn document_symbol(&self, params: DocumentSymbolParams) -> jsonrpc::Result<Option<DocumentSymbolResponse>> { self.core.document_symbol(params).await }
    async fn semantic_tokens_full(&self, params: SemanticTokensParams) -> jsonrpc::Result<Option<SemanticTokensResult>> { self.core.semantic_tokens_full(params).await }
    async fn semantic_tokens_full_delta(&self, params: SemanticTokensDeltaParams) -> jsonrpc::Result<Option<SemanticTokensFullDeltaResult>> { self.core.semantic_tokens_full_delta(params).await }
    async fn semantic_tokens_range(&self, params: SemanticTokensRangeParams) -> jsonrpc::Result<Option<SemanticTokensRangeResult>> { self.core.semantic_tokens_range(params).await }
    async fn inline_value(&self, params: InlineValueParams) -> jsonrpc::Result<Option<Vec<InlineValue>>> { self.core.inline_value(params).await }
    async fn inlay_hint(&self, params: InlayHintParams) -> jsonrpc::Result<Option<Vec<InlayHint>>> { self.core.inlay_hint(params).await }
    async fn inlay_hint_resolve(&self, params: InlayHint) -> jsonrpc::Result<InlayHint> { self.core.inlay_hint_resolve(params).await }
    async fn moniker(&self, params: MonikerParams) -> jsonrpc::Result<Option<Vec<Moniker>>> { self.core.moniker(params).await }
    async fn completion(&self, params: CompletionParams) -> jsonrpc::Result<Option<CompletionResponse>> { self.core.completion(params).await }
    async fn completion_resolve(&self, params: CompletionItem) -> jsonrpc::Result<CompletionItem> { self.core.completion_resolve(params).await }
    async fn diagnostic(&self, params: DocumentDiagnosticParams) -> jsonrpc::Result<DocumentDiagnosticReportResult> { self.core.diagnostic(params).await }
    async fn workspace_diagnostic(&self, params: WorkspaceDiagnosticParams) -> jsonrpc::Result<WorkspaceDiagnosticReportResult> { self.core.workspace_diagnostic(params).await }
    async fn signature_help(&self, params: SignatureHelpParams) -> jsonrpc::Result<Option<SignatureHelp>> { self.core.signature_help(params).await }
    async fn code_action(&self, params: CodeActionParams) -> jsonrpc::Result<Option<CodeActionResponse>> { self.core.code_action(params).await }
    async fn code_action_resolve(&self, params: CodeAction) -> jsonrpc::Result<CodeAction> { self.core.code_action_resolve(params).await }
    async fn document_color(&self, params: DocumentColorParams) -> jsonrpc::Result<Vec<ColorInformation>> { self.core.document_color(params).await }
    async fn color_presentation(&self, params: ColorPresentationParams) -> jsonrpc::Result<Vec<ColorPresentation>> { self.core.color_presentation(params).await }
    async fn formatting(&self, params: DocumentFormattingParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> { self.core.formatting(params).await }
    async fn range_formatting(&self, params: DocumentRangeFormattingParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> { self.core.range_formatting(params).await }
    async fn on_type_formatting(&self, params: DocumentOnTypeFormattingParams) -> jsonrpc::Result<Option<Vec<TextEdit>>> { self.core.on_type_formatting(params).await }
    async fn rename(&self, params: RenameParams) -> jsonrpc::Result<Option<WorkspaceEdit>> { self.core.rename(params).await }
    async fn prepare_rename(&self, params: TextDocumentPositionParams) -> jsonrpc::Result<Option<PrepareRenameResponse>> { self.core.prepare_rename(params).await }
    async fn linked_editing_range(&self, params: LinkedEditingRangeParams) -> jsonrpc::Result<Option<LinkedEditingRanges>> { self.core.linked_editing_range(params).await }
    async fn symbol(&self, params: WorkspaceSymbolParams) -> jsonrpc::Result<Option<Vec<SymbolInformation>>> { self.core.symbol(params).await }
    async fn symbol_resolve(&self, params: WorkspaceSymbol) -> jsonrpc::Result<WorkspaceSymbol> { self.core.symbol_resolve(params).await }
    async fn did_change_configuration(&self, params: DidChangeConfigurationParams) { self.core.did_change_configuration(params).await }
    async fn did_change_workspace_folders(&self, params: DidChangeWorkspaceFoldersParams) { self.core.did_change_workspace_folders(params).await }
    async fn will_create_files(&self, params: CreateFilesParams) -> jsonrpc::Result<Option<WorkspaceEdit>> { self.core.will_create_files(params).await }
    async fn did_create_files(&self, params: CreateFilesParams) { self.core.did_create_files(params).await }
    async fn will_rename_files(&self, params: RenameFilesParams) -> jsonrpc::Result<Option<WorkspaceEdit>> { self.core.will_rename_files(params).await }
    async fn did_rename_files(&self, params: RenameFilesParams) { self.core.did_rename_files(params).await }
    async fn will_delete_files(&self, params: DeleteFilesParams) -> jsonrpc::Result<Option<WorkspaceEdit>> { self.core.will_delete_files(params).await }
    async fn did_delete_files(&self, params: DeleteFilesParams) { self.core.did_delete_files(params).await }
    async fn did_change_watched_files(&self, params: DidChangeWatchedFilesParams) { self.core.did_change_watched_files(params).await }
    async fn execute_command(&self, params: ExecuteCommandParams) -> jsonrpc::Result<Option<Value>> { self.core.execute_command(params).await }
}
