use log::{error, warn};
use serde_json::Value;
use tower_lsp::jsonrpc;
use tower_lsp::jsonrpc::Error;
use tower_lsp::lsp_types::{CreateFilesParams, DeleteFilesParams, DidChangeConfigurationParams, DidChangeWatchedFilesParams, DidChangeWorkspaceFoldersParams, ExecuteCommandParams, RenameFilesParams, SymbolInformation, WorkspaceEdit, WorkspaceSymbol, WorkspaceSymbolParams};

use crate::server::core::ServerCore;

impl ServerCore {
    pub async fn symbol(&self, _params: WorkspaceSymbolParams) -> jsonrpc::Result<Option<Vec<SymbolInformation>>> {
        error!("Got a workspace/symbol request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn symbol_resolve(&self, _params: WorkspaceSymbol) -> jsonrpc::Result<WorkspaceSymbol> {
        error!("Got a workspaceSymbol/resolve request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn did_change_configuration(&self, _params: DidChangeConfigurationParams) {
        warn!("Got a workspace/didChangeConfiguration notification, but it is not implemented");
    }

    pub async fn did_change_workspace_folders(&self, _params: DidChangeWorkspaceFoldersParams) {
        warn!("Got a workspace/didChangeWorkspaceFolders notification, but it is not implemented");
    }

    pub async fn will_create_files(&self, _params: CreateFilesParams) -> jsonrpc::Result<Option<WorkspaceEdit>> {
        error!("Got a workspace/willCreateFiles request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn did_create_files(&self, _params: CreateFilesParams) {
        warn!("Got a workspace/didCreateFiles notification, but it is not implemented");
    }

    pub async fn will_rename_files(&self, _params: RenameFilesParams) -> jsonrpc::Result<Option<WorkspaceEdit>> {
        error!("Got a workspace/willRenameFiles request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn did_rename_files(&self, _params: RenameFilesParams) {
        warn!("Got a workspace/didRenameFiles notification, but it is not implemented");
    }

    pub async fn will_delete_files(&self, _params: DeleteFilesParams) -> jsonrpc::Result<Option<WorkspaceEdit>> {
        error!("Got a workspace/willDeleteFiles request, but it is not implemented");
        Err(Error::method_not_found())
    }

    pub async fn did_delete_files(&self, _params: DeleteFilesParams) {
        warn!("Got a workspace/didDeleteFiles notification, but it is not implemented");
    }

    pub async fn did_change_watched_files(&self, _params: DidChangeWatchedFilesParams) {
        warn!("Got a workspace/didChangeWatchedFiles notification, but it is not implemented");
    }

    pub async fn execute_command(&self, _params: ExecuteCommandParams) -> jsonrpc::Result<Option<Value>> {
        error!("Got a workspace/executeCommand request, but it is not implemented");
        Err(Error::method_not_found())
    }
}