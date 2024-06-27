use tower_lsp::{LspService, Server};
use lsp_server::server::core::ServerCore;
use lsp_server::server::wrapper::ServerWrapper;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // TODO support different transports
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| {
        ServerWrapper { core: ServerCore {client} }
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
