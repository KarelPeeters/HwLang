use tower_lsp::Client;

#[derive(Debug)]
pub struct ServerCore {
    pub client: Client
}
