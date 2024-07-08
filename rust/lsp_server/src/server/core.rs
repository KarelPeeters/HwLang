use std::collections::HashMap;

use tokio::sync::Mutex;
use tower_lsp::{
    lsp_types::{MessageType, Url},
    Client,
};

#[derive(Debug)]
pub struct ServerCore {
    pub client: Client,
    pub state: Mutex<State>,
}

#[derive(Debug)]
pub struct State {
    pub documents: HashMap<Url, String>,
}

impl ServerCore {
    pub async fn log_error(&self, msg: String) {
        self.client.log_message(MessageType::ERROR, msg).await;
    }

    pub async fn log_warning(&self, msg: String) {
        self.client.log_message(MessageType::WARNING, msg).await;
    }

    pub async fn log_info(&self, msg: String) {
        self.client.log_message(MessageType::INFO, msg).await;
    }

    pub async fn log_log(&self, msg: String) {
        self.client.log_message(MessageType::LOG, msg).await;
    }
}
