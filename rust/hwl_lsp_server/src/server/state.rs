use crossbeam_channel::{SendError, Sender};
use lsp_server::Message;
use lsp_types::{InitializeParams, PositionEncodingKind, SemanticTokensFullOptions, SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities, WorkDoneProgressOptions};

pub struct Settings {
    pub initialize_params: InitializeParams,
    pub server_capabilities: ServerCapabilities,
}

pub struct ServerState {
    settings: Settings,
    sender: Sender<Message>,
}

impl Settings {
    pub fn new(initialize_params: InitializeParams) -> Self {
        const NO_WORK_DONE: WorkDoneProgressOptions = WorkDoneProgressOptions { work_done_progress: None };

        let server_capabilities = ServerCapabilities {
            // TODO use utf-8 if the client supports it, should be faster on both sides
            position_encoding: Some(PositionEncodingKind::UTF16),
            semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                work_done_progress_options: NO_WORK_DONE,
                legend: Default::default(),
                range: None,
                full: Some(SemanticTokensFullOptions::Bool(true)),
            })),
            ..Default::default()
        };

        Self { initialize_params, server_capabilities }
    }

    pub fn server_capabilities(&self) -> &ServerCapabilities {
        &self.server_capabilities
    }
}

impl ServerState {
    pub fn new(settings: Settings, sender: Sender<Message>) -> Self {
        Self { settings, sender }
    }

    pub fn handle_message(&mut self, msg: Message) -> Result<(), SendError<Message>> {
        match msg {
            Message::Request(msg) => {
                eprintln!("received request: {msg:?}")
            }
            Message::Response(_) => {
                eprintln!("received response: {msg:?}")
            }
            Message::Notification(_) => {
                eprintln!("received notification: {msg:?}")
            }
        }

        Ok(())
    }
}
