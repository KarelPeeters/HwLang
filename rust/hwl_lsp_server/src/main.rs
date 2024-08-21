use crossbeam_channel::SendError;
use hwl_lsp_server::server::settings::Settings;
use hwl_lsp_server::server::state::ServerState;
use lsp_server::{Connection, ErrorCode, Message, ProtocolError, Response};
use lsp_types::{InitializeParams, InitializeResult, ServerInfo};
use serde_json::to_value;

fn main() -> Result<(), TopError> {
    eprintln!("LSP server started");

    // open connection
    // TODO allow runtime selection of different protocols
    let (connection, io_threads) = Connection::stdio();

    // initialization
    let settings = {
        let (initialize_id, initialization_params) = match connection.initialize_start() {
            Ok(d) => d,
            Err(e_protocol) => {
                if e_protocol.channel_is_disconnected() {
                    match io_threads.join() {
                        Ok(()) => {}
                        Err(e_io) => return Err(TopError::Both(e_io, e_protocol)),
                    }
                }
                return Err(TopError::Protocol(e_protocol));
            }
        };
        let initialization_params = match serde_json::from_value::<InitializeParams>(initialization_params.clone()) {
            Ok(p) => p,
            Err(e) => {
                let response = Response::new_err(initialize_id, ErrorCode::ParseError as _, format!("failed to parse initialization parameters: {e:?}"));
                connection.sender.send(Message::Response(response)).map_err(|e| TopError::SendError(e))?;
                return Err(TopError::Anyhow(e.into()));
            }
        };

        let settings = Settings::new(initialization_params);
        let server_capabilities = settings.server_capabilities();
        let initialize_result = InitializeResult {
            capabilities: server_capabilities.clone(),
            server_info: Some(ServerInfo {
                name: env!("CARGO_PKG_NAME").to_string(),
                version: Some(format!("{}-dev", env!("CARGO_PKG_VERSION"))),
            }),
        };
        connection.initialize_finish(initialize_id, to_value(initialize_result).unwrap())?;

        settings
    };
    let mut state = ServerState::new(settings, connection.sender);

    // main loop
    loop {
        match connection.receiver.recv() {
            Ok(msg) => {
                match state.handle_message(msg) {
                    Ok(()) => {}
                    Err(e) => return Err(TopError::SendError(e)),
                }
            }
            Err(_) => {
                // Receive error, which means the input channel was closed
                // no need to raise an error here, this could happen in normal operation
                break;
            }
        }
    }

    io_threads.join()?;
    Ok(())
}


#[derive(Debug)]
#[allow(dead_code)]
pub enum TopError {
    Protocol(ProtocolError),
    IO(std::io::Error),
    Both(std::io::Error, ProtocolError),
    Anyhow(anyhow::Error),
    SendError(SendError<Message>),
}

impl From<ProtocolError> for TopError {
    fn from(value: ProtocolError) -> Self {
        TopError::Protocol(value)
    }
}

impl From<std::io::Error> for TopError {
    fn from(value: std::io::Error) -> Self {
        TopError::IO(value)
    }
}

impl From<anyhow::Error> for TopError {
    fn from(value: anyhow::Error) -> Self {
        TopError::Anyhow(value)
    }
}
