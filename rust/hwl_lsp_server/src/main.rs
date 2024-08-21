use hwl_lsp_server::util::from_json;
use lsp_server::{Connection, Message, ProtocolError};
use lsp_types::{InitializeParams, InitializeResult, ServerCapabilities, ServerInfo};
use serde_json::to_value;

struct ServerState {}

impl ServerState {
    pub fn new(_init: InitializeParams) -> Self {
        ServerState {}
    }

    pub fn capabilities(&self) -> ServerCapabilities {
        Default::default()
    }
}

fn main() -> Result<(), TopError> {
    // TODO allow runtime selection of different protocols
    eprintln!("Starting LSP server");

    let (connection, io_threads) = Connection::stdio();

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
    let initialization_params = from_json::<InitializeParams>("InitializeParams", &initialization_params)?;

    let state = ServerState::new(initialization_params);

    let initialize_result = InitializeResult {
        capabilities: state.capabilities(),
        server_info: Some(ServerInfo {
            name: "hwl_lsp_server".to_owned(),
            version: Some(format!("{}-dev", env!("CARGO_PKG_VERSION"))),
        }),
    };
    connection.initialize_finish(initialize_id, to_value(initialize_result).unwrap())?;

    loop {
        match connection.receiver.recv() {
            Ok(msg) => {
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
enum TopError {
    Protocol(ProtocolError),
    IO(std::io::Error),
    Both(std::io::Error, ProtocolError),
    Anyhow(anyhow::Error),
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
