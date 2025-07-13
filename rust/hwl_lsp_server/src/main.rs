use crossbeam_channel::{RecvError, SendError, TryRecvError};
use hwl_language::throw;
use hwl_lsp_server::server::logger::Logger;
use hwl_lsp_server::server::sender::ServerSender;
use hwl_lsp_server::server::settings::Settings;
use hwl_lsp_server::server::state::{HandleMessageOutcome, OrSendError, RequestError, ServerState};
use lsp_server::{Connection, ErrorCode, Message, ProtocolError, Response};
use lsp_types::{InitializeParams, InitializeResult, ServerInfo};
use serde_json::to_value;
use std::path::Path;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), TopError> {
    std::panic::catch_unwind(main_inner).unwrap_or_else(|e| {
        let s = match e.downcast::<String>() {
            Err(e) => format!("{:?} with type {:?}", e, (*e).type_id()),
            Ok(e_str) => format!("{e_str:?}"),
        };

        std::fs::write("panic.txt", s).unwrap();
        Err(TopError::Panic)
    })
}

fn main_inner() -> Result<(), TopError> {
    // TODO make this configurable through env var at least
    let mut logger = Logger::new(false, Some(Path::new("log.txt")));
    logger.log("server started");

    // open connection
    // TODO allow runtime selection of different protocols
    let (connection, io_threads) = Connection::stdio();

    // initialization
    let settings = {
        let (initialize_id, initialization_params) = connection.initialize_start()?;
        let initialization_params = match serde_json::from_value::<InitializeParams>(initialization_params.clone()) {
            Ok(p) => p,
            Err(e) => {
                let mut sender = ServerSender::new(connection.sender, logger);
                sender.send_notification_error(RequestError::ParamParse(e), "initialization")?;
                return Ok(());
            }
        };

        let settings = match Settings::new(initialization_params) {
            Ok(settings) => settings,
            Err(e) => {
                let mut sender = ServerSender::new(connection.sender, logger);
                sender.send_response(Response::new_err(
                    initialize_id,
                    ErrorCode::RequestFailed as i32,
                    e.0.clone(),
                ))?;
                return Ok(());
            }
        };

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

    let mut state = ServerState::new(settings, ServerSender::new(connection.sender, logger));
    state.initial_registrations()?;

    // main loop
    let exit_code = 'outer: loop {
        state.log("waiting for message");
        let mut msg = connection.receiver.recv();

        // inner loop to handle a bunch of non-blocking events at once
        'inner: loop {
            match msg {
                Ok(msg) => {
                    state.log(format!("==> {msg:?}"));

                    let outcome = state.handle_message(msg)?;
                    match outcome {
                        HandleMessageOutcome::Continue => {}
                        HandleMessageOutcome::Exit(exit_code) => break 'outer exit_code,
                    }
                }
                Err(e) => {
                    let _: RecvError = e;
                    // Receive error, which means the input channel was closed
                    // no need to raise an error here, this could happen in normal operation
                    // TODO should the server not have sent a shutdown before?
                    break 'outer 2;
                }
            }

            // immediately get the next message if any non-blocking messages are available
            msg = match connection.receiver.try_recv() {
                Ok(msg) => Ok(msg),
                Err(TryRecvError::Empty) => break 'inner,
                Err(TryRecvError::Disconnected) => Err(RecvError),
            };
            state.log("got another non-blocking message");
        }

        // there are no more messages immediately available, spend some time doing other things
        // (eg. incremental compilation, collecting and pushing diagnostics, ...)
        state.log("doing background work");
        match state.do_background_work() {
            Ok(()) => {
                state.log("finished background work");
            }
            Err(e) => match e {
                OrSendError::SendError(e) => throw!(e),
                OrSendError::Error(e) => {
                    state.sender.send_notification_error(e, "background work")?;
                }
            },
        };
    };

    // TODO monitor the client process ID, and stop the server if it every dies somehow without closing the IO channel
    // TODO do we need to close channels?
    state.log("joining io threads");
    io_threads.join()?;
    state.log(format!("exiting with code {exit_code}"));
    std::process::exit(exit_code);
}

#[derive(Debug)]
pub enum TopError {
    IO(std::io::Error),
    SendError(SendError<Message>),
    Protocol(ProtocolError),
    InitJson(String),
    Panic,
}

impl From<std::io::Error> for TopError {
    fn from(value: std::io::Error) -> Self {
        TopError::IO(value)
    }
}

impl From<SendError<Message>> for TopError {
    fn from(value: SendError<Message>) -> Self {
        TopError::SendError(value)
    }
}

impl From<ProtocolError> for TopError {
    fn from(value: ProtocolError) -> Self {
        TopError::Protocol(value)
    }
}
