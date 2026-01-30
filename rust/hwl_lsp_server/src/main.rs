use clap::Parser;
use crossbeam_channel::{RecvError, TryRecvError, bounded};
use hwl_language::throw;
use hwl_lsp_server::server::settings::Settings;
use hwl_lsp_server::server::state::{HandleMessageOutcome, RequestError, ServerState};
use hwl_lsp_server::util::logger::Logger;
use hwl_lsp_server::util::sender::{SendError, SendErrorOr, ServerSender};
use hwl_util::constants::{HWL_LSP_NAME, HWL_VERSION};
use lsp_server::{Connection, ErrorCode, Message, ProtocolError, Response};
use lsp_types::{InitializeParams, InitializeResult, ServerInfo};
use serde_json::to_value;
use std::fs::File;
use std::io::{BufWriter, stdin, stdout};
use std::path::PathBuf;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(clap::Parser)]
struct Args {
    #[clap(
        long,
        help = "The log file to use. This can also be set through the environment variable HWL_LSP_LOG_FILE."
    )]
    log_file: Option<PathBuf>,
}

// TODO replace types with https://github.com/gluon-lang/lsp-types/issues/284#issuecomment-2720488780
// TODO think about error handling some more
// TODO clean up this panicking mode, write to stderr and/or the log_file instead
fn main() -> Result<(), TopError> {
    std::panic::catch_unwind(main_inner).unwrap_or_else(|e| {
        let s = match e.downcast::<String>() {
            Err(e) => format!("{:?} with type {:?}", e, (*e).type_id()),
            Ok(e_str) => format!("{e_str:?}"),
        };

        std::fs::write("hwl_lsp_server_panic.txt", s).unwrap();
        Err(TopError::Panic)
    })
}

fn main_inner() -> Result<(), TopError> {
    let Args { log_file } = Args::parse();

    let log_file = log_file.or_else(|| std::env::var_os("HWL_LOG_FILE").map(PathBuf::from));
    let log_writer = log_file
        .map::<Result<_, TopError>, _>(|log_file| {
            let file = File::create(&log_file).map_err(|_| TopError::Init)?;
            Ok(BufWriter::new(file))
        })
        .transpose()?;

    let mut logger = Logger::new(false, log_writer);
    logger.log("server started");

    // open connection
    // TODO allow runtime selection of different protocols
    let (connection, io_threads) = connection_stdio_with_input_capacity();

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
                name: HWL_LSP_NAME.to_owned(),
                version: Some(HWL_VERSION.to_owned()),
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
        let should_stop = || !connection.receiver.is_empty();
        match state.do_background_work(&should_stop) {
            Ok(()) => {
                state.log("finished background work");
            }
            Err(e) => match e {
                SendErrorOr::SendError(e) => throw!(e),
                SendErrorOr::Other(e) => {
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

/// Copy of [lsp_server::stdio::stdio_transport], with non-zero capacity on the input channel,
/// to allow using [crossbeam_channel::Receiver::is_empty].
fn connection_stdio_with_input_capacity() -> (Connection, IoThreads) {
    let (writer_sender, writer_receiver) = bounded::<Message>(0);
    let writer = std::thread::spawn(move || {
        let stdout = stdout();
        let mut stdout = stdout.lock();
        writer_receiver.into_iter().try_for_each(|it| it.write(&mut stdout))
    });

    let (reader_sender, reader_receiver) = bounded::<Message>(1);
    let reader = std::thread::spawn(move || {
        let stdin = stdin();
        let mut stdin = stdin.lock();

        while let Some(msg) = Message::read(&mut stdin)? {
            let is_exit = matches!(&msg, Message::Notification(n) if n.method == "exit");

            reader_sender
                .send(msg)
                .expect("receiver was dropped, failed to send a message");

            if is_exit {
                break;
            }
        }
        Ok(())
    });

    let connection = Connection {
        sender: writer_sender,
        receiver: reader_receiver,
    };
    let threads = IoThreads { reader, writer };
    (connection, threads)
}

struct IoThreads {
    reader: std::thread::JoinHandle<std::io::Result<()>>,
    writer: std::thread::JoinHandle<std::io::Result<()>>,
}

impl IoThreads {
    fn join(self) -> std::io::Result<()> {
        match self.reader.join() {
            Ok(r) => r?,
            Err(err) => std::panic::panic_any(err),
        }
        match self.writer.join() {
            Ok(r) => r,
            Err(err) => std::panic::panic_any(err),
        }
    }
}

#[derive(Debug)]
pub enum TopError {
    Init,
    IO(std::io::Error),
    SendError(SendError),
    Protocol(ProtocolError),
    InitJson(String),
    Panic,
}

impl From<std::io::Error> for TopError {
    fn from(value: std::io::Error) -> Self {
        TopError::IO(value)
    }
}

impl From<SendError> for TopError {
    fn from(value: SendError) -> Self {
        TopError::SendError(value)
    }
}

impl From<ProtocolError> for TopError {
    fn from(value: ProtocolError) -> Self {
        TopError::Protocol(value)
    }
}
