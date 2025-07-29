use crate::engine::vfs::{Vfs, VfsError};
use crate::server::sender::{SendError, SendErrorOr, ServerSender};
use crate::server::settings::Settings;
use indexmap::IndexSet;
use lsp_server::{ErrorCode, Message, RequestId, Response};
use lsp_types::notification::Notification;
use lsp_types::request::RegisterCapability;
use lsp_types::{notification, DidChangeWatchedFilesRegistrationOptions, FileSystemWatcher, Registration, Uri};

pub struct ServerState {
    pub settings: Settings,
    pub sender: ServerSender,

    pub has_received_shutdown_request: bool,

    pub open_files: IndexSet<Uri>,
    pub vfs: Vfs,

    // local model of the client-side state, used to correctly send incremental messages
    pub curr_watchers: Vec<FileSystemWatcher>,
    pub curr_files_with_diagnostics: IndexSet<Uri>,
}

// TODO move these to some common place
// TODO rename these to something better
pub type RequestResult<T> = Result<T, RequestError>;

// TODO rethink error handling, do we want to distinguish between hard failures and soft failures?
#[derive(Debug)]
pub enum RequestError {
    ParamParse(serde_json::Error),
    MethodNotImplemented(String),
    Invalid(String),
    Vfs(VfsError),
    Internal(String),
}

#[derive(Debug, Copy, Clone)]
pub enum HandleMessageOutcome {
    Continue,
    Exit(i32),
}

impl ServerState {
    pub fn new(settings: Settings, sender: ServerSender) -> Self {
        ServerState {
            settings,
            sender,
            has_received_shutdown_request: false,
            open_files: IndexSet::new(),
            vfs: Vfs::new(),

            curr_watchers: vec![],
            curr_files_with_diagnostics: IndexSet::new(),
        }
    }

    pub fn initial_registrations(&mut self) -> Result<(), SendError> {
        // TODO register initial watchers here?
        Ok(())
    }

    /// Set the file watchers. This overrides any existing watchers, so ensure that the list passed here is complete.
    /// To avoid missed events, set the watchers before reading the corresponding files from disk.
    pub fn set_watchers(&mut self, watchers: Vec<FileSystemWatcher>) -> Result<(), SendError> {
        // TODO assert that the client has the right capability
        // avoid watcher churn
        if watchers == self.curr_watchers {
            return Ok(());
        }

        let register_options = Some(
            serde_json::to_value(DidChangeWatchedFilesRegistrationOptions {
                watchers: watchers.clone(),
            })
            .unwrap(),
        );
        let registrations = vec![Registration {
            id: self.sender.next_unique_id(),
            method: notification::DidChangeWatchedFiles::METHOD.to_string(),
            register_options,
        }];
        let params = lsp_types::RegistrationParams { registrations };
        self.sender.send_request::<RegisterCapability>(params)?;

        self.curr_watchers = watchers;
        Ok(())
    }

    pub fn handle_message(&mut self, msg: Message) -> Result<HandleMessageOutcome, SendError> {
        match msg {
            Message::Request(request) => {
                eprintln!("received request: {request:?}");

                // evaluate the request
                let result = if self.has_received_shutdown_request {
                    Err(RequestError::Invalid("no requests allowed after shutdown".to_owned()))
                } else {
                    self.dispatch_request(&request.method, request.params)
                };

                // send response back
                let response = match result {
                    Ok(result) => Response::new_ok(request.id, result),
                    Err(e) => e.to_response(request.id, &request.method),
                };
                self.sender.send_response(response)?;
            }
            Message::Response(_) => {
                // We don't expect to receive any yet, if there are any we can safely ignore them.
                eprintln!("received response: {msg:?}")
            }
            Message::Notification(notification) => {
                eprintln!("received notification: {notification:?}");
                let method = &notification.method;

                // handle exit notification
                if method == notification::Exit::METHOD {
                    return if self.has_received_shutdown_request {
                        Ok(HandleMessageOutcome::Exit(0))
                    } else {
                        Ok(HandleMessageOutcome::Exit(1))
                    };
                }

                // evaluate notification
                let result = self.dispatch_notification(method, notification.params);

                // maybe send error back
                match result {
                    Ok(()) => {}
                    Err(e) => self
                        .sender
                        .send_notification_error(e, &format!("notification {method:?}"))?,
                }
            }
        }

        Ok(HandleMessageOutcome::Continue)
    }

    pub fn do_background_work(&mut self) -> Result<(), SendErrorOr<RequestError>> {
        self.compile_project_and_send_diagnostics()
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        self.sender.logger.log(msg);
    }
}

impl RequestError {
    pub fn to_response(self, id: RequestId, method: &str) -> Response {
        let (code, message) = self.to_code_message();
        Response::new_err(id, code as i32, format!("error during request {method:?}: {message}"))
    }

    pub fn to_code_message(self) -> (ErrorCode, String) {
        match self {
            RequestError::ParamParse(e) => (ErrorCode::InvalidParams, format!("failed to parameters: {e:?}")),
            RequestError::MethodNotImplemented(m) => {
                (ErrorCode::MethodNotFound, format!("method not implemented: {m}"))
            }
            RequestError::Invalid(reason) => (ErrorCode::InvalidRequest, format!("invalid request {reason:?}")),
            RequestError::Vfs(e) => (ErrorCode::InternalError, format!("vfs error {e:?}")),
            RequestError::Internal(e) => (ErrorCode::InternalError, format!("internal error {e:?}")),
        }
    }
}

impl From<VfsError> for RequestError {
    fn from(value: VfsError) -> Self {
        RequestError::Vfs(value)
    }
}
