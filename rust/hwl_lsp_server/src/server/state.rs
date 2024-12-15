use crate::engine::vfs::VfsError;
use crate::engine::vfs::VirtualFileSystem;
use crate::server::sender::ServerSender;
use crate::server::settings::Settings;
use crossbeam_channel::SendError;
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use lsp_server::{ErrorCode, Message, RequestId, Response};
use lsp_types::notification::Notification;
use lsp_types::request::RegisterCapability;
use lsp_types::{
    notification, DidChangeWatchedFilesRegistrationOptions, FileSystemWatcher, GlobPattern, Registration,
    RegistrationParams, Uri,
};
use std::collections::HashSet;

pub struct ServerState {
    pub settings: Settings,
    pub sender: ServerSender,

    pub has_received_shutdown_request: bool,

    pub open_files: HashSet<Uri>,
    pub vfs: VirtualFileSystemWrapper,
}

pub struct VirtualFileSystemWrapper {
    inner: Option<VirtualFileSystem>,
    root: Uri,
}

// TODO move these to some common place
// TODO rename these to something better
pub type RequestResult<T> = Result<T, RequestError>;

#[derive(Debug)]
pub enum OrSendError<E> {
    SendError(SendError<Message>),
    Error(E),
}

#[derive(Debug)]
pub enum RequestError {
    ParamParse(serde_json::Error),
    MethodNotImplemented,
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
        // TODO support multiple workspaces through a list of VFSs instead of just a single one
        let vfs = VirtualFileSystemWrapper::new(settings.initialize_params.root_uri.clone().unwrap());

        Self {
            settings,
            sender,
            has_received_shutdown_request: false,
            open_files: HashSet::new(),
            vfs,
        }
    }

    pub fn initial_registrations(&mut self) -> Result<(), SendError<Message>> {
        // subscribe to file changes
        let pattern = format!("**/{{*.{}}}", LANGUAGE_FILE_EXTENSION);
        let params = RegistrationParams {
            registrations: vec![Registration {
                id: self.sender.next_unique_id(),
                method: notification::DidChangeWatchedFiles::METHOD.to_string(),
                register_options: Some(
                    serde_json::to_value(DidChangeWatchedFilesRegistrationOptions {
                        watchers: vec![FileSystemWatcher {
                            // TODO use relative?
                            glob_pattern: GlobPattern::String(pattern),
                            kind: None,
                        }],
                    })
                        .unwrap(),
                ),
            }],
        };
        self.sender.send_request::<RegisterCapability>(params)?;

        Ok(())
    }

    pub fn handle_message(&mut self, msg: Message) -> Result<HandleMessageOutcome, SendError<Message>> {
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

    pub fn do_background_work(&mut self) -> Result<(), OrSendError<RequestError>> {
        self.compile_project_and_send_diagnostics()?;
        Ok(())
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        self.sender.logger.log(msg);
    }
}

impl VirtualFileSystemWrapper {
    pub fn new(root: Uri) -> Self {
        Self { inner: None, root }
    }

    pub fn inner(&mut self) -> Result<&mut VirtualFileSystem, VfsError> {
        if self.inner.is_none() {
            self.inner = Some(VirtualFileSystem::new(self.root.clone())?);
        }
        Ok(self.inner.as_mut().unwrap())
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
            RequestError::MethodNotImplemented => (ErrorCode::MethodNotFound, "method not implemented".to_string()),
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

impl<E> From<E> for OrSendError<E> {
    fn from(value: E) -> Self {
        OrSendError::Error(value)
    }
}
