use crate::server::document::VirtualFileSystem;
use crate::server::settings::Settings;
use crossbeam_channel::{SendError, Sender};
use lsp_server::{ErrorCode, Message, RequestId, Response};
use lsp_types::notification::Notification;
use lsp_types::request::{RegisterCapability, Request};
use lsp_types::{notification, request, DidChangeWatchedFilesRegistrationOptions, FileSystemWatcher, GlobPattern, MessageType, Registration, RegistrationParams, ShowMessageParams, Uri};
use std::collections::HashSet;

pub struct ServerState {
    sender: Sender<Message>,
    pub settings: Settings,
    pub has_received_shutdown_request: bool,

    next_id: u64,
    request_ids_expecting_null_response: HashSet<String>,

    pub open_files: HashSet<Uri>,
    pub virtual_file_system: VirtualFileSystem,
}

// TODO unify these?
#[derive(Debug)]
pub enum RequestError {
    ParamParse(serde_json::Error),
    MethodNotImplemented,
    Invalid(String),
}

// TODO unify these?
#[derive(Debug)]
pub enum NotificationError {
    ParamParse(serde_json::Error),
    MethodNotImplemented,
    Invalid(String),
}

pub trait RequestHandler<R: Request> {
    fn handle_request(&mut self, params: R::Params) -> Result<R::Result, RequestError>;
}

pub trait NotificationHandler<N: Notification> {
    fn handle_notification(&mut self, params: N::Params) -> Result<(), NotificationError>;
}

macro_rules! handle_request {
    ($self:ident, $params:expr, $ty:ty) => {
        handle_request_impl::<$ty>($params, |p| <Self as RequestHandler<$ty>>::handle_request($self, p))
    };
}

macro_rules! handle_notification {
    ($self:ident, $params:expr, $ty:ty) => {
        handle_notification_impl::<$ty>($params, |p| <Self as NotificationHandler<$ty>>::handle_notification($self, p))
    };
}

#[derive(Debug, Copy, Clone)]
pub enum HandleMessageOutcome {
    Continue,
    Exit(i32),
}

impl ServerState {
    pub fn new(settings: Settings, sender: Sender<Message>) -> Self {
        Self {
            settings,
            sender,
            has_received_shutdown_request: false,
            next_id: 0,
            request_ids_expecting_null_response: HashSet::new(),
            open_files: HashSet::new(),
            virtual_file_system: VirtualFileSystem::new(),
        }
    }

    pub fn initial_registrations(&mut self) -> Result<(), SendError<Message>> {
        // subscript to file changes
        let params = RegistrationParams {
            registrations: vec![
                Registration {
                    id: self.next_unique_id(),
                    method: notification::DidChangeWatchedFiles::METHOD.to_string(),
                    register_options: Some(serde_json::to_value(DidChangeWatchedFilesRegistrationOptions {
                        watchers: vec![
                            FileSystemWatcher {
                                // TODO use relative?
                                glob_pattern: GlobPattern::String("**/{*.kh,.kh_config.toml}".to_owned()),
                                kind: None,
                            }
                        ],
                    }).unwrap()),
                }
            ],
        };
        self.send_request::<RegisterCapability>(params)?;

        Ok(())
    }

    fn next_unique_id(&mut self) -> String {
        let id = self.next_id.to_string();
        self.next_id += 1;
        id
    }

    // TODO support non-void requests
    pub fn send_request<R: Request<Result=()>>(&mut self, args: R::Params) -> Result<(), SendError<Message>> {
        let id = self.next_unique_id();
        let request = lsp_server::Request {
            id: RequestId::from(id.clone()),
            method: R::METHOD.to_owned(),
            params: serde_json::to_value(args).unwrap(),
        };
        self.sender.send(Message::Request(request))?;
        assert!(self.request_ids_expecting_null_response.insert(id));
        Ok(())
    }

    pub fn send_notification<N: Notification>(&mut self, args: N::Params) -> Result<(), SendError<Message>> {
        let notification = lsp_server::Notification {
            method: notification::ShowMessage::METHOD.to_owned(),
            params: serde_json::to_value(args).unwrap(),
        };
        self.sender.send(Message::Notification(notification))?;
        Ok(())
    }

    pub fn handle_message(&mut self, msg: Message) -> Result<HandleMessageOutcome, SendError<Message>> {
        match msg {
            Message::Request(request) => {
                eprintln!("received request: {request:?}");
                let method = &request.method;

                // evaluate the request
                let result = if self.has_received_shutdown_request {
                    Err(RequestError::Invalid("no requests allowed after shutdown".to_owned()))
                } else {
                    self.dispatch_request(method, request.params)
                };

                // format the result
                let response = match result {
                    Ok(result) => Response::new_ok(request.id, result),
                    Err(e) => {
                        let (code, message) = match e {
                            RequestError::ParamParse(e) => (
                                ErrorCode::InvalidParams,
                                format!("failed to parse request {method:?} parameters: {e:?}")
                            ),
                            RequestError::MethodNotImplemented => (
                                ErrorCode::MethodNotFound,
                                format!("method not implemented for request {method:?}")
                            ),
                            RequestError::Invalid(reason) => (
                                ErrorCode::InvalidRequest,
                                format!("invalid request {method:?}: {reason:?}"),
                            ),
                        };
                        Response::new_err(request.id, code as i32, message)
                    }
                };

                // send response
                self.sender.send(Message::Response(response))?;
            }
            Message::Response(_) => {
                // We don't expect to receive any yet, if there are any we can safely ignore them.
                eprintln!("received response: {msg:?}")
            }
            Message::Notification(notification) => {
                eprintln!("received notification: {notification:?}");

                if notification.method == notification::Exit::METHOD {
                    return if self.has_received_shutdown_request {
                        Ok(HandleMessageOutcome::Exit(0))
                    } else {
                        Ok(HandleMessageOutcome::Exit(1))
                    };
                }

                let method = &notification.method;
                match self.dispatch_notification(method, notification.params) {
                    Ok(()) => {}
                    Err(e) => {
                        let message = match e {
                            NotificationError::ParamParse(e) =>
                                format!("failed to parse parameters of notification {method:?}: {e:?}"),
                            NotificationError::MethodNotImplemented =>
                                format!("method not implemented for notification {method:?}"),
                            NotificationError::Invalid(reason) =>
                                format!("invalid notification {method:?}: {reason:?}"),
                        };
                        self.send_notification::<notification::ShowMessage>(ShowMessageParams {
                            typ: MessageType::ERROR,
                            message,
                        })?;
                    }
                }
            }
        }

        Ok(HandleMessageOutcome::Continue)
    }

    fn dispatch_request(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, RequestError> {
        match method {
            request::SemanticTokensFullRequest::METHOD =>
                handle_request!(self, params, request::SemanticTokensFullRequest),
            request::Shutdown::METHOD =>
                handle_request!(self, params, request::Shutdown),
            _ => Err(RequestError::MethodNotImplemented),
        }
    }

    fn dispatch_notification(&mut self, method: &str, params: serde_json::Value) -> Result<(), NotificationError> {
        match method {
            notification::DidOpenTextDocument::METHOD =>
                handle_notification!(self, params, notification::DidOpenTextDocument),
            notification::DidCloseTextDocument::METHOD =>
                handle_notification!(self, params, notification::DidCloseTextDocument),
            notification::DidChangeTextDocument::METHOD =>
                handle_notification!(self, params, notification::DidChangeTextDocument),
            notification::DidChangeWatchedFiles::METHOD =>
                handle_notification!(self, params, notification::DidChangeWatchedFiles),
            notification::WillSaveTextDocument::METHOD => {
                // ignored, this doesn't have any effect
                Ok(())
            },
            _ => {
                if method.starts_with("$/") {
                    // these notifications are allowed to be ignored
                    Ok(())
                } else {
                    // this could have been an important notification
                    Err(NotificationError::MethodNotImplemented)
                }
            },
        }
    }
}

fn handle_request_impl<R: Request>(params_value: serde_json::Value, f: impl FnOnce(R::Params) -> Result<R::Result, RequestError>) -> Result<serde_json::Value, RequestError> {
    let params = serde_json::from_value::<R::Params>(params_value)
        .map_err(RequestError::ParamParse)?;
    let result = f(params)?;
    let result_value = serde_json::to_value(result).unwrap();
    Ok(result_value)
}

fn handle_notification_impl<N: Notification>(params_value: serde_json::Value, f: impl FnOnce(N::Params) -> Result<(), NotificationError>) -> Result<(), NotificationError> {
    let params = serde_json::from_value::<N::Params>(params_value)
        .map_err(NotificationError::ParamParse)?;
    f(params)?;
    Ok(())
}
