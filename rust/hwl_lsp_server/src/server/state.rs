use crate::server::document::OpenFileInfo;
use crate::server::settings::Settings;
use crossbeam_channel::{SendError, Sender};
use indexmap::IndexMap;
use lsp_server::{ErrorCode, Message, RequestId, Response, ResponseError};
use lsp_types::notification::{DidChangeWatchedFiles, Notification};
use lsp_types::request::{RegisterCapability, Request};
use lsp_types::{notification, request, DidChangeWatchedFilesRegistrationOptions, FileSystemWatcher, GlobPattern, Registration, RegistrationParams, Uri};
use std::collections::HashSet;

pub struct ServerState {
    sender: Sender<Message>,
    pub settings: Settings,
    pub open_files: IndexMap<Uri, OpenFileInfo>,

    next_id: u64,
    request_ids_expecting_null_response: HashSet<String>,
}

pub trait RequestHandler<R: Request> {
    // TODO better handling, but not just exposing compiler errors: they should be rendered, not errors in the protocol
    fn handle_request(&mut self, params: R::Params) -> Result<R::Result, String>;
}

pub trait NotificationHandler<N: Notification> {
    // TODO error handling
    fn handle_notification(&mut self, params: N::Params);
}

macro_rules! handle_notification {
    ($self:ident, $params:expr, $ty:ty) => {
        handle_notification_impl::<$ty>($params, |p| <Self as NotificationHandler<$ty>>::handle_notification($self, p))
    };
}

impl ServerState {
    pub fn new(settings: Settings, sender: Sender<Message>) -> Self {
        Self {
            settings,
            sender,
            open_files: IndexMap::new(),
            next_id: 0,
            request_ids_expecting_null_response: HashSet::new(),
        }
    }

    pub fn initial_registrations(&mut self) -> Result<(), SendError<Message>> {
        // subscript to file changes
        let params = RegistrationParams {
            registrations: vec![
                Registration {
                    id: self.next_unique_id(),
                    method: DidChangeWatchedFiles::METHOD.to_string(),
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
        let req = lsp_server::Request {
            id: RequestId::from(id.clone()),
            method: R::METHOD.to_owned(),
            params: serde_json::to_value(args).unwrap(),
        };
        self.sender.send(Message::Request(req))?;
        assert!(self.request_ids_expecting_null_response.insert(id));
        Ok(())
    }

    pub fn handle_message(&mut self, msg: Message) -> Result<(), SendError<Message>> {
        match msg {
            Message::Request(msg) => {
                eprintln!("received request: {msg:?}");

                let response = match self.dispatch_request(&msg.method, msg.params) {
                    Ok(result) => Response::new_ok(msg.id, result),
                    Err(error) => Response::new_err(msg.id, error.code, error.message),
                };
                self.sender.send(Message::Response(response))?;
            }
            Message::Response(_) => {
                // We don't expect to receive any yet, if there are any we can safely ignore them.
                eprintln!("received response: {msg:?}")
            }
            Message::Notification(msg) => {
                eprintln!("received notification: {msg:?}");
                // TODO handle parsing error?
                self.dispatch_notification(&msg.method, msg.params).unwrap();
            }
        }

        Ok(())
    }

    fn dispatch_request(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, ResponseError> {
        match method {
            request::SemanticTokensFullRequest::METHOD =>
                handle_request_impl::<request::SemanticTokensFullRequest>(params, |p| self.handle_request(p)),
            _ => {
                Err(ResponseError {
                    code: ErrorCode::MethodNotFound as _,
                    message: format!("unexpected method {:?}", method),
                    data: None,
                })
            }
        }
    }

    fn dispatch_notification(&mut self, method: &str, params: serde_json::Value) -> Result<(), serde_json::Error> {
        match method {
            notification::DidOpenTextDocument::METHOD =>
                handle_notification!(self, params, notification::DidOpenTextDocument),
            notification::DidCloseTextDocument::METHOD =>
                handle_notification!(self, params, notification::DidCloseTextDocument),
            notification::DidChangeTextDocument::METHOD =>
                handle_notification!(self, params, notification::DidChangeTextDocument),
            _ => {
                // ignoring notifications is usually fine
                eprintln!("ignoring notification: {method}");
                Ok(())
            },
        }
    }
}

fn handle_request_impl<R: Request>(params: serde_json::Value, f: impl FnOnce(R::Params) -> Result<R::Result, String>) -> Result<serde_json::Value, ResponseError> {
    match parse_params_request::<R>(params) {
        Ok(params) => {
            match f(params) {
                Ok(res) => Ok(serde_json::to_value(res).unwrap()),
                Err(e) => Err(ResponseError {
                    code: ErrorCode::InternalError as _,
                    message: format!("failed to handle request: {e}"),
                    data: None,
                }),
            }
        }
        Err(e) => Err(ResponseError {
            code: ErrorCode::InvalidParams as _,
            message: format!("failed to parse parameters: {e:?}"),
            data: None,
        }),
    }
}

fn handle_notification_impl<N: Notification>(params: serde_json::Value, f: impl FnOnce(N::Params)) -> Result<(), serde_json::Error> {
    // TODO error handling
    match parse_params_notification::<N>(params) {
        Ok(params) => f(params),
        Err(e) => eprintln!("failed to parse parameters: {e:?}"),
    }

    Ok(())
}

fn parse_params_request<R: Request>(params: serde_json::Value) -> serde_json::Result<R::Params> {
    serde_json::from_value::<R::Params>(params)
}

fn parse_params_notification<N: Notification>(params: serde_json::Value) -> serde_json::Result<N::Params> {
    serde_json::from_value::<N::Params>(params)
}
