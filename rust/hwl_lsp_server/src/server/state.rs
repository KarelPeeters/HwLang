use crate::server::settings::Settings;
use crossbeam_channel::{SendError, Sender};
use indexmap::IndexMap;
use lsp_server::{ErrorCode, Message, Response, ResponseError};
use lsp_types::notification::Notification;
use lsp_types::request::Request;
use lsp_types::{notification, request, Uri};

pub struct ServerState {
    sender: Sender<Message>,
    pub settings: Settings,
    pub open_files: IndexMap<Uri, String>
}

pub trait RequestHandler<R: Request> {
    fn handle_request(&mut self, params: R::Params) -> Result<R::Result, String>;
}

pub trait NotificationHandler<N: Notification> {
    // TODO error handling
    fn handle_notification(&mut self, params: N::Params);
}

impl ServerState {
    pub fn new(settings: Settings, sender: Sender<Message>) -> Self {
        Self {
            settings,
            sender,
            open_files: IndexMap::new(),
        }
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

    // TODO macros to generate these dispatches? or does a &dyn vec work too?
    fn dispatch_request(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, ResponseError> {
        match method {
            request::SemanticTokensFullRequest::METHOD => {
                self.handle_request_impl::<request::SemanticTokensFullRequest>(params)
            }
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
            notification::DidOpenTextDocument::METHOD => {
                let params = parse_params_notification::<notification::DidOpenTextDocument>(params)?;
                <ServerState as NotificationHandler<notification::DidOpenTextDocument>>::handle_notification(self, params);
            }
            notification::DidCloseTextDocument::METHOD => {
                let params = parse_params_notification::<notification::DidCloseTextDocument>(params)?;
                <ServerState as NotificationHandler<notification::DidCloseTextDocument>>::handle_notification(self, params);
            }
            notification::DidChangeTextDocument::METHOD => {
                let params = parse_params_notification::<notification::DidChangeTextDocument>(params)?;
                <ServerState as NotificationHandler<notification::DidChangeTextDocument>>::handle_notification(self, params);
            }
            _ => {
                eprintln!("ignoring notification: {method}")
            }
        }

        Ok(())
    }

    fn handle_request_impl<R: Request>(&mut self, params: serde_json::Value) -> Result<serde_json::Value, ResponseError> {
        match parse_params_request::<request::SemanticTokensFullRequest>(params) {
            Ok(params) => {
                match self.handle_request(params) {
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
}

fn parse_params_request<R: Request>(params: serde_json::Value) -> serde_json::Result<R::Params> {
    serde_json::from_value::<R::Params>(params)
}

fn parse_params_notification<N: Notification>(params: serde_json::Value) -> serde_json::Result<N::Params> {
    serde_json::from_value::<N::Params>(params)
}
