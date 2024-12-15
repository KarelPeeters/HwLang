use crate::server::state::{RequestError, RequestResult, ServerState};
use lsp_types::notification::Notification;
use lsp_types::request::Request;
use lsp_types::{notification, request};

pub trait RequestHandler<R: Request> {
    fn handle_request(&mut self, params: R::Params) -> RequestResult<R::Result>;
}

pub trait NotificationHandler<N: Notification> {
    fn handle_notification(&mut self, params: N::Params) -> RequestResult<()>;
}

macro_rules! handle_request {
    ($self:ident, $params:expr, $ty:ty) => {
        handle_request_impl::<$ty>($params, |p| <Self as RequestHandler<$ty>>::handle_request($self, p))
    };
}

macro_rules! handle_notification {
    ($self:ident, $params:expr, $ty:ty) => {
        handle_notification_impl::<$ty>($params, |p| {
            <Self as NotificationHandler<$ty>>::handle_notification($self, p)
        })
    };
}

impl ServerState {
    pub fn dispatch_request(&mut self, method: &str, params: serde_json::Value) -> RequestResult<serde_json::Value> {
        match method {
            request::SemanticTokensFullRequest::METHOD => {
                handle_request!(self, params, request::SemanticTokensFullRequest)
            }
            request::Shutdown::METHOD => handle_request!(self, params, request::Shutdown),
            _ => Err(RequestError::MethodNotImplemented),
        }
    }

    pub fn dispatch_notification(&mut self, method: &str, params: serde_json::Value) -> RequestResult<()> {
        match method {
            notification::DidOpenTextDocument::METHOD => {
                handle_notification!(self, params, notification::DidOpenTextDocument)
            }
            notification::DidCloseTextDocument::METHOD => {
                handle_notification!(self, params, notification::DidCloseTextDocument)
            }
            notification::DidChangeTextDocument::METHOD => {
                handle_notification!(self, params, notification::DidChangeTextDocument)
            }
            notification::DidChangeWatchedFiles::METHOD => {
                handle_notification!(self, params, notification::DidChangeWatchedFiles)
            }
            notification::WillSaveTextDocument::METHOD | notification::DidSaveTextDocument::METHOD => {
                // ignored, this doesn't have any effect
                Ok(())
            }
            _ => {
                if method.starts_with("$/") {
                    // these notifications are allowed to be ignored
                    Ok(())
                } else {
                    // this could have been an important notification
                    Err(RequestError::MethodNotImplemented)
                }
            }
        }
    }
}

fn handle_request_impl<R: Request>(
    params_value: serde_json::Value,
    f: impl FnOnce(R::Params) -> RequestResult<R::Result>,
) -> Result<serde_json::Value, RequestError> {
    let params = serde_json::from_value::<R::Params>(params_value).map_err(RequestError::ParamParse)?;
    Ok(serde_json::to_value(f(params)?).unwrap())
}

fn handle_notification_impl<N: Notification>(
    params_value: serde_json::Value,
    f: impl FnOnce(N::Params) -> RequestResult<()>,
) -> RequestResult<()> {
    let params = serde_json::from_value::<N::Params>(params_value).map_err(RequestError::ParamParse)?;
    f(params)
}
