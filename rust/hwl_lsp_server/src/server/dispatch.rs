use crate::server::state::{NotificationError, RequestError, ServerState};
use lsp_types::notification::Notification;
use lsp_types::request::Request;
use lsp_types::{notification, request};

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

impl ServerState {
    pub fn dispatch_request(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, RequestError> {
        match method {
            request::SemanticTokensFullRequest::METHOD =>
                handle_request!(self, params, request::SemanticTokensFullRequest),
            request::Shutdown::METHOD =>
                handle_request!(self, params, request::Shutdown),
            _ => Err(RequestError::MethodNotImplemented),
        }
    }

    pub fn dispatch_notification(&mut self, method: &str, params: serde_json::Value) -> Result<(), NotificationError> {
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
            }
            _ => {
                if method.starts_with("$/") {
                    // these notifications are allowed to be ignored
                    Ok(())
                } else {
                    // this could have been an important notification
                    Err(NotificationError::MethodNotImplemented)
                }
            }
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
