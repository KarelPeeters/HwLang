use crate::server::state::RequestError;
use crossbeam_channel::{SendError, Sender};
use lsp_server::{Message, RequestId, Response};
use lsp_types::notification::Notification;
use lsp_types::request::Request;
use lsp_types::{notification, MessageType, ShowMessageParams};
use std::collections::HashSet;

pub struct ServerSender {
    sender: Sender<Message>,

    next_id: u64,
    request_ids_expecting_null_response: HashSet<String>,
}

pub type SendResult<T = ()> = Result<T, SendError<Message>>;

impl ServerSender {
    pub fn new(sender: Sender<Message>) -> Self {
        Self {
            next_id: 0,
            request_ids_expecting_null_response: HashSet::new(),
            sender,
        }
    }

    pub fn next_unique_id(&mut self) -> String {
        let id = self.next_id.to_string();
        self.next_id += 1;
        id
    }

    // TODO support non-void requests
    pub fn send_request<R: Request<Result=()>>(&mut self, args: R::Params) -> SendResult {
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

    pub fn send_response(&mut self, response: Response) -> SendResult {
        self.sender.send(Message::Response(response))
    }

    pub fn send_notification<N: Notification>(&mut self, args: N::Params) -> SendResult {
        let notification = lsp_server::Notification {
            method: notification::ShowMessage::METHOD.to_owned(),
            params: serde_json::to_value(args).unwrap(),
        };
        self.sender.send(Message::Notification(notification))?;
        Ok(())
    }

    pub fn send_notification_error(&mut self, error: RequestError, during: &str) -> SendResult {
        let (_, message) = error.to_code_message();
        let params = ShowMessageParams {
            typ: MessageType::ERROR,
            message: format!("error during {during}: {message}"),
        };
        self.send_notification::<notification::ShowMessage>(params)
    }
}
