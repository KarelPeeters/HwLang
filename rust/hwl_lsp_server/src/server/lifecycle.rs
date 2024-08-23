use crate::server::state::{NotificationHandler, RequestHandler, ServerState};
use lsp_types::notification::Exit;
use lsp_types::request::Shutdown;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LifecycleState {
    Running,
    Shutdown,
    Exit,
}

impl RequestHandler<Shutdown> for ServerState {
    fn handle_request(&mut self, _: ()) -> Result<(), String> {
        if self.lifecycle_state != LifecycleState::Running {
            return Err(format!("tried to shutdown server which is not running but in state {:?}", self.lifecycle_state));
        }
        self.lifecycle_state = LifecycleState::Shutdown;
        Ok(())
    }
}

impl NotificationHandler<Exit> for ServerState {
    fn handle_notification(&mut self, _: ()) {
        if self.lifecycle_state != LifecycleState::Shutdown {
            // TODO proper error handling
            panic!("tried to shutdown server which is not shutdown but in state {:?}", self.lifecycle_state);
        }
        self.lifecycle_state = LifecycleState::Exit;
    }
}
