use crate::handlers::dispatch::RequestHandler;
use crate::server::state::{RequestResult, ServerState};
use lsp_types::request::Shutdown;

impl RequestHandler<Shutdown> for ServerState {
    fn handle_request(&mut self, _: ()) -> RequestResult<()> {
        assert!(
            !self.has_received_shutdown_request,
            "this should have been checked in the main loop already"
        );
        self.has_received_shutdown_request = true;
        Ok(())
    }
}
