use std::sync::Mutex;

// TODO rename/expand to handle all external interactions: IO, env vars, ...
pub trait PrintHandler {
    fn print(&self, s: &str);
}

pub struct IgnorePrintHandler;

impl PrintHandler for IgnorePrintHandler {
    fn print(&self, _: &str) {}
}

pub struct StdoutPrintHandler;

impl PrintHandler for StdoutPrintHandler {
    #[allow(clippy::print_stdout)]
    fn print(&self, s: &str) {
        print!("{s}");
    }
}

pub struct CollectPrintHandler(Mutex<Vec<String>>);

impl CollectPrintHandler {
    pub fn new() -> Self {
        CollectPrintHandler(Mutex::new(Vec::new()))
    }

    pub fn finish(self) -> Vec<String> {
        self.0.into_inner().unwrap()
    }
}

impl PrintHandler for CollectPrintHandler {
    fn print(&self, s: &str) {
        self.0.lock().unwrap().push(s.to_string());
    }
}
