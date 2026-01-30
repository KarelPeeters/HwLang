use std::fs::File;
use std::io::{BufWriter, Write};

pub struct Logger {
    log_to_stderr: bool,
    log_writer: Option<BufWriter<File>>,
}

// TODO replace with proper logging setup, ideally the same one the lsp-server crate uses
#[allow(clippy::print_stderr)]
impl Logger {
    pub fn new(log_to_stderr: bool, log_writer: Option<BufWriter<File>>) -> Self {
        Self {
            log_to_stderr,
            log_writer,
        }
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let msg = msg.into();

        if self.log_to_stderr {
            eprintln!("{msg}");
        }

        if let Some(log_file) = &mut self.log_writer {
            // ignore errors, we don't want to add additional server crashes
            let r0 = writeln!(log_file, "{msg}");
            let r1 = log_file.flush();
            if r0.is_err() || r1.is_err() {
                eprintln!("error writing to log file: {r0:?} {r1:?}");
            }
        }
    }
}
