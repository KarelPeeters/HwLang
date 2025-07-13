use std::io::{BufWriter, Write};
use std::path::Path;

pub struct Logger {
    log_to_stderr: bool,
    log_file: Option<BufWriter<std::fs::File>>,
}

impl Logger {
    pub fn new(log_to_stderr: bool, log_path: Option<&Path>) -> Self {
        let log_file = log_path.and_then(|log_path| match std::fs::File::create(log_path) {
            Ok(log_file) => Some(BufWriter::new(log_file)),
            Err(e) => {
                eprintln!("failed to open log file {log_path:?}: {e:?}");
                None
            }
        });
        Self {
            log_to_stderr,
            log_file,
        }
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let msg = msg.into();

        if self.log_to_stderr {
            eprintln!("{msg}");
        }

        if let Some(log_file) = &mut self.log_file {
            // ignore errors, we don't want to add additional server crashes
            let r0 = writeln!(log_file, "{msg}");
            let r1 = log_file.flush();
            if r0.is_err() || r1.is_err() {
                eprintln!("error writing to log file: {r0:?} {r1:?}");
            }
        }
    }
}
