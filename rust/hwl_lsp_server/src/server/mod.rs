pub mod logger;
pub mod sender;

pub mod settings;
pub mod state;
pub mod dispatch;

// The different parts of the LSP protocol are spread over the following modules, mirroring the chapters in
// <https://microsoft.github.io/language-server-protocol/specifications/lsp/>
pub mod lifecycle;
pub mod language;
pub mod document;
// pub mod wrapper;
// pub mod workspace;
