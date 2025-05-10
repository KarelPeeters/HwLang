pub mod logger;
pub mod sender;

pub mod dispatch;
pub mod settings;
pub mod state;

// The different parts of the LSP protocol are spread over the following modules, mirroring the chapters in
// <https://microsoft.github.io/language-server-protocol/specifications/lsp/>
pub mod document;
pub mod language;
pub mod lifecycle;
