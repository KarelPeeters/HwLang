//! Crate which contains the stdlib source code as constants,
//! so the compile can access them while still being a standalone binary.
//!
//! This is separate from the `hwl_language` create to allow for separate build scripts,
//! so the grammar does not get recompiled every time the stdlib changes.

#[derive(Debug)]
pub struct StdSourceFile {
    pub path: &'static str,
    pub steps: &'static [&'static str],
    pub content: &'static str,
}

pub const STD_SOURCE_FILES: &[StdSourceFile] = include!(concat!(env!("OUT_DIR"), "/std_source_files.in"));
