use std::io::Write;
use crate::syntax::{parse_file_content, ParseError};
use crate::syntax::pos::FileId;

pub struct CompileSet {
    pub file_ids: Vec<Vec<String>>,
}

pub struct CheckedCompileSet {

}

impl CompileSet {
    pub fn new() -> Self {
        CompileSet {
            file_ids: vec![],
        }
    }

    pub fn add_file(&mut self, path: Vec<String>, source: String) -> Result<(), ParseError> {
        let file_id = FileId(self.file_ids.len());
        self.file_ids.push(path);
        let ast = parse_file_content(&source, file_id)?;
        // TODO do something with ast
        Ok(())
    }

    pub fn add_external_vhdl(&mut self, library: String, source: String) {
        todo!()
    }

    pub fn add_external_verilog(&mut self, source: String) {
        todo!()
    }

    pub fn check(&self) -> CheckedCompileSet {
        CheckedCompileSet {}
    }
}

impl CheckedCompileSet {
    pub fn export(&self, f: impl Write) -> std::fmt::Result {
        todo!()
    }
}
