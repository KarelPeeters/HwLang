use std::io::Write;
use indexmap::IndexMap;
use crate::syntax::{parse_file_content, ParseError};
use crate::syntax::pos::FileId;
use crate::syntax::ast;

pub struct CompileSet {
    pub files: IndexMap<Vec<String>, (String, ast::FileContent)>,
}

pub struct CheckedCompileSet {

}

impl CompileSet {
    pub fn new() -> Self {
        CompileSet {
            files: Default::default(),
        }
    }

    pub fn add_file(&mut self, path: Vec<String>, source: String) -> Result<(), ParseError> {
        let file_id = FileId(self.files.len());
        let parsed = parse_file_content(&source, file_id)?;
        let mut prev = self.files.insert(path, (source, parsed));
        // TODO properly handle this error
        assert!(prev.is_none());
        Ok(())
    }

    pub fn add_external_vhdl(&mut self, library: String, source: String) {
        todo!()
    }

    pub fn add_external_verilog(&mut self, source: String) {
        todo!()
    }

    pub fn check(&self) -> CheckedCompileSet {
        // Steps:
        // * create scopes for all the modules, populated with placeholders for each of:
        //   * root libraries (including external)
        //   * items defined in the current file (including imports)

        // * step over each item and resolve the signature, following and breaking cycles when possible
        // * step over each item and resolve and typecheck the body
        CheckedCompileSet {}

        /* old item list;
        //   * root files
        //   * child files
        //     -> should these exist at all? we want to avoid non-local references,
        //        ideally every id that's accessible is mentioned in the file
        //   * imports in the current file
        //   * items defined in the current file
         */
    }
}

impl CheckedCompileSet {
    pub fn export(&self, f: impl Write) -> std::fmt::Result {
        todo!()
    }
}
