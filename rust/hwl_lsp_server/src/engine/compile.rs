use crate::server::state::{RequestResult, ServerState};
use hwl_language::constants::LANGUAGE_FILE_EXTENSION;
use hwl_language::data::compiled::CompiledDatabase;
use hwl_language::data::source::{FilePath, SourceDatabase};
use hwl_language::error::CompileResult;
use hwl_language::front::driver::compile;
use hwl_language::try_inner;
use std::path::Component;

impl ServerState {
    pub fn compile_project(&mut self) -> RequestResult<CompileResult<()>> {
        // build source database
        let vfs = self.vfs.inner()?;
        let mut source = SourceDatabase::new();

        for (path, content) in vfs.iter() {
            let mut steps = vec![];
            for c in path.components() {
                match c {
                    Component::Normal(c) => steps.push(c.to_str()
                        .expect("the vfs layer should already check for non-utf8 paths ")
                        .to_owned()
                    ),
                    _ => unreachable!("the vfs path should be a simple relative path, got {:?}", path)
                }
            }

            // remove final extension
            let last = steps.last_mut().unwrap();
            *last = last.strip_suffix(LANGUAGE_FILE_EXTENSION)
                .expect("only language source files should be in the VFS")
                .to_owned();

            let text = content.get_text(path)?;
            try_inner!(source.add_file(FilePath(steps), path.to_str().unwrap().to_owned(), text.to_owned()));
        }

        // actual compilation
        let _: CompiledDatabase = try_inner!(compile(&source));

        Ok(Ok(()))
    }
}
