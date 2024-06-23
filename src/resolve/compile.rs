use indexmap::{IndexMap, IndexSet};

use crate::error::CompileError;
use crate::resolve::scope::{Scope, Visibility};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::pos::FileId;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FilePath(pub Vec<String>);

pub struct CompileSet {
    pub paths: IndexSet<FilePath>,
    // TODO use arena for this too?
    pub files: IndexMap<FileId, FileInfo>,
}

pub struct FileInfo {
    id: FileId,
    path: FilePath,
    source: String,
    ast: Option<ast::FileContent>,
    scope_declared: Option<Scope<'static, ScopedItem>>,
    item_placeholders: Option<Vec<PlaceholderItem>>,
}

pub struct ModuleInfo {
    path: FilePath,
    file: Option<FileId>,
    scope: Scope<'static, ScopedItem>,
}

pub struct Node {
    children: IndexMap<String, Node>,
}

#[derive(Debug, Copy, Clone)]
pub enum ScopedItem {
    File(FileId),
    Placeholder(PlaceholderItem),
}

#[derive(Debug, Copy, Clone)]
pub struct PlaceholderItem(u64);

#[derive(Debug)]
pub enum CompileSetError {
    EmptyPath,
    DuplicatePath(FilePath),
}

impl CompileSet {
    pub fn new() -> Self {
        CompileSet {
            files: IndexMap::default(),
            paths: IndexSet::default(),
        }
    }

    pub fn add_file(&mut self, path: FilePath, source: String) -> Result<(), CompileSetError> {
        if path.0.is_empty() {
            return Err(CompileSetError::EmptyPath);
        }
        if !self.paths.insert(path.clone()) {
            return Err(CompileSetError::DuplicatePath(path.clone()));
        }

        let id = FileId(self.files.len());
        println!("adding {:?} => {:?}", id, path);
        let info = FileInfo::new(id, path, source);
        
        let prev = self.files.insert(id, info);
        assert!(prev.is_none());
        Ok(())
    }

    pub fn add_external_vhdl(&mut self, library: String, source: String) {
        todo!()
    }

    pub fn add_external_verilog(&mut self, source: String) {
        todo!()
    }
}

impl CompileSet {
    pub fn compile(mut self) -> Result<(), CompileError> {
        // Use separate steps to flag all errors of the same type before moving on to the next level.
        // TODO: some error recovery and continuation, eg. return all parse errors at once
        
        let mut next_placeholder = 0;
        let mut next_placeholder = move || {
            let i = next_placeholder;
            next_placeholder += 1;
            PlaceholderItem(i)
        };

        // parse all files
        for file in self.files.values_mut() {
            let ast = parse_file_content(&file.source, file.id)?;
            file.ast = Some(ast);
        }

        // build import scope of each file
        for file in self.files.values_mut() {
            let ast = file.ast.as_ref().unwrap();
            
            let mut scope_declared = Scope::default();
            let mut item_placeholders = vec![];
            
            for item in &ast.items {
                let (id, ast_vis) = item.id_vis();
                let vis = match ast_vis {
                    ast::Visibility::Public(_) => Visibility::Public, 
                    ast::Visibility::Private => Visibility::Private,
                };
                
                let placeholder = next_placeholder();
                item_placeholders.push(placeholder);
                scope_declared.maybe_declare(&id, ScopedItem::Placeholder(placeholder), vis)?;
            }
            
            file.scope_declared = Some(scope_declared);
            file.item_placeholders = Some(item_placeholders);
        }
        
        // type-check everything
        // TODO support recursion between files

        Ok(())
    }
}

impl FileInfo {
    pub fn new(id: FileId, path: FilePath, source: String) -> Self {
        FileInfo {
            id,
            path,
            source,
            ast: None,
            scope_declared: None,
            item_placeholders: None,
        }
    }
}
