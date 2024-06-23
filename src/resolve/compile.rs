use std::io::Write;
use indexmap::{IndexMap, IndexSet};
use crate::new_index_type;
use crate::resolve::scope::Scope;
use crate::syntax::pos::FileId;
use crate::syntax::ast;
use crate::util::arena::Arena;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FilePath(pub Vec<String>);

pub struct CompileSet {
    pub paths: IndexSet<FilePath>,
    // TODO use arena for this too?
    pub files: IndexMap<FileId, FileInfo>,
}

pub struct FileInfo {
    file_id: FileId,
    path: FilePath,
    source: String,
    ast: Option<ast::FileContent>,
    module: Option<Module>,
}

new_index_type!(Module);

pub struct ModuleInfo {
    path: FilePath,
    file: Option<FileId>,
    scope: Scope<'static, ScopedValue>,
}

pub struct Node {
    children: IndexMap<String, Node>,
    module: Module,
}

#[derive(Debug)]
pub enum ScopedValue {
    Module(Module),
    Item(/*TODO*/)
}

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
        let info = FileInfo {
            file_id: id,
            source,
            path,
            module: None,
            ast: None,
        };

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
    pub fn compile(&mut self) {
        // create all modules
        let mut modules: Arena<Module, ModuleInfo> = Arena::default();

        let root_module = modules.push(ModuleInfo {
            path: FilePath(vec![]),
            file: None,
            scope: Default::default(),
        });
        let mut root = Node { children: Default::default(), module: root_module };

        for info in self.files.values() {
            let mut node = &mut root;
            for elem in &(&info.path).0 {
                let child = node.children.entry(elem.clone()).or_insert_with(|| {
                    let module = modules.push(ModuleInfo {
                        path: info.path.clone(),
                        file: Some(info.file_id),
                        scope: Default::default(),
                    });
                    Node { children: Default::default(), module }
                });
                node = child;
            }
        }

        // populate module scopes with root and sibling paths
        recurse_populate_children(&root, &mut modules);
    }
}

fn recurse_populate_children(node: &Node, modules: &mut Arena<Module, ModuleInfo>) {
    // add children to scope
    let info = &mut modules[node.module];
    for (name, child) in &node.children {
        info.scope.declare_str(name, ScopedValue::Module(child.module));
    }

    // recurse
    for child in node.children.values() {
        recurse_populate_children(child, modules);
    }
}
