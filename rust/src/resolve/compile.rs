use std::net::Incoming;
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope;
use crate::resolve::scope::Visibility;
use crate::resolve::types::{ItemReference, Type, Types};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Expression, ItemDefType, Path};
use crate::syntax::pos::FileId;
use crate::util::arena::Arena;

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
    local_scope: Option<Scope<'static>>,
}

pub struct ModuleInfo {
    path: FilePath,
    file: Option<FileId>,
    scope: Scope<'static>,
}

pub struct Node {
    children: IndexMap<String, Node>,
}

type Scope<'s> = scope::Scope<'s, ScopedValue>;

#[derive(Debug, Copy, Clone)]
pub enum ScopedValue {
    File(FileId),
    Item(Item),
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

new_index_type!(pub Item);

#[derive(Debug)]
pub struct ItemInfo {
    item_reference: ItemReference,
    resolved: Option<ResolvedValue>,
}

#[derive(Debug, Copy, Clone)]
pub enum ResolvedValue {
    Type(Type),
    Const(/*TODO*/),
    Function(/*TODO*/),
    Module(/*TODO*/),
    Interface(/*TODO*/),
    // TODO others, in particular constants
}

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<Item>),
    InvalidTypeExpression(Expression),
    MissingTypeExpression(ItemDefType),
}

pub struct CompileState<'a> {
    files: &'a IndexMap<FileId, FileInfo>,
    items: Arena<Item, ItemInfo>,
    types: Types,
}

type CheckResult<T> = Result<T, CheckFirstOr<FrontError>>;

#[derive(Debug, Copy, Clone)]
struct CheckFirst(Item);

#[derive(Debug, Copy, Clone)]
enum CheckFirstOr<E> {
    CheckFirst(CheckFirst),
    Error(E),
}

impl CompileSet {
    pub fn compile(mut self) -> Result<(), CompileError> {
        // Use separate steps to flag all errors of the same type before moving on to the next level.
        // TODO: some error recovery and continuation, eg. return all parse errors at once

        let mut items: Arena<Item, ItemInfo> = Arena::default();

        // parse all files
        for file in self.files.values_mut() {
            let ast = parse_file_content(&file.source, file.id)?;
            file.ast = Some(ast);
        }

        // build import scope of each file
        for file in self.files.values_mut() {
            let ast = file.ast.as_ref().unwrap();

            let mut scope_declared = Scope::default();

            for (ast_item_index, ast_item) in enumerate(&ast.items) {
                let (id, ast_vis) = ast_item.id_vis();
                let vis = match ast_vis {
                    ast::Visibility::Public(_) => Visibility::Public,
                    ast::Visibility::Private => Visibility::Private,
                };

                let item_reference = ItemReference { file: file.id, item_index: ast_item_index };
                let item = items.push(ItemInfo { item_reference, resolved: None });
                scope_declared.maybe_declare(&id, ScopedValue::Item(item), vis)?;
            }

            file.local_scope = Some(scope_declared);
        }

        // visit all items
        let mut state = CompileState {
            files: &self.files,
            items,
            types: Types::default(),
        };
        let item_keys = state.items.keys().collect_vec();

        for item in item_keys {
            let mut stack = vec![item];
            while let Some(curr) = stack.pop() {
                // TODO separate signature and content as visit actions
                if state.items[item].resolved.is_some() {
                    continue;
                }

                let result = state.resolve_item(curr);

                match result {
                    Ok(resolved) => {
                        assert!(state.items[item].resolved.is_none());
                        state.items[item].resolved = Some(resolved);
                        
                        println!("Resolved {:?} to {:?}", item, resolved);
                    }
                    Err(CheckFirstOr::Error(e)) => {
                        return Err(e.into());
                    }
                    Err(CheckFirstOr::CheckFirst(CheckFirst(first))) => {
                        stack.push(curr);
                        if stack.contains(&first) {
                            stack.push(first);
                            return Err(FrontError::CyclicTypeDependency(stack).into());
                        }
                        stack.push(first);
                    }
                }
            }

            // TODO assert more stuff?
            assert!(state.items[item].resolved.is_some());
        }

        Ok(())
    }
}

impl CompileState<'_> {
    fn resolve_item(&mut self, item: Item) -> Result<ResolvedValue, CheckFirstOr<CompileError>> {
        assert!(self.items[item].resolved.is_none());

        // item lookup
        let info = &self.items[item];
        let item_reference = info.item_reference;
        let ItemReference { file, item_index } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        let item_ast = &file_info.ast.as_ref().unwrap().items[item_index];
        let scope = file_info.local_scope.as_ref().unwrap();

        // actual item
        let resolved= match item_ast {
            ast::Item::Use(item_ast) => {
                self.resolve_path(scope, &item_ast.path)?
            }
            ast::Item::Type(item_ast) => { 
                assert!(item_ast.params.params.is_empty());
                let expr = item_ast.inner.as_ref().ok_or(FrontError::MissingTypeExpression(item_ast.clone()))?;
                self.resolve_expression(scope, expr)?
            },
            ast::Item::Struct(_) => todo!(),
            ast::Item::Enum(_) => todo!(),
            ast::Item::Const(_) => ResolvedValue::Const(),
            ast::Item::Function(_) => ResolvedValue::Function(),
            ast::Item::Module(_) => ResolvedValue::Module(),
            ast::Item::Interface(_) => ResolvedValue::Interface(),
        };

        Ok(resolved)
    }

    fn resolve_path(&mut self, scope: &Scope, path: &Path) -> Result<ResolvedValue, CheckFirst> {
        todo!()
    }

    fn get_resolved(&mut self, item: Item) -> Result<ResolvedValue, CheckFirst> {
        self.items[item].resolved.ok_or(CheckFirst(item))
    }
    
    fn resolve_expression(&mut self, scope: &Scope, expr: &Expression) -> Result<ResolvedValue, CheckFirstOr<CompileError>> {
        todo!()
    }
}

impl<E: Into<CompileError>> From<E> for CheckFirstOr<CompileError> { 
    fn from(value: E) -> Self {
        CheckFirstOr::Error(value.into())
    }
}

impl <E> From<CheckFirst> for CheckFirstOr<E> {
    fn from(first: CheckFirst) -> Self {
        CheckFirstOr::CheckFirst(first)
    }
}

impl FileInfo {
    pub fn new(id: FileId, path: FilePath, source: String) -> Self {
        FileInfo {
            id,
            path,
            source,
            ast: None,
            local_scope: None,
        }
    }
}
