use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope::{Scope, Visibility};
use crate::resolve::types::{Type, TypeArena, TypeFunction, TypeInfo, TypeModule};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::Expression;
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
    local_scope: Option<Scope<'static, ScopedValue>>,
}

pub struct ModuleInfo {
    path: FilePath,
    file: Option<FileId>,
    scope: Scope<'static, ScopedValue>,
}

pub struct Node {
    children: IndexMap<String, Node>,
}

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
    file: FileId,
    ast_item_index: usize,
    ty: Option<Type>,
}

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<Item>),
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

                let item = items.push(ItemInfo { file: file.id, ast_item_index, ty: None });
                scope_declared.maybe_declare(&id, ScopedValue::Item(item), vis)?;
            }
            
            file.local_scope = Some(scope_declared);
        }

        // type-resolve all items
        // TODO rewrite this using coroutines?
        let mut types = TypeArena::default();
        let item_keys = items.keys().collect_vec();
        for item in item_keys {
            let mut stack = vec![item];
            while let Some(curr) = stack.pop() {
                if items[curr].ty.is_some() {
                    continue;
                }

                let result = self.resolve_type_for_item(&mut types, &items, curr);
                
                match result {
                    Ok(ty) => {
                        assert!(items[item].ty.is_none());
                        items[item].ty = Some(ty);
                    }
                    Err(CheckFirstOr::CheckFirst(CheckFirst(first))) => {
                        stack.push(curr);
                        if stack.contains(&first) {
                            stack.push(first);
                            return Err(FrontError::CyclicTypeDependency(stack).into());
                        }
                        stack.push(first);
                    }
                    Err(CheckFirstOr::Error(e)) => {
                        return Err(e.into());
                    }
                }
            }
            
            assert!(items[item].ty.is_some());
        }
        // TODO support recursion between files

        Ok(())
    }
    
    fn resolve_type_for_expression(&self, types: &mut TypeArena, expr: &Expression) -> CheckResult<Type> {
        todo!()
    }

    fn resolve_type_for_item(&self, types: &mut TypeArena, items: &Arena<Item, ItemInfo>, item: Item) -> CheckResult<Type>  {
        let info = &items[item];

        let file_info = self.files.get(&info.file).unwrap();
        let ast_file = file_info.ast.as_ref().unwrap();
        let ast_item = &ast_file.items[info.ast_item_index];
        let file_scope = file_info.local_scope.as_ref().unwrap();
        
        let ty = match ast_item {
            ast::Item::Use(ast_item) => todo!(),
            ast::Item::Const(ast_item) => todo!(),
            ast::Item::Type(ast_item) => todo!(),
            ast::Item::Struct(ast_item) => todo!(),
            ast::Item::Enum(ast_item) => todo!(),
            ast::Item::Function(ast_item) => {
                let mut params = vec![];
                for p in &ast_item.params.params {
                    params.push(self.resolve_type_for_expression(types, &p.ty)?);
                }
                let ret = match &ast_item.ret_ty {
                    None => types.push(TypeInfo::Void),
                    Some(ret_ty) => self.resolve_type_for_expression(types, ret_ty)?,
                };
                
                types.push(TypeInfo::Function(TypeFunction { params, ret: Box::new(ret) }))
            },
            ast::Item::Module(ast_item) => {
                types.push(TypeInfo::Module(TypeModule {}))
            },
            ast::Item::Interface(ast_item) => todo!(),
        };
        
        Ok(ty)
    }
}

type CheckResult<T> = Result<T, CheckFirstOr<FrontError>>;

#[derive(Debug, Copy, Clone)]
struct CheckFirst(Item);

#[derive(Debug, Copy, Clone)]
enum CheckFirstOr<E> {
    CheckFirst(CheckFirst),
    Error(E),
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
