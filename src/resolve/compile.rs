use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope;
use crate::resolve::scope::Visibility;
use crate::resolve::types::{ItemReference, Type, TypeEnum, TypeFunction, TypeInfo, Types, TypeStruct};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Expression, ExpressionKind, ItemDefType};
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
    ty: Option<Option<Type>>,
}

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<Item>),
    InvalidTypeExpression(Expression),
    MissingTypeExpression(ItemDefType),
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
                let item = items.push(ItemInfo { item_reference, ty: None });
                scope_declared.maybe_declare(&id, ScopedValue::Item(item), vis)?;
            }
            
            file.local_scope = Some(scope_declared);
        }

        // type-resolve all items
        // TODO rewrite this using coroutines?
        let mut types = Types::default();
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
        
        // visit the bodies of all items
        // TODO
        
        // TODO merge type checking and body visiting, we want them to be mixable

        Ok(())
    }

    fn find_path(&self, scope: &Scope, path: &ast::Path) -> Result<ScopedValue, CompileError> {
        todo!()
    }

    fn resolve_type_for_path(&self, types: &mut Types, scope: &Scope, path: &ast::Path) -> CheckResult<Type> {
        todo!()
    }

    /// Resolve an expression type as a type.
    /// This is not the type _of_ the given expression, the expression has to be a valid type by itself.
    fn resolve_type_for_expression(&self, types: &mut Types, scope: &Scope, expr: &Expression) -> CheckResult<Type> {
        match &expr.kind {
            ExpressionKind::Dummy => todo!("this will become a type var once we have type inference"),
            ExpressionKind::Path(path) => self.resolve_type_for_path(types, scope, path),
            ExpressionKind::Wrapped(inner) => self.resolve_type_for_expression(types, scope, inner),
            ExpressionKind::TypeFunc(params, ret) => {
                let params = params.iter().map(|p| self.resolve_type_for_expression(types, scope, p)).collect::<CheckResult<_>>()?;
                let ret = self.resolve_type_for_expression(types, scope, ret)?;
                Ok(types.push(TypeInfo::Function(TypeFunction { params, ret: Box::new(ret) })))
            },
            ExpressionKind::Call(_, _) => todo!(),
            ExpressionKind::TupleLiteral(_) => todo!(),

            // other expressions don't define types
            _ => Err(FrontError::InvalidTypeExpression(expr.clone()).into()),
        }
    }

    // TODO is this the type _of_ an item or the type that an item is by itself?
    // eg. functions don't declare a type, so disallow them
    fn resolve_type_for_item(&self, types: &mut Types, items: &Arena<Item, ItemInfo>, item: Item) -> CheckResult<Option<Type>> {
        let info = &items[item];

        // item lookup
        let item_reference = info.item_reference;
        let ItemReference { file, item_index } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        let ast_file = file_info.ast.as_ref().unwrap();
        let ast_item = &ast_file.items[item_index];
        let file_scope = file_info.local_scope.as_ref().unwrap();

        // actual type match
        let ty = match ast_item {
            ast::Item::Use(ast_item) =>
                Some(self.resolve_type_for_path(types, &file_scope, &ast_item.path)?),
            // TODO at what point should type parameters be filled in?
            //   should all of this machinery return type constructors instead?
            ast::Item::Type(ast_item) => {
                assert!(ast_item.params.params.is_empty());
                match &ast_item.inner {
                    None => 
                        return Err(FrontError::MissingTypeExpression(ast_item.clone()).into()),
                    Some(inner) =>
                        Some(self.resolve_type_for_expression(types, &file_scope, inner)?),
                }
            },
            ast::Item::Struct(ast_item) => {
                assert!(ast_item.params.params.is_empty());
                Some(types.push(TypeInfo::Struct(TypeStruct { item_reference })))
            }
            ast::Item::Enum(ast_item) => {
                assert!(ast_item.params.params.is_empty());
                Some(types.push(TypeInfo::Enum(TypeEnum { item_reference })))
            }

            // these items don't define types
            ast::Item::Const(_) => None,
            ast::Item::Module(_) => None,
            ast::Item::Interface(_) => None,
            ast::Item::Function(_) => None,
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

impl<E> From<E> for CheckFirstOr<E> {
    fn from(value: E) -> Self {
        CheckFirstOr::Error(value)
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
