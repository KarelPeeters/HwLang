use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope;
use crate::resolve::scope::Visibility;
use crate::resolve::types::ItemReference;
use crate::resolve::values::{ResolvedValue, ResolvedValueInfo, ResolvedValues};
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
enum ScopedValue {
    Item(Item),
    File(FileId),
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

    signature: Option<ResolvedValue>,
    content: Option<ResolvedValue>,
}

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<ResolveQuery>),
    InvalidTypeExpression(Expression),
    MissingTypeExpression(ItemDefType),
    UnexpectedPathToFile(Path, FileId),
    UnexpectedPathToItem(Path, Item),
}

pub struct CompileState<'a> {
    files: &'a IndexMap<FileId, FileInfo>,
    items: Arena<Item, ItemInfo>,
    values: ResolvedValues,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ResolveQuery {
    item: Item,
    kind: ResolveQueryKind,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ResolveQueryKind {
    Signature,
    Content,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ResolveFirst(ResolveQuery);

#[derive(Debug, Copy, Clone)]
enum ResolveFirstOr<E> {
    ResolveFirst(ResolveQuery),
    Error(E),
}

impl CompileSet {
    pub fn compile(mut self) -> Result<(), CompileError> {
        // Use separate steps to flag all errors of the same type before moving on to the next level.
        // TODO: some error recovery and continuation, eg. return all parse errors at once

        // items only exists to serve as a level of indirection between values,
        //   so we can easily do the graph solution in a single pass
        let mut items: Arena<Item, ItemInfo> = Arena::default();
        let values = ResolvedValues::default();

        // parse all files
        for file in self.files.values_mut() {
            let ast = parse_file_content(&file.source, file.id)?;
            file.ast = Some(ast);
        }

        // build import scope of each file
        for file in self.files.values_mut() {
            let ast = file.ast.as_ref().unwrap();

            // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
            let mut scope_declared = Scope::default();

            for (ast_item_index, ast_item) in enumerate(&ast.items) {
                let (id, ast_vis) = ast_item.id_vis();
                let vis = match ast_vis {
                    ast::Visibility::Public(_) => Visibility::Public,
                    ast::Visibility::Private => Visibility::Private,
                };

                let item_reference = ItemReference { file: file.id, item_index: ast_item_index };
                let item = items.push(ItemInfo { item_reference, signature: None, content: None });
                scope_declared.maybe_declare(&id, ScopedValue::Item(item), vis)?;
            }

            file.local_scope = Some(scope_declared);
        }

        let mut state = CompileState {
            files: &self.files,
            items,
            values,
        };

        // fully resolve all items
        let item_keys = state.items.keys().collect_vec();
        for item in item_keys {
            println!("Starting full resolution of {:?}", item);
            let query = ResolveQuery { item, kind: ResolveQueryKind::Content };
            state.resolve_fully(query)?;
        }

        Ok(())
    }
}

impl CompileState<'_> {
    fn resolve_fully(&mut self, query: ResolveQuery) -> Result<(), CompileError> {
        let mut stack = vec![query];

        while let Some(curr) = stack.pop() {
            if self.resolve(curr).is_ok() {
                continue;
            }

            let result = self.resolve_new(curr);

            match result {
                Ok(resolved) => {
                    println!("Resolved {:?} to {:?}", query, resolved);
                    *curr.kind.slot_mut(&mut self.items[curr.item]) = Some(resolved);
                }
                Err(ResolveFirstOr::Error(e)) => {
                    return Err(e.into());
                }
                Err(ResolveFirstOr::ResolveFirst(first)) => {
                    stack.push(curr);
                    let has_cycle = stack.contains(&first);
                    stack.push(first);
                    if has_cycle {
                        return Err(FrontError::CyclicTypeDependency(stack).into());
                    }
                }
            }
        }

        Ok(())
    }

    fn resolve(&self, query: ResolveQuery) -> Result<ResolvedValue, ResolveFirst> {
        match query.kind.slot(&self.items[query.item]) {
            &Some(result) => Ok(result),
            None => Err(ResolveFirst(query))
        }
    }

    fn resolve_new(&mut self, query: ResolveQuery) -> Result<ResolvedValue, ResolveFirstOr<CompileError>> {
        // check that this is indeed a new query
        assert!(query.kind.slot(&self.items[query.item]).is_none());

        let ResolveQuery { item: curr_item, kind: curr_kind } = query;

        // item lookup
        let info = &self.items[curr_item];
        let item_reference = info.item_reference;
        let ItemReference { file, item_index } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        let item_ast = &file_info.ast.as_ref().unwrap().items[item_index];
        let scope = file_info.local_scope.as_ref().unwrap();

        // actual resolution
        let resolved= match item_ast {
            ast::Item::Use(item_ast) => {
                let next_item = self.follow_path(scope, &item_ast.path)?;
                self.resolve(ResolveQuery { item: next_item, kind: curr_kind })?
            }
            ast::Item::Type(item_ast) => {
                assert!(item_ast.params.params.is_empty());

                match query.kind {
                    ResolveQueryKind::Signature => self.values.push(ResolvedValueInfo::SignatureType),
                    ResolveQueryKind::Content => todo!(),
                }
            },
            ast::Item::Struct(_) => todo!(),
            ast::Item::Enum(_) => todo!(),
            ast::Item::Const(_) => todo!(),
            ast::Item::Function(_) => todo!(),
            ast::Item::Module(_) => todo!(),
            ast::Item::Interface(_) => todo!(),
        };

        Ok(resolved)
    }

    fn follow_path(&self, scope: &Scope, path: &Path) -> Result<Item, ResolveFirstOr<CompileError>> {
        let mut scope = scope;
        let mut vis = Visibility::Private;

        for parent in &path.parents {
            match *scope.find(None, &parent, vis)? {
                ScopedValue::Item(item) =>
                    return Err(FrontError::UnexpectedPathToItem(path.clone(), item).into()),
                ScopedValue::File(file) => {
                    scope = self.files.get(&file).unwrap().local_scope.as_ref().unwrap();
                    vis = Visibility::Public;
                }
            }
        }

        match *scope.find(None, &path.id, vis)? {
            ScopedValue::Item(item) =>
                Ok(item),
            ScopedValue::File(file) =>
                Err(FrontError::UnexpectedPathToFile(path.clone(), file).into()),
        }
    }

    fn eval_expression(&self, scope: &Scope, expr: &Expression) -> Result<ResolvedValue, ResolveFirstOr<CompileError>> {
        todo!()
    }
}

impl ResolveQueryKind {
    pub fn slot(self, info: &ItemInfo) -> &Option<ResolvedValue> {
        match self {
            ResolveQueryKind::Signature => &info.signature,
            ResolveQueryKind::Content => &info.content,
        }
    }

    pub fn slot_mut(self, info: &mut ItemInfo) -> &mut Option<ResolvedValue> {
        match self {
            ResolveQueryKind::Signature => &mut info.signature,
            ResolveQueryKind::Content => &mut info.content,
        }
    }
}

impl<E: Into<CompileError>> From<E> for ResolveFirstOr<CompileError> {
    fn from(value: E) -> Self {
        ResolveFirstOr::Error(value.into())
    }
}

impl From<ResolveFirst> for ResolveFirstOr<CompileError> {
    fn from(value: ResolveFirst) -> Self {
        ResolveFirstOr::ResolveFirst(value.0)
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
