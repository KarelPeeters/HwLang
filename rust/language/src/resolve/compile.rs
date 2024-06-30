use std::collections::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope;
use crate::resolve::scope::Visibility;
use crate::resolve::types::{BasicTypes, ItemReference, Type, TypeInfo, TypeInfoFunction, Types};
use crate::resolve::values::{Value, ValueFunctionInfo, ValueInfo, Values};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Args, Expression, ExpressionKind, Identifier, ItemDefEnum, ItemDefStruct, ItemDefType, Path, Spanned, TypeParam};
use crate::syntax::pos::FileId;
use crate::util::arena::Arena;

macro_rules! throw {
    ($e:expr) => { return Err($e.into()) };
}

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

pub type Scope<'s> = scope::Scope<'s, ScopedValue>;

#[derive(Debug, Copy, Clone)]
pub enum ScopedValue {
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
            throw!(CompileSetError::EmptyPath);
        }
        if !self.paths.insert(path.clone()) {
            throw!(CompileSetError::DuplicatePath(path.clone()));
        }

        let id = FileId(self.files.len());
        println!("adding {:?} => {:?}", id, path);
        let info = FileInfo::new(id, path, source);

        let prev = self.files.insert(id, info);
        assert!(prev.is_none());
        Ok(())
    }

    pub fn add_external_vhdl(&mut self, library: String, source: String) {
        todo!("compile VHDL library={:?}, source.len={}", library, source.len())
    }

    pub fn add_external_verilog(&mut self, source: String) {
        todo!("compile verilog source.len={}", source.len())
    }
}

new_index_type!(pub Item);

#[derive(Debug)]
pub struct ItemInfo {
    item_reference: ItemReference,

    signature: Option<Value>,
    content: Option<Value>,
}

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<ResolveQuery>),

    ExpectedTypeExpression(Expression),
    ExpectedFunctionExpression(Expression),

    ExpectedPathToItemNotFile(Path, FileId),
    ExpectedPathToFileNotItem(Path, Item),

    InvalidBuiltinIdentifier(Expression, Identifier),
    InvalidBuiltinArgs(Expression, Args),

    DuplicateParameterName(Identifier, Identifier),
}

pub struct CompileState<'a> {
    files: &'a IndexMap<FileId, FileInfo>,
    items: Arena<Item, ItemInfo>,
    values: Values,
    types: Types,
    basic_values: BasicTypes<Value>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ResolveQuery {
    item: Item,
    kind: ResolveQueryKind,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ResolveQueryKind {
    Signature,
    Value,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ResolveFirst(ResolveQuery);

#[derive(Debug, Copy, Clone)]
pub enum ResolveFirstOr<E> {
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
        let mut values = Values::default();
        let types = Types::default();

        // parse all files
        for file in self.files.values_mut() {
            let ast = parse_file_content(file.id, &file.source)?;
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

        let basic_values = types.basic().map(|&ty| values.push(ValueInfo::Type(ty)));

        let mut state = CompileState {
            files: &self.files,
            items,
            values,
            types,
            basic_values
        };

        // fully resolve all items
        let item_keys = state.items.keys().collect_vec();
        for item in item_keys {
            println!("Starting full resolution of {:?}", item);
            let query = ResolveQuery { item, kind: ResolveQueryKind::Value };
            state.resolve_fully(query)?;
        }

        Ok(())
    }
}

pub type ResolveResult<T> = Result<T, ResolveFirstOr<CompileError>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum FunctionBody {
    /// type alias, enum, or struct
    TypeConstructor(ItemReference),
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
                    println!("  resolved {:?} to {:?}", query, resolved);
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

    fn resolve(&self, query: ResolveQuery) -> Result<Value, ResolveFirst> {
        match query.kind.slot(&self.items[query.item]) {
            &Some(result) => Ok(result),
            None => Err(ResolveFirst(query))
        }
    }

    fn resolve_new(&mut self, query: ResolveQuery) -> ResolveResult<Value> {
        // check that this is indeed a new query
        assert!(query.kind.slot(&self.items[query.item]).is_none());

        // item lookup
        let ResolveQuery { item, kind } = query;
        let info = &self.items[item];
        let item_reference = info.item_reference;
        let ItemReference { file, item_index } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        let item_ast = &file_info.ast.as_ref().unwrap().items[item_index];
        let scope = file_info.local_scope.as_ref().unwrap();

        println!("  resolve_new {:?} at {:?}", query, item_ast.span());

        // actual resolution
        let resolved= match item_ast {
            ast::Item::Use(item_ast) => {
                let next_item = self.follow_path(scope, &item_ast.path)?;
                self.resolve(ResolveQuery { item: next_item, kind })?
            }
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, params, inner: _ }) |
            ast::Item::Struct(ItemDefStruct { span: _, vis: _, id: _, params, fields: _ }) |
            ast::Item::Enum(ItemDefEnum { span: _, vis: _, id: _, params, variants: _ })
            => {
                self.resolve_new_type_def_item(query, scope, item_reference, params)?
            },
            ast::Item::Const(_) => todo!(),
            ast::Item::Function(_) => todo!(),
            ast::Item::Module(_) => todo!(),
            ast::Item::Interface(_) => todo!(),
        };

        Ok(resolved)
    }

    fn resolve_new_type_def_item(
        &mut self,
        query: ResolveQuery,
        scope: &Scope,
        item_reference: ItemReference,
        params: &Option<Spanned<Vec<TypeParam>>>,
    ) -> ResolveResult<Value> {
        let build_value = FunctionBody::TypeConstructor(item_reference);
        
        let value = match params {
            None => {
                // no params, this is just a straight type definition
                match query.kind {
                    ResolveQueryKind::Signature => self.type_as_value(self.types.basic().ty_type),
                    ResolveQueryKind::Value => self.call_function_body(build_value)?,
                }
            }
            Some(params) => {
                // params, this is a type constructor, equivalent to a function that returns a type
                let ty = self.type_constructor_signature(scope, &params.inner)?;
                match query.kind {
                    ResolveQueryKind::Signature => self.type_as_value(ty),
                    ResolveQueryKind::Value => {
                        // check parameter names for uniqueness
                        // TODO this is a weird place to do this,
                        //  can't we type-check the item itself including its header first?
                        let mut unique: HashMap<&str, &Identifier> = Default::default();
                        for p in &params.inner {
                            if let Some(prev)  = unique.insert(&p.id.string, &p.id) {
                                throw!(FrontError::DuplicateParameterName(prev.clone(), p.id.clone()))
                            }
                        }

                        // construct the actual function value
                        let params = params.inner.iter().map(|p| p.id.clone()).collect_vec();
                        let func = ValueFunctionInfo { item_reference, ty, params, body: build_value };
                        self.values.push(ValueInfo::Function(func))
                    }
                }
            }
        };

        Ok(value)
    }

    fn type_constructor_signature(&mut self, scope: &Scope, params: &Vec<TypeParam>) -> ResolveResult<Type> {
        let mut param_types = vec![];
        for param in params {
            let TypeParam { id: _, ty, span: _ } = param;
            let param_ty = self.eval_expression_as_ty(scope, ty)?;
            param_types.push(param_ty);
        }
        let ty_info = TypeInfo::Function(TypeInfoFunction {
            params: param_types,
            ret: self.types.basic().ty_type,
        });
        Ok(self.types.push(ty_info))
    }

    fn follow_path(&self, scope: &Scope, path: &Path) -> ResolveResult<Item> {
        let mut scope = scope;
        let mut vis = Visibility::Private;

        for parent in &path.parents {
            match *scope.find(None, &parent, vis)? {
                ScopedValue::Item(item) =>
                    return Err(FrontError::ExpectedPathToFileNotItem(path.clone(), item).into()),
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
                Err(FrontError::ExpectedPathToItemNotFile(path.clone(), file).into()),
        }
    }

    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Value> {
        match &expr.inner {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Path(_) => todo!(),
            ExpressionKind::Wrapped(_) => todo!(),
            ExpressionKind::Type => Ok(self.basic_values.ty_type),
            ExpressionKind::TypeFunc(_, _) => todo!(),
            ExpressionKind::Block(_) => todo!(),
            ExpressionKind::If(_) => todo!(),
            ExpressionKind::Loop(_) => todo!(),
            ExpressionKind::While(_) => todo!(),
            ExpressionKind::For(_) => todo!(),
            ExpressionKind::Sync(_) => todo!(),
            ExpressionKind::Return(_) => todo!(),
            ExpressionKind::Break(_) => todo!(),
            ExpressionKind::Continue => todo!(),
            ExpressionKind::IntPattern(_) => todo!(),
            ExpressionKind::BoolLiteral(_) => todo!(),
            ExpressionKind::StringLiteral(_) => todo!(),
            ExpressionKind::ArrayLiteral(_) => todo!(),
            ExpressionKind::TupleLiteral(_) => todo!(),
            ExpressionKind::StructLiteral(_) => todo!(),
            ExpressionKind::Range { .. } => todo!(),
            ExpressionKind::UnaryOp(_, _) => todo!(),
            ExpressionKind::BinaryOp(_, _, _) => todo!(),
            ExpressionKind::TernarySelect(_, _, _) => todo!(),
            ExpressionKind::ArrayIndex(_, _) => todo!(),
            ExpressionKind::DotIdIndex(_, _) => todo!(),
            ExpressionKind::DotIntIndex(_, _) => todo!(),
            ExpressionKind::Call(target, args) => {
                if let ExpressionKind::Path(Path { parents, id, span: _ }) = &target.inner {
                    if parents.is_empty() {
                        if let Some(name) = id.string.strip_prefix("__builtin_") {
                            return self.eval_builtin_call(expr, name, id, args);
                        }
                    }
                }

                let target_value = self.eval_expression(scope, target)?;
                match &self.values[target_value] {
                    ValueInfo::Function(target_value) => {
                        todo!()
                    }
                    _ => throw!(FrontError::ExpectedFunctionExpression((&**target).clone())),
                }
            },
        }
    }

    fn eval_expression_as_ty(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Type> {
        let value = self.eval_expression(scope, expr)?;
        match &self.values[value] {
            &ValueInfo::Type(ty) => Ok(ty),
            _ => Err(FrontError::ExpectedTypeExpression(expr.clone()).into()),
        }
    }

    fn eval_builtin_call(&mut self, expr: &Expression, name: &str, id: &Identifier, args: &Args) -> ResolveResult<Value> {
        // TODO disallow calling builtin in user modules?
        match name {
            "type" => {
                if args.inner.len() >= 1 {
                    if let ExpressionKind::StringLiteral(ty) = &args.inner[0].inner {
                        match ty.as_str() {
                            "bool" if args.inner.len() == 1 => {
                                return Ok(self.basic_values.ty_bool);
                            }
                            _ => {},
                        }
                    }
                }

                throw!(FrontError::InvalidBuiltinArgs(expr.clone(), args.clone()));
            }
            _ => throw!(FrontError::InvalidBuiltinIdentifier(expr.clone(), id.clone())),
        }
    }

    fn type_as_value(&mut self, ty: Type) -> Value {
        self.values.push(ValueInfo::Type(ty))
    }

    fn call_function_body(&mut self, body: FunctionBody) -> ResolveResult<Value> {
        match body {
            FunctionBody::TypeConstructor(item_reference) => {
                let ItemReference { file, item_index } = item_reference;
                let file_info = self.files.get(&file).unwrap();
                let item_ast = &file_info.ast.as_ref().unwrap().items[item_index];
                let scope = file_info.local_scope.as_ref().unwrap();

                match item_ast {
                    ast::Item::Type(item_ast) => {
                        let ty = self.eval_expression_as_ty(scope, &item_ast.inner)?;
                        Ok(self.type_as_value(ty))
                    }
                    ast::Item::Struct(_) => todo!(),
                    ast::Item::Enum(_) => todo!(),
                    _ => unreachable!(),
                }
            },
        }
    }
}

impl ResolveQueryKind {
    pub fn slot(self, info: &ItemInfo) -> &Option<Value> {
        match self {
            ResolveQueryKind::Signature => &info.signature,
            ResolveQueryKind::Value => &info.content,
        }
    }

    pub fn slot_mut(self, info: &mut ItemInfo) -> &mut Option<Value> {
        match self {
            ResolveQueryKind::Signature => &mut info.signature,
            ResolveQueryKind::Value => &mut info.content,
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
