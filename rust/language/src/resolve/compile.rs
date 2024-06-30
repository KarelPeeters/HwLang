use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope;
use crate::resolve::scope::Visibility;
use crate::resolve::types::{BasicTypes, ItemReference, Type, TypeInfo, TypeInfoFunction, Types};
use crate::resolve::values::{Value, ValueFunctionInfo, ValueInfo, Values};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Args, Expression, ExpressionKind, Identifier, ItemDefType, Path, TypeParam};
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

    signature: Option<Value>,
    content: Option<Value>,
}

#[derive(Debug)]
pub enum FrontError {
    CyclicTypeDependency(Vec<ResolveQuery>),
    InvalidTypeExpression(Expression),
    MissingTypeExpression(ItemDefType),
    UnexpectedPathToFile(Path, FileId),
    UnexpectedPathToItem(Path, Item),
    InvalidBuiltinIdentifier(Expression, Identifier),
    InvalidBuiltinArgs(Expression, Args),
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

type ResolveResult<T> = Result<T, ResolveFirstOr<CompileError>>;

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
            ast::Item::Type(item_ast) => {
                let ItemDefType { span: _, vis: _, id: _, params, inner } = item_ast;
                
                // let ty = self.define_maybe_type_constructor(scope, &params)?;
                
                let value_info = match params {
                    None => {
                        // no params, this is just a straight type definition
                        match query.kind {
                            ResolveQueryKind::Signature => ValueInfo::Type(self.types.basic().ty_type),
                            ResolveQueryKind::Value => ValueInfo::Type(self.eval_expression_as_ty(scope, inner)?),
                        }
                    }
                    Some(params) => {
                        // params, this is a type constructor, equivalent to a function that returns a type
                        let ty = self.type_constructor_signature(scope, &params.inner)?;
                        match query.kind {
                            ResolveQueryKind::Signature => ValueInfo::Type(ty),
                            ResolveQueryKind::Value => {
                                let func = ValueFunctionInfo { ty };
                                ValueInfo::Function(func)
                            }
                        }
                    }
                };
                
                self.values.push(value_info)
            },
            ast::Item::Struct(_) => todo!(),
            ast::Item::Enum(item_ast) => {
                // let ItemDefEnum { span: _, vis: _, id: _, params, variants } = item_ast;
                // let ty = self.define_maybe_type_constructor(scope, params)?;
                // 
                // match query.kind {
                //     ResolveQueryKind::Signature => {
                //         self.values.push(ValueInfo::Type(ty));
                //     }
                //     ResolveQueryKind::Value => {
                //         let _ = variants;
                //         let 
                //     }
                // }
                
                todo!()
            },
            ast::Item::Const(_) => todo!(),
            ast::Item::Function(_) => todo!(),
            ast::Item::Module(_) => todo!(),
            ast::Item::Interface(_) => todo!(),
        };

        Ok(resolved)
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

    fn eval_expression(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Value> {
        match &expr.inner {
            ExpressionKind::Dummy => todo!("ExpressionKind::Dummy at {:?}", expr.span),
            ExpressionKind::Path(_) => todo!("ExpressionKind::Path at {:?}", expr.span),
            ExpressionKind::Wrapped(_) => todo!("ExpressionKind::Wrapped at {:?}", expr.span),
            ExpressionKind::Type => todo!("ExpressionKind::Type at {:?}", expr.span),
            ExpressionKind::TypeFunc(_, _) => todo!("ExpressionKind::TypeFunc at {:?}", expr.span),
            ExpressionKind::Block(_) => todo!("ExpressionKind::Block at {:?}", expr.span),
            ExpressionKind::If(_) => todo!("ExpressionKind::If at {:?}", expr.span),
            ExpressionKind::Loop(_) => todo!("ExpressionKind::Loop at {:?}", expr.span),
            ExpressionKind::While(_) => todo!("ExpressionKind::While at {:?}", expr.span),
            ExpressionKind::For(_) => todo!("ExpressionKind::For at {:?}", expr.span),
            ExpressionKind::Sync(_) => todo!("ExpressionKind::Sync at {:?}", expr.span),
            ExpressionKind::Return(_) => todo!("ExpressionKind::Return at {:?}", expr.span),
            ExpressionKind::Break(_) => todo!("ExpressionKind::Break at {:?}", expr.span),
            ExpressionKind::Continue => todo!("ExpressionKind::Continue at {:?}", expr.span),
            ExpressionKind::IntPattern(_) => todo!("ExpressionKind::IntPattern at {:?}", expr.span),
            ExpressionKind::BoolLiteral(_) => todo!("ExpressionKind::BoolLiteral at {:?}", expr.span),
            ExpressionKind::StringLiteral(_) => todo!("ExpressionKind::StringLiteral at {:?}", expr.span),
            ExpressionKind::ArrayLiteral(_) => todo!("ExpressionKind::ArrayLiteral at {:?}", expr.span),
            ExpressionKind::TupleLiteral(_) => todo!("ExpressionKind::TupleLiteral at {:?}", expr.span),
            ExpressionKind::StructLiteral(_) => todo!("ExpressionKind::StructLiteral at {:?}", expr.span),
            ExpressionKind::Range { .. } => todo!("ExpressionKind::Range at {:?}", expr.span),
            ExpressionKind::UnaryOp(_, _) => todo!("ExpressionKind::UnaryOp at {:?}", expr.span),
            ExpressionKind::BinaryOp(_, _, _) => todo!("ExpressionKind::BinaryOp at {:?}", expr.span),
            ExpressionKind::TernarySelect(_, _, _) => todo!("ExpressionKind::TernarySelect at {:?}", expr.span),
            ExpressionKind::ArrayIndex(_, _) => todo!("ExpressionKind::ArrayIndex at {:?}", expr.span),
            ExpressionKind::DotIdIndex(_, _) => todo!("ExpressionKind::DotIdIndex at {:?}", expr.span),
            ExpressionKind::DotIntIndex(_, _) => todo!("ExpressionKind::DotIntIndex at {:?}", expr.span),
            ExpressionKind::Call(target, args) => {
                if let ExpressionKind::Path(Path { parents, id, span: _ }) = &target.inner {
                    if parents.is_empty() {
                        if let Some(name) = id.string.strip_prefix("__builtin_") {
                            return self.eval_builtin_call(expr, name, id, args);
                        }
                    }
                }

                todo!("ExpressionKind::Call at {:?}", expr.span)
            },
        }
    }

    fn eval_expression_as_ty(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Type> {
        let value = self.eval_expression(scope, expr)?;
        match &self.values[value] {
            &ValueInfo::Type(ty) => Ok(ty),
            _ => Err(FrontError::InvalidTypeExpression(expr.clone()).into()),
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

                return Err(FrontError::InvalidBuiltinArgs(expr.clone(), args.clone()).into());
            }
            _ => return Err(FrontError::InvalidBuiltinIdentifier(expr.clone(), id.clone()).into()),
        }
    }

    fn type_as_value(&mut self, ty: Type) -> Value {
        self.values.push(ValueInfo::Type(ty))
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
