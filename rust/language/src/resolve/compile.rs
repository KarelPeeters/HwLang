use std::collections::HashMap;
use std::ops::Add;

use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools, zip_eq};
use num_bigint::BigInt;
use num_traits::One;

use crate::error::CompileError;
use crate::new_index_type;
use crate::resolve::scope;
use crate::resolve::scope::Visibility;
use crate::resolve::types::{BasicTypes, Type, TypeInfo, TypeInfoEnum, TypeInfoFunction, TypeInfoInteger, TypeInfoStruct, Types, TypeUnique};
use crate::resolve::values::{Value, ValueFunctionInfo, ValueInfo, Values};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Args, Expression, ExpressionKind, Identifier, IntPattern, ItemDefEnum, ItemDefStruct, ItemDefType, Path, RangeLiteral, Spanned, TypeParam};
use crate::syntax::pos::FileId;
use crate::util::arena::Arena;

macro_rules! throw {
    ($e:expr) => { return Err($e.into()) };
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FilePath(pub Vec<String>);

pub struct CompileSet {
    pub paths: IndexSet<FilePath>,
    // TODO use arena for this too?
    pub files: IndexMap<FileId, FileInfo>,
}

// TODO distinguish between true and temporary options 
pub struct FileInfo {
    id: FileId,
    path: FilePath,
    source: String,
    ast: Option<ast::FileContent>,
    local_scope: Option<Scope<'static>>,
    module: Option<Module>,
}

new_index_type!(Module);

pub struct ModuleInfo {
    path: FilePath,
    file: Option<FileId>,
    children: IndexMap<String, Module>,
}

pub type Scope<'s> = scope::Scope<'s, ScopedValue>;

#[derive(Debug, Copy, Clone)]
pub enum ScopedValue {
    // TODO should items and files also just be values, freely storable by the user?
    Value(Value),
    Item(Item),
    // TODO remove file as a scoped value? it can only ever come from use 
    //  statements which look things up in a separate namespace anyway
    File(FileId),
}

#[derive(Debug, Copy, Clone)]
pub enum ValueOrItem {
    Value(Value),
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
    ExpectedIntegerExpression(Expression),
    ExpectedRangeExpression(Expression),

    ExpectedPathToFileNotValueOrItem(Path, ValueOrItem),
    ExpectedPathToItemNotValue(Path, Value),
    ExpectedPathToItemOrValueNotFile(Path, FileId),

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
    // TODO: add some error recovery and continuation, eg. return all parse errors at once
    pub fn compile(mut self) -> Result<(), CompileError> {
        // sort files to ensure platform-independence
        // TODO should this be here or at the higher-level path walker?
        self.files.sort_by(|_, v1, _, v2| {
            v1.path.cmp(&v2.path)
        });

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
        
        // connect files and modules
        // TODO this feels overcomplicated
        let mut modules: Arena<Module, ModuleInfo> = Arena::default();
        let root_module = modules.push(ModuleInfo { path: FilePath(vec![]), file: None, children: Default::default() });

        let mut get_module = |modules: &mut Arena<Module, ModuleInfo>, path: &FilePath| -> Module {
            let mut parent = root_module;
            for (i, path_item) in enumerate(&path.0) {
                let child = (&modules[parent]).children.get(path_item);
                match child {
                    Some(&child) => parent = child,
                    None => {
                        let new_module = modules.push(ModuleInfo {
                            path: FilePath(path.0[..=i].to_vec()),
                            file: None,
                            children: Default::default(),
                        });
                        modules[parent].children.insert(path_item.clone(), new_module);
                        parent = new_module;
                    }
                }
            }
            parent
        };
        for file in self.files.values_mut() {
            let module = get_module(&mut modules, &file.path);
            modules[module].file = Some(file.id);
            file.module = Some(module);
        }

        // build import scope of each file
        for file in self.files.values_mut() {
            let ast = file.ast.as_ref().unwrap();

            // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
            let mut local_scope = Scope::default();

            for (ast_item_index, ast_item) in enumerate(&ast.items) {
                let (id, ast_vis) = ast_item.id_vis();
                let vis = match ast_vis {
                    ast::Visibility::Public(_) => Visibility::Public,
                    ast::Visibility::Private => Visibility::Private,
                };

                let item_reference = ItemReference { file: file.id, item_index: ast_item_index };
                let item = items.push(ItemInfo { item_reference, signature: None, content: None });
                local_scope.maybe_declare(&id, ScopedValue::Item(item), vis)?;
            }

            file.local_scope = Some(local_scope);
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

impl<'a> CompileState<'a> {
    fn resolve_fully(&mut self, query: ResolveQuery) -> Result<(), CompileError> {
        let mut stack = vec![query];

        while let Some(curr) = stack.pop() {
            if self.resolve(curr).is_ok() {
                continue;
            }

            let result = self.resolve_new(curr);

            match result {
                Ok(resolved) => {
                    *curr.kind.slot_mut(&mut self.items[curr.item]) = Some(resolved);
                }
                Err(ResolveFirstOr::Error(e)) => {
                    return Err(e.into());
                }
                Err(ResolveFirstOr::ResolveFirst(first)) => {
                    stack.push(curr);
                    let cycle_start_index = stack.iter().position(|s| s == &first);
                    stack.push(first);
                    if let Some(cycle_start_index) = cycle_start_index {
                        drop(stack.drain(..cycle_start_index));
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

    // TODO split this (and the corresponding functions) into signature and value for extra type safety
    fn resolve_new(&mut self, query: ResolveQuery) -> ResolveResult<Value> {
        // check that this is indeed a new query
        assert!(query.kind.slot(&self.items[query.item]).is_none());

        // item lookup
        let ResolveQuery { item, kind } = query;
        let item_reference = self.items[item].item_reference;
        let ItemReference { file, item_index: _ } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        let item_ast = self.get_item_ast(item_reference);
        let scope = file_info.local_scope.as_ref().unwrap();

        // actual resolution
        let resolved= match item_ast {
            ast::Item::Use(item_ast) => {
                let next_item = self.follow_path(scope, &item_ast.path)?;

                let next_item = match next_item {
                    ValueOrItem::Value(value) => throw!(FrontError::ExpectedPathToItemNotValue(item_ast.path.clone(), value)),
                    ValueOrItem::Item(next_item) => next_item,
                };

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
                    ResolveQueryKind::Value => self.call_function_body(scope, &build_value, vec![])?,
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

    // TODO should single identifiers be separate from paths? paths can never resolve to a value,
    //  only to items and modules (which may correspond to files)
    // TODO resolution order: child items, sibling items, lib_exit::std::..., lib_curr::root_file::...
    //  or do we want separate child/sibling keywords too?
    fn follow_path(&mut self, scope: &Scope, path: &Path) -> ResolveResult<ValueOrItem> {
        let mut scope = scope;
        let mut vis = Visibility::Private;

        for parent in &path.parents {
            match *scope.find(None, &parent, vis)? {
                ScopedValue::Item(item) =>
                    throw!(FrontError::ExpectedPathToFileNotValueOrItem(path.clone(), ValueOrItem::Item(item))),
                ScopedValue::Value(value) =>
                    throw!(FrontError::ExpectedPathToFileNotValueOrItem(path.clone(), ValueOrItem::Value(value))),
                ScopedValue::File(file) => {
                    scope = self.files.get(&file).unwrap().local_scope.as_ref().unwrap();
                    vis = Visibility::Public;
                }
            }
        }

        match *scope.find(None, &path.id, vis)? {
            ScopedValue::Value(value) =>
                Ok(ValueOrItem::Value(value)),
            ScopedValue::Item(item) =>
                Ok(ValueOrItem::Item(item)),
            ScopedValue::File(file) =>
                Err(FrontError::ExpectedPathToItemOrValueNotFile(path.clone(), file).into()),
        }
    }

    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Value> {
        match &expr.inner {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Path(path) => {
                match self.follow_path(scope, path)? {
                    ValueOrItem::Value(value) =>
                        Ok(value),
                    ValueOrItem::Item(item) =>
                        Ok(self.resolve(ResolveQuery { item, kind: ResolveQueryKind::Value })?),
                }
            },
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
            ExpressionKind::IntPattern(int) => {
                let value = match int {
                    IntPattern::Hex(_) => todo!("hex with wildcards"),
                    IntPattern::Bin(_) => todo!("bin with wildcards"),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        str_clean.parse::<BigInt>().unwrap()
                    }
                };

                Ok(self.values.push(ValueInfo::Int(value)))
            },
            ExpressionKind::BoolLiteral(_) => todo!(),
            ExpressionKind::StringLiteral(_) => todo!(),
            ExpressionKind::ArrayLiteral(_) => todo!(),
            ExpressionKind::TupleLiteral(_) => todo!(),
            ExpressionKind::StructLiteral(_) => todo!(),
            ExpressionKind::RangeLiteral(range) => {
                let &RangeLiteral { end_inclusive, ref start, ref end } = range;

                let mut map_point = |point: &Option<Box<Expression>>| -> ResolveResult<Option<BigInt>> {
                    match point {
                        None => Ok(None),
                        Some(point) => {
                            let point_value = self.eval_expression(scope, point)?;
                            match &self.values[point_value] {
                                ValueInfo::Int(value) => Ok(Some(value.clone())),
                                _ => throw!(FrontError::ExpectedIntegerExpression((&**point).clone())),
                            }
                        }
                    }
                };

                let start = map_point(start)?;
                let end_partial = map_point(end)?;

                let end=  if end_inclusive {
                    Some(end_partial.unwrap().add(&BigInt::one()))
                } else {
                    end_partial
                };

                Ok(self.values.push(ValueInfo::Range { start, end }))
            },
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
                            return self.eval_builtin_call(scope, expr, name, id, args);
                        }
                    }
                }

                let target_value = self.eval_expression(scope, target)?;
                match &self.values[target_value] {
                    ValueInfo::Function(target_value) => {
                        let ValueFunctionInfo { item_reference, ty, params, body } = target_value;
                        // TODO skip these clones
                        let params = params.clone();
                        let body = body.clone();

                        // define params
                        // TODO this is a bit strange, shouldn't the function body have been fully checked at this point?
                        let mut body_scope = scope.nest(Visibility::Private);
                        let mut param_values = vec![];
                        // TODO check param/arg length and type match, return proper errors
                        for (param, arg) in zip_eq(&params, &args.inner) {
                            // note: args are evaluated in the parent scope
                            let arg = self.eval_expression(scope, arg)?;
                            // TODO visibility does not really sense here, this scope is never accessible from outside
                            body_scope.declare(param, ScopedValue::Value(arg), Visibility::Private)?;
                            param_values.push(arg);
                        }

                        // actually run the function
                        // TODO time/step/recursion limit?
                        let result = self.call_function_body(&body_scope, &body, param_values)?;
                        Ok(result)
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

    fn eval_builtin_call(&mut self, scope: &Scope, expr: &Expression, name: &str, id: &Identifier, args: &Args) -> ResolveResult<Value> {
        // TODO disallow calling builtin in user modules?
        match name {
            "type" => {
                if args.inner.len() >= 1 {
                    if let ExpressionKind::StringLiteral(ty) = &args.inner[0].inner {
                        match ty.as_str() {
                            "bool" if args.inner.len() == 1 => return Ok(self.basic_values.ty_bool),
                            "int" if args.inner.len() == 1 => return Ok(self.basic_values.ty_int),
                            "int_range" if args.inner.len() == 2 => {
                                let range = self.eval_expression(scope, &args.inner[1])?;
                                let (start, end) = match &self.values[range] {
                                    ValueInfo::Range { start, end } => (start, end),
                                    _ => throw!(FrontError::ExpectedRangeExpression((&args.inner[1]).clone())),
                                };
                                let ty_info = TypeInfoInteger { min: start.clone(), max: end.clone() };
                                let ty = self.types.push(TypeInfo::Integer(ty_info));
                                return Ok(self.type_as_value(ty));
                            },
                            "Range" if args.inner.len() == 1 => return Ok(self.basic_values.ty_range),
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

    fn call_function_body(&mut self, scope_with_params: &Scope, body: &FunctionBody, params_for_unique: Vec<Value>) -> ResolveResult<Value> {
        match body {
            &FunctionBody::TypeConstructor(item_reference) => {
                match self.get_item_ast(item_reference) {
                    ast::Item::Type(item_ast) => {
                        let ty = self.eval_expression_as_ty(scope_with_params, &item_ast.inner)?;
                        Ok(self.type_as_value(ty))
                    }
                    ast::Item::Struct(item_ast) => {
                        let unique = TypeUnique { item_reference, params: params_for_unique };

                        // map fields
                        let mut fields = vec![];
                        for field in &item_ast.fields {
                            let field_ty = self.eval_expression_as_ty(scope_with_params, &field.ty)?;
                            fields.push((field.id.string.clone(), field_ty));
                        }

                        // define new type
                        let info = TypeInfoStruct { unique, fields };
                        let ty = self.types.push(TypeInfo::Struct(info));
                        Ok(self.type_as_value(ty))
                    },
                    ast::Item::Enum(item_ast) => {
                        let unique = TypeUnique { item_reference, params: params_for_unique };

                        // map variants
                        let mut variants = vec![];
                        for variant in &item_ast.variants {
                            let content = variant.content.as_ref()
                                .map(|content| self.eval_expression_as_ty(scope_with_params, content))
                                .transpose()?;
                            variants.push((variant.id.string.clone(), content));
                        }

                        // define new type
                        let info = TypeInfoEnum { unique, variants };
                        let ty = self.types.push(TypeInfo::Enum(info));
                        Ok(self.type_as_value(ty))
                    },
                    _ => unreachable!(),
                }
            },
        }
    }

    fn get_item_ast(&self, item_reference: ItemReference) -> &'a ast::Item {
        let ItemReference { file, item_index } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        &file_info.ast.as_ref().unwrap().items[item_index]
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
            module: None,
        }
    }
}

/// Utility type to refer to a specific item in a specific file.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ItemReference {
    pub file: FileId,
    pub item_index: usize,
}
