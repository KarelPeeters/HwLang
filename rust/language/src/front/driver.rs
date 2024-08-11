use std::collections::HashMap;

use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};
use num_bigint::BigInt;

use crate::error::CompileError;
use crate::front::error::FrontError;
use crate::front::param::{GenericParameter, GenericTypeParameter, GenericValueParameter};
use crate::front::scope;
use crate::front::scope::Visibility;
use crate::front::types::{Constructor, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{Value, ValueRangeInfo};
use crate::new_index_type;
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Args, Expression, ExpressionKind, GenericParam, GenericParamKind, Identifier, IntPattern, ItemDefEnum, ItemDefModule, ItemDefStruct, ItemDefType, Path, RangeLiteral, Spanned};
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

    pub directories: Arena<Directory, DirectoryInfo>,
    pub root_directory: Directory,
}

// TODO distinguish between true and temporary options
pub struct FileInfo {
    id: FileId,
    path: FilePath,
    directory: Directory,
    source: String,
    ast: Option<ast::FileContent>,
    local_scope: Option<Scope<'static>>,
}

new_index_type!(Directory);

pub struct DirectoryInfo {
    path: FilePath,
    file: Option<FileId>,
    children: IndexMap<String, Directory>,
}

pub type Scope<'s> = scope::Scope<'s, ScopedEntry>;

// TODO pick a better name for this
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(Item),
    Direct(ScopedEntryDirect),
}

// TODO transpose or not?
pub type ScopedEntryDirect = MaybeConstructor<TypeOrValue>;
pub type ItemKind = TypeOrValue<(), ()>;

#[derive(Debug, Clone)]
pub enum TypeOrValue<T = Type, V = Value> {
    Type(T),
    Value(V),
}

#[derive(Debug)]
pub enum CompileSetError {
    EmptyPath,
    DuplicatePath(FilePath),
}

impl CompileSet {
    pub fn new() -> Self {
        let mut directories = Arena::default();
        let root_directory = directories.push(DirectoryInfo { path: FilePath(vec![]), file: None, children: Default::default() });

        CompileSet {
            files: IndexMap::default(),
            paths: IndexSet::default(),
            directories,
            root_directory,
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
        let directory = self.get_directory(&path);
        let info = FileInfo::new(id, path, directory, source);

        let slot = &mut self.directories[directory].file;
        assert_eq!(*slot, None);
        *slot = Some(id);

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

    fn get_directory(&mut self, path: &FilePath) -> Directory {
        let mut curr_dir = self.root_directory;
        for (i, path_item) in enumerate(&path.0) {
            curr_dir = match self.directories[curr_dir].children.get(path_item) {
                Some(&child) => child,
                None => {
                    let child = self.directories.push(DirectoryInfo {
                        path: FilePath(path.0[..=i].to_vec()),
                        file: None,
                        children: Default::default(),
                    });
                    self.directories[curr_dir].children.insert(path_item.clone(), child);
                    child
                }
            };
        }
        curr_dir
    }
}

new_index_type!(pub Item);

#[derive(Debug)]
pub struct ItemInfo {
    item_reference: ItemReference,

    // `None` if this item has not been resolved yet.
    // For a type: the type
    // For a value: the signature
    ty: Option<MaybeConstructor<Type>>,

    // TODO where to store the actual value? do we just leave this abstract?
}

pub struct CompileState<'a> {
    files: &'a IndexMap<FileId, FileInfo>,
    directories: &'a Arena<Directory, DirectoryInfo>,
    root_directory: Directory,
    items: Arena<Item, ItemInfo>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ResolveFirst(Item);

#[derive(Debug, Copy, Clone)]
pub enum ResolveFirstOr<E> {
    ResolveFirst(Item),
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

        // parse all files
        for file in self.files.values_mut() {
            let ast = parse_file_content(file.id, &file.source)?;
            file.ast = Some(ast);
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
                let item = items.push(ItemInfo { item_reference, ty: None });
                local_scope.maybe_declare(&id, ScopedEntry::Item(item), vis)?;
            }

            file.local_scope = Some(local_scope);
        }

        let mut state = CompileState {
            files: &self.files,
            directories: &self.directories,
            root_directory: self.root_directory,
            items,
        };

        // fully resolve all items
        let item_keys = state.items.keys().collect_vec();
        for item in item_keys {
            // TODO extra pass that actually looks at the bodies?
            //  or just call "typecheck_item_fully" instead?
            state.resolve_item_type_fully(item)?;
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
    fn resolve_item_type_fully(&mut self, item: Item) -> Result<(), CompileError> {
        let mut stack = vec![item];

        // TODO avoid repetitive work by switching to async instead?
        while let Some(curr) = stack.pop() {
            if self.resolve_item_type(curr).is_ok() {
                continue;
            }

            let result = self.resolve_item_type_new(curr);

            match result {
                Ok(resolved) => {
                    let slot = &mut self.items[item].ty;
                    assert!(slot.is_none(), "someone else already set the type");
                    *slot = Some(resolved.clone());
                }
                Err(ResolveFirstOr::Error(e)) => {
                    throw!(e);
                }
                Err(ResolveFirstOr::ResolveFirst(first)) => {
                    stack.push(curr);
                    let cycle_start_index = stack.iter().position(|s| s == &first);
                    stack.push(first);
                    if let Some(cycle_start_index) = cycle_start_index {
                        drop(stack.drain(..cycle_start_index));
                        throw!(FrontError::CyclicTypeDependency(stack));
                    }
                }
            }
        }

        Ok(())
    }

    fn resolve_item_type(&self, item: Item) -> Result<MaybeConstructor<Type>, ResolveFirst> {
        match &self.items[item].ty {
            Some(result) => Ok(result.clone()),
            None => Err(ResolveFirst(item))
        }
    }

    // TODO split this (and the corresponding functions) into signature and value for extra type safety
    fn resolve_item_type_new(&mut self, item: Item) -> ResolveResult<MaybeConstructor<Type>> {
        // check that this is indeed a new query
        assert!(self.items[item].ty.is_none());

        // item lookup
        let item_reference = self.items[item].item_reference;
        let ItemReference { file, item_index: _ } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        let item_ast = self.get_item_ast(item_reference);
        let scope = file_info.local_scope.as_ref().unwrap();

        // actual resolution
        match item_ast {
            // TODO why are we handling use items here? can they not be eliminated by scope building
            // use indirection
            ast::Item::Use(item_ast) => {
                let next_item = self.resolve_use_path(&item_ast.path)?;
                Ok(self.resolve_item_type(next_item)?)
            }
            // type definitions
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, params, inner: _ }) => {
                self.resolve_new_type_def_item(scope, params, |scope_inner| {
                    todo!()
                })
            }
            ast::Item::Struct(ItemDefStruct { span: _, vis: _, id: _, params, fields: _ }) => {
                self.resolve_new_type_def_item(scope, params, |scope_inner| {
                    todo!()
                })
            }
            ast::Item::Enum(ItemDefEnum { span: _, vis: _, id: _, params, variants: _ }) => {
                self.resolve_new_type_def_item(scope, params, |scope_inner| {
                    todo!()
                })
            },
            // value definitions
            ast::Item::Module(ItemDefModule { span: _, vis: _, id: _, params, ports: _, body: _ }) => todo!(),
            ast::Item::Const(_) => todo!(),
            ast::Item::Function(_) => todo!(),
            ast::Item::Interface(_) => todo!(),
        }
    }

    fn resolve_new_type_def_item(
        &mut self,
        scope_outer: &Scope,
        params: &Option<Spanned<Vec<GenericParam>>>,
        build_ty: impl FnOnce(&Scope) -> Type,
    ) -> ResolveResult<MaybeConstructor<Type>> {
        match params {
            None => {
                // there are no parameters, just map directly
                Ok(MaybeConstructor::Immediate(build_ty(scope_outer)))
            }
            Some(params) => {
                // build inner scope
                let mut unique: HashMap<&str, &Identifier> = Default::default();
                let mut parameters = vec![];
                let mut scope_inner = scope_outer.nest(Visibility::Private);

                for param_ast in &params.inner {
                    // check parameter names for uniqueness
                    // TODO this is a weird place to do this, this can already happen during parsing
                    if let Some(prev) = unique.insert(&param_ast.id.string, &param_ast.id) {
                        throw!(FrontError::DuplicateParameterName(prev.clone(), param_ast.id.clone()))
                    }

                    let (param, entry) = match &param_ast.kind {
                        GenericParamKind::Type => {
                            let param = GenericTypeParameter { id: param_ast.id.clone() };
                            (GenericParameter::Type(param.clone()), TypeOrValue::Type(Type::Generic(param)))
                        },
                        GenericParamKind::ValueOfType(ty) => {
                            let ty = self.eval_expression_as_ty(&scope_inner, ty)?;
                            let param = GenericValueParameter { id: param_ast.id.clone(), ty };
                            (GenericParameter::Value(param.clone()), TypeOrValue::Value(Value::Generic(param)))
                        },
                    };

                    parameters.push(param);
                    let entry = ScopedEntry::Direct(MaybeConstructor::Immediate(entry));
                    scope_inner.declare(&param_ast.id, entry, Visibility::Private)?;
                }

                // map inner to actual type
                let ty_constr = Constructor {
                    parameters,
                    inner: build_ty(&scope_inner),
                };
                Ok(MaybeConstructor::Constructor(ty_constr))
            }
        }
    }

    fn resolve_use_path(&self, path: &Path) -> ResolveResult<Item> {
        // TODO the current path design does not allow private sub-modules
        //   are they really necessary? if all inner items are private it's effectively equivalent

        let mut vis = Visibility::Private;
        let mut curr_dir = self.root_directory;

        let Path { span: _, steps, id } = path;

        for step in steps {
            curr_dir = self.directories[curr_dir].children.get(&step.string).copied().ok_or_else(|| {
                let mut options = self.directories[curr_dir].children.keys().cloned().collect_vec();
                options.sort();
                FrontError::InvalidPathStep(step.clone(), options)
            })?;
        }

        let file = self.directories[curr_dir].file.ok_or_else(|| {
            FrontError::ExpectedPathToFile(path.clone())
        })?;

        let file_info = self.files.get(&file).unwrap();
        let file_scope = file_info.local_scope.as_ref().unwrap();

        // TODO change root scope to just be a map instead of a scope so we can avoid this unwrap
        let value = file_scope.find(None, id, vis)?;
        match value {
            &ScopedEntry::Item(item) => Ok(item),
            // TODO is this still true?
            ScopedEntry::Direct(_) => unreachable!("file root entries should not exist"),
        }
    }

    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<ScopedEntryDirect> {
        let result = match &expr.inner {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Id(id) => {
                match scope.find(None, id, Visibility::Private)? {
                    &ScopedEntry::Item(item) => {
                        // TODO properly support value items, and in general fix "type" vs "value" resolution
                        //  maybe through checking the item kind first?
                        //  each of them clearly only defines a type or value, right?
                        //    or do we want to support "type A = if(cond) B else C"?
                        self.resolve_item_type(item)?
                            .map(TypeOrValue::Type)
                    }
                    ScopedEntry::Direct(entry) => entry.clone(),
                }
            },
            ExpressionKind::Wrapped(_) => todo!(),
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
                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Int(value)))
            },
            ExpressionKind::BoolLiteral(_) => todo!(),
            ExpressionKind::StringLiteral(_) => todo!(),
            ExpressionKind::ArrayLiteral(_) => todo!(),
            ExpressionKind::TupleLiteral(_) => todo!(),
            ExpressionKind::StructLiteral(_) => todo!(),
            ExpressionKind::RangeLiteral(range) => {
                let &RangeLiteral { end_inclusive, ref start, ref end } = range;

                let mut map_point = |point: &Option<Box<Expression>>| -> ResolveResult<_> {
                    match point {
                        None => Ok(None),
                        Some(point) => Ok(Some(Box::new(self.eval_expression_as_value(scope, point)?))),
                    }
                };

                let start = map_point(start)?;
                let end = map_point(end)?;

                let value = Value::Range(ValueRangeInfo::new(start, end, end_inclusive));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(value))
            },
            ExpressionKind::UnaryOp(_, _) => todo!(),
            ExpressionKind::BinaryOp(_, _, _) => todo!(),
            ExpressionKind::TernarySelect(_, _, _) => todo!(),
            ExpressionKind::ArrayIndex(_, _) => todo!(),
            ExpressionKind::DotIdIndex(_, _) => todo!(),
            ExpressionKind::DotIntIndex(_, _) => todo!(),
            ExpressionKind::Call(target, args) => {
                if let ExpressionKind::Id(id) = &target.inner {
                    if let Some(name) = id.string.strip_prefix("__builtin_") {
                        return Ok(MaybeConstructor::Immediate(self.eval_builtin_call(scope, expr, name, id, args)?));
                    }
                }

                let _ = self.eval_expression(scope, target)?;
                // TODO support both value function calls and type constructor calls

                todo!("call expression")

                // match &self.values[target_value] {
                //     ValueInfo::Function(target_value) => {
                //         let FunctionValue { item_reference, ty, params, body } = target_value;
                //         // TODO skip these clones
                //         let params = params.clone();
                //         let body = body.clone();
                //
                //         // define params
                //         // TODO this is a bit strange, shouldn't the function body have been fully checked at this point?
                //         let mut body_scope = scope.nest(Visibility::Private);
                //         let mut param_values = vec![];
                //         // TODO check param/arg length and type match, return proper errors
                //         for (param, arg) in zip_eq(&params, &args.inner) {
                //             // note: args are evaluated in the parent scope
                //             let arg = self.eval_expression(scope, arg)?;
                //             // TODO visibility does not really sense here, this scope is never accessible from outside
                //             body_scope.declare(param, ScopedEntry::Value(arg), Visibility::Private)?;
                //             param_values.push(arg);
                //         }
                //
                //         // actually run the function
                //         // TODO time/step/recursion limit?
                //         let result = self.call_function_body(&body_scope, &body, param_values)?;
                //         Ok(result)
                //     }
                //     _ => throw!(FrontError::ExpectedFunctionExpression((&**target).clone())),
                // }
            },
        };
        Ok(result)
    }

    fn eval_expression_as_ty(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Type> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => throw!(FrontError::ExpectedTypeExpressionGotConstructor(expr.clone())),
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(ty) => Ok(ty),
                TypeOrValue::Value(_) => throw!(FrontError::ExpectedTypeExpressionGotValue(expr.clone())),
            }
        }
    }

    fn eval_expression_as_value(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Value> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => throw!(FrontError::ExpectedValueExpressionGotConstructor(expr.clone())),
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(_) => throw!(FrontError::ExpectedValueExpressionGotType(expr.clone())),
                TypeOrValue::Value(value) => Ok(value),
            }
        }
    }

    fn eval_builtin_call(&mut self, scope: &Scope, expr: &Expression, name: &str, id: &Identifier, args: &Args) -> ResolveResult<TypeOrValue> {
        // TODO disallow calling builtin in user modules?
        let result = match name {
            "type" => {
                let first_arg = args.inner.get(0).map(|e| &e.inner);
                if let Some(ExpressionKind::StringLiteral(ty)) = first_arg {
                    match ty.as_str() {
                        "bool" if args.inner.len() == 1 =>
                            TypeOrValue::Type(Type::Boolean),
                        "int" if args.inner.len() == 1 => {
                            let range = Box::new(Value::Range(ValueRangeInfo::unbounded()));
                            TypeOrValue::Type(Type::Integer(IntegerTypeInfo { range }))
                        }
                        "int_range" if args.inner.len() == 2 => {
                            // TODO typecheck (range must be integer)
                            let range = Box::new(self.eval_expression_as_value(scope, &args.inner[1])?);
                            let ty_info = IntegerTypeInfo { range };
                            TypeOrValue::Type(Type::Integer(ty_info))
                        },
                        "Range" if args.inner.len() == 1 =>
                            TypeOrValue::Type(Type::Range),
                        "bits" if args.inner.len() == 2 => {
                            // TODO typecheck (bits must be non-negative integer)
                            let bits = self.eval_expression_as_value(scope, &args.inner[1])?;
                            TypeOrValue::Type(Type::Bits(Box::new(bits)))
                        }
                        _ => throw!(FrontError::InvalidBuiltinArgs(expr.clone(), args.clone())),
                    }
                } else {
                    throw!(FrontError::InvalidBuiltinArgs(expr.clone(), args.clone()))
                }
            }
            _ => throw!(FrontError::InvalidBuiltinIdentifier(expr.clone(), id.clone())),
        };

        Ok(result)
    }

    // TODO move this content into actual type def stuff
    // fn call_function_body(&mut self, scope_with_params: &Scope, body: &FunctionBody, params_for_unique: Vec<Value>) -> ResolveResult<Value> {
    //     match body {
    //         &FunctionBody::TypeConstructor(item_reference) => {
    //             match self.get_item_ast(item_reference) {
    //                 ast::Item::Type(item_ast) => {
    //                     let ItemDefType { span: _, vis: _, id: _, params: _, inner } = item_ast;
    //                     let ty = self.eval_expression_as_ty(scope_with_params, inner)?;
    //                     Ok(self.type_as_value(ty))
    //                 }
    //                 ast::Item::Struct(item_ast) => {
    //                     let ItemDefStruct { span: _, vis: _, id: _, params: _, fields } = item_ast;
    //                     let unique = TypeUnique { item_reference, params: params_for_unique };
    //
    //                     // map fields
    //                     let mut field_types = vec![];
    //                     for field in fields {
    //                         let field_ty = self.eval_expression_as_ty(scope_with_params, &field.ty)?;
    //                         field_types.push((field.id.string.clone(), field_ty));
    //                     }
    //
    //                     // define new type
    //                     let info = StructTypeInfo { unique, fields: field_types };
    //                     let ty = self.types.push(Type::Struct(info));
    //                     Ok(self.type_as_value(ty))
    //                 },
    //                 ast::Item::Enum(item_ast) => {
    //                     let ItemDefEnum { span: _, vis: _, id: _, params: _, variants } = item_ast;
    //                     let unique = TypeUnique { item_reference, params: params_for_unique };
    //
    //                     // map variants
    //                     let mut variant_types = vec![];
    //                     for variant in variants {
    //                         let content = variant.content.as_ref()
    //                             .map(|content| self.eval_expression_as_ty(scope_with_params, content))
    //                             .transpose()?;
    //                         variant_types.push((variant.id.string.clone(), content));
    //                     }
    //
    //                     // define new type
    //                     let info = EnumTypeInfo { unique, variants: variant_types };
    //                     let ty = self.types.push(Type::Enum(info));
    //                     Ok(self.type_as_value(ty))
    //                 },
    //                 ast::Item::Module(item_ast) => {
    //                     let ItemDefModule { span: _, vis: _, id: _, params: _, ports, body } = item_ast;
    //                     let unique = TypeUnique { item_reference, params: params_for_unique };
    //
    //                     // map ports
    //                     let mut port_types = vec![];
    //
    //                     for port in &ports.inner {
    //                         let ModulePort { span: _, id: _, direction, kind } = port;
    //
    //                         let kind_ty = match &kind.inner {
    //                             PortKind::Clock => PortKind::Clock,
    //                             PortKind::Normal { sync, ty } => {
    //                                 let sync_index = match &sync.inner {
    //                                     SyncKind::Async => SyncKind::Async,
    //                                     SyncKind::Sync(id) => {
    //                                         // TODO expand to full expressions and real scope lookups
    //                                         let index = port_types.iter().position(|(port_id, _)| port_id == &id.string).ok_or_else(|| {
    //                                             FrontError::UnknownClock(id.clone())
    //                                         })?;
    //                                         SyncKind::Sync(index)
    //                                     }
    //                                 };
    //
    //                                 let ty = self.eval_expression_as_ty(scope_with_params, ty)?;
    //                                 PortKind::Normal { sync: sync_index, ty }
    //                             }
    //                         };
    //
    //                         port_types.push((port.id.string.clone(), PortTypeInfo { direction: port.direction.inner, kind: kind_ty }));
    //                     }
    //
    //                     // define new type
    //                     let info = ModuleTypeInfo { unique, ports: port_types };
    //                     let ty = self.types.push(Type::Module(info));
    //                     Ok(self.type_as_value(ty))
    //                 }
    //                 _ => unreachable!(),
    //             }
    //         },
    //     }
    // }

    fn get_item_ast(&self, item_reference: ItemReference) -> &'a ast::Item {
        let ItemReference { file, item_index } = item_reference;
        let file_info = self.files.get(&file).unwrap();
        &file_info.ast.as_ref().unwrap().items[item_index]
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
    pub fn new(id: FileId, path: FilePath, directory: Directory, source: String) -> Self {
        FileInfo {
            id,
            path,
            directory,
            source,
            ast: None,
            local_scope: None,
        }
    }
}

/// Utility type to refer to a specific item in a specific file.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ItemReference {
    pub file: FileId,
    pub item_index: usize,
}
