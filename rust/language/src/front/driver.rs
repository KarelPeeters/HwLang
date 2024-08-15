use std::cmp::min;
use std::collections::HashMap;

use annotate_snippets::{Level, Renderer, Snippet};
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};
use logos::Source;
use num_bigint::BigInt;

use crate::{new_index_type, throw};
use crate::error::{CompileError, SnippetError};
use crate::front::{scope, TypeOrValue};
use crate::front::error::FrontError;
use crate::front::param::{GenericArgs, GenericParameter, GenericParams, GenericTypeParameter, GenericValueParameter};
use crate::front::scope::Visibility;
use crate::front::types::{EnumTypeInfo, EnumTypeInfoInner, Generic, IntegerTypeInfo, MaybeConstructor, StructTypeInfo, StructTypeInfoInner, Type};
use crate::front::values::{Value, ValueRangeInfo};
use crate::syntax::{ast, parse_file_content};
use crate::syntax::ast::{Args, Expression, ExpressionKind, GenericParam, GenericParamKind, Identifier, IntPattern, ItemDefEnum, ItemDefModule, ItemDefStruct, ItemDefType, Path, RangeLiteral, Spanned};
use crate::syntax::pos::{FileId, FileOffsets, Span};
use crate::util::arena::Arena;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FilePath(pub Vec<String>);

pub struct SourceDatabase {
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
    offsets: FileOffsets,
    local_scope: Option<Scope<'static>>,
}

// TODO rename to "FilePath" or "SourcePath"
new_index_type!(pub Directory);

pub struct DirectoryInfo {
    path: FilePath,
    file: Option<FileId>,
    children: IndexMap<String, Directory>,

    // only intended for use in user-visible diagnostic messages
    diagnostic_path: String,
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

#[derive(Debug)]
pub enum CompileSetError {
    EmptyPath,
    DuplicatePath(FilePath),
}

impl SourceDatabase {
    pub fn new() -> Self {
        let mut directories = Arena::default();
        let root_directory = directories.push(DirectoryInfo {
            path: FilePath(vec![]),
            file: None,
            children: Default::default(),
            diagnostic_path: "/".to_owned(),
        });

        SourceDatabase {
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
                    let curr_path = &path.0[..=i];
                    let child = self.directories.push(DirectoryInfo {
                        path: FilePath(curr_path.to_vec()),
                        file: None,
                        children: Default::default(),
                        diagnostic_path: curr_path.iter().join("/"),
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

pub struct CompileState<'d> {
    database: &'d SourceDatabase,
    items: Arena<Item, ItemInfo>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ResolveFirst(Item);

#[derive(Debug, Copy, Clone)]
pub enum ResolveFirstOr<E> {
    ResolveFirst(Item),
    Error(E),
}

impl SourceDatabase {
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
            let ast = parse_file_content(&file.source, &file.offsets)?;
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
            database: &self,
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
        let file_info = &self.database[file];
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
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, params, inner }) => {
                self.resolve_new_type_def_item(item_reference, scope, params, |s, params: _, args: _, scope_inner| {
                    Ok(s.eval_expression_as_ty(scope_inner, inner)?)
                })
            }
            ast::Item::Struct(ItemDefStruct { span: _, vis: _, id: _, params, fields }) => {
                self.resolve_new_type_def_item(item_reference, scope, params, |s, params, args, scope_inner| {
                    // map fields
                    let mut field_types = vec![];
                    for field in fields {
                        let field_ty = s.eval_expression_as_ty(scope_inner, &field.ty)?;
                        field_types.push((field.id.string.clone(), field_ty));
                    }

                    let ty = StructTypeInfo {
                        generic_struct: Generic {
                            parameters: params.clone(),
                            inner: StructTypeInfoInner {
                                item_reference,
                                fields: field_types,
                            },
                        },
                        args,
                    };
                    Ok(Type::Struct(ty))
                })
            }
            ast::Item::Enum(ItemDefEnum { span: _, vis: _, id: _, params, variants }) => {
                self.resolve_new_type_def_item(item_reference, scope, params, |s, params, args, scope_inner| {
                    // map variants
                    let mut variant_types = vec![];
                    for variant in variants {
                        let content = variant.content.as_ref()
                            .map(|content| s.eval_expression_as_ty(scope_inner, content))
                            .transpose()?;
                        variant_types.push((variant.id.string.clone(), content));
                    }

                    let ty = EnumTypeInfo {
                        generic_enum: Generic {
                            parameters: params.clone(),
                            inner: EnumTypeInfoInner {
                                item_reference,
                                variants: variant_types,
                            },
                        },
                        args,
                    };
                    Ok(Type::Enum(ty))
                })
            },
            // value definitions
            ast::Item::Module(ItemDefModule { span: _, vis: _, id: _, params, ports: _, body: _ }) => todo!(),
            ast::Item::Const(_) => todo!(),
            ast::Item::Function(_) => todo!(),
            ast::Item::Interface(_) => todo!(),
        }
    }

    fn resolve_new_type_def_item<T>(
        &mut self,
        defining_item: ItemReference,
        scope_outer: &Scope,
        params: &Option<Spanned<Vec<GenericParam>>>,
        build_ty: impl FnOnce(&mut Self, GenericParams, GenericArgs, &Scope) -> ResolveResult<T>,
    ) -> ResolveResult<MaybeConstructor<T>> {
        match params {
            None => {
                // there are no parameters, just map directly
                let parameters = GenericParams { vec: vec![] };
                let arguments = GenericArgs { vec: vec![] };
                Ok(MaybeConstructor::Immediate(build_ty(self, parameters, arguments, scope_outer)?))
            }
            Some(params) => {
                // build inner scope
                let mut unique: HashMap<&str, &Identifier> = Default::default();
                let mut parameters = vec![];
                let mut arguments = vec![];
                let mut scope_inner = scope_outer.nest(Visibility::Private);

                for param_ast in &params.inner {
                    // check parameter names for uniqueness
                    // TODO this is a weird place to do this, this can already happen during parsing
                    if let Some(prev) = unique.insert(&param_ast.id.string, &param_ast.id) {
                        throw!(FrontError::DuplicateParameterName(prev.clone(), param_ast.id.clone()))
                    }

                    let (param, arg) = match &param_ast.kind {
                        GenericParamKind::Type => {
                            let param = GenericTypeParameter { defining_item, id: param_ast.id.clone() };
                            (GenericParameter::Type(param.clone()), TypeOrValue::Type(Type::Generic(param)))
                        },
                        GenericParamKind::ValueOfType(ty) => {
                            let ty = self.eval_expression_as_ty(&scope_inner, ty)?;
                            let param = GenericValueParameter { defining_item, id: param_ast.id.clone(), ty };
                            (GenericParameter::Value(param.clone()), TypeOrValue::Value(Value::Generic(param)))
                        },
                    };

                    parameters.push(param);
                    arguments.push(arg.clone());
                    let entry = ScopedEntry::Direct(MaybeConstructor::Immediate(arg));
                    scope_inner.declare(&param_ast.id, entry, Visibility::Private)?;
                }

                // map inner to actual type
                let parameters = GenericParams { vec: parameters };
                let arguments = GenericArgs { vec: arguments };
                let ty_constr = Generic {
                    parameters: parameters.clone(),
                    inner: build_ty(self, parameters, arguments, &scope_inner)?,
                };
                Ok(MaybeConstructor::Constructor(ty_constr))
            }
        }
    }

    fn resolve_use_path(&self, path: &Path) -> ResolveResult<Item> {
        // TODO the current path design does not allow private sub-modules
        //   are they really necessary? if all inner items are private it's effectively equivalent

        let mut vis = Visibility::Private;
        let mut curr_dir = self.database.root_directory;

        let Path { span: _, steps, id } = path;

        for step in steps {
            let curr_dir_info = &self.database[curr_dir];
            curr_dir = curr_dir_info.children.get(&step.string).copied().ok_or_else(|| {
                let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                options.sort();
                FrontError::InvalidPathStep(step.clone(), options)
            })?;
        }

        let file = self.database[curr_dir].file.ok_or_else(|| {
            FrontError::ExpectedPathToFile(path.clone())
        })?;
        let file_scope = self.database[file].local_scope.as_ref().unwrap();

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

                let target_entry = self.eval_expression(scope, target)?;

                match target_entry {
                    ScopedEntryDirect::Constructor(constr) => {
                        todo!("{:?}", expr)
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => throw!(self.single_span_error(
                                "invalid call target",
                                target.span,
                                "invalid call target kind 'type'"
                            )),
                            TypeOrValue::Value(_) => {
                                todo!("call target value")
                            },
                        }
                    }
                }
                
                // TODO support both value function calls and type constructor calls

                todo!("call expression at {:?}", expr)

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

    fn get_item_ast(&self, item_reference: ItemReference) -> &'a ast::Item {
        let ItemReference { file, item_index } = item_reference;
        let file_info = &self.database[file];
        let ast = file_info.ast.as_ref().unwrap();
        &ast.items[item_index]
    }

    fn single_span_error(&self, title: impl Into<String>, span: Span, detail: impl Into<String>) -> SnippetError {
        let mut diag = Diagnostic::new(title);
        diag.error(span, detail);
        return diag.finish(&self.database);
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
            offsets: FileOffsets::new(id, &source),
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

// Diagnostic error formatting
const SNIPPET_CONTEXT_LINES: usize = 2;

pub struct Diagnostic {
    pub title: String,
    pub errors: Vec<(Span, String)>,
}

impl Diagnostic {
    pub fn new(title: impl Into<String>) -> Diagnostic {
        Diagnostic { title: title.into(), errors: vec![] }
    }

    // TODO allow reporting multiple errors in a single span
    pub fn error(&mut self, span: Span, message: impl Into<String>) {
        self.errors.push((span, message.into()))
    }

    pub fn finish(self, database: &SourceDatabase) -> SnippetError {
        let Self { title, errors } = self;

        let mut message = Level::Error.title(&title);

        for &(span, ref error) in &errors {
            let file_info = &database[span.start.file];
            let offsets = &file_info.offsets;

            let span = offsets.expand_span(span);
            let start_line_0 = span.start.line_0.saturating_sub(SNIPPET_CONTEXT_LINES);
            let end_line_0 = min(span.end.line_0 + SNIPPET_CONTEXT_LINES, offsets.line_count() - 1);
            let start_byte = offsets.line_start_byte(start_line_0);
            let end_byte = offsets.line_start_byte(end_line_0);
            let source = &file_info.source[start_byte..end_byte];

            let path = &database[file_info.directory].diagnostic_path;

            message = message.snippet(
                Snippet::source(source)
                    .origin(path)
                    .line_start(start_line_0 + 1)
                    .annotation(
                        Level::Error.span((span.start.byte - start_byte)..(span.end.byte - start_byte))
                            .label(&error)
                    )
            );
        }

        let renderer = Renderer::styled();
        let string = renderer.render(message).to_string();
        SnippetError { string }
    }
}

impl std::ops::Index<FileId> for SourceDatabase {
    type Output = FileInfo;
    fn index(&self, index: FileId) -> &Self::Output {
        self.files.get(&index).unwrap()
    }
}

impl std::ops::Index<Directory> for SourceDatabase {
    type Output = DirectoryInfo;
    fn index(&self, index: Directory) -> &Self::Output {
        &self.directories[index]
    }
}