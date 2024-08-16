use annotate_snippets::{Level, Renderer, Snippet};
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, zip_eq, Itertools};
use num_bigint::BigInt;
use std::cmp::min;
use std::collections::HashMap;

use crate::error::{CompileError, DiagnosticError};
use crate::front::param::{GenericArgs, GenericContainer, GenericParameter, GenericParameterUniqueId, GenericParams, GenericTypeParameter, GenericValueParameter};
use crate::front::scope::Visibility;
use crate::front::types::{Constructor, EnumTypeInfo, IntegerTypeInfo, MaybeConstructor, ModuleTypeInfo, NominalTypeUnique, PortTypeInfo, StructTypeInfo, Type};
use crate::front::values::{Value, ValueRangeInfo};
use crate::front::{scope, TypeOrValue};
use crate::syntax::ast::{Args, BinaryOp, EnumVariant, Expression, ExpressionKind, GenericParam, GenericParamKind, Identifier, IntPattern, ItemDefEnum, ItemDefModule, ItemDefStruct, ItemDefType, ItemUse, ModulePort, Path, PortKind, RangeLiteral, Spanned, StructField, SyncKind, UnaryOp};
use crate::syntax::pos::{FileId, FileOffsets, Pos, PosFull, Span, SpanFull};
use crate::syntax::{ast, parse_file_content, ParseError};
use crate::util::arena::Arena;
use crate::{new_index_type, throw};

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
    #[allow(dead_code)]
    id: FileId,
    path: FilePath,
    /// only intended for use in user-visible diagnostic messages
    path_raw: String,
    #[allow(dead_code)]
    directory: Directory,
    source: String,
    ast: Option<ast::FileContent>,
    offsets: FileOffsets,
    local_scope: Option<Scope<'static>>,
}

// TODO rename to "FilePath" or "SourcePath"
new_index_type!(pub Directory);

pub struct DirectoryInfo {
    #[allow(dead_code)]
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

#[must_use]
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
        });

        SourceDatabase {
            files: IndexMap::default(),
            paths: IndexSet::default(),
            directories,
            root_directory,
        }
    }

    pub fn add_file(&mut self, path: FilePath, path_raw: String, source: String) -> Result<(), CompileSetError> {
        if path.0.is_empty() {
            throw!(CompileSetError::EmptyPath);
        }
        if !self.paths.insert(path.clone()) {
            throw!(CompileSetError::DuplicatePath(path.clone()));
        }

        let id = FileId(self.files.len());
        println!("adding {:?} => {:?}", id, path);
        let directory = self.get_directory(&path);
        let info = FileInfo::new(id, path, path_raw, directory, source);

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
                    });
                    self.directories[curr_dir].children.insert(path_item.clone(), child);
                    child
                }
            };
        }
        curr_dir
    }

    pub fn expand_pos(&self, pos: Pos) -> PosFull {
        self[pos.file].offsets.expand_pos(pos)
    }

    pub fn expand_span(&self, span: Span) -> SpanFull {
        self[span.start.file].offsets.expand_span(span)
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

        // TODO split up database into even more immutable and mutable part
        let files = self.files.keys().copied().collect_vec();

        // parse all files
        for &file in &files {
            let file_info = &self[file];
            let ast = parse_file_content(&file_info.source, &file_info.offsets)
                .map_err(|e| self.map_parser_error(e))?;
            self.files.get_mut(&file).unwrap().ast = Some(ast);
        }

        // build import scope of each file
        for &file in &files {
            let file_info = &self[file];
            let ast = file_info.ast.as_ref().unwrap();

            // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
            let mut local_scope = Scope::new_root(file_info.offsets.full_span());

            for (ast_item_index, ast_item) in enumerate(&ast.items) {
                let common_info = ast_item.common_info();
                let vis = match common_info.vis {
                    ast::Visibility::Public(_) => Visibility::Public,
                    ast::Visibility::Private => Visibility::Private,
                };

                let item_reference = ItemReference { file, item_index: ast_item_index };
                let item = items.push(ItemInfo { item_reference, ty: None });
                local_scope.maybe_declare(&self, &common_info.id, ScopedEntry::Item(item), vis)?;
            }

            self.files.get_mut(&file).unwrap().local_scope = Some(local_scope);
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

    fn map_parser_error(&self, e: ParseError) -> DiagnosticError {
        match e {
            ParseError::InvalidToken { location } => {
                let span = Span::empty_at(location);
                self.diagnostic("invalid token")
                    .add_error(span, "invalid token")
                    .finish()
            }
            ParseError::UnrecognizedEof { location, expected } => {
                let span = Span::empty_at(location);
                let expected = expected.iter().map(|s| &s[1..s.len() - 1]).collect_vec();

                self.diagnostic("unexpected eof")
                    .add_error(span, "invalid token")
                    .footer(Level::Info, format!("expected one of {:?}", expected))
                    .finish()
            },
            ParseError::UnrecognizedToken { token, expected } => {
                let (start, _, end) = token;
                let span = Span::new(start, end);
                let expected = expected.iter().map(|s| &s[1..s.len() - 1]).collect_vec();

                self.diagnostic("unexpected token")
                    .add_error(span, "unexpected token")
                    .footer(Level::Info, format!("expected one of {:?}", expected))
                    .finish()
            },
            ParseError::ExtraToken { token } => {
                let (start, _, end) = token;
                let span = Span::new(start, end);

                self.diagnostic("unexpected extra token")
                    .add_error(span, "extra token")
                    .finish()
            },
            ParseError::User { .. } => unreachable!("no user errors are generated by the grammer")
        }
    }

    pub fn diagnostic(&self, title: impl Into<String>) -> Diagnostic<'_> {
        Diagnostic::new(self, title)
    }
}

pub type ResolveResult<T> = Result<T, ResolveFirstOr<CompileError>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum FunctionBody {
    /// type alias, enum, or struct
    TypeConstructor(ItemReference),
}

impl<'d> CompileState<'d> {
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

                        let mut diag = self.diagnostic("cyclic type dependency");
                        for item in stack {
                            let item_ast = self.get_item_ast(self.items[item].item_reference);
                            diag = diag.add_error(item_ast.common_info().span_short, "part of cycle");
                        }
                        throw!(diag.finish())
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
    // TODO should this do both signatures and values, or only the latter?
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
        match *item_ast {
            // use indirection
            ast::Item::Use(ItemUse { span: _, ref path, as_: _ }) => {
                // TODO why are we handling use items here? can they not be eliminated by scope building
                //  this is really weird, use items don't even really have signatures
                let next_item = self.resolve_use_path(path)?;
                Ok(self.resolve_item_type(next_item)?)
            }
            // type definitions
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, ref params, ref inner }) => {
                self.resolve_new_generic_type_def(item_reference, scope, params, |s, _args, scope_inner| {
                    Ok(s.eval_expression_as_ty(scope_inner, inner)?)
                })
            }
            ast::Item::Struct(ItemDefStruct { span, vis: _, id: _, ref params, ref fields }) => {
                self.resolve_new_generic_type_def(item_reference, scope, params, |s, args, scope_inner| {
                    // map fields
                    let mut fields_map = IndexMap::new();
                    for field in fields {
                        let StructField { span: _, id: field_id, ty } = field;
                        let field_ty = s.eval_expression_as_ty(scope_inner, ty)?;

                        let prev = fields_map.insert(field_id.string.clone(), (field_id, field_ty));
                        if let Some(prev) = prev {
                            throw!(s.diagnostic_defined_twice("struct field", span, field_id, prev.0))
                        }
                    }

                    // result
                    let ty = StructTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item_reference, args },
                        fields: fields_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Type::Struct(ty))
                })
            }
            ast::Item::Enum(ItemDefEnum { span, vis: _, id: _, ref params, ref variants }) => {
                self.resolve_new_generic_type_def(item_reference, scope, params, |s, args, scope_inner| {
                    // map variants
                    let mut variants_map = IndexMap::new();
                    for variant in variants {
                        let EnumVariant { span: _, id: variant_id, content } = variant;

                        let content = content.as_ref()
                            .map(|content| s.eval_expression_as_ty(scope_inner, content))
                            .transpose()?;

                        let prev = variants_map.insert(variant_id.string.clone(), (variant_id, content));
                        if let Some(prev) = prev {
                            throw!(s.diagnostic_defined_twice("enum variant", span, variant_id, prev.0))
                        }
                    }

                    // result
                    let ty = EnumTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item_reference, args },
                        variants: variants_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Type::Enum(ty))
                })
            },
            // value definitions
            ast::Item::Module(ItemDefModule { span: _, vis: _, id: _, ref params, ref ports, body: _ }) => {
                self.resolve_new_generic_type_def(item_reference, scope, params, |s, args, scope_inner| {
                    // yet another sub-scope for the ports that refer to each other
                    // TODO get a more accurate span
                    let mut scope_ports = scope_inner.nest(ports.span, Visibility::Private);

                    // map ports
                    // TODO extract duplicate code between all these id uniqueness checking places
                    let mut ports_map = IndexMap::new();
                    for port in &ports.inner {
                        let ModulePort { span: _, id: port_id, direction, kind, } = port;

                        let info = PortTypeInfo {
                            direction: direction.inner,
                            kind: match &kind.inner {
                                PortKind::Clock => PortKind::Clock,
                                PortKind::Normal { sync, ty } => {
                                    PortKind::Normal {
                                        sync: match &sync.inner {
                                            SyncKind::Async => SyncKind::Async,
                                            SyncKind::Sync(clk) => {
                                                let clk = s.eval_expression_as_value(&scope_ports, clk)?;
                                                SyncKind::Sync(clk)
                                            },
                                        },
                                        ty: s.eval_expression_as_ty(scope_inner, ty)?,
                                    }
                                }
                            },
                        };

                        let prev = ports_map.insert(port_id.string.clone(), (port_id, info));
                        if let Some(prev) = prev {
                            throw!(s.diagnostic_defined_twice("module port", ports.span, &port_id, prev.0))
                        }

                        scope_ports.declare(
                            &s.database,
                            &port_id,
                            ScopedEntry::Direct(MaybeConstructor::Immediate(TypeOrValue::Value(Value::Port(port_id.clone())))),
                            Visibility::Private,
                        )?;
                    }

                    // result
                    let ty = ModuleTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item_reference, args },
                        ports: ports_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Type::Module(ty))
                })
            },
            ast::Item::Const(_) => self.diagnostic_todo(item_ast.common_info().span_short, "const definition"),
            ast::Item::Function(_) => self.diagnostic_todo(item_ast.common_info().span_short, "function definition"),
            ast::Item::Interface(_) => self.diagnostic_todo(item_ast.common_info().span_short, "interface definition"),
        }
    }

    fn resolve_new_generic_type_def<T>(
        &mut self,
        defining_item: ItemReference,
        scope_outer: &Scope,
        params: &Option<Spanned<Vec<GenericParam>>>,
        build_ty: impl FnOnce(&mut Self, GenericArgs, &Scope) -> ResolveResult<T>,
    ) -> ResolveResult<MaybeConstructor<T>> {
        match params {
            None => {
                // there are no parameters, just map directly
                let arguments = GenericArgs { vec: vec![] };
                Ok(MaybeConstructor::Immediate(build_ty(self, arguments, scope_outer)?))
            }
            Some(params) => {
                // build inner scope
                let mut unique: HashMap<&str, &Identifier> = Default::default();
                let mut parameters = vec![];
                let mut arguments = vec![];

                let item_span = self.get_item_ast(defining_item).common_info().span_full;
                let mut scope_inner = scope_outer.nest(item_span, Visibility::Private);

                for (param_index, param_ast) in enumerate(&params.inner) {
                    // check parameter names for uniqueness
                    if let Some(prev) = unique.insert(&param_ast.id.string, &param_ast.id) {
                        throw!(self.diagnostic_defined_twice("generic parameter", params.span, prev, &param_ast.id))
                    }

                    let unique_id = GenericParameterUniqueId { defining_item, param_index };

                    let (param, arg) = match &param_ast.kind {
                        GenericParamKind::Type => {
                            let param = GenericTypeParameter { unique_id, id: param_ast.id.clone() };
                            (GenericParameter::Type(param.clone()), TypeOrValue::Type(Type::Generic(param)))
                        },
                        GenericParamKind::ValueOfType(ty) => {
                            let ty = self.eval_expression_as_ty(&scope_inner, ty)?;
                            let param = GenericValueParameter { unique_id, id: param_ast.id.clone(), ty };
                            (GenericParameter::Value(param.clone()), TypeOrValue::Value(Value::Generic(param)))
                        },
                    };

                    parameters.push(param);
                    arguments.push(arg.clone());

                    // TODO should we nest scopes here, or is incremental declaration in a single scope equivalent?
                    let entry = ScopedEntry::Direct(MaybeConstructor::Immediate(arg));
                    scope_inner.declare(&self.database, &param_ast.id, entry, Visibility::Private)?;
                }

                // map inner to actual type
                let parameters = GenericParams { vec: parameters };
                let arguments = GenericArgs { vec: arguments };
                let ty_constr = Constructor {
                    parameters,
                    inner: build_ty(self, arguments, &scope_inner)?,
                };
                Ok(MaybeConstructor::Constructor(ty_constr))
            }
        }
    }

    fn resolve_use_path(&self, path: &Path) -> ResolveResult<Item> {
        // TODO the current path design does not allow private sub-modules
        //   are they really necessary? if all inner items are private it's effectively equivalent

        // TODO allow private visibility in child and sibling paths
        let vis = Visibility::Public;
        let mut curr_dir = self.database.root_directory;

        let Path { span: _, steps, id } = path;

        for step in steps {
            let curr_dir_info = &self.database[curr_dir];
            curr_dir = curr_dir_info.children.get(&step.string).copied().ok_or_else(|| {
                let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                options.sort();

                self.diagnostic("invalid path step")
                    .snippet(path.span)
                    .add_error(step.span, "invalid step")
                    .finish()
                    .footer(Level::Info, format!("possible options: {:?}", options))
                    .finish()
            })?;
        }

        let file = self.database[curr_dir].file.ok_or_else(|| {
            self.diagnostic_simple("expected path to file", path.span, "no file exists at this path")
        })?;
        let file_scope = self.database[file].local_scope.as_ref().unwrap();

        // TODO change root scope to just be a map instead of a scope so we can avoid this unwrap
        let value = file_scope.find(&self.database, None, id, vis)?;
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
        let result = match expr.inner {
            ExpressionKind::Dummy => self.diagnostic_todo(expr.span, "dummy expression"),
            ExpressionKind::Wrapped(ref inner) => self.eval_expression(scope, inner)?,
            ExpressionKind::Id(ref id) => {
                match scope.find(&self.database, None, id, Visibility::Private)? {
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
            ExpressionKind::TypeFunc(_, _) => self.diagnostic_todo(expr.span, "typefunc expression"),
            ExpressionKind::Block(_) => self.diagnostic_todo(expr.span, "block expression"),
            ExpressionKind::If(_) => self.diagnostic_todo(expr.span, "if expression"),
            ExpressionKind::Loop(_) => self.diagnostic_todo(expr.span, "loop expression"),
            ExpressionKind::While(_) => self.diagnostic_todo(expr.span, "while expression"),
            ExpressionKind::For(_) => self.diagnostic_todo(expr.span, "for expression"),
            ExpressionKind::Sync(_) => self.diagnostic_todo(expr.span, "sync expression"),
            ExpressionKind::Return(_) => self.diagnostic_todo(expr.span, "return expression"),
            ExpressionKind::Break(_) => self.diagnostic_todo(expr.span, "break expression"),
            ExpressionKind::Continue => self.diagnostic_todo(expr.span, "continue expression"),
            ExpressionKind::IntPattern(ref pattern) => {
                let value = match pattern {
                    IntPattern::Hex(_) => self.diagnostic_todo(expr.span, "hex int-pattern expression"),
                    IntPattern::Bin(_) => self.diagnostic_todo(expr.span, "bin int-pattern expression"),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        str_clean.parse::<BigInt>().unwrap()
                    }
                };
                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Int(value)))
            }
            ExpressionKind::BoolLiteral(_) => self.diagnostic_todo(expr.span, "boolliteral expression"),
            ExpressionKind::StringLiteral(_) => self.diagnostic_todo(expr.span, "stringliteral expression"),
            ExpressionKind::ArrayLiteral(_) => self.diagnostic_todo(expr.span, "arrayliteral expression"),
            ExpressionKind::TupleLiteral(_) => self.diagnostic_todo(expr.span, "tupleliteral expression"),
            ExpressionKind::StructLiteral(_) => self.diagnostic_todo(expr.span, "structliteral expression"),
            ExpressionKind::RangeLiteral(ref range) => {
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
            ExpressionKind::UnaryOp(op, ref inner) => {
                let result = match op {
                    UnaryOp::Neg => {
                        Value::Binary(
                            BinaryOp::Sub,
                            Box::new(Value::Int(BigInt::ZERO)),
                            Box::new(self.eval_expression_as_value(scope, inner)?),
                        )
                    }
                    UnaryOp::Not => self.diagnostic_todo(expr.span, "unaryop not expression"),
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            },
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_as_value(scope, left)?;
                let right = self.eval_expression_as_value(scope, right)?;

                let result = Value::Binary(op, Box::new(left), Box::new(right));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            },
            ExpressionKind::TernarySelect(_, _, _) => self.diagnostic_todo(expr.span, "ternaryselect expression"),
            ExpressionKind::ArrayIndex(_, _) => self.diagnostic_todo(expr.span, "arrayindex expression"),
            ExpressionKind::DotIdIndex(_, _) => self.diagnostic_todo(expr.span, "dotidindex expression"),
            ExpressionKind::DotIntIndex(_, _) => self.diagnostic_todo(expr.span, "dotintindex expression"),
            ExpressionKind::Call(ref target, ref args) => {
                if let ExpressionKind::Id(id) = &target.inner {
                    if let Some(name) = id.string.strip_prefix("__builtin_") {
                        return Ok(MaybeConstructor::Immediate(self.eval_builtin_call(scope, expr.span, name, args)?));
                    }
                }

                let target_entry = self.eval_expression(scope, target)?;

                match target_entry {
                    ScopedEntryDirect::Constructor(constr) => {
                        // goal: replace parameters with the arguments of this call
                        let Constructor { inner, parameters } = constr;

                        // check count match
                        if parameters.vec.len() != args.inner.len() {
                            throw!(self.diagnostic_simple(
                                format!("constructor argument count mismatch, expected {}, got {}", parameters.vec.len(), args.inner.len()),
                                args.span,
                                format!("expected {} arguments, got {}", parameters.vec.len(), args.inner.len()),
                            ))
                        }

                        // check kind and type match, and collect in replacement map
                        let mut replacement_map: IndexMap<GenericParameterUniqueId, TypeOrValue> = IndexMap::new();
                        for (param, arg) in zip_eq(&parameters.vec, &args.inner) {
                            let arg_evaluated = match param {
                                GenericParameter::Type(_param) => {
                                    let arg_ty = self.eval_expression_as_ty(scope, arg)?;
                                    // TODO bound-check
                                    TypeOrValue::Type(arg_ty)
                                }
                                GenericParameter::Value(_param) => {
                                    let arg_value = self.eval_expression_as_value(scope, arg)?;
                                    // TODO type-check
                                    TypeOrValue::Value(arg_value)
                                }
                            };
                            replacement_map.insert(param.unique_id(), arg_evaluated);
                        }

                        // do the actual replacement
                        let result = inner.replace_generic_params(&replacement_map);
                        MaybeConstructor::Immediate(result)
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => throw!(
                                self.diagnostic_simple("invalid call target", target.span, "invalid call target kind 'type'")
                            ),
                            TypeOrValue::Value(_) => {
                                self.diagnostic_todo(target.span, "value call target")
                            },
                        }
                    }
                }
            },
        };
        Ok(result)
    }

    fn eval_expression_as_ty(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Type> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => throw!(
                self.diagnostic_simple("expected type, got constructor", expr.span, "constructor")
            ),
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(ty) => Ok(ty),
                TypeOrValue::Value(_) => throw!(
                    self.diagnostic_simple("expected type, got value", expr.span, "value")
                ),
            }
        }
    }

    fn eval_expression_as_value(&mut self, scope: &Scope, expr: &Expression) -> ResolveResult<Value> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => throw!(
                self.diagnostic_simple("expected value, got constructor", expr.span, "constructor")
            ),
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(_) => throw!({
                    self.diagnostic_simple("expected value, got type", expr.span, "type")
                }),
                TypeOrValue::Value(value) => Ok(value),
            }
        }
    }

    fn eval_builtin_call(&mut self, scope: &Scope, expr_span: Span, name: &str, args: &Args) -> ResolveResult<TypeOrValue> {
        // TODO disallow calling builtin outside of stdlib?
        match name {
            "type" => {
                let first_arg = args.inner.get(0).map(|e| &e.inner);
                if let Some(ExpressionKind::StringLiteral(ty)) = first_arg {
                    match ty.as_str() {
                        "bool" if args.inner.len() == 1 =>
                            return Ok(TypeOrValue::Type(Type::Boolean)),
                        "int" if args.inner.len() == 1 => {
                            let range = Box::new(Value::Range(ValueRangeInfo::unbounded()));
                            return Ok(TypeOrValue::Type(Type::Integer(IntegerTypeInfo { range })));
                        }
                        "int_range" if args.inner.len() == 2 => {
                            // TODO typecheck (range must be integer)
                            let range = Box::new(self.eval_expression_as_value(scope, &args.inner[1])?);
                            let ty_info = IntegerTypeInfo { range };
                            return Ok(TypeOrValue::Type(Type::Integer(ty_info)));
                        },
                        "Range" if args.inner.len() == 1 =>
                            return Ok(TypeOrValue::Type(Type::Range)),
                        "bits" if args.inner.len() == 2 => {
                            // TODO typecheck (bits must be non-negative integer)
                            let bits = self.eval_expression_as_value(scope, &args.inner[1])?;
                            return Ok(TypeOrValue::Type(Type::Bits(Box::new(bits))));
                        }
                        "Array" if args.inner.len() == 3 => {
                            let ty = self.eval_expression_as_ty(scope, &args.inner[1])?;
                            let len = self.eval_expression_as_value(scope, &args.inner[2])?;
                            return Ok(TypeOrValue::Type(Type::Array(Box::new(ty), Box::new(len))));
                        }
                        // fallthrough
                        _ => {},
                    }
                }
            }
            // fallthrough
            _ => {},
        }

        Err(
            self.diagnostic("invalid builtin arguments")
                .snippet(expr_span)
                .add_error(args.span, "invalid arguments")
                .finish()
                .finish()
                .into()
        )
    }

    fn get_item_ast(&self, item_reference: ItemReference) -> &'d ast::Item {
        let ItemReference { file, item_index } = item_reference;
        let file_info = &self.database[file];
        let ast = file_info.ast.as_ref().unwrap();
        &ast.items[item_index]
    }

    fn diagnostic(&self, title: impl Into<String>) -> Diagnostic<'d> {
        self.database.diagnostic(title)
    }

    fn diagnostic_defined_twice(&self, kind: &str, span: Span, prev: &Identifier, curr: &Identifier) -> DiagnosticError {
        self.diagnostic(format!("duplicate {:?}", kind))
            .snippet(span)
            .add_info(prev.span, "previously defined here")
            .add_error(curr.span, "defined for the second time here")
            .finish()
            .finish()
    }

    fn diagnostic_simple(&self, title: impl Into<String>, span: Span, label: impl Into<String>) -> DiagnosticError {
        self.diagnostic(title)
            .add_error(span, label)
            .finish()
    }

    #[track_caller]
    fn diagnostic_todo(&self, span: Span, feature: &str) -> ! {
        let message = format!("feature not yet implemented: {}", feature);
        let err = self.diagnostic(&message)
            .add_error(span, "used here")
            .finish();
        println!("{}", err.string);
        panic!("{}", message)
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
    pub fn new(id: FileId, path: FilePath, path_raw: String, directory: Directory, source: String) -> Self {
        FileInfo {
            id,
            path,
            path_raw,
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

// TODO double-check that this was actually finished in the drop implementation? same for snippet
// TODO switch to different error collection system to support multiple errors and warnings
#[must_use]
pub struct Diagnostic<'d> {
    title: String,
    snippets: Vec<(Span, Vec<(Level, Span, String)>)>,
    footers: Vec<(Level, String)>,

    // This is only stored here to make the finish call slightly neater,
    //   but could be removed again if the lifetimes are too tricky.
    database: &'d SourceDatabase,
}

#[must_use]
pub struct DiagnosticSnippet<'d> {
    diag: Diagnostic<'d>,
    span: Span,
    annotations: Vec<(Level, Span, String)>,
}

impl<'d> Diagnostic<'d> {
    pub fn new(database: &'d SourceDatabase, title: impl Into<String>) -> Self {
        Diagnostic {
            title: title.into(),
            snippets: vec![],
            footers: vec![],
            database,
        }
    }

    pub fn snippet(self, span: Span) -> DiagnosticSnippet<'d> {
        DiagnosticSnippet {
            diag: self,
            span,
            annotations: vec![],
        }
    }

    pub fn footer(mut self, level: Level, footer: impl Into<String>) -> Self {
        self.footers.push((level, footer.into()));
        self
    }

    pub fn finish(self) -> DiagnosticError {
        let Self { title, snippets, footers, database } = self;
        assert!(!snippets.is_empty(), "Diagnostic without any snippets is not allowed");

        let mut message = Level::Error.title(&title);

        for &(span, ref annotations) in &snippets {
            let file_info = &database[span.start.file];
            let offsets = &file_info.offsets;

            // select lines and convert to bytes
            let span_snippet = offsets.expand_span(span);
            let start_line_0 = span_snippet.start.line_0.saturating_sub(SNIPPET_CONTEXT_LINES);
            let end_line_0 = min(span_snippet.end.line_0 + SNIPPET_CONTEXT_LINES, offsets.line_count() - 1);
            let start_byte = offsets.line_start_byte(start_line_0);
            let end_byte = offsets.line_start_byte(end_line_0);
            let source = &file_info.source[start_byte..end_byte];

            // create snippet
            let mut snippet = Snippet::source(source)
                .origin(&file_info.path_raw)
                .line_start(start_line_0 + 1);
            for (level, span_annotation, label) in annotations {
                let span_byte = (span_annotation.start.byte - start_byte)..(span_annotation.end.byte - start_byte);
                snippet = snippet.annotation(level.span(span_byte).label(label));
            }

            message = message.snippet(snippet);
        }

        for &(level, ref footer) in &footers {
            message = message.footer(level.title(footer));
        }

        let renderer = Renderer::styled();
        let string = renderer.render(message).to_string();
        DiagnosticError { string }
    }
}

impl DiagnosticAddable for Diagnostic<'_> {
    fn add(self, level: Level, span: Span, label: impl Into<String>) -> Self {
        self.snippet(span).add(level, span, label).finish()
    }
}

impl<'d> DiagnosticSnippet<'d> {
    pub fn finish(self) -> Diagnostic<'d> {
        let Self { mut diag, span, annotations } = self;
        assert!(!annotations.is_empty(), "DiagnosticSnippet without any annotations is not allowed");
        diag.snippets.push((span, annotations));
        diag
    }
}

impl DiagnosticAddable for DiagnosticSnippet<'_> {
    fn add(mut self, level: Level, span: Span, label: impl Into<String>) -> Self {
        assert!(self.span.contains(span), "DiagnosticSnippet labels must fall within snippet span");
        self.annotations.push((level, span, label.into()));
        self
    }
}

pub trait DiagnosticAddable: Sized {
    fn add(self, level: Level, span: Span, label: impl Into<String>) -> Self;

    fn add_error(self, span: Span, label: impl Into<String>) -> Self {
        self.add(Level::Error, span, label)
    }

    fn add_info(self, span: Span, label: impl Into<String>) -> Self {
        self.add(Level::Info, span, label)
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