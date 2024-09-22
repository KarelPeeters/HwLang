use crate::data::compiled::{CompiledDatabase, CompiledDatabasePartial, FunctionParameter, FunctionSignatureInfo, FunctionTypeParameterInfo, FunctionValueParameterInfo, GenericParameter, GenericTypeParameter, GenericTypeParameterInfo, GenericValueParameter, GenericValueParameterInfo, Item, ItemBody, ItemInfo, ItemInfoPartial, ModulePortInfo, ModuleSignatureInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::data::parsed::{ItemAstReference, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::common::{GenericContainer, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::scope::{Scope, Scopes, Visibility};
use crate::front::types::{Constructor, EnumTypeInfo, FunctionParameters, FunctionTypeInfo, GenericArguments, GenericParameters, IntegerTypeInfo, MaybeConstructor, ModuleTypeInfo, NominalTypeUnique, StructTypeInfo, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::{Args, BinaryOp, EnumVariant, Expression, ExpressionKind, GenericParameterKind, IntPattern, ItemDefEnum, ItemDefFunction, ItemDefModule, ItemDefStruct, ItemDefType, ItemUse, Path, PortKind, RangeLiteral, Spanned, StructField, SyncDomain, SyncKind, UnaryOp};
use crate::syntax::pos::Span;
use crate::syntax::{ast, parse_error_to_diagnostic, parse_file_content};
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Itertools};
use num_bigint::BigInt;

pub fn compile(diagnostics: &Diagnostics, database: &SourceDatabase) -> (ParsedDatabase, CompiledDatabase) {
    // sort files to ensure platform-independence
    // TODO make this the responsibility of the database builder, now file ids are still not deterministic
    let files_sorted = database.files.keys()
        .copied()
        .sorted_by_key(|&file| &database[database[file].directory].path)
        .collect_vec();

    // items only exists to serve as a level of indirection between values,
    //   so we can easily do the graph solution in a single pass
    let mut items: Arena<Item, ItemInfoPartial> = Arena::default();

    // parse all files and populate local scopes
    let mut file_ast = IndexMap::new();
    let mut file_scope = IndexMap::new();
    let mut scopes = Scopes::default();

    for file in files_sorted {
        let file_info = &database[file];

        // parse
        let (ast, scope) = match parse_file_content(file, &file_info.source) {
            Ok(ast) => {
                // build local scope
                // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
                let local_scope = scopes.new_root(file_info.offsets.full_span(file));
                let local_scope_info = &mut scopes[local_scope];

                for (file_item_index, ast_item) in enumerate(&ast.items) {
                    let common_info = ast_item.common_info();
                    let vis = match common_info.vis {
                        ast::Visibility::Public(_) => Visibility::Public,
                        ast::Visibility::Private => Visibility::Private,
                    };

                    let item = items.push(ItemInfo {
                        ast_ref: ItemAstReference { file, file_item_index },
                        ty: None,
                        body: None,
                    });

                    local_scope_info.maybe_declare(diagnostics, &common_info.id, ScopedEntry::Item(item), vis);
                }

                (Ok(ast), Ok(local_scope))
            }
            Err(e) => {
                let e = diagnostics.report(parse_error_to_diagnostic(e));
                (Err(e), Err(e))
            }
        };

        file_ast.insert_first(file, ast);
        file_scope.insert_first(file, scope);
    }

    let parsed = ParsedDatabase { file_ast };

    let mut state = CompileState {
        diag: diagnostics,
        source: database,
        parsed: &parsed,
        compiled: CompiledDatabase {
            items,
            file_scope,
            generic_type_params: Arena::default(),
            generic_value_params: Arena::default(),
            function_type_params: Arena::default(),
            function_value_params: Arena::default(),
            module_ports: Arena::default(),
            module_info: IndexMap::new(),
            function_info: IndexMap::new(),
            scopes,
        },
        log_const_eval: false,
    };

    // resolve all item types (which is mostly their signatures)
    // TODO randomize order to check for dependency bugs? but then diagnostics have random orders
    let item_keys = state.compiled.items.keys().collect_vec();
    for &item in &item_keys {
        state.resolve_item_type_fully(item);
    }

    // typecheck all item bodies
    // TODO merge this with the previous pass: better for LSP and maybe for local items
    for &item in &item_keys {
        assert!(state.compiled[item].body.is_none());
        let body = match state.resolve_item_body(item) {
            Ok(body) => body,
            Err(ResolveFirst(_)) => panic!("all types should be resolved by now"),
        };

        let slot = &mut state.compiled[item].body;
        assert!(slot.is_none());
        *slot = Some(body);
    }

    // map to final database
    let items = state.compiled.items.map_values(|info| ItemInfo {
        ast_ref: info.ast_ref,
        ty: info.ty.unwrap(),
        body: info.body.unwrap(),
    });

    let compiled = CompiledDatabase {
        file_scope: state.compiled.file_scope,
        scopes: state.compiled.scopes,
        items,
        generic_type_params: state.compiled.generic_type_params,
        generic_value_params: state.compiled.generic_value_params,
        function_type_params: state.compiled.function_type_params,
        function_value_params: state.compiled.function_value_params,
        module_info: state.compiled.module_info,
        module_ports: state.compiled.module_ports,
        function_info: state.compiled.function_info,
    };

    (parsed, compiled)
}

// TODO create some dedicated auxiliary data structure, with dense and non-dense variants
pub(super) struct CompileState<'d, 'a> {
    pub(super) log_const_eval: bool,
    pub(super) diag: &'d Diagnostics,
    pub(super) source: &'d SourceDatabase,
    pub(super) parsed: &'a ParsedDatabase,
    pub(super) compiled: CompiledDatabasePartial,
}

#[derive(Debug, Copy, Clone)]
pub struct ResolveFirst(Item);

pub type ResolveResult<T> = Result<T, ResolveFirst>;

#[derive(Debug, Copy, Clone)]
pub enum EvalTrueError {
    False,
    Unknown,
}

impl EvalTrueError {
    pub fn to_message(self) -> &'static str {
        match self {
            EvalTrueError::False => "must be true but is false",
            // TODO ask user to report issue if they think it actually is provable
            EvalTrueError::Unknown => "must be true but is unknown",
        }
    }
}

impl<'d, 'a> CompileState<'d, 'a> {
    fn resolve_item_type_fully(&mut self, item: Item) {
        let mut stack = vec![item];

        // TODO avoid repetitive work by switching to async instead?
        while let Some(curr) = stack.pop() {
            if self.compiled[curr].ty.is_some() {
                // already resolved, skip
                continue;
            }

            let resolved = match self.resolve_item_type_new(curr) {
                Ok(resolved) => resolved,
                Err(ResolveFirst(first)) => {
                    assert!(self.compiled[first].ty.is_none(), "request to resolve {first:?} first, but it already has a type");

                    // push curr failed attempt back on the stack
                    stack.push(curr);

                    // check for cycle
                    let cycle_start_index = stack.iter().position(|s| s == &first);
                    if let Some(cycle_start_index) = cycle_start_index {
                        // cycle detected, report error
                        let cycle = &stack[cycle_start_index..];

                        // build diagnostic
                        // TODO the order is nondeterministic, it depends on which items happened to be visited first
                        let mut diag = Diagnostic::new("cyclic type dependency");
                        for &stack_item in cycle {
                            let item_ast = self.parsed.item_ast(self.compiled[stack_item].ast_ref);
                            diag = diag.add_error(item_ast.common_info().span_short, "part of cycle");
                        }
                        let err = self.diag.report(diag.finish());

                        // set slot of all involved items
                        for &stack_item in cycle {
                            let slot = &mut self.compiled[stack_item].ty;
                            assert!(slot.is_none(), "someone else already set the type for {curr:?}");
                            *slot = Some(MaybeConstructor::Error(err));
                        }

                        // remove cycle from the stack
                        drop(stack.drain(cycle_start_index..));
                        continue;
                    } else {
                        // no cycle, visit the next item
                        stack.push(first);
                        continue;
                    }
                }
            };

            // managed to resolve the current item, store its type
            let slot = &mut self.compiled[curr].ty;
            assert!(slot.is_none(), "someone else already set the type for {curr:?}");
            *slot = Some(resolved);
        }
    }

    fn resolve_item_type(&self, item: Item) -> ResolveResult<&MaybeConstructor<Type>> {
        match self.compiled[item].ty {
            Some(ref r) => Ok(r),
            None => Err(ResolveFirst(item)),
        }
    }

    // TODO this signature is wrong: items are not always type constructors
    // TODO clarify: this resolves the _signature_, not the body, right?
    //   for type aliases it appears to resolve the body too.
    fn resolve_item_type_new(&mut self, item: Item) -> ResolveResult<MaybeConstructor<Type>> {
        // check that this is indeed a new query
        assert!(self.compiled[item].ty.is_none());

        // item lookup
        let item_ast = self.parsed.item_ast(self.compiled[item].ast_ref);
        let scope_file = match *self.compiled.file_scope.get(&self.compiled[item].ast_ref.file).unwrap() {
            Ok(scope_file) => scope_file,
            Err(e) => return Ok(MaybeConstructor::Error(e)),
        };

        // actual resolution
        match *item_ast {
            // use indirection
            ast::Item::Use(ItemUse { span: _, ref path, as_: _ }) => {
                // TODO why are we handling use items here? can they not be eliminated by scope building
                //  this is really weird, use items don't even really have signatures
                let next_item = self.resolve_use_path(path)?;
                match next_item {
                    Ok(next_item) => Ok(self.resolve_item_type(next_item)?.clone()),
                    Err(e) => Ok(MaybeConstructor::Error(e)),
                }
            }
            // type definitions
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, ref params, ref inner }) => {
                self.resolve_new_generic_type_def(item, scope_file, params, |s, _args, scope_inner| {
                    Ok(Ok(s.eval_expression_as_ty(scope_inner, inner)?))
                })
            }
            ast::Item::Struct(ItemDefStruct { span, vis: _, id: _, ref params, ref fields }) => {
                self.resolve_new_generic_type_def(item, scope_file, params, |s, args, scope_inner| {
                    // map fields
                    let mut fields_map = IndexMap::new();
                    for field in fields {
                        let StructField { span: _, id: field_id, ty } = field;
                        let field_ty = s.eval_expression_as_ty(scope_inner, ty)?;

                        let prev = fields_map.insert(field_id.string.clone(), (field_id, field_ty));
                        if let Some(prev) = prev {
                            let diag = Diagnostic::new_defined_twice("struct field", span, field_id, prev.0);
                            let err = s.diag.report(diag);
                            return Ok(Err(err));
                        }
                    }

                    // result
                    let ty = StructTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        fields: fields_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Ok(Type::Struct(ty)))
                })
            }
            ast::Item::Enum(ItemDefEnum { span, vis: _, id: _, ref params, ref variants }) => {
                self.resolve_new_generic_type_def(item, scope_file, params, |s, args, scope_inner| {
                    // map variants
                    let mut variants_map = IndexMap::new();
                    for variant in variants {
                        let EnumVariant { span: _, id: variant_id, content } = variant;

                        let content = content.as_ref()
                            .map(|content| s.eval_expression_as_ty(scope_inner, content))
                            .transpose()?;

                        let prev = variants_map.insert(variant_id.string.clone(), (variant_id, content));
                        if let Some(prev) = prev {
                            let diag = Diagnostic::new_defined_twice("enum variant", span, variant_id, prev.0);
                            let err = s.diag.report(diag);
                            return Ok(Err(err));
                        }
                    }

                    // result
                    let ty = EnumTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        variants: variants_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Ok(Type::Enum(ty)))
                })
            }
            // value definitions
            ast::Item::Module(ItemDefModule { span: _, vis: _, id: _, ref params, ref ports, ref body }) => {
                self.resolve_new_generic_type_def(item, scope_file, params, |s, args, scope_inner| {
                    // yet another sub-scope for the ports that refer to each other
                    let scope_ports = s.compiled.scopes.new_child(scope_inner, ports.span.join(body.span), Visibility::Private);

                    // map ports
                    let mut port_vec = vec![];

                    for port in &ports.inner {
                        let ast::ModulePort { span: _, id: port_id, direction, kind, } = port;

                        let module_port_info = ModulePortInfo {
                            defining_item: item,
                            defining_id: port_id.clone(),
                            direction: direction.inner,
                            kind: match &kind.inner {
                                PortKind::Clock => PortKind::Clock,
                                PortKind::Normal { sync, ty } => {
                                    PortKind::Normal {
                                        sync: match &sync.inner {
                                            SyncKind::Async => SyncKind::Async,
                                            SyncKind::Sync(SyncDomain { clock, reset }) => {
                                                let clock = s.eval_expression_as_value(scope_ports, clock)?;
                                                let reset = s.eval_expression_as_value(scope_ports, reset)?;
                                                SyncKind::Sync(SyncDomain { clock, reset })
                                            }
                                        },
                                        ty: s.eval_expression_as_ty(scope_ports, ty)?,
                                    }
                                }
                            },
                        };
                        let module_port = s.compiled.module_ports.push(module_port_info);
                        port_vec.push(module_port);

                        s.compiled.scopes[scope_ports].declare(
                            s.diag,
                            &port_id,
                            ScopedEntry::Direct(MaybeConstructor::Immediate(TypeOrValue::Value(Value::ModulePort(module_port)))),
                            Visibility::Private,
                        );
                    }

                    // result
                    let module_info = ModuleSignatureInfo { scope_ports };
                    s.compiled.module_info.insert_first(item, module_info);

                    let module_ty_info = ModuleTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        ports: port_vec,
                    };

                    Ok(Ok(Type::Module(module_ty_info)))
                })
            }
            ast::Item::Const(_) =>
                Ok(MaybeConstructor::Error(self.diag.report_todo(item_ast.common_info().span_short, "const definition"))),
            ast::Item::Function(ItemDefFunction { span: _, vis: _, id: _, ref params, ref ret_ty, ref body }) => {
                let scope_params = self.compiled.scopes.new_child(scope_file, params.span.join(body.span), Visibility::Private);

                let mut parameters = vec![];

                for param_ast in &params.inner {
                    // TODO name arg/param better
                    let (param, arg) = match &param_ast.kind {
                        &GenericParameterKind::Type(_span) => {
                            let param = self.compiled.function_type_params.push(FunctionTypeParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                            });
                            (FunctionParameter::Type(param), TypeOrValue::Type(Type::FunctionParameter(param)))
                        }
                        GenericParameterKind::Value(ty_expr) => {
                            let ty = self.eval_expression_as_ty(scope_params, ty_expr)?;
                            let param = self.compiled.function_value_params.push(FunctionValueParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                                ty,
                                ty_span: ty_expr.span,
                            });
                            (FunctionParameter::Value(param), TypeOrValue::Value(Value::FunctionParameter(param)))
                        }
                    };

                    parameters.push(param);

                    // TODO should we nest scopes here, or is incremental declaration in a single scope equivalent?
                    let entry = ScopedEntry::Direct(MaybeConstructor::Immediate(arg));
                    self.compiled.scopes[scope_params].declare(self.diag, &param_ast.id, entry, Visibility::Private);
                }

                let ret_ty = match ret_ty {
                    None => Type::Unit,
                    Some(ret_ty) => self.eval_expression_as_ty(scope_params, ret_ty)?,
                };

                // keep scope for later
                self.compiled.function_info.insert_first(item, FunctionSignatureInfo { scope_params });

                // result type
                let ty_info = FunctionTypeInfo {
                    params: FunctionParameters { vec: parameters },
                    ret: Box::new(ret_ty),
                };
                Ok(MaybeConstructor::Immediate(Type::Function(ty_info)))
            }
            ast::Item::Interface(_) =>
                Ok(MaybeConstructor::Error(self.diag.report_todo(item_ast.common_info().span_short, "interface definition"))),
        }
    }

    // TODO can this be used for functions? they don't really define a type independent of the parameters
    fn resolve_new_generic_type_def<T>(
        &mut self,
        item: Item,
        scope_outer: Scope,
        params: &Option<Spanned<Vec<ast::GenericParameter>>>,
        build_ty: impl FnOnce(&mut Self, GenericArguments, Scope) -> ResolveResult<Result<T, ErrorGuaranteed>>,
    ) -> ResolveResult<MaybeConstructor<T>> {
        let item_span = self.parsed.item_ast(self.compiled[item].ast_ref).common_info().span_full;
        let scope_inner = self.compiled.scopes.new_child(scope_outer, item_span, Visibility::Private);

        match params {
            None => {
                // there are no parameters, just map directly
                // the scope still needs to be "nested" since the builder needs an owned scope
                let arguments = GenericArguments { vec: vec![] };
                match build_ty(self, arguments, scope_inner)? {
                    Ok(ty) => Ok(MaybeConstructor::Immediate(ty)),
                    Err(e) => Ok(MaybeConstructor::Error(e)),
                }
            }
            Some(params) => {
                // build inner scope
                let mut parameters = vec![];
                let mut arguments = vec![];

                for param_ast in &params.inner {
                    // TODO name arg/param better
                    let (param, arg) = match &param_ast.kind {
                        &GenericParameterKind::Type(_span) => {
                            let param = self.compiled.generic_type_params.push(GenericTypeParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                            });
                            (GenericParameter::Type(param), TypeOrValue::Type(Type::GenericParameter(param)))
                        }
                        GenericParameterKind::Value(ty_expr) => {
                            let ty = self.eval_expression_as_ty(scope_inner, ty_expr)?;
                            let param = self.compiled.generic_value_params.push(GenericValueParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                                ty,
                                ty_span: ty_expr.span,
                            });
                            (GenericParameter::Value(param), TypeOrValue::Value(Value::GenericParameter(param)))
                        }
                    };

                    parameters.push(param);
                    arguments.push(arg.clone());

                    // TODO should we nest scopes here, or is incremental declaration in a single scope equivalent?
                    let entry = ScopedEntry::Direct(MaybeConstructor::Immediate(arg));
                    self.compiled.scopes[scope_inner].declare(self.diag, &param_ast.id, entry, Visibility::Private);
                }

                let parameters = GenericParameters { vec: parameters };
                let arguments = GenericArguments { vec: arguments };

                // build inner type
                let inner = match build_ty(self, arguments, scope_inner) {
                    Ok(Ok(inner)) => inner,
                    Ok(Err(e)) => return Ok(MaybeConstructor::Error(e)),
                    Err(first) => return Err(first),
                };

                // result
                let ty_constr = Constructor { parameters, inner };
                Ok(MaybeConstructor::Constructor(ty_constr))
            }
        }
    }

    fn resolve_item_body(&mut self, item: Item) -> ResolveResult<ItemBody> {
        assert!(self.compiled[item].body.is_none());

        let item_ty = self.compiled[item].ty.as_ref().expect("item should already have been checked");
        let item_ty_err = match item_ty {
            &MaybeConstructor::Error(e) => Some(e),
            _ => None,
        };

        let item_ast = self.parsed.item_ast(self.compiled[item].ast_ref);
        let item_span = item_ast.common_info().span_short;

        let body = match item_ast {
            // these items are fully defined by their type, which was already checked earlier
            ast::Item::Use(_) | ast::Item::Type(_) | ast::Item::Struct(_) | ast::Item::Enum(_) => ItemBody::None,
            ast::Item::Const(_) => {
                match item_ty_err {
                    Some(e) => ItemBody::Error(e),
                    None => ItemBody::Error(self.diag.report_todo(item_span, "const body")),
                }
            }
            ast::Item::Function(_) => {
                match item_ty_err {
                    Some(e) => ItemBody::Error(e),
                    None => ItemBody::Error(self.diag.report_todo(item_span, "function body")),
                }
            }
            ast::Item::Module(item_ast) =>
                ItemBody::Module(self.resolve_module_body(item, item_ast)?),
            ast::Item::Interface(_) => {
                match item_ty_err {
                    Some(e) => ItemBody::Error(e),
                    None => ItemBody::Error(self.diag.report_todo(item_span, "interface body")),
                }
            }
        };
        Ok(body)
    }

    fn resolve_use_path(&self, path: &Path) -> ResolveResult<Result<Item, ErrorGuaranteed>> {
        // TODO the current path design does not allow private sub-modules
        //   are they really necessary? if all inner items are private it's effectively equivalent
        //   -> no it's not equivalent, things can also be private from the parent

        // TODO allow private visibility in child and sibling paths
        let vis = Visibility::Public;
        let mut curr_dir = self.source.root_directory;

        let Path { span: _, steps, id } = path;

        for step in steps {
            let curr_dir_info = &self.source[curr_dir];

            curr_dir = match curr_dir_info.children.get(&step.string) {
                Some(&child_dir) => child_dir,
                None => {
                    let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                    options.sort();

                    let diag = Diagnostic::new("invalid path step")
                        .snippet(path.span)
                        .add_error(step.span, "invalid step")
                        .finish()
                        .footer(Level::Info, format!("possible options: {:?}", options))
                        .finish();
                    let err = self.diag.report(diag);
                    return Ok(Err(err));
                }
            };
        }

        let file = match self.source[curr_dir].file {
            Some(file) => file,
            None => {
                let diag = Diagnostic::new_simple("expected path to file", path.span, "no file exists at this path");
                let err = self.diag.report(diag);
                return Ok(Err(err));
            }
        };

        let file_scope = match *self.compiled.file_scope.get(&file).unwrap() {
            Ok(scope) => scope,
            Err(e) => return Ok(Err(e)),
        };

        // TODO change root scope to just be a map instead of a scope so we can avoid this unwrap
        let entry = match self.compiled.scopes[file_scope].find(&self.compiled.scopes, self.diag, id, vis) {
            Err(e) => return Ok(Err(e)),
            Ok(entry) => entry,
        };
        match entry.value {
            &ScopedEntry::Item(item) => Ok(Ok(item)),
            // TODO is this still true?
            ScopedEntry::Direct(_) => unreachable!("file root entries should not exist"),
        }
    }

    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&mut self, scope: Scope, expr: &Expression) -> ResolveResult<ScopedEntryDirect> {
        let result = match expr.inner {
            ExpressionKind::Dummy =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "dummy expression")),
            ExpressionKind::Any =>
                ScopedEntryDirect::Immediate(TypeOrValue::Type(Type::Any)),
            ExpressionKind::Wrapped(ref inner) =>
                self.eval_expression(scope, inner)?,
            ExpressionKind::Id(ref id) => {
                let entry = match self.compiled.scopes[scope].find(&self.compiled.scopes, self.diag, id, Visibility::Private) {
                    Err(e) => return Ok(ScopedEntryDirect::Error(e)),
                    Ok(entry) => entry,
                };
                match entry.value {
                    &ScopedEntry::Item(item) => {
                        // TODO properly support value items, and in general fix "type" vs "value" resolution
                        //  maybe through checking the item kind first?
                        //  each of them clearly only defines a type or value, right?
                        //    or do we want to support "type A = if(cond) B else C"?
                        self.resolve_item_type(item)?
                            .clone()
                            .map(TypeOrValue::Type)
                    }
                    ScopedEntry::Direct(entry) => entry.clone(),
                }
            }
            ExpressionKind::TypeFunc(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "type func expression")),
            ExpressionKind::Block(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "block expression")),
            ExpressionKind::If(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "if expression")),
            ExpressionKind::Loop(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "loop expression")),
            ExpressionKind::While(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "while expression")),
            ExpressionKind::For(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "for expression")),
            ExpressionKind::Return(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "return expression")),
            ExpressionKind::Break(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "break expression")),
            ExpressionKind::Continue =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "continue expression")),
            ExpressionKind::IntPattern(ref pattern) => {
                match pattern {
                    IntPattern::Hex(_) =>
                        ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        let value = str_clean.parse::<BigInt>().unwrap();
                        ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Int(value)))
                    }
                }
            }
            ExpressionKind::BoolLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "bool literal expression")),
            ExpressionKind::StringLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "string literal expression")),
            ExpressionKind::ArrayLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "array literal expression")),
            ExpressionKind::TupleLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "tuple literal expression")),
            ExpressionKind::StructLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "struct literal expression")),
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

                if let (Some(start), Some(end)) = (&start, &end) {
                    let op = if end_inclusive { BinaryOp::CmpLt } else { BinaryOp::CmpLte };
                    match self.require_value_true_for_range(expr.span, &Value::Binary(op, start.clone(), end.clone())) {
                        Ok(()) => {}
                        Err(e) => return Ok(ScopedEntryDirect::Error(e)),
                    }
                }

                let value = Value::Range(RangeInfo::new(start, end, end_inclusive));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(value))
            }
            ExpressionKind::UnaryOp(op, ref inner) => {
                let result = match op {
                    UnaryOp::Neg => {
                        Value::Binary(
                            BinaryOp::Sub,
                            Box::new(Value::Int(BigInt::ZERO)),
                            Box::new(self.eval_expression_as_value(scope, inner)?),
                        )
                    }
                    UnaryOp::Not =>
                        Value::UnaryNot(Box::new(self.eval_expression_as_value(scope, inner)?)),
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_as_value(scope, left)?;
                let right = self.eval_expression_as_value(scope, right)?;

                let result = Value::Binary(op, Box::new(left), Box::new(right));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::TernarySelect(_, _, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "ternary select expression")),
            ExpressionKind::ArrayIndex(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "array index expression")),
            ExpressionKind::DotIdIndex(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "dot id index expression")),
            ExpressionKind::DotIntIndex(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "dot int index expression")),
            ExpressionKind::Call(ref target, ref args) => {
                if let ExpressionKind::Id(id) = &target.inner {
                    if let Some(name) = id.string.strip_prefix("__builtin_") {
                        return match self.eval_builtin_call(scope, expr.span, name, args)? {
                            Ok(result) => Ok(MaybeConstructor::Immediate(result)),
                            Err(e) => Ok(MaybeConstructor::Error(e)),
                        };
                    }
                }

                let target_entry = self.eval_expression(scope, target)?;

                match target_entry {
                    ScopedEntryDirect::Constructor(constr) => {
                        // goal: replace parameters with the arguments of this call
                        let Constructor { inner, parameters } = constr;

                        // check count match
                        if parameters.vec.len() != args.inner.len() {
                            let err = Diagnostic::new_simple(
                                format!("constructor argument count mismatch, expected {}, got {}", parameters.vec.len(), args.inner.len()),
                                args.span,
                                format!("expected {} arguments, got {}", parameters.vec.len(), args.inner.len()),
                            );
                            return Ok(ScopedEntryDirect::Error(self.diag.report(err)));
                        }

                        // check kind and type match, and collect in replacement map
                        let mut map_ty: IndexMap<GenericTypeParameter, Type> = IndexMap::new();
                        let mut map_value: IndexMap<GenericValueParameter, Value> = IndexMap::new();
                        let mut last_err = None;

                        for (&param, arg) in zip_eq(&parameters.vec, &args.inner) {
                            match param {
                                GenericParameter::Type(param) => {
                                    let arg_ty = self.eval_expression_as_ty(scope, arg)?;
                                    // TODO use for bound-check (once we add type bounds)
                                    let _param_info = &self.compiled[param];
                                    map_ty.insert_first(param, arg_ty)
                                }
                                GenericParameter::Value(param) => {
                                    let arg_value = self.eval_expression_as_value(scope, arg)?;
                                    let param_info = &self.compiled[param];
                                    match self.check_type_contains(param_info.ty_span, arg.span, &param_info.ty, &arg_value) {
                                        Ok(()) => {}
                                        Err(e) => last_err = Some(e),
                                    }
                                    map_value.insert_first(param, arg_value)
                                }
                            }
                        }

                        // only bail once all parameters have been checked
                        if let Some(e) = last_err {
                            return Ok(ScopedEntryDirect::Error(e));
                        }

                        // do the actual replacement
                        let result = inner.replace_generic_params(&mut self.compiled, &map_ty, &map_value);
                        MaybeConstructor::Immediate(result)
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => {
                                let diag = Diagnostic::new_simple("invalid call target", target.span, "invalid call target kind 'type'");
                                ScopedEntryDirect::Error(self.diag.report(diag))
                            }
                            TypeOrValue::Value(_) =>
                                ScopedEntryDirect::Error(self.diag.report_todo(target.span, "value call target")),
                        }
                    }
                    ScopedEntryDirect::Error(e) => ScopedEntryDirect::Error(e),
                }
            }
        };
        Ok(result)
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> ResolveResult<Type> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            // TODO unify these error strings somewhere
            // TODO maybe move back to central error collection place for easier unit testing?
            ScopedEntryDirect::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type, got constructor", expr.span, "constructor");
                Ok(Type::Error(self.diag.report(diag)))
            }
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(ty) => Ok(ty),
                TypeOrValue::Value(_) => {
                    let diag = Diagnostic::new_simple("expected type, got value", expr.span, "value");
                    Ok(Type::Error(self.diag.report(diag)))
                }
            }
            ScopedEntryDirect::Error(e) => Ok(Type::Error(e))
        }
    }

    pub fn eval_expression_as_value(&mut self, scope: Scope, expr: &Expression) -> ResolveResult<Value> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => {
                let err = Diagnostic::new_simple("expected value, got constructor", expr.span, "constructor");
                Ok(Value::Error(self.diag.report(err)))
            }
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(_) => {
                    let err = Diagnostic::new_simple("expected value, got type", expr.span, "type");
                    Ok(Value::Error(self.diag.report(err)))
                }
                TypeOrValue::Value(value) => Ok(value),
            }
            ScopedEntryDirect::Error(e) => Ok(Value::Error(e)),
        }
    }

    pub fn eval_sync_domain(&mut self, scope: Scope, domain: &SyncDomain<Box<Expression>>) -> ResolveResult<SyncDomain<Value>> {
        let clock = self.eval_expression_as_value(scope, &domain.clock)?;
        let reset = self.eval_expression_as_value(scope, &domain.reset)?;

        // TODO check that clock is a clock
        // TODO check that reset is a boolean
        // TODO check that reset is either async or sync to the same clock

        Ok(SyncDomain {
            clock,
            reset,
        })
    }

    fn eval_builtin_call(
        &mut self,
        scope: Scope,
        expr_span: Span,
        name: &str,
        args: &Args,
    ) -> ResolveResult<Result<TypeOrValue, ErrorGuaranteed>> {
        // TODO disallow calling builtin outside of stdlib?
        match name {
            "type" => {
                let first_arg = args.inner.get(0).map(|e| &e.inner);
                if let Some(ExpressionKind::StringLiteral(ty)) = first_arg {
                    match ty.as_str() {
                        "bool" if args.inner.len() == 1 =>
                            return Ok(Ok(TypeOrValue::Type(Type::Boolean))),
                        "int" if args.inner.len() == 1 => {
                            let range = Box::new(Value::Range(RangeInfo::unbounded()));
                            return Ok(Ok(TypeOrValue::Type(Type::Integer(IntegerTypeInfo { range }))));
                        }
                        "int_range" if args.inner.len() == 2 => {
                            // TODO typecheck (range must be integer)
                            let range = Box::new(self.eval_expression_as_value(scope, &args.inner[1])?);
                            let ty_info = IntegerTypeInfo { range };
                            return Ok(Ok(TypeOrValue::Type(Type::Integer(ty_info))));
                        }
                        "Range" if args.inner.len() == 1 =>
                            return Ok(Ok(TypeOrValue::Type(Type::Range))),
                        "bits_inf" if args.inner.len() == 1 => {
                            return Ok(Ok(TypeOrValue::Type(Type::Bits(None))));
                        }
                        "bits" if args.inner.len() == 2 => {
                            // TODO typecheck (bits must be non-negative integer)
                            let bits = self.eval_expression_as_value(scope, &args.inner[1])?;
                            return Ok(Ok(TypeOrValue::Type(Type::Bits(Some(Box::new(bits))))));
                        }
                        "Array" if args.inner.len() == 3 => {
                            let ty = self.eval_expression_as_ty(scope, &args.inner[1])?;
                            let len = self.eval_expression_as_value(scope, &args.inner[2])?;
                            return Ok(Ok(TypeOrValue::Type(Type::Array(Box::new(ty), Box::new(len)))));
                        }
                        // fallthrough
                        _ => {}
                    }
                }
            }
            // fallthrough
            _ => {}
        }

        let err = Diagnostic::new("invalid arguments for __builtin call")
            .snippet(expr_span)
            .add_error(args.span, "invalid arguments")
            .finish()
            .finish()
            .into();
        Ok(Err(self.diag.report(err)))
    }
}
