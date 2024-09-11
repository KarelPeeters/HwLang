use crate::data::compiled::{CompiledDatabase, FileAuxiliary, FunctionParameter, FunctionParameterInfo, GenericParameter, GenericTypeParameter, GenericTypeParameterInfo, GenericValueParameter, GenericValueParameterInfo, Item, ItemBody, ItemInfo, ModulePort, ModulePortInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, DiagnosticError, DiagnosticResult, Diagnostics};
use crate::data::module_body::{CombinatorialStatement, ModuleBlock, ModuleBlockCombinatorial, ModuleBody};
use crate::data::source::SourceDatabase;
use crate::front::common::{GenericContainer, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::scope::{Scope, Scopes, Visibility};
use crate::front::types::{Constructor, EnumTypeInfo, GenericArguments, GenericParameters, IntegerTypeInfo, MaybeConstructor, ModuleTypeInfo, NominalTypeUnique, StructTypeInfo, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::{Args, BinaryOp, EnumVariant, Expression, ExpressionKind, GenericParameterKind, Identifier, IntPattern, ItemDefEnum, ItemDefModule, ItemDefStruct, ItemDefType, ItemUse, Path, PortKind, RangeLiteral, Spanned, StatementKind, StructField, SyncKind, UnaryOp};
use crate::syntax::pos::{FileId, Span};
use crate::syntax::{ast, parse_file_content};
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use crate::{throw, try_opt_result};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Itertools};
use num_bigint::BigInt;
use num_traits::{One, Signed};
use std::collections::HashMap;

pub fn compile(diagnostics: &Diagnostics, database: &SourceDatabase) -> CompiledDatabase {
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
    let mut file_auxiliary = IndexMap::new();
    let mut scopes = Scopes::default();

    for file in files_sorted {
        let file_info = &database[file];

        // parse
        let aux = match parse_file_content(file, &file_info.source) {
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

                    let item = items.push(ItemInfoPartial {
                        file,
                        file_item_index,
                        ty: None,
                        body: None,
                    });

                    match local_scope_info.maybe_declare(diagnostics, &common_info.id, ScopedEntry::Item(item), vis) {
                        Ok(()) => {}
                        Err(e) => {
                            // this is is a non-fatal error
                            // TODO clean this up, use report_and_continue inside of .maybe_declare
                            let _: DiagnosticError = e;
                        },
                    }
                }

                Ok(FileAuxiliary { ast, local_scope })
            }
            Err(e) => {
                Err(diagnostics.report(database.map_parser_error_to_diagnostic(e)))
            },
        };

        file_auxiliary.insert_first(file, aux);
    }

    let mut state = CompileState {
        diag: diagnostics,
        database,
        items,
        file_auxiliary: &file_auxiliary,
        generic_type_params: Arena::default(),
        generic_value_params: Arena::default(),
        function_params: Arena::default(),
        module_ports: Arena::default(),
        module_info: IndexMap::new(),
        scopes,
        log_const_eval: false,
    };

    // resolve all item types (which is mostly their signatures)
    // TODO randomize order to check for dependency bugs? but then diagnostics have random orders
    let item_keys = state.items.keys().collect_vec();
    for &item in &item_keys {
        state.resolve_item_type_fully(item);
    }

    // typecheck all item bodies
    // TODO merge this with the previous pass: better for LSP and maybe for local items
    for &item in &item_keys {
        assert!(state[item].body.is_none());
        let body = match state.type_check_item_body(item) {
            Ok(body) => Ok(body),
            Err(ResolveError::Diagnostic(e)) => Err(e),
            // TODO replace panics with internal compiler errors
            Err(ResolveError::ResolveFirst(_)) => panic!("all types should be resolved by now"),
        };

        let slot = &mut state[item].body;
        assert!(slot.is_none());
        *slot = Some(body);
    }

    let items = state.items.map_values(|info| ItemInfo {
        file: info.file,
        file_item_index: info.file_item_index,
        ty: info.ty.unwrap(),
        body: info.body.unwrap(),
    });

    CompiledDatabase {
        items,
        generic_type_params: state.generic_type_params,
        generic_value_params: state.generic_value_params,
        function_params: state.function_params,
        module_ports: state.module_ports,
        scopes: state.scopes,
        file_auxiliary,
    }
}

// TODO create some dedicated auxiliary data structure, with dense and non-dense variants
struct CompileState<'d, 'a> {
    diag: &'d Diagnostics,
    
    database: &'d SourceDatabase,
    file_auxiliary: &'a IndexMap<FileId, DiagnosticResult<FileAuxiliary>>,
    items: Arena<Item, ItemInfoPartial>,

    // TODO reduce duplication with CompileDatabase (including all index accessors through a trait)
    generic_type_params: Arena<GenericTypeParameter, GenericTypeParameterInfo>,
    generic_value_params: Arena<GenericValueParameter, GenericValueParameterInfo>,
    function_params: Arena<FunctionParameter, FunctionParameterInfo>,
    module_ports: Arena<ModulePort, ModulePortInfo>,
    module_info: IndexMap<Item, ModuleInfo>,
    scopes: Scopes<ScopedEntry>,

    log_const_eval: bool,
}

struct ModuleInfo {
    scope_ports: Scope,
    // TODO scope_inner?
}

#[derive(Debug)]
struct ItemInfoPartial {
    file: FileId,
    file_item_index: usize,

    // `None` if this item has not been resolved yet.
    // For a type: the type
    // For a value: the signature
    ty: Option<DiagnosticResult<MaybeConstructor<Type>>>,
    // `None` if this items body has not been checked yet.
    body: Option<DiagnosticResult<ItemBody>>,
}

#[derive(Debug, Copy, Clone)]
pub struct ResolveFirst(Item);

#[derive(Debug)]
pub enum ResolveError {
    Diagnostic(DiagnosticError),
    ResolveFirst(ResolveFirst),
}

pub type ResolveResult<T> = Result<T, ResolveError>;

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
            if self[curr].ty.is_some() {
                // already resolved, skip
                continue;
            }

            let resolved = match self.resolve_item_type_new(curr) {
                Ok(resolved) => Ok(resolved),
                Err(ResolveError::Diagnostic(e)) => Err(e),
                Err(ResolveError::ResolveFirst(ResolveFirst(first))) => {
                    assert!(self[first].ty.is_none(), "request to resolve {first:?} first, but it already has a type");

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
                            let item_ast = self.get_item_ast(stack_item).expect("cycle xor diagnostic");
                            diag = diag.add_error(item_ast.common_info().span_short, "part of cycle");
                        }
                        let err = self.diag.report(diag.finish());

                        // set slot of all involved items
                        for &stack_item in cycle {
                            let slot = &mut self[stack_item].ty;
                            assert!(slot.is_none(), "someone else already set the type for {curr:?}");
                            *slot = Some(Err(err));
                        }

                        // remove cycle from the stack
                        drop(stack.drain(cycle_start_index..));
                        continue
                    } else {
                        // no cycle, visit the next item
                        stack.push(first);
                        continue
                    }
                }
            };

            // managed to resolve the current item, store its type
            let slot = &mut self[curr].ty;
            assert!(slot.is_none(), "someone else already set the type for {curr:?}");
            *slot = Some(resolved);
        }
    }

    fn resolve_item_type(&self, item: Item) -> ResolveResult<&MaybeConstructor<Type>> {
        match self[item].ty {
            Some(Ok(ref r)) => Ok(r),
            Some(Err(e)) => Err(ResolveError::Diagnostic(e)),
            None => Err(ResolveError::ResolveFirst(ResolveFirst(item))),
        }
    }

    fn resolve_item_type_new(&mut self, item: Item) -> ResolveResult<MaybeConstructor<Type>> {
        // check that this is indeed a new query
        assert!(self[item].ty.is_none());

        // item lookup
        let item_ast = self.get_item_ast(item)?;
        let scope = self.file_auxiliary.get(&self[item].file).unwrap().as_ref()?.local_scope;

        // actual resolution
        match *item_ast {
            // use indirection
            ast::Item::Use(ItemUse { span: _, ref path, as_: _ }) => {
                // TODO why are we handling use items here? can they not be eliminated by scope building
                //  this is really weird, use items don't even really have signatures
                let next_item = self.resolve_use_path(path)?;
                Ok(self.resolve_item_type(next_item)?.clone())
            }
            // type definitions
            ast::Item::Type(ItemDefType { span: _, vis: _, id: _, ref params, ref inner }) => {
                self.resolve_new_generic_type_def(item, scope, params, |s, _args, scope_inner| {
                    Ok(s.eval_expression_as_ty(scope_inner, inner)?)
                })
            }
            ast::Item::Struct(ItemDefStruct { span, vis: _, id: _, ref params, ref fields }) => {
                self.resolve_new_generic_type_def(item, scope, params, |s, args, scope_inner| {
                    // map fields
                    let mut fields_map = IndexMap::new();
                    for field in fields {
                        let StructField { span: _, id: field_id, ty } = field;
                        let field_ty = s.eval_expression_as_ty(scope_inner, ty)?;

                        let prev = fields_map.insert(field_id.string.clone(), (field_id, field_ty));
                        if let Some(prev) = prev {
                            let diag = Diagnostic::new_defined_twice("struct field", span, field_id, prev.0);
                            throw!(s.diag.report(diag))
                        }
                    }

                    // result
                    let ty = StructTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        fields: fields_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Type::Struct(ty))
                })
            }
            ast::Item::Enum(ItemDefEnum { span, vis: _, id: _, ref params, ref variants }) => {
                self.resolve_new_generic_type_def(item, scope, params, |s, args, scope_inner| {
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
                            throw!(s.diag.report(diag))
                        }
                    }

                    // result
                    let ty = EnumTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        variants: variants_map.into_iter().map(|(k, v)| (k, v.1)).collect(),
                    };
                    Ok(Type::Enum(ty))
                })
            },
            // value definitions
            ast::Item::Module(ItemDefModule { span: _, vis: _, id: _, ref params, ref ports, ref body }) => {
                self.resolve_new_generic_type_def(item, scope, params, |s, args, scope_inner| {
                    // yet another sub-scope for the ports that refer to each other
                    let scope_ports = s.scopes.new_child(scope_inner, ports.span.join(body.span), Visibility::Private);

                    // map ports
                    let mut port_name_map = IndexMap::new();
                    let mut port_vec = vec![];

                    for port in &ports.inner {
                        let ast::ModulePort { span: _, id: port_id, direction, kind, } = port;

                        if let Some(prev) = port_name_map.insert(port_id.string.clone(), port_id) {
                            let diag = Diagnostic::new_defined_twice("module port", ports.span, &port_id, prev);
                            throw!(s.diag.report(diag))
                        }

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
                                            SyncKind::Sync(clk) => {
                                                let clk = s.eval_expression_as_value(scope_ports, clk)?;
                                                SyncKind::Sync(clk)
                                            },
                                        },
                                        ty: s.eval_expression_as_ty(scope_ports, ty)?,
                                    }
                                }
                            },
                        };
                        let module_port = s.module_ports.push(module_port_info);
                        port_vec.push(module_port);

                        s.scopes[scope_ports].declare(
                            s.diag,
                            &port_id,
                            ScopedEntry::Direct(MaybeConstructor::Immediate(TypeOrValue::Value(Value::ModulePort(module_port)))),
                            Visibility::Private,
                        )?;
                    }

                    // result
                    let module_info = ModuleInfo { scope_ports };
                    s.module_info.insert_first(item, module_info);

                    let module_ty_info = ModuleTypeInfo {
                        nominal_type_unique: NominalTypeUnique { item, args },
                        ports: port_vec,
                    };

                    Ok(Type::Module(module_ty_info))
                })
            },
            ast::Item::Const(_) =>
                throw!(self.diag.report_todo(item_ast.common_info().span_short, "const definition")),
            ast::Item::Function(_) =>
                throw!(self.diag.report_todo(item_ast.common_info().span_short, "function definition")),
            ast::Item::Interface(_) =>
                throw!(self.diag.report_todo(item_ast.common_info().span_short, "interface definition")),
        }
    }

    fn resolve_new_generic_type_def<T>(
        &mut self,
        item: Item,
        scope_outer: Scope,
        params: &Option<Spanned<Vec<ast::GenericParameter>>>,
        build_ty: impl FnOnce(&mut Self, GenericArguments, Scope) -> ResolveResult<T>,
    ) -> ResolveResult<MaybeConstructor<T>> {
        let item_span = self.get_item_ast(item)?.common_info().span_full;
        let scope_inner = self.scopes.new_child(scope_outer, item_span, Visibility::Private);

        match params {
            None => {
                // there are no parameters, just map directly
                // the scope still needs to be "nested" since the builder needs an owned scope
                let arguments = GenericArguments { vec: vec![] };
                Ok(MaybeConstructor::Immediate(build_ty(self, arguments, scope_inner)?))
            }
            Some(params) => {
                // build inner scope
                let mut unique: HashMap<&str, &Identifier> = Default::default();
                let mut parameters = vec![];
                let mut arguments = vec![];

                for param_ast in &params.inner {
                    // check parameter names for uniqueness
                    if let Some(prev) = unique.insert(&param_ast.id.string, &param_ast.id) {
                        let diag = Diagnostic::new_defined_twice("generic parameter", params.span, prev, &param_ast.id);
                        throw!(self.diag.report(diag));
                    }

                    let (param, arg) = match &param_ast.kind {
                        &GenericParameterKind::Type(_span) => {
                            let param = self.generic_type_params.push(GenericTypeParameterInfo {
                                defining_item: item,
                                defining_id: param_ast.id.clone(),
                            });
                            (GenericParameter::Type(param), TypeOrValue::Type(Type::GenericParameter(param)))
                        },
                        GenericParameterKind::Value(ty_expr) => {
                            let ty = self.eval_expression_as_ty(scope_inner, ty_expr)?;
                            let param = self.generic_value_params.push(GenericValueParameterInfo {
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
                    self.scopes[scope_inner].declare(self.diag, &param_ast.id, entry, Visibility::Private)?;
                }

                // map inner to actual type
                let parameters = GenericParameters { vec: parameters };
                let arguments = GenericArguments { vec: arguments };
                let ty_constr = Constructor {
                    parameters,
                    inner: build_ty(self, arguments, scope_inner)?,
                };
                Ok(MaybeConstructor::Constructor(ty_constr))
            }
        }
    }

    fn type_check_item_body(&mut self, item: Item) -> ResolveResult<ItemBody> {
        assert!(self[item].body.is_none());

        let _item_ty = self[item].ty.as_ref().expect("item should already have been checked");

        let item_ast = self.get_item_ast(item)?;
        let item_span = item_ast.common_info().span_short;

        let body = match item_ast {
            // these items are fully defined by their type, which was already checked earlier
            ast::Item::Use(_) | ast::Item::Type(_) | ast::Item::Struct(_) | ast::Item::Enum(_) => ItemBody::None,
            ast::Item::Const(_) =>
                throw!(self.diag.report_todo(item_span, "const body")),
            ast::Item::Function(_) =>
                throw!(self.diag.report_todo(item_span, "function body")),
            ast::Item::Module(item_ast) => {
                let ItemDefModule { span: _, vis: _, id: _, params: _, ports: _, body } = item_ast;
                let ast::Block { span: _, statements } = body;

                // TODO add to this scope any locally-defined items first
                //   this might require spawning a new sub-driver instance,
                //     or can we avoid that and do everything in a single graph?
                //   do we _want_ to avoid splits? that's good for concurrency!
                let scope_body = self.scopes.new_child(self.module_info[&item].scope_ports, body.span, Visibility::Private);

                let mut module_blocks = vec![];

                for top_statement in statements {
                    match top_statement.kind {
                        StatementKind::Declaration(_) =>
                            throw!(self.diag.report_todo(top_statement.span, "module top-level declaration")),
                        StatementKind::Assignment(_) =>
                            throw!(self.diag.report_todo(top_statement.span, "module top-level assignment")),
                        StatementKind::Expression(_) =>
                            throw!(self.diag.report_todo(top_statement.span, "module top-level expression")),
                        StatementKind::CombinatorialBlock(ref comb_block) => {
                            let mut result_statements = vec![];

                            for statement in &comb_block.block.statements {
                                match statement.kind {
                                    StatementKind::Declaration(_) =>
                                        throw!(self.diag.report_todo(statement.span, "combinatorial declaration")),
                                    StatementKind::Assignment(ref assignment) => {
                                        let &ast::Assignment { span: _, op, ref target, ref value } = assignment;
                                        if op.inner.is_some() {
                                            throw!(self.diag.report_todo(statement.span, "combinatorial assignment with operator"))
                                        }

                                        // TODO type and sync checking

                                        let target = self.eval_expression_as_value(scope_body, target)?;
                                        let value = self.eval_expression_as_value(scope_body, value)?;

                                        if let (Value::ModulePort(target), Value::ModulePort(value)) = (target, value) {
                                            result_statements.push(CombinatorialStatement::PortPortAssignment(target, value));
                                        } else {
                                            throw!(self.diag.report_todo(statement.span, "general combinatorial assignment"))
                                        }
                                    },
                                    StatementKind::Expression(_) =>
                                        throw!(self.diag.report_todo(statement.span, "combinatorial expression")),

                                    StatementKind::CombinatorialBlock(ref comb_block_inner) => {
                                        let err = Diagnostic::new("nested combinatorial block")
                                            .add_info(comb_block.span_keyword, "outer")
                                            .add_error(comb_block_inner.span_keyword, "inner")
                                            .finish();
                                        throw!(self.diag.report(err))
                                    }

                                    StatementKind::ClockedBlock(ref clock_block_inner) => {
                                        let err = Diagnostic::new("nested clock block in combinatorial block")
                                            .add_info(comb_block.span_keyword, "outer")
                                            .add_error(clock_block_inner.span_keyword, "inner")
                                            .finish();
                                        throw!(self.diag.report(err))
                                    },
                                }
                            }

                            let comb_block = ModuleBlockCombinatorial {
                                statements: result_statements,
                            };
                            module_blocks.push(ModuleBlock::Combinatorial(comb_block));
                        }
                        StatementKind::ClockedBlock(_) =>
                            throw!(self.diag.report_todo(top_statement.span, "clocked block")),
                    }
                }

                ItemBody::Module(ModuleBody {
                    blocks: module_blocks,
                })
            },
            ast::Item::Interface(_) =>
                throw!(self.diag.report_todo(item_span, "interface body")),
        };
        Ok(body)
    }

    fn resolve_use_path(&self, path: &Path) -> ResolveResult<Item> {
        // TODO the current path design does not allow private sub-modules
        //   are they really necessary? if all inner items are private it's effectively equivalent
        //   -> no it's not equivalent, things can also be private from the parent

        // TODO allow private visibility in child and sibling paths
        let vis = Visibility::Public;
        let mut curr_dir = self.database.root_directory;

        let Path { span: _, steps, id } = path;

        for step in steps {
            let curr_dir_info = &self.database[curr_dir];
            curr_dir = curr_dir_info.children.get(&step.string).copied().ok_or_else(|| {
                let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                options.sort();

                let diag = Diagnostic::new("invalid path step")
                    .snippet(path.span)
                    .add_error(step.span, "invalid step")
                    .finish()
                    .footer(Level::Info, format!("possible options: {:?}", options))
                    .finish();
                self.diag.report(diag)
            })?;
        }

        let file = self.database[curr_dir].file.ok_or_else(|| {
            let diag = Diagnostic::new_simple("expected path to file", path.span, "no file exists at this path");
            self.diag.report(diag)
        })?;
        let file_scope = self[file].as_ref_ok()?.local_scope;

        // TODO change root scope to just be a map instead of a scope so we can avoid this unwrap
        match self.scopes[file_scope].find(&self.scopes, &self.database, self.diag, id, vis)?.value {
            &ScopedEntry::Item(item) => Ok(item),
            // TODO is this still true?
            ScopedEntry::Direct(_) => unreachable!("file root entries should not exist"),
        }
    }

    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&self, scope: Scope, expr: &Expression) -> ResolveResult<ScopedEntryDirect> {
        let result = match expr.inner {
            ExpressionKind::Dummy =>
                throw!(self.diag.report_todo(expr.span, "dummy expression")),
            ExpressionKind::Wrapped(ref inner) => self.eval_expression(scope, inner)?,
            ExpressionKind::Id(ref id) => {
                match self.scopes[scope].find(&self.scopes, &self.database, self.diag, id, Visibility::Private)?.value {
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
            },
            ExpressionKind::TypeFunc(_, _) =>
                throw!(self.diag.report_todo(expr.span, "type func expression")),
            ExpressionKind::Block(_) =>
                throw!(self.diag.report_todo(expr.span, "block expression")),
            ExpressionKind::If(_) =>
                throw!(self.diag.report_todo(expr.span, "if expression")),
            ExpressionKind::Loop(_) =>
                throw!(self.diag.report_todo(expr.span, "loop expression")),
            ExpressionKind::While(_) =>
                throw!(self.diag.report_todo(expr.span, "while expression")),
            ExpressionKind::For(_) =>
                throw!(self.diag.report_todo(expr.span, "for expression")),
            ExpressionKind::Sync(_) =>
                throw!(self.diag.report_todo(expr.span, "sync expression")),
            ExpressionKind::Return(_) =>
                throw!(self.diag.report_todo(expr.span, "return expression")),
            ExpressionKind::Break(_) =>
                throw!(self.diag.report_todo(expr.span, "break expression")),
            ExpressionKind::Continue =>
                throw!(self.diag.report_todo(expr.span, "continue expression")),
            ExpressionKind::IntPattern(ref pattern) => {
                let value = match pattern {
                    IntPattern::Hex(_) =>
                        throw!(self.diag.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        throw!(self.diag.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        str_clean.parse::<BigInt>().unwrap()
                    }
                };
                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Int(value)))
            }
            ExpressionKind::BoolLiteral(_) =>
                throw!(self.diag.report_todo(expr.span, "bool literal expression")),
            ExpressionKind::StringLiteral(_) =>
                throw!(self.diag.report_todo(expr.span, "string literal expression")),
            ExpressionKind::ArrayLiteral(_) =>
                throw!(self.diag.report_todo(expr.span, "array literal expression")),
            ExpressionKind::TupleLiteral(_) =>
                throw!(self.diag.report_todo(expr.span, "tuple literal expression")),
            ExpressionKind::StructLiteral(_) =>
                throw!(self.diag.report_todo(expr.span, "struct literal expression")),
            ExpressionKind::RangeLiteral(ref range) => {
                let &RangeLiteral { end_inclusive, ref start, ref end } = range;

                let map_point = |point: &Option<Box<Expression>>| -> ResolveResult<_> {
                    match point {
                        None => Ok(None),
                        Some(point) => Ok(Some(Box::new(self.eval_expression_as_value(scope, point)?))),
                    }
                };

                let start = map_point(start)?;
                let end = map_point(end)?;

                if let (Some(start), Some(end)) = (&start, &end) {
                    let op = if end_inclusive { BinaryOp::CmpLt } else { BinaryOp::CmpLte };
                    self.require_value_true_for_range(expr.span, &Value::Binary(op, start.clone(), end.clone()))?;
                }

                let value = Value::Range(RangeInfo::new(start, end, end_inclusive));
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
                    UnaryOp::Not =>
                        Value::UnaryNot(Box::new(self.eval_expression_as_value(scope, inner)?)),
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            },
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_as_value(scope, left)?;
                let right = self.eval_expression_as_value(scope, right)?;

                let result = Value::Binary(op, Box::new(left), Box::new(right));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            },
            ExpressionKind::TernarySelect(_, _, _) =>
                throw!(self.diag.report_todo(expr.span, "ternary select expression")),
            ExpressionKind::ArrayIndex(_, _) =>
                throw!(self.diag.report_todo(expr.span, "array index expression")),
            ExpressionKind::DotIdIndex(_, _) =>
                throw!(self.diag.report_todo(expr.span, "dot id index expression")),
            ExpressionKind::DotIntIndex(_, _) =>
                throw!(self.diag.report_todo(expr.span, "dot int index expression")),
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
                            let err = Diagnostic::new_simple(
                                format!("constructor argument count mismatch, expected {}, got {}", parameters.vec.len(), args.inner.len()),
                                args.span,
                                format!("expected {} arguments, got {}", parameters.vec.len(), args.inner.len()),
                            );
                            throw!(self.diag.report(err));
                        }

                        // check kind and type match, and collect in replacement map
                        let mut map_ty: IndexMap<GenericTypeParameter, Type> = IndexMap::new();
                        let mut map_value: IndexMap<GenericValueParameter, Value> = IndexMap::new();
                        for (&param, arg) in zip_eq(&parameters.vec, &args.inner) {
                            match param {
                                GenericParameter::Type(param) => {
                                    let arg_ty = self.eval_expression_as_ty(scope, arg)?;
                                    // TODO use for bound-check (once we add type bounds)
                                    let _param_info = &self[param];
                                    map_ty.insert_first(param, arg_ty)
                                }
                                GenericParameter::Value(param) => {
                                    let arg_value = self.eval_expression_as_value(scope, arg)?;
                                    let param_info = &self[param];
                                    self.check_type_contains(param_info.ty_span, arg.span, &param_info.ty, &arg_value)?;
                                    map_value.insert_first(param, arg_value)
                                }
                            }
                        }

                        // do the actual replacement
                        let result = inner.replace_generic_params(&map_ty, &map_value);
                        MaybeConstructor::Immediate(result)
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => {
                                let err = Diagnostic::new_simple("invalid call target", target.span, "invalid call target kind 'type'");
                                throw!(self.diag.report(err));
                            }
                            TypeOrValue::Value(_) =>
                                throw!(self.diag.report_todo(target.span, "value call target")),
                        }
                    }
                }
            },
        };
        Ok(result)
    }

    fn eval_expression_as_ty(&self, scope: Scope, expr: &Expression) -> ResolveResult<Type> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            // TODO unify these error strings somewhere
            // TODO maybe move back to central error collection place for easier unit testing?
            ScopedEntryDirect::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type, got constructor", expr.span, "constructor");
                throw!(self.diag.report(diag))
            },
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(ty) => Ok(ty),
                TypeOrValue::Value(_) => {
                    let diag = Diagnostic::new_simple("expected type, got value", expr.span, "value");
                    throw!(self.diag.report(diag));
                },
            }
        }
    }

    fn eval_expression_as_value(&self, scope: Scope, expr: &Expression) -> ResolveResult<Value> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => {
                let err = Diagnostic::new_simple("expected value, got constructor", expr.span, "constructor");
                throw!(self.diag.report(err));
            },
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(_) => {
                    let err = Diagnostic::new_simple("expected value, got type", expr.span, "type");
                    throw!(self.diag.report(err));
                },
                TypeOrValue::Value(value) => Ok(value),
            }
        }
    }

    fn eval_builtin_call(&self, scope: Scope, expr_span: Span, name: &str, args: &Args) -> ResolveResult<TypeOrValue> {
        // TODO disallow calling builtin outside of stdlib?
        match name {
            "type" => {
                let first_arg = args.inner.get(0).map(|e| &e.inner);
                if let Some(ExpressionKind::StringLiteral(ty)) = first_arg {
                    match ty.as_str() {
                        "bool" if args.inner.len() == 1 =>
                            return Ok(TypeOrValue::Type(Type::Boolean)),
                        "int" if args.inner.len() == 1 => {
                            let range = Box::new(Value::Range(RangeInfo::unbounded()));
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
                        "bits_inf" if args.inner.len() == 1 => {
                            return Ok(TypeOrValue::Type(Type::Bits(None)));
                        }
                        "bits" if args.inner.len() == 2 => {
                            // TODO typecheck (bits must be non-negative integer)
                            let bits = self.eval_expression_as_value(scope, &args.inner[1])?;
                            return Ok(TypeOrValue::Type(Type::Bits(Some(Box::new(bits)))));
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

        let err = Diagnostic::new("invalid arguments for __builtin call")
            .snippet(expr_span)
            .add_error(args.span, "invalid arguments")
            .finish()
            .finish()
            .into();
        throw!(self.diag.report(err));
    }

    fn check_type_contains(&self, span_ty: Span, span_value: Span, ty: &Type, value: &Value) -> ResolveResult<()> {
        match (ty, value) {
            (Type::Range, Value::Range(_)) => return Ok(()),
            (Type::Integer(IntegerTypeInfo { range }), value) => {
                if let Value::Range(range) = range.as_ref() {
                    let &RangeInfo { ref start, ref end, end_inclusive } = range;
                    if let Some(start) = start {
                        let cond = Value::Binary(BinaryOp::CmpLte, start.clone(), Box::new(value.clone()));
                        self.require_value_true_for_type_check(span_ty, span_value, &cond)?;
                    }
                    if let Some(end) = end {
                        let cmp_op = if end_inclusive { BinaryOp::CmpLte } else { BinaryOp::CmpLt };
                        let cond = Value::Binary(cmp_op, Box::new(value.clone()), end.clone());
                        self.require_value_true_for_type_check(span_ty, span_value, &cond)?;
                    }
                    return Ok(())
                }
            }
            _ => {},
        }

        let feature = &format!("type-check {:?} contains {:?}", ty, value);
        throw!(self.diag.report_todo(span_value, feature))
    }

    fn require_value_true_for_range(&self, span_range: Span, value: &Value) -> ResolveResult<()> {
        self.try_eval_bool_true(span_range, value)?.map_err(|e| {
            let err = Diagnostic::new(format!("range valid check failed: value {} {}", self.value_to_readable_str(value), e.to_message()))
                .add_error(span_range, "when checking that this range is non-decreasing")
                .finish();
            self.diag.report(err).into()
        })
    }

    fn require_value_true_for_type_check(&self, span_ty: Span, span_value: Span, value: &Value) -> ResolveResult<()> {
        self.try_eval_bool_true(span_value, value)?.map_err(|e| {
            let err = Diagnostic::new(format!("type check failed: value {} {}", self.value_to_readable_str(value), e.to_message()))
                .add_error(span_value, "when type checking this value")
                .add_info(span_ty, "against this type")
                .finish();
            self.diag.report(err).into()
        })
    }

    fn try_eval_bool_true(&self, origin: Span, value: &Value) -> DiagnosticResult<Result<(), EvalTrueError>> {
        match self.try_eval_bool(origin, value)? {
            Some(true) => Ok(Ok(())),
            Some(false) => Ok(Err(EvalTrueError::False)),
            None => Ok(Err(EvalTrueError::Unknown)),
        }
    }

    // TODO what is the general algorithm for this? equivalence graphs?
    // TODO add boolean-proving cache
    // TODO it it possible to keep boolean proving and type inference separate? it probably is
    //   if we don't allow user-defined type selections
    //   even then, we can probably brute-force our way through those relatively easily
    // TODO convert lte/gte into +1/-1 fixes instead?
    // TODO convert inclusive/exclusive into +1/-1 fixes instead?
    // TODO check lt, lte, gt, gte, ... all together elegantly
    // TODO return true for vacuous truths, eg. comparisons between empty ranges?
    fn try_eval_bool(&self, origin: Span, value: &Value) -> DiagnosticResult<Option<bool>> {
        let result = self.try_eval_bool_inner(origin, value);
        if self.log_const_eval {
            eprintln!(
                "try_eval_bool({}) -> {:?}",
                self.value_to_readable_str(value),
                result
            );
        }
        result
    }

    fn try_eval_bool_inner(&self, origin: Span, value: &Value) -> DiagnosticResult<Option<bool>> {
        // TODO this is wrong, we should be returning None a lot more, eg. if the ranges of the operands are not tight
        match *value {
            Value::Binary(binary_op, ref left, ref right) => {
                let left = try_opt_result!(self.range_of_value(origin, left)?);
                let right = try_opt_result!(self.range_of_value(origin, right)?);

                let compare_lt = |allow_eq: bool| {
                    let end_delta = if left.end_inclusive { 0 } else { 1 };
                    let left_end = value_as_int(left.end.as_ref()?)? - end_delta;
                    let right_start = value_as_int(right.start.as_ref()?)?;

                    if allow_eq {
                        Some(&left_end <= right_start)
                    } else {
                        Some(&left_end < right_start)
                    }
                };

                match binary_op {
                    BinaryOp::CmpLt => return Ok(compare_lt(false)),
                    BinaryOp::CmpLte => return Ok(compare_lt(true)),
                    _ => {},
                }
            }
            // TODO support more values
            _ => {},
        }
        Ok(None)
    }

    // TODO This needs to be split up into a tight and loose range:
    //   we _know_ all values in the tight range are reachable,
    //   and we _know_ no values outside of the loose range are
    fn range_of_value(&self, origin: Span, value: &Value) -> DiagnosticResult<Option<RangeInfo<Box<Value>>>> {
        let result = self.range_of_value_inner(origin, value)?;
        let result_simplified = result.clone().map(|r| {
            RangeInfo {
                start: r.start.map(|v| Box::new(simplify_value(*v))),
                end: r.end.map(|v| Box::new(simplify_value(*v))),
                end_inclusive: r.end_inclusive,
            }
        });
        if self.log_const_eval {
            eprintln!(
                "range_of_value({}) -> {:?} -> {:?}",
                self.value_to_readable_str(value),
                result.as_ref().map_or("None".to_string(), |r| self.range_to_readable_str(&r)),
                result_simplified.as_ref().map_or("None".to_string(), |r| self.range_to_readable_str(&r)),
            );
        }
        Ok(result_simplified)
    }

    fn range_of_value_inner(&self, origin: Span, value: &Value) -> DiagnosticResult<Option<RangeInfo<Box<Value>>>> {
        // TODO if range ends are themselves params with ranges, assume the worst case
        //   although that misses things like (n < n+1)
        fn ty_as_range(ty: &Type) -> DiagnosticResult<Option<RangeInfo<Box<Value>>>> {
            if let Type::Integer(IntegerTypeInfo { range }) = ty {
                if let Value::Range(range) = range.as_ref() {
                    return Ok(Some(range.clone()));
                }
            }
            Ok(None)
        }

        match *value {
            // params have types which we can use to extract a range
            Value::GenericParameter(param) => ty_as_range(&self[param].ty),
            Value::FunctionParameter(param) => ty_as_range(&self[param].ty),
            // a single integer corresponds to the range containing only that integer
            // TODO should we generate an inclusive or exclusive range here?
            //   this will become moot once we switch to +1 deltas
            Value::Int(ref value) => Ok(Some(RangeInfo {
                start: Some(Box::new(Value::Int(value.clone()))),
                end: Some(Box::new(Value::Int(value + 1u32))),
                end_inclusive: false,
            })),
            Value::ModulePort(_) => Ok(None),
            Value::Binary(op, ref left, ref right) => {
                let left = try_opt_result!(self.range_of_value(origin, left)?);
                let right = try_opt_result!(self.range_of_value(origin, right)?);

                // TODO replace end_inclusive with a +1 delta to get rid of this trickiness forever
                if left.end_inclusive || right.end_inclusive {
                    return Ok(None);
                }

                match op {
                    BinaryOp::Add => {
                        let range = RangeInfo {
                            start: option_pair(left.start.as_ref(), right.start.as_ref())
                                .map(|(left_start, right_start)|
                                    Box::new(Value::Binary(BinaryOp::Add, left_start.clone(), right_start.clone()))
                                ),
                            end: option_pair(left.end.as_ref(), right.end.as_ref())
                                .map(|(left_end, right_end)|
                                    Box::new(Value::Binary(BinaryOp::Add, left_end.clone(), right_end.clone()))
                                ),
                            end_inclusive: false,
                        };
                        Ok(Some(range))
                    }
                    BinaryOp::Sub => {
                        let range = RangeInfo {
                            start: option_pair(left.start.as_ref(), right.end.as_ref())
                                .map(|(left_start, right_end)| {
                                    let right_end_inclusive = Box::new(Value::Binary(BinaryOp::Sub, right_end.clone(), Box::new(Value::Int(BigInt::one()))));
                                    Box::new(Value::Binary(BinaryOp::Sub, left_start.clone(), right_end_inclusive))
                                }),
                            end: option_pair(left.end.as_ref(), right.start.as_ref())
                                .map(|(left_end, right_start)|
                                    Box::new(Value::Binary(BinaryOp::Sub, left_end.clone(), right_start.clone()))
                                ),
                            end_inclusive: false,
                        };
                        Ok(Some(range))
                    }
                    BinaryOp::Pow => {
                        // check that exponent is non-negative
                        let right_start = right.start.as_ref().ok_or_else(|| {
                            let err = Diagnostic::new(format!("power exponent cannot be negative, got range without lower bound {:?}", self.range_to_readable_str(&right)))
                                .add_error(origin, "while checking this expression")
                                .finish();
                            self.diag.report(err)
                        })?;
                        let cond = Value::Binary(BinaryOp::CmpLte, Box::new(Value::Int(BigInt::ZERO)), right_start.clone());
                        self.try_eval_bool_true(origin, &cond)?
                            .map_err(|e| {
                                let err = Diagnostic::new(format!("power exponent range check failed: value {} {}", self.value_to_readable_str(&cond), e.to_message()))
                                    .add_error(origin, "while checking this expression")
                                    .finish();
                                self.diag.report(err)
                            })?;

                        let left_start = try_opt_result!(left.start);

                        // if base is >0, then the result is >0 too
                        // TODO this range can be improved a lot: consider cases +,0,- separately
                        let base_positive = self.try_eval_bool(origin, &Value::Binary(BinaryOp::CmpLt, Box::new(Value::Int(BigInt::ZERO)), left_start))?;
                        if base_positive == Some(true) {
                            Ok(Some(RangeInfo {
                                start: Some(Box::new(Value::Int(BigInt::one()))),
                                end: None,
                                end_inclusive: false,
                            }))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(None),
                }
            },
            Value::UnaryNot(_) => Ok(None),
            Value::Range(_) => panic!("range can't itself have a range type"),
            Value::Function(_) => panic!("function can't have a range type"),
            Value::Module(_) => panic!("module can't have a range type"),
        }
    }

    // TODO make sure generic variables are properly disambiguated
    fn range_to_readable_str(&self, range: &RangeInfo<Box<Value>>) -> String {
        let RangeInfo { ref start, ref end, end_inclusive } = *range;

        let start = start.as_ref().map_or(String::new(), |v| self.value_to_readable_str(v));
        let end = end.as_ref().map_or(String::new(), |v| self.value_to_readable_str(v));
        let symbol = if end_inclusive { "..=" } else { ".." };

        format!("({}{}{})", start, symbol, end)
    }

    // TODO integrate generic parameters properly in the diagnostic, by pointing to them
    fn value_to_readable_str(&self, value: &Value) -> String {
        match *value {
            // TODO include span to disambiguate
            Value::GenericParameter(p) => {
                let id = &self[p].defining_id;
                format!("generic_param({:?}, {:?})", id.string, self.database.expand_pos(id.span.start))
            }
            Value::FunctionParameter(p) => {
                let id = &self[p].defining_id;
                format!("function_param({:?}, {:?})", id.string(), self.database.expand_pos(id.span().start))
            }
            Value::ModulePort(p) => {
                let id = &self[p].defining_id;
                format!("module_port({:?}, {:?})", id.string, self.database.expand_pos(id.span.start))
            }
            Value::Int(ref v) => {
                if v.is_negative() {
                    format!("({})", v)
                } else {
                    format!("{}", v)
                }
            },
            Value::Range(ref range) => self.range_to_readable_str(range),
            Value::Binary(op, ref left, ref right) => {
                let left = self.value_to_readable_str(left);
                let right = self.value_to_readable_str(right);
                let symbol = op.symbol();
                format!("({} {} {})", left, symbol, right)
            }
            Value::UnaryNot(ref inner) => {
                let inner = self.value_to_readable_str(inner);
                format!("(!{})", inner)
            }
            Value::Function(_) => todo!(),
            Value::Module(_) => todo!(),
        }
    }

    fn get_item_ast(&self, item: Item) -> DiagnosticResult<&'a ast::Item> {
        let info = &self[item];
        self.file_auxiliary.get(&info.file).unwrap()
            .as_ref_ok()
            .map(|aux| &aux.ast.items[info.file_item_index])
    }
}

pub fn simplify_value(value: Value) -> Value {
    match value {
        Value::Binary(op, left, right) => {
            let left = simplify_value(*left);
            let right = simplify_value(*right);

            if let Some((left, right)) = option_pair(value_as_int(&left), value_as_int(&right)) {
                match op {
                    BinaryOp::Add => return Value::Int(left + right),
                    BinaryOp::Sub => return Value::Int(left - right),
                    BinaryOp::Mul => return Value::Int(left * right),
                    _ => {}
                }
            }

            Value::Binary(op, Box::new(left), Box::new(right))
        }
        // TODO at least recursively call simplify
        value => value,
    }
}

pub fn value_as_int(value: &Value) -> Option<&BigInt> {
    match value {
        Value::Int(value) => Some(value),
        _ => None,
    }
}

fn option_pair<A, B>(left: Option<A>, right: Option<B>) -> Option<(A, B)> {
    match (left, right) {
        (Some(left), Some(right)) => Some((left, right)),
        _ => None,
    }
}

impl From<DiagnosticError> for ResolveError {
    fn from(value: DiagnosticError) -> Self {
        ResolveError::Diagnostic(value)
    }
}

impl From<&DiagnosticError> for ResolveError {
    fn from(value: &DiagnosticError) -> Self {
        ResolveError::Diagnostic(*value)
    }
}

impl From<ResolveFirst> for ResolveError {
    fn from(value: ResolveFirst) -> Self {
        ResolveError::ResolveFirst(value)
    }
}

impl std::ops::Index<FileId> for CompileState<'_, '_> {
    type Output = DiagnosticResult<FileAuxiliary>;
    fn index(&self, index: FileId) -> &Self::Output {
        self.file_auxiliary.get(&index).unwrap()
    }
}

impl std::ops::Index<Item> for CompileState<'_, '_> {
    type Output = ItemInfoPartial;
    fn index(&self, index: Item) -> &Self::Output {
        &self.items[index]
    }
}

impl std::ops::IndexMut<Item> for CompileState<'_, '_> {
    fn index_mut(&mut self, index: Item) -> &mut Self::Output {
        &mut self.items[index]
    }
}

impl std::ops::Index<GenericTypeParameter> for CompileState<'_, '_> {
    type Output = GenericTypeParameterInfo;
    fn index(&self, index: GenericTypeParameter) -> &Self::Output {
        &self.generic_type_params[index]
    }
}

impl std::ops::Index<GenericValueParameter> for CompileState<'_, '_> {
    type Output = GenericValueParameterInfo;
    fn index(&self, index: GenericValueParameter) -> &Self::Output {
        &self.generic_value_params[index]
    }
}

impl std::ops::Index<FunctionParameter> for CompileState<'_, '_> {
    type Output = FunctionParameterInfo;
    fn index(&self, index: FunctionParameter) -> &Self::Output {
        &self.function_params[index]
    }
}

impl std::ops::Index<ModulePort> for CompileState<'_, '_> {
    type Output = ModulePortInfo;
    fn index(&self, index: ModulePort) -> &Self::Output {
        &self.module_ports[index]
    }
}
