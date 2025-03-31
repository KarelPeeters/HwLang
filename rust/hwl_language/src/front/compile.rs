use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use crate::constants::THREAD_STACK_SIZE;
use crate::front::block::TypedIrExpression;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrDatabase, IrExpression, IrModule, IrModuleInfo, IrModules, IrPort, IrRegister, IrWire};
use crate::front::misc::{DomainSignal, Polarized, PortDomain, ScopedEntry, Signal, ValueDomain};
use crate::front::module::{ElaboratedModule, ElaboratedModuleHeader, ModuleElaborationCacheKey};
use crate::front::scope::{Scope, ScopeFile, ScopeInfo, Scopes, Visibility};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast;
use crate::syntax::ast::{Args, DomainKind, Identifier, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::syntax::parsed::{AstRefItem, AstRefModule, ParsedDatabase};
use crate::syntax::pos::{FileId, Span};
use crate::syntax::source::SourceDatabase;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::sync::{ComputeOnce, SharedQueue};
use crate::util::ResultExt;
use crate::{new_index_type, throw};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};

// TODO add test that randomizes order of files and items to check for dependency bugs,
//   assert that result and diagnostics are the same
// TODO extend the set of "type-checking" root points:
//   * project settings: multiple top modules
//   * type-checking-only generic instantiations of modules
//   * type-check all modules without generics automatically
//   * type-check modules with generics partially
pub fn compile(
    diags: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    print_handler: &mut (dyn PrintHandler + Sync),
    thread_count: NonZeroUsize,
) -> Result<IrDatabase, ErrorGuaranteed> {
    let PopulatedFileScopes {
        file_scopes,
        file_to_scopes,
        all_items_except_imports,
    } = populate_file_scopes(diags, source, parsed);

    // TODO randomize order to avoid all threads working on similar items and running into each other?
    let thread_count = thread_count.get();
    let work_queue = SharedQueue::new(thread_count);
    work_queue.push_batch(all_items_except_imports.iter().copied().map(WorkItem::EvaluateItem));

    let fixed = CompileFixed { source, parsed };
    let shared = CompileShared {
        file_scopes,
        file_to_scopes,
        work_queue,
        cache_item_values: all_items_except_imports
            .iter()
            .map(|&item| (item, ComputeOnce::new()))
            .collect(),
        cache_modules: Mutex::new(CompileSharedModules {
            elaborations: IndexMap::new(),
            ir_modules: Arena::default(),
        }),
        print_handler,
    };

    // get the top module
    let top_module_and_item = find_top_module(diags, fixed, &shared).and_then(|top_item| {
        let mut tmp_state = CompileState::new(fixed, &shared, diags);
        let elaborated: ElaboratedModule = tmp_state.elaborate_module(top_item, None)?;
        tmp_state.finish();
        Ok((elaborated.ir_module, top_item))
    });

    // run until everything is elaborated
    if thread_count > 1 {
        // TODO manage thread pool externally, and re-use it between parsing and elaboration
        std::thread::scope(|s| {
            let mut handles = vec![];
            for thread_index in 0..thread_count {
                let shared = &shared;
                let f = move || {
                    let thread_diags = Diagnostics::new();
                    run_elaboration_loop(&thread_diags, fixed, shared);
                    thread_diags
                };

                let name = format!("compile_{thread_index}");
                let h = std::thread::Builder::new()
                    .name(name)
                    .stack_size(THREAD_STACK_SIZE)
                    .spawn_scoped(s, f)
                    .unwrap();

                handles.push(h);
            }

            // TODO sort diags
            for h in handles {
                let thread_diags = h.join().unwrap();
                for d in thread_diags.finish() {
                    diags.report(d);
                }
            }
        });
    } else {
        run_elaboration_loop(diags, fixed, &shared);
    }

    // return result (at this point all modules should have been fully elaborated)
    let (top_module, top_item) = top_module_and_item?;
    let modules = shared.finish_ir_modules(diags, parsed[top_item].span)?;

    let db = IrDatabase { top_module, modules };
    db.validate(diags)?;
    Ok(db)
}

fn run_elaboration_loop(diags: &Diagnostics, fixed: CompileFixed, shared: &CompileShared) {
    while let Some(work_item) = shared.work_queue.pop() {
        let mut state = CompileState::new(fixed, shared, diags);

        match work_item {
            WorkItem::EvaluateItem(item) => {
                let slot = shared.cache_item_values.get(&item).unwrap();
                let result = state.with_recursion(ElaborationStackEntry::Item(item), |s| {
                    slot.offer_to_compute(|| s.eval_item_new(item));
                });

                // this is the first stack entry so it can't be a cycle yet
                result.unwrap();
            }
            WorkItem::ElaborateModule(ir_module, header) => {
                // TODO this lock is a bottleneck, it prevents elaborating multiple modules at once, which was the whole point!
                let cache = shared.cache_modules.lock().unwrap();
                let slot = &cache.ir_modules[ir_module];
                slot.offer_to_compute(|| state.elaborate_module_body_new(header));
            }
        }

        state.finish();
    }
}

/// globally shared, constant state
#[derive(Copy, Clone)]
pub struct CompileFixed<'a> {
    pub source: &'a SourceDatabase,
    pub parsed: &'a ParsedDatabase,
}

pub enum WorkItem {
    EvaluateItem(AstRefItem),
    ElaborateModule(IrModule, ElaboratedModuleHeader),
}

/// long-term shared between threads
pub struct CompileShared<'a> {
    pub file_scopes: Arena<ScopeFile, ScopeInfo>,
    pub file_to_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,

    pub work_queue: SharedQueue<WorkItem>,

    pub cache_item_values: IndexMap<AstRefItem, ComputeOnce<Result<CompileValue, ErrorGuaranteed>>>,
    // TODO turn this mutex into an RWLock and add a ComputeOnce wrapper to the values
    // TODO combine elaborations with ir_modules?
    pub cache_modules: Mutex<CompileSharedModules>,

    // TODO is there a reasonable way to get deterministic prints?
    pub print_handler: &'a (dyn PrintHandler + Sync),
}

pub struct CompileSharedModules {
    elaborations: IndexMap<ModuleElaborationCacheKey, Result<ElaboratedModule, ErrorGuaranteed>>,
    ir_modules: Arena<IrModule, ComputeOnce<Result<IrModuleInfo, ErrorGuaranteed>>>,
}

// short-term state for a single item evaluation or module elaboration
pub struct CompileState<'a> {
    pub fixed: CompileFixed<'a>,
    pub shared: &'a CompileShared<'a>,
    pub diags: &'a Diagnostics,

    // TODO maybe these should not be here, but only when in the relevant context?
    pub variables: Arena<Variable, VariableInfo>,
    pub ports: Arena<Port, PortInfo>,
    pub wires: Arena<Wire, WireInfo>,
    pub registers: Arena<Register, RegisterInfo>,
    pub scopes: Scopes<'a>,

    pub stack: Vec<ElaborationStackEntry>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct StackNotEq(usize);

// TODO cleanup
// TODO also set some stack limit to ensure we get properly formatted error messages (diags instead of rust stacktraces)
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ElaborationStackEntry {
    // TODO properly refer to ast instantiation site
    /// Module instantiation is only here to provide nicer error messages.
    ModuleInstantiation(Span, StackNotEq),
    // TODO better name, is this ItemEvaluation, ItemSignature, ...?
    Item(AstRefItem),
    // TODO better names
    FunctionCall(Span, StackNotEq),
    FunctionRun(AstRefItem, Vec<MaybeCompile<TypedIrExpression>>),
}

new_index_type!(pub Variable);
new_index_type!(pub Port);
new_index_type!(pub Wire);
new_index_type!(pub Register);

#[derive(Debug)]
pub struct ConstantInfo {
    pub id: MaybeIdentifier,
    pub value: CompileValue,
}

#[derive(Debug)]
pub struct VariableInfo {
    pub id: MaybeIdentifier,
    pub mutable: bool,
    pub ty: Option<Spanned<Type>>,
}

#[derive(Debug)]
pub struct PortInfo {
    pub id: Identifier,
    pub direction: Spanned<PortDirection>,
    pub domain: Spanned<PortDomain>,
    pub ty: Spanned<HardwareType>,
    pub ir: IrPort,
}

#[derive(Debug)]
pub struct WireInfo {
    pub id: MaybeIdentifier,
    pub domain: Spanned<ValueDomain>,
    pub ty: Spanned<HardwareType>,
    pub ir: IrWire,
}

#[derive(Debug)]
pub struct RegisterInfo {
    pub id: MaybeIdentifier,
    pub domain: Spanned<SyncDomain<DomainSignal>>,
    pub ty: Spanned<HardwareType>,
    pub ir: IrRegister,
}

impl PortInfo {
    pub fn typed_ir_expr(&self) -> TypedIrExpression {
        TypedIrExpression {
            ty: self.ty.inner.clone(),
            domain: ValueDomain::from_port_domain(self.domain.inner),
            expr: IrExpression::Port(self.ir),
        }
    }
}

impl WireInfo {
    pub fn typed_ir_expr(&self) -> TypedIrExpression {
        TypedIrExpression {
            ty: self.ty.inner.clone(),
            domain: self.domain.inner.clone(),
            expr: IrExpression::Wire(self.ir),
        }
    }
}

impl RegisterInfo {
    pub fn typed_ir_expr(&self) -> TypedIrExpression {
        TypedIrExpression {
            ty: self.ty.inner.clone(),
            domain: ValueDomain::from_domain_kind(DomainKind::Sync(self.domain.inner)),
            expr: IrExpression::Register(self.ir),
        }
    }
}

fn add_import_to_scope(
    diags: &Diagnostics,
    source: &SourceDatabase,
    file_to_scopes: &IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    file_scopes: &mut Arena<ScopeFile, ScopeInfo>,
    target_scope: ScopeFile,
    item: &ast::ItemImport,
) {
    // TODO the current path design does not allow private sub-modules
    //   are they really necessary? if all inner items are private it's effectively equivalent
    //   -> no it's not equivalent, things can also be private from the parent

    let ast::ItemImport {
        span: _,
        parents,
        entry,
    } = item;

    let parent_scope = find_path_delcare_scope(diags, source, file_to_scopes, parents);

    let import_entries = match &entry.inner {
        ast::ImportFinalKind::Single(entry) => std::slice::from_ref(entry),
        ast::ImportFinalKind::Multi(entries) => entries,
    };
    for import_entry in import_entries {
        let ast::ImportEntry { span: _, id, as_ } = import_entry;

        // TODO allow private visibility into child scopes?
        let entry = parent_scope.and_then(|parent_scope| {
            let tmp_scopes = Scopes::new(file_scopes);
            file_scopes[parent_scope]
                .find(&tmp_scopes, diags, id, Visibility::Public)
                .map(|entry| entry.value.clone())
        });

        let target_scope = &mut file_scopes[target_scope];
        match as_ {
            Some(as_) => target_scope.maybe_declare(diags, as_.as_ref(), entry, Visibility::Private),
            None => target_scope.declare(diags, id, entry, Visibility::Private),
        };
    }
}

fn find_path_delcare_scope(
    diags: &Diagnostics,
    source: &SourceDatabase,
    file_scopes: &IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    path: &Spanned<Vec<Identifier>>,
) -> Result<ScopeFile, ErrorGuaranteed> {
    // TODO the current path design does not allow private sub-modules
    //   are they really necessary? if all inner items are private it's effectively equivalent
    //   -> no it's not equivalent, things can also be private from the parent
    let mut curr_dir = source.root_directory;

    // get the span without the trailing separator
    let parents_span = if path.inner.is_empty() {
        path.span
    } else {
        path.inner.first().unwrap().span.join(path.inner.last().unwrap().span)
    };

    for step in &path.inner {
        let curr_dir_info = &source[curr_dir];

        curr_dir = match curr_dir_info.children.get(&step.string) {
            Some(&child_dir) => child_dir,
            None => {
                let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                options.sort();

                // TODO without trailing separator
                let diag = Diagnostic::new("import not found")
                    .snippet(path.span)
                    .add_error(step.span, "failed step")
                    .finish()
                    .footer(Level::Info, format!("possible options: {:?}", options))
                    .finish();
                throw!(diags.report(diag));
            }
        };
    }

    let file = match source[curr_dir].file {
        Some(file) => file,
        None => {
            throw!(diags.report_simple("expected path to file", parents_span, "no file exists at this path"))
        }
    };

    file_scopes
        .get(&file)
        .unwrap()
        .as_ref_ok()
        .map(|scopes| scopes.scope_outer_declare)
}

// TODO move to the scope module?
pub struct PopulatedFileScopes {
    pub file_scopes: Arena<ScopeFile, ScopeInfo>,
    pub file_to_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    // TODO maybe imports should not be items in the first place?
    pub all_items_except_imports: Vec<AstRefItem>,
}

fn populate_file_scopes(diags: &Diagnostics, source: &SourceDatabase, parsed: &ParsedDatabase) -> PopulatedFileScopes {
    let mut file_to_scopes = IndexMap::new();
    let mut file_scopes: Arena<ScopeFile, _> = Arena::default();

    let files = source.files();
    let mut all_items_except_imports = vec![];

    for &file in &files {
        let file_source = &source[file];

        let scope = parsed[file].as_ref_ok().map(|ast| {
            // build declaration scope
            // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
            let file_span = file_source.offsets.full_span(file);
            let scope_declare = file_scopes.push(ScopeInfo::new(file_span, None));
            let scope_import = file_scopes.push(ScopeInfo::new(
                file_span,
                Some((Scope::File(scope_declare), Visibility::Private)),
            ));

            let declare_scope_info = &mut file_scopes[scope_declare];

            for (ast_item_ref, ast_item) in ast.items_with_ref() {
                if let Some(declaration_info) = ast_item.declaration_info() {
                    let vis = match declaration_info.vis {
                        ast::Visibility::Public(_) => Visibility::Public,
                        ast::Visibility::Private => Visibility::Private,
                    };
                    declare_scope_info.maybe_declare(
                        diags,
                        declaration_info.id,
                        Ok(ScopedEntry::Item(ast_item_ref)),
                        vis,
                    );
                }

                match ast_item {
                    ast::Item::Import(_) => {}
                    _ => all_items_except_imports.push(ast_item_ref),
                }
            }

            FileScopes {
                scope_outer_declare: scope_declare,
                scope_inner_import: scope_import,
            }
        });

        file_to_scopes.insert_first(file, scope);
    }

    // populate import scopes
    for &file in &files {
        if let Ok(scopes) = file_to_scopes.get(&file).as_ref().unwrap() {
            let file_ast = parsed[file].as_ref_ok().unwrap();
            for item in &file_ast.items {
                if let ast::Item::Import(item) = item {
                    add_import_to_scope(
                        diags,
                        source,
                        &file_to_scopes,
                        &mut file_scopes,
                        scopes.scope_inner_import,
                        item,
                    );
                }
            }
        }
    }

    PopulatedFileScopes {
        file_scopes,
        file_to_scopes,
        all_items_except_imports,
    }
}

fn find_top_module(
    diags: &Diagnostics,
    fixed: CompileFixed,
    shared: &CompileShared,
) -> Result<AstRefModule, ErrorGuaranteed> {
    let top_file = fixed.source[fixed.source.root_directory]
        .children
        .get("top")
        .and_then(|&top_dir| fixed.source[top_dir].file)
        .ok_or_else(|| {
            let title = "no top file found, should be called `top` and be in the root directory of the project";
            diags.report(Diagnostic::new(title).finish())
        })?;
    let top_file_scope = shared
        .file_to_scopes
        .get(&top_file)
        .unwrap()
        .as_ref_ok()?
        .scope_outer_declare;
    let top_entry = shared.file_scopes[top_file_scope].find_immediate_str(diags, "top", Visibility::Public)?;

    match top_entry.value {
        &ScopedEntry::Item(item) => match &fixed.parsed[item] {
            ast::Item::Module(module) => match &module.params {
                None => Ok(AstRefModule::new_unchecked(item)),
                Some(_) => {
                    Err(diags.report_simple("`top` cannot have generic parameters", module.id.span, "defined here"))
                }
            },
            _ => Err(diags.report_simple("`top` should be a module", top_entry.defining_span, "defined here")),
        },
        ScopedEntry::Named(_) | ScopedEntry::Value(_) => {
            // TODO include "got" string
            // TODO is this even ever possible? direct should only be inside of scopes
            Err(diags.report_simple(
                "top should be an item, got a named/value",
                top_entry.defining_span,
                "defined here",
            ))
        }
    }
}

impl CompileShared<'_> {
    pub fn finish_ir_modules(self, diags: &Diagnostics, dummy_span: Span) -> Result<IrModules, ErrorGuaranteed> {
        let ir_modules = self.cache_modules.into_inner().unwrap().ir_modules;
        ir_modules.try_map_values(|v| match v.into_inner() {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(diags.report_internal_error(dummy_span, "not all modules were elaborated")),
        })
    }
}

impl<'a> CompileState<'a> {
    pub fn new(fixed: CompileFixed<'a>, shared: &'a CompileShared, diags: &'a Diagnostics) -> Self {
        CompileState {
            fixed,
            shared,
            diags,
            variables: Arena::default(),
            ports: Arena::default(),
            wires: Arena::default(),
            registers: Arena::default(),
            stack: vec![],
            scopes: Scopes::new(&shared.file_scopes),
        }
    }

    pub fn finish(self) {
        assert!(self.stack.is_empty());
        let _ = self;
    }

    pub fn not_eq_stack(&self) -> StackNotEq {
        StackNotEq(self.stack.len())
    }

    // TODO add stack limit
    pub fn with_recursion<T>(
        &mut self,
        entry: ElaborationStackEntry,
        f: impl FnOnce(&mut Self) -> T,
    ) -> Result<T, ErrorGuaranteed> {
        let diags = self.diags;
        let parsed = self.fixed.parsed;

        if let Some(loop_start) = self.stack.iter().position(|x| x == &entry) {
            // report elaboration loop
            let cycle = &self.stack[loop_start..];

            let mut diag = Diagnostic::new("encountered elaboration cycle");
            for (i, elem) in enumerate(cycle) {
                match elem {
                    &ElaborationStackEntry::ModuleInstantiation(span, _not_eq) => {
                        // TODO include generic args
                        diag = diag.add_error(span, format!("({i}): module instantiation"));
                    }
                    &ElaborationStackEntry::Item(item) => {
                        diag = diag.add_error(parsed[item].common_info().span_short, format!("({i}): item"));
                    }
                    &ElaborationStackEntry::FunctionCall(expr_span, _not_eq) => {
                        diag = diag.add_error(expr_span, format!("({i}): function call"));
                    }
                    &ElaborationStackEntry::FunctionRun(item, _) => {
                        // TODO include args
                        diag = diag.add_error(parsed[item].common_info().span_short, format!("({i}): function run"));
                    }
                }
            }

            Err(diags.report(diag.finish()))
        } else {
            self.stack.push(entry);
            let len_expected = self.stack.len();

            let result = f(self);

            assert_eq!(self.stack.len(), len_expected);
            self.stack.pop().unwrap();

            Ok(result)
        }
    }

    pub fn elaborate_module(
        &mut self,
        module: AstRefModule,
        args: Option<Args<Option<Identifier>, Spanned<CompileValue>>>,
    ) -> Result<ElaboratedModule, ErrorGuaranteed> {
        let params = self.elaborate_module_params(module, args)?;

        // check cache
        let cache_key = params.cache_key();
        let mut cache = self.shared.cache_modules.lock().unwrap();
        if let Some(result) = cache.elaborations.get(&cache_key) {
            return result.clone();
        }

        // new elaboration
        let elaborated = self.elaborate_module_ports_new(params).map(|header| {
            // reserve placeholder ir module, will be filled in later
            let ir_module: IrModule = cache.ir_modules.push(ComputeOnce::new());
            let elaborated = ElaboratedModule {
                ir_module,
                ports: header.ports.clone(),
            };
            self.shared
                .work_queue
                .push(WorkItem::ElaborateModule(ir_module, header));
            elaborated
        });

        // insert into cache
        cache.elaborations.insert_first(cache_key, elaborated.clone());
        drop(cache);

        elaborated
    }

    pub fn file_scope(&self, file: FileId) -> Result<ScopeFile, ErrorGuaranteed> {
        let diags = self.diags;
        let source = self.fixed.source;

        match self.shared.file_to_scopes.get(&file) {
            None => Err(diags.report_internal_error(source[file].offsets.full_span(file), "file scopes not found")),
            Some(Ok(scopes)) => Ok(scopes.scope_inner_import),
            Some(&Err(e)) => Err(e),
        }
    }

    pub fn domain_signal_to_ir(&self, signal: &DomainSignal) -> IrExpression {
        let &Polarized { inverted, signal } = signal;
        let inner = match signal {
            Signal::Port(port) => IrExpression::Port(self.ports[port].ir),
            Signal::Wire(wire) => IrExpression::Wire(self.wires[wire].ir),
            Signal::Register(reg) => IrExpression::Register(self.registers[reg].ir),
        };
        if inverted {
            IrExpression::BoolNot(Box::new(inner))
        } else {
            inner
        }
    }
}

#[derive(Debug)]
pub struct FileScopes {
    /// The scope that only includes top-level items defined in this file.
    pub scope_outer_declare: ScopeFile,
    /// Child scope of [scope_outer_declare] that includes all imported items.
    pub scope_inner_import: ScopeFile,
}

// TODO rename/expand to handle all external interation: IO, env vars, ...
pub trait PrintHandler {
    fn println(&self, s: &str);
}

pub struct NoPrintHandler;

impl PrintHandler for NoPrintHandler {
    fn println(&self, _: &str) {}
}

pub struct StdoutPrintHandler;

impl PrintHandler for StdoutPrintHandler {
    fn println(&self, s: &str) {
        println!("{}", s);
    }
}

pub struct CollectPrintHandler(RefCell<Vec<String>>);

impl CollectPrintHandler {
    pub fn new() -> Self {
        CollectPrintHandler(RefCell::new(Vec::new()))
    }

    pub fn finish(self) -> Vec<String> {
        self.0.into_inner()
    }
}

impl PrintHandler for CollectPrintHandler {
    fn println(&self, s: &str) {
        self.0.borrow_mut().push(s.to_string());
    }
}
