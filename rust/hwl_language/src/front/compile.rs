use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticError, Diagnostics};
use crate::front::flow::NextFlowRootId;
use crate::front::item::{ElaboratedModule, ElaborationArenas};
use crate::front::module_header::ElaboratedModuleHeader;
use crate::front::print::PrintHandler;
use crate::front::scope::{DeclaredValueSingle, FrozenScope, ScopeKey, ScopedEntry};
use crate::front::signal::{
    Port, PortInfo, PortInterface, PortInterfaceInfo, Wire, WireInfo, WireInterface, WireInterfaceInfo,
};
use crate::front::value::{CompileValue, Value};
use crate::mid::graph::ir_modules_check_no_instance_cycles;
use crate::mid::ir::{IrDatabase, IrLargeArena, IrModule, IrModuleInfo};
use crate::syntax::ast::{self, Identifier, MaybeIdentifier, Visibility};
use crate::syntax::hierarchy::SourceHierarchy;
use crate::syntax::parsed::{AstRefItem, AstRefModuleInternal, ParsedDatabase};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::syntax::source::{FileId, SourceDatabase};
use crate::util::arena::Arena;
use crate::util::data::{IndexMapExt, NonEmptyVec};
use crate::util::pool::ThreadPool;
use crate::util::sync::{ComputeOnceArena, SharedQueue};
use crate::util::{ResultDoubleExt, ResultExt};
use hwl_util::constants::{STACK_OVERFLOW_ERROR_ENTRIES_SHOWN, STACK_OVERFLOW_STACK_LIMIT};
use indexmap::IndexMap;
use itertools::{Itertools, zip_eq};
use rand::seq::SliceRandom;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum ResolveError {
    EmptyPath,
    ItemNotFound,
}

impl<'a, 's> CompileRefs<'a, 's> {
    pub fn run_compile_loop(self, diags: &Diagnostics, pool: Option<&ThreadPool>) {
        let mut diags_raw = if let Some(pool) = pool {
            // distribute elaboration access the pool
            let thread_diags = pool.broadcast(|| self.run_compile_loop_thread());
            thread_diags.into_iter().flatten().collect_vec()
        } else {
            // run elaboration on the current thread
            self.run_compile_loop_thread()
        };

        // TODO rework this once we have a proper query system
        diags_raw.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        for d in diags_raw {
            diags.push(d);
        }
    }

    fn run_compile_loop_thread(self) -> Vec<Diagnostic> {
        let mut diags_thread = vec![];

        while let Some(work_item) = self.shared.work_queue.pop() {
            match work_item {
                WorkItem::EvaluateItem(item) => {
                    self.shared.item_values.offer_to_compute(item, || {
                        let diags = Diagnostics::new();
                        let mut ctx = CompileItemContext::new(self, &diags, Some(item), None);
                        let result = ctx.eval_item_new(item);
                        diags_thread.extend(diags.finish());
                        result
                    });
                }
                WorkItem::ElaborateModuleBody(header, ir_module) => {
                    // do elaboration
                    let diags = Diagnostics::new();
                    let ir_module_info = self.elaborate_module_body_new(&diags, header);
                    diags_thread.extend(diags.finish());

                    // store result
                    let slot = &mut self.shared.ir_database.lock().unwrap().modules[ir_module];
                    assert!(slot.is_none());
                    *slot = Some(ir_module_info);
                }
            }
        }

        diags_thread
    }

    pub fn check_should_stop(&self, diags: &Diagnostics, span: Span) -> DiagResult {
        if (self.should_stop)() {
            Err(diags.report_error_simple("compilation interrupted", span, "while elaborating here"))
        } else {
            Ok(())
        }
    }

    pub fn resolve_item_by_path(&self, diags: &Diagnostics, path: Spanned<&str>) -> DiagResult<AstRefItem> {
        // TODO share code with resolve_import_path
        // split path
        let path_split = path.inner.split(".").collect_vec();
        let (&name, steps) = path_split.split_last().ok_or_else(|| {
            diags.report_error_simple(
                "invalid path: cannot be empty",
                path.span,
                format!("got empty path `{}`", path.inner),
            )
        })?;

        // follow steps to get node
        let mut curr_node = self.hierarchy.root_node();
        for &step in steps {
            curr_node = curr_node.children.get(step).ok_or_else(|| {
                diags.report_error_simple(
                    "invalid path: step does not exist in hierarchy",
                    path.span,
                    format!("hierarchy step `{}` does not exist in path `{}`", step, path.inner),
                )
            })?;
        }

        // get file scope
        let file = curr_node.file.ok_or_else(|| {
            diags.report_error_simple(
                "invalid path: expected file at end of hierarchy",
                path.span,
                format!("expected file at end of hierarchy for path `{}`", path.inner),
            )
        })?;
        let scope = self.shared.file_scope(file)?;

        // get item in scope
        // TODO check public?
        let entry = scope.find(diags, Spanned::new(path.span, name))?;
        let item = match entry.value {
            ScopedEntry::Item(item) => item,
            ScopedEntry::Named(_) | ScopedEntry::Captured(_) | ScopedEntry::Value(_) => {
                return Err(diags.report_error_internal(path.span, "file scopes should only contain items"));
            }
        };

        // queue item elaboration
        self.shared.work_queue.push(WorkItem::EvaluateItem(item));

        Ok(item)
    }
}

// TODO rename to CompileRefs?
#[derive(Copy, Clone)]
pub struct CompileRefs<'a, 's> {
    pub settings: &'a CompileSettings,
    pub source: &'a SourceDatabase,
    pub hierarchy: &'a SourceHierarchy,
    pub parsed: &'a ParsedDatabase,

    pub shared: &'s CompileShared,
    // TODO is there a reasonable way to get deterministic prints?
    //   -> yes, handle these the same way as diagnostics,
    //   make both use a gated-streaming model in CLI mode
    pub print_handler: &'a (dyn PrintHandler + Sync),
    pub should_stop: &'a (dyn Fn() -> bool + Sync),
}

#[derive(Debug, Clone)]
pub struct CompileSettings {
    pub do_ir_cleanup: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum QueueItems {
    None,
    All,
}

#[derive(Debug)]
pub enum WorkItem {
    EvaluateItem(AstRefItem),
    ElaborateModuleBody(ElaboratedModuleHeader<AstRefModuleInternal>, IrModule),
}

/// long-term shared between threads
pub struct CompileShared {
    pub file_scopes: FileScopes,

    pub work_queue: SharedQueue<WorkItem>,

    pub item_values: ComputeOnceArena<AstRefItem, DiagResult<CompileValue>, StackEntry>,
    pub elaboration_arenas: ElaborationArenas,
    // TODO make this a non-blocking collection thing, could be thread-local collection and merging or a channel
    //   or maybe just another sharded DashMap
    pub ir_database: Mutex<IrDatabase<Option<DiagResult<IrModuleInfo>>>>,
    pub next_flow_root_id: NextFlowRootId,
}

pub type FileScopes = IndexMap<FileId, DiagResult<Arc<FrozenScope>>>;

pub type ArenaPorts = Arena<Port, PortInfo>;
pub type ArenaPortInterfaces = Arena<PortInterface, PortInterfaceInfo>;

pub struct CompileItemContext<'a, 's, 'd> {
    pub refs: CompileRefs<'a, 's>,
    // TODO find a better name for this
    pub state: CompileItemContextState,
    pub diags: &'d Diagnostics,
}

#[derive(Debug)]
pub struct CompileItemContextState {
    // TODO replace with query
    pub origin: Option<AstRefItem>,
    pub call_stack: Vec<StackEntry>,

    // TODO all of this should really be part of optional module-specific state, not always-present fields
    pub curr_module: Option<ElaboratedModule>,
    pub ports: ArenaPorts,
    pub port_interfaces: Arena<PortInterface, PortInterfaceInfo>,
    pub wires: Arena<Wire, WireInfo>,
    pub wire_interfaces: Arena<WireInterface, WireInterfaceInfo>,
    pub large: IrLargeArena,
}

#[derive(Debug, Copy, Clone)]
pub enum StackEntry {
    ItemUsage(Span),
    ItemEvaluation(Span),
    FunctionCall(Span),
    FunctionRun(Span),
}

impl StackEntry {
    pub fn into_span_message(self, depth: usize) -> (Span, String) {
        let (span, msg) = match self {
            StackEntry::ItemUsage(span) => (span, "item used here"),
            StackEntry::ItemEvaluation(span) => (span, "item declared here"),
            StackEntry::FunctionCall(span) => (span, "function call here"),
            StackEntry::FunctionRun(span) => (span, "function declared here"),
        };

        (span, format!("[{depth}] {msg}"))
    }
}

impl<'a, 's, 'd> CompileItemContext<'a, 's, 'd> {
    pub fn new(
        refs: CompileRefs<'a, 's>,
        diags: &'d Diagnostics,
        origin: Option<AstRefItem>,
        curr_module: Option<ElaboratedModule>,
    ) -> Self {
        let state = CompileItemContextState {
            origin,
            call_stack: vec![],
            curr_module,
            ports: Arena::new(),
            port_interfaces: Arena::new(),
            wires: Arena::new(),
            wire_interfaces: Arena::new(),
            large: Arena::new(),
        };
        CompileItemContext { refs, diags, state }
    }

    pub fn restore(fixed: CompileRefs<'a, 's>, diags: &'d Diagnostics, state: CompileItemContextState) -> Self {
        CompileItemContext {
            refs: fixed,
            diags,
            state,
        }
    }

    pub fn into_content(self) -> CompileItemContextState {
        let CompileItemContext {
            refs: _,
            diags: _,
            state,
        } = self;
        state
    }

    pub fn check_should_stop(&self, span: Span) -> DiagResult {
        self.refs.check_should_stop(self.diags, span)
    }

    pub fn recurse<R>(&mut self, entry: StackEntry, f: impl FnOnce(&mut Self) -> R) -> DiagResult<R> {
        if self.state.call_stack.len() >= STACK_OVERFLOW_STACK_LIMIT {
            return Err(stack_overflow_diagnostic(&self.state.call_stack).report(self.diags));
        }

        self.state.call_stack.push(entry);
        let len = self.state.call_stack.len();

        let result = f(self);

        assert_eq!(self.state.call_stack.len(), len);
        self.state.call_stack.pop().unwrap();

        Ok(result)
    }

    pub fn eval_item(&mut self, item: AstRefItem) -> DiagResult<&'s CompileValue> {
        let item_span = self.refs.parsed[item].info().span_short;
        let stack_entry = StackEntry::ItemEvaluation(item_span);

        self.recurse(stack_entry, |s| {
            let origin = s.state.origin.map(|origin| (origin, s.state.call_stack.clone()));
            let f_compute = || {
                // TODO change diags propagation here
                // TODO we're possibly creating a new redundant context here, is that normal?
                let mut ctx = CompileItemContext::new(s.refs, self.diags, Some(item), None);
                ctx.eval_item_new(item)
            };
            // TODO careful about where cycles get reported!
            let f_cycle = |stack: Vec<&StackEntry>| cycle_diagnostic(stack).report(s.diags);
            s.refs
                .shared
                .item_values
                .get_or_compute(origin, item, f_compute, f_cycle)
                .map(ResultExt::as_ref_ok)
                .flatten_err()
        })
        .flatten_err()
    }
}

fn cycle_diagnostic(mut stack: Vec<&StackEntry>) -> DiagnosticError {
    // rotate the stack to keep error messages deterministic
    assert!(!stack.is_empty());
    let min_index = stack
        .iter()
        .position_min_by_key(|entry| entry.into_span_message(0).0)
        .unwrap();
    stack.rotate_left(min_index);
    let stack = NonEmptyVec::try_from(stack).unwrap();

    // create the diagnostic
    let mut next_index: usize = 0;
    let messages = stack.map(|entry| {
        let index = next_index;
        next_index += 1;
        entry.into_span_message(index)
    });

    DiagnosticError::new_multiple("encountered cyclic dependency", messages)
}

fn stack_overflow_diagnostic(stack: &Vec<StackEntry>) -> DiagnosticError {
    assert!(!stack.is_empty());

    let mut messages = vec![];

    let skipped = if stack.len() <= 2 * STACK_OVERFLOW_ERROR_ENTRIES_SHOWN {
        for depth in 0..stack.len() {
            messages.push(stack[depth].into_span_message(depth));
        }
        None
    } else {
        for depth in 0..STACK_OVERFLOW_ERROR_ENTRIES_SHOWN {
            messages.push(stack[depth].into_span_message(depth));
        }
        for depth in (stack.len() - STACK_OVERFLOW_ERROR_ENTRIES_SHOWN)..stack.len() {
            messages.push(stack[depth].into_span_message(depth));
        }
        Some(stack.len() - 2 * STACK_OVERFLOW_ERROR_ENTRIES_SHOWN)
    };

    let mut diag = DiagnosticError::new_multiple(
        format!("stack overflow, stack depth {}", stack.len()),
        NonEmptyVec::try_from(messages).unwrap(),
    );

    if let Some(skipped) = skipped {
        diag = diag.add_footer_info(format!("skipped showing {} stack entries in the middle", skipped))
    }

    diag
}

#[derive(Debug, Clone)]
pub enum CompileStackEntry {
    Item(AstRefItem),
    FunctionCall(Span),
    FunctionRun(AstRefItem, Vec<Value>),
}

fn populate_file_scopes(
    diags: &Diagnostics,
    source: &SourceDatabase,
    hierarchy: &SourceHierarchy,
    parsed: &ParsedDatabase,
) -> FileScopes {
    // pass 0: add all declared items to the file scope
    let mut file_scopes = IndexMap::new();
    for file in hierarchy.files() {
        let scope = parsed[file].as_ref_ok().map(|ast| {
            let mut scope = FrozenScope::new(ast.span);
            for (ast_item_ref, ast_item) in ast.items_with_ref() {
                if let Some(info) = ast_item.info().declaration {
                    scope.declare(diags, info.id.spanned_str(source), Ok(ScopedEntry::Item(ast_item_ref)));
                }
            }
            scope
        });

        file_scopes.insert_first(file, scope);
    }

    // pass 1: resolve all imports and collect the imported items
    // (don't immediately add them, then then would already be visible for later imports from other files)
    let mut file_imported_items: Vec<Vec<(MaybeIdentifier, DiagResult<ScopedEntry>)>> = vec![];
    for target_file in hierarchy.files() {
        let mut curr_imported_items = vec![];

        if let Ok(target_file_ast) = &parsed[target_file] {
            for item in &target_file_ast.items {
                if let ast::Item::Import(item) = item {
                    let ast::ItemImport {
                        span: _,
                        parents,
                        entry,
                    } = item;

                    let source_scope = resolve_import_path(diags, source, hierarchy, parents)
                        .and_then(|source_file| file_scopes.get(&source_file).unwrap().as_ref_ok());

                    let entries = match &entry.inner {
                        ast::ImportFinalKind::Single(entry) => std::slice::from_ref(entry),
                        ast::ImportFinalKind::Multi(entries) => entries,
                    };

                    for entry in entries {
                        let &ast::ImportEntry { span: _, id, as_ } = entry;

                        // TODO suggest alternatives, like for parent imports
                        let source_value = source_scope
                            .and_then(|source_scope| source_scope.find(diags, id.spanned_str(source)))
                            .map(|found| found.value);

                        // check visibility, but still proceed as if the import succeeded
                        if let Ok(ScopedEntry::Item(source_item)) = source_value {
                            let decl_info = parsed[source_item].info().declaration.unwrap();
                            match decl_info.vis {
                                Visibility::Public { span: _ } => {}
                                Visibility::Private => {
                                    let _ = DiagnosticError::new(
                                        format!("cannot access identifier `{}`", id.str(source)),
                                        id.span,
                                        "not accessible here",
                                    )
                                    .add_info(decl_info.id.span(), "identifier declared here")
                                    .add_footer_info("private items cannot be accessed outside of the declaring file")
                                    .report(diags);
                                }
                            }
                        }

                        let target_id: MaybeIdentifier = match as_ {
                            Some(as_) => as_,
                            None => MaybeIdentifier::Identifier(id),
                        };
                        curr_imported_items.push((target_id, source_value));
                    }
                }
            }
        }

        file_imported_items.push(curr_imported_items);
    }

    // pass 2: actually add the imported items to the file scopes
    for (target_file, items) in zip_eq(hierarchy.files(), file_imported_items) {
        if let Ok(scope) = file_scopes.get_mut(&target_file).unwrap() {
            for (target_id, value) in items {
                scope.declare(diags, target_id.spanned_str(source), value);
            }
        }
    }

    // pass 3: add prelude items to all files scopes
    // TODO this silently does nothing if there is no std library, is that okay?
    // TODO this causes errors if there are duplicate identifiers
    let mut prelude_imported_items: Vec<(String, DeclaredValueSingle)> = vec![];
    for std_file in ["types", "util", "math"] {
        let file = hierarchy
            .root_node()
            .children
            .get("std")
            .and_then(|n| n.children.get(std_file))
            .and_then(|n| n.file);

        if let Some(file) = file {
            let scope = &file_scopes.get(&file).unwrap();
            if let Ok(scope) = scope {
                scope.for_each_immediate_entry(|name, value| {
                    if let ScopeKey::Id(name) = name {
                        prelude_imported_items.push((name.to_owned(), value.cloned()));
                    }
                });
            }
        }
    }
    for file in hierarchy.files() {
        if let Ok(scope) = file_scopes.get_mut(&file).unwrap() {
            for (name, value) in &prelude_imported_items {
                if !scope.has_immediate_entry(ScopeKey::Id(name.as_str())) {
                    scope.declare_already_checked(name.clone(), value.clone());
                }
            }
        }
    }

    file_scopes.into_iter().map(|(k, v)| (k, v.map(Arc::new))).collect()
}

fn resolve_import_path(
    diags: &Diagnostics,
    source: &SourceDatabase,
    hierarchy: &SourceHierarchy,
    path: &Spanned<Vec<Identifier>>,
) -> DiagResult<FileId> {
    // TODO the current path design does not allow private sub-modules
    //   are they really necessary? if all inner items are private it's effectively equivalent
    //   -> no it's not equivalent, things can also be private from the parent
    let mut curr_node = hierarchy.root_node();

    // get the span without the trailing separator
    let parents_span = if path.inner.is_empty() {
        // TODO is an empty path even possible?
        path.span
    } else {
        path.inner.first().unwrap().span.join(path.inner.last().unwrap().span)
    };

    for step in &path.inner {
        curr_node = match curr_node.children.get(step.str(source)) {
            Some(child_node) => child_node,
            None => {
                let mut options = curr_node.children.keys().cloned().collect_vec();
                options.sort();

                // TODO without trailing separator
                return Err(DiagnosticError::new("import not found", step.span, "failed step")
                    .add_footer_hint(format!("possible options: {options:?}"))
                    .report(diags));
            }
        };
    }

    curr_node
        .file
        .ok_or_else(|| diags.report_error_simple("expected path to file", parents_span, "no file exists at this path"))
}

impl CompileShared {
    pub fn new(
        diags: &Diagnostics,
        source: &SourceDatabase,
        hierarchy: &SourceHierarchy,
        parsed: &ParsedDatabase,
        queue_items: QueueItems,
        thread_count: NonZeroUsize,
    ) -> Self {
        let file_scopes = populate_file_scopes(diags, source, hierarchy, parsed);

        // pass over all items, to:
        // * collect all non-import items for the compute arena
        // * find all external modules
        // TODO make also skip trivial items already, eg. functions and generic modules
        let mut items = vec![];
        let mut external_modules: IndexMap<String, Vec<Span>> = IndexMap::new();
        for file in hierarchy.files() {
            if let Ok(file_ast) = &parsed[file] {
                for (item_ref, item) in file_ast.items_with_ref() {
                    if !matches!(item, ast::Item::Import(_)) {
                        items.push(item_ref);
                    }
                    if let ast::Item::ModuleExternal(module) = item {
                        external_modules
                            .entry(module.id.str(source).to_owned())
                            .or_default()
                            .push(module.id.span);
                    }
                }
            }
        }

        // check for duplicate external modules
        for (name, spans) in &external_modules {
            if spans.len() > 1 {
                let messages = spans
                    .iter()
                    .map(|&span| (span, "defined here".to_owned()))
                    .collect_vec();
                let messages = NonEmptyVec::try_from(messages).unwrap();

                let _ = DiagnosticError::new_multiple(
                    format!("external module with name `{name}` declared twice"),
                    messages,
                )
                .report(diags);
            }
        }
        let external_modules = external_modules.into_keys().collect();

        let item_values = ComputeOnceArena::new(thread_count);

        // populate work queue
        // TODO shuffle or not?
        //   * which is actually faster? (for mutex contention)
        //   * do we want to shuffle anyway to test that the compiler is deterministic?
        let work_queue = SharedQueue::new(thread_count);
        match queue_items {
            QueueItems::None => {}
            QueueItems::All => {
                items.shuffle(&mut rand::thread_rng());
                work_queue.push_batch(items.into_iter().map(WorkItem::EvaluateItem));
            }
        }

        CompileShared {
            file_scopes,
            work_queue,
            item_values,
            elaboration_arenas: ElaborationArenas::new(),
            ir_database: Mutex::new(IrDatabase {
                modules: Arena::new(),
                external_modules,
            }),
            next_flow_root_id: NextFlowRootId::default(),
        }
    }

    pub fn file_scope(&self, file: FileId) -> DiagResult<&Arc<FrozenScope>> {
        self.file_scopes.get(&file).unwrap().as_ref_ok()
    }

    /// Look up the given item, returning its value if its evaluation has already finished.
    /// This is useful when the caller expects the item to already have been evaluated,
    ///   to avoid accidentally queuing up more work.
    pub fn eval_item_if_complete(&self, item: AstRefItem) -> Option<DiagResult<&CompileValue>> {
        self.item_values.get(item).map(Result::as_ref_ok)
    }

    pub fn finish_ir_database(self, diags: &Diagnostics, dummy_span: Span) -> DiagResult<IrDatabase<IrModuleInfo>> {
        finish_ir_database_impl(
            diags,
            dummy_span,
            &self.work_queue,
            self.ir_database.into_inner().unwrap(),
        )
    }

    pub fn finish_ir_database_ref(
        &self,
        diags: &Diagnostics,
        dummy_span: Span,
    ) -> DiagResult<IrDatabase<IrModuleInfo>> {
        let ir_database = self.ir_database.lock().unwrap().clone();
        finish_ir_database_impl(diags, dummy_span, &self.work_queue, ir_database)
    }
}

fn finish_ir_database_impl(
    diags: &Diagnostics,
    dummy_span: Span,
    work_queue: &SharedQueue<WorkItem>,
    ir_database: IrDatabase<Option<DiagResult<IrModuleInfo>>>,
) -> DiagResult<IrDatabase<IrModuleInfo>> {
    // check that work queue is empty
    if work_queue.pop().is_some() {
        return Err(diags.report_error_internal(dummy_span, "not all work items have been processed"));
    }

    // check that all modules have been elaborated
    let IrDatabase {
        modules,
        external_modules,
    } = ir_database;
    let modules = modules.try_map_values(|_, v| match v {
        Some(Ok(v)) => Ok(v),
        Some(Err(e)) => Err(e),
        None => Err(diags.report_error_internal(dummy_span, "not all modules were elaborated")),
    })?;

    // check that there are no cycles
    ir_modules_check_no_instance_cycles(diags, &modules)?;

    Ok(IrDatabase {
        modules,
        external_modules,
    })
}
