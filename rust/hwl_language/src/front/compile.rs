use crate::constants::{MAX_STACK_ENTRIES, STACK_OVERFLOW_ERROR_ENTRIES_SHOWN, THREAD_STACK_SIZE};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, DiagnosticBuilder, Diagnostics, ErrorGuaranteed};
use crate::front::domain::DomainSignal;
use crate::front::item::{ElaboratedModule, ElaborationArenas};
use crate::front::module::ElaboratedModuleHeader;
use crate::front::print::PrintHandler;
use crate::front::scope::{Scope, ScopedEntry};
use crate::front::signal::Signal;
use crate::front::signal::{
    Polarized, Port, PortInfo, PortInterface, PortInterfaceInfo, Register, RegisterInfo, Wire, WireInfo, WireInterface,
    WireInterfaceInfo,
};
use crate::front::value::{CompileValue, Value};
use crate::front::variables::{Variable, VariableInfo};
use crate::mid::ir::{IrDatabase, IrExpression, IrExpressionLarge, IrLargeArena, IrModule, IrModuleInfo};
use crate::syntax::ast::Spanned;
use crate::syntax::ast::{self, Expression, ExpressionKind, Identifier, MaybeIdentifier, Visibility};
use crate::syntax::parsed::{AstRefItem, AstRefModuleInternal, ParsedDatabase};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::throw;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::sync::{ComputeOnceArena, SharedQueue};
use crate::util::{ResultDoubleExt, ResultExt};
use annotate_snippets::Level;
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, zip_eq, Itertools};
use rand::seq::SliceRandom;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::Mutex;

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
    elaboration_set: ElaborationSet,
    print_handler: &mut (dyn PrintHandler + Sync),
    should_stop: &(dyn Fn() -> bool + Sync),
    thread_count: NonZeroUsize,
) -> Result<IrDatabase, ErrorGuaranteed> {
    let fixed = CompileFixed { source, parsed };

    let queue_all_items = match elaboration_set {
        ElaborationSet::TopOnly => false,
        ElaborationSet::AsMuchAsPossible => true,
    };
    let shared = CompileShared::new(diags, fixed, queue_all_items, thread_count);

    // get the top module
    // TODO we don't really need to do this any more, all non-generic modules are elaborated anyway
    // TODO change this once we're not hardcoding a single top item any more
    let top_item_and_ir_module = {
        let refs = CompileRefs {
            fixed,
            shared: &shared,
            diags,
            print_handler,
            should_stop,
        };
        find_top_module(diags, fixed, &shared).and_then(|top_item| {
            let mut ctx = CompileItemContext::new_empty(refs, None);
            let result = ctx.eval_item(top_item.item())?;

            match result {
                &CompileValue::Module(ElaboratedModule::Internal(elab)) => {
                    let info = shared.elaboration_arenas.module_internal_info(elab);
                    Ok((top_item, info.module_ir))
                }
                _ => Err(diags.report_internal_error(parsed[top_item].id.span(), "top items should be modules")),
            }
        })
    };

    // run until everything is elaborated
    if thread_count.get() > 1 {
        // TODO manage thread pool externally, and re-use it between parsing and elaboration
        // TODO use the current thread as one of thread loops instead of just joining?
        std::thread::scope(|s| {
            let mut handles = vec![];

            for thread_index in 0..thread_count.get() {
                let f = || {
                    let thread_diags = Diagnostics::new();
                    let thread_refs = CompileRefs {
                        fixed,
                        shared: &shared,
                        diags: &thread_diags,
                        print_handler,
                        should_stop,
                    };
                    thread_refs.run_elaboration_loop();
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

            // merge diagnostics, sorting to keep them deterministic
            // TODO some kind of topological sort "as if visited by single thread" might be nicer
            let mut all_diags = vec![];
            for h in handles {
                // TODO propagate panics better here, ideally all threads would stop and program would fully exit
                let thread_diags = h.join().unwrap();
                all_diags.extend(thread_diags.finish());
            }
            all_diags.sort_by_key(|d| d.main_annotation().map(|a| a.span));
            for d in all_diags {
                diags.report(d);
            }
        });
    } else {
        let thread_refs = CompileRefs {
            fixed,
            shared: &shared,
            diags,
            print_handler,
            should_stop,
        };
        thread_refs.run_elaboration_loop();
    }

    // return result (at this point all modules should have been fully elaborated)
    let (top_item, top_ir_module) = top_item_and_ir_module?;

    let db_partial = shared.finish_ir_database(diags, parsed[top_item].span)?;

    let db = IrDatabase {
        top_module: top_ir_module,
        modules: db_partial.ir_modules,
        external_modules: db_partial.external_modules.into_iter().sorted().collect(),
    };

    // TODO add an option to always do this?
    #[cfg(debug_assertions)]
    db.validate(diags)?;

    Ok(db)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ElaborationSet {
    TopOnly,
    AsMuchAsPossible,
}

impl<'a> CompileRefs<'a, '_> {
    pub fn check_should_stop(&self, span: Span) -> Result<(), ErrorGuaranteed> {
        // TODO report only one error, now all threads report the same error
        //   put an Option<ErrorGuaranteed> somewhere in a mutex?
        if (self.should_stop)() {
            Err(self
                .diags
                .report_simple("compilation interrupted", span, "while elaborating here"))
        } else {
            Ok(())
        }
    }

    pub fn run_elaboration_loop(self) {
        while let Some(work_item) = self.shared.work_queue.pop() {
            match work_item {
                WorkItem::EvaluateItem(item) => {
                    self.shared.item_values.offer_to_compute(item, || {
                        let mut ctx = CompileItemContext::new_empty(self, Some(item));
                        ctx.eval_item_new(item)
                    });
                }
                WorkItem::ElaborateModuleBody(header, ir_module) => {
                    // do elaboration
                    let ir_module_info = self.elaborate_module_body_new(header);

                    // store result
                    let slot = &mut self.shared.ir_database.lock().unwrap().ir_modules[ir_module];
                    assert!(slot.is_none());
                    *slot = Some(ir_module_info);
                }
            }
        }
    }

    pub fn get_expr(&self, expr: Expression) -> &'a ExpressionKind {
        self.fixed.parsed.get_expr(expr)
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
    ElaborateModuleBody(ElaboratedModuleHeader<AstRefModuleInternal>, IrModule),
}

/// long-term shared between threads
pub struct CompileShared {
    pub file_scopes: FileScopes,

    pub work_queue: SharedQueue<WorkItem>,

    pub item_values: ComputeOnceArena<AstRefItem, Result<CompileValue, ErrorGuaranteed>, StackEntry>,
    pub elaboration_arenas: ElaborationArenas,
    // TODO make this a non-blocking collection thing, could be thread-local collection and merging or a channel
    //   or maybe just another sharded DashMap
    pub ir_database: Mutex<PartialIrDatabase<Option<Result<IrModuleInfo, ErrorGuaranteed>>>>,
}

pub struct PartialIrDatabase<M> {
    pub external_modules: IndexSet<String>,
    pub ir_modules: Arena<IrModule, M>,
}

pub type FileScopes = IndexMap<FileId, Result<Scope<'static>, ErrorGuaranteed>>;

#[derive(Copy, Clone)]
pub struct CompileRefs<'a, 's> {
    // TODO maybe inline this
    pub fixed: CompileFixed<'a>,
    pub shared: &'s CompileShared,
    pub diags: &'a Diagnostics,
    // TODO is there a reasonable way to get deterministic prints?
    pub print_handler: &'a (dyn PrintHandler + Sync),
    pub should_stop: &'a dyn Fn() -> bool,
}

pub type ArenaVariables = Arena<Variable, VariableInfo>;
pub type ArenaPorts = Arena<Port, PortInfo>;
pub type ArenaPortInterfaces = Arena<PortInterface, PortInterfaceInfo>;

pub struct CompileItemContext<'a, 's> {
    // TODO maybe inline this
    pub refs: CompileRefs<'a, 's>,

    pub variables: ArenaVariables,
    pub ports: ArenaPorts,
    pub port_interfaces: Arena<PortInterface, PortInterfaceInfo>,
    pub wires: Arena<Wire, WireInfo>,
    pub wire_interfaces: Arena<WireInterface, WireInterfaceInfo>,
    pub registers: Arena<Register, RegisterInfo>,
    pub large: IrLargeArena,

    pub origin: Option<AstRefItem>,
    pub stack: Vec<StackEntry>,
}

#[derive(Debug, Copy, Clone)]
pub enum StackEntry {
    ItemUsage(Span),
    ItemEvaluation(Span),
    FunctionCall(Span),
    FunctionRun(Span),
}

impl StackEntry {
    pub fn into_span_message(self) -> (Span, &'static str) {
        match self {
            StackEntry::ItemUsage(span) => (span, "item used here"),
            StackEntry::ItemEvaluation(entry_span) => (entry_span, "item declared here"),
            StackEntry::FunctionCall(entry_span) => (entry_span, "function call here"),
            StackEntry::FunctionRun(entry_span) => (entry_span, "function declared here"),
        }
    }
}

impl<'a, 's> CompileItemContext<'a, 's> {
    pub fn new_empty(refs: CompileRefs<'a, 's>, origin: Option<AstRefItem>) -> Self {
        Self::new_restore(refs, origin, Arena::new(), Arena::new(), Arena::new())
    }

    pub fn new_restore(
        refs: CompileRefs<'a, 's>,
        origin: Option<AstRefItem>,
        ports: ArenaPorts,
        port_interfaces: ArenaPortInterfaces,
        variables: ArenaVariables,
    ) -> Self {
        CompileItemContext {
            refs,
            variables,
            ports,
            port_interfaces,
            wires: Arena::new(),
            wire_interfaces: Arena::new(),
            registers: Arena::new(),
            large: IrLargeArena::new(),
            origin,
            stack: vec![],
        }
    }

    pub fn recurse<R>(&mut self, entry: StackEntry, f: impl FnOnce(&mut Self) -> R) -> Result<R, ErrorGuaranteed> {
        if self.stack.len() > MAX_STACK_ENTRIES {
            return Err(self.refs.diags.report(stack_overflow_diagnostic(&self.stack)));
        }

        self.stack.push(entry);
        let len = self.stack.len();

        let result = f(self);

        assert_eq!(self.stack.len(), len);
        self.stack.pop().unwrap();

        Ok(result)
    }

    pub fn eval_item(&mut self, item: AstRefItem) -> Result<&CompileValue, ErrorGuaranteed> {
        let item_span = self.refs.fixed.parsed[item].info().span_short;
        let stack_entry = StackEntry::ItemEvaluation(item_span);

        self.recurse(stack_entry, |s| {
            let origin = s.origin.map(|origin| (origin, s.stack.clone()));
            let f_compute = || {
                let mut ctx = CompileItemContext::new_empty(s.refs, Some(item));
                ctx.eval_item_new(item)
            };
            let f_cycle = |stack: Vec<&StackEntry>| s.refs.diags.report(cycle_diagnostic(stack));
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

fn cycle_diagnostic(mut stack: Vec<&StackEntry>) -> Diagnostic {
    // sort the stack to keep error messages deterministic
    assert!(!stack.is_empty());
    let min_index = stack
        .iter()
        .position_min_by_key(|entry| {
            let (span, _) = entry.into_span_message();
            span
        })
        .unwrap();
    stack.rotate_left(min_index);

    // create the diagnostic
    let mut diag = Diagnostic::new("encountered cyclic dependency");
    for (entry_index, &entry) in enumerate(stack) {
        let (span, label) = entry.into_span_message();
        diag = diag.add_error(span, format!("[{entry_index}] {label}"));
    }
    diag.finish()
}

fn stack_overflow_diagnostic(stack: &Vec<StackEntry>) -> Diagnostic {
    let mut diag = Diagnostic::new(format!("encountered stack overflow, stack depth {}", stack.len()));

    let add_entry = |diag: DiagnosticBuilder, index: usize, entry: &StackEntry| {
        let (span, label) = entry.into_span_message();
        diag.add_error(span, format!("[{index}] {label}"))
    };

    if stack.len() <= 2 * STACK_OVERFLOW_ERROR_ENTRIES_SHOWN {
        for (entry_index, entry) in enumerate(stack) {
            diag = add_entry(diag, entry_index, entry);
        }
    } else {
        for entry_index in 0..STACK_OVERFLOW_ERROR_ENTRIES_SHOWN {
            diag = add_entry(diag, entry_index, &stack[entry_index]);
        }
        for entry_index in (stack.len() - STACK_OVERFLOW_ERROR_ENTRIES_SHOWN)..stack.len() {
            diag = add_entry(diag, entry_index, &stack[entry_index]);
        }
        diag = diag.footer(
            Level::Info,
            format!(
                "skipped showing {} stack entries in the middle",
                stack.len() - STACK_OVERFLOW_ERROR_ENTRIES_SHOWN
            ),
        )
    }

    diag.finish()
}

#[derive(Debug, Clone)]
pub enum CompileStackEntry {
    Item(AstRefItem),
    FunctionCall(Span),
    FunctionRun(AstRefItem, Vec<Value>),
}

fn populate_file_scopes(diags: &Diagnostics, fixed: CompileFixed) -> FileScopes {
    let CompileFixed { source, parsed } = fixed;

    // pass 0: add all items to their own file scope
    let mut file_scopes: FileScopes = IndexMap::new();
    for file in source.files() {
        let scope = parsed[file].as_ref_ok().map(|ast| {
            let mut scope = Scope::new_root(ast.span, file);
            for (ast_item_ref, ast_item) in ast.items_with_ref() {
                if let Some(info) = ast_item.info().declaration {
                    scope.maybe_declare(
                        diags,
                        Ok(info.id.spanned_str(source)),
                        Ok(ScopedEntry::Item(ast_item_ref)),
                    );
                }
            }
            scope
        });

        file_scopes.insert_first(file, scope);
    }

    // pass 1: collect import items from source scopes for each target file
    let mut file_imported_items: Vec<Vec<(MaybeIdentifier, Result<ScopedEntry, ErrorGuaranteed>)>> = vec![];
    for target_file in source.files() {
        let mut curr_imported_items = vec![];

        if let Ok(target_file_ast) = &parsed[target_file] {
            for item in &target_file_ast.items {
                if let ast::Item::Import(item) = item {
                    let ast::ItemImport {
                        span: _,
                        parents,
                        entry,
                    } = item;

                    let source_scope = resolve_import_path(diags, source, parents)
                        .and_then(|source_file| file_scopes.get(&source_file).unwrap().as_ref_ok());

                    let entries = match &entry.inner {
                        ast::ImportFinalKind::Single(entry) => std::slice::from_ref(entry),
                        ast::ImportFinalKind::Multi(entries) => entries,
                    };

                    for entry in entries {
                        let &ast::ImportEntry { span: _, id, as_ } = entry;
                        let source_value = source_scope
                            .and_then(|source_scope| source_scope.find(diags, id.spanned_str(source)))
                            .map(|found| found.value.clone());

                        // check visibility, but still proceed as if the import succeeded
                        if let Ok(ScopedEntry::Item(source_item)) = source_value {
                            let decl_info = parsed[source_item].info().declaration.unwrap();
                            match decl_info.vis {
                                Visibility::Public(_) => {}
                                Visibility::Private => {
                                    let err = Diagnostic::new(format!("cannot access identifier `{}`", id.str(source)))
                                        .add_info(decl_info.id.span(), "identifier declared here")
                                        .add_error(id.span, "not accessible here")
                                        .footer(
                                            Level::Info,
                                            "private items cannot be accessed outside of the declaring file",
                                        )
                                        .finish();
                                    diags.report(err);
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

    // pass 3: add imported items to the target file scope
    for (target_file, items) in zip_eq(source.files(), file_imported_items) {
        if let Ok(scope) = file_scopes.get_mut(&target_file).unwrap() {
            for (target_id, value) in items {
                scope.maybe_declare(diags, Ok(target_id.spanned_str(source)), value);
            }
        }
    }

    file_scopes
}

fn resolve_import_path(
    diags: &Diagnostics,
    source: &SourceDatabase,
    path: &Spanned<Vec<Identifier>>,
) -> Result<FileId, ErrorGuaranteed> {
    // TODO the current path design does not allow private sub-modules
    //   are they really necessary? if all inner items are private it's effectively equivalent
    //   -> no it's not equivalent, things can also be private from the parent
    let mut curr_dir = source.root_directory();

    // get the span without the trailing separator
    let parents_span = if path.inner.is_empty() {
        path.span
    } else {
        path.inner.first().unwrap().span.join(path.inner.last().unwrap().span)
    };

    for step in &path.inner {
        let curr_dir_info = &source[curr_dir];

        curr_dir = match curr_dir_info.children.get(step.str(source)) {
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

    source[curr_dir]
        .file
        .ok_or_else(|| diags.report_simple("expected path to file", parents_span, "no file exists at this path"))
}

fn find_top_module(
    diags: &Diagnostics,
    fixed: CompileFixed,
    shared: &CompileShared,
) -> Result<AstRefModuleInternal, ErrorGuaranteed> {
    // TODO make the top module if any configurable or at least an external parameter, not hardcoded here
    //   maybe we can even remove the concept entirely, by now we're elaborating all items without generics already
    let top_file = fixed.source[fixed.source.root_directory()]
        .children
        .get("top")
        .and_then(|&top_dir| fixed.source[top_dir].file)
        .ok_or_else(|| {
            let title = "no top file found, should be called `top` and be in the root directory of the project";
            diags.report(Diagnostic::new(title).finish())
        })?;
    let top_file_scope = shared.file_scopes.get(&top_file).unwrap().as_ref_ok()?;
    let top_entry = top_file_scope.find_immediate_str(diags, "top")?;

    match top_entry.value {
        &ScopedEntry::Item(item) => match &fixed.parsed[item] {
            ast::Item::ModuleInternal(module) => match &module.params {
                None => Ok(AstRefModuleInternal::new_unchecked(item, module)),
                Some(_) => {
                    Err(diags.report_simple("`top` cannot have generic parameters", module.id.span(), "defined here"))
                }
            },
            _ => Err(diags.report_simple("`top` should be a module", top_entry.defining_span, "defined here")),
        },
        ScopedEntry::Named(_) => {
            // TODO include "got" string
            // TODO is this even ever possible? direct should only be inside of scopes
            Err(diags.report_simple(
                "top should be an item, got a named value",
                top_entry.defining_span,
                "defined here",
            ))
        }
    }
}

impl CompileShared {
    pub fn new(diags: &Diagnostics, fixed: CompileFixed, queue_all_items: bool, thread_count: NonZeroUsize) -> Self {
        let CompileFixed { source, parsed } = fixed;
        let file_scopes = populate_file_scopes(diags, fixed);

        // pass over all items, to:
        // * collect all non-import items for the compute arena
        // * find all external modules
        // TODO make also skip trivial items already, eg. functions and generic modules
        let mut items = vec![];
        let mut external_modules: IndexMap<String, Vec<Span>> = IndexMap::new();
        for file in source.files() {
            if let Ok(file_ast) = &parsed[file] {
                for (item_ref, item) in file_ast.items_with_ref() {
                    if !matches!(item, ast::Item::Import(_)) {
                        items.push(item_ref);
                    }
                    if let ast::Item::ModuleExternal(module) = item {
                        external_modules
                            .entry(module.id.str(fixed.source).to_owned())
                            .or_default()
                            .push(module.id.span);
                    }
                }
            }
        }

        // check for duplicate external modules
        for (name, spans) in &external_modules {
            if spans.len() > 1 {
                let mut diag = Diagnostic::new(format!("external module with name `{name}` declared twice"));
                for &span in spans {
                    diag = diag.add_error(span, "defined here");
                }
                diags.report(diag.finish());
            }
        }
        let external_modules = external_modules.into_keys().collect();

        let item_values = ComputeOnceArena::new(items.iter().copied());

        // populate work queue
        // TODO shuffle or not?
        //   * which is actually faster? (for mutex contention)
        //   * do we want to shuffle anyway to test that the compiler is deterministic?
        let work_queue = SharedQueue::new(thread_count);
        if queue_all_items {
            items.shuffle(&mut rand::thread_rng());
            work_queue.push_batch(items.into_iter().map(WorkItem::EvaluateItem));
        }

        CompileShared {
            file_scopes,
            work_queue,
            item_values,
            elaboration_arenas: ElaborationArenas::new(),
            ir_database: Mutex::new(PartialIrDatabase {
                ir_modules: Arena::new(),
                external_modules,
            }),
        }
    }

    pub fn file_scope(&self, file: FileId) -> Result<&Scope<'static>, ErrorGuaranteed> {
        self.file_scopes.get(&file).unwrap().as_ref_ok()
    }

    pub fn finish_ir_database(
        self,
        diags: &Diagnostics,
        dummy_span: Span,
    ) -> Result<PartialIrDatabase<IrModuleInfo>, ErrorGuaranteed> {
        if self.work_queue.pop().is_some() {
            diags.report_internal_error(dummy_span, "not all work items have been processed");
        }

        let PartialIrDatabase {
            external_modules,
            ir_modules,
        } = self.ir_database.into_inner().unwrap();
        let ir_modules = ir_modules.try_map_values(|_, v| match v {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(diags.report_internal_error(dummy_span, "not all modules were elaborated")),
        })?;

        Ok(PartialIrDatabase {
            ir_modules,
            external_modules,
        })
    }
}

// TODO move somewhere else
impl CompileItemContext<'_, '_> {
    pub fn domain_signal_to_ir(&mut self, signal: Spanned<DomainSignal>) -> Result<IrExpression, ErrorGuaranteed> {
        let signal_span = signal.span;
        let Polarized { inverted, signal } = signal.inner;

        let inner = match signal {
            Signal::Port(port) => IrExpression::Port(self.ports[port].ir),
            Signal::Wire(wire) => {
                let typed = self.wires[wire].typed(self.refs, &self.wire_interfaces, signal_span)?;
                IrExpression::Wire(typed.ir)
            }
            Signal::Register(reg) => IrExpression::Register(self.registers[reg].ir),
        };
        let result = if inverted {
            self.large.push_expr(IrExpressionLarge::BoolNot(inner))
        } else {
            inner
        };
        Ok(result)
    }
}
