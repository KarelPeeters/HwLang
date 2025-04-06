use crate::constants::THREAD_STACK_SIZE;
use crate::front::block::TypedIrExpression;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrDatabase, IrExpression, IrModule, IrModuleInfo, IrModules, IrPort, IrRegister, IrWire};
use crate::front::misc::{DomainSignal, Polarized, PortDomain, ScopedEntry, Signal, ValueDomain};
use crate::front::module::{ElaboratedModule, ElaboratedModuleHeader, ModuleElaborationCacheKey};
use crate::front::scope::Scope;
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast::{self, Visibility};
use crate::syntax::ast::{Args, DomainKind, Identifier, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::syntax::parsed::{AstRefItem, AstRefModule, ParsedDatabase};
use crate::syntax::pos::{FileId, Span};
use crate::syntax::source::SourceDatabase;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::sync::{ComputeOnceArena, ComputeOnceMap, SharedQueue};
use crate::util::{ResultDoubleExt, ResultExt};
use crate::{new_index_type, throw};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Itertools};
use rand::seq::SliceRandom;
use std::num::NonZeroUsize;
use std::sync::Mutex;
// TODO keep this file for the core compile loop and multithreading, move everything else out

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
    let fixed = CompileFixed { source, parsed };

    let files = source.files();
    let file_scopes = populate_file_scopes(diags, fixed, &files);

    // TODO make also skip trivial items already, eg. functions and generic modules
    let all_items_except_imports = || {
        files.iter().flat_map(|&file: &FileId| {
            parsed[file].as_ref_ok().into_iter().flat_map(|file_ast| {
                file_ast
                    .items_with_ref()
                    .filter(|(_, item)| !matches!(item, ast::Item::Import(_)))
                    .map(|(item, _)| item)
            })
        })
    };

    // TODO shuffle or not?
    //   * which is actually faster? (for mutex contention)
    //   * do we want to shuffle anyway to test that the compiler is deterministic?
    let work_queue = SharedQueue::new(thread_count);
    {
        let mut all_items_except_imports = all_items_except_imports().collect_vec();
        all_items_except_imports.shuffle(&mut rand::thread_rng());
        work_queue.push_batch(all_items_except_imports.into_iter().map(WorkItem::EvaluateItem));
    }

    let shared = CompileShared {
        file_scopes,
        work_queue,
        item_values: ComputeOnceArena::new(all_items_except_imports()),
        elaborated_modules: ComputeOnceMap::new(),
        ir_modules: Mutex::new(Arena::default()),
        print_handler,
    };

    // get the top module
    // TODO we don't really need to do this any more, all non-generic modules are elaborated anyway
    let top_item_and_ir_module = {
        let refs = CompileRefs {
            fixed,
            shared: &shared,
            diags,
        };
        find_top_module(diags, fixed, &shared).and_then(|top_item| {
            let &ElaboratedModule { ir_module, ports: _ } = refs.elaborate_module(top_item, None)?;
            Ok((top_item, ir_module))
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
        };
        thread_refs.run_elaboration_loop();
    }

    // return result (at this point all modules should have been fully elaborated)
    let (top_item, top_ir_module) = top_item_and_ir_module?;

    let modules = shared.finish_ir_modules(diags, parsed[top_item].span)?;

    let db = IrDatabase {
        modules,
        top_module: top_ir_module,
    };
    db.validate(diags)?;
    Ok(db)
}

impl<'s> CompileRefs<'_, 's> {
    fn run_elaboration_loop(self) {
        while let Some(work_item) = self.shared.work_queue.pop() {
            match work_item {
                WorkItem::EvaluateItem(item) => {
                    self.shared.item_values.offer_to_compute(item, || {
                        let mut ctx = CompileItemContext::new(self, Some(item));
                        let result = ctx.eval_item_new(item);
                        result
                    });
                }
                WorkItem::ElaborateModule(header, ir_module) => {
                    // do elaboration
                    let ir_module_info = self.elaborate_module_body_new(header);

                    // store result
                    let slot = &mut self.shared.ir_modules.lock().unwrap()[ir_module];
                    assert!(slot.is_none());
                    *slot = Some(ir_module_info);
                }
            }
        }
    }

    pub fn elaborate_module(
        &self,
        module: AstRefModule,
        args: Option<Args<Option<Identifier>, Spanned<CompileValue>>>,
    ) -> Result<&'s ElaboratedModule, ErrorGuaranteed> {
        let shared = self.shared;

        // elaborate params
        let params = self.elaborate_module_params_new(module, args)?;
        let key = params.cache_key();

        // if necessary, elaborate header and queue body
        let elaborated = shared
            .elaborated_modules
            .get_or_compute(key, |_| {
                // elaborate ports, immediately pushing and returning error if that fails
                let (ports, header) = match self.elaborate_module_ports_new(params) {
                    Ok(p) => p,
                    Err(e) => {
                        shared.ir_modules.lock().unwrap().push(Some(Err(e)));
                        return Err(e);
                    }
                };

                // reserve ir module key, will be filled in later during body elaboration
                let ir_module = { shared.ir_modules.lock().unwrap().push(None) };

                // queue body elaboration
                self.shared
                    .work_queue
                    .push(WorkItem::ElaborateModule(header, ir_module));

                Ok(ElaboratedModule { ir_module, ports })
            })
            .as_ref_ok();

        elaborated
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
    ElaborateModule(ElaboratedModuleHeader, IrModule),
}

/// long-term shared between threads
pub struct CompileShared<'p> {
    pub file_scopes: FileScopes,

    pub work_queue: SharedQueue<WorkItem>,
    pub item_values: ComputeOnceArena<AstRefItem, Result<CompileValue, ErrorGuaranteed>, StackEntry>,
    pub elaborated_modules: ComputeOnceMap<ModuleElaborationCacheKey, Result<ElaboratedModule, ErrorGuaranteed>>,

    // TODO make this a non-blocking collection thing, could be thread-local collection and merging or a channel
    pub ir_modules: Mutex<Arena<IrModule, Option<Result<IrModuleInfo, ErrorGuaranteed>>>>,

    // TODO is there a reasonable way to get deterministic prints?
    // TODO move this into refs, what is this doing here? Shared should be usable as a long-term value type, it can't have lifetime params
    pub print_handler: &'p (dyn PrintHandler + Sync),
}

pub type FileScopes = IndexMap<FileId, Result<Scope<'static>, ErrorGuaranteed>>;

pub struct CompileSharedModules {
    pub elaborations: IndexMap<ModuleElaborationCacheKey, Result<ElaboratedModule, ErrorGuaranteed>>,
    pub ir_modules: Arena<IrModule, Option<Result<IrModuleInfo, ErrorGuaranteed>>>,
}

#[derive(Copy, Clone)]
pub struct CompileRefs<'a, 's> {
    // TODO maybe inline this
    pub fixed: CompileFixed<'a>,
    pub shared: &'s CompileShared<'a>,
    pub diags: &'a Diagnostics,
}

pub struct CompileItemContext<'a, 's> {
    // TODO maybe inline this
    pub refs: CompileRefs<'a, 's>,

    pub variables: Arena<Variable, VariableInfo>,
    pub ports: Arena<Port, PortInfo<Port>>,
    pub wires: Arena<Wire, WireInfo>,
    pub registers: Arena<Register, RegisterInfo>,

    pub origin: Option<AstRefItem>,
    pub stack: Vec<StackEntry>,
}

#[derive(Debug, Copy, Clone)]
pub enum StackEntry {
    ItemUsage(Span),
    ItemEvaluation(AstRefItem),
    FunctionCall(Span),
    FunctionRun(AstRefItem),
}

impl StackEntry {
    pub fn into_span_message(self, parsed: &ParsedDatabase) -> (Span, &'static str) {
        match self {
            StackEntry::ItemUsage(span) => (span, "item used here"),
            StackEntry::ItemEvaluation(entry_item) => {
                let entry_span = parsed[entry_item].common_info().span_short;
                (entry_span, "item declared here")
            }
            StackEntry::FunctionCall(entry_span) => (entry_span, "function call here"),
            StackEntry::FunctionRun(entry_item) => {
                let entry_span = parsed[entry_item].common_info().span_short;
                (entry_span, "function declared here")
            }
        }
    }
}

impl<'a, 's> CompileItemContext<'a, 's> {
    pub fn new(refs: CompileRefs<'a, 's>, origin: Option<AstRefItem>) -> Self {
        CompileItemContext {
            refs,
            variables: Arena::default(),
            ports: Arena::default(),
            wires: Arena::default(),
            registers: Arena::default(),
            origin,
            stack: vec![],
        }
    }

    pub fn recurse<R>(&mut self, entry: StackEntry, f: impl FnOnce(&mut Self) -> R) -> R {
        self.stack.push(entry);
        let len = self.stack.len();

        let result = f(self);

        assert_eq!(self.stack.len(), len);
        self.stack.pop().unwrap();

        result
    }

    pub fn eval_item(&mut self, item: AstRefItem) -> Result<&CompileValue, ErrorGuaranteed> {
        self.recurse(StackEntry::ItemEvaluation(item), |s| {
            let origin = s.origin.map(|origin| (origin, s.stack.clone()));
            let f_compute = || {
                let mut ctx = CompileItemContext::new(s.refs, Some(item));
                ctx.eval_item_new(item)
            };
            let f_cycle = |stack: Vec<&StackEntry>| s.refs.diags.report(cycle_diagnostic(s.refs.fixed.parsed, stack));
            s.refs
                .shared
                .item_values
                .get_or_compute(origin, item, f_compute, f_cycle)
                .map(ResultExt::as_ref_ok)
                .flatten_err()
        })
    }
}

fn cycle_diagnostic(parsed: &ParsedDatabase, mut stack: Vec<&StackEntry>) -> Diagnostic {
    // sort the stack to keep error messages deterministic
    assert!(!stack.is_empty());
    let min_index = stack
        .iter()
        .position_min_by_key(|entry| {
            let (span, _) = entry.into_span_message(parsed);
            span
        })
        .unwrap();
    stack.rotate_left(min_index);

    // create the diagnostic
    let mut diag = Diagnostic::new("encountered cyclic dependency");
    for (entry_index, &entry) in enumerate(stack) {
        let (span, label) = entry.into_span_message(parsed);
        diag = diag.add_error(span, format!("[{entry_index}] {label}"));
    }
    diag.finish()
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct StackNotEq(usize);

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CompileStackEntry {
    Item(AstRefItem),
    FunctionCall(Span, StackNotEq),
    FunctionRun(AstRefItem, Vec<MaybeCompile<TypedIrExpression>>),
}

// TODO move these somewhere else
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
pub struct PortInfo<P> {
    pub id: Identifier,
    pub direction: Spanned<PortDirection>,
    pub domain: Spanned<PortDomain<P>>,
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

impl PortInfo<Port> {
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

fn populate_file_scopes(diags: &Diagnostics, fixed: CompileFixed, files: &[FileId]) -> FileScopes {
    let CompileFixed { source, parsed } = fixed;

    // pass 0: add all items to their own file scope
    let mut file_scopes: FileScopes = IndexMap::new();
    for &file in files {
        let scope = parsed[file].as_ref_ok().map(|ast| {
            let mut scope = Scope::new_root(ast.span);
            for (ast_item_ref, ast_item) in ast.items_with_ref() {
                if let Some(declaration_info) = ast_item.declaration_info() {
                    scope.maybe_declare(diags, declaration_info.id, Ok(ScopedEntry::Item(ast_item_ref)));
                }
            }
            scope
        });

        file_scopes.insert_first(file, scope);
    }

    // pass 1: collect import items from source scopes for each target file
    let mut file_imported_items: Vec<Vec<(MaybeIdentifier<&Identifier>, Result<ScopedEntry, ErrorGuaranteed>)>> =
        vec![];
    for &target_file in files {
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
                        let ast::ImportEntry { span: _, id, as_ } = entry;
                        let source_value = source_scope
                            .and_then(|source_scope| source_scope.find(diags, id))
                            .map(|found| found.value.clone());

                        // check visibility, but still proceed as if the import succeeded
                        if let Ok(ScopedEntry::Item(source_item)) = source_value {
                            let decl_info = parsed[source_item].declaration_info().unwrap();
                            match decl_info.vis {
                                Visibility::Public(_) => {}
                                Visibility::Private => {
                                    let err = Diagnostic::new(format!("cannot access identifier `{}`", id.string))
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

                        let target_id: MaybeIdentifier<&Identifier> = match as_ {
                            Some(as_) => as_.as_ref(),
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
    for (&target_file, items) in zip_eq(files, file_imported_items) {
        if let Ok(scope) = file_scopes.get_mut(&target_file).unwrap() {
            for (target_id, value) in items {
                scope.maybe_declare(diags, target_id, value);
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

    source[curr_dir]
        .file
        .ok_or_else(|| diags.report_simple("expected path to file", parents_span, "no file exists at this path"))
}

fn find_top_module(
    diags: &Diagnostics,
    fixed: CompileFixed,
    shared: &CompileShared,
) -> Result<AstRefModule, ErrorGuaranteed> {
    // TODO make the top module if any configurable or at least an external parameter, not hardcoded here
    //   maybe we can even remove the concept entirely, by now we're elaborating all items without generics already
    let top_file = fixed.source[fixed.source.root_directory]
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
    pub fn file_scope(&self, file: FileId) -> Result<&Scope<'static>, ErrorGuaranteed> {
        self.file_scopes.get(&file).unwrap().as_ref_ok()
    }

    pub fn finish_ir_modules(self, diags: &Diagnostics, dummy_span: Span) -> Result<IrModules, ErrorGuaranteed> {
        let ir_modules = self.ir_modules.into_inner().unwrap();
        ir_modules.try_map_values(|v| match v {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(diags.report_internal_error(dummy_span, "not all modules were elaborated")),
        })
    }
}

// TODO move somewhere else
impl CompileItemContext<'_, '_> {
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

// TODO rename/expand to handle all external interactions: IO, env vars, ...
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

pub struct CollectPrintHandler(Mutex<Vec<String>>);

impl CollectPrintHandler {
    pub fn new() -> Self {
        CollectPrintHandler(Mutex::new(Vec::new()))
    }

    pub fn finish(self) -> Vec<String> {
        self.0.into_inner().unwrap()
    }
}

impl PrintHandler for CollectPrintHandler {
    fn println(&self, s: &str) {
        self.0.lock().unwrap().push(s.to_string());
    }
}
