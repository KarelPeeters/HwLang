use std::collections::VecDeque;

use crate::front::block::TypedIrExpression;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrExpression, IrModule, IrModuleInfo, IrPort, IrRegister, IrWire};
use crate::front::misc::{DomainSignal, Polarized, PortDomain, ScopedEntry, Signal, ValueDomain};
use crate::front::scope::{Scope, Scopes, Visibility};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, MaybeCompile};
use crate::syntax::ast;
use crate::syntax::ast::{Args, DomainKind, Identifier, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::syntax::parsed::{AstRefItem, AstRefModule, ParsedDatabase};
use crate::syntax::pos::{FileId, Span};
use crate::syntax::source::SourceDatabase;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use crate::{new_index_type, throw};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};

use super::ir::{IrDatabase, IrModules};
use super::module::{ElaboratedModule, ElaboratedModuleHeader, ModuleElaborationCacheKey};

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
    print_handler: &mut (dyn PrintHandler),
) -> Result<IrDatabase, ErrorGuaranteed> {
    let PopulatedScopes {
        scopes,
        file_scopes,
        all_items_except_imports,
    } = PopulatedScopes::new(diags, source, parsed);

    let mut state_long = CompileStateLong::new(scopes, file_scopes);
    let mut state = CompileState::new(diags, source, parsed, &mut state_long, print_handler);

    // visit all items, possibly using them as an elaboration starting point
    for item in all_items_except_imports {
        let _ = state.eval_item(item);
    }
    // get the top module
    let (top_module, top_item) = state.find_top_module().and_then(|top_item| {
        let elaborated = state.elaborate_module(top_item, None)?;
        Ok((elaborated.ir_module, top_item))
    })?;

    // run until everything is elaborated
    state.run_elaboration_loop();

    // return result (at this point all modules should have been fully elaborated)
    assert!(state.elaboration_stack.is_empty());
    let modules = state_long.finish_ir_modules(diags, parsed[top_item].span)?;

    let db = IrDatabase { top_module, modules };
    db.validate(diags)?;
    Ok(db)
}

// TODO split into per-file and longer term state, so we can parralize the per-module stuff
//   file scopes, constants, items and module elaboration details are global and long term,
//   but variables, wires, ports, registers, ... are all only relevant within modules
pub struct CompileStateLong {
    pub scopes: Scopes,
    pub file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,

    pub variables: Arena<Variable, VariableInfo>,
    pub ports: Arena<Port, PortInfo>,
    pub wires: Arena<Wire, WireInfo>,
    pub registers: Arena<Register, RegisterInfo>,

    // contains None for placeholder modules that have not yet had their bodies elaborated
    pub ir_modules: Arena<IrModule, Option<Result<IrModuleInfo, ErrorGuaranteed>>>,
    pub elaborated_modules_cache: IndexMap<ModuleElaborationCacheKey, Result<ElaboratedModule, ErrorGuaranteed>>,
    pub elaborated_modules_todo: VecDeque<(IrModule, ElaboratedModuleHeader)>,

    pub items: IndexMap<AstRefItem, Result<CompileValue, ErrorGuaranteed>>,
}

pub struct CompileState<'a> {
    pub diags: &'a Diagnostics,
    pub source: &'a SourceDatabase,
    pub parsed: &'a ParsedDatabase,

    pub state: &'a mut CompileStateLong,
    pub print_handler: &'a mut dyn PrintHandler,

    elaboration_stack: Vec<ElaborationStackEntry>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct StackNotEq(usize);

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ElaborationStackEntry {
    // TODO properly refer to ast instantiation site
    /// Module instantiation is only here to provide nicer error messages.
    ModuleInstantiation(Span, StackNotEq),
    ModuleElaboration(ModuleElaborationCacheKey),
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
    scopes: &mut Scopes,
    file_scopes: &IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    target_scope: Scope,
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

    let parent_scope = find_parent_scope(diags, source, file_scopes, parents);

    let import_entries = match &entry.inner {
        ast::ImportFinalKind::Single(entry) => std::slice::from_ref(entry),
        ast::ImportFinalKind::Multi(entries) => entries,
    };

    for import_entry in import_entries {
        let ast::ImportEntry { span: _, id, as_ } = import_entry;

        // TODO allow private visibility into child scopes?
        let entry = parent_scope.and_then(|parent_scope| {
            scopes[parent_scope]
                .find(scopes, diags, id, Visibility::Public)
                .map(|entry| entry.value.clone())
        });

        let target_scope = &mut scopes[target_scope];
        match as_ {
            Some(as_) => target_scope.maybe_declare(diags, as_.as_ref(), entry, Visibility::Private),
            None => target_scope.declare(diags, id, entry, Visibility::Private),
        };
    }
}

fn find_parent_scope(
    diags: &Diagnostics,
    source: &SourceDatabase,
    file_scopes: &IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    parents: &Spanned<Vec<Identifier>>,
) -> Result<Scope, ErrorGuaranteed> {
    // TODO the current path design does not allow private sub-modules
    //   are they really necessary? if all inner items are private it's effectively equivalent
    //   -> no it's not equivalent, things can also be private from the parent
    let mut curr_dir = source.root_directory;

    // get the span without the trailing separator
    let parents_span = if parents.inner.is_empty() {
        parents.span
    } else {
        parents
            .inner
            .first()
            .unwrap()
            .span
            .join(parents.inner.last().unwrap().span)
    };

    for step in &parents.inner {
        let curr_dir_info = &source[curr_dir];

        curr_dir = match curr_dir_info.children.get(&step.string) {
            Some(&child_dir) => child_dir,
            None => {
                let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                options.sort();

                // TODO without trailing separator
                let diag = Diagnostic::new("import not found")
                    .snippet(parents.span)
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

pub struct PopulatedScopes {
    pub scopes: Scopes,
    pub file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    // TODO maybe imports should not be items in the first place?
    pub all_items_except_imports: Vec<AstRefItem>,
}

impl PopulatedScopes {
    pub fn new(diags: &Diagnostics, source: &SourceDatabase, parsed: &ParsedDatabase) -> Self {
        // populate file scopes
        let mut map_file_scopes = IndexMap::new();
        let mut scopes = Scopes::new();

        let files = source.files();
        let mut all_items_except_imports = vec![];

        for &file in &files {
            let file_source = &source[file];

            let scope = parsed[file].as_ref_ok().map(|ast| {
                // build declaration scope
                // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
                let file_span = file_source.offsets.full_span(file);
                let scope_declare = scopes.new_root(file_span);
                let scope_import = scopes.new_child(scope_declare, file_span, Visibility::Private);

                let local_scope_info = &mut scopes[scope_declare];

                for (ast_item_ref, ast_item) in ast.items_with_ref() {
                    if let Some(declaration_info) = ast_item.declaration_info() {
                        let vis = match declaration_info.vis {
                            ast::Visibility::Public(_) => Visibility::Public,
                            ast::Visibility::Private => Visibility::Private,
                        };
                        local_scope_info.maybe_declare(
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

            map_file_scopes.insert_first(file, scope);
        }

        // populate import scopes
        for &file in &files {
            if let Ok(file_scopes) = map_file_scopes.get(&file).as_ref().unwrap() {
                let file_ast = parsed[file].as_ref_ok().unwrap();
                for item in &file_ast.items {
                    if let ast::Item::Import(item) = item {
                        add_import_to_scope(
                            diags,
                            source,
                            &mut scopes,
                            &map_file_scopes,
                            file_scopes.scope_inner_import,
                            item,
                        );
                    }
                }
            }
        }

        PopulatedScopes {
            scopes,
            file_scopes: map_file_scopes,
            all_items_except_imports,
        }
    }
}

impl CompileStateLong {
    pub fn new(scopes: Scopes, file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>) -> Self {
        CompileStateLong {
            scopes,
            file_scopes,
            registers: Arena::default(),
            ports: Arena::default(),
            wires: Arena::default(),
            variables: Arena::default(),
            ir_modules: Arena::default(),
            elaborated_modules_cache: IndexMap::new(),
            elaborated_modules_todo: VecDeque::new(),
            items: IndexMap::default(),
        }
    }

    pub fn finish_ir_modules(self, diags: &Diagnostics, dummy_span: Span) -> Result<IrModules, ErrorGuaranteed> {
        self.ir_modules.try_map_values(|v| match v {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(e),
            None => Err(diags.report_internal_error(dummy_span, "not all modules were elaborated")),
        })
    }
}

impl<'a> CompileState<'a> {
    pub fn new(
        diags: &'a Diagnostics,
        source: &'a SourceDatabase,
        parsed: &'a ParsedDatabase,
        state: &'a mut CompileStateLong,
        print_handler: &'a mut dyn PrintHandler,
    ) -> Self {
        CompileState {
            diags,
            source,
            parsed,
            state,
            print_handler,
            elaboration_stack: vec![],
        }
    }

    pub fn not_eq_stack(&self) -> StackNotEq {
        StackNotEq(self.elaboration_stack.len())
    }

    // TODO add stack limit
    pub fn check_compile_loop<T>(
        &mut self,
        entry: ElaborationStackEntry,
        f: impl FnOnce(&mut Self) -> T,
    ) -> Result<T, ErrorGuaranteed> {
        if let Some(loop_start) = self.elaboration_stack.iter().position(|x| x == &entry) {
            // report elaboration loop
            let cycle = &self.elaboration_stack[loop_start..];

            let mut diag = Diagnostic::new("encountered elaboration cycle");
            for (i, elem) in enumerate(cycle) {
                match elem {
                    &ElaborationStackEntry::ModuleInstantiation(span, _not_eq) => {
                        // TODO include generic args
                        diag = diag.add_error(span, format!("({i}): module instantiation"));
                    }
                    ElaborationStackEntry::ModuleElaboration(elem) => {
                        diag = diag.add_error(self.parsed[elem.module].id.span, format!("({i}): module"));
                    }
                    &ElaborationStackEntry::Item(item) => {
                        diag = diag.add_error(self.parsed[item].common_info().span_short, format!("({i}): item"));
                    }
                    &ElaborationStackEntry::FunctionCall(expr_span, _not_eq) => {
                        diag = diag.add_error(expr_span, format!("({i}): function call"));
                    }
                    &ElaborationStackEntry::FunctionRun(item, _) => {
                        // TODO include args
                        diag = diag.add_error(
                            self.parsed[item].common_info().span_short,
                            format!("({i}): function run"),
                        );
                    }
                }
            }

            Err(self.diags.report(diag.finish()))
        } else {
            self.elaboration_stack.push(entry);
            let len_expected = self.elaboration_stack.len();

            let result = f(self);

            assert_eq!(self.elaboration_stack.len(), len_expected);
            self.elaboration_stack.pop().unwrap();

            Ok(result)
        }
    }

    pub fn run_elaboration_loop(&mut self) {
        while let Some((ir_module, header)) = self.state.elaborated_modules_todo.pop_front() {
            let info = self.elaborate_module_body_new(header);

            let slot = &mut self.state.ir_modules[ir_module];
            assert!(slot.is_none());
            *slot = Some(info);
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
        if let Some(result) = self.state.elaborated_modules_cache.get(&cache_key) {
            return result.clone();
        }

        // new elaboration
        let header = self.elaborate_module_ports_new(params);
        match header {
            Ok(header) => {
                // reserve placeholder ir module, will be filled in later
                let ir_module: IrModule = self.state.ir_modules.push(None);
                let result = ElaboratedModule {
                    ir_module,
                    ports: header.ports.clone(),
                };
                self.state.elaborated_modules_todo.push_back((ir_module, header));
                Ok(result)
            }
            Err(e) => {
                self.state.elaborated_modules_cache.insert_first(cache_key, Err(e));
                Err(e)
            }
        }
    }

    fn find_top_module(&self) -> Result<AstRefModule, ErrorGuaranteed> {
        let diags = self.diags;

        let top_file = self.source[self.source.root_directory]
            .children
            .get("top")
            .and_then(|&top_dir| self.source[top_dir].file)
            .ok_or_else(|| {
                let title = "no top file found, should be called `top` and be in the root directory of the project";
                diags.report(Diagnostic::new(title).finish())
            })?;
        let top_file_scope = self
            .state
            .file_scopes
            .get(&top_file)
            .unwrap()
            .as_ref_ok()?
            .scope_outer_declare;
        let top_entry = self.state.scopes[top_file_scope].find_immediate_str(diags, "top", Visibility::Public)?;

        match top_entry.value {
            &ScopedEntry::Item(item) => match &self.parsed[item] {
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

    pub fn file_scope(&self, file: FileId) -> Result<Scope, ErrorGuaranteed> {
        match self.state.file_scopes.get(&file) {
            None => Err(self
                .diags
                .report_internal_error(self.source[file].offsets.full_span(file), "file scopes not found")),
            Some(Ok(scopes)) => Ok(scopes.scope_inner_import),
            Some(&Err(e)) => Err(e),
        }
    }

    pub fn domain_signal_to_ir(&self, signal: &DomainSignal) -> IrExpression {
        let &Polarized { inverted, signal } = signal;
        let inner = match signal {
            Signal::Port(port) => IrExpression::Port(self.state.ports[port].ir),
            Signal::Wire(wire) => IrExpression::Wire(self.state.wires[wire].ir),
            Signal::Register(reg) => IrExpression::Register(self.state.registers[reg].ir),
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
    pub scope_outer_declare: Scope,
    /// Child scope of [scope_outer_declare] that includes all imported items.
    pub scope_inner_import: Scope,
}

// TODO rename/expand to handle all external interation: IO, env vars, ...
pub trait PrintHandler {
    fn println(&mut self, s: &str);
}

pub struct NoPrintHandler;

impl PrintHandler for NoPrintHandler {
    fn println(&mut self, _: &str) {}
}

pub struct StdoutPrintHandler;

impl PrintHandler for StdoutPrintHandler {
    fn println(&mut self, s: &str) {
        println!("{}", s);
    }
}

pub struct CollectPrintHandler(Vec<String>);

impl CollectPrintHandler {
    pub fn new() -> Self {
        CollectPrintHandler(Vec::new())
    }

    pub fn finish(self) -> Vec<String> {
        self.0
    }
}

impl PrintHandler for CollectPrintHandler {
    fn println(&mut self, s: &str) {
        self.0.push(s.to_owned());
    }
}
