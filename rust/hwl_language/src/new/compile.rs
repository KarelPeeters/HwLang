use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::data::parsed::{AstRefItem, AstRefModule, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::scope::{Scope, ScopeInfo, Scopes, Visibility};
use crate::new::ir::{IrDesign, IrModule, IrModuleInfo};
use crate::new::misc::{DomainSignal, PortDomain, ScopedEntry};
use crate::new::types::{HardwareType, Type};
use crate::new::value::CompileValue;
use crate::syntax::ast;
use crate::syntax::ast::{Args, DomainKind, Identifier, PortDirection, Spanned, SyncDomain};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::{Arena, ArenaSet};
use crate::util::data::IndexMapExt;
use crate::util::{ResultDoubleExt, ResultExt};
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
pub fn compile(diags: &Diagnostics, source: &SourceDatabase, parsed: &ParsedDatabase) -> IrDesign {
    // populate file scopes
    let mut map_file_scopes = IndexMap::new();
    let mut scopes = Scopes::default();

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
                    local_scope_info.maybe_declare(diags, declaration_info.id, Ok(ScopedEntry::Item(ast_item_ref)), vis);
                }

                match ast_item {
                    ast::Item::Import(_) => {}
                    _ => all_items_except_imports.push(ast_item_ref),
                }
            }

            FileScopes { scope_outer_declare: scope_declare, scope_inner_import: scope_import }
        });

        map_file_scopes.insert_first(file, scope);
    }

    // populate import scopes
    for &file in &files {
        if let Ok(file_scopes) = map_file_scopes.get(&file).as_ref().unwrap() {
            let file_ast = parsed[file].as_ref_ok().unwrap();
            for item in &file_ast.items {
                if let ast::Item::Import(item) = item {
                    add_import_to_scope(diags, &source, &mut scopes, &map_file_scopes, file_scopes.scope_inner_import, item);
                }
            }
        }
    }

    // group into state
    let mut state = CompileState {
        diags,
        source,
        parsed,
        scopes,
        file_scopes: map_file_scopes,
        constants: Arena::default(),
        parameters: Arena::default(),
        registers: Arena::default(),
        ports: Arena::default(),
        wires: Arena::default(),
        variables: Arena::default(),
        ir_modules: Arena::default(),
        elaborated_modules: ArenaSet::default(),
        elaborated_modules_to_ir: IndexMap::new(),
        elaboration_stack: vec![],
        items: IndexMap::default(),
    };

    // visit all items, possibly using them as an elaboration starting point
    for item in all_items_except_imports {
        let _ = state.eval_item_as_ty_or_value(item);
    }

    // get the top module (it will always hit the elaboration cache)
    let top_module = state.find_top_module()
        .and_then(|top_item| {
            let elaboration = state.elaborated_modules.push(ModuleElaborationInfo { item: top_item, args: None });
            state.elaborate_module(elaboration)
        });

    // return result
    assert!(state.elaboration_stack.is_empty());
    IrDesign {
        top_module,
        modules: state.ir_modules,
    }
}

pub struct CompileState<'a> {
    pub diags: &'a Diagnostics,
    pub source: &'a SourceDatabase,
    pub parsed: &'a ParsedDatabase,

    pub scopes: Scopes,
    file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,

    pub constants: Arena<Constant, ConstantInfo>,
    pub parameters: Arena<Parameter, ConstantInfo>,
    pub variables: Arena<Variable, VariableInfo>,
    pub ports: Arena<Port, PortInfo>,
    pub wires: Arena<Wire, WireInfo>,
    pub registers: Arena<Register, RegisterInfo>,

    pub ir_modules: Arena<IrModule, IrModuleInfo>,
    pub elaborated_modules: ArenaSet<ModuleElaboration, ModuleElaborationInfo>,
    pub elaborated_modules_to_ir: IndexMap<ModuleElaboration, Result<IrModule, ErrorGuaranteed>>,

    pub items: IndexMap<AstRefItem, Result<CompileValue, ErrorGuaranteed>>,

    elaboration_stack: Vec<ElaborationStackEntry>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ElaborationStackEntry {
    // TODO properly refer to ast instantiation site
    /// Module instantiation is only here to provide nicer error messages.
    ModuleInstantiation(Span),
    ModuleElaboration(ModuleElaboration),
    // TODO better name, is this ItemEvaluation, ItemSignature, ...?
    Item(AstRefItem),
    // TODO better names
    FunctionCall(Span, Args<CompileValue>),
    FunctionRun(AstRefItem, Args<CompileValue>),
}

new_index_type!(pub ModuleElaboration);
new_index_type!(pub Constant);
new_index_type!(pub Parameter);
new_index_type!(pub Variable);
new_index_type!(pub Port);
new_index_type!(pub Wire);
new_index_type!(pub Register);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleElaborationInfo {
    pub item: AstRefModule,
    pub args: Option<Vec<CompileValue>>,
}

#[derive(Debug)]
pub struct ConstantInfo {
    pub def_id_span: Span,
    pub value: CompileValue,
}

#[derive(Debug)]
pub struct VariableInfo {
    pub def_id_span: Span,
    pub ty: Type,
}

#[derive(Debug)]
pub struct PortInfo {
    pub def_id_span: Span,
    pub direction: Spanned<PortDirection>,
    pub domain: Spanned<PortDomain<DomainSignal>>,
    pub ty: Spanned<HardwareType>,
}

#[derive(Debug)]
pub struct WireInfo {
    pub def_id_span: Span,
    pub domain: Spanned<DomainKind<DomainSignal>>,
    pub ty: Spanned<HardwareType>,
}

#[derive(Debug)]
pub struct RegisterInfo {
    pub def_id_span: Span,
    pub domain: SyncDomain<DomainSignal>,
    pub ty: Spanned<HardwareType>,
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

    let ast::ItemImport { span: _, parents, entry } = item;

    let parent_scope = find_parent_scope(diags, source, file_scopes, parents);

    let import_entries = match &entry.inner {
        ast::ImportFinalKind::Single(entry) => std::slice::from_ref(entry),
        ast::ImportFinalKind::Multi(entries) => entries,
    };

    for import_entry in import_entries {
        let ast::ImportEntry { span: _, id, as_ } = import_entry;

        // TODO allow private visibility into child scopes?
        let entry = parent_scope.and_then(|parent_scope| {
            scopes[parent_scope].find(&scopes, diags, id, Visibility::Public)
                .map(|entry| entry.value.clone())
        }
        );

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
        parents.inner.first().unwrap().span.join(parents.inner.last().unwrap().span)
    };

    for step in &parents.inner {
        let curr_dir_info = &source[curr_dir];

        curr_dir = match curr_dir_info.children.get(&step.string) {
            Some(&child_dir) => child_dir,
            None => {
                let mut options = curr_dir_info.children.keys().cloned().collect_vec();
                options.sort();

                // TODO without trailing separator
                let diag = Diagnostic::new("invalid path step")
                    .snippet(parents.span)
                    .add_error(step.span, "invalid step")
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

    file_scopes.get(&file).unwrap().as_ref_ok()
        .map(|scopes| scopes.scope_outer_declare)
}

impl CompileState<'_> {
    // TODO add stack limit
    pub fn check_compile_loop<T>(&mut self, entry: ElaborationStackEntry, f: impl FnOnce(&mut Self) -> T) -> Result<T, ErrorGuaranteed> {
        if let Some(loop_start) = self.elaboration_stack.iter().position(|x| x == &entry) {
            // report elaboration loop
            let cycle = &self.elaboration_stack[loop_start..];

            let mut diag = Diagnostic::new("encountered elaboration cycle");
            for (i, elem) in enumerate(cycle) {
                match elem {
                    &ElaborationStackEntry::ModuleInstantiation(span) => {
                        // TODO include generic args
                        diag = diag.add_error(span, format!("({i}): module instantiation"));
                    }
                    &ElaborationStackEntry::ModuleElaboration(elem) => {
                        let item = self.elaborated_modules[elem].item;
                        diag = diag.add_error(self.parsed[item].id.span, format!("({i}): module"));
                    }
                    &ElaborationStackEntry::Item(item) => {
                        diag = diag.add_error(self.parsed[item].common_info().span_short, format!("({i}): item"));
                    }
                    &ElaborationStackEntry::FunctionCall(expr_span, _) => {
                        // TODO include args
                        diag = diag.add_error(expr_span, format!("({i}): function call"));
                    }
                    &ElaborationStackEntry::FunctionRun(item, _) => {
                        // TODO include args
                        diag = diag.add_error(self.parsed[item].common_info().span_short, format!("({i}): function run"));
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

    pub fn elaborate_module(&mut self, module_elaboration: ModuleElaboration) -> Result<IrModule, ErrorGuaranteed> {
        // check cache
        if let Some(&result) = self.elaborated_modules_to_ir.get(&module_elaboration) {
            return result;
        }

        // new elaboration
        let ir_module = self.check_compile_loop(ElaborationStackEntry::ModuleElaboration(module_elaboration), |s| {
            s.elaborate_module_new(module_elaboration)
                .map(|ir_content| s.ir_modules.push(ir_content))
        }).flatten_err();

        // put into cache and return
        // this is correct even for errors caused by cycles:
        //   _every_ item in the cycle would always trigger the cycle, so we can mark all of them as errors
        self.elaborated_modules_to_ir.insert_first(module_elaboration, ir_module);

        ir_module
    }

    fn find_top_module(&self) -> Result<AstRefModule, ErrorGuaranteed> {
        let diags = self.diags;

        let top_file = self.source[self.source.root_directory].children.get("top")
            .and_then(|&top_dir| self.source[top_dir].file)
            .ok_or_else(|| {
                let title = "no top file found, should be called `top` and be in the root directory of the project";
                diags.report(Diagnostic::new(title).finish())
            })?;
        let top_file_scope = self.file_scopes.get(&top_file).unwrap().as_ref_ok()?.scope_outer_declare;
        let top_entry = self[top_file_scope].find_immediate_str(diags, "top", Visibility::Public)?;

        match top_entry.value {
            &ScopedEntry::Item(item) => {
                match &self.parsed[item] {
                    ast::Item::Module(module) => {
                        match &module.params {
                            None => Ok(AstRefModule::new_unchecked(item)),
                            Some(_) => {
                                Err(diags.report_simple(
                                    "`top` cannot have generic parameters",
                                    module.id.span,
                                    "defined here",
                                ))
                            }
                        }
                    }
                    _ => {
                        Err(diags.report_simple(
                            "`top` should be a module",
                            top_entry.defining_span,
                            "defined here",
                        ))
                    }
                }
            }
            ScopedEntry::Direct(_) => {
                // TODO include "got" string
                // TODO is this even ever possible? direct should only be inside of scopes
                Err(diags.report_simple(
                    "top should be an item, got a direct",
                    top_entry.defining_span,
                    "defined here",
                ))
            }
        }
    }

    pub fn file_scope(&self, file: FileId) -> Result<Scope, ErrorGuaranteed> {
        match self.file_scopes.get(&file) {
            None => Err(self.diags.report_internal_error(self.source[file].offsets.full_span(file), "file scopes not found")),
            Some(Ok(scopes)) => Ok(scopes.scope_inner_import),
            Some(&Err(e)) => Err(e),
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

macro_rules! impl_index {
    ($arena:ident, $index:ty, $info:ty) => {
        impl std::ops::Index<$index> for CompileState<'_> {
            type Output = $info;
            fn index(&self, index: $index) -> &Self::Output {
                &self.$arena[index]
            }
        }
    };
}

impl_index!(scopes, Scope, ScopeInfo);
