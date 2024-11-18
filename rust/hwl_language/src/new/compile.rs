use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::data::parsed::{AstRefModule, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::scope::{Scope, ScopeInfo, Scopes, Visibility};
use crate::new::ir::{IrDesign, IrModule, IrModuleContent};
use crate::new::misc::{MaybeUnchecked, ScopedEntry, TypeOrValue, Unchecked};
use crate::new::types::{RangeInfo, Type};
use crate::new::value::{KnownCompileValue, Value};
use crate::syntax::ast;
use crate::syntax::ast::{GenericParameterKind, Identifier, Spanned};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::{Arena, ArenaSet};
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use crate::{new_index_type, throw};
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{zip_eq, Itertools};
use num_bigint::BigUint;

pub struct CompileState<'a> {
    diags: &'a Diagnostics,
    source: &'a SourceDatabase,
    parsed: &'a ParsedDatabase,

    scopes: Scopes<ScopedEntry>,
    file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,

    ir_modules: Arena<IrModule, IrModuleContent>,
    elaborated_modules: ArenaSet<ModuleElaboration, ModuleElaborationInfo>,
    elaborated_modules_to_ir: IndexMap<ModuleElaboration, Result<IrModule, ErrorGuaranteed>>,

    elaboration_stack: Vec<ModuleElaboration>,
}

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
                // TODO add enum-match safety here
                if let Some(declaration_info) = ast_item.declaration_info() {
                    let vis = match declaration_info.vis {
                        ast::Visibility::Public(_) => Visibility::Public,
                        ast::Visibility::Private => Visibility::Private,
                    };
                    local_scope_info.maybe_declare(diags, declaration_info.id, ScopedEntry::Item(ast_item_ref), vis);
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
        elaboration_stack: vec![],
        ir_modules: Arena::default(),
        elaborated_modules: ArenaSet::default(),
        elaborated_modules_to_ir: IndexMap::new(),
    };

    // start elaborating from the top module
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

new_index_type!(pub ModuleElaboration);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleElaborationInfo {
    pub item: AstRefModule,
    pub args: Option<Vec<TypeOrValue<KnownCompileValue>>>,
}

fn add_import_to_scope(
    diags: &Diagnostics,
    source: &SourceDatabase,
    scopes: &mut Scopes<ScopedEntry>,
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
        let entry = match parent_scope {
            Ok(parent_scope) => scopes[parent_scope].find(&scopes, diags, id, Visibility::Public),
            Err(e) => Err(e),
        }
            .map(|entry| entry.value.clone())
            .unwrap_or_else(|e| ScopedEntry::Direct(TypeOrValue::Error(e)));

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
    fn elaborate_module(&mut self, module_elaboration: ModuleElaboration) -> Result<IrModule, ErrorGuaranteed> {
        let diags = self.diags;

        // check cache
        if let Some(&result) = self.elaborated_modules_to_ir.get(&module_elaboration) {
            return result;
        }

        let ir_module = if let Some(loop_start) = self.elaboration_stack.iter().position(|&x| x == module_elaboration) {
            // report elaboration loop
            let cycle = &self.elaboration_stack[loop_start..];

            // TODO include instantiation site in error message
            let mut diag = Diagnostic::new("encountered elaboration cycle");
            for &elem in cycle {
                let item = self.elaborated_modules[elem].item;
                diag = diag.add_error(self.parsed[item].id.span, "module part of elaboration cycle");
            }

            Err(diags.report(diag.finish()))
        } else {
            // elaborate new module
            self.elaboration_stack.push(module_elaboration);
            let elaboration_info = self.elaborated_modules[module_elaboration].clone();
            self.elaborate_module_new(elaboration_info)
                .map(|ir_content| self.ir_modules.push(ir_content))
        };

        // put into cache and return
        self.elaborated_modules_to_ir.insert_first(module_elaboration, ir_module);
        ir_module
    }

    // TODO clarify that argument type checking has already happened at the call site?
    fn elaborate_module_new(&mut self, module_elaboration: ModuleElaborationInfo) -> Result<IrModuleContent, ErrorGuaranteed> {
        // elaborate new module
        let diags = self.diags;
        let ModuleElaborationInfo { item, args } = module_elaboration;

        let file_scope = self[item.file()].as_ref_ok()?.scope_inner_import;
        let &ast::ItemDefModule { span: def_span, vis: _, id: _, ref params, ref ports, ref body } = &self.parsed[item];

        // check params and add to scope
        let params_scope = self.scopes.new_child(file_scope, def_span, Visibility::Private);
        match (params, args) {
            (None, None) => {}
            (Some(params), Some(args)) => {
                if params.inner.len() != args.len() {
                    throw!(diags.report_internal_error(params.span, "mismatched number of arguments"));
                }

                for (param, arg) in zip_eq(&params.inner, args) {
                    let entry = match (&param.kind, arg) {
                        (&GenericParameterKind::Type(param_span), TypeOrValue::Type(arg)) => {
                            let _: Span = param_span;
                            TypeOrValue::Type(arg)
                        }
                        (GenericParameterKind::Value(param_ty), TypeOrValue::Value(arg)) => {
                            let param_ty = self.eval_expression_as_ty(param_ty);
                            match self.check_type_contains(&param_ty, &arg) {
                                Ok(true) => {}
                                Ok(false) => todo!(),
                                Err(_) => todo!(),
                            }
                            TypeOrValue::Value(Value::Compile(MaybeUnchecked::Checked(arg)))
                        }

                        (_, TypeOrValue::Error(e)) => TypeOrValue::Error(e),

                        (GenericParameterKind::Value(_), TypeOrValue::Type(_)) |
                        (GenericParameterKind::Type(_), TypeOrValue::Value(_)) => {
                            throw!(diags.report_internal_error(param.span, "mismatched generic arg kind"))
                        }
                    };

                    let entry = ScopedEntry::Direct(entry);
                    self.scopes[params_scope].declare(diags, &param.id, entry, Visibility::Private);
                }
            }
            _ => throw!(diags.report_internal_error(def_span, "mismatched presence of arguments")),
        };

        // check ports and add to scope
        let ports_scope = self.scopes.new_child(params_scope, def_span, Visibility::Private);
        // TODO check ports (mostly domains)
        // TODO add ports to scope

        // elaborate module body
        // TODO elaborate module body, probably best in a different file?

        todo!()
    }

    fn find_top_module(&self) -> Result<AstRefModule, ErrorGuaranteed> {
        let diags = self.diags;

        let top_file = self.source[self.source.root_directory].children.get("top")
            .and_then(|&top_dir| self.source[top_dir].file)
            .ok_or_else(|| {
                let title = "no top file found, should be called `top` and be in the root directory of the project";
                diags.report(Diagnostic::new(title).finish())
            })?;
        let top_file_scope = self[top_file].as_ref_ok()?.scope_outer_declare;
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

    fn check_type_contains(&self, ty: &Type, value: &KnownCompileValue) -> Result<bool, Unchecked> {
        let mut any_unchecked = None;

        match (ty, value) {
            (Type::Bool, KnownCompileValue::Bool(_)) => {},
            (Type::String, KnownCompileValue::String(_)) => {},
            (Type::Int(range), KnownCompileValue::Int(value)) => {
                let RangeInfo { start_inc, end_inc } = range;

                match start_inc {
                    None => {}
                    Some(MaybeUnchecked::Checked(start_inc)) => {
                        if value < start_inc {
                            return Ok(false);
                        }
                    }
                    &Some(MaybeUnchecked::Unchecked(u)) => any_unchecked = Some(u),
                }

                match end_inc {
                    None => {}
                    Some(MaybeUnchecked::Checked(end_inc)) => {
                        if value > end_inc {
                            return Ok(false);
                        }
                    }
                    &Some(MaybeUnchecked::Unchecked(u)) => any_unchecked = Some(u),
                }
            }
            (Type::Array(inner, len), KnownCompileValue::Array(values)) => {
                match len {
                    MaybeUnchecked::Checked(len) => {
                        if len != &BigUint::from(values.len()) {
                            return Ok(false);
                        }
                    }
                    &MaybeUnchecked::Unchecked(u) => any_unchecked = Some(u),
                }

                for value in values {
                    match self.check_type_contains(inner, value) {
                        Ok(true) => {}
                        Ok(false) => return Ok(false),
                        Err(u) => any_unchecked = Some(u),
                    }
                }
            }

            (&Type::Unchecked(u), _) => any_unchecked = Some(u),

            (
                Type::Bool | Type::String | Type::Int(_) | Type::Array(_, _),
                KnownCompileValue::Bool(_) | KnownCompileValue::String(_) | KnownCompileValue::Int(_) | KnownCompileValue::Array(_)
            ) => return Ok(false),
        }

        match any_unchecked {
            None => Ok(true),
            Some(u) => Err(u),
        }
    }
}

#[derive(Debug)]
pub struct FileScopes {
    /// The scope that only includes top-level items defined in this file.
    scope_outer_declare: Scope,
    /// Child scope of [scope_outer_declare] that includes all imported items.
    scope_inner_import: Scope,
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

impl_index!(scopes, Scope, ScopeInfo<ScopedEntry>);

impl std::ops::Index<FileId> for CompileState<'_> {
    type Output = Result<FileScopes, ErrorGuaranteed>;
    fn index(&self, file: FileId) -> &Self::Output {
        self.file_scopes.get(&file).unwrap()
    }
}