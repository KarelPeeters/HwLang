use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::data::parsed::{AstRefItem, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::scope::{Scope, ScopeInfo, Scopes, Visibility};
use crate::new::ir::{IrDesign, IrModule};
use crate::new::misc::{Item, ScopedEntry, TypeOrValue};
use crate::new::value::KnownCompileValue;
use crate::new_index_type;
use crate::syntax::ast;
use crate::syntax::ast::{Identifier, MaybeIdentifier, Spanned};
use crate::syntax::pos::FileId;
use crate::util::arena::{Arena, ArenaSet};
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::Itertools;

pub struct CompileState<'a> {
    diags: &'a Diagnostics,
    source: &'a SourceDatabase,
    parsed: &'a ParsedDatabase,

    scopes: Scopes<ScopedEntry>,
    file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,

    item_visit_stack: Vec<Item>,
    items: Arena<Item, ItemInfo>,
}

// TODO add test that randomizes order of files and items to check for dependency bugs,
//   assert that result and diagnostics are the same
// TODO extend the set of "type-checking" root points:
//   * project settings: multiple top modules
//   * type-checking-only generic instantiations of modules
//   * type-check all modules without generics automatically
//   * type-check modules with generics partially
pub fn compile(diags: &Diagnostics, source: &SourceDatabase, parsed: &ParsedDatabase) -> IrDesign {
    // items only exists to serve as a level of indirection between values,
    //   so we can easily do the graph solution in a single pass
    let mut items: Arena<Item, ItemInfo> = Arena::default();

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

                    let item = items.push(ItemInfo {
                        defining_id: declaration_info.id.as_ref().map_inner(|&id| id.clone()),
                        ast_ref: ast_item_ref,
                    });

                    local_scope_info.maybe_declare(diags, declaration_info.id, ScopedEntry::Item(item), vis);
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
        items,
        item_visit_stack: Vec::new(),
    };

    // start resolving from the top module
    let mut modules = Arena::default();
    let mut module_todo = vec![];
    let mut instances = ArenaSet::<ModuleElaboration, ModuleElaborationInfo>::default();
    let mut instance_ir_map = IndexMap::<ModuleElaboration, IrModule>::new();

    let top_module = state.find_top_module()
        .map(|top_item| {
            let instance = instances.push(ModuleElaborationInfo { item: top_item, generic_args: vec![] });
            module_todo.push(instance);
            (top_item, instance)
        });

    while let Some(instance) = module_todo.pop() {
        // TODO
    }

    let top_module = top_module.and_then(|(top_item, top_instance)| {
        instance_ir_map.get(&top_instance).copied().ok_or_else(|| {
            diags.report_internal_error(
                state.items[top_item].defining_id.span(),
                "missing IR module for top module instance",
            )
        })
    });

    IrDesign {
        top_module,
        modules,
    }
}

new_index_type!(pub ModuleElaboration);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleElaborationInfo {
    pub item: Item,
    pub generic_args: Vec<TypeOrValue<KnownCompileValue>>,
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
                return Err(diags.report(diag));
            }
        };
    }

    let file = match source[curr_dir].file {
        Some(file) => file,
        None => {
            return Err(diags.report_simple("expected path to file", parents_span, "no file exists at this path"))
        }
    };

    file_scopes.get(&file).unwrap().as_ref_ok()
        .map(|scopes| scopes.scope_outer_declare)
}

impl CompileState<'_> {
    fn find_top_module(&self) -> Result<Item, ErrorGuaranteed> {
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
                match &self.parsed[self[item].ast_ref] {
                    ast::Item::Module(module) => {
                        match &module.params {
                            None => Ok(item),
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
}

// TODO maybe just delete this, it seems pretty redundant
#[derive(Debug)]
pub struct ItemInfo {
    defining_id: MaybeIdentifier,
    ast_ref: AstRefItem,
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

impl_index!(items, Item, ItemInfo);
impl_index!(scopes, Scope, ScopeInfo<ScopedEntry>);

impl std::ops::Index<FileId> for CompileState<'_> {
    type Output = Result<FileScopes, ErrorGuaranteed>;
    fn index(&self, file: FileId) -> &Self::Output {
        self.file_scopes.get(&file).unwrap()
    }
}