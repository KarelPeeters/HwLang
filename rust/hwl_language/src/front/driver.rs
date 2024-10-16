use crate::data::compiled::{CompiledDatabase, CompiledDatabasePartial, FileScopes, Item, ItemInfo, ItemInfoPartial};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::data::parsed::{ItemAstReference, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::scope::{Scope, Scopes, Visibility};
use crate::front::types::MaybeConstructor;
use crate::syntax::ast::{Identifier, ImportEntry, ImportFinalKind, ItemImport, MaybeIdentifier, Spanned};
use crate::syntax::pos::FileId;
use crate::syntax::{ast, parse_error_to_diagnostic, parse_file_content};
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};

pub fn compile(diag: &Diagnostics, source: &SourceDatabase) -> (ParsedDatabase, CompiledDatabase) {
    // sort files to ensure platform-independence
    // TODO make this the responsibility of the database builder, now file ids are still not deterministic
    let files_sorted = source.files.keys()
        .copied()
        .sorted_by_key(|&file| &source[source[file].directory].path)
        .collect_vec();

    // items only exists to serve as a level of indirection between values,
    //   so we can easily do the graph solution in a single pass
    let mut items: Arena<Item, ItemInfoPartial> = Arena::default();

    // parse all files and populate local scopes
    let mut map_file_ast = IndexMap::new();
    let mut map_file_scopes = IndexMap::new();
    let mut scopes = Scopes::default();

    for &file in &files_sorted {
        let file_source = &source[file];

        // parse
        let (ast, scope) = match parse_file_content(file, &file_source.source) {
            Ok(ast) => {
                // build declaration scope
                // TODO should users declare other libraries they will be importing from to avoid scope conflict issues?
                let file_span = file_source.offsets.full_span(file);
                let scope_declare = scopes.new_root(file_span);
                let scope_import = scopes.new_child(scope_declare, file_span, Visibility::Private);

                let local_scope_info = &mut scopes[scope_declare];

                for (file_item_index, ast_item) in enumerate(&ast.items) {
                    // TODO add enum-match safety here
                    if let Some(declaration_info) = ast_item.declaration_info() {
                        let vis = match declaration_info.vis {
                            ast::Visibility::Public(_) => Visibility::Public,
                            ast::Visibility::Private => Visibility::Private,
                        };

                        let item = items.push(ItemInfo {
                            defining_id: MaybeIdentifier::Identifier(declaration_info.id.clone()),
                            ast_ref: ItemAstReference { file, file_item_index },
                            signature: None,
                            body: None,
                        });

                        local_scope_info.declare(diag, declaration_info.id, ScopedEntry::Item(item), vis);
                    }
                }

                (Ok(ast), Ok(FileScopes { scope_outer_declare: scope_declare, scope_inner_import: scope_import }))
            }
            Err(e) => {
                let e = diag.report(parse_error_to_diagnostic(e));
                (Err(e), Err(e))
            }
        };

        map_file_ast.insert_first(file, ast);
        map_file_scopes.insert_first(file, scope);
    }

    // populate import scopes
    for &file in &files_sorted {
        if let (Ok(file_ast), Ok(file_scopes)) = (map_file_ast.get(&file).as_ref().unwrap(), map_file_scopes.get(&file).as_ref().unwrap()) {
            for item in &file_ast.items {
                if let ast::Item::Import(item) = item {
                    add_import_to_scope(diag, &source, &mut scopes, &map_file_scopes, file_scopes.scope_inner_import, item);
                }
            }
        }
    }

    // group into state
    let parsed = ParsedDatabase { file_ast: map_file_ast };

    let mut state = CompileState {
        diags: diag,
        source,
        parsed: &parsed,
        log_type_check: false,
        item_signature_stack: Vec::new(),
        item_signatures_finished: false,
        compiled: CompiledDatabase {
            items,
            file_scopes: map_file_scopes,
            scopes,
            generic_type_params: Arena::default(),
            generic_value_params: Arena::default(),
            module_ports: Arena::default(),
            module_info: IndexMap::new(),
            function_info: IndexMap::new(),
            registers: Arena::default(),
            wires: Arena::default(),
            variables: Arena::default(),
        },
    };

    // resolve all item types (which is mostly their signatures)
    // TODO randomize order to check for dependency bugs? but then diagnostics have random orders
    let item_keys = state.compiled.items.keys().collect_vec();
    for &item in &item_keys {
        let _ = state.resolve_item_signature(item);
    }

    // tell future checks that they're expected to complete immediately
    state.item_signatures_finished = true;

    // typecheck all item bodies
    // TODO merge this with the previous pass: better for LSP and maybe for local items
    // TODO alternatively: don't merge, this part can easily be parallelized
    for &item in &item_keys {
        assert!(state.compiled[item].body.is_none());
        let body = state.check_item_body(item);

        let slot = &mut state.compiled[item].body;
        assert!(slot.is_none());
        *slot = Some(body);
    }

    // map to final database
    let items = state.compiled.items.map_values(|info| ItemInfo {
        defining_id: info.defining_id,
        ast_ref: info.ast_ref,
        signature: info.signature.unwrap(),
        body: info.body.unwrap(),
    });

    let compiled = CompiledDatabase {
        scopes: state.compiled.scopes,
        file_scopes: state.compiled.file_scopes,
        items,
        generic_type_params: state.compiled.generic_type_params,
        generic_value_params: state.compiled.generic_value_params,
        module_info: state.compiled.module_info,
        module_ports: state.compiled.module_ports,
        function_info: state.compiled.function_info,
        registers: state.compiled.registers,
        wires: state.compiled.wires,
        variables: state.compiled.variables,
    };

    (parsed, compiled)
}

// TODO create some dedicated auxiliary data structure, with dense and non-dense variants
pub(super) struct CompileState<'d, 'a> {
    pub(super) diags: &'d Diagnostics,
    pub(super) source: &'d SourceDatabase,
    pub(super) parsed: &'a ParsedDatabase,

    pub(super) log_type_check: bool,
    /// The stack of items that are currently being resolved.
    /// This is used to detect cycles in type resolution.
    item_signature_stack: Vec<Item>,
    item_signatures_finished: bool,

    pub(super) compiled: CompiledDatabasePartial,
}

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

impl CompileState<'_, '_> {
    pub fn resolve_item_signature(&mut self, item: Item) -> &MaybeConstructor<TypeOrValue> {
        // return existing signature if there is sone
        //   ideally we could just do `if let Some(...)` here, but for some reason the borrow checker rejects that
        if self.compiled.items[item].signature.is_some() {
            return &self.compiled.items[item].signature.as_ref().unwrap();
        }

        // check for unexpected new signatures
        if self.item_signatures_finished {
            self.diags.report_internal_error(
                self.compiled[item].defining_id.span(),
                "all signatures should be resolved before body checking starts",
            );
        }

        // check for cycle
        let cycle_start_index = self.item_signature_stack.iter().position(|s| s == &item);
        let result: MaybeConstructor<TypeOrValue> = if let Some(cycle_start_index) = cycle_start_index {
            // cycle detected, report error
            let cycle = &self.item_signature_stack[cycle_start_index..];

            // build diagnostic
            // TODO the order is nondeterministic, it depends on which items happened to be visited first
            let mut err = Diagnostic::new("cyclic signature dependency");
            for &stack_item in cycle {
                let item_ast = self.parsed.item_ast(self.compiled[stack_item].ast_ref);
                err = err.add_error(item_ast.common_info().span_short, "part of cycle");
            }
            let err = self.diags.report(err.finish());
            MaybeConstructor::Error(err)
        } else {
            // push current onto stack
            self.item_signature_stack.push(item);

            // resolve new signature
            let result = self.resolve_item_signature_new(item);

            // pop current from stack
            let popped = self.item_signature_stack.pop();
            assert_eq!(popped, Some(item));

            result
        };

        // store and return result
        let slot = &mut self.compiled[item].signature;
        assert!(slot.is_none(), "someone else already set the signature for {item:?}");
        let result = slot.insert(result);
        result
    }
}

fn add_import_to_scope(
    diags: &Diagnostics,
    source: &SourceDatabase,
    scopes: &mut Scopes<ScopedEntry>,
    file_scopes: &IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    target_scope: Scope,
    item: &ItemImport,
) {
    // TODO the current path design does not allow private sub-modules
    //   are they really necessary? if all inner items are private it's effectively equivalent
    //   -> no it's not equivalent, things can also be private from the parent

    let ItemImport { span: _, parents, entry } = item;

    let parent_scope = find_parent_scope(diags, source, file_scopes, parents);

    let import_entries = match &entry.inner {
        ImportFinalKind::Single(entry) => std::slice::from_ref(entry),
        ImportFinalKind::Multi(entries) => entries,
    };

    for import_entry in import_entries {
        let ImportEntry { span: _, id, as_ } = import_entry;

        // TODO allow private visibility into child scopes?
        let entry = match parent_scope {
            Ok(parent_scope) => scopes[parent_scope].find(&scopes, diags, id, Visibility::Public),
            Err(e) => Err(e),
        }
            .map(|entry| entry.value.clone())
            .unwrap_or_else(|e| ScopedEntry::Direct(ScopedEntryDirect::Error(e)));

        let target_scope = &mut scopes[target_scope];
        match as_ {
            Some(as_) => target_scope.maybe_declare(diags, as_, entry, Visibility::Private),
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