use crate::data::compiled::{CompiledDatabase, CompiledDatabasePartial, Item, ItemInfo, ItemInfoPartial};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics};
use crate::data::parsed::{ItemAstReference, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, TypeOrValue};
use crate::front::scope::{Scopes, Visibility};
use crate::front::types::MaybeConstructor;
use crate::syntax::{ast, parse_error_to_diagnostic, parse_file_content};
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};

pub fn compile(diagnostics: &Diagnostics, database: &SourceDatabase) -> (ParsedDatabase, CompiledDatabase) {
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
    let mut file_ast = IndexMap::new();
    let mut file_scope = IndexMap::new();
    let mut scopes = Scopes::default();

    for file in files_sorted {
        let file_info = &database[file];

        // parse
        let (ast, scope) = match parse_file_content(file, &file_info.source) {
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

                    let item = items.push(ItemInfo {
                        ast_ref: ItemAstReference { file, file_item_index },
                        signature: None,
                        body: None,
                    });

                    local_scope_info.maybe_declare(diagnostics, &common_info.id, ScopedEntry::Item(item), vis);
                }

                (Ok(ast), Ok(local_scope))
            }
            Err(e) => {
                let e = diagnostics.report(parse_error_to_diagnostic(e));
                (Err(e), Err(e))
            }
        };

        file_ast.insert_first(file, ast);
        file_scope.insert_first(file, scope);
    }

    let parsed = ParsedDatabase { file_ast };

    let mut state = CompileState {
        diag: diagnostics,
        source: database,
        parsed: &parsed,
        compiled: CompiledDatabase {
            items,
            file_scope,
            generic_type_params: Arena::default(),
            generic_value_params: Arena::default(),
            function_type_params: Arena::default(),
            function_value_params: Arena::default(),
            module_ports: Arena::default(),
            module_info: IndexMap::new(),
            function_info: IndexMap::new(),
            scopes,
        },
        log_const_eval: false,
    };

    // resolve all item types (which is mostly their signatures)
    // TODO randomize order to check for dependency bugs? but then diagnostics have random orders
    let item_keys = state.compiled.items.keys().collect_vec();
    for &item in &item_keys {
        state.resolve_item_signature_fully(item);
    }

    // typecheck all item bodies
    // TODO merge this with the previous pass: better for LSP and maybe for local items
    for &item in &item_keys {
        assert!(state.compiled[item].body.is_none());
        let body = match state.resolve_item_body(item) {
            Ok(body) => body,
            Err(ResolveFirst(_)) => panic!("all types should be resolved by now"),
        };

        let slot = &mut state.compiled[item].body;
        assert!(slot.is_none());
        *slot = Some(body);
    }

    // map to final database
    let items = state.compiled.items.map_values(|info| ItemInfo {
        ast_ref: info.ast_ref,
        signature: info.signature.unwrap(),
        body: info.body.unwrap(),
    });

    let compiled = CompiledDatabase {
        file_scope: state.compiled.file_scope,
        scopes: state.compiled.scopes,
        items,
        generic_type_params: state.compiled.generic_type_params,
        generic_value_params: state.compiled.generic_value_params,
        function_type_params: state.compiled.function_type_params,
        function_value_params: state.compiled.function_value_params,
        module_info: state.compiled.module_info,
        module_ports: state.compiled.module_ports,
        function_info: state.compiled.function_info,
    };

    (parsed, compiled)
}

// TODO create some dedicated auxiliary data structure, with dense and non-dense variants
pub(super) struct CompileState<'d, 'a> {
    pub(super) log_const_eval: bool,
    pub(super) diag: &'d Diagnostics,
    pub(super) source: &'d SourceDatabase,
    pub(super) parsed: &'a ParsedDatabase,
    pub(super) compiled: CompiledDatabasePartial,
}

#[derive(Debug, Copy, Clone)]
pub struct ResolveFirst(Item);

pub type ResolveResult<T> = Result<T, ResolveFirst>;

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
    fn resolve_item_signature_fully(&mut self, item: Item) {
        let mut stack = vec![item];

        // TODO avoid repetitive work by switching to async instead?
        while let Some(curr) = stack.pop() {
            if self.compiled[curr].signature.is_some() {
                // already resolved, skip
                continue;
            }

            let resolved = match self.resolve_item_signature_new(curr) {
                Ok(resolved) => resolved,
                Err(ResolveFirst(first)) => {
                    assert!(self.compiled[first].signature.is_none(), "request to resolve {first:?} first, but it already has a signature");

                    // push curr failed attempt back on the stack
                    stack.push(curr);

                    // check for cycle
                    let cycle_start_index = stack.iter().position(|s| s == &first);
                    if let Some(cycle_start_index) = cycle_start_index {
                        // cycle detected, report error
                        let cycle = &stack[cycle_start_index..];

                        // build diagnostic
                        // TODO the order is nondeterministic, it depends on which items happened to be visited first
                        let mut diag = Diagnostic::new("cyclic signature dependency");
                        for &stack_item in cycle {
                            let item_ast = self.parsed.item_ast(self.compiled[stack_item].ast_ref);
                            diag = diag.add_error(item_ast.common_info().span_short, "part of cycle");
                        }
                        let err = self.diag.report(diag.finish());

                        // set slot of all involved items
                        for &stack_item in cycle {
                            let slot = &mut self.compiled[stack_item].signature;
                            assert!(slot.is_none(), "someone else already set the signature for {curr:?}");
                            *slot = Some(MaybeConstructor::Error(err));
                        }

                        // remove cycle from the stack
                        drop(stack.drain(cycle_start_index..));
                        continue;
                    } else {
                        // no cycle, visit the next item
                        stack.push(first);
                        continue;
                    }
                }
            };

            // managed to resolve the current item, store its type
            let slot = &mut self.compiled[curr].signature;
            assert!(slot.is_none(), "someone else already set the type for {curr:?}");
            *slot = Some(resolved);
        }
    }

    pub fn resolve_item_signature(&self, item: Item) -> ResolveResult<&MaybeConstructor<TypeOrValue>> {
        match self.compiled[item].signature {
            Some(ref r) => Ok(r),
            None => Err(ResolveFirst(item)),
        }
    }
}
