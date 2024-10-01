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

pub fn compile(diag: &Diagnostics, database: &SourceDatabase) -> (ParsedDatabase, CompiledDatabase) {
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
                        defining_id: common_info.id.clone(),
                        ast_ref: ItemAstReference { file, file_item_index },
                        signature: None,
                        body: None,
                    });

                    local_scope_info.maybe_declare(diag, &common_info.id, ScopedEntry::Item(item), vis);
                }

                (Ok(ast), Ok(local_scope))
            }
            Err(e) => {
                let e = diag.report(parse_error_to_diagnostic(e));
                (Err(e), Err(e))
            }
        };

        file_ast.insert_first(file, ast);
        file_scope.insert_first(file, scope);
    }

    let parsed = ParsedDatabase { file_ast };

    let mut state = CompileState {
        diags: diag,
        source: database,
        parsed: &parsed,
        log_const_eval: false,
        item_signature_stack: Vec::new(),
        item_signatures_finished: false,
        compiled: CompiledDatabase {
            items,
            file_scope,
            scopes,
            generic_type_params: Arena::default(),
            generic_value_params: Arena::default(),
            module_ports: Arena::default(),
            module_info: IndexMap::new(),
            function_info: IndexMap::new(),
            registers: Arena::default(),
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
        file_scope: state.compiled.file_scope,
        scopes: state.compiled.scopes,
        items,
        generic_type_params: state.compiled.generic_type_params,
        generic_value_params: state.compiled.generic_value_params,
        module_info: state.compiled.module_info,
        module_ports: state.compiled.module_ports,
        function_info: state.compiled.function_info,
        registers: state.compiled.registers,
        variables: state.compiled.variables,
    };

    (parsed, compiled)
}

// TODO create some dedicated auxiliary data structure, with dense and non-dense variants
pub(super) struct CompileState<'d, 'a> {
    pub(super) diags: &'d Diagnostics,
    pub(super) source: &'d SourceDatabase,
    pub(super) parsed: &'a ParsedDatabase,

    pub(super) log_const_eval: bool,
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
