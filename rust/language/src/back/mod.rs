use crate::data::compiled::CompiledDataBase;
use crate::data::lowered::LoweredDatabase;
use crate::data::source::SourceDatabase;
use crate::error::CompileError;
use crate::front::common::ScopedEntry;
use crate::front::diagnostic::DiagnosticAddable;
use crate::front::driver::Item;
use crate::front::scope::Visibility;
use crate::front::types::{GenericArguments, MaybeConstructor, Type};
use crate::syntax::ast::MaybeIdentifier;
use crate::util::data::IndexMapExt;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::collections::VecDeque;

#[derive(Debug)]
pub enum LowerError {
    NoTopFileFound,
}

// TODO make backend configurable between verilog and VHDL?
pub fn lower(source: &SourceDatabase, compiled: &CompiledDataBase) -> Result<LoweredDatabase, CompileError> {
    // find top module
    // TODO allow for multiple top-levels, all in a single compilation and with shared common modules
    let top_module = find_top_module(source, compiled)?;

    // generate module sources
    // delay concatenation, we still need to flip the order
    let mut verilog_sources_rev = vec![];
    let mut used_instance_names = IndexSet::new();
    let mut module_instance_names: IndexMap<ModuleInstance, String> = IndexMap::new();

    // TODO pick some nice traversal order
    let mut todo = VecDeque::new();
    todo.push_back(ModuleInstance { module: top_module, args: GenericArguments { vec: Vec::new() } });

    while let Some(instance) = todo.pop_front() {
        let item_info = &compiled[instance.module];

        // pick name
        let ast = compiled.get_item_ast(item_info.item_reference);
        let module_name = pick_unique_name(&ast.common_info().id, &mut used_instance_names);

        // generate source
        let verilog_source = generate_module_source(source, compiled, &instance, &module_name);
        verilog_sources_rev.push(verilog_source);

        // insert in deduplication map
        module_instance_names.insert_first(instance, module_name.clone());
    }

    // concatenate sources and build result
    let verilog_source = verilog_sources_rev.iter().rev().join("\n\n");
    let result = LoweredDatabase { top_module_name: "top".to_string(), verilog_source };
    Ok(result)
}

// TODO expose the elaborated tree as a user-facing API, next to the ast and the type-checked files
#[derive(Eq, PartialEq, Hash)]
struct ModuleInstance {
    module: Item,
    /// These args are constant and fully evaluated, without any remaining outer generic parameters.
    args: GenericArguments,
}

fn generate_module_source(_: &SourceDatabase, _: &CompiledDataBase, _: &ModuleInstance, module_name: &str) -> String {
    // TODO generate ports
    // TODO generate body
    format!("module {} ();\nendmodule", module_name)
}

fn find_top_module(source: &SourceDatabase, compiled: &CompiledDataBase) -> Result<Item, CompileError> {
    let top_dir = *source[source.root_directory].children.get("top")
        .ok_or(LowerError::NoTopFileFound)?;
    let top_file = source[top_dir].file.ok_or(LowerError::NoTopFileFound)?;
    let top_entry = &compiled[top_file].local_scope.find_immediate_str(source, "top", Visibility::Public)?;
    match top_entry.value {
        &ScopedEntry::Item(item) => {
            match &compiled[item].ty {
                MaybeConstructor::Constructor(_) => {
                    let err = source.diagnostic("top should be a module, got a constructor")
                        .add_error(top_entry.defining_span, "defined here")
                        .finish();
                    Err(err.into())
                }
                MaybeConstructor::Immediate(ty) => {
                    if let Type::Module(_) = ty {
                        Ok(item)
                    } else {
                        let err = source.diagnostic("top should be a module, got a non-module type")
                            .add_error(top_entry.defining_span, "defined here")
                            .finish();
                        Err(err.into())
                    }
                }
            }
        }
        ScopedEntry::Direct(_) => {
            // TODO include "got" string
            // TODO is this even ever possible? direct should only be inside of scopes
            let err = source.diagnostic("top should be an item, got a direct")
                .add_error(top_entry.defining_span, "defined here")
                .finish();
            Err(err.into())
        }
    }
}

// TODO proper filename uniqueness scheme: combination of path, raw name and generic args
//   it might be worth getting a bit clever about this, or we can also just always use the full name
//   but using the paths generic args fully might generate _very_ long strings
fn pick_unique_name(id: &MaybeIdentifier, used_names: &mut IndexSet<String>) -> String {
    let raw_name: &str = match id {
        // TODO this is not allowed, maybe just panic?
        MaybeIdentifier::Dummy(_) => "unnamed",
        MaybeIdentifier::Identifier(id) => &id.string,
    };

    if used_names.insert(raw_name.to_owned()) {
        // immediate success
        raw_name.to_owned()
    } else {
        // append suffix
        for i in 0.. {
            let cand = format!("{}_{}", raw_name, i);
            if used_names.insert(cand.clone()) {
                return cand;
            }
        }
        unreachable!()
    }
}
