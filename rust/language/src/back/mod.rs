use crate::data::compiled::{CompiledDatabase, ModulePort, ModulePortInfo};
use crate::data::lowered::LoweredDatabase;
use crate::data::source::SourceDatabase;
use crate::error::CompileError;
use crate::front::common::ScopedEntry;
use crate::front::diagnostic::{DiagnosticAddable, DiagnosticContext};
use crate::front::driver::Item;
use crate::front::scope::Visibility;
use crate::front::types::{GenericArguments, MaybeConstructor, Type};
use crate::front::values::Value;
use crate::syntax::ast;
use crate::syntax::ast::{MaybeIdentifier, PortDirection, PortKind, SyncKind};
use crate::util::data::IndexMapExt;
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::collections::VecDeque;
use unwrap_match::unwrap_match;

#[derive(Debug)]
pub enum LowerError {
    NoTopFileFound,
}

// TODO make backend configurable between verilog and VHDL?
pub fn lower(source: &SourceDatabase, compiled: &CompiledDatabase) -> Result<LoweredDatabase, CompileError> {
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
    todo.push_back(ModuleInstance { module: top_module, args: None });

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

const INDENT: &str = "    ";

// TODO expose the elaborated tree as a user-facing API, next to the ast and the type-checked files
#[derive(Eq, PartialEq, Hash)]
struct ModuleInstance {
    module: Item,
    /// These args are constant and fully evaluated, without any remaining outer generic parameters.
    args: Option<GenericArguments>,
}

// TODO write straight into a single string buffer instead of repeated concatenation
fn generate_module_source(source: &SourceDatabase, compiled: &CompiledDatabase, instance: &ModuleInstance, module_name: &str) -> String {
    let item_info = &compiled[instance.module];
    let item_ast = unwrap_match!(compiled.get_item_ast(item_info.item_reference), ast::Item::Module(item_ast) => item_ast);

    let module_info = match &item_info.ty {
        MaybeConstructor::Immediate(Type::Module(info)) => {
            assert!(instance.args.is_none());
            info
        },
        MaybeConstructor::Constructor(_) => {
            source.diagnostic_todo(item_ast.span, "module instance with generic params/args");
        }
        _ => unreachable!(),
    };

    // TODO build this in a single pass?
    let mut port_string = String::new();
    for (port_index, &port) in enumerate(&module_info.ports) {
        if port_index == 0 {
            port_string.push_str("\n");
        }
        let comma_str = if port_index == module_info.ports.len() - 1 { "" } else { "," };

        port_string.push_str(INDENT);
        port_string.push_str(&port_to_verilog(compiled, port, comma_str));
        port_string.push_str("\n");
    }

    // TODO generate ports
    // TODO generate body
    format!("module {module_name} ({port_string});\n{INDENT}// TODO module content\nendmodule")
}

fn port_to_verilog(compiled: &CompiledDatabase, port: ModulePort, comma_str: &str) -> String {
    let &ModulePortInfo {
        defining_item,
        ref defining_id,
        direction,
        ref kind
    } = &compiled[port];

    let dir_str = match direction {
        PortDirection::Input => "input",
        PortDirection::Output => "output",
    };

    let (ty_str, comment) = match kind {
        PortKind::Clock => ("".to_owned(), "clock".to_owned()),
        PortKind::Normal { sync, ty } => {
            let ty_str = match type_to_verilog(ty) {
                VerilogType::SingleBit => "".to_string(),
                VerilogType::MultiBit(n) => format!("[{}:0] ", n - 1),
            };

            // TODO include full type in comment
            let comment = match sync {
                SyncKind::Sync(clk) => {
                    let clk_port = unwrap_match!(clk, &Value::ModulePort(port) => port);
                    let clk_port_info = &compiled[clk_port];
                    assert_eq!(defining_item, clk_port_info.defining_item);
                    format!("sync({})", &clk_port_info.defining_id.string)
                },
                SyncKind::Async => "async".to_owned(),
            };

            (ty_str, comment)
        }
    };

    let name_str = &defining_id.string;

    format!("{dir_str} wire {ty_str}{name_str}{comma_str} // {comment}")
}

// TODO signed/unsigned? does it ever matter?
// TODO does width 0 work properly or should we avoid instantiating those?
// TODO what is the max bus width allowed in verilog?
pub enum VerilogType {
    SingleBit,
    MultiBit(usize),
}

fn type_to_verilog(ty: &Type) -> VerilogType {
    match ty {
        Type::Boolean => VerilogType::SingleBit,
        Type::Bits(n) => {
            match n {
                None => panic!("infinite bit widths should never materialize in RTL"),
                Some(n) => VerilogType::MultiBit(value_evaluate_int(n).to_usize().expect("bit width too large")),
            }
        }
        // TODO convert all of these to bit representations
        Type::Array(_, _) => todo!(),
        Type::Integer(_) => todo!(),
        Type::Tuple(_) => todo!(),
        Type::Struct(_) => todo!(),
        Type::Enum(_) => todo!(),
        // invalid RTL types
        // TODO materialize generics in RTL anyway, where possible? reduces some code duplication
        //   with optional generics, maybe even always have them? but then if there are different types it gets tricky
        Type::GenericParameter(_) => panic!("generics should never materialize in RTL"),
        Type::Range => panic!("ranges should never materialize in RTL"),
        Type::Function(_) => panic!("functions should never materialize in RTL"),
        Type::Module(_) => panic!("modules should never materialize in RTL"),
    }
}

fn value_evaluate_int(value: &Value) -> BigInt {
    match value {
        Value::Int(i) => i.clone(),
        Value::GenericParameter(_) => todo!(),
        Value::FunctionParameter(_) => todo!(),
        Value::ModulePort(_) => todo!(),
        Value::Range(_) => todo!(),
        Value::Binary(_, _, _) => todo!(),
        Value::Function(_) => todo!(),
        Value::Module(_) => todo!(),
    }
}

fn find_top_module(source: &SourceDatabase, compiled: &CompiledDatabase) -> Result<Item, CompileError> {
    let top_dir = *source[source.root_directory].children.get("top")
        .ok_or(LowerError::NoTopFileFound)?;
    let top_file = source[top_dir].file.ok_or(LowerError::NoTopFileFound)?;
    let top_entry = &compiled[top_file].local_scope.find_immediate_str(source, "top", Visibility::Public)?;
    match top_entry.value {
        &ScopedEntry::Item(item) => {
            match &compiled[item].ty {
                MaybeConstructor::Constructor(_) => {
                    let err = source.diagnostic("top should be a module without generic parameters, got a constructor")
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
