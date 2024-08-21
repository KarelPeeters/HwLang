use crate::data::compiled::{CompiledDatabase, Item, ItemBody, ModulePort, ModulePortInfo};
use crate::data::lowered::LoweredDatabase;
use crate::data::module_body::{CombinatorialStatement, ModuleBlock, ModuleBlockCombinatorial, ModuleBody};
use crate::data::source::SourceDatabase;
use crate::error::CompileError;
use crate::front::common::ScopedEntry;
use crate::front::diagnostic::{DiagnosticAddable, DiagnosticContext};
use crate::front::scope::Visibility;
use crate::front::types::{GenericArguments, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{BinaryOp, MaybeIdentifier, PortDirection, PortKind, SyncKind};
use crate::util::data::IndexMapExt;
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};
use num_bigint::BigInt;
use num_traits::{Signed as _, ToPrimitive};
use std::cmp::max;
use std::collections::VecDeque;
use std::fmt::Write;
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
        // pick name
        let ast = compiled.get_item_ast(instance.module);
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
    let &ModuleInstance { module: item, args: ref module_args } = instance;
    let item_info = &compiled[item];
    let item_ast = unwrap_match!(compiled.get_item_ast(item), ast::Item::Module(item_ast) => item_ast);

    let module_info = match &item_info.ty {
        MaybeConstructor::Immediate(Type::Module(info)) => {
            assert!(module_args.is_none());
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

    let body = unwrap_match!(&item_info.body, ItemBody::Module(body) => body);
    let body_str = module_body_to_verilog(compiled, body).unwrap();

    format!("module {module_name} ({port_string});\n{body_str}endmodule\n")
}

fn module_body_to_verilog(compiled: &CompiledDatabase, body: &ModuleBody) -> Result<String, std::fmt::Error> {
    let mut result = String::new();
    let f = &mut result;

    let ModuleBody { blocks } = body;
    for (block_index, block) in enumerate(blocks) {
        if block_index != 0 {
            write!(f, "\n\n")?;
        }

        match block {
            ModuleBlock::Combinatorial(ModuleBlockCombinatorial { statements }) => {
                // TODO collect RHS expressions and use those instead of this star
                // TODO add metadata pointing to source as comments
                writeln!(f, "{INDENT}always(*) begin")?;

                for statement in statements {
                    match statement {
                        &CombinatorialStatement::PortPortAssignment(target, value) => {
                            let target_str = &compiled[target].defining_id.string;
                            let value_str = &compiled[value].defining_id.string;
                            writeln!(f, "{INDENT}{INDENT}{target_str} <= {value_str};")?;
                        }
                    }
                }

                writeln!(f, "{INDENT}end")?;
            }
            ModuleBlock::Clocked(_) => todo!()
        }
    }

    Ok(result)
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
                VerilogType::SingleBit =>
                    "".to_string(),
                VerilogType::MultiBit(signed, n) => {
                    assert!(n > 0, "zero-width signals are not allowed in verilog");
                    format!("{}[{}:0] ", signed.to_verilog_str(), n - 1)
                },
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
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VerilogType {
    SingleBit,
    /// Signedness is really just here for documentation purposes,
    ///   the internal code will always case to the right value anyway
    /// The width includes the sign bit.
    MultiBit(Signed, u64),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Signed {
    Unsigned,
    Signed,
}

impl Signed {
    fn to_verilog_str(self) -> &'static str {
        match self {
            Signed::Unsigned => "",
            Signed::Signed => "signed ",
        }
    }
}

fn type_to_verilog(ty: &Type) -> VerilogType {
    match ty {
        Type::Boolean => VerilogType::SingleBit,
        Type::Bits(n) => {
            match n {
                None => panic!("infinite bit widths should never materialize in RTL"),
                Some(n) => {
                    let n = value_evaluate_int(n).to_u64().expect("negative or too large to convert to integer");
                    VerilogType::MultiBit(Signed::Unsigned, n)
                },
            }
        }
        // TODO convert all of these to bit representations
        Type::Array(_, _) => todo!(),
        Type::Integer(info) => {
            let IntegerTypeInfo { range } = info;
            let range = value_evaluate_int_range(range);
            verilog_type_for_int_range(range)
        },
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
        Value::Binary(op, left, right) => {
            let left = value_evaluate_int(left);
            let right = value_evaluate_int(right);
            match op {
                BinaryOp::Add => left + right,
                BinaryOp::Sub => left - right,
                BinaryOp::Mul => left * right,
                BinaryOp::Div => left / right,
                BinaryOp::Mod => left % right,
                BinaryOp::BitAnd => left & right,
                BinaryOp::BitOr => left | right,
                BinaryOp::BitXor => left ^ right,
                BinaryOp::Pow => left.pow(right.to_u32().unwrap()),
                _ => todo!("evaluate binary value {value:?}")
            }
        },
        Value::GenericParameter(_) => todo!(),
        Value::FunctionParameter(_) => todo!(),
        Value::ModulePort(_) => todo!(),
        Value::Range(_) => todo!(),
        Value::Function(_) => todo!(),
        Value::Module(_) => todo!(),
    }
}

fn value_evaluate_int_range(value: &Value) -> std::ops::Range<BigInt> {
    match value {
        Value::Range(info) => {
            let &RangeInfo { ref start, ref end, end_inclusive } = info;

            // unbound ranges should never show up in RTL
            let start = value_evaluate_int(start.as_ref().unwrap());
            let end = value_evaluate_int(end.as_ref().unwrap());

            if end_inclusive {
                start..(end + 1)
            } else {
                start..end
            }
        },
        // TODO relax once ranges are less built-in
        _ => panic!("only range values can be evaluated as ranges"),
    }
}

fn find_top_module(source: &SourceDatabase, compiled: &CompiledDatabase) -> Result<Item, CompileError> {
    let top_dir = *source[source.root_directory].children.get("top")
        .ok_or(LowerError::NoTopFileFound)?;
    let top_file = source[top_dir].file.ok_or(LowerError::NoTopFileFound)?;
    let top_entry = compiled[compiled[top_file].local_scope].find_immediate_str(source, "top", Visibility::Public)?;
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

// TODO move this to some common place, we will need it all over the place
// TODO do we want to save bits for small ranges that are far away from 0? is this common or worth it?
// Pick signedness and the smallest width such that the entire range can be represented.
fn verilog_type_for_int_range(range: std::ops::Range<BigInt>) -> VerilogType {
    assert!(range.start < range.end, "Range needs to contain at least one value, got {range:?}");

    let min_value = range.start;
    let max_value = range.end - 1u32;

    if min_value < BigInt::ZERO {
        // signed
        // prevent max value underflow
        let max_value = if max_value.is_negative() {
            BigInt::ZERO
        } else {
            max_value
        };
        VerilogType::MultiBit(Signed::Signed, max(
            1 + (min_value + 1u32).bits(),
            1 + max_value.bits(),
        ))
    } else {
        // unsigned
        VerilogType::MultiBit(Signed::Unsigned, max_value.bits())
    }
}

#[cfg(test)]
mod test {
    use crate::back::{verilog_type_for_int_range, Signed, VerilogType};
    use std::ops::Range;

    #[track_caller]
    fn test_case(range: Range<i64>, signed: Signed, width: u32) {
        let expected = VerilogType::MultiBit(signed, width.into());
        let result = verilog_type_for_int_range(range.start.into()..range.end.into());
        println!("range {:?} => {:?}", range, result);
        assert_eq!(
            expected,
            result,
            "mismatch for range {range:?}"
        );
    }

    #[test]
    fn int_range_type_manual() {
        // positive
        test_case(0..1, Signed::Unsigned, 0);
        test_case(0..2, Signed::Unsigned, 1);
        test_case(0..6, Signed::Unsigned, 3);
        test_case(0..7, Signed::Unsigned, 3);
        test_case(0..8, Signed::Unsigned, 3);
        test_case(0..9, Signed::Unsigned, 4);

        // negative
        test_case(-1..0, Signed::Signed, 1);
        test_case(-2..0, Signed::Signed, 2);
        test_case(-6..0, Signed::Signed, 4);
        test_case(-7..0, Signed::Signed, 4);
        test_case(-8..0, Signed::Signed, 4);
        test_case(-9..0, Signed::Signed, 5);

        // mixed
        test_case(-1..1, Signed::Signed, 1);
        test_case(-2..1, Signed::Signed, 2);
        test_case(-1..2, Signed::Signed, 2);
        test_case(-7..8, Signed::Signed, 4);
        test_case(-8..7, Signed::Signed, 4);
        test_case(-8..8, Signed::Signed, 4);
        test_case(-9..8, Signed::Signed, 5);
        test_case(-8..9, Signed::Signed, 5);
    }

    #[test]
    fn int_range_type_automatic() {
        // test that the typical 2s complement ranges behave as expected
        for w in 0u32..32u32 {
            println!("testing w={w}");
            // unsigned
            if w > 1 {
                // for w=0 this case doesn't make any sense
                // for w=1 the bit width should actually get smaller
                test_case(0..2i64.pow(w) - 1, Signed::Unsigned, w);
            }
            test_case(0..2i64.pow(w), Signed::Unsigned, w);
            test_case(0..2i64.pow(w) + 1, Signed::Unsigned, w + 1);

            // singed (only possible if there is room for a sign bit)
            if w > 0 {
                if w > 1 {
                    test_case((-2i64.pow(w - 1) + 1)..2i64.pow(w - 1), Signed::Signed, w);
                }
                test_case(-2i64.pow(w - 1)..(2i64.pow(w - 1) - 1), Signed::Signed, w);
                test_case(-2i64.pow(w - 1)..2i64.pow(w - 1), Signed::Signed, w);
                test_case((-2i64.pow(w - 1) - 1)..2i64.pow(w - 1), Signed::Signed, w + 1);
                test_case(-2i64.pow(w - 1)..(2i64.pow(w - 1) + 1), Signed::Signed, w + 1);
            }
        }
    }
}