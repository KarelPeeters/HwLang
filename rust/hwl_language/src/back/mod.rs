use crate::data::compiled::{CompiledDatabase, Item, ItemChecked, ModulePort, ModulePortInfo, RegisterInfo};
use crate::data::diagnostic::{Diagnostic, Diagnostics, ErrorGuaranteed};
use crate::data::lowered::LoweredDatabase;
use crate::data::module_body::{LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleChecked, ModuleStatement};
use crate::data::parsed::ParsedDatabase;
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, TypeOrValue};
use crate::front::scope::Visibility;
use crate::front::types::{GenericArguments, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{BinaryOp, DomainKind, Identifier, PortDirection, PortKind, SyncDomain};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use crate::{swriteln, throw};
use indexmap::{IndexMap, IndexSet};
use itertools::{enumerate, Itertools};
use num_bigint::BigInt;
use num_traits::{Signed as _, ToPrimitive};
use std::cmp::max;
use std::collections::VecDeque;
use std::fmt::Write;
use unwrap_match::unwrap_match;

// TODO make backend configurable between verilog and VHDL?
// TODO ban keywords
// TODO should we still be doing diagnostics here, or should lowering just never start?
pub fn lower(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
) -> LoweredDatabase {
    // find top module
    // TODO allow for multiple top-levels, all in a single compilation and with shared common modules
    let top_module = match find_top_module(diag, source, compiled) {
        Ok(top_module) => top_module,
        Err(e) => {
            return LoweredDatabase {
                top_module_name: Err(e),
                verilog_source: "".to_owned(),
            };
        }
    };

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
        let ast = parsed.item_ast(compiled[instance.module].ast_ref);
        let ast = unwrap_match!(ast, ast::Item::Module(ast) => ast);
        let module_name = pick_unique_name(&ast.id, &mut used_instance_names);

        // generate source
        let verilog_source = generate_module_source(diag, source, parsed, compiled, &instance, &module_name);
        verilog_sources_rev.push(verilog_source);

        // insert in deduplication map
        module_instance_names.insert_first(instance, module_name.clone());
    }

    // concatenate sources and build result
    let verilog_source = verilog_sources_rev.iter().rev().join("\n\n");
    LoweredDatabase {
        top_module_name: Ok("top".to_string()),
        verilog_source,
    }
}

const I: &str = "    ";

// TODO expose the elaborated tree as a user-facing API, next to the ast and the type-checked files
#[derive(Eq, PartialEq, Hash)]
struct ModuleInstance {
    module: Item,
    /// These args are constant and fully evaluated, without any remaining outer generic parameters.
    args: Option<GenericArguments>,
}

// TODO write straight into a single string buffer instead of repeated concatenation
fn generate_module_source(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
    instance: &ModuleInstance,
    module_name: &str,
) -> String {
    let &ModuleInstance { module: item, args: ref module_args } = instance;
    let item_info = &compiled[item];
    let item_ast = unwrap_match!(parsed.item_ast(compiled[item].ast_ref), ast::Item::Module(item_ast) => item_ast);

    let module_info = match &item_info.signature {
        MaybeConstructor::Immediate(TypeOrValue::Value(Value::Module(info))) => {
            assert!(module_args.is_none());
            info
        }
        MaybeConstructor::Constructor(_) => {
            diag.report_todo(item_ast.span, "module instance with generic params/args");
            return "".to_string();
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

        port_string.push_str(I);
        port_string.push_str(&port_to_verilog(diag, source, parsed, compiled, port, comma_str));
        port_string.push_str("\n");
    }

    let body = unwrap_match!(&item_info.body, ItemChecked::Module(body) => body);
    let body_str = module_body_to_verilog(diag, source, parsed, compiled, item_ast.span, body);

    format!("module {module_name} ({port_string});\n{body_str}endmodule\n")
}

fn module_body_to_verilog(diag: &Diagnostics, source: &SourceDatabase, parsed: &ParsedDatabase, compiled: &CompiledDatabase, module_span: Span, body: &ModuleChecked) -> String {
    let mut result = String::new();
    let f = &mut result;

    let ModuleChecked { statements, regs } = body;

    for (reg_index, &(reg, ref init)) in enumerate(regs) {
        if reg_index != 0 {
            swriteln!(f);
        }

        // TODO also allow declaration init for FPGAs and simulation?
        // initialization happens in the corresponding clocked block, not at declaration time
        let _ = init;

        // TODO use id in the name?
        let RegisterInfo { defining_item: _, defining_id: _, domain: sync, ty } = &compiled[reg];
        let ty_str = verilog_ty_to_str(diag, module_span, type_to_verilog(diag, module_span, &ty));
        let sync_str = sync_to_comment_str(source, parsed, compiled, &DomainKind::Sync(sync.clone()));

        swriteln!(f, "{I}reg {ty_str}module_reg_{reg_index}; // {sync_str}");
    }

    if regs.len() > 0 {
        swriteln!(f);
    }

    for (statement_index, statement) in enumerate(statements) {
        if statement_index != 0 {
            swriteln!(f);
        }

        match statement {
            ModuleStatement::Combinatorial(block) => {
                let ModuleBlockCombinatorial { span: _, statements } = block;
                // TODO collect RHS expressions and use those instead of this star
                // TODO add metadata pointing to source as comments
                swriteln!(f, "{I}always @(*) begin");
                for statement in statements {
                    swriteln!(f, "{I}{I}{}", statement_to_string(parsed, compiled, statement));
                }
                swriteln!(f, "{I}end");
            }
            ModuleStatement::Clocked(block) => {
                let &ModuleBlockClocked {
                    span, ref domain, ref on_reset, ref on_block
                } = block;
                let SyncDomain { clock, reset } = domain;

                let sensitivity_value_to_string = |value: &Value| -> (&str, &str, &str) {
                    match value {
                        Value::Error(_) =>
                            return ("0 /* error */", "posedge", ""),
                        &Value::ModulePort(port) =>
                            return (&parsed.module_port_ast(compiled[port].ast).id.string, "posedge", ""),
                        Value::UnaryNot(inner) => {
                            if let &Value::ModulePort(port) = &**inner {
                                return (&parsed.module_port_ast(compiled[port].ast).id.string, "negedge", "!");
                            }
                            // fallthrough
                        }
                        // fallthrough
                        _ => {}
                    }

                    diag.report_todo(span, "general sensitivity");
                    ("0 /* error */", "posedge", "")
                };

                let (clock_str, clock_edge, _) = sensitivity_value_to_string(clock);
                let (reset_str, reset_edge, reset_prefix) = sensitivity_value_to_string(reset);

                swriteln!(f, "{I}always @({clock_edge} {clock_str} or {reset_edge} {reset_str}) begin");
                swriteln!(f, "{I}{I}if ({reset_prefix}{reset_str}) begin");
                for statement in on_reset {
                    swriteln!(f, "{I}{I}{}", statement_to_string(parsed, compiled, statement));
                }
                swriteln!(f, "{I}{I}end else begin");
                for statement in on_block {
                    swriteln!(f, "{I}{I}{I}{}", statement_to_string(parsed, compiled, statement));
                }
                swriteln!(f, "{I}{I}end");

                swriteln!(f, "{I}end");
            }
            ModuleStatement::Instance(_) => {
                diag.report_todo(module_span, "module instance lowering");
            }
            &ModuleStatement::Err(e) => {
                let _: ErrorGuaranteed = e;
                swriteln!(f, "{I}// error statement");
            }
        }
    }

    result
}

fn statement_to_string(parsed: &ParsedDatabase, compiled: &CompiledDatabase, statement: &LowerStatement) -> String {
    match statement {
        &LowerStatement::PortPortAssignment(target, value) => {
            let target_str = &parsed.module_port_ast(compiled[target].ast).id.string;
            let value_str = &parsed.module_port_ast(compiled[value].ast).id.string;
            format!("{target_str} <= {value_str};")
        }
        &LowerStatement::Error(_) => {
            "// error statement".to_string()
        }
    }
}

fn port_to_verilog(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
    port: ModulePort,
    comma_str: &str,
) -> String {
    let &ModulePortInfo {
        ast,
        direction,
        ref kind
    } = &compiled[port];
    let defining_id = &parsed.module_port_ast(ast).id;

    let dir_str = match direction {
        PortDirection::Input => "input",
        PortDirection::Output => "output",
    };

    let (ty_str, comment) = match kind {
        PortKind::Clock => ("".to_owned(), "clock".to_owned()),
        PortKind::Normal { domain: sync, ty } => {
            // TODO include full type in comment
            let ty_str = verilog_ty_to_str(diag, defining_id.span, type_to_verilog(diag, defining_id.span, ty));
            let comment = sync_to_comment_str(source, parsed, compiled, sync);
            (ty_str, comment)
        }
    };

    let name_str = &defining_id.string;

    format!("{dir_str} wire {ty_str}{name_str}{comma_str} // {comment}")
}

fn sync_to_comment_str(source: &SourceDatabase, parsed: &ParsedDatabase, compiled: &CompiledDatabase, sync: &DomainKind<Value>) -> String {
    match sync {
        DomainKind::Sync(SyncDomain { clock, reset }) => {
            let clock_str = compiled.value_to_readable_str(source, parsed, clock);
            let reset_str = compiled.value_to_readable_str(source, parsed, reset);
            format!("sync({}, {})", clock_str, reset_str)
        }
        DomainKind::Async => "async".to_owned(),
    }
}

fn verilog_ty_to_str(diag: &Diagnostics, span: Span, ty: VerilogType) -> String {
    match ty {
        VerilogType::SingleBit =>
            "".to_string(),
        VerilogType::MultiBit(signed, n) => {
            if n == 0 && signed == Signed::Signed {
                let e = diag.report_internal_error(span, "zero-width signed signal");
                return verilog_ty_to_str(diag, span, VerilogType::Error(e));
            }

            if n == 0 {
                // TODO filter out zero-width signals earlier in the process
                let e = diag.report_todo(span, "zero-width signals");
                return verilog_ty_to_str(diag, span, VerilogType::Error(e));
            }

            format!("{}[{}:0] ", signed.to_verilog_str(), n - 1)
        }
        VerilogType::Error(_) =>
            "/* error */ ".to_string(),
    }
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
    MultiBit(Signed, u32),
    Error(ErrorGuaranteed),
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

fn type_to_verilog(diag: &Diagnostics, span: Span, ty: &Type) -> VerilogType {
    match ty {
        &Type::Error(e) =>
            VerilogType::Error(e),
        Type::Unit => VerilogType::MultiBit(Signed::Unsigned, 0),
        Type::Boolean => VerilogType::SingleBit,
        Type::Clock => VerilogType::SingleBit,
        Type::Bits(n) => {
            match n {
                None => {
                    VerilogType::Error(diag.report_internal_error(span, "infinite bit widths should never materialize"))
                }
                Some(n) => {
                    match value_evaluate_int(diag, span, n).map(|n| n.to_u32()) {
                        Ok(Some(n)) => VerilogType::MultiBit(Signed::Unsigned, n),
                        Ok(None) => {
                            // TODO negative should be an internal error
                            let e = diag.report_simple(format!("width {n:?} negative or too large"), span, "used here");
                            VerilogType::Error(e)
                        }
                        Err(e) => VerilogType::Error(e),
                    }
                }
            }
        }
        // TODO convert all of these to bit representations
        Type::Array(_, _) =>
            VerilogType::Error(diag.report_todo(span, "lower type 'array'")),
        Type::Integer(info) => {
            let IntegerTypeInfo { range } = info;
            let range = match value_evaluate_int_range(diag, span, range) {
                Ok(range) => range,
                Err(e) => return VerilogType::Error(e),
            };
            verilog_type_for_int_range(diag, span, range)
        }
        Type::Tuple(_) =>
            VerilogType::Error(diag.report_todo(span, "lower type 'tuple'")),
        Type::Struct(_) =>
            VerilogType::Error(diag.report_todo(span, "lower type 'struct'")),
        Type::Enum(_) =>
            VerilogType::Error(diag.report_todo(span, "lower type 'enum'")),
        // invalid RTL types
        // TODO redesign type-type such that these are statically impossible at this point?
        Type::Any | Type::String | Type::GenericParameter(_) | Type::Range | Type::Function(_) | Type::Unchecked | Type::Never =>
            VerilogType::Error(diag.report_internal_error(span, format!("type '{ty:?}' should not materialize"))),
    }
}

fn value_evaluate_int(diag: &Diagnostics, span: Span, value: &Value) -> Result<BigInt, ErrorGuaranteed> {
    match value {
        &Value::Error(e) => Err(e),
        Value::IntConstant(i) => Ok(i.clone()),
        Value::Binary(op, left, right) => {
            let left = value_evaluate_int(diag, span, left)?;
            let right = value_evaluate_int(diag, span, right)?;
            match op {
                BinaryOp::Add => Ok(left + right),
                BinaryOp::Sub => Ok(left - right),
                BinaryOp::Mul => Ok(left * right),
                BinaryOp::Div => Ok(left / right),
                BinaryOp::Mod => Ok(left % right),
                BinaryOp::BitAnd => Ok(left & right),
                BinaryOp::BitOr => Ok(left | right),
                BinaryOp::BitXor => Ok(left ^ right),
                BinaryOp::Pow => {
                    match right.to_u32() {
                        Some(right) =>
                            Ok(left.pow(right)),
                        None =>
                            Err(diag.report_simple(format!("power exponent too large: {}", right), span, "used here")),
                    }
                }
                _ => Err(diag.report_todo(span, format!("value_evaluate_int binary value {value:?}")))
            }
        }
        Value::BoolConstant(_) => Err(diag.report_todo(span, "value_evaluate_int value BoolConstant")),
        Value::StringConstant(_) => Err(diag.report_todo(span, "value_evaluate_int value StringConstant")),
        Value::Unit => Err(diag.report_todo(span, "value_evaluate_int value Unit")),
        Value::UnaryNot(_) => Err(diag.report_todo(span, "value_evaluate_int value UnaryNot")),
        Value::GenericParameter(_) => Err(diag.report_todo(span, "value_evaluate_int value GenericParameter")),
        Value::ModulePort(_) => Err(diag.report_todo(span, "value_evaluate_int value ModulePort")),
        Value::Range(_) => Err(diag.report_todo(span, "value_evaluate_int value Range")),
        Value::FunctionReturn(_) => Err(diag.report_todo(span, "value_evaluate_int value Function")),
        Value::Module(_) => Err(diag.report_todo(span, "value_evaluate_int value Module")),
        Value::Wire => Err(diag.report_todo(span, "value_evaluate_int value Wire")),
        Value::Register(_) => Err(diag.report_todo(span, "value_evaluate_int value Reg")),
        Value::Variable(_) => Err(diag.report_todo(span, "value_evaluate_int value Variable")),
        Value::Never => Err(diag.report_todo(span, "value_evaluate_int value Never")),
    }
}

fn value_evaluate_int_range(diag: &Diagnostics, span: Span, value: &Value) -> Result<std::ops::Range<BigInt>, ErrorGuaranteed> {
    match value {
        &Value::Error(e) => Err(e),
        Value::Range(info) => {
            let &RangeInfo { ref start, ref end } = info;

            let (start, end) = match (start, end) {
                (Some(start), Some(end)) => (start, end),
                _ => throw!(diag.report_internal_error(span, "unbound integers should not materialize")),
            };

            let start = value_evaluate_int(diag, span, start.as_ref())?;
            let end = value_evaluate_int(diag, span, end.as_ref())?;

            Ok(start..end)
        }
        _ => Err(diag.report_todo(span, format!("evaluate range of non-range value {value:?}"))),
    }
}

fn find_top_module(
    diag: &Diagnostics,
    source: &SourceDatabase,
    compiled: &CompiledDatabase,
) -> Result<Item, ErrorGuaranteed> {
    let top_file = source[source.root_directory].children.get("top")
        .and_then(|&top_dir| {
            source[top_dir].file
        }).ok_or_else(|| {
        let title = "no top file found, should be called `top` and be in the root directory of the project";
        diag.report(Diagnostic::new(title).finish())
    })?;
    let top_file_scope = compiled[top_file].as_ref_ok()?.scope_outer_declare;
    let top_entry = compiled[top_file_scope].find_immediate_str(diag, "top", Visibility::Public)?;

    match top_entry.value {
        &ScopedEntry::Item(item) => {
            match &compiled[item].signature {
                &MaybeConstructor::Error(e) =>
                    Err(e),
                MaybeConstructor::Constructor(_) => {
                    Err(diag.report_simple(
                        "top should be a module without generic parameters, got a constructor",
                        top_entry.defining_span,
                        "defined here",
                    ))
                }
                MaybeConstructor::Immediate(imm) => {
                    match imm {
                        TypeOrValue::Value(Value::Module(_)) =>
                            Ok(item),
                        TypeOrValue::Type(_) => {
                            Err(diag.report_simple(
                                "top should be a module, got a type",
                                top_entry.defining_span,
                                "defined here",
                            ))
                        }
                        TypeOrValue::Value(_) => {
                            Err(diag.report_simple(
                                "top should be a module, got a non-module value",
                                top_entry.defining_span,
                                "defined here",
                            ))
                        }
                        &TypeOrValue::Error(e) => Err(e),
                    }
                }
            }
        }
        ScopedEntry::Direct(_) => {
            // TODO include "got" string
            // TODO is this even ever possible? direct should only be inside of scopes
            Err(diag.report_simple(
                "top should be an item, got a direct",
                top_entry.defining_span,
                "defined here",
            ))
        }
    }
}

// TODO proper filename uniqueness scheme: combination of path, raw name and generic args
//   it might be worth getting a bit clever about this, or we can also just always use the full name
//   but using the paths generic args fully might generate _very_ long strings
fn pick_unique_name(id: &Identifier, used_names: &mut IndexSet<String>) -> String {
    let raw_name = &id.string;

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
fn verilog_type_for_int_range(diag: &Diagnostics, span: Span, range: std::ops::Range<BigInt>) -> VerilogType {
    assert!(range.start < range.end, "Range needs to contain at least one value, got {range:?}");

    let min_value = range.start;
    let max_value = range.end - 1u32;

    let (signed, bits) = if min_value < BigInt::ZERO {
        // signed
        // prevent max value underflow
        let max_value = if max_value.is_negative() {
            BigInt::ZERO
        } else {
            max_value
        };
        let max_bits = max(
            1 + (min_value + 1u32).bits(),
            1 + max_value.bits(),
        );

        (Signed::Signed, max_bits)
    } else {
        // unsigned
        (Signed::Unsigned, max_value.bits())
    };

    match bits.to_u32() {
        Some(bits) => VerilogType::MultiBit(signed, bits),
        None => {
            let e = diag.report_simple(
                format!("integer range needs more bits ({bits}) than are possible in verilog"),
                span,
                "used here",
            );
            VerilogType::Error(e)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::back::{verilog_type_for_int_range, Signed, VerilogType};
    use crate::data::diagnostic::Diagnostics;
    use crate::syntax::pos::{FileId, Pos, Span};
    use std::ops::Range;

    #[track_caller]
    fn test_case(range: Range<i64>, signed: Signed, width: u32) {
        let diag = Diagnostics::new();
        let span = Span::empty_at(Pos { file: FileId::SINGLE, byte: 0 });

        let expected = VerilogType::MultiBit(signed, width.into());
        let result = verilog_type_for_int_range(&diag, span, range.start.into()..range.end.into());
        println!("range {:?} => {:?}", range, result);
        assert_eq!(
            expected,
            result,
            "mismatch for range {range:?}"
        );

        assert!(diag.finish().is_empty());
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