use crate::back::todo::{BackModule, BackModuleList, BackModuleName};
use crate::data::compiled::{CompiledDatabase, GenericParameter, Item, ItemChecked, ModulePort, ModulePortInfo, Register, RegisterInfo, Wire, WireInfo};
use crate::data::diagnostic::{Diagnostic, Diagnostics, ErrorGuaranteed};
use crate::data::lowered::LoweredDatabase;
use crate::data::module_body::{LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleChecked, ModuleInstance, ModuleStatement};
use crate::data::parsed::ParsedDatabase;
use crate::data::source::SourceDatabase;
use crate::front::common::{GenericContainer, GenericMap, ScopedEntry, TypeOrValue};
use crate::front::scope::Visibility;
use crate::front::types::{Constructor, GenericArguments, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{BinaryOp, DomainKind, PortDirection, PortKind, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::ResultExt;
use crate::{swrite, swriteln, throw};
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Itertools};
use num_bigint::BigInt;
use num_traits::{Signed as _, ToPrimitive};
use std::cmp::max;
use std::fmt::{Display, Formatter, Write};
use unwrap_match::unwrap_match;

// TODO make backend configurable between verilog and VHDL?
// TODO ban keywords
// TODO should we still be doing diagnostics here, or should lowering just never start?
pub fn lower(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
) -> LoweredDatabase {
    // generate module sources
    // delay concatenation, we still need to flip the order
    let mut verilog_sources_rev = vec![];
    let mut todo = BackModuleList::new();

    // add top module to queue
    // TODO allow for multiple top-levels, all in a single compilation and with shared common modules
    let top_module_name = find_top_module(diag, source, compiled)
        .map(|top_module| todo.push(parsed, compiled, BackModule { item: top_module, args: None }));

    // generate source
    while let Some((module_name, module)) = todo.pop() {
        let verilog_source = generate_module_source(diag, source, parsed, compiled, &mut todo, module_name, module);
        verilog_sources_rev.push(verilog_source);
    }

    // concatenate sources and build result
    let verilog_source = verilog_sources_rev.iter().rev().join("\n\n");
    LoweredDatabase {
        top_module_name,
        module_names: todo.finish(),
        verilog_source,
    }
}

/// Indentation used in the generated code.
const I: &str = "    ";

// TODO write straight into a single string buffer instead of repeated concatenation
fn generate_module_source(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
    todo: &mut BackModuleList,
    module_name: BackModuleName,
    module: BackModule,
) -> String {
    let BackModule { item, args: module_args } = module;
    let item_info = &compiled[item];
    let item_ast = unwrap_match!(parsed.item_ast(compiled[item].ast_ref), ast::Item::Module(item_ast) => item_ast);

    let (generic_map, generics_comment, module_info) = match &item_info.signature {
        MaybeConstructor::Immediate(TypeOrValue::Value(Value::Module(info))) => {
            assert!(module_args.is_none());
            (GenericMap::empty(), "\n".to_string(), info)
        }
        MaybeConstructor::Constructor(Constructor { parameters, inner: TypeOrValue::Value(Value::Module(info)) }) => {
            let module_args = module_args.unwrap();
            assert_eq!(parameters.vec.len(), module_args.vec.len());

            let mut map = GenericMap::empty();
            let mut generics_comment = " with generic args:\n".to_string();

            for (&param, arg) in zip_eq(&parameters.vec, &module_args.vec) {
                match (param, arg) {
                    (GenericParameter::Type(param), TypeOrValue::Type(arg)) => {
                        swriteln!(
                            &mut generics_comment,
                            "//{I}{} = {}",
                            compiled.type_to_readable_str(source, parsed, &Type::GenericParameter(param)),
                            compiled.type_to_readable_str(source, parsed, arg),
                        );
                        map.generic_ty.insert_first(param, arg.clone());
                    }
                    (GenericParameter::Value(param), TypeOrValue::Value(arg)) => {
                        swriteln!(
                            &mut generics_comment,
                            "//{I}{} = {}",
                            compiled.value_to_readable_str(source, parsed, &Value::GenericParameter(param)),
                            compiled.value_to_readable_str(source, parsed, arg),
                        );
                        map.generic_value.insert_first(param, arg.clone());
                    }
                    _ => unreachable!(),
                }
            }
            (map, generics_comment, info)
        }
        _ => {
            diag.report_internal_error(item_ast.span, "trying to lower module with value that is not a module");
            return format!("// error module declaration of {module_name}");
        }
    };

    // TODO build this in a single pass?
    let mut port_string = String::new();
    for (port_index, &port) in enumerate(&module_info.ports) {
        if port_index == 0 {
            port_string.push_str("\n");
        }
        let comma_str = if port_index == module_info.ports.len() - 1 { "" } else { "," };

        port_string.push_str(I);
        port_string.push_str(&port_to_verilog(diag, source, parsed, compiled, &generic_map, port, comma_str));
        port_string.push_str("\n");
    }

    // TODO remove this clone once generics are implemented to avoid substitution
    let body = unwrap_match!(&item_info.body, ItemChecked::Module(body) => body).clone();
    let body_str = module_body_to_verilog(diag, source, parsed, compiled, todo, &generic_map, item_ast.span, &body);
    let module_id_str = &item_ast.id.string;

    let comment_str = format!("// module {module_id_str:?}{generics_comment}");
    format!("{comment_str}module {module_name} ({port_string});\n{body_str}endmodule\n")
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum Signal {
    Reg(Register),
    Wire(Wire),
}

#[derive(Debug, Copy, Clone)]
struct SignalName(usize);

impl Display for SignalName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "module_signal_{}", self.0)
    }
}

fn module_body_to_verilog(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
    todo: &mut BackModuleList,
    map: &GenericMap,
    module_span: Span,
    body: &ModuleChecked,
) -> String {
    let mut result = String::new();
    let f = &mut result;

    let ModuleChecked { statements, regs, wires } = body;
    let mut signal_map = IndexMap::new();

    let mut newline = NewlineGenerator::new();

    for &reg in regs {
        newline.before_item(f);
        // TODO use id in the name?
        let RegisterInfo { defining_item: _, defining_id, domain: sync, ty } = &compiled[reg];

        let name_str = compiled.defining_id_to_readable_string(defining_id);
        let ty_str = verilog_ty_to_str(diag, module_span, type_to_verilog(diag, compiled, map, module_span, ty));
        let sync_str = sync_ty_to_comment_str(source, parsed, compiled, &DomainKind::Sync(sync.clone()), ty);

        let name = SignalName(signal_map.len());
        swriteln!(f, "{I}reg {ty_str}{name}; // reg {name_str:?} {sync_str}");
        signal_map.insert_first(Signal::Reg(reg), name);
    }

    newline.start_new_block();

    for &(wire, ref value) in wires {
        newline.before_item(f);
        let WireInfo { defining_item: _, defining_id, domain, ty, has_declaration_value: _ } = &compiled[wire];

        let name_str = compiled.defining_id_to_readable_string(defining_id);
        let ty_str = verilog_ty_to_str(diag, module_span, type_to_verilog(diag, compiled, map, module_span, ty));
        let comment_info = sync_ty_to_comment_str(source, parsed, compiled, domain, ty);

        let (keyword_str, assign_str) = if let Some(value) = value {
            let value_spanned = Spanned { span: defining_id.span(), inner: value };
            let value_str = value_to_verilog(diag, parsed, compiled, &signal_map, value_spanned);

            let def_str = match value_str {
                Ok(s) => format!(" = {}", s),
                Err(VerilogValueUndefined) => " /* = undefined */".to_string(),
            };
            ("wire", def_str)
        } else {
            ("reg", "".to_string())
        };

        let name = SignalName(signal_map.len());
        swriteln!(f, "{I}{keyword_str} {ty_str}{name}{assign_str}; // wire {name_str:?} {comment_info}");
        signal_map.insert_first(Signal::Wire(wire), name);
    }

    for statement in statements {
        newline.start_new_block();
        newline.before_item(f);

        match statement {
            ModuleStatement::Combinatorial(block) => {
                let ModuleBlockCombinatorial { span: _, statements } = block;
                // TODO collect RHS expressions and use those instead of this star
                // TODO add metadata pointing to source as comments
                swriteln!(f, "{I}always @(*) begin");
                for statement in statements {
                    swriteln!(f, "{I}{I}{}", statement_to_string(diag, parsed, compiled, &signal_map, statement));
                }
                swriteln!(f, "{I}end");
            }
            ModuleStatement::Clocked(block) => {
                let &ModuleBlockClocked {
                    span, ref domain, statements_reset: ref on_reset, statements: ref on_block
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
                    swriteln!(f, "{I}{I}{I}{}", statement_to_string(diag, parsed, compiled, &signal_map, statement));
                }
                swriteln!(f, "{I}{I}end else begin");
                for statement in on_block {
                    swriteln!(f, "{I}{I}{I}{}", statement_to_string(diag, parsed, compiled, &signal_map, statement));
                }
                swriteln!(f, "{I}{I}end");

                swriteln!(f, "{I}end");
            }
            ModuleStatement::Instance(instance) => {
                let &ModuleInstance {
                    module: child,
                    name: ref child_instance_name,
                    generic_arguments: ref child_generic_arguments_raw,
                    port_connections: ref child_port_connections
                } = instance;

                let child_generic_arguments = child_generic_arguments_raw.as_ref().map(|args| {
                    GenericArguments {
                        vec: args.vec.iter().map(|arg| {
                            arg.replace_generics(compiled, map)
                        }).collect(),
                    }
                });
                let child_module_name = todo.push(parsed, compiled, BackModule { item: child, args: child_generic_arguments });

                if let Some(child_instance_name) = child_instance_name {
                    swrite!(f, "{I}{child_module_name} {child_instance_name} (");
                } else {
                    swrite!(f, "{I}{child_module_name} (");
                }

                if !child_port_connections.vec.is_empty() {
                    swriteln!(f);
                    let module_ports = &parsed.module_ast(compiled[child].ast_ref).ports.inner;
                    for (i, (port, connection)) in enumerate(zip_eq(module_ports, &child_port_connections.vec)) {
                        let value_str = value_to_verilog(diag, parsed, compiled, &signal_map, connection.as_ref())
                            .unwrap_or_else(|_: VerilogValueUndefined| "/* undefined */".to_string());

                        swrite!(f, "{I}{I}.{}({})", port.id.string, value_str);
                        // no trailing comma
                        if i != child_port_connections.vec.len() - 1 {
                            swrite!(f, ",");
                        }
                        swriteln!(f);
                    }
                }
                swriteln!(f, "{I});");
            }
            &ModuleStatement::Err(e) => {
                let _: ErrorGuaranteed = e;
                swriteln!(f, "{I}// error statement");
            }
        }
    }

    result
}

fn statement_to_string(
    diag: &Diagnostics,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
    signal_map: &IndexMap<Signal, SignalName>,
    statement: &LowerStatement,
) -> String {
    match statement {
        LowerStatement::Assignment { target, value } => {
            // TODO create shadow variables for all assignments inside blocks, and only assign to those
            //  then finally at the end of the block, non-blocking assign to everything
            let mut commented = false;

            let target_str = value_to_verilog(diag, parsed, compiled, signal_map, target.as_ref());
            let target_str = match &target_str {
                Ok(s) => s.as_str(),
                Err(VerilogValueUndefined) => {
                    commented = true;
                    "undefined"
                }
            };
            let value_str = value_to_verilog(diag, parsed, compiled, signal_map, value.as_ref());
            let value_str = match &value_str {
                Ok(s) => s.as_str(),
                Err(VerilogValueUndefined) => {
                    commented = true;
                    "undefined"
                }
            };

            let prefix = if commented { "// " } else { "" };
            format!("{prefix}{target_str} = {value_str};")
        }
        &LowerStatement::Error(e) => {
            let _: ErrorGuaranteed = e;
            "// error statement".to_string()
        }
    }
}

fn port_to_verilog(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
    map: &GenericMap,
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
            let ty_str = verilog_ty_to_str(diag, defining_id.span, type_to_verilog(diag, compiled, map, defining_id.span, ty));
            let comment = sync_ty_to_comment_str(source, parsed, compiled, sync, ty);
            (ty_str, comment)
        }
    };

    let name_str = &defining_id.string;

    format!("{dir_str} wire {ty_str}{name_str}{comma_str} // {comment}")
}

fn sync_ty_to_comment_str(source: &SourceDatabase, parsed: &ParsedDatabase, compiled: &CompiledDatabase, sync: &DomainKind<Value>, ty: &Type) -> String {
    let sync_str = match sync {
        DomainKind::Sync(SyncDomain { clock, reset }) => {
            let clock_str = compiled.value_to_readable_str(source, parsed, clock);
            let reset_str = compiled.value_to_readable_str(source, parsed, reset);
            format!("sync({}, {})", clock_str, reset_str)
        }
        DomainKind::Async => "async".to_owned(),
    };
    let ty_str = compiled.type_to_readable_str(source, parsed, ty);
    format!("{} {}", sync_str, ty_str)
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

fn value_to_verilog(
    diag: &Diagnostics,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
    signal_map: &IndexMap<Signal, SignalName>,
    value: Spanned<&Value>,
) -> Result<String, VerilogValueUndefined> {
    match value_to_verilog_inner(diag, parsed, compiled, signal_map, value) {
        Ok(value) => Ok(value),
        Err(VerilogValueError::Diag(e)) => {
            let _: ErrorGuaranteed = e;
            Ok("/* error */".to_string())
        }
        Err(VerilogValueError::Undefined) => Err(VerilogValueUndefined),
    }
}

#[derive(Debug, Copy, Clone)]
struct VerilogValueUndefined;

#[derive(Debug, Copy, Clone, derive_more::From)]
enum VerilogValueError {
    Diag(ErrorGuaranteed),
    Undefined,
}

// TODO careful about scoping, are we sure we're never accidentally referring to the wrong value?
fn value_to_verilog_inner(
    diag: &Diagnostics,
    parsed: &ParsedDatabase,
    compiled: &CompiledDatabase,
    signal_map: &IndexMap<Signal, SignalName>,
    value: Spanned<&Value>,
) -> Result<String, VerilogValueError> {
    let Spanned { span, inner: value } = value;
    match value {
        &Value::Error(e) => Err(e.into()),
        &Value::GenericParameter(_) =>
            Err(diag.report_internal_error(span, "generic parameters should not materialize").into()),
        &Value::ModulePort(port) =>
            Ok(parsed.module_port_ast(compiled[port].ast).id.string.clone()),

        &Value::Undefined => Err(VerilogValueError::Undefined),
        &Value::BoolConstant(b) => Ok(if b { "1" } else { "0" }.to_string()),
        Value::IntConstant(i) => Ok(i.to_string()),

        Value::Binary(op, a, b) => {
            // TODO reduce the amount of parentheses
            let op = binary_op_to_verilog(diag, span, *op)?;
            let a = value_to_verilog_inner(diag, parsed, compiled, signal_map, Spanned { span, inner: a })?;
            let b = value_to_verilog_inner(diag, parsed, compiled, signal_map, Spanned { span, inner: b })?;
            Ok(format!("({} {} {})", a, op, b))
        }
        Value::UnaryNot(x) => {
            Ok(format!("(!{})", value_to_verilog_inner(diag, parsed, compiled, signal_map, Spanned { span, inner: x })?))
        }

        &Value::Wire(wire) => {
            match signal_map.get(&Signal::Wire(wire)) {
                None => Err(diag.report_internal_error(span, "wire not found in signal map").into()),
                Some(wire_name) => Ok(format!("{}", wire_name)),
            }
        }
        &Value::Register(reg) => {
            match signal_map.get(&Signal::Reg(reg)) {
                None => Err(diag.report_internal_error(span, "register not found in signal map").into()),
                Some(reg_name) => Ok(format!("{}", reg_name)),
            }
        }
        &Value::Variable(_) =>
            Err(diag.report_todo(span, "value_to_verilog for variables").into()),
        // forward to the inner value
        &Value::Constant(c) =>
            value_to_verilog_inner(diag, parsed, compiled, signal_map, Spanned { span, inner: &compiled[c].value }),

        Value::Never | Value::Unit | Value::StringConstant(_) | Value::Range(_) | Value::FunctionReturn(_) | Value::Module(_) =>
            Err(diag.report_internal_error(span, format!("value '{value:?}' should not materialize")).into()),
    }
}

// TODO check that all of these exist and behave as expected
fn binary_op_to_verilog(diag: &Diagnostics, span: Span, op: BinaryOp) -> Result<&'static str, ErrorGuaranteed> {
    match op {
        BinaryOp::Add => Ok("+"),
        BinaryOp::Sub => Ok("-"),
        BinaryOp::Mul => Ok("*"),
        BinaryOp::Div => Ok("/"),
        BinaryOp::Mod => Ok("%"),
        BinaryOp::Pow => Ok("**"),
        BinaryOp::BitAnd => Ok("&"),
        BinaryOp::BitOr => Ok("|"),
        BinaryOp::BitXor => Ok("^"),
        BinaryOp::BoolAnd => Ok("&&"),
        BinaryOp::BoolOr => Ok("||"),
        BinaryOp::Shl => Ok("<<"),
        BinaryOp::Shr => Ok(">>"),
        BinaryOp::CmpEq => Ok("=="),
        BinaryOp::CmpNeq => Ok("!="),
        BinaryOp::CmpLt => Ok("<"),
        BinaryOp::CmpLte => Ok("<="),
        BinaryOp::CmpGt => Ok(">"),
        BinaryOp::CmpGte => Ok(">="),
        BinaryOp::In => Err(diag.report_todo(span, "binary op 'in'")),
    }
}

fn type_to_verilog(diag: &Diagnostics, compiled: &CompiledDatabase, map: &GenericMap, span: Span, ty: &Type) -> VerilogType {
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
                    match value_evaluate_int(diag, compiled, map, span, n).map(|n| n.to_u32()) {
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
            let range = match value_evaluate_int_range(diag, compiled, map, span, range) {
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

fn value_evaluate_int(diag: &Diagnostics, compiled: &CompiledDatabase, map: &GenericMap, span: Span, value: &Value) -> Result<BigInt, ErrorGuaranteed> {
    match value {
        &Value::Error(e) => Err(e),
        Value::IntConstant(i) => Ok(i.clone()),
        Value::Binary(op, left, right) => {
            let left = value_evaluate_int(diag, compiled, map, span, left)?;
            let right = value_evaluate_int(diag, compiled, map, span, right)?;
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
        Value::GenericParameter(param) => {
            match map.generic_value.get(param) {
                Some(value) => value_evaluate_int(diag, compiled, map, span, value),
                None => Err(diag.report_internal_error(span, "unbound generic parameter")),
            }
        }
        Value::ModulePort(_) => Err(diag.report_todo(span, "value_evaluate_int value ModulePort")),
        Value::Range(_) => Err(diag.report_todo(span, "value_evaluate_int value Range")),
        Value::FunctionReturn(_) => Err(diag.report_todo(span, "value_evaluate_int value Function")),
        Value::Module(_) => Err(diag.report_todo(span, "value_evaluate_int value Module")),
        Value::Wire(_) => Err(diag.report_todo(span, "value_evaluate_int value Wire")),
        Value::Register(_) => Err(diag.report_todo(span, "value_evaluate_int value Reg")),
        Value::Variable(_) => Err(diag.report_todo(span, "value_evaluate_int value Variable")),
        // forward to the inner value
        Value::Constant(c) =>
            value_evaluate_int(diag, compiled, map, span, &compiled[*c].value),
        Value::Never => Err(diag.report_todo(span, "value_evaluate_int value Never")),
        Value::Undefined => Err(diag.report_todo(span, "value_evaluate_int value Undefined")),
    }
}

fn value_evaluate_int_range(diag: &Diagnostics, compiled: &CompiledDatabase, map: &GenericMap, span: Span, value: &Value) -> Result<std::ops::Range<BigInt>, ErrorGuaranteed> {
    match value {
        &Value::Error(e) => Err(e),
        Value::Range(info) => {
            let &RangeInfo { ref start, ref end } = info;

            let (start, end) = match (start, end) {
                (Some(start), Some(end)) => (start, end),
                _ => throw!(diag.report_internal_error(span, "unbound integers should not materialize")),
            };

            let start = value_evaluate_int(diag, compiled, map, span, start.as_ref())?;
            let end = value_evaluate_int(diag, compiled, map, span, end.as_ref())?;

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

#[derive(Debug, Clone, Copy)]
struct NewlineGenerator {
    any_prev: bool,
    any_curr: bool,
}

impl NewlineGenerator {
    pub fn new() -> Self {
        Self {
            any_prev: false,
            any_curr: false,
        }
    }

    pub fn start_new_block(&mut self) {
        self.any_prev |= self.any_curr;
        self.any_curr = false;
    }

    pub fn before_item(&mut self, f: &mut String) {
        if self.any_prev && !self.any_curr {
            swriteln!(f);
        }
        self.any_curr = true;
    }
}

#[cfg(test)]
mod test {
    use crate::back::core::{verilog_type_for_int_range, Signed, VerilogType};
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