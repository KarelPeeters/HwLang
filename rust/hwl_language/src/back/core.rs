use crate::back::todo::{BackModule, BackModuleList, BackModuleName};
use crate::data::compiled::{CompiledDatabase, GenericParameter, Item, ItemChecked, ModulePort, ModulePortInfo, Register, RegisterInfo, Wire, WireInfo};
use crate::data::diagnostic::{Diagnostic, Diagnostics, ErrorGuaranteed};
use crate::data::lowered::LoweredDatabase;
use crate::data::module_body::{LowerBlock, LowerIfStatement, LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleChecked, ModuleInstance, ModuleStatement};
use crate::data::parsed::ParsedDatabase;
use crate::data::source::SourceDatabase;
use crate::front::common::{GenericContainer, GenericMap, ScopedEntry, TypeOrValue};
use crate::front::module::Driver;
use crate::front::scope::Visibility;
use crate::front::types::{Constructor, GenericArguments, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{BinaryOp, DomainKind, Identifier, MaybeIdentifier, PortDirection, PortKind, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::{result_pair, ResultExt};
use crate::{swrite, swriteln, throw};
use indexmap::IndexMap;
use itertools::{enumerate, zip_eq, Itertools};
use num_bigint::BigInt;
use num_traits::{Signed as _, ToPrimitive};
use std::cmp::max;
use std::fmt::{Display, Formatter, Write};
use std::ops::RangeInclusive;
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

    // TODO remove this clone once generics are implemented to avoid substitution
    let body = unwrap_match!(&item_info.body, ItemChecked::Module(body) => body).clone();

    // TODO build this in a single pass?
    let mut port_string = String::new();
    for (port_index, &port) in enumerate(&module_info.ports) {
        if port_index == 0 {
            port_string.push_str("\n");
        }
        let comma_str = if port_index == module_info.ports.len() - 1 { "" } else { "," };

        let driver = body.output_port_driver.get(&port).copied();

        port_string.push_str(I);
        port_string.push_str(&port_to_verilog(diag, source, parsed, compiled, &generic_map, port, driver, comma_str));
        port_string.push_str("\n");
    }

    let body_str = module_body_to_verilog(diag, source, parsed, compiled, todo, &generic_map, item_ast.span, &body);
    let module_id_str = &item_ast.id.string;

    let comment_str = format!("// module {module_id_str:?}{generics_comment}");
    format!("{comment_str}module {module_name} ({port_string});\n{body_str}endmodule\n")
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum Signal {
    Reg(Register),
    Wire(Wire),
    Port(ModulePort),
}

impl Signal {
    fn to_value(self) -> Value {
        match self {
            Signal::Reg(reg) => Value::Register(reg),
            Signal::Wire(wire) => Value::Wire(wire),
            Signal::Port(port) => Value::ModulePort(port),
        }
    }

    fn ty(self, compiled: &CompiledDatabase) -> &Type {
        match self {
            Signal::Reg(reg) => &compiled[reg].ty,
            Signal::Wire(wire) => &compiled[wire].ty,
            Signal::Port(port) => &compiled[port].kind.ty(),
        }
    }

    fn defining_id<'d>(self, parsed: &'d ParsedDatabase, compiled: &'d CompiledDatabase) -> MaybeIdentifier<&'d Identifier> {
        match self {
            Signal::Reg(reg) => compiled[reg].defining_id.as_ref(),
            Signal::Wire(wire) => compiled[wire].defining_id.as_ref(),
            Signal::Port(port) => MaybeIdentifier::Identifier(parsed.module_port_ast(compiled[port].ast).id()),
        }
    }
}

// TODO rework this, try to use the original name or some reasonable derivative
#[derive(Debug, Copy, Clone)]
struct SignalName {
    kind: SignalNameKind,
    index: usize,
}

#[derive(Debug, Copy, Clone)]

enum SignalNameKind {
    ModuleReg,
    ModuleWire,
    LocalVar,
}

impl Display for SignalName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let &SignalName { kind, index } = self;

        let prefix = match kind {
            SignalNameKind::ModuleReg => "module_reg",
            SignalNameKind::ModuleWire => "module_wire",
            SignalNameKind::LocalVar => "local_var",
        };

        write!(f, "{prefix}_{index}")
    }
}

#[derive(Debug, Copy, Clone)]
struct Indent {
    depth: usize,
}

impl Indent {
    pub fn new(depth: usize) -> Indent {
        Indent { depth }
    }

    pub fn nest(self) -> Indent {
        Indent { depth: self.depth + 1 }
    }
}

impl Display for Indent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.depth {
            write!(f, "{I}")?;
        }
        Ok(())
    }
}

fn module_body_to_verilog(
    diag: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
    todo: &mut BackModuleList,
    generic_map: &GenericMap,
    module_span: Span,
    body: &ModuleChecked,
) -> String {
    let mut result = String::new();
    let f = &mut result;

    let ModuleChecked { statements, regs, wires, output_port_driver: _ } = body;
    let mut signal_map = IndexMap::new();

    let mut newline = NewlineGenerator::new();

    for &reg in regs {
        newline.before_item(f);
        // TODO use id in the name?
        let RegisterInfo { defining_item: _, defining_id, domain: sync, ty } = &compiled[reg];

        let name_str = compiled.defining_id_to_readable_string(defining_id.as_ref());
        let ty_str = verilog_ty_to_str(diag, module_span, type_to_verilog(diag, compiled, generic_map, module_span, ty));
        let sync_str = sync_ty_to_comment_str(source, parsed, compiled, &DomainKind::Sync(sync.clone()), ty);

        let name = SignalName { kind: SignalNameKind::ModuleReg, index: signal_map.len() };
        swriteln!(f, "{I}reg {ty_str}{name}; // reg {name_str:?} {sync_str}");
        signal_map.insert_first(Signal::Reg(reg), name);
    }

    newline.start_new_block();

    for &(wire, ref value) in wires {
        newline.before_item(f);
        let WireInfo { defining_item: _, defining_id, domain, ty, has_declaration_value: _ } = &compiled[wire];

        let name_str = compiled.defining_id_to_readable_string(defining_id.as_ref()).to_owned();
        let ty_str = verilog_ty_to_str(diag, module_span, type_to_verilog(diag, compiled, generic_map, module_span, ty));
        let comment_info = sync_ty_to_comment_str(source, parsed, compiled, domain, ty);

        let (keyword_str, assign_str) = if let Some(value) = value {
            let value_spanned = Spanned { span: defining_id.span(), inner: value };
            let value_str = value_to_verilog(diag, parsed, compiled, generic_map, &signal_map, value_spanned);

            let def_str = match value_str {
                Ok(s) => format!(" = {}", s),
                Err(VerilogValueUndefined) => " /* = undefined */".to_string(),
            };
            ("wire", def_str)
        } else {
            ("reg", "".to_string())
        };

        let name = SignalName { kind: SignalNameKind::ModuleWire, index: signal_map.len() };

        swriteln!(f, "{I}{keyword_str} {ty_str}{name}{assign_str}; // wire {name_str:?} {comment_info}");
        signal_map.insert_first(Signal::Wire(wire), name);
    }

    for statement in statements {
        newline.start_new_block();
        newline.before_item(f);

        match statement {
            ModuleStatement::Combinatorial(block) => {
                let ModuleBlockCombinatorial { span: _, block } = block;
                // TODO collect RHS expressions and use those instead of this star
                // TODO add metadata pointing to source as comments
                swriteln!(f, "{I}always @(*) begin");
                block_to_verilog(diag, parsed, compiled, generic_map, &signal_map, block, f, Indent::new(2));
                swriteln!(f, "{I}end");
            }
            ModuleStatement::Clocked(block) => {
                clocked_block_to_verilog(diag, parsed, compiled, generic_map, &signal_map, f, block);
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
                            arg.replace_generics(compiled, generic_map)
                        }).collect(),
                    }
                });
                let child_module_name = todo.push(parsed, compiled, BackModule { item: child, args: child_generic_arguments });

                if let Some(child_instance_name) = child_instance_name {
                    swrite!(f, "{I}{child_module_name} {child_instance_name} (");
                } else {
                    swrite!(f, "{I}{child_module_name} (");
                }

                if child_port_connections.vec.is_empty() {
                    swriteln!(f, ");");
                } else {
                    swriteln!(f);

                    let module_ports = &compiled.module_info.get(&child).as_ref().unwrap().ports;
                    assert_eq!(module_ports.len(), child_port_connections.vec.len());

                    // manual for loop for lifetime reasons
                    // TODO clean this up once compiled does no longer need to be mutable
                    for i in 0..module_ports.len() {
                        let module_ports = &compiled.module_info.get(&child).as_ref().unwrap().ports;

                        let port = module_ports[i];
                        let connection = &child_port_connections.vec[i];

                        let port_id = parsed.module_port_ast(compiled[port].ast).id();
                        let value_str = value_to_verilog(diag, parsed, compiled, generic_map, &signal_map, connection.as_ref())
                            .unwrap_or_else(|_: VerilogValueUndefined| "".to_string());
                        swrite!(f, "{I}{I}.{}({})", port_id.string, value_str);

                        // no trailing comma
                        if i != child_port_connections.vec.len() - 1 {
                            swrite!(f, ",");
                        }
                        swriteln!(f);
                    }
                    swriteln!(f, "{I});");
                }
            }
            &ModuleStatement::Err(e) => {
                let _: ErrorGuaranteed = e;
                swriteln!(f, "{I}// error statement");
            }
        }
    }

    result
}

fn clocked_block_to_verilog(
    diag: &Diagnostics,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
    generic_map: &GenericMap,
    signal_map: &IndexMap<Signal, SignalName>,
    f: &mut String,
    block: &ModuleBlockClocked,
) {
    let &ModuleBlockClocked {
        span, ref domain, ref on_reset, ref on_clock
    } = block;

    // figure out sensitivity list
    let SyncDomain { clock, reset } = domain;

    let sensitivity_value_to_string = |value: &Value| -> (&str, &str, &str) {
        match value {
            Value::Error(_) =>
                return ("0 /* error */", "posedge", ""),
            &Value::ModulePort(port) =>
                return (&parsed.module_port_ast(compiled[port].ast).id().string, "posedge", ""),
            Value::UnaryNot(inner) => {
                if let &Value::ModulePort(port) = &**inner {
                    return (&parsed.module_port_ast(compiled[port].ast).id().string, "negedge", "!");
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

    // collect written regs and ports for shadowing
    let mut inner_signal_map = IndexMap::new();
    collect_clocked_block_written_externals(diag, on_clock, &mut |s| {
        let next_index = inner_signal_map.len();
        inner_signal_map.entry(s).or_insert_with(|| {
            SignalName { kind: SignalNameKind::LocalVar, index: next_index }
        });
    });

    // combine signal maps
    // inner map overrides outer map, reads/writes should have to the inner shadow variables
    let mut combined_signal_map = signal_map.clone();
    for (&k, &v) in &inner_signal_map {
        combined_signal_map.insert(k, v);
    }

    // block signature
    let mut newline = NewlineGenerator::new();
    swriteln!(f, "{I}always @({clock_edge} {clock_str} or {reset_edge} {reset_str}) begin");

    // define local variables
    for (&k, &v) in &inner_signal_map {
        newline.before_item(f);

        let ty = k.ty(compiled);
        let defining_id = k.defining_id(parsed, compiled);

        let ty_str = verilog_ty_to_str(diag, defining_id.span(), type_to_verilog(diag, compiled, generic_map, defining_id.span(), ty));
        let name_str = compiled.defining_id_to_readable_string(defining_id);

        swriteln!(f, "{I}{I}reg {}{}; // local copy of \"{}\" ", ty_str, v, name_str);
    }

    newline.start_new_block();
    newline.before_item(f);

    // reset block
    swriteln!(f, "{I}{I}if ({reset_prefix}{reset_str}) begin");
    block_to_verilog(diag, parsed, compiled, generic_map, &signal_map, on_reset, f, Indent::new(3));
    swriteln!(f, "{I}{I}end else begin");

    // copy signals to local variables
    for (&k, &v) in &inner_signal_map {
        newline.before_item(f);

        let value_spanned = Spanned { span: k.defining_id(parsed, compiled).span(), inner: &k.to_value() };
        let signal_outer_verilog = value_to_verilog(diag, parsed, compiled, generic_map, &signal_map, value_spanned).unwrap();

        swriteln!(f, "{I}{I}{I}{} = {};", v, signal_outer_verilog);
    }

    // clock block
    newline.start_new_block();
    if on_clock.statements.len() > 0 {
        newline.before_item(f);
    }
    block_to_verilog(diag, parsed, compiled, generic_map, &combined_signal_map, on_clock, f, Indent::new(3));

    // copy local variables back to signals
    newline.start_new_block();
    for (&k, &v) in &inner_signal_map {
        newline.before_item(f);

        let value_spanned = Spanned { span: k.defining_id(parsed, compiled).span(), inner: &k.to_value() };
        let signal_outer_verilog = value_to_verilog(diag, parsed, compiled, generic_map, &signal_map, value_spanned).unwrap();

        // use blocking assignments here
        swriteln!(f, "{I}{I}{I}{} <= {};", signal_outer_verilog, v);
    }

    // end
    swriteln!(f, "{I}{I}end");
    swriteln!(f, "{I}end");
}

fn collect_clocked_block_written_externals(diag: &Diagnostics, block: &LowerBlock, report: &mut impl FnMut(Signal)) {
    let LowerBlock { statements } = block;

    for stmt in statements {
        match &stmt.inner {
            LowerStatement::Error(_) => {}
            LowerStatement::Block(block) => collect_clocked_block_written_externals(diag, block, report),
            // for now expressions can't contain statements, no no assignments either
            LowerStatement::Expression(_) => {}
            LowerStatement::Assignment { target, value: _ } => {
                match &target.inner {
                    // collect registers and ports
                    &Value::ModulePort(port) => report(Signal::Port(port)),
                    &Value::Register(reg) => report(Signal::Reg(reg)),
                    // variables are already local and don't need to be tracked
                    Value::Variable(_) => {}
                    // ignore errors
                    Value::Error(_) => {}

                    // wires shouldn't be write targets in clocked blocks, if they are it's an error anyway
                    // TODO properly mark this as an error target, instead of connecting the wire through
                    Value::Wire(_) => {}
                    // invalid assignment targets
                    Value::GenericParameter(_) |
                    Value::Never | Value::Unit | Value::Undefined |
                    Value::BoolConstant(_) | Value::IntConstant(_) | Value::StringConstant(_) |
                    Value::Range(_) | Value::Binary(_, _, _) | Value::UnaryNot(_) |
                    Value::FunctionReturn(_) | Value::Module(_) |
                    Value::Constant(_) => {
                        diag.report_internal_error(stmt.span, format!("assignment in clocked block to invalid target {:?}", target.inner));
                    }
                }
            }
            LowerStatement::If(stmt) => {
                let LowerIfStatement { condition: _, then_block, else_block } = stmt;
                collect_clocked_block_written_externals(diag, then_block, report);
                if let Some(else_block) = else_block {
                    collect_clocked_block_written_externals(diag, else_block, report);
                }
            }
            // TODO
            LowerStatement::For => {}
            LowerStatement::While => {}
            LowerStatement::Return(_) => {}
        }
    }
}

fn block_to_verilog(
    diag: &Diagnostics,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
    generic_map: &GenericMap,
    signal_map: &IndexMap<Signal, SignalName>,
    block: &LowerBlock,
    f: &mut String,
    indent: Indent,
) {
    let LowerBlock { statements } = block;
    for statement in statements {
        statement_to_verilog(diag, parsed, compiled, generic_map, signal_map, statement.as_ref(), f, indent);
    }
}

fn statement_to_verilog(
    diag: &Diagnostics,
    parsed: &ParsedDatabase,
    compiled: &mut CompiledDatabase,
    generic_map: &GenericMap,
    signal_map: &IndexMap<Signal, SignalName>,
    statement: Spanned<&LowerStatement>,
    f: &mut String,
    indent: Indent,
) {
    match statement.inner {
        LowerStatement::Block(block) => {
            swriteln!(f, "{indent}begin");
            block_to_verilog(diag, parsed, compiled, generic_map, signal_map, block, f, indent.nest());
            swriteln!(f, "{indent}end");
        }
        LowerStatement::Expression(value) => {
            let value_str = value_to_verilog(diag, parsed, compiled, generic_map, signal_map, value.as_ref());
            let value_str = match &value_str {
                Ok(s) => s.as_str(),
                Err(VerilogValueUndefined) => {
                    swriteln!(f, "{indent}// undefined");
                    return;
                }
            };
            swriteln!(f, "{indent}{value_str};");
        }
        LowerStatement::Assignment { target, value } => {
            // TODO create shadow variables for all assignments inside blocks, and only assign to those
            //  then finally at the end of the block, non-blocking assign to everything
            let mut commented = false;

            let target_str = value_to_verilog(diag, parsed, compiled, generic_map, signal_map, target.as_ref());
            let target_str = match &target_str {
                Ok(s) => s.as_str(),
                Err(VerilogValueUndefined) => {
                    commented = true;
                    "undefined"
                }
            };
            let value_str = value_to_verilog(diag, parsed, compiled, generic_map, signal_map, value.as_ref());
            let value_str = match &value_str {
                Ok(s) => s.as_str(),
                Err(VerilogValueUndefined) => {
                    commented = true;
                    "undefined"
                }
            };

            let prefix = if commented { "// " } else { "" };
            swriteln!(f, "{indent}{prefix}{target_str} = {value_str};");
        }
        LowerStatement::If(statement) => {
            // TODO if the condition is constant, only emit the relevant branch
            let LowerIfStatement { condition, then_block, else_block } = statement;

            let condition_str = value_to_verilog(diag, parsed, compiled, generic_map, signal_map, condition.as_ref());
            let condition_str = match &condition_str {
                Ok(s) => s.as_str(),
                Err(VerilogValueUndefined) => {
                    swriteln!(f, "{indent}// if(undefined)");
                    return;
                }
            };

            // TODO if the else branch is just a single if statement again, don't keep indenting
            swriteln!(f, "{indent}if ({condition_str}) begin");
            block_to_verilog(diag, parsed, compiled, generic_map, signal_map, then_block, f, indent.nest());

            match else_block {
                Some(else_block) => {
                    swriteln!(f, "{indent}end else begin");
                    block_to_verilog(diag, parsed, compiled, generic_map, signal_map, else_block, f, indent.nest());
                    swriteln!(f, "{indent}end");
                }
                None => {
                    swriteln!(f, "{indent}end");
                }
            }
        }
        LowerStatement::For => {
            diag.report_internal_error(statement.span, "lower for loop");
            swriteln!(f, "{indent}// TODO lower for loop");
        }
        LowerStatement::While => {
            diag.report_todo(statement.span, "while loops should never materialize");
            swriteln!(f, "{indent}// error while loop");
        }
        LowerStatement::Return(_) => {
            diag.report_internal_error(statement.span, "return should never materialize");
            swriteln!(f, "{indent}// error return");
        }
        &LowerStatement::Error(e) => {
            let _: ErrorGuaranteed = e;
            swriteln!(f, "{indent}// error statement");
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
    driver: Option<Driver>,
    comma_str: &str,
) -> String {
    let &ModulePortInfo {
        ast,
        direction,
        ref kind
    } = &compiled[port];
    let defining_id = &parsed.module_port_ast(ast).id();

    let dir_str = match direction {
        PortDirection::Input => "input",
        PortDirection::Output => "output",
    };

    let driver_str = match driver {
        None | Some(Driver::WireDeclaration | Driver::InstancePortConnection(_)) => "wire",
        Some(Driver::CombinatorialBlock(_) | Driver::ClockedBlock(_)) => "reg",
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

    format!("{dir_str} {driver_str} {ty_str}{name_str}{comma_str} // {comment}")
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

impl VerilogType {
    fn width(self) -> Result<u32, ErrorGuaranteed> {
        match self {
            VerilogType::SingleBit => Ok(1),
            VerilogType::MultiBit(_, w) => Ok(w),
            VerilogType::Error(e) => Err(e),
        }
    }
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
    compiled: &mut CompiledDatabase,
    generic_map: &GenericMap,
    signal_map: &IndexMap<Signal, SignalName>,
    value: Spanned<&Value>,
) -> Result<String, VerilogValueUndefined> {
    let value_replaced = value.map_inner(|value| value.replace_generics(compiled, generic_map));

    match value_to_verilog_inner(diag, parsed, compiled, signal_map, value_replaced.as_ref()) {
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

        &Value::ModulePort(port) => {
            match signal_map.get(&Signal::Port(port)) {
                None => Ok(parsed.module_port_ast(compiled[port].ast).id().string.clone()),
                Some(name) => Ok(format!("{}", name)),
            }
        }
        &Value::Wire(wire) => {
            match signal_map.get(&Signal::Wire(wire)) {
                None => Err(diag.report_internal_error(span, "wire not found in signal map").into()),
                Some(name) => Ok(format!("{}", name)),
            }
        }
        &Value::Register(reg) => {
            match signal_map.get(&Signal::Reg(reg)) {
                None => Err(diag.report_internal_error(span, "register not found in signal map").into()),
                Some(name) => Ok(format!("{}", name)),
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
        BinaryOp::BoolXor => Ok("^"),
        BinaryOp::Shl => Ok("<<"),
        BinaryOp::Shr => Ok(">>"),
        BinaryOp::CmpEq => Ok("=="),
        BinaryOp::CmpNeq => Ok("!="),
        BinaryOp::CmpLt => Ok("<"),
        BinaryOp::CmpLte => Ok("<="),
        BinaryOp::CmpGt => Ok(">"),
        BinaryOp::CmpGte => Ok(">="),
        BinaryOp::In => Err(diag.report_todo(span, "lower binary op 'in'")),
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
        Type::Array(inner, n) => {
            let w = type_to_verilog(diag, compiled, map, span, inner).width();
            let n = value_evaluate_int(diag, compiled, map, span, n);

            match result_pair(w, n) {
                Ok((w, n)) => {
                    match diag_big_int_to_u32(diag, span, &(w * n), "total array bit width too large") {
                        Ok(total) => VerilogType::MultiBit(Signed::Unsigned, total),
                        Err(e) => VerilogType::Error(e),
                    }
                }
                Err(e) => VerilogType::Error(e),
            }
        }
        Type::Integer(info) => {
            let IntegerTypeInfo { range } = info;
            let range = match value_evaluate_int_range(diag, compiled, map, span, range) {
                Ok(range) => range,
                Err(e) => return VerilogType::Error(e),
            };
            verilog_type_for_int_range(diag, span, range)
        }
        // TODO convert all of these to their bit representations, or split them up into multiple ports?
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
                    diag_big_int_to_u32(diag, span, &right, "power exponent too large")
                        .map(|right| left.pow(right))
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

fn value_evaluate_int_range(diag: &Diagnostics, compiled: &CompiledDatabase, map: &GenericMap, span: Span, value: &Value) -> Result<RangeInclusive<BigInt>, ErrorGuaranteed> {
    match value {
        &Value::Error(e) => Err(e),
        Value::Range(info) => {
            let &RangeInfo { ref start_inc, ref end_inc } = info;

            let (start, end) = match (start_inc, end_inc) {
                (Some(start), Some(end)) => (start, end),
                _ => throw!(diag.report_internal_error(span, "unbound integers should not materialize")),
            };

            let start = value_evaluate_int(diag, compiled, map, span, start.as_ref())?;
            let end = value_evaluate_int(diag, compiled, map, span, end.as_ref())?;

            Ok(start..=end)
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
fn verilog_type_for_int_range(diag: &Diagnostics, span: Span, range: RangeInclusive<BigInt>) -> VerilogType {
    assert!(!range.is_empty(), "Range needs to contain at least one value, got {range:?}");
    let (start, end) = range.into_inner();

    let (signed, bits) = if start < BigInt::ZERO {
        // signed
        // prevent max value underflow
        let max_value = if end.is_negative() {
            BigInt::ZERO
        } else {
            end
        };
        let max_bits = max(
            1 + (start + 1u32).bits(),
            1 + max_value.bits(),
        );

        (Signed::Signed, max_bits)
    } else {
        // unsigned
        (Signed::Unsigned, end.bits())
    };

    match diag_u64_to_u32(diag, span, bits, "integer range too large") {
        Ok(bits) => VerilogType::MultiBit(signed, bits),
        Err(e) => VerilogType::Error(e)
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
        let range_inc = range.start.into()..=(range.end - 1).into();
        let result = verilog_type_for_int_range(&diag, span, range_inc);
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

fn diag_u64_to_u32(diag: &Diagnostics, span: Span, value: u64, message: &str) -> Result<u32, ErrorGuaranteed> {
    value.try_into().map_err(|_| {
        diag.report_simple(
            format!("{message}: overflow when converting {value} to u32"),
            span,
            "used here",
        )
    })
}

fn diag_big_int_to_u32(diag: &Diagnostics, span: Span, value: &BigInt, message: &str) -> Result<u32, ErrorGuaranteed> {
    value.try_into().map_err(|_| {
        diag.report_simple(
            format!("{message}: overflow when converting {value} to u32"),
            span,
            "used here",
        )
    })
}