use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAssignmentTargetBase, IrBlock, IrBoolBinaryOp, IrClockedProcess,
    IrCombinatorialProcess, IrExpression, IrIfStatement, IrIntArithmeticOp, IrIntCompareOp, IrModule, IrModuleChild,
    IrModuleInfo, IrModuleInstance, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrStatement,
    IrTargetStep, IrType, IrVariable, IrVariableInfo, IrVariables, IrWire, IrWireInfo, IrWireOrPort,
};
use crate::front::types::HardwareType;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, Spanned, SyncDomain};
use crate::syntax::parsed::ParsedDatabase;
use crate::syntax::pos::Span;
use crate::syntax::source::SourceDatabase;
use crate::util::arena::Arena;
use crate::util::data::IndexMapExt;
use crate::util::int::IntRepresentation;
use crate::util::{Indent, ResultExt};
use crate::{swrite, swriteln, throw};
use indexmap::IndexMap;
use itertools::enumerate;
use lazy_static::lazy_static;
use num_bigint::{BigInt, BigUint};
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::num::NonZeroU32;

use super::ir::{ir_modules_topological_sort, IrModules};

#[derive(Debug, Clone)]
pub struct LoweredVerilog {
    pub verilog_source: String,

    // TODO should this be a string or a lowered name? we don't want to expose too many implementation details
    pub top_module_name: String,
    pub debug_info_module_map: IndexMap<IrModule, String>,
}

// TODO make backend configurable between verilog and VHDL?
// TODO ban keywords
// TODO should we still be doing diagnostics here, or should lowering just never start?
pub fn lower(
    diags: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    modules: &IrModules,
    top_module: IrModule,
) -> Result<LoweredVerilog, ErrorGuaranteed> {
    // the fact that we're not using `parsed` is good, all information should be contained in `compiled`
    // ideally `source` would also not be used
    let _ = parsed;

    let mut ctx = LowerContext {
        diags,
        source,
        modules,
        module_map: IndexMap::new(),
        top_name_scope: LoweredNameScope::default(),
        lowered_modules: vec![],
    };

    let modules = ir_modules_topological_sort(modules, top_module);
    for module in modules {
        let result = lower_module(&mut ctx, module)?;
        ctx.module_map.insert_first(module, result);
    }

    let top_name = ctx.module_map.get(&top_module).unwrap().name.clone();
    Ok(LoweredVerilog {
        verilog_source: ctx.lowered_modules.join("\n\n"),
        top_module_name: top_name.0.clone(),
        debug_info_module_map: ctx.module_map.into_iter().map(|(k, v)| (k, v.name.0)).collect(),
    })
}

struct LowerContext<'a> {
    diags: &'a Diagnostics,
    source: &'a SourceDatabase,
    modules: &'a IrModules,
    module_map: IndexMap<IrModule, LoweredModule>,
    top_name_scope: LoweredNameScope,
    lowered_modules: Vec<String>,
}

#[derive(Debug, Clone)]
struct LoweredName(String);

#[derive(Debug, Clone)]
struct LoweredModule {
    name: LoweredName,
    ports: IndexMap<IrPort, LoweredName>,
}

#[derive(Default)]
struct LoweredNameScope {
    used: HashSet<String>,
}

impl LoweredNameScope {
    pub fn exact_for_new_id(&mut self, diags: &Diagnostics, id: &Identifier) -> Result<LoweredName, ErrorGuaranteed> {
        check_identifier_valid(
            diags,
            Spanned {
                span: id.span,
                inner: &id.string,
            },
        )?;

        if !self.used.insert(id.string.clone()) {
            throw!(diags.report_internal_error(
                id.span,
                format!("lowered identifier `{}` already used its scope", id.string)
            ))
        }

        Ok(LoweredName(id.string.clone()))
    }

    pub fn make_unique_maybe_id(
        &mut self,
        diags: &Diagnostics,
        id: &MaybeIdentifier,
    ) -> Result<LoweredName, ErrorGuaranteed> {
        self.make_unique_str(diags, id.span(), id.string())
    }

    pub fn make_unique_id(&mut self, diags: &Diagnostics, id: &Identifier) -> Result<LoweredName, ErrorGuaranteed> {
        self.make_unique_str(diags, id.span, &id.string)
    }

    pub fn make_unique_str(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        string: &str,
    ) -> Result<LoweredName, ErrorGuaranteed> {
        check_identifier_valid(diags, Spanned { span, inner: string })?;

        if self.used.insert(string.to_owned()) {
            return Ok(LoweredName(string.to_owned()));
        }

        // TODO speed this up?
        for i in 0u32.. {
            let suffixed = format!("{}_{}", string, i);
            if self.used.insert(suffixed.clone()) {
                return Ok(LoweredName(suffixed));
            }
        }

        throw!(diags.report_internal_error(
            span,
            format!("failed to generate unique lowered identifier for `{}`", string)
        ))
    }
}

// TODO replace with name mangling that forces everything to be valid
fn check_identifier_valid(diags: &Diagnostics, id: Spanned<&str>) -> Result<(), ErrorGuaranteed> {
    let s = id.inner;

    if s.is_empty() {
        throw!(diags.report_simple(
            "invalid verilog identifier: identifier cannot be empty",
            id.span,
            "identifier used here"
        ))
    }
    let first = s.chars().next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        throw!(diags.report_simple(
            "invalid verilog identifier: first character must be alphabetic or underscore",
            id.span,
            "identifier used here"
        ))
    }
    for c in s.chars() {
        if !(c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '$' || c == '_') {
            throw!(diags.report_simple(
                format!("invalid verilog identifier: character `{c}` not allowed in identifier"),
                id.span,
                "identifier used here"
            ))
        }
    }

    Ok(())
}

#[derive(Debug, Copy, Clone)]
struct NameMap<'a> {
    ports: &'a IndexMap<IrPort, LoweredName>,
    registers: &'a IndexMap<IrRegister, LoweredName>,
    wires: &'a IndexMap<IrWire, LoweredName>,
    variables: &'a IndexMap<IrVariable, LoweredName>,
}

fn lower_module(ctx: &mut LowerContext, module: IrModule) -> Result<LoweredModule, ErrorGuaranteed> {
    let diags = ctx.diags;
    assert!(!ctx.module_map.contains_key(&module));

    // TODO careful with name scoping: we don't want eg. ports to accidentally shadow other modules
    //   or maybe verilog has separate namespaces, then it's fine

    let module_info = &ctx.modules[module];
    let IrModuleInfo {
        debug_info_id,
        debug_info_generic_args,
        ports,
        registers,
        wires,
        children: processes,
    } = module_info;
    let module_name = ctx.top_name_scope.make_unique_id(diags, debug_info_id)?;

    let mut f = String::new();

    swriteln!(f, "// module {}", debug_info_id.string);
    swriteln!(
        f,
        "//   defined in \"{}\"",
        ctx.source[debug_info_id.span.start.file].path_raw
    );

    if let Some(generic_args) = debug_info_generic_args {
        swriteln!(f, "//   instantiated with generic arguments:");
        for (arg_name, arg_value) in generic_args {
            swriteln!(f, "//     {}={}", arg_name.string, arg_value.to_diagnostic_string());
        }
    }

    let mut module_name_scope = LoweredNameScope::default();

    swrite!(f, "module {}(", module_name);
    let port_name_map = lower_module_ports(diags, ports, &mut module_name_scope, &mut f)?;
    swriteln!(f, ");");

    let mut newline = NewlineGenerator::new();
    let (reg_name_map, wire_name_map) =
        lower_module_signals(diags, &mut module_name_scope, registers, wires, &mut newline, &mut f)?;

    lower_module_statements(
        ctx,
        &mut module_name_scope,
        &port_name_map,
        &reg_name_map,
        &wire_name_map,
        registers,
        processes,
        &mut newline,
        &mut f,
    )?;

    swriteln!(f, "endmodule");

    let lowered_module = LoweredModule {
        name: module_name,
        ports: port_name_map,
    };

    ctx.lowered_modules.push(f);
    Ok(lowered_module)
}

fn lower_module_ports(
    diags: &Diagnostics,
    ports: &Arena<IrPort, IrPortInfo>,
    module_name_scope: &mut LoweredNameScope,
    f: &mut String,
) -> Result<IndexMap<IrPort, LoweredName>, ErrorGuaranteed> {
    let mut port_lines = vec![];
    let mut port_name_map = IndexMap::new();
    let mut last_actual_port_index = None;

    for (port_index, (port, port_info)) in enumerate(ports) {
        let IrPortInfo {
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
            direction,
            ty,
        } = port_info;

        // TODO check that port names are valid and unique
        let lower_name = module_name_scope.exact_for_new_id(diags, debug_info_id)?;
        let port_ty = VerilogType::from_ir_ty(diags, debug_info_id.span, ty)?;

        let ty_prefix_str = port_ty.to_prefix_str();
        let (is_actual_port, ty_str) = match ty_prefix_str.as_ref().map(|s| s.as_str()) {
            Ok(ty_str) => (true, ty_str),
            Err(ZeroWidth) => (false, "[empty]"),
        };
        let dir_str = match direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
        };
        let ty_debug_str = debug_info_ty.to_diagnostic_string();

        if is_actual_port {
            last_actual_port_index = Some(port_index);
        }
        port_lines.push((
            is_actual_port,
            format!("{dir_str} wire {ty_str}{lower_name}"),
            format!("{debug_info_domain} {ty_debug_str}"),
        ));

        port_name_map.insert_first(port, lower_name);
    }

    for (port_index, (is_actual_port, main_str, comment_str)) in enumerate(port_lines) {
        swrite!(f, "\n    ");

        let start_str = if is_actual_port { "" } else { "//" };
        let end_str = if Some(port_index) == last_actual_port_index {
            ""
        } else {
            ","
        };

        swrite!(f, "{start_str}{main_str}{end_str} // {comment_str}")
    }

    if ports.len() > 0 {
        swriteln!(f);
    }

    Ok(port_name_map)
}

fn lower_module_signals(
    diags: &Diagnostics,
    module_name_scope: &mut LoweredNameScope,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    wires: &Arena<IrWire, IrWireInfo>,
    newline: &mut NewlineGenerator,
    f: &mut String,
) -> Result<(IndexMap<IrRegister, LoweredName>, IndexMap<IrWire, LoweredName>), ErrorGuaranteed> {
    let mut lower_signal = |signal_type,
                            ty,
                            debug_info_id,
                            debug_info_ty: &HardwareType,
                            debug_info_domain,
                            f: &mut String| {
        let name = module_name_scope.make_unique_maybe_id(diags, debug_info_id)?;
        let ty_prefix_str = VerilogType::from_ir_ty(diags, debug_info_id.span(), ty)?.to_prefix_str();
        let (prefix_str, ty_prefix_str) = match ty_prefix_str.as_ref_ok() {
            Ok(ty_prefix_str) => ("", ty_prefix_str.as_str()),
            Err(ZeroWidth) => ("// ", "[empty]"),
        };

        let name_debug_str = &debug_info_id.string();
        let ty_debug_str = debug_info_ty.to_diagnostic_string();

        // both regs and wires lower to verilog "regs", which are really just "signals that are written by processes"
        swriteln!(f, "{I}{prefix_str}reg {ty_prefix_str}{name}; // {signal_type} {name_debug_str}: {debug_info_domain} {ty_debug_str}");
        Ok(name)
    };

    let mut reg_name_map = IndexMap::new();
    let mut wire_name_map = IndexMap::new();

    newline.start_new_block();
    for (register, register_info) in registers {
        newline.before_item(f);

        let IrRegisterInfo {
            ty,
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
        } = register_info;
        let name = lower_signal("reg", ty, debug_info_id, debug_info_ty, debug_info_domain, f)?;
        reg_name_map.insert_first(register, name);
    }

    newline.start_new_block();
    for (wire, wire_info) in wires {
        newline.before_item(f);

        let IrWireInfo {
            ty,
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
        } = &wire_info;
        let name = lower_signal("wire", ty, debug_info_id, debug_info_ty, debug_info_domain, f)?;
        wire_name_map.insert_first(wire, name);
    }

    Ok((reg_name_map, wire_name_map))
}

fn lower_module_statements(
    ctx: &mut LowerContext,
    module_name_scope: &mut LoweredNameScope,
    port_name_map: &IndexMap<IrPort, LoweredName>,
    reg_name_map: &IndexMap<IrRegister, LoweredName>,
    wire_name_map: &IndexMap<IrWire, LoweredName>,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    children: &[IrModuleChild],
    newline: &mut NewlineGenerator,
    f: &mut String,
) -> Result<(), ErrorGuaranteed> {
    let diags = ctx.diags;

    for child in children {
        newline.start_new_block();
        newline.before_item(f);

        let mut newline = NewlineGenerator::new();

        match &child {
            IrModuleChild::CombinatorialProcess(process) => {
                let IrCombinatorialProcess { locals, block } = process;

                swriteln!(f, "{I}always @(*) begin");
                let variables = declare_locals(diags, module_name_scope, locals, f, &mut newline)?;

                let name_map = NameMap {
                    ports: port_name_map,
                    registers: reg_name_map,
                    wires: wire_name_map,
                    variables: &variables,
                };

                newline.start_new_block();
                lower_block(diags, name_map, block, f, Indent::new(2), &mut newline)?;
                swriteln!(f, "{I}end");
            }
            IrModuleChild::ClockedProcess(process) => {
                let IrClockedProcess {
                    domain,
                    locals,
                    on_clock,
                    on_reset,
                } = process;

                let outer_name_map = NameMap {
                    ports: port_name_map,
                    registers: reg_name_map,
                    wires: wire_name_map,
                    variables: &IndexMap::new(),
                };

                let SyncDomain { clock, reset } = &domain.inner;
                let clock_edge = lower_edge_to_str(diags, outer_name_map, domain.span, clock)?;
                let reset_edge = lower_edge_to_str(diags, outer_name_map, domain.span, reset)?;
                swriteln!(
                    f,
                    "{I}always @({} {}, {} {}) begin",
                    clock_edge.edge,
                    clock_edge.value,
                    reset_edge.edge,
                    reset_edge.value
                );

                let mut written_regs = HashSet::new();
                collect_written_registers(on_clock, &mut written_regs);
                collect_written_registers(on_reset, &mut written_regs);

                let shadowing_reg_name_map = lower_shadow_registers(
                    diags,
                    module_name_scope,
                    reg_name_map,
                    registers,
                    &written_regs,
                    f,
                    &mut newline,
                )?;

                let variables = declare_locals(diags, module_name_scope, locals, f, &mut newline)?;

                // populate shadow registers
                newline.start_new_block();
                for (&reg, shadow_name) in &shadowing_reg_name_map {
                    newline.before_item(f);
                    let orig_name = reg_name_map.get(&reg).unwrap();
                    swriteln!(f, "{I}{I}{shadow_name} = {orig_name};");
                }

                let inner_name_map = NameMap {
                    ports: port_name_map,
                    registers: &shadowing_reg_name_map,
                    wires: wire_name_map,
                    variables: &variables,
                };

                // main reset/clock structure
                newline.start_new_block();
                newline.before_item(f);
                swriteln!(f, "{I}{I}if ({}{}) begin", reset_edge.if_prefix, reset_edge.value);
                lower_block(diags, inner_name_map, on_reset, f, Indent::new(3), &mut newline)?;
                swriteln!(f, "{I}{I}end else begin");
                lower_block(diags, inner_name_map, on_clock, f, Indent::new(3), &mut newline)?;
                swriteln!(f, "{I}{I}end");

                // commit shadow writes
                newline.start_new_block();

                for (&reg, shadow_name) in &shadowing_reg_name_map {
                    newline.before_item(f);
                    let orig_name = reg_name_map.get(&reg).unwrap();
                    swriteln!(f, "{I}{I}{orig_name} <= {shadow_name};");
                }

                swriteln!(f, "{I}end");
            }
            IrModuleChild::ModuleInstance(instance) => {
                let IrModuleInstance {
                    name,
                    module,
                    port_connections,
                } = instance;

                let inner_module = ctx.module_map.get(module).unwrap();
                let inner_module_name = &inner_module.name;

                if let Some(name) = name {
                    let name_safe = LoweredName(name.clone());
                    swrite!(f, "{I}{inner_module_name} {name_safe}(");
                } else {
                    swrite!(f, "{I}{inner_module_name}(");
                }

                if port_connections.is_empty() {
                    swriteln!(f, ");");
                    return Ok(());
                }
                swriteln!(f);

                let name_map = NameMap {
                    ports: port_name_map,
                    registers: reg_name_map,
                    wires: wire_name_map,
                    variables: &IndexMap::new(),
                };

                for (port_index, (&port, connection)) in enumerate(port_connections) {
                    let port_name = inner_module.ports.get(&port).unwrap();
                    swrite!(f, "{I}{I}.{port_name}(");
                    match &connection.inner {
                        IrPortConnection::Input(expr) => {
                            lower_expression(diags, name_map, expr.span, &expr.inner, f)?;
                        }
                        &IrPortConnection::Output(signal) => {
                            match signal {
                                None => {
                                    // write nothing, causing empty `()`
                                }
                                Some(IrWireOrPort::Wire(signal_wire)) => {
                                    let wire_name = name_map.wires.get(&signal_wire).unwrap();
                                    swrite!(f, "{wire_name}");
                                }
                                Some(IrWireOrPort::Port(signal_port)) => {
                                    let port_name = name_map.ports.get(&signal_port).unwrap();
                                    swrite!(f, "{port_name}");
                                }
                            }
                        }
                    }

                    if port_index == port_connections.len() - 1 {
                        swriteln!(f, ")");
                    } else {
                        swriteln!(f, "),");
                    }
                }

                swriteln!(f, "{I});");
            }
        }
    }

    Ok(())
}

// TODO blocks with variables must be named
fn declare_locals(
    diags: &Diagnostics,
    module_name_scope: &mut LoweredNameScope,
    locals: &IrVariables,
    f: &mut String,
    newline: &mut NewlineGenerator,
) -> Result<IndexMap<IrVariable, LoweredName>, ErrorGuaranteed> {
    newline.start_new_block();

    let mut result = IndexMap::new();

    for (variable, variable_info) in locals {
        newline.before_item(f);

        let IrVariableInfo { ty, debug_info_id } = variable_info;
        let name = module_name_scope.make_unique_maybe_id(diags, debug_info_id)?;

        let ty_prefix_str = VerilogType::from_ir_ty(diags, debug_info_id.span(), ty)?.to_prefix_str();
        let (prefix_str, ty_prefix_str) = match ty_prefix_str.as_ref_ok() {
            Ok(ty_prefix_str) => ("", ty_prefix_str.as_str()),
            Err(ZeroWidth) => ("// ", "[empty]"),
        };

        swriteln!(f, "{I}{I}{prefix_str}reg {ty_prefix_str}{name};");
        result.insert_first(variable, name);
    }

    Ok(result)
}

fn lower_shadow_registers(
    diags: &Diagnostics,
    module_name_scope: &mut LoweredNameScope,
    reg_name_map: &IndexMap<IrRegister, LoweredName>,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    written_regs: &HashSet<IrRegister>,
    f: &mut String,
    newline: &mut NewlineGenerator,
) -> Result<IndexMap<IrRegister, LoweredName>, ErrorGuaranteed> {
    let mut shadowing_reg_name_map = IndexMap::new();

    newline.start_new_block();

    for reg in registers.keys() {
        let mapped = if written_regs.contains(&reg) {
            let register_info = &registers[reg];
            let ty = VerilogType::from_ir_ty(diags, register_info.debug_info_id.span(), &register_info.ty)?;

            let register_name = register_info.debug_info_id.string();
            let shadow_name = module_name_scope.make_unique_str(
                diags,
                register_info.debug_info_id.span(),
                &format!("shadow_{}", register_name),
            )?;

            match ty.to_prefix_str() {
                Ok(ty_prefix_str) => {
                    newline.before_item(f);
                    swriteln!(f, "{I}{I}reg {ty_prefix_str}{shadow_name};");
                }
                Err(ZeroWidth) => {
                    // don't declare shadows for zero-width registers
                }
            }

            shadow_name
        } else {
            reg_name_map.get(&reg).unwrap().clone()
        };

        shadowing_reg_name_map.insert_first(reg, mapped);
    }

    Ok(shadowing_reg_name_map)
}

fn collect_written_registers(block: &IrBlock, result: &mut HashSet<IrRegister>) {
    let IrBlock { statements } = block;

    for stmt in statements {
        match &stmt.inner {
            IrStatement::Assign(IrAssignmentTarget { base, steps: _ }, _) => match base {
                &IrAssignmentTargetBase::Register(reg) => {
                    result.insert(reg);
                }
                IrAssignmentTargetBase::Wire(_)
                | IrAssignmentTargetBase::Port(_)
                | IrAssignmentTargetBase::Variable(_) => {}
            },
            IrStatement::Block(inner) => {
                collect_written_registers(inner, result);
            }
            IrStatement::If(stmt) => {
                let IrIfStatement {
                    condition: _,
                    then_block,
                    else_block,
                } = stmt;

                collect_written_registers(then_block, result);
                if let Some(else_block) = else_block {
                    collect_written_registers(else_block, result);
                }
            }
            IrStatement::PrintLn(_) => {}
        }
    }
}

fn lower_block(
    diag: &Diagnostics,
    name_map: NameMap,
    block: &IrBlock,
    f: &mut String,
    indent: Indent,
    newline: &mut NewlineGenerator,
) -> Result<(), ErrorGuaranteed> {
    let IrBlock { statements } = block;

    for stmt in statements {
        newline.before_item(f);

        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                swrite!(f, "{indent}");
                lower_assign_target(diag, name_map, stmt.span, target, f)?;
                swrite!(f, " = ");
                lower_expression(diag, name_map, stmt.span, source, f)?;
                swriteln!(f, ";");
            }
            IrStatement::Block(inner) => {
                swriteln!(f, "{indent}begin");
                lower_block(diag, name_map, inner, f, indent.nest(), newline)?;
                swriteln!(f, "{indent}end");
            }
            IrStatement::If(IrIfStatement {
                condition,
                then_block,
                else_block,
            }) => {
                swrite!(f, "{indent}if (");
                lower_expression(diag, name_map, stmt.span, condition, f)?;
                swriteln!(f, ") begin");
                lower_block(diag, name_map, then_block, f, indent.nest(), newline)?;
                swrite!(f, "{indent}end");

                if let Some(else_block) = else_block {
                    swriteln!(f, " else begin");
                    lower_block(diag, name_map, else_block, f, indent.nest(), newline)?;
                    swrite!(f, "{indent}end");
                }

                swriteln!(f);
            }
            IrStatement::PrintLn(s) => {
                // TODO properly escape string
                swriteln!(f, "{indent}$display(\"{s}\");");
            }
        }
    }

    Ok(())
}

fn lower_assign_target(
    diag: &Diagnostics,
    name_map: NameMap,
    span: Span,
    target: &IrAssignmentTarget,
    f: &mut String,
) -> Result<(), ErrorGuaranteed> {
    // TODO this is probably wrong, we might need intermediate variables for the base and after each step
    let IrAssignmentTarget { base, steps } = target;

    match *base {
        IrAssignmentTargetBase::Port(port) => swrite!(f, "{}", name_map.ports.get(&port).unwrap()),
        IrAssignmentTargetBase::Wire(wire) => swrite!(f, "{}", name_map.wires.get(&wire).unwrap()),
        IrAssignmentTargetBase::Register(reg) => swrite!(f, "{}", name_map.registers.get(&reg).unwrap()),
        IrAssignmentTargetBase::Variable(var) => swrite!(f, "{}", name_map.variables.get(&var).unwrap()),
    }

    // TODO both of these are wrong, we're not taking element type sizes into account
    // TODO this entire thing should just be flattened to a single slice
    for step in steps {
        match step {
            IrTargetStep::ArrayIndex(start) => {
                swrite!(f, "[");
                lower_expression(diag, name_map, span, start, f)?;
                swrite!(f, "]");
            }

            IrTargetStep::ArraySlice(start, len) => {
                swrite!(f, "[");
                lower_expression(diag, name_map, span, start, f)?;
                swrite!(f, "+:{}", lower_uint_str(len));
                swrite!(f, "]");
            }
        }
    }

    Ok(())
}

// TODO allow this to use intermediate variables and to generate multi-line expressions
fn lower_expression(
    diags: &Diagnostics,
    name_map: NameMap,
    span: Span,
    expr: &IrExpression,
    f: &mut String,
) -> Result<(), ErrorGuaranteed> {
    fn try_get<'m, K: Eq + Hash, V>(
        diags: &Diagnostics,
        span: Span,
        map: &'m IndexMap<K, V>,
        key: K,
        kind: &str,
    ) -> Result<&'m V, ErrorGuaranteed> {
        map.get(&key)
            .ok_or_else(|| diags.report_internal_error(span, format!("failed to find {kind} in name map")))
    }

    match expr {
        &IrExpression::Bool(x) => swrite!(f, "1'b{}", x as u8),
        IrExpression::Int(x) => swrite!(f, "{}", lower_int_str(x)),

        &IrExpression::Port(port) => swrite!(f, "{}", try_get(diags, span, name_map.ports, port, "port")?),
        &IrExpression::Wire(wire) => swrite!(f, "{}", try_get(diags, span, name_map.wires, wire, "wire")?),
        &IrExpression::Register(reg) => swrite!(f, "{}", try_get(diags, span, name_map.registers, reg, "register")?),
        &IrExpression::Variable(var) => swrite!(f, "{}", try_get(diags, span, name_map.variables, var, "variable")?),

        IrExpression::BoolNot(inner) => {
            swrite!(f, "(!");
            lower_expression(diags, name_map, span, inner, f)?;
            swrite!(f, ")");
        }
        &IrExpression::BoolBinary(op, ref left, ref right) => {
            // logical and bitwise operators would both work,
            //   bitwise is more consistent since it also has an xor operator
            let op_str = match op {
                IrBoolBinaryOp::And => "&",
                IrBoolBinaryOp::Or => "|",
                IrBoolBinaryOp::Xor => "^",
            };

            swrite!(f, "(");
            lower_expression(diags, name_map, span, left, f)?;
            swrite!(f, " {} ", op_str);
            lower_expression(diags, name_map, span, right, f)?;
            swrite!(f, ")");
        }
        IrExpression::IntArithmetic(op, ty, left, right) => {
            // TODO bit-widths are not correct
            let op_str = match op {
                IrIntArithmeticOp::Add => "+",
                IrIntArithmeticOp::Sub => "-",
                IrIntArithmeticOp::Mul => "*",
                IrIntArithmeticOp::Div => "/",
                IrIntArithmeticOp::Mod => "%",
                IrIntArithmeticOp::Pow => "**",
            };

            let _ = ty;
            swrite!(f, "(");
            lower_expression(diags, name_map, span, left, f)?;
            swrite!(f, " {} ", op_str);
            lower_expression(diags, name_map, span, right, f)?;
            swrite!(f, ")");
        }
        IrExpression::IntCompare(op, left, right) => {
            // TODO bit-widths are not correct
            let op_str = match op {
                IrIntCompareOp::Eq => "==",
                IrIntCompareOp::Neq => "!=",
                IrIntCompareOp::Lt => "<",
                IrIntCompareOp::Lte => "<=",
                IrIntCompareOp::Gt => ">",
                IrIntCompareOp::Gte => ">=",
            };

            swrite!(f, "(");
            lower_expression(diags, name_map, span, left, f)?;
            swrite!(f, " {} ", op_str);
            lower_expression(diags, name_map, span, right, f)?;
            swrite!(f, ")");
        }

        IrExpression::TupleLiteral(elements) => {
            // verilog does not care much about types, this is just a concatenation
            //  (assuming all sub-expression have the right width, which they should)
            // TODO this is probably incorrect in general, we need to store the tuple in a variable first
            swrite!(f, "{{");
            for (i, elem) in enumerate(elements) {
                if i != 0 {
                    swrite!(f, ", ");
                }

                lower_expression(diags, name_map, span, elem, f)?;
            }
            swrite!(f, "}}");
        }
        IrExpression::ArrayLiteral(_inner_ty, _len, elements) => {
            // verilog does not care much about types, this is just a concatenation
            //  (assuming all sub-expression have the right width, which they should)
            // TODO skip for zero-sized array? we probably need a more general way to skip zero-sized expressions
            // TODO use repeat operator if array elements are repeated
            swrite!(f, "{{");
            for (i, elem) in enumerate(elements) {
                if i != 0 {
                    swrite!(f, ", ");
                }

                let inner = match elem {
                    IrArrayLiteralElement::Spread(inner) => inner,
                    IrArrayLiteralElement::Single(inner) => inner,
                };
                lower_expression(diags, name_map, span, inner, f)?;
            }
            swrite!(f, "}}");
        }

        IrExpression::ArrayIndex { base, index } => {
            // TODO this is probably incorrect in general, we need to store the array in a variable first
            swrite!(f, "(");
            lower_expression(diags, name_map, span, base, f)?;
            swrite!(f, "[");
            lower_expression(diags, name_map, span, index, f)?;
            swrite!(f, "])");
        }
        IrExpression::ArraySlice { base, start, len } => {
            // TODO this is probably incorrect in general, we need to store the array in a variable first
            swrite!(f, "(");
            lower_expression(diags, name_map, span, base, f)?;
            swrite!(f, "[");
            lower_expression(diags, name_map, span, start, f)?;
            swrite!(f, "+:{}])", lower_uint_str(len));
        }

        IrExpression::IntToBits(_, _) => throw!(diags.report_todo(span, "lower int to bits")),
        IrExpression::IntFromBits(_, _) => throw!(diags.report_todo(span, "lower bits to int")),
        IrExpression::ExpandIntRange(target, value) => {
            // just add zero of the right width to expand the range
            // TODO skip if unnecessary?
            let target_repr = IntRepresentation::for_range(target);
            swrite!(f, "({}'d0 + ", target_repr.width);
            lower_expression(diags, name_map, span, value, f)?;
            swrite!(f, ")");
        }
        IrExpression::ConstrainIntRange(target, value) => {
            // TODO this not correct, we're not actually lowering the bit width
            let target_repr = IntRepresentation::for_range(target);
            let _ = target_repr;
            lower_expression(diags, name_map, span, value, f)?;
        }
    }

    Ok(())
}

fn lower_int_str(x: &BigInt) -> String {
    // TODO zero-width literals are probably not allowed in verilog
    // TODO double-check integer bit-width promotion rules
    let sign = match x.sign() {
        num_bigint::Sign::Plus | num_bigint::Sign::NoSign => "",
        num_bigint::Sign::Minus => "-",
    };
    let repr = IntRepresentation::for_single(x);
    format!("{}{}'d{}", sign, repr.width, x.magnitude())
}

fn lower_uint_str(x: &BigUint) -> String {
    // TODO zero-width literals are probably not allowed in verilog
    // TODO double-check integer bit-width promotion rules
    // TODO avoid clone
    let repr = IntRepresentation::for_single(&BigInt::from(x.clone()));
    format!("{}'d{}", repr.width, x)
}

#[derive(Debug)]
struct EdgeString {
    edge: &'static str,
    if_prefix: &'static str,
    value: String,
}

fn lower_edge_to_str(
    diags: &Diagnostics,
    name_map: NameMap,
    span: Span,
    expr: &IrExpression,
) -> Result<EdgeString, ErrorGuaranteed> {
    // unwrap not layers, their parity will determine the edge
    let mut not_count: u32 = 0;
    let mut curr = expr;
    while let IrExpression::BoolNot(inner) = curr {
        not_count += 1;
        curr = inner;
    }

    let (edge, if_prefix) = if not_count % 2 == 0 {
        ("posedge", "")
    } else {
        ("negedge", "!")
    };

    // lower expression
    let mut s = String::new();
    let f = &mut s;
    lower_expression(diags, name_map, span, curr, f)?;

    Ok(EdgeString {
        edge,
        if_prefix,
        value: s,
    })
}

const I: &str = Indent::I;

#[derive(Debug, Copy, Clone)]
struct ZeroWidth;

// TODO maybe simplify this to just a single "width" value?
// TODO maybe expand this to represent multi-dim arrays?
// TODO should empty types be represented as single bits or should all interacting code be skipped?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum VerilogType {
    /// Width-0 type, should be skipped in generated code since Verilog does not support it.
    ZeroWidth,
    /// Single bit values.
    Bit,
    /// Potentially multi-bit values. This can still happen to have width 1,
    /// The difference with [SingleBit] in that case is that this should still be represented as an array.
    Array(NonZeroU32),
}

impl VerilogType {
    pub fn array(diags: &Diagnostics, span: Span, w: BigUint) -> Result<VerilogType, ErrorGuaranteed> {
        let w = diag_big_int_to_u32(diags, span, &w.into(), "array width too large")?;
        match NonZeroU32::new(w) {
            None => Ok(VerilogType::ZeroWidth),
            Some(w) => Ok(VerilogType::Array(w)),
        }
    }

    // TODO split tuples and short arrays into multiple ports instead?
    pub fn from_ir_ty(diags: &Diagnostics, span: Span, ty: &IrType) -> Result<VerilogType, ErrorGuaranteed> {
        match ty {
            IrType::Bool => Ok(VerilogType::Bit),
            IrType::Int(_) | IrType::Tuple(_) | IrType::Array(_, _) => Self::array(diags, span, ty.bit_width()),
        }
    }

    pub fn to_prefix_str(self) -> Result<String, ZeroWidth> {
        match self {
            VerilogType::ZeroWidth => Err(ZeroWidth),
            VerilogType::Bit => Ok("".to_string()),
            VerilogType::Array(width) => Ok(format!("[{}:0] ", width.get() - 1)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NewlineGenerator {
    any_prev: bool,
    any_curr: bool,
}

// TODO rename these functions to make more sense, and create a function for the combination of both
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

fn diag_big_int_to_u32(diags: &Diagnostics, span: Span, value: &BigInt, message: &str) -> Result<u32, ErrorGuaranteed> {
    value.try_into().map_err(|_| {
        diags.report_simple(
            format!("{message}: overflow when converting {value} to u32"),
            span,
            "used here",
        )
    })
}

impl Display for LoweredName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO use hashset or at least binary search for this
        if VERILOG_KEYWORDS.contains(&self.0.as_str()) {
            // emit escaped identifier,
            //   including extra trailing space just to be sure
            f.write_str(&format!("\\{} ", self.0))
        } else {
            f.write_str(&self.0)
        }
    }
}

lazy_static! {
    /// Updated to "IEEE Standard for SystemVerilog", IEEE 1800-2023
    static ref VERILOG_KEYWORDS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        for line in include_str!("verilog_keywords.txt").lines() {
            let line = line.trim();
            if !line.is_empty() {
                set.insert(line);
            }
        }
        set
    };
}
