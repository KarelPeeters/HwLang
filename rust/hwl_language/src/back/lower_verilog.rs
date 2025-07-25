use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::types::HardwareType;
use crate::mid::ir::{
    ir_modules_topological_sort, IrArrayLiteralElement, IrAssignmentTarget, IrAssignmentTargetBase, IrAsyncResetInfo,
    IrBlock, IrBoolBinaryOp, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrIfStatement,
    IrIntArithmeticOp, IrIntCompareOp, IrLargeArena, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo,
    IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrStatement,
    IrTargetStep, IrType, IrVariable, IrVariableInfo, IrVariables, IrWire, IrWireInfo, IrWireOrPort,
};
use crate::syntax::ast::{PortDirection, Spanned};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint, Sign};
use crate::util::data::IndexMapExt;
use crate::util::int::IntRepresentation;
use crate::util::{separator_non_trailing, Indent, ResultExt};
use hwl_util::{swrite, swriteln};
use indexmap::{IndexMap, IndexSet};
use itertools::enumerate;
use lazy_static::lazy_static;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;

#[derive(Debug, Clone)]
pub struct LoweredVerilog {
    pub source: String,

    // TODO should this be a string or a lowered name? we don't want to expose too many implementation details
    pub top_module_name: String,
    pub debug_info_module_map: IndexMap<IrModule, String>,
}

// TODO make backend configurable between verilog and VHDL?
// TODO ban keywords
// TODO should we still be doing diagnostics here, or should lowering just never start?
// TODO identifier ID: prefix _all_ signals with something: wire, reg, local, ...,
//   so nothing can conflict with ports/module names. Not fully right yet, but maybe a good idea.
pub fn lower_to_verilog(
    diags: &Diagnostics,
    modules: &IrModules,
    external_modules: &IndexSet<String>,
    top_module: IrModule,
) -> DiagResult<LoweredVerilog> {
    let mut ctx = LowerContext {
        diags,
        modules,
        module_map: IndexMap::new(),
        top_name_scope: LoweredNameScope {
            used: external_modules.clone(),
        },
        lowered_modules: vec![],
    };

    let modules = ir_modules_topological_sort(modules, top_module);
    for module in modules {
        let result = lower_module(&mut ctx, module)?;
        ctx.module_map.insert_first(module, result);
    }

    let top_name = ctx.module_map.get(&top_module).unwrap().name.clone();
    Ok(LoweredVerilog {
        source: ctx.lowered_modules.join("\n\n"),
        top_module_name: top_name.0.clone(),
        debug_info_module_map: ctx.module_map.into_iter().map(|(k, v)| (k, v.name.0)).collect(),
    })
}

struct LowerContext<'a> {
    diags: &'a Diagnostics,
    modules: &'a IrModules,
    module_map: IndexMap<IrModule, LoweredModule>,
    top_name_scope: LoweredNameScope,
    lowered_modules: Vec<String>,
}

#[derive(Debug, Clone)]
struct LoweredName<S: AsRef<str> = String>(S);

#[derive(Debug, Clone)]
struct LoweredModule {
    name: LoweredName,
    ports: IndexMap<IrPort, LoweredName>,
}

#[derive(Default)]
struct LoweredNameScope {
    used: IndexSet<String>,
}

impl LoweredNameScope {
    pub fn exact_for_new_id(&mut self, diags: &Diagnostics, span: Span, id: &str) -> DiagResult<LoweredName> {
        check_identifier_valid(diags, Spanned { span, inner: id })?;
        if !self.used.insert(id.to_owned()) {
            throw!(diags.report_internal_error(span, format!("lowered identifier `{id}` already used its scope")))
        }
        Ok(LoweredName(id.to_owned()))
    }

    pub fn make_unique_maybe_id(&mut self, diags: &Diagnostics, id: Spanned<Option<&str>>) -> DiagResult<LoweredName> {
        self.make_unique_str(diags, id.span, id.inner.unwrap_or("_"))
    }

    #[allow(dead_code)]
    pub fn make_unique_id(&mut self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<LoweredName> {
        self.make_unique_str(diags, id.span, id.inner)
    }

    pub fn make_unique_str(&mut self, diags: &Diagnostics, span: Span, string: &str) -> DiagResult<LoweredName> {
        check_identifier_valid(diags, Spanned { span, inner: string })?;

        if self.used.insert(string.to_owned()) {
            return Ok(LoweredName(string.to_owned()));
        }

        // TODO speed this up?
        for i in 0u32.. {
            let suffixed = format!("{string}_{i}");
            if self.used.insert(suffixed.clone()) {
                return Ok(LoweredName(suffixed));
            }
        }

        throw!(diags.report_internal_error(
            span,
            format!("failed to generate unique lowered identifier for `{string}`")
        ))
    }
}

// TODO replace with name mangling that forces everything to be valid
fn check_identifier_valid(diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<()> {
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
    registers_inner: &'a IndexMap<IrRegister, LoweredName>,
    registers_outer: &'a IndexMap<IrRegister, LoweredName>,
    wires: &'a IndexMap<IrWire, LoweredName>,
    variables: &'a IndexMap<IrVariable, LoweredName>,
}

impl NameMap<'_> {
    pub fn map_reg(&self, reg: IrRegister) -> &LoweredName {
        self.registers_inner
            .get(&reg)
            .unwrap_or_else(|| self.registers_outer.get(&reg).unwrap())
    }
}

fn lower_module(ctx: &mut LowerContext, module: IrModule) -> DiagResult<LoweredModule> {
    let diags = ctx.diags;
    assert!(!ctx.module_map.contains_key(&module));

    // TODO careful with name scoping: we don't want eg. ports to accidentally shadow other modules
    //   or maybe verilog has separate namespaces, then it's fine

    let module_info = &ctx.modules[module];
    let IrModuleInfo {
        ports,
        large,
        registers,
        wires,
        children: processes,
        debug_info_file,
        debug_info_id,
        debug_info_generic_args,
    } = module_info;
    let debug_info_id = maybe_id_as_ref(debug_info_id);
    let module_name = ctx.top_name_scope.make_unique_maybe_id(diags, debug_info_id)?;

    let mut f = String::new();

    swriteln!(f, "// module {}", debug_info_id.inner.unwrap_or("_"));
    swriteln!(f, "//   defined in \"{debug_info_file}\"",);

    if let Some(generic_args) = debug_info_generic_args {
        swriteln!(f, "//   instantiated with generic arguments:");
        for (arg_name, arg_value) in generic_args {
            swriteln!(f, "//     {}={}", arg_name, arg_value.diagnostic_string());
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
        large,
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
) -> DiagResult<IndexMap<IrPort, LoweredName>> {
    let mut port_lines = vec![];
    let mut port_name_map = IndexMap::new();
    let mut last_actual_port_index = None;

    for (port_index, (port, port_info)) in enumerate(ports) {
        let IrPortInfo {
            name,
            direction,
            ty,
            debug_span,
            debug_info_ty,
            debug_info_domain,
        } = port_info;

        // TODO check that port names are valid and unique
        let lower_name = module_name_scope.exact_for_new_id(diags, *debug_span, name)?;
        let port_ty = VerilogType::from_ir_ty(diags, *debug_span, ty)?;

        let ty_prefix_str = port_ty.to_prefix_str();
        let (is_actual_port, ty_str) = match ty_prefix_str.as_ref().map(|s| s.as_str()) {
            Ok(ty_str) => (true, ty_str),
            Err(ZeroWidth) => (false, "[empty]"),
        };
        let dir_str = match direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output reg",
        };

        if is_actual_port {
            last_actual_port_index = Some(port_index);
        }
        port_lines.push((
            is_actual_port,
            format!("{dir_str} {ty_str}{lower_name}"),
            format!("{debug_info_domain} {}", debug_info_ty.inner),
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

    if !ports.is_empty() {
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
) -> DiagResult<(IndexMap<IrRegister, LoweredName>, IndexMap<IrWire, LoweredName>)> {
    let mut lower_signal = |signal_type,
                            ty,
                            debug_info_id: &Spanned<Option<String>>,
                            debug_info_ty: &HardwareType,
                            debug_info_domain,
                            f: &mut String| {
        let debug_info_id = maybe_id_as_ref(debug_info_id);
        let name = module_name_scope.make_unique_maybe_id(diags, debug_info_id)?;
        let ty_prefix_str = VerilogType::from_ir_ty(diags, debug_info_id.span, ty)?.to_prefix_str();
        let (prefix_str, ty_prefix_str) = match ty_prefix_str.as_ref_ok() {
            Ok(ty_prefix_str) => ("", ty_prefix_str.as_str()),
            Err(ZeroWidth) => ("// ", "[empty]"),
        };

        let name_debug_str = debug_info_id.inner.unwrap_or("_");
        let ty_debug_str = debug_info_ty.diagnostic_string();

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
    large: &IrLargeArena,
    module_name_scope: &mut LoweredNameScope,
    port_name_map: &IndexMap<IrPort, LoweredName>,
    reg_name_map: &IndexMap<IrRegister, LoweredName>,
    wire_name_map: &IndexMap<IrWire, LoweredName>,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    children: &[Spanned<IrModuleChild>],
    newline: &mut NewlineGenerator,
    f: &mut String,
) -> DiagResult<()> {
    let diags = ctx.diags;

    for (child_index, child) in enumerate(children) {
        newline.start_new_block();
        newline.before_item(f);

        let mut newline = NewlineGenerator::new();

        match &child.inner {
            IrModuleChild::CombinatorialProcess(process) => {
                let IrCombinatorialProcess { locals, block } = process;

                swriteln!(f, "{I}always @(*) begin");
                let variables = declare_locals(diags, module_name_scope, locals, f, &mut newline)?;

                let name_map = NameMap {
                    ports: port_name_map,
                    registers_outer: reg_name_map,
                    registers_inner: &IndexMap::default(),
                    wires: wire_name_map,
                    variables: &variables,
                };

                newline.start_new_block();
                lower_block(large, name_map, block, f, Indent::new(2), &mut newline)?;
                swriteln!(f, "{I}end");
            }
            IrModuleChild::ClockedProcess(process) => {
                let IrClockedProcess {
                    locals,
                    clock_signal,
                    clock_block,
                    async_reset,
                } = process;

                let outer_name_map = NameMap {
                    ports: port_name_map,
                    registers_outer: reg_name_map,
                    registers_inner: &IndexMap::default(),
                    wires: wire_name_map,
                    variables: &IndexMap::new(),
                };

                let clock_edge = lower_edge_to_str(large, outer_name_map, clock_signal.as_ref())?;
                let async_reset = async_reset
                    .as_ref()
                    .map(|info| {
                        let reset_edge = lower_edge_to_str(large, outer_name_map, info.signal.as_ref())?;
                        Ok((reset_edge, info))
                    })
                    .transpose()?;

                match &async_reset {
                    Some((reset_edge, _)) => swriteln!(
                        f,
                        "{I}always @({} {}, {} {}) begin",
                        clock_edge.edge,
                        clock_edge.value,
                        reset_edge.edge,
                        reset_edge.value,
                    ),
                    None => swriteln!(f, "{I}always @({} {}) begin", clock_edge.edge, clock_edge.value,),
                }

                // shadowing is only for writes in the clocked body, we don't care about the resets
                //   (although typically those should be a subset)
                let mut written_regs = IndexSet::new();
                collect_written_registers(clock_block, &mut written_regs);
                let shadowing_reg_name_map =
                    lower_shadow_registers(diags, module_name_scope, registers, &written_regs, f, &mut newline)?;
                let variables = declare_locals(diags, module_name_scope, locals, f, &mut newline)?;

                let inner_name_map = NameMap {
                    ports: port_name_map,
                    registers_outer: reg_name_map,
                    registers_inner: &shadowing_reg_name_map,
                    wires: wire_name_map,
                    variables: &variables,
                };

                // main reset/clock structure
                newline.start_new_block();
                newline.before_item(f);

                // reset header
                let indent_clocked = match &async_reset {
                    None => Indent::new(2),
                    Some((reset_edge, reset_info)) => {
                        // reset, using outer name map (no shadowing)
                        swriteln!(f, "{I}{I}if ({}{}) begin", reset_edge.if_prefix, reset_edge.value);

                        let IrAsyncResetInfo { signal: _, resets } = reset_info;
                        for reset in resets {
                            let (reg, value) = &reset.inner;
                            let reg_name = reg_name_map.get(reg).unwrap();
                            swrite!(f, "{I}{I}{I}{reg_name} <= ");
                            lower_expression(large, outer_name_map, value, f)?;
                            swriteln!(f, ";");
                        }

                        swriteln!(f, "{I}{I}end else begin");
                        Indent::new(3)
                    }
                };

                // populate shadow registers
                for (&reg, shadow_name) in &shadowing_reg_name_map {
                    newline.before_item(f);
                    let orig_name = reg_name_map.get(&reg).unwrap();
                    swriteln!(f, "{indent_clocked}{shadow_name} = {orig_name};");
                }

                // block itself, using inner name map (with shadowing)
                newline.start_new_block();
                lower_block(large, inner_name_map, clock_block, f, indent_clocked, &mut newline)?;

                // write-back shadow registers
                newline.start_new_block();
                for (&reg, shadow_name) in &shadowing_reg_name_map {
                    newline.before_item(f);
                    let orig_name = reg_name_map.get(&reg).unwrap();
                    swriteln!(f, "{indent_clocked}{orig_name} <= {shadow_name};");
                }

                // reset tail
                if async_reset.is_some() {
                    swriteln!(f, "{I}{I}end");
                }

                swriteln!(f, "{I}end");
            }
            IrModuleChild::ModuleInternalInstance(instance) => {
                let IrModuleInternalInstance {
                    name,
                    module,
                    port_connections,
                } = instance;

                let inner_module = ctx.module_map.get(module).unwrap();
                let inner_module_name = &inner_module.name;

                if let Some(name) = name {
                    let name_safe = LoweredName(name.clone());
                    swrite!(f, "{I}{inner_module_name} {name_safe}");
                } else {
                    swrite!(f, "{I}{inner_module_name} instance_{child_index}");
                }

                let name_map = NameMap {
                    ports: port_name_map,
                    registers_outer: reg_name_map,
                    registers_inner: &IndexMap::default(),
                    wires: wire_name_map,
                    variables: &IndexMap::new(),
                };
                let port_name = |port_index| {
                    // TODO avoid clone here
                    let (_port, name) = inner_module.ports.get_index(port_index).unwrap();
                    name.clone()
                };
                lower_port_connections(f, large, name_map, port_name, port_connections)?;
            }
            IrModuleChild::ModuleExternalInstance(instance) => {
                let IrModuleExternalInstance {
                    name,
                    module_name,
                    generic_args,
                    port_names,
                    port_connections,
                } = instance;

                swrite!(f, "{I}{module_name}");

                if let Some(generic_args) = generic_args {
                    if generic_args.is_empty() {
                        swrite!(f, " #()");
                    } else {
                        swriteln!(f, " #(");
                        for (arg_index, (arg_name, arg_value)) in enumerate(generic_args) {
                            let arg_name = LoweredName(arg_name);
                            let sep = separator_non_trailing(",", arg_index, generic_args.len());
                            swriteln!(f, "{I}{I}.{arg_name}({arg_value}){sep}");
                        }
                        swrite!(f, "{I})");
                    }
                }

                if let Some(name) = name {
                    swrite!(f, " {}", LoweredName(name));
                } else {
                    swrite!(f, " instance_{child_index}");
                }

                let name_map = NameMap {
                    ports: port_name_map,
                    registers_outer: reg_name_map,
                    registers_inner: &IndexMap::default(),
                    wires: wire_name_map,
                    variables: &IndexMap::new(),
                };
                let port_name = |port_index: usize| LoweredName(&port_names[port_index]);
                lower_port_connections(f, large, name_map, port_name, port_connections)?;
            }
        }
    }

    Ok(())
}

fn lower_port_connections<S: AsRef<str>>(
    f: &mut String,
    large: &IrLargeArena,
    name_map: NameMap,
    port_name: impl Fn(usize) -> LoweredName<S>,
    port_connections: &Vec<Spanned<IrPortConnection>>,
) -> DiagResult<()> {
    swrite!(f, "(");

    if port_connections.is_empty() {
        swriteln!(f, ");");
        return Ok(());
    }
    swriteln!(f);

    for (port_index, connection) in enumerate(port_connections) {
        let port_name = port_name(port_index);
        swrite!(f, "{I}{I}.{port_name}(");

        match &connection.inner {
            IrPortConnection::Input(expr) => {
                lower_expression(large, name_map, &expr.inner, f)?;
            }
            &IrPortConnection::Output(signal) => {
                match signal {
                    None => {
                        // write nothing, resulting in an empty `()`
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

        swriteln!(
            f,
            "){}",
            separator_non_trailing(",", port_index, port_connections.len())
        );
    }

    swriteln!(f, "{I});");
    Ok(())
}

// TODO blocks with variables must be named
fn declare_locals(
    diags: &Diagnostics,
    module_name_scope: &mut LoweredNameScope,
    locals: &IrVariables,
    f: &mut String,
    newline: &mut NewlineGenerator,
) -> DiagResult<IndexMap<IrVariable, LoweredName>> {
    newline.start_new_block();

    let mut result = IndexMap::new();

    for (variable, variable_info) in locals {
        newline.before_item(f);

        let IrVariableInfo { ty, debug_info_id } = variable_info;
        let name = module_name_scope.make_unique_maybe_id(diags, maybe_id_as_ref(debug_info_id))?;

        let ty_prefix_str = VerilogType::from_ir_ty(diags, debug_info_id.span, ty)?.to_prefix_str();
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
    registers: &Arena<IrRegister, IrRegisterInfo>,
    written_regs: &IndexSet<IrRegister>,
    f: &mut String,
    newline: &mut NewlineGenerator,
) -> DiagResult<IndexMap<IrRegister, LoweredName>> {
    let mut shadowing_reg_name_map = IndexMap::new();

    newline.start_new_block();

    for &reg in written_regs {
        let register_info = &registers[reg];
        let debug_info_id = maybe_id_as_ref(&register_info.debug_info_id);

        let ty = VerilogType::from_ir_ty(diags, debug_info_id.span, &register_info.ty)?;

        let register_name = debug_info_id.inner.unwrap_or("_");
        let shadow_name =
            module_name_scope.make_unique_str(diags, debug_info_id.span, &format!("shadow_{register_name}"))?;

        match ty.to_prefix_str() {
            Ok(ty_prefix_str) => {
                newline.before_item(f);
                swriteln!(f, "{I}{I}reg {ty_prefix_str}{shadow_name};");
            }
            Err(ZeroWidth) => {
                // don't declare shadows for zero-width registers
            }
        }

        shadowing_reg_name_map.insert_first(reg, shadow_name);
    }

    Ok(shadowing_reg_name_map)
}

fn collect_written_registers(block: &IrBlock, result: &mut IndexSet<IrRegister>) {
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
    large: &IrLargeArena,
    name_map: NameMap,
    block: &IrBlock,
    f: &mut String,
    indent: Indent,
    newline: &mut NewlineGenerator,
) -> DiagResult<()> {
    let IrBlock { statements } = block;

    for stmt in statements {
        newline.before_item(f);

        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                swrite!(f, "{indent}");
                lower_assign_target(large, name_map, target, f)?;
                swrite!(f, " = ");
                lower_expression(large, name_map, source, f)?;
                swriteln!(f, ";");
            }
            IrStatement::Block(inner) => {
                swriteln!(f, "{indent}begin");
                lower_block(large, name_map, inner, f, indent.nest(), newline)?;
                swriteln!(f, "{indent}end");
            }
            IrStatement::If(IrIfStatement {
                condition,
                then_block,
                else_block,
            }) => {
                swrite!(f, "{indent}if (");
                lower_expression(large, name_map, condition, f)?;
                swriteln!(f, ") begin");
                lower_block(large, name_map, then_block, f, indent.nest(), newline)?;
                swrite!(f, "{indent}end");

                if let Some(else_block) = else_block {
                    swriteln!(f, " else begin");
                    lower_block(large, name_map, else_block, f, indent.nest(), newline)?;
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
    large: &IrLargeArena,
    name_map: NameMap,
    target: &IrAssignmentTarget,
    f: &mut String,
) -> DiagResult<()> {
    // TODO this is probably wrong, we might need intermediate variables for the base and after each step
    let IrAssignmentTarget { base, steps } = target;

    match *base {
        IrAssignmentTargetBase::Port(port) => swrite!(f, "{}", name_map.ports.get(&port).unwrap()),
        IrAssignmentTargetBase::Wire(wire) => swrite!(f, "{}", name_map.wires.get(&wire).unwrap()),
        IrAssignmentTargetBase::Register(reg) => swrite!(f, "{}", name_map.map_reg(reg)),
        IrAssignmentTargetBase::Variable(var) => swrite!(f, "{}", name_map.variables.get(&var).unwrap()),
    }

    // TODO both of these are wrong, we're not taking element type sizes into account
    // TODO this entire thing should just be flattened to a single slice
    for step in steps {
        match step {
            IrTargetStep::ArrayIndex(start) => {
                swrite!(f, "[");
                lower_expression(large, name_map, start, f)?;
                swrite!(f, "]");
            }

            IrTargetStep::ArraySlice(start, len) => {
                swrite!(f, "[");
                lower_expression(large, name_map, start, f)?;
                swrite!(f, "+:{}", lower_uint_str(len));
                swrite!(f, "]");
            }
        }
    }

    Ok(())
}

// TODO allow this to use intermediate variables and to generate multi-line expressions
fn lower_expression(large: &IrLargeArena, name_map: NameMap, expr: &IrExpression, f: &mut String) -> DiagResult<()> {
    match expr {
        &IrExpression::Bool(x) => swrite!(f, "1'b{}", x as u8),
        IrExpression::Int(x) => swrite!(f, "{}", lower_int_str(x)),

        &IrExpression::Port(port) => swrite!(f, "{}", name_map.ports.get(&port).unwrap()),
        &IrExpression::Wire(wire) => swrite!(f, "{}", name_map.wires.get(&wire).unwrap()),
        &IrExpression::Register(reg) => swrite!(f, "{}", name_map.map_reg(reg)),
        &IrExpression::Variable(var) => swrite!(f, "{}", name_map.variables.get(&var).unwrap()),

        &IrExpression::Large(expr) => {
            match &large[expr] {
                IrExpressionLarge::BoolNot(inner) => {
                    swrite!(f, "(!");
                    lower_expression(large, name_map, inner, f)?;
                    swrite!(f, ")");
                }
                IrExpressionLarge::BoolBinary(op, ref left, ref right) => {
                    // logical and bitwise operators would both work,
                    //   bitwise is more consistent since it also has an xor operator
                    let op_str = match op {
                        IrBoolBinaryOp::And => "&",
                        IrBoolBinaryOp::Or => "|",
                        IrBoolBinaryOp::Xor => "^",
                    };

                    swrite!(f, "(");
                    lower_expression(large, name_map, left, f)?;
                    swrite!(f, " {} ", op_str);
                    lower_expression(large, name_map, right, f)?;
                    swrite!(f, ")");
                }
                IrExpressionLarge::IntArithmetic(op, ty, left, right) => {
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
                    lower_expression(large, name_map, left, f)?;
                    swrite!(f, " {} ", op_str);
                    lower_expression(large, name_map, right, f)?;
                    swrite!(f, ")");
                }
                IrExpressionLarge::IntCompare(op, left, right) => {
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
                    lower_expression(large, name_map, left, f)?;
                    swrite!(f, " {} ", op_str);
                    lower_expression(large, name_map, right, f)?;
                    swrite!(f, ")");
                }

                IrExpressionLarge::TupleLiteral(elements) => {
                    // verilog does not care much about types, this is just a concatenation
                    //  (assuming all sub-expression have the right width, which they should)
                    // TODO this is probably incorrect in general, we need to store the tuple in a variable first
                    swrite!(f, "{{");
                    for (i, elem) in enumerate(elements) {
                        if i != 0 {
                            swrite!(f, ", ");
                        }

                        lower_expression(large, name_map, elem, f)?;
                    }
                    swrite!(f, "}}");
                }
                IrExpressionLarge::ArrayLiteral(_inner_ty, _len, elements) => {
                    // verilog does not care much about types, this is just a concatenation
                    //  (assuming all sub-expression have the right width, which they should)
                    // TODO skip for zero-sized array? we probably need a more general way to skip zero-sized expressions
                    // TODO use repeat operator if array elements are repeated
                    // TODO the order is wrong, the verilog array operator is the wrong way around
                    swrite!(f, "{{");
                    for (i, elem) in enumerate(elements) {
                        if i != 0 {
                            swrite!(f, ", ");
                        }

                        let inner = match elem {
                            IrArrayLiteralElement::Spread(inner) => inner,
                            IrArrayLiteralElement::Single(inner) => inner,
                        };
                        lower_expression(large, name_map, inner, f)?;
                    }
                    swrite!(f, "}}");
                }

                IrExpressionLarge::TupleIndex { base, index } => {
                    // TODO this is completely wrong
                    swrite!(f, "(");
                    lower_expression(large, name_map, base, f)?;
                    swrite!(f, "[{}])", index);
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    // TODO this is probably incorrect in general, we need to store the array in a variable first
                    // TODO we're incorrectly using array indices as bit indices here
                    swrite!(f, "(");
                    lower_expression(large, name_map, base, f)?;
                    swrite!(f, "[");
                    lower_expression(large, name_map, index, f)?;
                    swrite!(f, "])");
                }
                IrExpressionLarge::ArraySlice { base, start, len } => {
                    // TODO this is probably incorrect in general, we need to store the array in a variable first
                    // TODO we're incorrectly using array indices as bit indices here
                    swrite!(f, "(");
                    lower_expression(large, name_map, base, f)?;
                    swrite!(f, "[");
                    lower_expression(large, name_map, start, f)?;
                    swrite!(f, "+:{}])", lower_uint_str(len));
                }

                IrExpressionLarge::ToBits(_ty, value) => {
                    // in verilog everything is just a bit vector, so we don't need to do anything
                    lower_expression(large, name_map, value, f)?;
                }
                IrExpressionLarge::FromBits(_ty, value) => {
                    // in verilog everything is just a bit vector, so we don't need to do anything
                    lower_expression(large, name_map, value, f)?;
                }
                IrExpressionLarge::ExpandIntRange(target, value) => {
                    // just add zero of the right width to expand the range
                    // TODO skip if unnecessary?
                    // TODO this is probably wrong for signed values, and definitely for zero-width values
                    let target_repr = IntRepresentation::for_range(target);
                    swrite!(f, "({}'d0 + ", target_repr.size_bits());
                    lower_expression(large, name_map, value, f)?;
                    swrite!(f, ")");
                }
                IrExpressionLarge::ConstrainIntRange(target, value) => {
                    // TODO this not correct, we're not actually lowering the bit width
                    let target_repr = IntRepresentation::for_range(target);
                    let _ = target_repr;
                    lower_expression(large, name_map, value, f)?;
                }
            }
        }
    }

    Ok(())
}

fn lower_int_str(x: &BigInt) -> String {
    // TODO zero-width literals are probably not allowed in verilog
    // TODO double-check integer bit-width promotion rules
    let sign = match x.sign() {
        Sign::Positive | Sign::Zero => "",
        Sign::Negative => "-",
    };
    let repr = IntRepresentation::for_single(x);
    format!("{}{}'d{}", sign, repr.size_bits(), x.abs())
}

fn lower_uint_str(x: &BigUint) -> String {
    // TODO zero-width literals are probably not allowed in verilog
    // TODO double-check integer bit-width promotion rules
    // TODO avoid clone
    let repr = IntRepresentation::for_single(&x.into());
    format!("{}'d{}", repr.size_bits(), x)
}

#[derive(Debug)]
struct EdgeString {
    edge: &'static str,
    if_prefix: &'static str,
    value: String,
}

fn lower_edge_to_str(large: &IrLargeArena, name_map: NameMap, expr: Spanned<&IrExpression>) -> DiagResult<EdgeString> {
    // unwrap not layers, their parity will determine the edge
    let mut not_count: u32 = 0;
    let mut curr = expr.inner;
    loop {
        if let &IrExpression::Large(curr_large) = curr {
            if let IrExpressionLarge::BoolNot(inner) = &large[curr_large] {
                not_count += 1;
                curr = inner;
                continue;
            }
        }
        break;
    }

    let (edge, if_prefix) = if not_count % 2 == 0 {
        ("posedge", "")
    } else {
        ("negedge", "!")
    };

    // lower expression
    let mut s = String::new();
    let f = &mut s;
    lower_expression(large, name_map, curr, f)?;

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
    pub fn array(diags: &Diagnostics, span: Span, w: BigUint) -> DiagResult<VerilogType> {
        let w = diag_big_int_to_u32(diags, span, &w.into(), "array width too large")?;
        match NonZeroU32::new(w) {
            None => Ok(VerilogType::ZeroWidth),
            Some(w) => Ok(VerilogType::Array(w)),
        }
    }

    // TODO split tuples and short arrays into multiple ports instead?
    pub fn from_ir_ty(diags: &Diagnostics, span: Span, ty: &IrType) -> DiagResult<VerilogType> {
        match ty {
            IrType::Bool => Ok(VerilogType::Bit),
            IrType::Int(_) | IrType::Tuple(_) | IrType::Array(_, _) => Self::array(diags, span, ty.size_bits()),
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

#[derive(Debug, Copy, Clone)]
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

fn diag_big_int_to_u32(diags: &Diagnostics, span: Span, value: &BigInt, message: &str) -> DiagResult<u32> {
    value.try_into().map_err(|_| {
        diags.report_simple(
            format!("{message}: overflow when converting {value} to u32"),
            span,
            "used here",
        )
    })
}

impl<S: AsRef<str>> Display for LoweredName<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = self.0.as_ref();
        if VERILOG_KEYWORDS.contains(s) {
            // emit escaped identifier,
            //   including extra trailing space just to be sure
            f.write_str(&format!("\\{s} "))
        } else {
            // TODO check for invalid chars and escape those, or at least throw an error
            f.write_str(s)
        }
    }
}

lazy_static! {
    // TODO also include vhdl keywords and ban both in generated output?
    /// Updated to "IEEE Standard for SystemVerilog", IEEE 1800-2023
    static ref VERILOG_KEYWORDS: IndexSet<&'static str> = {
        include_str!("verilog_keywords.txt").lines().map(str::trim).filter(|line| !line.is_empty()).collect()
    };
}

fn maybe_id_as_ref(id: &Spanned<Option<String>>) -> Spanned<Option<&str>> {
    id.as_ref().map_inner(|s| s.as_ref().map(String::as_ref))
}
