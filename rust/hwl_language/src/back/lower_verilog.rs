use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::signal::Polarized;
use crate::front::types::HardwareType;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAssignmentTargetBase, IrAsyncResetInfo, IrBlock, IrBoolBinaryOp,
    IrClockedProcess, IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrIfStatement, IrIntArithmeticOp,
    IrIntCompareOp, IrLargeArena, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo,
    IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrSignal,
    IrStatement, IrTargetStep, IrType, IrVariable, IrVariableInfo, IrVariables, IrWire, IrWireInfo, IrWireOrPort,
    ir_modules_topological_sort,
};
use crate::syntax::ast::PortDirection;
use crate::syntax::pos::{Span, Spanned};
use crate::throw;
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint, Sign};
use crate::util::data::{GrowVec, IndexMapExt};
use crate::util::int::{IntRepresentation, Signed};
use crate::util::{Indent, ResultExt, separator_non_trailing};
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
        top_name_scope: LoweredNameScope::new_root(external_modules.clone()),
        result_source: vec![],
    };

    let modules = ir_modules_topological_sort(modules, top_module);
    for module in modules {
        let result = lower_module(&mut ctx, module)?;
        ctx.module_map.insert_first(module, result);
    }

    let top_name = ctx.module_map.get(&top_module).unwrap().name.clone();
    Ok(LoweredVerilog {
        source: ctx.result_source.join("\n\n"),
        top_module_name: top_name.0.clone(),
        debug_info_module_map: ctx.module_map.into_iter().map(|(k, v)| (k, v.name.0)).collect(),
    })
}

struct LowerContext<'a> {
    diags: &'a Diagnostics,
    modules: &'a IrModules,
    module_map: IndexMap<IrModule, LoweredModule>,
    top_name_scope: LoweredNameScope<'static>,
    result_source: Vec<String>,
}

#[derive(Debug, Clone)]
struct LoweredName<S: AsRef<str> = String>(S);

#[derive(Debug, Clone)]
struct LoweredModule {
    name: LoweredName,
    ports: IndexMap<IrPort, LoweredName>,
}

#[derive(Default)]
struct LoweredNameScope<'p> {
    parent: Option<&'p LoweredNameScope<'p>>,
    local_used: IndexSet<String>,
}

impl<'p> LoweredNameScope<'p> {
    pub fn new_root(used: IndexSet<String>) -> Self {
        Self {
            parent: None,
            local_used: used,
        }
    }

    pub fn new_child(&'p self) -> Self {
        Self {
            parent: Some(self),
            local_used: IndexSet::new(),
        }
    }

    pub fn exact_for_new_id(&mut self, diags: &Diagnostics, span: Span, id: &str) -> DiagResult<LoweredName> {
        check_identifier_valid(diags, Spanned { span, inner: id })?;
        if self.is_used(id) {
            throw!(diags.report_internal_error(span, format!("lowered identifier `{id}` already used its scope")))
        }
        Ok(LoweredName(id.to_owned()))
    }

    pub fn make_unique_maybe_id(&mut self, diags: &Diagnostics, id: Spanned<Option<&str>>) -> DiagResult<LoweredName> {
        self.make_unique_str(diags, id.span, id.inner.unwrap_or("_"), false)
    }

    #[allow(dead_code)]
    pub fn make_unique_id(&mut self, diags: &Diagnostics, id: Spanned<&str>) -> DiagResult<LoweredName> {
        self.make_unique_str(diags, id.span, id.inner, false)
    }

    pub fn make_unique_str(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        string: &str,
        force_index: bool,
    ) -> DiagResult<LoweredName> {
        // TODO avoid repeated allocations in this function
        //   * for each str, store the next (potentially) valid suffix
        //   * repeatedly truncate and re-add suffix, instead of creating new strings
        check_identifier_valid(diags, Spanned { span, inner: string })?;

        if !force_index && !self.is_used(string) {
            self.local_used.insert(string.to_owned());
            return Ok(LoweredName(string.to_owned()));
        }

        for i in 0u32.. {
            let suffixed = format!("{string}_{i}");
            if !self.is_used(&suffixed) {
                self.local_used.insert(suffixed.clone());
                return Ok(LoweredName(suffixed));
            }
        }

        throw!(diags.report_internal_error(
            span,
            format!("failed to generate unique lowered identifier for `{string}`")
        ))
    }

    fn is_used(&self, s: &str) -> bool {
        let mut curr = self;
        loop {
            if curr.local_used.contains(s) {
                return true;
            }
            match curr.parent {
                Some(p) => curr = p,
                None => return false,
            }
        }
    }
}

// TODO replace with name mangling that forces everything to be valid
fn check_identifier_valid(diags: &Diagnostics, id: Spanned<&str>) -> DiagResult {
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
struct NameMap<'n> {
    ports: &'n IndexMap<IrPort, LoweredName>,
    registers_inner: &'n IndexMap<IrRegister, LoweredName>,
    registers_outer: &'n IndexMap<IrRegister, LoweredName>,
    wires: &'n IndexMap<IrWire, LoweredName>,
    variables: &'n IndexMap<IrVariable, LoweredName>,
}

impl<'n> NameMap<'n> {
    pub fn map_signal(&self, signal: IrSignal) -> &'n LoweredName {
        match signal {
            IrSignal::Port(port) => self.ports.get(&port).unwrap(),
            IrSignal::Wire(wire) => self.wires.get(&wire).unwrap(),
            IrSignal::Register(reg) => self.map_reg(reg),
        }
    }

    pub fn map_reg(&self, reg: IrRegister) -> &'n LoweredName {
        self.registers_inner
            .get(&reg)
            .unwrap_or_else(|| self.registers_outer.get(&reg).unwrap())
    }

    pub fn map_var(self, var: IrVariable) -> &'n LoweredName {
        self.variables.get(&var).unwrap()
    }
}

fn lower_module(ctx: &mut LowerContext, module: IrModule) -> DiagResult<LoweredModule> {
    let diags = ctx.diags;
    assert!(!ctx.module_map.contains_key(&module));

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

    ctx.result_source.push(f);
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
        let end_str = separator_non_trailing(",", port_index, last_actual_port_index.unwrap_or(0) + 1);

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
        swriteln!(
            f,
            "{I}{prefix_str}reg {ty_prefix_str}{name}; // {signal_type} {name_debug_str}: {debug_info_domain} {ty_debug_str}"
        );
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
) -> DiagResult {
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

                let temporaries = GrowVec::new();
                let mut ctx = LowerBlockContext {
                    diags,
                    large,
                    name_map,
                    name_scope: module_name_scope.new_child(),
                    temporaries: &temporaries,
                    indent: Indent::new(2),
                    newline: &mut newline,
                    f,
                };

                ctx.newline.start_new_block();
                ctx.lower_block(block)?;

                if !temporaries.is_empty() {
                    todo!()
                }

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

                let clock_edge = lower_edge(outer_name_map, clock_signal.inner)?;
                let async_reset = async_reset
                    .as_ref()
                    .map(|info| {
                        let reset_edge = lower_edge(outer_name_map, info.signal.inner)?;
                        Ok((reset_edge, info))
                    })
                    .transpose()?;

                match &async_reset {
                    Some((reset_edge, _)) => swriteln!(
                        f,
                        "{I}always @({} {}, {} {}) begin",
                        clock_edge.edge,
                        clock_edge.signal,
                        reset_edge.edge,
                        reset_edge.signal,
                    ),
                    None => swriteln!(f, "{I}always @({} {}) begin", clock_edge.edge, clock_edge.signal,),
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
                        let IrAsyncResetInfo { signal: _, resets } = reset_info;

                        // reset, using outer name map (no shadowing)
                        swriteln!(f, "{I}{I}if ({}{}) begin", reset_edge.if_prefix, reset_edge.signal);
                        let indent_inner = Indent::new(3);

                        for reset in resets {
                            let (reg, value) = &reset.inner;
                            let reg_name = reg_name_map.get(reg).unwrap();

                            // TODO maybe we can avoid this here if we only allow single-expression reset values?
                            let temporaries = GrowVec::new();
                            let mut ctx = LowerBlockContext {
                                diags,
                                large,
                                name_map: outer_name_map,
                                name_scope: module_name_scope.new_child(),
                                temporaries: &temporaries,
                                indent: indent_inner,
                                newline: &mut newline,
                                f,
                            };
                            let value = ctx.lower_expression(reset.span, value)?;
                            if !temporaries.is_empty() {
                                todo!()
                            }

                            swriteln!(f, "{indent_inner}{reg_name} <= {value};");
                        }

                        swriteln!(f, "{I}{I}end else begin");
                        indent_inner
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

                let temporaries = GrowVec::new();
                let mut ctx = LowerBlockContext {
                    diags,
                    large,
                    name_map: inner_name_map,
                    name_scope: module_name_scope.new_child(),
                    temporaries: &temporaries,
                    indent: indent_clocked,
                    newline: &mut newline,
                    f,
                };
                ctx.lower_block(clock_block)?;
                if temporaries.is_empty() {
                    todo!()
                }

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
                lower_port_connections(f, name_map, port_name, port_connections)?;
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
                lower_port_connections(f, name_map, port_name, port_connections)?;
            }
        }
    }

    Ok(())
}

fn lower_port_connections<S: AsRef<str>>(
    f: &mut String,
    name_map: NameMap,
    port_name: impl Fn(usize) -> LoweredName<S>,
    port_connections: &Vec<Spanned<IrPortConnection>>,
) -> DiagResult {
    swrite!(f, "(");

    if port_connections.is_empty() {
        swriteln!(f, ");");
        return Ok(());
    }
    swriteln!(f);

    for (port_index, connection) in enumerate(port_connections) {
        let port_name = port_name(port_index);
        swrite!(f, "{I}{I}.{port_name}(");

        match connection.inner {
            IrPortConnection::Input(expr) => {
                let expr = name_map.map_signal(expr.inner);
                swrite!(f, "{expr}");
            }
            IrPortConnection::Output(signal) => {
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
            module_name_scope.make_unique_str(diags, debug_info_id.span, &format!("shadow_{register_name}"), false)?;

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

#[derive(Debug)]
enum Evaluated<'n> {
    Name(&'n LoweredName),
    #[allow(dead_code)]
    Temporary(Temporary<'n>),
    String(String),
    Str(&'static str),
}

#[derive(Debug, Copy, Clone)]
struct Temporary<'n>(&'n LoweredName);

#[allow(dead_code)]
#[derive(Debug)]
struct TemporaryInfo {
    name: LoweredName,
    ty: VerilogType,
}

impl Display for Temporary<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Evaluated<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Evaluated::Name(n) => write!(f, "{}", n),
            Evaluated::Temporary(tmp) => write!(f, "{}", tmp),
            Evaluated::String(s) => write!(f, "{}", s),
            Evaluated::Str(s) => write!(f, "{}", s),
        }
    }
}

#[allow(dead_code)]
struct LowerBlockContext<'a, 'n> {
    diags: &'a Diagnostics,
    large: &'a IrLargeArena,

    name_map: NameMap<'n>,
    name_scope: LoweredNameScope<'a>,
    temporaries: &'n GrowVec<TemporaryInfo>,

    indent: Indent,
    newline: &'a mut NewlineGenerator,
    f: &'a mut String,
}

impl<'a, 'n> LowerBlockContext<'a, 'n> {
    fn indent(&mut self, f: impl FnOnce(&mut Self) -> DiagResult) -> DiagResult {
        let old_indent = self.indent;
        self.indent = self.indent.nest();
        let result = f(self);
        self.indent = old_indent;
        result
    }

    fn lower_block(&mut self, block: &IrBlock) -> DiagResult {
        let IrBlock { statements } = block;

        for stmt in statements {
            self.newline.before_item(self.f);
            self.lower_statement(stmt.as_ref())?;
        }

        Ok(())
    }

    fn lower_statement(&mut self, stmt: Spanned<&IrStatement>) -> DiagResult {
        let indent = self.indent;

        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                let target = self.lower_assign_target(stmt.span, target)?;
                let source = self.lower_expression(stmt.span, source)?;
                swriteln!(self.f, "{indent}{target} = {source};");
            }
            IrStatement::Block(inner) => {
                swriteln!(self.f, "{indent}begin");
                self.indent(|s| s.lower_block(inner))?;
                swriteln!(self.f, "{indent}end");
            }
            IrStatement::If(IrIfStatement {
                condition,
                then_block,
                else_block,
            }) => {
                let cond = self.lower_expression(stmt.span, condition)?;
                swrite!(self.f, "{indent}if ({cond}) begin");
                self.indent(|s| s.lower_block(then_block))?;
                swrite!(self.f, "{indent}end");

                // TODO skip empty else blocks here or in a common IR optimization pass?
                if let Some(else_block) = else_block {
                    swriteln!(self.f, " else begin");
                    self.indent(|s| s.lower_block(else_block))?;
                    swrite!(self.f, "{indent}end");
                }

                swriteln!(self.f);
            }
            IrStatement::PrintLn(s) => {
                // TODO properly escape string
                swriteln!(self.f, "{indent}$display(\"{s}\");");
            }
        }
        Ok(())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn lower_expression(&mut self, span: Span, expr: &IrExpression) -> DiagResult<Evaluated<'n>> {
        let name_map = self.name_map;

        let eval = match expr {
            &IrExpression::Bool(x) => {
                let s = if x { "1'b0" } else { "1'b1" };
                Evaluated::Str(s)
            }
            IrExpression::Int(x) => Evaluated::String(lower_int_str(x)),

            &IrExpression::Port(s) => Evaluated::Name(name_map.map_signal(IrSignal::Port(s))),
            &IrExpression::Wire(s) => Evaluated::Name(name_map.map_signal(IrSignal::Wire(s))),
            &IrExpression::Register(s) => Evaluated::Name(name_map.map_signal(IrSignal::Register(s))),
            &IrExpression::Variable(v) => Evaluated::Name(name_map.map_var(v)),

            &IrExpression::Large(expr) => {
                match &self.large[expr] {
                    IrExpressionLarge::BoolNot(inner) => {
                        let inner = self.lower_expression(span, inner)?;
                        Evaluated::String(format!("(!{inner}"))
                    }
                    IrExpressionLarge::BoolBinary(op, left, right) => {
                        // logical and bitwise operators would both work,
                        //   bitwise is more consistent since it also has an xor operator
                        let op_str = match op {
                            IrBoolBinaryOp::And => "&",
                            IrBoolBinaryOp::Or => "|",
                            IrBoolBinaryOp::Xor => "^",
                        };

                        let left = self.lower_expression(span, left)?;
                        let right = self.lower_expression(span, right)?;

                        Evaluated::String(format!("({left} {op_str} {right})"))
                    }
                    IrExpressionLarge::IntArithmetic(op, ty, left, right) => {
                        let _ = ty;

                        // TODO div/mod truncation directions are not right
                        // TODO pow is very likely not right
                        // TODO lower mod to branch + sub, lower mul/pow to shift if possible
                        let left = self.lower_expression(span, left)?;
                        let right = self.lower_expression(span, right)?;
                        let op_str = match op {
                            IrIntArithmeticOp::Add => "+",
                            IrIntArithmeticOp::Sub => "-",
                            IrIntArithmeticOp::Mul => "*",
                            IrIntArithmeticOp::Div => "/",
                            IrIntArithmeticOp::Mod => "%",
                            IrIntArithmeticOp::Pow => "**",
                        };
                        Evaluated::String(format!("({left} {op_str} {right})"))
                    }
                    IrExpressionLarge::IntCompare(op, left, right) => {
                        // TODO bit-widths are probably not correct
                        let op_str = match op {
                            IrIntCompareOp::Eq => "==",
                            IrIntCompareOp::Neq => "!=",
                            IrIntCompareOp::Lt => "<",
                            IrIntCompareOp::Lte => "<=",
                            IrIntCompareOp::Gt => ">",
                            IrIntCompareOp::Gte => ">=",
                        };

                        let left = self.lower_expression(span, left)?;
                        let right = self.lower_expression(span, right)?;

                        Evaluated::String(format!("({left} {op_str} {right})"))
                    }

                    IrExpressionLarge::TupleLiteral(elements) => {
                        // verilog does not care much about types, this is just a concatenation
                        //  (assuming all sub-expression have the right width, which they should)
                        // TODO this is probably incorrect in general, we need to store the tuple in a variable first
                        let mut g = String::new();
                        swrite!(g, "{{");
                        for (i, elem) in enumerate(elements) {
                            let elem = self.lower_expression(span, elem)?;
                            swrite!(g, "{elem}");
                            swrite!(g, "{}", separator_non_trailing(",", i, elements.len()));
                        }
                        swrite!(g, "}}");
                        Evaluated::String(g)
                    }
                    IrExpressionLarge::ArrayLiteral(_inner_ty, _len, elements) => {
                        // verilog does not care much about types, this is just a concatenation
                        //  (assuming all sub-expression have the right width, which they should)
                        // TODO skip for zero-sized array? we probably need a more general way to skip zero-sized expressions
                        // TODO use repeat operator if array elements are repeated
                        // TODO the order is wrong, the verilog array operator is the wrong way around
                        let mut g = String::new();
                        swrite!(g, "{{");
                        for (i, elem) in enumerate(elements) {
                            let inner = match elem {
                                IrArrayLiteralElement::Spread(inner) => inner,
                                IrArrayLiteralElement::Single(inner) => inner,
                            };

                            let inner = self.lower_expression(span, inner)?;
                            swrite!(g, "{}", inner);
                            swrite!(g, "{}", separator_non_trailing(",", i, elements.len()));
                        }
                        swrite!(g, "}}");
                        Evaluated::String(g)
                    }

                    IrExpressionLarge::TupleIndex { base, index } => {
                        // TODO this is completely wrong
                        let base = self.lower_expression(span, base)?;
                        Evaluated::String(format!("({base}[{index}])"))
                    }
                    IrExpressionLarge::ArrayIndex { base, index } => {
                        // TODO this is probably incorrect in general, we need to store the array in a variable first
                        // TODO we're incorrectly using array indices as bit indices here
                        let base = self.lower_expression(span, base)?;
                        let index = self.lower_expression(span, index)?;
                        Evaluated::String(format!("({base}[{index}])"))
                    }
                    IrExpressionLarge::ArraySlice { base, start, len } => {
                        // TODO this is probably incorrect in general, we need to store the array in a variable first
                        // TODO we're incorrectly using array indices as bit indices here
                        let base = self.lower_expression(span, base)?;
                        let start = self.lower_expression(span, start)?;
                        let len = lower_uint_str(len);
                        Evaluated::String(format!("({base}[{start}+:{len}])"))
                    }

                    IrExpressionLarge::ToBits(_ty, value) => {
                        // in verilog everything is just a bit vector, so we don't need to do anything
                        self.lower_expression(span, value)?
                    }
                    IrExpressionLarge::FromBits(_ty, value) => {
                        // in verilog everything is just a bit vector, so we don't need to do anything
                        self.lower_expression(span, value)?
                    }
                    IrExpressionLarge::ExpandIntRange(target, value) => {
                        // cast the value to the right signedness
                        //   and add a literal of the right sign and size to force expansion
                        let target_repr = IntRepresentation::for_range(target);
                        let target_size = target_repr.size_bits();

                        let value = self.lower_expression(span, value)?;

                        let s = match target_repr.signed() {
                            Signed::Signed => format!("$unsigned({target_size}'sd0 + $signed({value}))"),
                            Signed::Unsigned => format!("({target_size}'d0 + {value})"),
                        };
                        Evaluated::String(s)
                    }
                    IrExpressionLarge::ConstrainIntRange(target, value) => {
                        // TODO this not correct, we're not actually lowering the bit width
                        // TODO add assertions?
                        let target_repr = IntRepresentation::for_range(target);
                        let _ = target_repr;
                        self.lower_expression(span, value)?
                    }
                }
            }
        };
        Ok(eval)
    }

    fn lower_assign_target(&mut self, span: Span, target: &IrAssignmentTarget) -> DiagResult<Evaluated<'n>> {
        // TODO this is probably wrong, we might need intermediate variables for the base and after each step
        let IrAssignmentTarget { base, steps } = target;

        let name_map = self.name_map;
        let base = match *base {
            IrAssignmentTargetBase::Port(s) => name_map.map_signal(IrSignal::Port(s)),
            IrAssignmentTargetBase::Wire(s) => name_map.map_signal(IrSignal::Wire(s)),
            IrAssignmentTargetBase::Register(s) => name_map.map_signal(IrSignal::Register(s)),
            IrAssignmentTargetBase::Variable(r) => name_map.map_var(r),
        };

        // early exit to avoid string allocation
        if steps.is_empty() {
            return Ok(Evaluated::Name(base));
        }

        let mut g = String::new();
        swrite!(g, "{base}");

        // TODO both of these are wrong, we're not taking element type sizes into account
        // TODO this entire thing should just be flattened to a single slice
        for step in steps {
            match step {
                IrTargetStep::ArrayIndex(start) => {
                    let start = self.lower_expression(span, start)?;
                    swrite!(g, "[{start}]");
                }
                IrTargetStep::ArraySlice(start, len) => {
                    let start = self.lower_expression(span, start)?;
                    let len = lower_uint_str(len);
                    swrite!(g, "[{start}+:{len}]");
                }
            }
        }

        Ok(Evaluated::String(g))
    }

    #[allow(dead_code)]
    fn new_temporary(&mut self, ty: Spanned<IrType>) -> DiagResult<Temporary<'n>> {
        let ty_verilog = VerilogType::from_ir_ty(self.diags, ty.span, &ty.inner)?;
        let name = self.name_scope.make_unique_str(self.diags, ty.span, "tmp", true)?;
        let info = TemporaryInfo { name, ty: ty_verilog };

        let info = self.temporaries.push(info);
        Ok(Temporary(&info.name))
    }
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
struct EdgeString<'n> {
    edge: &'static str,
    if_prefix: &'static str,
    signal: &'n LoweredName,
}

fn lower_edge(name_map: NameMap, expr: Polarized<IrSignal>) -> DiagResult<EdgeString> {
    let Polarized { inverted, signal } = expr;
    let (edge, if_prefix) = match inverted {
        false => ("posedge", ""),
        true => ("negedge", "!"),
    };

    let signal = name_map.map_signal(signal);

    Ok(EdgeString {
        edge,
        if_prefix,
        signal,
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
