use crate::back::lower_verilog::non_zero_width::NonZeroWidthRange;
use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::signal::Polarized;
use crate::front::types::{ClosedIncRange, HardwareType};
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrBoolBinaryOp, IrClockedProcess,
    IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrIntArithmeticOp,
    IrIntCompareOp, IrIntegerRadix, IrLargeArena, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo,
    IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrSignal,
    IrSignalOrVariable, IrStatement, IrStringSubstitution, IrTargetStep, IrType, IrVariable, IrVariableInfo,
    IrVariables, IrWire, IrWireInfo, ValueAccess, ir_modules_topological_sort,
};
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::syntax::pos::{Span, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint, Sign};
use crate::util::data::{GrowVec, IndexMapExt};
use crate::util::int::{IntRepresentation, Signed};
use crate::util::{Indent, ResultExt, separator_non_trailing};
use crate::{throw, try_inner};
use hwl_util::{swrite, swriteln};
use indexmap::{IndexMap, IndexSet};
use itertools::{Either, enumerate};
use lazy_static::lazy_static;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use unwrap_match::unwrap_match;

const I: &str = Indent::I;

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
// TODO avoid a bunch of string allocations
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
    ports: IndexMap<IrPort, Result<LoweredName, ZeroWidth>>,
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
        self.local_used.insert(id.to_owned());
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

        for i in 0u64.. {
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

impl<'n> NameMap<'n> {
    pub fn map_signal_or_var(&self, v: IrSignalOrVariable) -> Result<&'n LoweredName, Either<ZeroWidth, NotRead>> {
        match v {
            IrSignalOrVariable::Signal(v) => self.map_signal(v).map_err(Either::Left),
            IrSignalOrVariable::Variable(v) => self.map_var(v),
        }
    }

    pub fn map_signal(&self, signal: IrSignal) -> Result<&'n LoweredName, ZeroWidth> {
        match signal {
            IrSignal::Port(port) => self.ports.get(&port).unwrap().as_ref_ok(),
            IrSignal::Wire(wire) => self.wires.get(&wire).unwrap().as_ref_ok(),
            IrSignal::Register(reg) => self.map_reg(reg),
        }
    }

    pub fn map_reg(&self, reg: IrRegister) -> Result<&'n LoweredName, ZeroWidth> {
        match self.regs_shadowed.get(&reg) {
            Some(name) => Ok(name),
            None => self.regs_outer.get(&reg).unwrap().as_ref_ok(),
        }
    }

    pub fn map_var(self, var: IrVariable) -> Result<&'n LoweredName, Either<ZeroWidth, NotRead>> {
        self.variables.get(&var).unwrap().as_ref_ok()
    }
}

#[derive(Debug, Copy, Clone)]
struct NameMap<'n> {
    ports: &'n IndexMap<IrPort, Result<LoweredName, ZeroWidth>>,
    wires: &'n IndexMap<IrWire, Result<LoweredName, ZeroWidth>>,
    regs_shadowed: &'n IndexMap<IrRegister, LoweredName>,
    regs_outer: &'n IndexMap<IrRegister, Result<LoweredName, ZeroWidth>>,
    variables: &'n IndexMap<IrVariable, Result<LoweredName, Either<ZeroWidth, NotRead>>>,
}

const SENSITIVITY_DUMMY_NAME_SIGNAL: &str = "dummy";
const SENSITIVITY_DUMMY_NAME_VAR: &str = "dummy_var";

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

    // TODO don't use absolute paths here, they cause non-reproducible builds
    swriteln!(f, "// module {}", debug_info_id.inner.unwrap_or("_"));
    swriteln!(f, "//   defined in \"{debug_info_file}\"",);

    if let Some(generic_args) = debug_info_generic_args {
        swriteln!(f, "//   instantiated with generic arguments:");
        for (arg_name, arg_value) in generic_args {
            swriteln!(f, "//     {}={}", arg_name, arg_value);
        }
    }

    let mut module_name_scope = LoweredNameScope::default();

    swrite!(f, "module {}(", module_name);
    let port_name_map = lower_module_ports(diags, ports, &mut module_name_scope, &mut f)?;
    swriteln!(f, ");");

    let mut newline_module = NewlineGenerator::new();
    let (reg_name_map, wire_name_map) = lower_module_signals(
        diags,
        &mut module_name_scope,
        registers,
        wires,
        &mut newline_module,
        &mut f,
    )?;

    // declare dummy signal, used to ensure combinatorial block sensitivity lists are never empty
    // TODO only declare when actually used?
    let dummy_name = module_name_scope.make_unique_str(
        diags,
        module_info.debug_info_id.span,
        SENSITIVITY_DUMMY_NAME_SIGNAL,
        false,
    )?;
    newline_module.start_item(&mut f);
    swriteln!(f, "{I}wire {dummy_name} = 1'b0;");

    lower_module_statements(
        ctx,
        large,
        module_info,
        &mut module_name_scope,
        &dummy_name,
        &port_name_map,
        &reg_name_map,
        &wire_name_map,
        registers,
        processes,
        &mut newline_module,
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
) -> DiagResult<IndexMap<IrPort, Result<LoweredName, ZeroWidth>>> {
    let mut port_lines = vec![];
    let mut port_name_map = IndexMap::new();
    let mut last_actual_port_index = None;

    // TODO remove second iteration,
    //   we only have it avoid redundant trailing commas which is also possible in a single pass
    for (port_index, (port, port_info)) in enumerate(ports) {
        let IrPortInfo {
            name,
            direction,
            ty,
            debug_span,
            debug_info_ty,
            debug_info_domain,
        } = port_info;

        // allocate port names even for zero-width ports to get some extra stability
        // TODO check that port names are valid and unique
        let lower_name = module_name_scope.exact_for_new_id(diags, *debug_span, name)?;

        let port_ty = VerilogType::new_from_ir(diags, *debug_span, ty)?;

        let slot;
        let (is_non_zero_width, ty_str) = match port_ty {
            Ok(ty_str) => {
                slot = ty_str.to_prefix().to_string();
                (true, slot.as_str())
            }
            Err(ZeroWidth) => (false, "[empty]"),
        };
        let dir_str = match direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output reg",
        };

        if is_non_zero_width {
            last_actual_port_index = Some(port_index);
        }
        port_lines.push((
            is_non_zero_width,
            format!("{dir_str} {ty_str}{lower_name}"),
            format!("{debug_info_domain} {}", debug_info_ty.inner),
        ));

        let insert_name = if is_non_zero_width {
            Ok(lower_name)
        } else {
            Err(ZeroWidth)
        };
        port_name_map.insert_first(port, insert_name);
    }

    for (port_index, (is_actual_port, main_str, comment_str)) in enumerate(port_lines) {
        swrite!(f, "\n    ");

        let start_str = if is_actual_port { "" } else { "// " };
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
) -> DiagResult<(
    IndexMap<IrRegister, Result<LoweredName, ZeroWidth>>,
    IndexMap<IrWire, Result<LoweredName, ZeroWidth>>,
)> {
    let mut lower_signal = |signal_kind,
                            ty,
                            debug_info_id: &Spanned<Option<String>>,
                            debug_info_ty: &HardwareType,
                            debug_domain,
                            newline: &mut NewlineGenerator,
                            f: &mut String| {
        // skip zero-width signals
        let ty_verilog = match VerilogType::new_from_ir(diags, debug_info_id.span, ty)? {
            Ok(ty) => ty,
            Err(ZeroWidth) => return Ok(Err(ZeroWidth)),
        };
        let ty_verilog_prefix = ty_verilog.to_prefix();

        // pick a unique name
        let debug_info_id = maybe_id_as_ref(debug_info_id);
        let name = module_name_scope.make_unique_maybe_id(diags, debug_info_id)?;

        // figure out debug info
        let debug_name = debug_info_id.inner.unwrap_or("_");
        let debug_ty = debug_info_ty.diagnostic_string();

        // both regs and wires lower to verilog "regs", which are really just "signals that are written by processes"
        newline.start_item(f);
        swriteln!(
            f,
            "{I}reg {ty_verilog_prefix}{name}; // {signal_kind} {debug_name}: {debug_domain} {debug_ty}"
        );

        Ok(Ok(name))
    };

    let mut reg_name_map = IndexMap::new();
    let mut wire_name_map = IndexMap::new();

    newline.start_group();
    for (register, register_info) in registers {
        let IrRegisterInfo {
            ty,
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
        } = register_info;

        let name = lower_signal("reg", ty, debug_info_id, debug_info_ty, debug_info_domain, newline, f)?;
        reg_name_map.insert_first(register, name);
    }

    newline.start_group();
    for (wire, wire_info) in wires {
        let IrWireInfo {
            ty,
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
        } = &wire_info;
        let name = lower_signal("wire", ty, debug_info_id, debug_info_ty, debug_info_domain, newline, f)?;
        wire_name_map.insert_first(wire, name);
    }

    Ok((reg_name_map, wire_name_map))
}

fn lower_module_statements(
    ctx: &mut LowerContext,
    large: &IrLargeArena,
    module: &IrModuleInfo,
    module_name_scope: &mut LoweredNameScope,
    dummy_name: &LoweredName,
    port_name_map: &IndexMap<IrPort, Result<LoweredName, ZeroWidth>>,
    reg_name_map: &IndexMap<IrRegister, Result<LoweredName, ZeroWidth>>,
    wire_name_map: &IndexMap<IrWire, Result<LoweredName, ZeroWidth>>,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    children: &[Spanned<IrModuleChild>],
    newline_module: &mut NewlineGenerator,
    f: &mut String,
) -> DiagResult {
    let diags = ctx.diags;

    for (child_index, child) in enumerate(children) {
        newline_module.start_group_and_item(f);

        match &child.inner {
            IrModuleChild::CombinatorialProcess(process) => {
                let IrCombinatorialProcess { locals, block } = process;
                let variables_read = collect_variables_read_and_shadow_registers(large, block, None);

                swriteln!(f, "{I}always @(*) begin");
                let indent = Indent::new(2);

                // Processes with empty sensitivity lists never run, not even at startup
                //   to avoid this, we make all combinatorial processes sensitive to a dummy signal.
                //   Determining when this can be skipped is tricky,
                //   not all IRSignals that are used actually end up being used in the generated verilog.
                let mut name_scope = module_name_scope.new_child();
                let dummy_name_internal =
                    name_scope.make_unique_str(diags, child.span, SENSITIVITY_DUMMY_NAME_VAR, false)?;
                swriteln!(f, "{indent}reg {dummy_name_internal};");

                let mut newline_process = NewlineGenerator::new();
                let variables =
                    declare_locals(diags, f, &mut newline_process, &mut name_scope, locals, variables_read)?;

                let temporaries = GrowVec::new();
                let temporaries_offset = f.len();

                newline_process.start_item(f);
                swriteln!(f, "{indent}{dummy_name_internal} = {dummy_name};");

                let name_map = NameMap {
                    ports: port_name_map,
                    wires: wire_name_map,
                    regs_outer: reg_name_map,
                    regs_shadowed: &IndexMap::default(),
                    variables: &variables,
                };
                newline_process.start_group();
                let mut ctx = LowerBlockContext {
                    diags,
                    large,
                    module,
                    locals,
                    name_map,
                    name_scope: &mut name_scope,
                    temporaries: &temporaries,
                    indent,
                    newline: &mut newline_process,
                    f,
                };
                ctx.lower_block(block)?;

                declare_temporaries(f, temporaries_offset, temporaries);

                swriteln!(f, "{I}end");
            }
            IrModuleChild::ClockedProcess(process) => {
                let &IrClockedProcess {
                    ref locals,
                    clock_signal,
                    ref clock_block,
                    ref async_reset,
                } = process;

                // lower the edges
                let outer_name_map = NameMap {
                    ports: port_name_map,
                    wires: wire_name_map,
                    regs_outer: reg_name_map,
                    regs_shadowed: &IndexMap::default(),
                    variables: &IndexMap::new(),
                };

                let clock_edge = lower_edge(diags, outer_name_map, clock_signal)?;
                let async_reset = async_reset
                    .as_ref()
                    .map(|info| {
                        let reset_edge = lower_edge(diags, outer_name_map, info.signal)?;
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

                let mut newline_process = NewlineGenerator::new();
                let mut name_scope = module_name_scope.new_child();

                // declare locals and shadow registers
                let mut shadow_regs = IndexMap::new();
                let variables_read =
                    collect_variables_read_and_shadow_registers(large, clock_block, Some(&mut shadow_regs));

                let variables =
                    declare_locals(diags, f, &mut newline_process, &mut name_scope, locals, variables_read)?;
                let reg_name_map_shadowed =
                    declare_shadow_registers(diags, f, &mut newline_process, &mut name_scope, registers, &shadow_regs)?;

                let temporaries = GrowVec::new();
                let temporaries_offset = f.len();

                // async reset header if any
                let indent_clocked = match &async_reset {
                    None => Indent::new(2),
                    Some((reset_edge, reset_info)) => {
                        let IrAsyncResetInfo { signal: _, resets } = reset_info;

                        // reset, using outer name map (no shadowing necessary since we only write)
                        swriteln!(f, "{I}{I}if ({}{}) begin", reset_edge.if_prefix, reset_edge.signal);
                        let indent_inner = Indent::new(3);

                        for reset in resets {
                            let (reg, value) = &reset.inner;
                            let reg_name = match reg_name_map.get(reg).unwrap() {
                                Ok(reg_name) => reg_name,
                                Err(ZeroWidth) => continue,
                            };

                            let mut ctx_reset = LowerBlockContext {
                                diags,
                                large,
                                module,
                                locals,
                                name_map: outer_name_map,
                                name_scope: &mut name_scope,
                                temporaries: &temporaries,
                                indent: indent_inner,
                                newline: &mut newline_process,
                                f,
                            };

                            let value = unwrap_zero_width(ctx_reset.lower_expression(reset.span, value)?);
                            swriteln!(f, "{indent_inner}{reg_name} <= {value};");
                        }

                        swriteln!(f, "{I}{I}end else begin");
                        indent_inner
                    }
                };

                // populate shadow registers
                newline_process.start_group();
                for (&reg, shadow_name) in &reg_name_map_shadowed {
                    let access = &shadow_regs[&reg];
                    if access.any_read {
                        newline_process.start_item(f);
                        let orig_name = match reg_name_map.get(&reg).unwrap() {
                            Ok(orig_name) => orig_name,
                            Err(ZeroWidth) => continue,
                        };
                        swriteln!(f, "{indent_clocked}{shadow_name} = {orig_name};");
                    }
                }

                // lower body itself, using inner name map (with shadowing)
                let inner_name_map = NameMap {
                    ports: port_name_map,
                    regs_outer: reg_name_map,
                    regs_shadowed: &reg_name_map_shadowed,
                    wires: wire_name_map,
                    variables: &variables,
                };
                newline_process.start_group();
                let mut ctx = LowerBlockContext {
                    diags,
                    large,
                    module,
                    locals,
                    name_map: inner_name_map,
                    name_scope: &mut name_scope,
                    temporaries: &temporaries,
                    indent: indent_clocked,
                    newline: &mut newline_process,
                    f,
                };
                ctx.lower_block(clock_block)?;

                declare_temporaries(f, temporaries_offset, temporaries);

                // write-back shadow registers
                newline_process.start_group();
                for (&reg, shadow_name) in &reg_name_map_shadowed {
                    newline_process.start_item(f);
                    let orig_name = match reg_name_map.get(&reg).unwrap() {
                        Ok(orig_name) => orig_name,
                        Err(ZeroWidth) => continue,
                    };
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
                    regs_outer: reg_name_map,
                    regs_shadowed: &IndexMap::default(),
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
                    regs_outer: reg_name_map,
                    regs_shadowed: &IndexMap::default(),
                    wires: wire_name_map,
                    variables: &IndexMap::new(),
                };
                let port_name = |port_index: usize| Ok(LoweredName(&port_names[port_index]));
                lower_port_connections(f, name_map, port_name, port_connections)?;
            }
        }
    }

    Ok(())
}

fn lower_port_connections<S: AsRef<str>>(
    f: &mut String,
    name_map: NameMap,
    port_name: impl Fn(usize) -> Result<LoweredName<S>, ZeroWidth>,
    port_connections: &Vec<Spanned<IrPortConnection>>,
) -> DiagResult {
    swrite!(f, "(");

    if port_connections.is_empty() {
        swriteln!(f, ");");
        return Ok(());
    }
    swriteln!(f);

    let mut first = true;

    for (port_index, connection) in enumerate(port_connections) {
        let port_name = match port_name(port_index) {
            Ok(port_name) => port_name,
            Err(ZeroWidth) => continue,
        };

        if !first {
            swriteln!(f, ",");
        }
        swrite!(f, "{I}{I}.{port_name}(");

        match connection.inner {
            IrPortConnection::Input(expr) => {
                let signal_name = unwrap_zero_width(name_map.map_signal(expr.inner));
                swrite!(f, "{signal_name}");
            }
            IrPortConnection::Output(signal) => {
                match signal {
                    None => {
                        // write nothing, resulting in an empty `()`
                    }
                    Some(signal) => {
                        let signal_name = unwrap_zero_width(name_map.map_signal(signal.as_signal()));
                        swrite!(f, "{signal_name}");
                    }
                }
            }
        }

        swrite!(f, ")",);
        first = false;
    }

    if !first {
        swriteln!(f);
    }
    swriteln!(f, "{I});");
    Ok(())
}

#[derive(Debug, Copy, Clone)]
struct NotRead;

#[derive(Debug, Copy, Clone)]
struct ZeroWidth;

// TODO blocks with variables must be named
/// Declare local variables.
/// We skip declaring variables that never read, any potential writes to them will also be skipped.
fn declare_locals(
    diags: &Diagnostics,
    f: &mut String,
    newline: &mut NewlineGenerator,
    name_scope: &mut LoweredNameScope,
    locals: &IrVariables,
    variables_read: IndexSet<IrVariable>,
) -> DiagResult<IndexMap<IrVariable, Result<LoweredName, Either<ZeroWidth, NotRead>>>> {
    let mut result = IndexMap::new();

    for (variable, variable_info) in locals {
        let name = if variables_read.contains(&variable) {
            let &IrVariableInfo {
                ref ty,
                debug_info_span,
                ref debug_info_id,
            } = variable_info;

            match VerilogType::new_from_ir(diags, debug_info_span, ty)? {
                Ok(ty_verilog) => {
                    let debug_info_id = Spanned::new(debug_info_span, debug_info_id.as_ref().map(String::as_str));
                    let name = name_scope.make_unique_maybe_id(diags, debug_info_id)?;

                    newline.start_item(f);
                    declare_local(f, ty_verilog, &name);
                    Ok(name)
                }
                Err(ZeroWidth) => Err(Either::Left(ZeroWidth)),
            }
        } else {
            Err(Either::Right(NotRead))
        };

        result.insert_first(variable, name);
    }

    Ok(result)
}

fn declare_shadow_registers(
    diags: &Diagnostics,
    f: &mut String,
    newline: &mut NewlineGenerator,
    module_name_scope: &mut LoweredNameScope,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    shadow_regs: &IndexMap<IrRegister, AccessInfo>,
) -> DiagResult<IndexMap<IrRegister, LoweredName>> {
    let mut shadowing_reg_name_map = IndexMap::new();

    for &reg in shadow_regs.keys() {
        let register_info = &registers[reg];
        let debug_info_id = maybe_id_as_ref(&register_info.debug_info_id);

        // skip zero-width registers
        let ty_verilog = match VerilogType::new_from_ir(diags, debug_info_id.span, &register_info.ty)? {
            Ok(ty) => ty,
            Err(ZeroWidth) => continue,
        };

        let register_name = debug_info_id.inner.unwrap_or("_");
        let shadow_name =
            module_name_scope.make_unique_str(diags, debug_info_id.span, &format!("shadow_{register_name}"), false)?;

        newline.start_item(f);
        let ty_verilog_refix = ty_verilog.to_prefix();
        swriteln!(f, "{I}{I}reg {ty_verilog_refix}{shadow_name};");

        shadowing_reg_name_map.insert_first(reg, shadow_name);
    }

    Ok(shadowing_reg_name_map)
}

fn declare_temporaries(f: &mut String, offset: usize, temporaries: GrowVec<TemporaryInfo>) {
    let mut f_inner = String::new();

    let mut any = false;
    for tmp in temporaries.into_vec() {
        let TemporaryInfo { name, ty } = *tmp;
        declare_local(&mut f_inner, ty, &name);
        any = true;
    }

    if any && !&f[offset..].starts_with("\n") {
        f_inner.push('\n');
    }

    f.insert_str(offset, &f_inner);
}

fn declare_local(f: &mut String, ty: VerilogType, name: &LoweredName) {
    // x-initialize all locals to avoid inferring latches
    let ty_prefix = ty.to_prefix();
    let ty_width = ty.width();
    swriteln!(f, "{I}{I}reg {ty_prefix}{name} = {ty_width}'dx;");
}

#[derive(Debug, Copy, Clone, Default)]
struct AccessInfo {
    any_read: bool,
    any_write: bool,
}

fn collect_variables_read_and_shadow_registers(
    large: &IrLargeArena,
    block: &IrBlock,
    mut shadow_regs: Option<&mut IndexMap<IrRegister, AccessInfo>>,
) -> IndexSet<IrVariable> {
    let mut variables_read = IndexSet::new();

    block.visit_values_accessed(large, &mut |signal, access| match (signal, access) {
        (IrSignalOrVariable::Variable(v), ValueAccess::Read) => {
            variables_read.insert(v);
        }
        (IrSignalOrVariable::Signal(IrSignal::Register(reg)), access) => {
            if let Some(shadow_regs) = &mut shadow_regs {
                let entry = shadow_regs.entry(reg).or_default();
                match access {
                    ValueAccess::Read => entry.any_read = true,
                    ValueAccess::Write => entry.any_write = true,
                }
            }
        }
        _ => {}
    });

    if let Some(shadow_regs) = &mut shadow_regs {
        // we only need to shadow registers that are written to
        shadow_regs.retain(|_, k| k.any_write);
    }

    variables_read
}

#[derive(Debug)]
enum Evaluated<'n> {
    Name(&'n LoweredName),
    Temporary(Temporary<'n>),
    String(String),
    Str(&'static str),

    // TODO doc (all variants, but especially this one)
    SignedString(String),
}

#[derive(Debug, Copy, Clone)]
struct Temporary<'n>(&'n LoweredName);

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

impl<'n> Evaluated<'n> {
    pub fn as_signed_maybe(&self, signed: bool) -> impl Display {
        struct S<'s, 'n> {
            inner: &'s Evaluated<'n>,
            signed: bool,
        }
        impl Display for S<'_, '_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                if self.signed {
                    match self.inner {
                        Evaluated::SignedString(s) => write!(f, "{}", s),
                        _ => write!(f, "$signed({})", self.inner),
                    }
                } else {
                    write!(f, "{}", self.inner)
                }
            }
        }
        S { inner: self, signed }
    }

    pub fn as_signed(&self) -> impl Display {
        self.as_signed_maybe(true)
    }
}

impl Display for Evaluated<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Evaluated::Name(n) => write!(f, "{}", n),
            Evaluated::Temporary(tmp) => write!(f, "{}", tmp),
            Evaluated::String(s) => write!(f, "{}", s),
            Evaluated::Str(s) => write!(f, "{}", s),
            Evaluated::SignedString(s) => write!(f, "$unsigned({})", s),
        }
    }
}

struct LowerBlockContext<'a, 'n> {
    diags: &'a Diagnostics,
    large: &'a IrLargeArena,
    module: &'a IrModuleInfo,
    locals: &'a IrVariables,

    name_map: NameMap<'n>,
    name_scope: &'a mut LoweredNameScope<'n>,
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

    fn lower_block_indented(&mut self, block: &IrBlock) -> DiagResult {
        self.indent(|s| s.lower_block(block))
    }

    fn lower_block(&mut self, block: &IrBlock) -> DiagResult {
        let IrBlock { statements } = block;
        for stmt in statements {
            self.lower_statement(stmt.as_ref())?;
        }
        Ok(())
    }

    fn lower_statement(&mut self, stmt: Spanned<&IrStatement>) -> DiagResult {
        let indent = self.indent;
        let diags = self.diags;
        self.newline.start_item(self.f);

        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                let target = match self.lower_assign_target(stmt.span, target)? {
                    Ok(target) => target,
                    Err(e) => {
                        let _: Either<ZeroWidth, NotRead> = e;
                        return Ok(());
                    }
                };
                let source = match self.lower_expression(stmt.span, source)? {
                    Ok(source) => source,
                    Err(ZeroWidth) => return Ok(()),
                };
                swriteln!(self.f, "{indent}{target} = {source};");
            }
            IrStatement::Block(inner) => {
                swriteln!(self.f, "{indent}begin");
                self.lower_block_indented(inner)?;
                swriteln!(self.f, "{indent}end");
            }
            IrStatement::If(IrIfStatement {
                condition,
                then_block,
                else_block,
            }) => {
                // TODO re-merge else-if chains to reduce indentation if there are no other statements in the else block?
                let cond = self.lower_expression_non_zero_width(stmt.span, condition, "condition")?;
                swriteln!(self.f, "{indent}if ({cond}) begin");
                self.lower_block_indented(then_block)?;
                swrite!(self.f, "{indent}end");

                if let Some(else_block) = else_block {
                    swriteln!(self.f, " else begin");
                    self.lower_block_indented(else_block)?;
                    swrite!(self.f, "{indent}end");
                }

                swriteln!(self.f);
            }
            IrStatement::For(IrForStatement { index, range, block }) => {
                let index = *index;
                let index_ty = IrExpression::Variable(index).ty(self.module, self.locals).unwrap_int();

                match NonZeroWidthRange::new(index_ty.as_ref()) {
                    Ok(index_ty) => {
                        let index = match self.name_map.map_var(index) {
                            Ok(index) => index,
                            Err(Either::Left(ZeroWidth)) => {
                                return Err(diags.report_internal_error(stmt.span, "index var zero-width"));
                            }
                            Err(Either::Right(NotRead)) => {
                                return Err(diags.report_internal_error(stmt.span, "index var not read"));
                            }
                        };

                        let start_inc = lower_int_constant(index_ty, &range.start_inc);
                        let end_inc = lower_int_constant(index_ty, &range.end_inc);
                        swriteln!(
                            self.f,
                            "{indent}for({index} = {start_inc}; {index} <= {end_inc}; {index} = {index} + 1) begin"
                        );
                        self.lower_block_indented(block)?;
                        swriteln!(self.f, "{indent}end");
                    }
                    Err(ZeroWidth) => {
                        // the loop would also run for zero iterations, so just skip it
                    }
                }
            }
            IrStatement::Print(pieces) => {
                let mut f_str = String::new();
                let mut f_args = String::new();

                for p in pieces {
                    match p {
                        StringPiece::Literal(p) => {
                            f_str.push_str(&escape_verilog_str(p));
                        }
                        StringPiece::Substitute(p) => match p {
                            IrStringSubstitution::Integer(p, radix) => {
                                // TODO how does signed bin/hex behave?
                                let signed =
                                    p.ty(self.module, self.locals).unwrap_int().as_ref().start_inc < &BigInt::ZERO;
                                let p = self.lower_expression(stmt.span, p)?;

                                let radix = match radix {
                                    IrIntegerRadix::Binary => "%b",
                                    IrIntegerRadix::Decimal => "%d",
                                    IrIntegerRadix::Hexadecimal => "%h",
                                };

                                match p {
                                    Ok(p) => {
                                        swrite!(f_str, "{radix}");
                                        swrite!(f_args, ", {}", p.as_signed_maybe(signed));
                                    }
                                    Err(ZeroWidth) => {
                                        swrite!(f_str, "0");
                                    }
                                }
                            }
                        },
                    }
                }

                swriteln!(self.f, "{indent}$write(\"{f_str}\"{f_args});");
            }
        }
        Ok(())
    }

    fn lower_expression(&mut self, span: Span, expr: &IrExpression) -> DiagResult<Result<Evaluated<'n>, ZeroWidth>> {
        let name_map = self.name_map;
        let indent = self.indent;

        // skip any zero-width expressions
        //   expressions are always pure, they can never have side-effects
        let result_ty = expr.ty(self.module, self.locals);
        let result_ty_verilog = match VerilogType::new_from_ir(self.diags, span, &result_ty)? {
            Ok(ty) => ty,
            Err(ZeroWidth) => {
                return Ok(Err(ZeroWidth));
            }
        };

        let eval = match expr {
            &IrExpression::Bool(x) => {
                let s = if x { "1'b1" } else { "1'b0" };
                Evaluated::Str(s)
            }
            IrExpression::Int(x) => {
                let range =
                    NonZeroWidthRange::new(ClosedIncRange::single(x)).expect("already checked for zero-width earlier");
                lower_int_constant(range, x)
            }

            &IrExpression::Signal(s) => Evaluated::Name(unwrap_zero_width(name_map.map_signal(s))),
            &IrExpression::Variable(v) => {
                Evaluated::Name(name_map.map_var(v).map_err(|e: Either<ZeroWidth, NotRead>| {
                    let _: NotRead = unwrap_zero_width(e.into());
                    self.diags.report_internal_error(span, "reading from not-read variable")
                })?)
            }

            &IrExpression::Large(expr) => {
                match &self.large[expr] {
                    IrExpressionLarge::Undefined(_) => {
                        let width = result_ty_verilog.width();
                        Evaluated::String(format!("{}'bx", width))
                    }
                    IrExpressionLarge::BoolNot(inner) => {
                        let inner = self.lower_expression_non_zero_width(span, inner, "boolean")?;
                        Evaluated::String(format!("(!{inner})"))
                    }
                    IrExpressionLarge::BoolBinary(op, left, right) => {
                        // logical and bitwise operators would both work,
                        //   bitwise is more consistent since it also has an xor operator
                        let op_str = match op {
                            IrBoolBinaryOp::And => "&",
                            IrBoolBinaryOp::Or => "|",
                            IrBoolBinaryOp::Xor => "^",
                        };

                        let left = self.lower_expression_non_zero_width(span, left, "boolean")?;
                        let right = self.lower_expression_non_zero_width(span, right, "boolean")?;

                        Evaluated::String(format!("({left} {op_str} {right})"))
                    }
                    &IrExpressionLarge::IntArithmetic(op, ref result_range, ref left, ref right) => {
                        let result_range = NonZeroWidthRange::new(result_range.as_ref())
                            .expect("already checked for zero-width earlier");
                        self.lower_arithmetic_expression(span, result_range, result_ty_verilog, op, left, right)?
                    }
                    IrExpressionLarge::IntCompare(op, left, right) => {
                        // find common range that contains all operands
                        let left_range = left.ty(self.module, self.locals).unwrap_int();
                        let right_range = right.ty(self.module, self.locals).unwrap_int();
                        let combined_range = left_range.as_ref().union(right_range.as_ref());
                        let combined_range =
                            NonZeroWidthRange::new(combined_range).unwrap_or(NonZeroWidthRange::ZERO_ONE);

                        // lower both operands to that range
                        let left_raw = self.lower_expression_int_expanded(span, combined_range, left)?;
                        let right_raw = self.lower_expression_int_expanded(span, combined_range, right)?;

                        let signed = combined_range.range().start_inc < &BigInt::ZERO;
                        let left = left_raw.as_signed_maybe(signed);
                        let right = right_raw.as_signed_maybe(signed);

                        // build the final expression
                        let op_str = match op {
                            IrIntCompareOp::Eq => "==",
                            IrIntCompareOp::Neq => "!=",
                            IrIntCompareOp::Lt => "<",
                            IrIntCompareOp::Lte => "<=",
                            IrIntCompareOp::Gt => ">",
                            IrIntCompareOp::Gte => ">=",
                        };
                        Evaluated::String(format!("({left} {op_str} {right})"))
                    }

                    IrExpressionLarge::TupleLiteral(elements) => {
                        // verilog does not care much about types, this is just a concatenation
                        //  (assuming all sub-expression have the right width, which they should)
                        // the order is flipped, verilog concatenation is high->low index
                        // TODO this is probably incorrect in general, we need to store the tuple in a variable first
                        let mut g = String::new();
                        let mut any_prev = false;
                        swrite!(g, "{{");
                        for elem in elements.iter().rev() {
                            let elem = match self.lower_expression(span, elem)? {
                                Ok(elem) => elem,
                                Err(ZeroWidth) => continue,
                            };

                            if any_prev {
                                swrite!(g, ", ");
                            }
                            swrite!(g, "{elem}");
                            any_prev = true;
                        }
                        swrite!(g, "}}");
                        Evaluated::String(g)
                    }
                    IrExpressionLarge::ArrayLiteral(_inner_ty, _len, elements) => {
                        // verilog does not care much about types, this is just a concatenation
                        //  (assuming all sub-expression have the right width, which they should)
                        // the order is flipped, verilog concatenation is high->low index
                        // TODO use repeat operator if array elements are repeated
                        let mut g = String::new();
                        let mut any_prev = false;
                        swrite!(g, "{{");
                        for elem in elements.iter().rev() {
                            let elem = match elem {
                                IrArrayLiteralElement::Spread(inner) => inner,
                                IrArrayLiteralElement::Single(inner) => inner,
                            };
                            let elem = match self.lower_expression(span, elem)? {
                                Ok(elem) => elem,
                                Err(ZeroWidth) => continue,
                            };

                            if any_prev {
                                swrite!(g, ", ");
                            }
                            swrite!(g, "{elem}");
                            any_prev = true;
                        }
                        swrite!(g, "}}");
                        Evaluated::String(g)
                    }

                    IrExpressionLarge::TupleIndex { base, index } => {
                        // TODO this is completely wrong
                        let base = match self.lower_expression(span, base)? {
                            Ok(base) => base,
                            Err(ZeroWidth) => return Ok(Err(ZeroWidth)),
                        };
                        Evaluated::String(format!("({base}[{index}])"))
                    }
                    IrExpressionLarge::ArrayIndex { base, index } => {
                        // TODO this is probably incorrect in general, we need to store the array in a variable first
                        // TODO we're incorrectly using array indices as bit indices here

                        let base_len =
                            unwrap_match!(base.ty(self.module, self.locals), IrType::Array(_, base_len) => base_len);
                        let index_range = ClosedIncRange {
                            start_inc: BigInt::ZERO,
                            end_inc: base_len - 1,
                        };
                        let index_range =
                            NonZeroWidthRange::new(index_range.as_ref()).unwrap_or(NonZeroWidthRange::ZERO_ONE);

                        let base = match self.lower_expression(span, base)? {
                            Ok(base) => base,
                            Err(ZeroWidth) => return Ok(Err(ZeroWidth)),
                        };
                        let index = self.lower_expression_int_expanded(span, index_range, index)?;

                        Evaluated::String(format!("({base}[{index}])"))
                    }
                    IrExpressionLarge::ArraySlice { base, start, len } => {
                        // TODO this is probably incorrect in general, we need to store the array in a variable first
                        // TODO we're incorrectly using array indices as bit indices here

                        let base_len =
                            unwrap_match!(base.ty(self.module, self.locals), IrType::Array(_, base_len) => base_len);
                        let start_range = ClosedIncRange {
                            start_inc: BigInt::ZERO,
                            end_inc: base_len.into(),
                        };
                        let start_range =
                            NonZeroWidthRange::new(start_range.as_ref()).unwrap_or(NonZeroWidthRange::ZERO_ONE);

                        let base = match self.lower_expression(span, base)? {
                            Ok(base) => base,
                            Err(ZeroWidth) => return Ok(Err(ZeroWidth)),
                        };
                        let start = self.lower_expression_int_expanded(span, start_range, start)?;
                        let len = lower_uint_str(len);

                        Evaluated::String(format!("({base}[{start}+:{len}])"))
                    }

                    IrExpressionLarge::ToBits(_ty, value) => {
                        // in verilog everything is just a bit vector, so we don't need to do anything
                        return self.lower_expression(span, value);
                    }
                    IrExpressionLarge::FromBits(_ty, value) => {
                        // in verilog everything is just a bit vector, so we don't need to do anything
                        return self.lower_expression(span, value);
                    }
                    IrExpressionLarge::ExpandIntRange(target, value) => {
                        let target =
                            NonZeroWidthRange::new(target.as_ref()).expect("already checked for zero-width earlier");
                        self.lower_expression_int_expanded(span, target, value)?
                    }
                    IrExpressionLarge::ConstrainIntRange(range, value) => {
                        // already handled through the result type
                        let _ = range;

                        // TODO add assertions? what exactly are the semantics of this operation?
                        let value = match self.lower_expression(span, value)? {
                            Ok(value) => value,
                            Err(ZeroWidth) => return Ok(Err(ZeroWidth)),
                        };

                        // store in temporary to force truncation
                        let tmp = self.new_temporary(span, result_ty_verilog)?;
                        swriteln!(self.f, "{indent}{tmp} = {value};");
                        Evaluated::Temporary(tmp)
                    }
                }
            }
        };
        Ok(Ok(eval))
    }

    fn lower_arithmetic_expression(
        &mut self,
        span: Span,
        result_range: NonZeroWidthRange,
        result_ty_verilog: VerilogType,
        op: IrIntArithmeticOp,
        left: &IrExpression,
        right: &IrExpression,
    ) -> DiagResult<Evaluated<'n>> {
        // TODO do strength reduction to common IR optimization pass?
        // TODO strength reduction (where possible):
        //    * mul -> shift when power of 2
        //    * div/mod by power of 2 -> bit slice
        //    * div by constant -> https://llvm.org/doxygen/structllvm_1_1SignedDivisionByConstantInfo.html#details
        //    * pow of two -> shift
        match op {
            IrIntArithmeticOp::Add => {
                self.lower_arithmetic_expression_simple(span, result_range, result_ty_verilog, "+", left, right)
            }
            IrIntArithmeticOp::Sub => {
                self.lower_arithmetic_expression_simple(span, result_range, result_ty_verilog, "-", left, right)
            }
            IrIntArithmeticOp::Mul => {
                self.lower_arithmetic_expression_simple(span, result_range, result_ty_verilog, "*", left, right)
            }
            IrIntArithmeticOp::Div => self.lower_arithmetic_expression_div_mod(
                span,
                result_range,
                result_ty_verilog,
                OperatorDivMod::Div,
                left,
                right,
            ),
            IrIntArithmeticOp::Mod => self.lower_arithmetic_expression_div_mod(
                span,
                result_range,
                result_ty_verilog,
                OperatorDivMod::Mod,
                left,
                right,
            ),
            IrIntArithmeticOp::Pow => {
                self.lower_arithmetic_expression_simple(span, result_range, result_ty_verilog, "**", left, right)
            }
        }
    }

    fn lower_arithmetic_expression_simple(
        &mut self,
        span: Span,
        result_range: NonZeroWidthRange,
        result_ty_verilog: VerilogType,
        op_str: &str,
        left: &IrExpression,
        right: &IrExpression,
    ) -> DiagResult<Evaluated<'n>> {
        // expand operands to a common range that contains all operands and the result
        let range_left = left.ty(self.module, self.locals).unwrap_int();
        let range_right = right.ty(self.module, self.locals).unwrap_int();
        let range_all = result_range
            .range()
            .union(range_left.as_ref())
            .union(range_right.as_ref());
        let range_all = NonZeroWidthRange::new(range_all).expect("result range is non-zero, so the union is too");

        // evaluate operands to that range
        let left = self.lower_expression_int_expanded(span, range_all, left)?;
        let right = self.lower_expression_int_expanded(span, range_all, right)?;

        // store result in a temporary to force truncation
        // TODO skip if no truncation is actually necessary
        let indent = self.indent;
        let res_tmp = self.new_temporary(span, result_ty_verilog)?;
        swriteln!(self.f, "{indent}{res_tmp} = {left} {op_str} {right};");

        Ok(Evaluated::Temporary(res_tmp))
    }

    fn lower_arithmetic_expression_div_mod(
        &mut self,
        span: Span,
        result_range: NonZeroWidthRange,
        result_ty_verilog: VerilogType,
        op: OperatorDivMod,
        a: &IrExpression,
        b: &IrExpression,
    ) -> DiagResult<Evaluated<'n>> {
        let indent = self.indent;
        let diags = self.diags;

        let range_a = a.ty(self.module, self.locals).unwrap_int();
        let range_b = b.ty(self.module, self.locals).unwrap_int();
        let range_all = result_range.range().union(range_a.as_ref()).union(range_b.as_ref());
        let range_all = NonZeroWidthRange::new(range_all).expect("result range is non-zero, so the union is too");

        // evaluate operands to that range
        let a_raw = self.lower_expression_int_expanded(span, range_all, a)?;
        let b_raw = self.lower_expression_int_expanded(span, range_all, b)?;

        // cast operands to signed if necessary
        let signed = range_all.range().start_inc < &BigInt::from(0);
        let a = a_raw.as_signed_maybe(signed);
        let b = b_raw.as_signed_maybe(signed);

        // make adjustments to match IR semantics (round down) instead of verilog (truncate towards zero)
        let a_is_neg = MaybeBool::is_negative(&a, range_a.as_ref());
        let b_is_neg = MaybeBool::is_negative(&b, range_b.as_ref());
        let signs_differ = MaybeBool::xor(&a_is_neg, &b_is_neg);
        let res_expr = match op {
            OperatorDivMod::Div => {
                let adj_one = b_is_neg.select("1", "-1");
                let adj = signs_differ.select(&format!("({b} + {adj_one})"), "0");
                let a_adj = format!("({a} - {adj})");
                format!("{a_adj} / {b};")
            }
            OperatorDivMod::Mod => {
                let tmp_mod = self.new_temporary(span, VerilogType::new_from_range(diags, span, range_all)?)?;
                swriteln!(self.f, "{indent}{tmp_mod} = {a} % {b};");
                let should_adjust =
                    MaybeBool::and(&MaybeBool::is_not_zero(&tmp_mod, result_range.range()), &signs_differ);
                should_adjust.select(&format!("{tmp_mod} + {b}"), &tmp_mod.to_string())
            }
        };

        // store in temporary to force truncation
        let res_tmp = self.new_temporary(span, result_ty_verilog)?;
        swriteln!(self.f, "{indent}{res_tmp} = {res_expr};");

        Ok(Evaluated::Temporary(res_tmp))
    }

    // TODO doc
    fn lower_expression_non_zero_width(
        &mut self,
        span: Span,
        expr: &IrExpression,
        reason: &str,
    ) -> DiagResult<Evaluated<'n>> {
        self.lower_expression(span, expr)?.map_err(|_: ZeroWidth| {
            self.diags
                .report_internal_error(span, format!("{reason} cannot be zero-width"))
        })
    }

    fn lower_expression_int_expanded(
        &mut self,
        span: Span,
        range: NonZeroWidthRange,
        value: &IrExpression,
    ) -> DiagResult<Evaluated<'n>> {
        // skip separate expansion step for integer constants
        if let IrExpression::Int(value) = value {
            return Ok(lower_int_constant(range, value));
        }

        // general case
        let value_ty = value.ty(self.module, self.locals);
        let value_ty = value_ty.unwrap_int();

        let value = match self.lower_expression(span, value)? {
            Ok(value) => lower_expand_int_range(range.range(), value_ty.as_ref(), value),
            Err(ZeroWidth) => {
                let value = value_ty.as_single().unwrap();
                lower_int_constant(range, value)
            }
        };

        Ok(value)
    }

    fn lower_assign_target(
        &mut self,
        span: Span,
        target: &IrAssignmentTarget,
    ) -> DiagResult<Result<Evaluated<'n>, Either<ZeroWidth, NotRead>>> {
        // TODO this is probably wrong, we might need intermediate variables for the base and after each step
        let &IrAssignmentTarget { base, ref steps } = target;

        let base_ty = base.as_expression().ty(self.module, self.locals);
        let base = try_inner!(self.name_map.map_signal_or_var(base));

        // early exit to avoid string allocation
        if steps.is_empty() {
            return Ok(Ok(Evaluated::Name(base)));
        }

        let mut g = String::new();
        swrite!(g, "{base}");

        // TODO both of these are wrong, we're not taking element type sizes into account
        // TODO this entire thing should just be flattened to a single slice
        // TODO handle non-consecutive slicing, probably similar to how the C++ backend does it
        let mut next_ty = base_ty;
        for step in steps {
            let (curr_inner, curr_len) =
                unwrap_match!(next_ty, IrType::Array(curr_inner, curr_len) => (curr_inner, curr_len));
            next_ty = *curr_inner;

            match step {
                IrTargetStep::ArrayIndex(index) => {
                    let index_range = ClosedIncRange {
                        start_inc: BigInt::ZERO,
                        end_inc: curr_len - 1,
                    };
                    let index_range =
                        NonZeroWidthRange::new(index_range.as_ref()).unwrap_or(NonZeroWidthRange::ZERO_ONE);
                    let index = self.lower_expression_int_expanded(span, index_range, index)?;
                    swrite!(g, "[{index}]");
                }
                IrTargetStep::ArraySlice(start, len) => {
                    let start_range = ClosedIncRange {
                        start_inc: BigInt::ZERO,
                        end_inc: curr_len.into(),
                    };
                    let start_range =
                        NonZeroWidthRange::new(start_range.as_ref()).unwrap_or(NonZeroWidthRange::ZERO_ONE);

                    let start = self.lower_expression_int_expanded(span, start_range, start)?;
                    let len = lower_uint_str(len);

                    swrite!(g, "[{start}+:{len}]");
                }
            }
        }

        Ok(Ok(Evaluated::String(g)))
    }

    fn new_temporary(&mut self, span: Span, ty: VerilogType) -> DiagResult<Temporary<'n>> {
        let name = self.name_scope.make_unique_str(self.diags, span, "tmp", true)?;

        let info = TemporaryInfo { name, ty };
        let info = self.temporaries.push(info);

        Ok(Temporary(&info.name))
    }
}

fn lower_expand_int_range<'n>(
    target_ty: ClosedIncRange<&BigInt>,
    value_ty: ClosedIncRange<&BigInt>,
    value: Evaluated<'n>,
) -> Evaluated<'n> {
    // cast the value to the right signedness
    //   and add a literal of the right sign and size to force expansion
    let target_repr = IntRepresentation::for_range(target_ty);
    let target_size = target_repr.size_bits();

    let value_repr = IntRepresentation::for_range(value_ty);
    let value_size = value_repr.size_bits();

    if target_size == value_size {
        return value;
    }

    // sign/zero-extend based on the signedness of the original value
    match value_repr.signed() {
        Signed::Signed => {
            // the result will be signed too, keep it that way for now to reduce unnecessary casts
            let value_signed = value.as_signed();
            Evaluated::SignedString(format!("({target_size}'sd0 + {value_signed})"))
        }
        Signed::Unsigned => Evaluated::String(format!("({target_size}'d0 + {value})")),
    }
}

fn lower_int_constant(ty: NonZeroWidthRange, x: &BigInt) -> Evaluated<'static> {
    let ty = ty.range();
    assert!(ty.contains(&x), "Trying to emit constant {x:?} encoded as range {ty:?}");

    let repr = IntRepresentation::for_range(ty);
    let bits = repr.size_bits();
    assert_ne!(bits, 0);

    match repr.signed() {
        Signed::Unsigned => Evaluated::String(format!("{bits}'d{x}")),
        Signed::Signed => {
            let s = match x.sign() {
                Sign::Negative => {
                    // Verilog does not actually have negative literals, it's just the unary not operator.
                    // For the most negative value this fails though, the absolute value already underflows,
                    //   so we should skip the preceding negative sign in that case.
                    let prefix_sign = if x == &repr.range().start_inc { "" } else { "-" };
                    format!("{prefix_sign}{bits}'sd{}", x.abs())
                }
                Sign::Zero | Sign::Positive => format!("{bits}'sd{x}"),
            };
            Evaluated::SignedString(s)
        }
    }
}

fn lower_uint_str(x: &BigUint) -> String {
    // TODO zero-width literals are probably not allowed in verilog
    // TODO double-check integer bit-width promotion rules
    // TODO avoid clone
    // TODO remove this redundant function and always use lower_int_constant
    let repr = IntRepresentation::for_single(&x.into());
    format!("{}'d{}", repr.size_bits(), x)
}

#[derive(Debug)]
struct EdgeString<'n> {
    edge: &'static str,
    if_prefix: &'static str,
    signal: &'n LoweredName,
}

fn lower_edge<'n>(
    diags: &Diagnostics,
    name_map: NameMap<'n>,
    expr: Spanned<Polarized<IrSignal>>,
) -> DiagResult<EdgeString<'n>> {
    let Polarized { inverted, signal } = expr.inner;
    let (edge, if_prefix) = match inverted {
        false => ("posedge", ""),
        true => ("negedge", "!"),
    };

    match name_map.map_signal(signal) {
        Ok(signal) => Ok(EdgeString {
            edge,
            if_prefix,
            signal,
        }),
        Err(ZeroWidth) => Err(diags.report_internal_error(expr.span, "zero-width edge signal")),
    }
}

mod non_zero_width {
    use crate::back::lower_verilog::ZeroWidth;
    use crate::front::types::ClosedIncRange;
    use crate::util::big_int::BigInt;
    use crate::util::int::IntRepresentation;

    #[derive(Debug, Copy, Clone)]
    pub struct NonZeroWidthRange<'a>(ClosedIncRange<&'a BigInt>);

    impl<'a> NonZeroWidthRange<'a> {
        pub const ZERO_ONE: NonZeroWidthRange<'static> = NonZeroWidthRange(ClosedIncRange {
            start_inc: &BigInt::ZERO,
            end_inc: &BigInt::ONE,
        });

        pub fn new(range: ClosedIncRange<&'a BigInt>) -> Result<Self, ZeroWidth> {
            let repr = IntRepresentation::for_range(range);
            if repr.size_bits() == 0 {
                Err(ZeroWidth)
            } else {
                Ok(NonZeroWidthRange(range))
            }
        }

        pub fn range(&self) -> ClosedIncRange<&'a BigInt> {
            self.0
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum OperatorDivMod {
    Div,
    Mod,
}

// TODO is this distinction actually useful?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum VerilogType {
    /// Single bit value.
    Bit,
    /// Potentially multi-bit values. This can still happen to have width 1,
    /// The difference with [SingleBit] in that case is that this should still be represented as an array.
    Array(NonZeroU32),
}

impl VerilogType {
    pub fn width(self) -> NonZeroU32 {
        match self {
            VerilogType::Bit => NonZeroU32::new(1).unwrap(),
            VerilogType::Array(w) => w,
        }
    }

    pub fn new_array(diags: &Diagnostics, span: Span, w: BigUint) -> DiagResult<Result<VerilogType, ZeroWidth>> {
        let w = diag_big_int_to_u32(diags, span, &w.into(), "array width too large")?;
        match NonZeroU32::new(w) {
            None => Ok(Err(ZeroWidth)),
            Some(w) => Ok(Ok(VerilogType::Array(w))),
        }
    }

    pub fn new_from_ir(diags: &Diagnostics, span: Span, ty: &IrType) -> DiagResult<Result<VerilogType, ZeroWidth>> {
        match ty {
            IrType::Bool => Ok(Ok(VerilogType::Bit)),
            IrType::Int(_) | IrType::Tuple(_) | IrType::Array(_, _) => Self::new_array(diags, span, ty.size_bits()),
        }
    }

    pub fn new_from_range(diags: &Diagnostics, span: Span, range: NonZeroWidthRange) -> DiagResult<VerilogType> {
        let repr = IntRepresentation::for_range(range.range());
        Ok(Self::new_array(diags, span, BigUint::from(repr.size_bits()))?.expect("range should be non-zero-width"))
    }

    pub fn to_prefix(self) -> impl Display {
        struct D(VerilogType);
        impl Display for D {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    VerilogType::Bit => Ok(()),
                    VerilogType::Array(width) => write!(f, "[{}:0] ", width.get() - 1),
                }
            }
        }

        D(self)
    }
}

#[derive(Debug, Copy, Clone)]
struct NewlineGenerator {
    any_prev_group: bool,
    any_prev_item: bool,
}

impl NewlineGenerator {
    pub fn new() -> Self {
        Self {
            any_prev_group: false,
            any_prev_item: false,
        }
    }

    pub fn start_group(&mut self) {
        self.any_prev_group |= self.any_prev_item;
        self.any_prev_item = false;
    }

    pub fn start_item(&mut self, f: &mut String) {
        if self.any_prev_group && !self.any_prev_item {
            swriteln!(f);
        }
        self.any_prev_item = true;
    }

    pub fn start_group_and_item(&mut self, f: &mut String) {
        self.start_group();
        self.start_item(f);
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
static ref VERILOG_KEYWORDS: IndexSet < & 'static str > = {
include_str ! ("verilog_keywords.txt").lines().map(str::trim).filter( |line | ! line.is_empty()).collect()
};
}

fn maybe_id_as_ref(id: &Spanned<Option<String>>) -> Spanned<Option<&str>> {
    id.as_ref().map_inner(|s| s.as_ref().map(String::as_ref))
}

enum MaybeBool {
    Const(bool),
    Runtime(String),
}

impl MaybeBool {
    fn is_not_zero(v: &impl Display, r: ClosedIncRange<&BigInt>) -> MaybeBool {
        let can_be_zero = r.contains(&&BigInt::ZERO);
        let can_be_non_zero = r != ClosedIncRange::single(&BigInt::ZERO);

        if can_be_zero && can_be_non_zero {
            MaybeBool::Runtime(format!("({v} != 0)"))
        } else if can_be_zero {
            MaybeBool::Const(false)
        } else {
            MaybeBool::Const(true)
        }
    }

    fn is_negative(v: &impl Display, r: ClosedIncRange<&BigInt>) -> MaybeBool {
        let can_be_neg = r.start_inc < &BigInt::ZERO;
        let can_be_non_neg = r.end_inc >= &BigInt::ZERO;
        if can_be_neg && can_be_non_neg {
            MaybeBool::Runtime(format!("({v} < 0)"))
        } else if can_be_neg {
            MaybeBool::Const(true)
        } else {
            MaybeBool::Const(false)
        }
    }

    fn select(&self, t: &str, f: &str) -> String {
        match self {
            MaybeBool::Const(true) => t.to_string(),
            MaybeBool::Const(false) => f.to_string(),
            MaybeBool::Runtime(c) => format!("({c} ? {t} : {f})"),
        }
    }

    fn xor(&self, other: &Self) -> MaybeBool {
        match (self, other) {
            (&MaybeBool::Const(a), &MaybeBool::Const(b)) => MaybeBool::Const(a ^ b),
            (MaybeBool::Const(false), MaybeBool::Runtime(other))
            | (MaybeBool::Runtime(other), MaybeBool::Const(false)) => MaybeBool::Runtime(other.to_string()),
            (MaybeBool::Const(true), MaybeBool::Runtime(other))
            | (MaybeBool::Runtime(other), MaybeBool::Const(true)) => MaybeBool::Runtime(format!("(!{})", other)),
            (MaybeBool::Runtime(a), MaybeBool::Runtime(b)) => MaybeBool::Runtime(format!("({} ^ {})", a, b)),
        }
    }

    fn and(&self, other: &Self) -> MaybeBool {
        match (self, other) {
            (&MaybeBool::Const(a), &MaybeBool::Const(b)) => MaybeBool::Const(a & b),
            (MaybeBool::Const(true), MaybeBool::Runtime(other))
            | (MaybeBool::Runtime(other), MaybeBool::Const(true)) => MaybeBool::Runtime(other.to_string()),
            (MaybeBool::Const(false), MaybeBool::Runtime(_)) | (MaybeBool::Runtime(_), MaybeBool::Const(false)) => {
                MaybeBool::Const(false)
            }
            (MaybeBool::Runtime(a), MaybeBool::Runtime(b)) => MaybeBool::Runtime(format!("({} && {})", a, b)),
        }
    }
}

fn unwrap_zero_width<T>(r: Result<T, ZeroWidth>) -> T {
    r.expect("zero width should have already been checked")
}

fn escape_verilog_str(s: &str) -> String {
    let mut f = String::new();

    for c in s.chars() {
        match c {
            '\n' => swrite!(f, "\\n"),
            '\t' => swrite!(f, "\\t"),
            '\\' => swrite!(f, "\\\\"),
            '"' => swrite!(f, "\\\""),
            '%' => swrite!(f, "%%"),
            _ => {
                if c.is_ascii() {
                    f.push(c);
                } else {
                    // verilog does not support unicode strings, so replace non-ascii with '?'
                    f.push('?');
                }
            }
        }
    }

    f
}
