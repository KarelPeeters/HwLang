use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::signal::Polarized;
use crate::mid::graph::ir_modules_topological_sort;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrBoolBinaryOp, IrClockedProcess,
    IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrIntArithmeticOp,
    IrIntCompareOp, IrIntegerRadix, IrLargeArena, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo,
    IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrSignal,
    IrSignalOrVariable, IrStatement, IrString, IrStringSubstitution, IrTargetStep, IrType, IrVariable, IrVariableInfo,
    IrVariables, IrWire, IrWireInfo, IrWireOrPort, ValueAccess,
};
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::syntax::pos::{Span, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint, Sign};
use crate::util::data::{GrowVec, IndexMapExt};
use crate::util::int::{IntRepresentation, Signed};
use crate::util::range::{ClosedNonEmptyRange, ClosedRange};
use crate::util::{Indent, ResultExt, separator_non_trailing};
use crate::{throw, try_inner};
use hwl_util::{swrite, swriteln};
use indexmap::{IndexMap, IndexSet};
use itertools::{Either, enumerate};
use lazy_static::lazy_static;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;

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
// TODO should we always shadow output ports with intermediate signals so we can read back from them,
//   even in old verilog versions?
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

    let modules = ir_modules_topological_sort(modules, [top_module]);
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
    // track the next suffix to try for each base string to avoid O(n**2) behavior
    next_suffix_to_try: IndexMap<String, u64>,
}

impl<'p> LoweredNameScope<'p> {
    pub fn new_root(used: IndexSet<String>) -> Self {
        Self {
            parent: None,
            local_used: used,
            next_suffix_to_try: IndexMap::new(),
        }
    }

    pub fn new_child(&'p self) -> Self {
        Self {
            parent: Some(self),
            local_used: IndexSet::new(),
            next_suffix_to_try: IndexMap::new(),
        }
    }

    pub fn exact_for_new_id(&mut self, diags: &Diagnostics, span: Span, id: &str) -> DiagResult<LoweredName> {
        check_identifier_valid(diags, Spanned { span, inner: id })?;
        if self.is_used(id) {
            throw!(diags.report_error_internal(span, format!("lowered identifier `{id}` already used its scope")))
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
        string_raw: &str,
        force_index: bool,
    ) -> DiagResult<LoweredName> {
        let string_valid = make_identifier_valid(string_raw);

        if !force_index && !self.is_used(&string_valid) {
            self.local_used.insert(string_valid.clone());
            return Ok(LoweredName(string_valid));
        }

        let start_suffix = self.next_suffix_to_try.get(&string_valid).copied().unwrap_or(0);

        let valid_len = string_valid.len();
        let mut buffer = string_valid;
        buffer.push('_');

        for i in start_suffix.. {
            swrite!(buffer, "{i}");

            if !self.is_used(&buffer) {
                let prefix = buffer[..valid_len].to_owned();
                self.next_suffix_to_try.insert(prefix, i + 1);

                self.local_used.insert(buffer.clone());
                return Ok(LoweredName(buffer));
            }

            buffer.truncate(valid_len + 1);
        }

        throw!(diags.report_error_internal(
            span,
            format!("failed to generate unique lowered identifier for `{string_raw}`")
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
//   then we can get rid of span
// TODO use __ for joining, not really here but for eg. interfaces
fn check_identifier_valid(diags: &Diagnostics, id: Spanned<&str>) -> DiagResult {
    let s = id.inner;

    if s.is_empty() {
        throw!(diags.report_error_simple(
            "invalid verilog identifier: identifier cannot be empty",
            id.span,
            "identifier used here"
        ))
    }
    let first = s.chars().next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        throw!(diags.report_error_simple(
            "invalid verilog identifier: first character must be alphabetic or underscore",
            id.span,
            "identifier used here"
        ))
    }
    for c in s.chars() {
        if !(c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '$' || c == '_') {
            throw!(diags.report_error_simple(
                format!("invalid verilog identifier: character `{c}` not allowed in identifier"),
                id.span,
                "identifier used here"
            ))
        }
    }

    Ok(())
}

fn make_identifier_valid(id: &str) -> String {
    if id.is_empty() {
        return "_".to_owned();
    }

    let mut id_clean = String::with_capacity(1 + id.len());

    for c in id.chars() {
        // first character has stricter rules, prepend _ if needed
        if id_clean.is_empty() {
            if !(c.is_ascii_alphabetic() || c == '_') {
                id_clean.push('_');
            }
        }

        // other characters can be digits too
        //   technically they can be $ too, but that's sketchy
        let c_clean = if c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '_' {
            c
        } else {
            // TODO preserve uniqueness
            '_'
        };
        id_clean.push(c_clean);
    }

    id_clean
}

#[derive(Debug, Copy, Clone)]
struct ModuleNameMap<'n> {
    ports: &'n IndexMap<IrPort, Result<LoweredName, ZeroWidth>>,
    wires: &'n IndexMap<IrWire, Result<LoweredName, ZeroWidth>>,
    regs: &'n IndexMap<IrRegister, Result<LoweredName, ZeroWidth>>,
}

#[derive(Debug, Copy, Clone)]
struct ProcessNameMap<'n> {
    module_name_map: ModuleNameMap<'n>,
    regs_shadowed: &'n IndexMap<IrRegister, LoweredName>,
    variables: &'n IndexMap<IrVariable, Result<LoweredName, ZeroWidth>>,
}

impl<'n> ModuleNameMap<'n> {
    pub fn map_signal(&self, s: IrSignal) -> Result<&'n LoweredName, ZeroWidth> {
        match s {
            IrSignal::Port(port) => self.ports.get(&port).unwrap().as_ref_ok(),
            IrSignal::Wire(wire) => self.wires.get(&wire).unwrap().as_ref_ok(),
            IrSignal::Register(reg) => self.regs.get(&reg).unwrap().as_ref_ok(),
        }
    }
}

impl<'n> ProcessNameMap<'n> {
    pub fn read_signal(&self, signal: IrSignal) -> Result<&'n LoweredName, ZeroWidth> {
        match signal {
            IrSignal::Port(_) | IrSignal::Wire(_) => self.module_name_map.map_signal(signal),
            IrSignal::Register(reg) => match self.regs_shadowed.get(&reg) {
                Some(name) => Ok(name),
                None => self.module_name_map.map_signal(signal),
            },
        }
    }

    pub fn map_var(&self, v: IrVariable) -> Result<&'n LoweredName, ZeroWidth> {
        self.variables.get(&v).unwrap().as_ref_ok()
    }
}

// Processes with empty sensitivity lists never run, not even at startup
// to avoid this, we make all combinatorial processes sensitive to a dummy signal.
// In many cases this dummy signal could be skipped, but that's extra unnecessary complexity.
const SENSITIVITY_DUMMY_NAME_SIGNAL: &str = "dummy";
const SENSITIVITY_DUMMY_NAME_VAR: &str = "dummy_var";

fn lower_module(ctx: &mut LowerContext, module: IrModule) -> DiagResult<LoweredModule> {
    let diags = ctx.diags;
    assert!(!ctx.module_map.contains_key(&module));

    let module_info = &ctx.modules[module];
    let IrModuleInfo {
        ports,
        large: _,
        registers,
        wires,
        children: _,
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
    let signals_driven_by_instances = collect_signals_driven_by_instances(module_info);

    swrite!(f, "module {}(", module_name);
    let port_name_map = lower_module_ports(
        diags,
        &signals_driven_by_instances,
        ports,
        &mut module_name_scope,
        &mut f,
    )?;
    swriteln!(f, ");");

    let mut newline_module = NewlineGenerator::new();
    let (reg_name_map, wire_name_map) = lower_module_signals(
        diags,
        &mut module_name_scope,
        &signals_driven_by_instances,
        registers,
        wires,
        &mut newline_module,
        &mut f,
    )?;
    let module_names = ModuleNameMap {
        ports: &port_name_map,
        wires: &wire_name_map,
        regs: &reg_name_map,
    };

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
        module_info,
        &mut module_name_scope,
        &dummy_name,
        module_names,
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
    signals_driven_by_child_instances: &IndexSet<IrWireOrPort>,
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

        let (is_non_zero_width, ty_prefix) = match port_ty {
            Ok(port_ty) => (true, Either::Left(port_ty.prefix())),
            Err(ZeroWidth) => (false, Either::Right("[empty] ")),
        };

        let dir_str = match direction {
            PortDirection::Input => "input",
            PortDirection::Output => {
                if signals_driven_by_child_instances.contains(&IrWireOrPort::Port(port)) {
                    "output"
                } else {
                    "output reg"
                }
            }
        };

        if is_non_zero_width {
            last_actual_port_index = Some(port_index);
        }
        port_lines.push((
            is_non_zero_width,
            format!("{dir_str} {ty_prefix}{lower_name}"),
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

// TODO Handle signals partially driven by different drivers? Or will that not be supported?
fn collect_signals_driven_by_instances(module: &IrModuleInfo) -> IndexSet<IrWireOrPort> {
    let mut signals = IndexSet::new();

    let mut visit_connections = |connections: &Vec<Spanned<IrPortConnection>>| {
        for connection in connections {
            match connection.inner {
                IrPortConnection::Input(_) => {}
                IrPortConnection::Output(signal) => {
                    if let Some(signal) = signal {
                        signals.insert(signal);
                    }
                }
            }
        }
    };

    for child in &module.children {
        match &child.inner {
            IrModuleChild::ClockedProcess(_) => {}
            IrModuleChild::CombinatorialProcess(_) => {}
            IrModuleChild::ModuleInternalInstance(inst) => visit_connections(&inst.port_connections),
            IrModuleChild::ModuleExternalInstance(inst) => visit_connections(&inst.port_connections),
        }
    }

    signals
}

fn lower_module_signals(
    diags: &Diagnostics,
    module_name_scope: &mut LoweredNameScope,
    signals_driven_by_child_instances: &IndexSet<IrWireOrPort>,
    registers: &Arena<IrRegister, IrRegisterInfo>,
    wires: &Arena<IrWire, IrWireInfo>,
    newline: &mut NewlineGenerator,
    f: &mut String,
) -> DiagResult<(
    IndexMap<IrRegister, Result<LoweredName, ZeroWidth>>,
    IndexMap<IrWire, Result<LoweredName, ZeroWidth>>,
)> {
    let mut lower_signal = |verilog_kind,
                            signal_kind,
                            ty,
                            debug_info_id: &Spanned<Option<String>>,
                            debug_info_ty: &String,
                            debug_info_domain,
                            newline: &mut NewlineGenerator,
                            f: &mut String| {
        // skip zero-width signals
        let ty_verilog = match VerilogType::new_from_ir(diags, debug_info_id.span, ty)? {
            Ok(ty) => ty,
            Err(ZeroWidth) => return Ok(Err(ZeroWidth)),
        };
        let ty_prefix = ty_verilog.prefix();

        // pick a unique name
        let debug_info_id = maybe_id_as_ref(debug_info_id);
        let name = module_name_scope.make_unique_maybe_id(diags, debug_info_id)?;

        // figure out debug info
        let debug_name = debug_info_id.inner.unwrap_or("_");

        // actually write the verilog
        newline.start_item(f);
        swriteln!(
            f,
            "{I}{verilog_kind} {ty_prefix}{name}; // {signal_kind} {debug_name}: {debug_info_domain} {debug_info_ty}"
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

        let name = lower_signal(
            "reg",
            "reg",
            ty,
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
            newline,
            f,
        )?;
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

        let verilog_kind = if signals_driven_by_child_instances.contains(&IrWireOrPort::Wire(wire)) {
            "wire"
        } else {
            "reg"
        };

        let name = lower_signal(
            verilog_kind,
            "wire",
            ty,
            debug_info_id,
            debug_info_ty,
            debug_info_domain,
            newline,
            f,
        )?;
        wire_name_map.insert_first(wire, name);
    }

    Ok((reg_name_map, wire_name_map))
}

fn lower_module_statements(
    ctx: &mut LowerContext,
    module_info: &IrModuleInfo,
    module_name_scope: &mut LoweredNameScope,
    dummy_name: &LoweredName,
    module_name_map: ModuleNameMap,
    newline_module: &mut NewlineGenerator,
    f: &mut String,
) -> DiagResult {
    let diags = ctx.diags;
    let large = &module_info.large;

    for (child_index, child) in enumerate(&module_info.children) {
        newline_module.start_group_and_item(f);

        match &child.inner {
            IrModuleChild::CombinatorialProcess(process) => {
                let IrCombinatorialProcess { locals, block } = process;

                swriteln!(f, "{I}always @(*) begin: comb_{child_index}");
                let indent = Indent::new(2);

                let mut name_scope = module_name_scope.new_child();
                let dummy_name_internal =
                    name_scope.make_unique_str(diags, child.span, SENSITIVITY_DUMMY_NAME_VAR, false)?;
                swriteln!(f, "{indent}reg {dummy_name_internal};");

                let mut newline_process = NewlineGenerator::new();
                let variables = declare_locals(diags, f, &mut newline_process, &mut name_scope, locals)?;

                let temporaries = GrowVec::new();
                let temporaries_offset = f.len();

                newline_process.start_item(f);
                swriteln!(f, "{indent}{dummy_name_internal} = {dummy_name};");

                // combinatorial processes don't have register shadowing, they can't write to any
                let process_name_map = ProcessNameMap {
                    module_name_map,
                    regs_shadowed: &IndexMap::default(),
                    variables: &variables,
                };
                newline_process.start_group();
                let mut ctx = LowerBlockContext {
                    diags,
                    large,
                    module: module_info,
                    locals,
                    name_map: process_name_map,
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
                let clock_edge = lower_edge(diags, module_name_map, clock_signal)?;
                let async_reset = async_reset
                    .as_ref()
                    .map(|info| {
                        let reset_edge = lower_edge(diags, module_name_map, info.signal)?;
                        Ok((reset_edge, info))
                    })
                    .transpose()?;

                match &async_reset {
                    Some((reset_edge, _)) => swriteln!(
                        f,
                        "{I}always @({} {}, {} {}) begin: clocked_{child_index}",
                        clock_edge.edge,
                        clock_edge.signal,
                        reset_edge.edge,
                        reset_edge.signal,
                    ),
                    None => swriteln!(
                        f,
                        "{I}always @({} {}) begin: clocked_{child_index}",
                        clock_edge.edge,
                        clock_edge.signal,
                    ),
                }

                let mut newline_process = NewlineGenerator::new();
                let mut name_scope = module_name_scope.new_child();

                // declare locals and shadow registers
                let shadow_regs = collect_shadow_signals(large, clock_block);

                let variables = declare_locals(diags, f, &mut newline_process, &mut name_scope, locals)?;
                let reg_name_map_shadowed = declare_shadow_registers(
                    diags,
                    f,
                    &mut newline_process,
                    &mut name_scope,
                    &module_info.registers,
                    &shadow_regs,
                )?;

                let temporaries = GrowVec::new();
                let temporaries_offset = f.len();

                // build new name map
                let process_name_map = ProcessNameMap {
                    module_name_map,
                    regs_shadowed: &reg_name_map_shadowed,
                    variables: &variables,
                };

                // async reset header if any
                let indent_clocked = match &async_reset {
                    None => Indent::new(2),
                    Some((reset_edge, reset_info)) => {
                        let IrAsyncResetInfo { signal: _, resets } = reset_info;

                        newline_process.start_group_and_item(f);
                        swriteln!(f, "{I}{I}if ({}{}) begin", reset_edge.if_prefix, reset_edge.signal);
                        let indent_inner = Indent::new(3);

                        let mut ctx_reset = LowerBlockContext {
                            diags,
                            large,
                            module: module_info,
                            locals,
                            name_map: process_name_map,
                            name_scope: &mut name_scope,
                            temporaries: &temporaries,
                            indent: indent_inner,
                            newline: &mut newline_process,
                            f,
                        };

                        // TODO should resets use blocking or non-blocking assignments?
                        for reset in resets {
                            let &(reg, ref value) = &reset.inner;

                            // use non-shadowed name
                            let reg_name_raw = match module_name_map.map_signal(IrSignal::Register(reg)) {
                                Ok(reg_name) => reg_name,
                                Err(ZeroWidth) => continue,
                            };

                            let value = unwrap_zero_width(ctx_reset.lower_expression(reset.span, value)?);
                            swriteln!(ctx_reset.f, "{indent_inner}{reg_name_raw} = {value};");
                        }

                        swriteln!(f, "{I}{I}end else begin");
                        indent_inner
                    }
                };

                // populate shadow registers
                for (&reg, reg_name_shadow) in &reg_name_map_shadowed {
                    if shadow_regs.contains(&reg) {
                        newline_process.start_item(f);
                        let reg_name_raw = match module_name_map.map_signal(IrSignal::Register(reg)) {
                            Ok(orig_name) => orig_name,
                            Err(ZeroWidth) => continue,
                        };
                        swriteln!(f, "{indent_clocked}{reg_name_shadow} = {reg_name_raw};");
                    }
                }

                // lower body itself
                newline_process.start_group();
                let mut ctx = LowerBlockContext {
                    diags,
                    large,
                    module: module_info,
                    locals,
                    name_map: process_name_map,
                    name_scope: &mut name_scope,
                    temporaries: &temporaries,
                    indent: indent_clocked,
                    newline: &mut newline_process,
                    f,
                };
                ctx.lower_block(clock_block)?;

                // go back and insert temporary declarations at the start of the process
                declare_temporaries(f, temporaries_offset, temporaries);

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

                let port_name = |port_index| {
                    // TODO avoid clone here
                    let (_port, name) = inner_module.ports.get_index(port_index).unwrap();
                    name.clone()
                };
                lower_port_connections(f, port_connections, module_name_map, port_name)?;
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

                let port_name = |port_index: usize| Ok(LoweredName(&port_names[port_index]));
                lower_port_connections(f, port_connections, module_name_map, port_name)?;
            }
        }
    }

    Ok(())
}

fn lower_port_connections<S: AsRef<str>>(
    f: &mut String,
    instance_connections: &Vec<Spanned<IrPortConnection>>,
    parent_name_map: ModuleNameMap,
    child_port_name: impl Fn(usize) -> Result<LoweredName<S>, ZeroWidth>,
) -> DiagResult {
    swrite!(f, "(");

    if instance_connections.is_empty() {
        swriteln!(f, ");");
        return Ok(());
    }
    swriteln!(f);

    let mut first = true;

    for (port_index, connection) in enumerate(instance_connections) {
        let port_name = match child_port_name(port_index) {
            Ok(port_name) => port_name,
            Err(ZeroWidth) => continue,
        };

        if !first {
            swriteln!(f, ",");
        }
        swrite!(f, "{I}{I}.{port_name}(");

        match connection.inner {
            IrPortConnection::Input(expr) => {
                let signal_name = unwrap_zero_width(parent_name_map.map_signal(expr.inner));
                swrite!(f, "{signal_name}");
            }
            IrPortConnection::Output(signal) => {
                match signal {
                    None => {
                        // write nothing, resulting in an empty `()`
                    }
                    Some(signal) => {
                        let signal_name = unwrap_zero_width(parent_name_map.map_signal(signal.as_signal()));
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
struct ZeroWidth;

/// Declare local variables.
/// We skip declaring variables that never read, any potential writes to them will also be skipped.
fn declare_locals(
    diags: &Diagnostics,
    f: &mut String,
    newline: &mut NewlineGenerator,
    name_scope: &mut LoweredNameScope,
    locals: &IrVariables,
) -> DiagResult<IndexMap<IrVariable, Result<LoweredName, ZeroWidth>>> {
    let mut result = IndexMap::new();

    for (variable, variable_info) in locals {
        let &IrVariableInfo {
            ref ty,
            debug_info_span,
            ref debug_info_id,
        } = variable_info;

        let ty_verilog = VerilogType::new_from_ir(diags, debug_info_span, ty)?;

        let name = match ty_verilog {
            Ok(ty_verilog) => {
                let debug_info_id = Spanned::new(debug_info_span, debug_info_id.as_ref().map(String::as_str));
                let name = name_scope.make_unique_maybe_id(diags, debug_info_id)?;

                newline.start_item(f);
                declare_local(f, ty_verilog, &name);
                Ok(name)
            }
            Err(ZeroWidth) => Err(ZeroWidth),
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
    shadow_regs: &IndexSet<IrRegister>,
) -> DiagResult<IndexMap<IrRegister, LoweredName>> {
    let mut shadowing_reg_name_map = IndexMap::new();

    for &reg in shadow_regs {
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
        let ty_prefix = ty_verilog.prefix();
        swriteln!(f, "{I}{I}reg {ty_prefix}{shadow_name};");

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
    let ty_prefix = ty.prefix();
    let ty_size_bits = ty.size_bits();
    swriteln!(f, "{I}{I}reg {ty_prefix}{name} = {ty_size_bits}'bx;");
}

#[derive(Debug, Copy, Clone, Default)]
struct AccessInfo {
    any_read: bool,
    any_write: bool,
}

/// Collect all registers that should be shadowed.
/// Shadowed registers get a local variable in the process that at all times stores a copy of the value in the signal.
/// Writes to the register write to both the signal and the shadow variable, reads read from the shadow variable.
///
/// This ensures we implement the correct signal write behavior:
/// Reads after writes in a process should read the new value, not the old one.
/// But clocked block register writes should use non-blocking `<=` assignments, which would have the wrong behavior.
///
/// We still implement write-through on every assignment instead of only once at the end of the process,
/// to keep the generated code similar to what users would write,
/// and to hopefully better pattern match synthesis tool optimizations like clock gating.
fn collect_shadow_signals(large: &IrLargeArena, block: &IrBlock) -> IndexSet<IrRegister> {
    let mut regs: IndexMap<IrRegister, AccessInfo> = IndexMap::new();

    block.visit_values_accessed(large, &mut |value, access| match value {
        IrSignalOrVariable::Signal(IrSignal::Register(reg)) => {
            let entry = regs.entry(reg).or_default();
            match access {
                ValueAccess::Read => entry.any_read = true,
                ValueAccess::Write => entry.any_write = true,
            }
        }
        _ => {}
    });

    // TODO avoid reallocation here
    regs.into_iter()
        .filter(|(_, access)| access.any_write && access.any_read)
        .map(|(s, _)| s)
        .collect()
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

    pub fn is_named(&self) -> bool {
        match self {
            Evaluated::Name(_) | Evaluated::Temporary(_) => true,
            Evaluated::String(_) | Evaluated::Str(_) | Evaluated::SignedString(_) => false,
        }
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

    name_map: ProcessNameMap<'n>,
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
                let source = match self.lower_expression(stmt.span, source)? {
                    Ok(source) => source,
                    Err(ZeroWidth) => return Ok(()),
                };
                self.lower_assignment(stmt.span, target, source)?;
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

                // we will need to form an inclusive range, which will give weird results for empty ranges
                if !range.is_empty() {
                    let ClosedRange { start, end } = range;
                    let end_inc = end - 1;

                    let index_ty = IrExpression::Variable(index).ty(self.module, self.locals).unwrap_int();

                    match NonZeroWidthRange::new(index_ty) {
                        Ok(index_ty) => {
                            let index = match self.name_map.map_var(index) {
                                Ok(index) => index,
                                Err(ZeroWidth) => {
                                    return Err(diags.report_error_internal(stmt.span, "index var zero-width"));
                                }
                            };

                            let start = lower_int_constant(&index_ty, start);
                            let end_inc = lower_int_constant(&index_ty, &end_inc);
                            swriteln!(
                                self.f,
                                "{indent}for({index} = {start}; {index} <= {end_inc}; {index} = {index} + 1) begin"
                            );
                            self.lower_block_indented(block)?;
                            swriteln!(self.f, "{indent}end");
                        }
                        Err(ZeroWidth) => {
                            // the loop would also run for zero iterations, so just skip it
                        }
                    }
                }
            }
            IrStatement::Print(string) => {
                self.lower_print(stmt.span, string)?;
            }
            IrStatement::AssertFailed => {
                // TODO if targeting SystemVerilog, use $fatal/$error
                swriteln!(self.f, "{indent}$finish;");
            }
        }
        Ok(())
    }

    fn lower_print(&mut self, span: Span, s: &IrString) -> DiagResult {
        let mut f_str = String::new();
        let mut f_args = String::new();

        for p in s {
            match p {
                StringPiece::Literal(p) => {
                    f_str.push_str(&escape_verilog_str(p));
                }
                StringPiece::Substitute(p) => match p {
                    IrStringSubstitution::Integer(p, radix) => {
                        // TODO how does signed bin/hex behave?
                        let signed = p.ty(self.module, self.locals).unwrap_int().as_ref().start.is_negative();
                        let p = self.lower_expression(span, p)?;

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

        let indent = self.indent;
        swriteln!(self.f, "{indent}$write(\"{f_str}\"{f_args});");

        Ok(())
    }

    fn lower_expression_as_named(
        &mut self,
        span: Span,
        expr: &IrExpression,
    ) -> DiagResult<Result<Evaluated<'n>, ZeroWidth>> {
        let eval = try_inner!(self.lower_expression(span, expr)?);

        if eval.is_named() {
            Ok(Ok(eval))
        } else {
            let ty = expr.ty(self.module, self.locals);
            let ty_verilog = try_inner!(VerilogType::new_from_ir(self.diags, span, &ty)?);

            let tmp = self.new_temporary(span, ty_verilog)?;
            let indent = self.indent;
            swriteln!(self.f, "{indent}{tmp} = {eval};");

            Ok(Ok(Evaluated::Temporary(tmp)))
        }
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
                let range = ClosedNonEmptyRange::single(x.clone());
                let range = NonZeroWidthRange::new(range).expect("already checked for zero-width earlier");
                lower_int_constant(&range, x)
            }

            &IrExpression::Signal(s) => Evaluated::Name(unwrap_zero_width(name_map.read_signal(s))),
            &IrExpression::Variable(v) => Evaluated::Name(unwrap_zero_width(name_map.map_var(v))),
            &IrExpression::Large(expr) => {
                match &self.large[expr] {
                    IrExpressionLarge::Undefined(_) => {
                        let size_bits = result_ty_verilog.size_bits();
                        Evaluated::String(format!("{}'bx", size_bits))
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
                        let result_range = NonZeroWidthRange::new(result_range.clone())
                            .expect("already checked for zero-width earlier");
                        self.lower_arithmetic_expression(span, &result_range, result_ty_verilog, op, left, right)?
                    }
                    IrExpressionLarge::IntCompare(op, left, right) => {
                        // find common range that contains all operands
                        let left_range = left.ty(self.module, self.locals).unwrap_int();
                        let right_range = right.ty(self.module, self.locals).unwrap_int();
                        let combined_range = left_range.union(right_range);
                        let combined_range =
                            NonZeroWidthRange::new(combined_range).unwrap_or(NonZeroWidthRange::ZERO_ONE);

                        // lower both operands to that range
                        let left_raw = self.lower_expression_int_expanded(span, &combined_range, left)?;
                        let right_raw = self.lower_expression_int_expanded(span, &combined_range, right)?;

                        let signed = combined_range.range().start.is_negative();
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
                    &IrExpressionLarge::TupleIndex { ref base, index } => {
                        let ty = base.ty(self.module, self.locals).unwrap_tuple();
                        let start_bits = ty[..index].iter().map(IrType::size_bits).sum::<BigUint>();
                        let size_bits = ty[index].size_bits();

                        let base = try_inner!(self.lower_expression_as_named(span, base)?);

                        let mut g = String::new();
                        swrite!(g, "({base}[{start_bits}");
                        if size_bits != BigUint::ONE {
                            swrite!(g, " +: {size_bits}");
                        }
                        swrite!(g, "])");
                        Evaluated::String(g)
                    }
                    IrExpressionLarge::ArrayIndex { base, index } => {
                        // TODO constant fold if index is a constant?
                        // TODO expose the extra knowledge we have about integer ranges to verilog?
                        let base_ty = base.ty(self.module, self.locals);
                        let bit_range = bit_index_range(&base_ty.size_bits());
                        let (element_ty, _) = base_ty.unwrap_array();
                        let element_size_bits = element_ty.size_bits();

                        let base = try_inner!(self.lower_expression_as_named(span, base)?);
                        let index = self.lower_expression_int_expanded(span, &bit_range, index)?;

                        let mut g = String::new();
                        swrite!(g, "({base}[{index}");
                        if element_size_bits != BigUint::ONE {
                            swrite!(g, " * {element_size_bits} +: {element_size_bits}");
                        }
                        swrite!(g, "])");
                        Evaluated::String(g)
                    }
                    IrExpressionLarge::ArraySlice { base, start, len } => {
                        // TODO constant fold if index is a constant?
                        // TODO expose the extra knowledge we have about integer ranges to verilog?
                        let base_ty = base.ty(self.module, self.locals);
                        let bit_range = bit_index_range(&base_ty.size_bits());
                        let (element_ty, _) = base_ty.unwrap_array();
                        let element_size_bits = element_ty.size_bits();
                        let len_bits = len * &element_size_bits;

                        let base = try_inner!(self.lower_expression_as_named(span, base)?);
                        let index = self.lower_expression_int_expanded(span, &bit_range, start)?;

                        let mut g = String::new();
                        swrite!(g, "({base}[{index}");
                        if element_size_bits != BigUint::ONE {
                            swrite!(g, "* {element_size_bits}")
                        };
                        if len_bits != BigUint::ONE {
                            swrite!(g, " +: {len_bits}");
                        }
                        swrite!(g, "])");
                        Evaluated::String(g)
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
                            NonZeroWidthRange::new(target.clone()).expect("already checked for zero-width earlier");
                        self.lower_expression_int_expanded(span, &target, value)?
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
        result_range: &NonZeroWidthRange,
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
        //    * >, >=, ... -> ==, != if possible due to known ranges
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
        result_range: &NonZeroWidthRange,
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
            .as_ref()
            .union(range_left.as_ref())
            .union(range_right.as_ref())
            .cloned();
        let range_all = NonZeroWidthRange::new(range_all).expect("result range is non-zero, so the union is too");

        // evaluate operands to that range
        let left = self.lower_expression_int_expanded(span, &range_all, left)?;
        let right = self.lower_expression_int_expanded(span, &range_all, right)?;

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
        result_range: &NonZeroWidthRange,
        result_ty_verilog: VerilogType,
        op: OperatorDivMod,
        a: &IrExpression,
        b: &IrExpression,
    ) -> DiagResult<Evaluated<'n>> {
        let indent = self.indent;
        let diags = self.diags;

        let range_a = a.ty(self.module, self.locals).unwrap_int();
        let range_b = b.ty(self.module, self.locals).unwrap_int();
        let range_all = result_range
            .range()
            .as_ref()
            .union(range_a.as_ref())
            .union(range_b.as_ref())
            .cloned();
        let range_all = NonZeroWidthRange::new(range_all).expect("result range is non-zero, so the union is too");

        // evaluate operands to that range
        let a_raw = self.lower_expression_int_expanded(span, &range_all, a)?;
        let b_raw = self.lower_expression_int_expanded(span, &range_all, b)?;

        // cast operands to signed if necessary
        let signed = range_all.range().start.is_negative();
        let a = a_raw.as_signed_maybe(signed);
        let b = b_raw.as_signed_maybe(signed);

        // make adjustments to match IR semantics (round down) instead of verilog (truncate towards zero)
        let a_is_neg = MaybeBool::is_negative(&a, ClosedRange::from(range_a.as_ref()));
        let b_is_neg = MaybeBool::is_negative(&b, ClosedRange::from(range_b.as_ref()));
        let signs_differ = MaybeBool::xor(&a_is_neg, &b_is_neg);
        let res_expr = match op {
            OperatorDivMod::Div => {
                let adj_one = b_is_neg.select("1", "-1");
                let adj = signs_differ.select(&format!("({b} + {adj_one})"), "0");
                let a_adj = format!("({a} - {adj})");
                format!("{a_adj} / {b}")
            }
            OperatorDivMod::Mod => {
                let tmp_mod = self.new_temporary(span, VerilogType::new_from_range(diags, span, range_all)?)?;
                swriteln!(self.f, "{indent}{tmp_mod} = {a} % {b};");
                let should_adjust = MaybeBool::and(
                    &MaybeBool::is_not_zero(&tmp_mod, ClosedRange::from(result_range.range().as_ref())),
                    &signs_differ,
                );
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
                .report_error_internal(span, format!("{reason} cannot be zero-width"))
        })
    }

    fn lower_expression_int_expanded(
        &mut self,
        span: Span,
        range: &NonZeroWidthRange,
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
            Ok(value) => lower_expand_int_range(range.range().as_ref(), value_ty.as_ref(), value),
            Err(ZeroWidth) => {
                let value = value_ty.as_ref().as_single().unwrap();
                lower_int_constant(range, value)
            }
        };

        Ok(value)
    }

    fn lower_assignment(&mut self, span: Span, target: &IrAssignmentTarget, source: Evaluated<'n>) -> DiagResult {
        // TODO constant fold if some indices are constant?
        // TODO expose the extra knowledge we have about integer ranges to verilog?
        let &IrAssignmentTarget { base, ref steps } = target;

        // evaluate indexing steps
        let base_ty = base.as_expression().ty(self.module, self.locals);
        let steps = self.build_target_steps(span, base_ty, steps)?;

        // actually do the assignment(s)
        let mut append_assign = |target: &LoweredName, assign_op: &str| {
            let indent = self.indent;
            swriteln!(self.f, "{indent}{target}{steps} {assign_op} {source};");
        };
        match base {
            IrSignalOrVariable::Variable(var) => {
                let name = unwrap_zero_width(self.name_map.map_var(var));
                append_assign(name, "=");
            }
            IrSignalOrVariable::Signal(signal) => match signal {
                IrSignal::Port(_) | IrSignal::Wire(_) => {
                    let name = unwrap_zero_width(self.name_map.module_name_map.map_signal(signal));
                    append_assign(name, "=");
                }
                IrSignal::Register(reg) => {
                    let name_raw = unwrap_zero_width(self.name_map.module_name_map.map_signal(signal));
                    let name_shadow = self.name_map.regs_shadowed.get(&reg);

                    append_assign(name_raw, "<=");
                    if let Some(name_shadow) = name_shadow {
                        append_assign(name_shadow, "=");
                    }
                }
            },
        }

        Ok(())
    }

    fn build_target_steps(&mut self, span: Span, base_ty: IrType, steps: &[IrTargetStep]) -> DiagResult<String> {
        // if there are no steps, skip entirely
        if steps.is_empty() {
            return Ok(String::new());
        }

        // string building utils
        let mut g = String::new();
        swrite!(g, "[");

        let mut is_first_offset = true;
        let mut add_offset = |index: Evaluated, size_bits: &BigUint| {
            if !is_first_offset {
                swrite!(g, " + ");
            }
            is_first_offset = false;

            swrite!(g, "{}", index);
            if size_bits != &BigUint::ONE {
                swrite!(g, " * {}", size_bits);
            }
        };

        // handle steps
        let bit_range = bit_index_range(&base_ty.size_bits());
        let mut curr_ty = base_ty;

        for step in steps {
            curr_ty = match step {
                IrTargetStep::ArrayIndex(index) => {
                    let (element_ty, _) = curr_ty.unwrap_array();
                    let element_size_bits = element_ty.size_bits();

                    let index = self.lower_expression_int_expanded(span, &bit_range, index)?;
                    add_offset(index, &element_size_bits);

                    element_ty
                }
                IrTargetStep::ArraySlice { start, len: length } => {
                    // TODO expose the extra knowledge this length gives us about the target range to verilog?
                    let (element_ty, _) = curr_ty.unwrap_array();
                    let element_size_bits = element_ty.size_bits();

                    let start = self.lower_expression_int_expanded(span, &bit_range, start)?;
                    add_offset(start, &element_size_bits);

                    IrType::Array(Box::new(element_ty), length.clone())
                }
            };
        }

        // correctly slice the final bit length
        let result_len_bits = curr_ty.size_bits();
        if result_len_bits != BigUint::ONE {
            swrite!(g, " +: {result_len_bits}");
        }
        swrite!(g, "]");

        Ok(g)
    }

    fn new_temporary(&mut self, span: Span, ty: VerilogType) -> DiagResult<Temporary<'n>> {
        let name = self.name_scope.make_unique_str(self.diags, span, "tmp", true)?;

        let info = TemporaryInfo { name, ty };
        let info = self.temporaries.push(info);

        Ok(Temporary(&info.name))
    }
}

/// Create a range that can be used for bit indexing, slicing and related math for the given size.
fn bit_index_range(size_bits: &BigUint) -> NonZeroWidthRange {
    let bit_index_range = ClosedNonEmptyRange {
        start: BigInt::ZERO,
        end: BigInt::from(size_bits + 1u8),
    };
    NonZeroWidthRange::new(bit_index_range).unwrap()
}

fn lower_expand_int_range<'n>(
    target_ty: ClosedNonEmptyRange<&BigInt>,
    value_ty: ClosedNonEmptyRange<&BigInt>,
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

fn lower_int_constant(ty: &NonZeroWidthRange, x: &BigInt) -> Evaluated<'static> {
    let ty = ty.range();
    assert!(ty.contains(x), "Trying to emit constant {x:?} encoded as range {ty:?}");

    let repr = IntRepresentation::for_range(ty.as_ref());
    let bits = repr.size_bits();
    assert_ne!(bits, 0);

    match repr.signed() {
        Signed::Unsigned => Evaluated::String(format!("{bits}'d{x}")),
        Signed::Signed => {
            let s = match x.sign() {
                Sign::Negative => {
                    // Verilog does not actually have negative literals, it's just the unary negation operator.
                    // This makes it tricky to expression the most negative value.
                    // The expression `max_pos_value` already overflows, becoming the most negative value,
                    //   so we should skip adding a leading negative sign.
                    let prefix_sign = if x == &repr.range().start { "" } else { "-" };
                    format!("{prefix_sign}{bits}'sd{}", x.abs())
                }
                Sign::Zero | Sign::Positive => format!("{bits}'sd{x}"),
            };
            Evaluated::SignedString(s)
        }
    }
}

#[derive(Debug)]
struct EdgeString<'n> {
    edge: &'static str,
    if_prefix: &'static str,
    signal: &'n LoweredName,
}

fn lower_edge<'n>(
    diags: &Diagnostics,
    name_map: ModuleNameMap<'n>,
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
        Err(ZeroWidth) => Err(diags.report_error_internal(expr.span, "zero-width edge signal")),
    }
}

/// Range that is guaranteed to contain multiple values,
///   which means that it will be represented with at least one bit in hardware.
#[derive(Debug, Clone)]
pub struct NonZeroWidthRange(ClosedNonEmptyRange<BigInt>);

impl NonZeroWidthRange {
    const ZERO_ONE: NonZeroWidthRange = NonZeroWidthRange(ClosedNonEmptyRange {
        start: BigInt::ZERO,
        end: BigInt::TWO,
    });

    fn new(range: ClosedNonEmptyRange<BigInt>) -> Result<Self, ZeroWidth> {
        let repr = IntRepresentation::for_range(range.as_ref());
        if repr.size_bits() == 0 {
            Err(ZeroWidth)
        } else {
            Ok(NonZeroWidthRange(range))
        }
    }

    fn range(&self) -> &ClosedNonEmptyRange<BigInt> {
        &self.0
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
    ///
    /// This becomes `reg x;` in Verilog.
    Bit,
    /// Potentially multi-bit values. This can still happen to have width 1,
    /// The difference with [SingleBit] in that case is that this should still be represented as an array.
    ///
    /// This becomes `reg [n-1:0] x;` in Verilog.
    Array(NonZeroU32),
}

impl VerilogType {
    pub fn new_array(
        diags: &Diagnostics,
        span: Span,
        size_bits: BigUint,
    ) -> DiagResult<Result<VerilogType, ZeroWidth>> {
        let size_bits = diag_big_int_to_u32(diags, span, &size_bits.into(), "array width too large")?;
        let size_bits = try_inner!(NonZeroU32::new(size_bits).ok_or(ZeroWidth));
        Ok(Ok(VerilogType::Array(size_bits)))
    }

    pub fn new_from_ir(diags: &Diagnostics, span: Span, ty: &IrType) -> DiagResult<Result<VerilogType, ZeroWidth>> {
        match ty {
            IrType::Bool => Ok(Ok(VerilogType::Bit)),
            IrType::Int(_) | IrType::Tuple(_) | IrType::Array(_, _) => Self::new_array(diags, span, ty.size_bits()),
        }
    }

    pub fn new_from_range(diags: &Diagnostics, span: Span, range: NonZeroWidthRange) -> DiagResult<VerilogType> {
        let repr = IntRepresentation::for_range(range.range().as_ref());
        Ok(Self::new_array(diags, span, BigUint::from(repr.size_bits()))?.expect("range should be non-zero-width"))
    }

    pub fn size_bits(self) -> NonZeroU32 {
        match self {
            VerilogType::Bit => NonZeroU32::new(1).unwrap(),
            VerilogType::Array(width) => width,
        }
    }

    pub fn prefix(self) -> impl Display {
        struct P(VerilogType);

        impl Display for P {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    VerilogType::Bit => Ok(()),
                    VerilogType::Array(width) => write!(f, "[{}:0] ", width.get() - 1),
                }
            }
        }

        P(self)
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
        diags.report_error_simple(
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

// Updated to "IEEE Standard for SystemVerilog", IEEE 1800-2023
// TODO also include vhdl keywords and ban both in generated output?
lazy_static! {
    static ref VERILOG_KEYWORDS: IndexSet<&'static str> = {
        include_str!("verilog_keywords.txt")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect()
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
    fn is_not_zero(v: &impl Display, r: ClosedRange<&BigInt>) -> MaybeBool {
        let can_be_zero = r.contains(&&BigInt::ZERO);
        let can_be_non_zero = r != ClosedRange::single(BigInt::ZERO).as_ref();

        if can_be_zero && can_be_non_zero {
            MaybeBool::Runtime(format!("({v} != 0)"))
        } else if can_be_zero {
            MaybeBool::Const(false)
        } else {
            MaybeBool::Const(true)
        }
    }

    fn is_negative(v: &impl Display, r: ClosedRange<&BigInt>) -> MaybeBool {
        let can_be_neg = r.start.is_negative();
        let can_be_non_neg = !r.end.is_negative();

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
