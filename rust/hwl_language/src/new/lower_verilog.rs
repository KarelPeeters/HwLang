use crate::data::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::data::parsed::ParsedDatabase;
use crate::data::source::SourceDatabase;
use crate::new::ir::{IrDatabase, IrModule, IrModuleInfo, IrPortInfo, IrType};
use crate::syntax::ast::{Identifier, PortDirection};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::int::IntRepresentation;
use crate::{swrite, swriteln, throw};
use indexmap::IndexMap;
use itertools::enumerate;
use lazy_static::lazy_static;
use num_bigint::{BigInt, BigUint};
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;

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
    compiled: &IrDatabase,
) -> Result<LoweredVerilog, ErrorGuaranteed> {
    let mut result = String::new();
    let mut module_map = IndexMap::new();
    let mut top_name_scope = LoweredNameScope::default();

    let top_name = lower_module(
        diags,
        source,
        parsed,
        compiled,
        &mut module_map,
        &mut top_name_scope,
        compiled.top_module?,
        &mut result,
    )?;

    Ok(LoweredVerilog {
        verilog_source: result,
        top_module_name: top_name.0,
        debug_info_module_map: module_map.into_iter().map(|(k, v)| (k, v.0)).collect(),
    })
}

#[derive(Debug, Clone)]
struct LoweredName(String);

#[derive(Default)]
struct LoweredNameScope {
    used: HashSet<String>,
}

impl LoweredNameScope {
    pub fn exact_for_new_id(&mut self, diags: &Diagnostics, id: &Identifier) -> Result<LoweredName, ErrorGuaranteed> {
        check_identifier_valid(diags, id)?;

        if !self.used.insert(id.string.clone()) {
            throw!(diags.report_internal_error(id.span, format!("lowered identifier `{}` already used its scope", id.string)))
        }

        Ok(LoweredName(id.string.clone()))
    }

    pub fn unique_for_new_id(&mut self, diags: &Diagnostics, id: &Identifier) -> Result<LoweredName, ErrorGuaranteed> {
        check_identifier_valid(diags, id)?;

        if self.used.insert(id.string.clone()) {
            return Ok(LoweredName(id.string.clone()));
        }

        for i in 0u32.. {
            let suffixed = format!("{}_{}", id.string, i);
            if self.used.insert(suffixed.clone()) {
                return Ok(LoweredName(suffixed));
            }
        }

        throw!(diags.report_internal_error(id.span, format!("failed to generate unique lowered identifier for `{}`", id.string)))
    }
}

// TODO replace with name mangling that forces everything to be valid
fn check_identifier_valid(diags: &Diagnostics, id: &Identifier) -> Result<(), ErrorGuaranteed> {
    let s = &id.string;

    if s.len() == 0 {
        throw!(diags.report_simple("invalid verilog identifier: identifier cannot be empty", id.span, "identifier used here"))
    }
    let first = s.chars().next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        throw!(diags.report_simple("invalid verilog identifier: first character must be alphabetic or underscore", id.span, "identifier used here"))
    }
    for c in s.chars() {
        if !(c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '$' || c == '_') {
            throw!(diags.report_simple(format!("invalid verilog identifier: character `{c}` not allowed in identifier"), id.span, "identifier used here"))
        }
    }

    Ok(())
}

/// Indentation used in the generated code.
const I: &str = "    ";

// TODO write straight into a single string buffer instead of repeated concatenation
fn lower_module(
    diags: &Diagnostics,
    source: &SourceDatabase,
    parsed: &ParsedDatabase,
    compiled: &IrDatabase,
    module_map: &mut IndexMap<IrModule, LoweredName>,
    top_name_scope: &mut LoweredNameScope,
    module: IrModule,
    f: &mut String,
) -> Result<LoweredName, ErrorGuaranteed> {
    // TODO careful with name scoping: we don't want eg. ports to accidentally shadow other modules
    //   or maybe verilog has separate namespaces, then it's fine

    let module_info = &compiled.modules[module];
    let IrModuleInfo { debug_info_id, debug_info_generic_args, ports, registers, wires, processes } = module_info;
    let module_name = top_name_scope.unique_for_new_id(diags, debug_info_id)?;

    let mut module_name_scope = LoweredNameScope::default();

    swriteln!(f, "// module {}", debug_info_id.string);
    swriteln!(f, "//   defined in \"{}\"", source[debug_info_id.span.start.file].path_raw);

    if let Some(generic_args) = debug_info_generic_args {
        swriteln!(f, "//   instantiated with generic arguments:");
        for (arg_name, arg_value) in generic_args {
            swriteln!(f, "//     {}={}", arg_name.string, arg_value.to_diagnostic_string());
        }
    }

    swrite!(f, "module {} (", module_name);
    let mut port_lines = vec![];
    let mut port_name_map = IndexMap::new();
    let mut last_actual_port_index = None;

    for (port_index, (port, port_info)) in enumerate(ports) {
        let IrPortInfo { debug_info_id, debug_info_ty, debug_info_domain, direction, ty } = port_info;

        // TODO check that port names are valid and unique
        let lower_name = module_name_scope.exact_for_new_id(diags, debug_info_id)?;
        let port_ty = VerilogType::from_ir_ty(diags, debug_info_id.span, ty)?;

        let ty_str = port_ty.to_prefix_str();
        let (is_actual_port, ty_str) = match ty_str.as_ref().map(|s| s.as_str()) {
            Ok(ty_str) => (true, ty_str),
            Err(ZeroWidth) => (false, "[empty]"),
        };
        let dir_str = match direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
        };
        let ty_debug_str = debug_info_ty.as_type().to_diagnostic_string();

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
        let end_str = if Some(port_index) == last_actual_port_index { "" } else { "," };

        swrite!(f, "{start_str}{main_str}{end_str} // {comment_str}")
    }

    if ports.len() > 0 {
        swrite!(f, "\n");
    }
    swriteln!(f, ");");

    // TODO body

    swriteln!(f, "endmodule");

    Ok(module_name)
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

    pub fn from_ir_ty(diags: &Diagnostics, span: Span, ty: &IrType) -> Result<VerilogType, ErrorGuaranteed> {
        match ty {
            IrType::Bool => Ok(VerilogType::Bit),
            IrType::Int(v) => {
                Self::array(diags, span, IntRepresentation::for_range(v).width)
            }
            IrType::Array(inner, len) => {
                Self::array(diags, span, inner.bit_width() * len)
            }
        }
    }

    pub fn to_prefix_str(self) -> Result<String, ZeroWidth> {
        match self {
            VerilogType::ZeroWidth =>
                Err(ZeroWidth),
            VerilogType::Bit =>
                Ok("".to_string()),
            VerilogType::Array(width) =>
                Ok(format!("[{}:0] ", width.get() - 1)),
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

fn diag_u64_to_u32(diags: &Diagnostics, span: Span, value: u64, message: &str) -> Result<u32, ErrorGuaranteed> {
    value.try_into().map_err(|_| {
        diags.report_simple(
            format!("{message}: overflow when converting {value} to u32"),
            span,
            "used here",
        )
    })
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
