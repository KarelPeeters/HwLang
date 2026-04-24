use crate::mid::ir::{
    IrEnumType, IrModule, IrModuleChild, IrModuleInfo, IrModuleInternalInstance, IrModules, IrPort, IrPortConnection,
    IrPortInfo, IrSignal, IrType, IrWire, IrWireInfo,
};
use crate::syntax::ast::PortDirection;
use crate::util::arena::{Idx, IndexType};
use crate::util::big_int::BigUint;
use crate::util::int::IntRepresentation;
use crate::util::Indent;
use fnv::FnvHasher;
use hwl_util::swriteln;
use itertools::enumerate;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct LoweredCppWrap {
    pub source: String,
    pub check_hash: u64,
}

#[derive(Debug, Clone)]
pub struct CppSignalInfo {
    pub id: usize,
    pub path: Vec<String>,
    pub name: String,
    pub ty: IrType,
}

pub fn lower_cpp_wrap(modules: &IrModules, top_module: IrModule, lowered_cpp_source: &str) -> LoweredCppWrap {
    let check_hash = {
        let mut hasher = FnvHasher::default();
        lowered_cpp_source.hash(&mut hasher);
        top_module.inner().index().hash(&mut hasher);
        "hwl cpp wrapper v0".hash(&mut hasher);
        hasher.finish()
    };

    let mut source = String::new();
    swriteln!(source, "#include \"lowered.cpp\"");
    swriteln!(source, "#include <cstddef>");
    swriteln!(source, "#include <cstdint>");
    swriteln!(source, "#include <cstring>");
    swriteln!(source, "#include <exception>");
    swriteln!(source, "#include <new>");
    swriteln!(source);

    emit_helpers(&mut source);
    emit_instance(&mut source, top_module);
    emit_api(modules, top_module, check_hash, &mut source);

    LoweredCppWrap { source, check_hash }
}

pub fn collect_cpp_signals(modules: &IrModules, top_module: IrModule) -> Vec<CppSignalInfo> {
    let top_info = &modules[top_module];
    let root_path = vec![module_name(top_module, top_info)];
    let root_ports = top_info
        .ports
        .iter()
        .map(|(port, port_info)| port_expr(port, port_info))
        .collect::<Vec<_>>();

    let mut signals = Vec::new();
    collect_signals_recursive(
        modules,
        top_module,
        &root_path,
        "instance->next_signals",
        &root_ports,
        &mut signals,
    );
    signals
}

fn emit_helpers(f: &mut String) {
    swriteln!(f, "namespace hwlang_cpp_wrap {{");
    swriteln!(f, "constexpr uint8_t RESULT_OK = 0;");
    swriteln!(f, "constexpr uint8_t RESULT_BAD_INDEX = 1;");
    swriteln!(f, "constexpr uint8_t RESULT_EXCEPTION = 2;");
    swriteln!(f, "constexpr uint8_t RESULT_BAD_DIRECTION = 3;");
    swriteln!(f);
    swriteln!(f, "void clear_data(std::size_t data_len, uint8_t *data) {{");
    swriteln!(f, "{I}std::memset(data, 0, data_len);");
    swriteln!(f, "}}");
    swriteln!(f);
    swriteln!(f, "void write_bit(uint8_t *data, std::size_t bit, bool value) {{");
    swriteln!(f, "{I}if (value) {{");
    swriteln!(f, "{I}{I}data[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));");
    swriteln!(f, "{I}}}");
    swriteln!(f, "}}");
    swriteln!(f);
    swriteln!(f, "bool read_bit(const uint8_t *data, std::size_t bit) {{");
    swriteln!(f, "{I}return ((data[bit / 8] >> (bit % 8)) & 1u) != 0;");
    swriteln!(f, "}}");
    swriteln!(f);
    swriteln!(f, "bool check_len(std::size_t expected_bits, std::size_t data_len) {{");
    swriteln!(f, "{I}std::size_t expected_bytes = expected_bits == 0 ? 0 : ((expected_bits + 7) / 8);");
    swriteln!(f, "{I}return data_len >= expected_bytes;");
    swriteln!(f, "}}");
    swriteln!(f, "}}");
    swriteln!(f);
}

fn emit_instance(f: &mut String, top_module: IrModule) {
    let module_index = top_module.inner().index();
    swriteln!(f, "struct HwlangCppSimInstance {{");
    swriteln!(f, "{I}ModuleSignals_{module_index} prev_signals{{}};");
    swriteln!(f, "{I}ModuleSignals_{module_index} next_signals{{}};");
    swriteln!(f, "{I}ModulePortsVal_{module_index} prev_ports{{}};");
    swriteln!(f, "{I}ModulePortsVal_{module_index} next_ports{{}};");
    swriteln!(f, "{I}uint64_t time = 0;");
    swriteln!(f, "}};");
    swriteln!(f);
}

fn emit_api(modules: &IrModules, top_module: IrModule, check_hash: u64, f: &mut String) {
    let module_index = top_module.inner().index();
    swriteln!(f, "extern \"C\" uint64_t check_hash() {{");
    swriteln!(f, "{I}return {check_hash}u;");
    swriteln!(f, "}}");
    swriteln!(f);

    swriteln!(f, "extern \"C\" void *create_instance() {{");
    swriteln!(f, "{I}return new (std::nothrow) HwlangCppSimInstance{{}};");
    swriteln!(f, "}}");
    swriteln!(f);

    swriteln!(f, "extern \"C\" void destroy_instance(void *instance_raw) {{");
    swriteln!(
        f,
        "{I}delete static_cast<HwlangCppSimInstance *>(instance_raw);"
    );
    swriteln!(f, "}}");
    swriteln!(f);

    swriteln!(f, "extern \"C\" uint8_t step(void *instance_raw, uint64_t increment_time) {{");
    swriteln!(f, "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);");
    swriteln!(f, "{I}try {{");
    swriteln!(f, "{I}{I}for (std::size_t i = 0; i < 32; i++) {{");
    swriteln!(f, "{I}{I}{I}module_{module_index}_all(");
    swriteln!(f, "{I}{I}{I}{I}instance->prev_signals,");
    swriteln!(f, "{I}{I}{I}{I}instance->prev_ports.as_ptrs(),");
    swriteln!(f, "{I}{I}{I}{I}instance->next_signals,");
    swriteln!(f, "{I}{I}{I}{I}instance->next_ports.as_ptrs()");
    swriteln!(f, "{I}{I}{I});");
    swriteln!(f, "{I}{I}{I}instance->prev_signals = instance->next_signals;");
    swriteln!(f, "{I}{I}{I}instance->prev_ports = instance->next_ports;");
    swriteln!(f, "{I}{I}}}");
    swriteln!(f, "{I}{I}instance->time += increment_time;");
    swriteln!(f, "{I}{I}return hwlang_cpp_wrap::RESULT_OK;");
    swriteln!(f, "{I}}} catch (...) {{");
    swriteln!(f, "{I}{I}return hwlang_cpp_wrap::RESULT_EXCEPTION;");
    swriteln!(f, "{I}}}");
    swriteln!(f, "}}");
    swriteln!(f);

    emit_get_port(modules, top_module, f);
    emit_set_port(modules, top_module, f);
    emit_get_signal(modules, top_module, f);
}

fn emit_get_port(modules: &IrModules, top_module: IrModule, f: &mut String) {
    let top_info = &modules[top_module];
    swriteln!(
        f,
        "extern \"C\" uint8_t get_port(void *instance_raw, std::size_t port_index, std::size_t data_len, uint8_t *data) {{"
    );
    swriteln!(f, "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);");
    swriteln!(f, "{I}switch (port_index) {{");
    for (port_index, (port, port_info)) in enumerate(&top_info.ports) {
        let expr = format!("instance->next_ports.{}", port_expr(port, port_info));
        emit_get_case(f, port_index, &expr, &port_info.ty);
    }
    swriteln!(f, "{I}{I}default: return hwlang_cpp_wrap::RESULT_BAD_INDEX;");
    swriteln!(f, "{I}}}");
    swriteln!(f, "}}");
    swriteln!(f);
}

fn emit_set_port(modules: &IrModules, top_module: IrModule, f: &mut String) {
    let top_info = &modules[top_module];
    swriteln!(
        f,
        "extern \"C\" uint8_t set_port(void *instance_raw, std::size_t port_index, std::size_t data_len, const uint8_t *data) {{"
    );
    swriteln!(f, "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);");
    swriteln!(f, "{I}switch (port_index) {{");
    for (port_index, (port, port_info)) in enumerate(&top_info.ports) {
        swriteln!(f, "{I}{I}case {port_index}: {{");
        if port_info.direction == PortDirection::Output {
            swriteln!(f, "{I}{I}{I}return hwlang_cpp_wrap::RESULT_BAD_DIRECTION;");
        } else {
            let size_bits = port_info.ty.size_bits();
            swriteln!(
                f,
                "{I}{I}{I}if (!hwlang_cpp_wrap::check_len({size_bits}, data_len)) return hwlang_cpp_wrap::RESULT_BAD_INDEX;"
            );
            let expr = format!("instance->next_ports.{}", port_expr(port, port_info));
            emit_unpack_value(f, Indent::new(3), &port_info.ty, &expr, "0");
            swriteln!(f, "{I}{I}{I}return hwlang_cpp_wrap::RESULT_OK;");
        }
        swriteln!(f, "{I}{I}}}");
    }
    swriteln!(f, "{I}{I}default: return hwlang_cpp_wrap::RESULT_BAD_INDEX;");
    swriteln!(f, "{I}}}");
    swriteln!(f, "}}");
    swriteln!(f);
}

fn emit_get_signal(modules: &IrModules, top_module: IrModule, f: &mut String) {
    let top_info = &modules[top_module];
    let root_ports = top_info
        .ports
        .iter()
        .map(|(port, port_info)| format!("instance->next_ports.{}", port_expr(port, port_info)))
        .collect::<Vec<_>>();
    let mut cases = Vec::new();
    collect_signal_cases_recursive(
        modules,
        top_module,
        "instance->next_signals",
        &root_ports,
        &mut cases,
    );

    swriteln!(
        f,
        "extern \"C\" uint8_t get_signal(void *instance_raw, std::size_t signal_index, std::size_t data_len, uint8_t *data) {{"
    );
    swriteln!(f, "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);");
    swriteln!(f, "{I}switch (signal_index) {{");
    for (signal_index, (expr, ty)) in enumerate(cases) {
        emit_get_case(f, signal_index, &expr, &ty);
    }
    swriteln!(f, "{I}{I}default: return hwlang_cpp_wrap::RESULT_BAD_INDEX;");
    swriteln!(f, "{I}}}");
    swriteln!(f, "}}");
    swriteln!(f);
}

fn emit_get_case(f: &mut String, index: usize, expr: &str, ty: &IrType) {
    let size_bits = ty.size_bits();
    swriteln!(f, "{I}{I}case {index}: {{");
    swriteln!(
        f,
        "{I}{I}{I}if (!hwlang_cpp_wrap::check_len({size_bits}, data_len)) return hwlang_cpp_wrap::RESULT_BAD_INDEX;"
    );
    swriteln!(f, "{I}{I}{I}hwlang_cpp_wrap::clear_data(data_len, data);");
    emit_pack_value(f, Indent::new(3), ty, expr, "0");
    swriteln!(f, "{I}{I}{I}return hwlang_cpp_wrap::RESULT_OK;");
    swriteln!(f, "{I}{I}}}");
}

fn emit_pack_value(f: &mut String, indent: Indent, ty: &IrType, value: &str, offset: &str) {
    match ty {
        IrType::Bool => {
            swriteln!(f, "{indent}hwlang_cpp_wrap::write_bit(data, {offset}, {value});");
        }
        IrType::Int(range) => {
            let width = IntRepresentation::for_range(range.as_ref()).size_bits();
            let tmp_i = format!("i_{}", offset_identifier(offset));
            swriteln!(f, "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {width}; {tmp_i}++) {{");
            swriteln!(
                f,
                "{indent}{I}hwlang_cpp_wrap::write_bit(data, {offset} + {tmp_i}, (({value}) >> {tmp_i}) & 1);"
            );
            swriteln!(f, "{indent}}}");
        }
        IrType::Array(inner, len) => {
            let inner_bits = inner.size_bits();
            let tmp_i = format!("i_{}", offset_identifier(offset));
            swriteln!(f, "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {len}; {tmp_i}++) {{");
            emit_pack_value(
                f,
                indent.nest(),
                inner,
                &format!("({value})[{tmp_i}]"),
                &format!("{offset} + {tmp_i} * {inner_bits}"),
            );
            swriteln!(f, "{indent}}}");
        }
        IrType::Tuple(elements) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(elements) {
                emit_pack_value(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Struct(info) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(info.fields.values()) {
                emit_pack_value(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Enum(info) => {
            emit_pack_value(
                f,
                indent,
                &IrType::Int(info.tag_range()),
                &format!("static_cast<int64_t>(({value}).index())"),
                offset,
            );
            let payload_offset = info.tag_size_bits();
            swriteln!(f, "{indent}switch (({value}).index()) {{");
            for (variant_i, payload_ty) in enumerate(info.variants.values()) {
                if let Some(payload_ty) = payload_ty {
                    swriteln!(f, "{indent}{I}case {variant_i}: {{");
                    emit_pack_value(
                        f,
                        indent.nest().nest(),
                        payload_ty,
                        &format!("std::get<{variant_i}>({value})"),
                        &format!("{offset} + {payload_offset}"),
                    );
                    swriteln!(f, "{indent}{I}{I}break;");
                    swriteln!(f, "{indent}{I}}}");
                }
            }
            swriteln!(f, "{indent}}}");
        }
    }
}

fn emit_unpack_value(f: &mut String, indent: Indent, ty: &IrType, value: &str, offset: &str) {
    match ty {
        IrType::Bool => {
            swriteln!(f, "{indent}{value} = hwlang_cpp_wrap::read_bit(data, {offset});");
        }
        IrType::Int(range) => {
            let repr = IntRepresentation::for_range(range.as_ref());
            let width = repr.size_bits();
            let tmp = format!("v_{}", offset_identifier(offset));
            let tmp_i = format!("i_{}", offset_identifier(offset));
            swriteln!(f, "{indent}uint64_t {tmp} = 0;");
            swriteln!(f, "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {width}; {tmp_i}++) {{");
            swriteln!(
                f,
                "{indent}{I}if (hwlang_cpp_wrap::read_bit(data, {offset} + {tmp_i})) {tmp} |= (uint64_t{{1}} << {tmp_i});"
            );
            swriteln!(f, "{indent}}}");
            match repr {
                IntRepresentation::Unsigned { .. } => {
                    swriteln!(f, "{indent}{value} = static_cast<int64_t>({tmp});");
                }
                IntRepresentation::Signed { width_1 } => {
                    let sign_adjust = width_1 + 1;
                    swriteln!(
                        f,
                        "{indent}{value} = hwlang_cpp_wrap::read_bit(data, {offset} + {width_1}) ? static_cast<int64_t>({tmp}) - (int64_t{{1}} << {sign_adjust}) : static_cast<int64_t>({tmp});"
                    );
                }
            }
        }
        IrType::Array(inner, len) => {
            let inner_bits = inner.size_bits();
            let tmp_i = format!("i_{}", offset_identifier(offset));
            swriteln!(f, "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {len}; {tmp_i}++) {{");
            emit_unpack_value(
                f,
                indent.nest(),
                inner,
                &format!("({value})[{tmp_i}]"),
                &format!("{offset} + {tmp_i} * {inner_bits}"),
            );
            swriteln!(f, "{indent}}}");
        }
        IrType::Tuple(elements) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(elements) {
                emit_unpack_value(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Struct(info) => {
            let mut child_offset = BigUint::ZERO;
            for (i, element) in enumerate(info.fields.values()) {
                emit_unpack_value(
                    f,
                    indent,
                    element,
                    &format!("std::get<{i}>({value})"),
                    &format!("{offset} + {child_offset}"),
                );
                child_offset += element.size_bits();
            }
        }
        IrType::Enum(info) => emit_unpack_enum(f, indent, info, value, offset),
    }
}

fn emit_unpack_enum(f: &mut String, indent: Indent, info: &IrEnumType, value: &str, offset: &str) {
    let tag_ty = IrType::Int(info.tag_range());
    let tag_tmp = format!("tag_{}", offset_identifier(offset));
    swriteln!(f, "{indent}int64_t {tag_tmp} = 0;");
    emit_unpack_value(f, indent, &tag_ty, &tag_tmp, offset);
    let payload_offset = info.tag_size_bits();
    swriteln!(f, "{indent}switch ({tag_tmp}) {{");
    for (variant_i, payload_ty) in enumerate(info.variants.values()) {
        swriteln!(f, "{indent}{I}case {variant_i}: {{");
        swriteln!(f, "{indent}{I}{I}({value}).template emplace<{variant_i}>();");
        if let Some(payload_ty) = payload_ty {
            emit_unpack_value(
                f,
                indent.nest().nest(),
                payload_ty,
                &format!("std::get<{variant_i}>({value})"),
                &format!("{offset} + {payload_offset}"),
            );
        }
        swriteln!(f, "{indent}{I}{I}break;");
        swriteln!(f, "{indent}{I}}}");
    }
    swriteln!(f, "{indent}}}");
}

fn collect_signals_recursive(
    modules: &IrModules,
    module: IrModule,
    path: &[String],
    signal_prefix: &str,
    port_exprs: &[String],
    signals: &mut Vec<CppSignalInfo>,
) {
    let module_info = &modules[module];

    for (port_index, (_port, port_info)) in enumerate(&module_info.ports) {
        signals.push(CppSignalInfo {
            id: signals.len(),
            path: path.to_owned(),
            name: port_info.name.clone(),
            ty: port_info.ty.clone(),
        });
        debug_assert!(port_exprs.get(port_index).is_some());
    }

    for (wire, wire_info) in &module_info.wires {
        signals.push(CppSignalInfo {
            id: signals.len(),
            path: path.to_owned(),
            name: wire_info
                .debug_info_id
                .inner
                .clone()
                .unwrap_or_else(|| format!("wire_{}", wire.inner().index())),
            ty: wire_info.ty.clone(),
        });
    }

    for (child_index, child) in enumerate(&module_info.children) {
        if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
            let child_path_name = instance.name.clone().unwrap_or_else(|| format!("inst_{child_index}"));
            let mut child_path = path.to_owned();
            child_path.push(child_path_name);
            let child_signal_prefix = format!("{signal_prefix}.child_{child_index}");
            let child_port_exprs = child_port_exprs(modules, module_info, signal_prefix, port_exprs, child_index, instance);
            collect_signals_recursive(
                modules,
                instance.module,
                &child_path,
                &child_signal_prefix,
                &child_port_exprs,
                signals,
            );
        }
    }
}

fn collect_signal_cases_recursive(
    modules: &IrModules,
    module: IrModule,
    signal_prefix: &str,
    port_exprs: &[String],
    cases: &mut Vec<(String, IrType)>,
) {
    let module_info = &modules[module];

    for (port_index, (_port, port_info)) in enumerate(&module_info.ports) {
        cases.push((port_exprs[port_index].clone(), port_info.ty.clone()));
    }

    for (wire, wire_info) in &module_info.wires {
        cases.push((
            format!("{signal_prefix}.{}", wire_expr(wire, wire_info)),
            wire_info.ty.clone(),
        ));
    }

    for (child_index, child) in enumerate(&module_info.children) {
        if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
            let child_signal_prefix = format!("{signal_prefix}.child_{child_index}");
            let child_port_exprs = child_port_exprs(modules, module_info, signal_prefix, port_exprs, child_index, instance);
            collect_signal_cases_recursive(
                modules,
                instance.module,
                &child_signal_prefix,
                &child_port_exprs,
                cases,
            );
        }
    }
}

fn child_port_exprs(
    modules: &IrModules,
    module_info: &IrModuleInfo,
    signal_prefix: &str,
    port_exprs: &[String],
    child_index: usize,
    instance: &IrModuleInternalInstance,
) -> Vec<String> {
    let child_module_info = &modules[instance.module];
    instance
        .port_connections
        .iter()
        .enumerate()
        .map(|(connection_index, connection)| match connection.inner {
            IrPortConnection::Input(signal) | IrPortConnection::Output(Some(signal)) => {
                signal_expr(module_info, signal_prefix, port_exprs, signal)
            }
            IrPortConnection::Output(None) => {
                let _port_info = child_module_info.ports.get_by_index(connection_index).unwrap().1;
                format!("{signal_prefix}.dummy_{child_index}_{connection_index}")
            }
        })
        .collect()
}

fn signal_expr(module_info: &IrModuleInfo, signal_prefix: &str, port_exprs: &[String], signal: IrSignal) -> String {
    match signal {
        IrSignal::Port(port) => port_exprs[port.inner().index()].clone(),
        IrSignal::Wire(wire) => format!("{signal_prefix}.{}", wire_expr(wire, &module_info.wires[wire])),
    }
}

fn module_name(module: IrModule, info: &IrModuleInfo) -> String {
    info.debug_info_id
        .inner
        .clone()
        .unwrap_or_else(|| format!("module_{}", module.inner().index()))
}

fn port_expr(port: IrPort, port_info: &IrPortInfo) -> String {
    name_str("port", port.inner(), Some(&port_info.name))
}

fn wire_expr(wire: IrWire, wire_info: &IrWireInfo) -> String {
    name_str(
        "wire",
        wire.inner(),
        wire_info.debug_info_id.inner.as_ref().map(String::as_ref),
    )
}

fn name_str(prefix: &str, index: Idx, id: Option<&str>) -> String {
    let index = index.index();
    match id {
        Some(id) => {
            let str_filtered: String = id.chars().filter(|&c| c.is_ascii_alphanumeric() || c == '_').collect();
            format!("{prefix}_{index}_{str_filtered}")
        }
        None => format!("{prefix}_{index}"),
    }
}

fn offset_identifier(offset: &str) -> String {
    offset
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

const I: &str = Indent::I;
