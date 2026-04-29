use crate::back::cpp_bits::{CppBitLoad, CppBitStore, emit_value_from_bits, emit_value_to_bits, offset_identifier};
use crate::mid::ir::{
    IrModule, IrModuleChild, IrModuleInfo, IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrPortInfo,
    IrSignal, IrType, IrWire, IrWireInfo,
};
use crate::syntax::ast::PortDirection;
use crate::util::Indent;
use crate::util::arena::{Idx, IndexType};
use fnv::FnvHasher;
use hwl_util::swriteln;
use itertools::enumerate;
use std::collections::HashSet;
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
    pub kind: CppSignalKind,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CppSignalKind {
    Port,
    Wire,
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
    swriteln!(
        f,
        "{I}std::size_t expected_bytes = expected_bits == 0 ? 0 : ((expected_bits + 7) / 8);"
    );
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
    swriteln!(f, "{I}delete static_cast<HwlangCppSimInstance *>(instance_raw);");
    swriteln!(f, "}}");
    swriteln!(f);

    swriteln!(
        f,
        "extern \"C\" uint8_t step(void *instance_raw, uint64_t increment_time) {{"
    );
    swriteln!(
        f,
        "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);"
    );
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
    swriteln!(
        f,
        "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);"
    );
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
    swriteln!(
        f,
        "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);"
    );
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
            emit_value_from_bits(
                f,
                Indent::new(3),
                &port_info.ty,
                &expr,
                "0",
                CppBitLoad::PackedData { data: "data" },
                &mut wrapped_temp_name,
            );
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
    collect_signal_cases_recursive(modules, top_module, "instance->next_signals", &root_ports, &mut cases);

    swriteln!(
        f,
        "extern \"C\" uint8_t get_signal(void *instance_raw, std::size_t signal_index, std::size_t data_len, uint8_t *data) {{"
    );
    swriteln!(
        f,
        "{I}auto *instance = static_cast<HwlangCppSimInstance *>(instance_raw);"
    );
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
    emit_value_to_bits(
        f,
        Indent::new(3),
        ty,
        expr,
        "0",
        CppBitStore::PackedData { data: "data" },
        &mut wrapped_temp_name,
    );
    swriteln!(f, "{I}{I}{I}return hwlang_cpp_wrap::RESULT_OK;");
    swriteln!(f, "{I}{I}}}");
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
    let hidden_wires = hidden_parent_bridge_wires(modules, module_info);

    for (port_index, (_port, port_info)) in enumerate(&module_info.ports) {
        signals.push(CppSignalInfo {
            id: signals.len(),
            path: path.to_owned(),
            name: port_info.name.clone(),
            ty: port_info.ty.clone(),
            kind: CppSignalKind::Port,
        });
        debug_assert!(port_exprs.get(port_index).is_some());
    }

    for (wire, wire_info) in &module_info.wires {
        if hidden_wires.contains(&wire.inner().index()) {
            continue;
        }
        signals.push(CppSignalInfo {
            id: signals.len(),
            path: path.to_owned(),
            name: wire_info
                .debug_info_id
                .inner
                .clone()
                .unwrap_or_else(|| format!("wire_{}", wire.inner().index())),
            ty: wire_info.ty.clone(),
            kind: CppSignalKind::Wire,
        });
    }

    for (child_index, child) in enumerate(&module_info.children) {
        if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
            let child_path_name = instance.name.clone().unwrap_or_else(|| format!("inst_{child_index}"));
            let mut child_path = path.to_owned();
            child_path.push(child_path_name);
            let child_signal_prefix = format!("{signal_prefix}.child_{child_index}");
            let child_port_exprs =
                child_port_exprs(modules, module_info, signal_prefix, port_exprs, child_index, instance);
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
    let hidden_wires = hidden_parent_bridge_wires(modules, module_info);

    for (port_index, (_port, port_info)) in enumerate(&module_info.ports) {
        cases.push((port_exprs[port_index].clone(), port_info.ty.clone()));
    }

    for (wire, wire_info) in &module_info.wires {
        if hidden_wires.contains(&wire.inner().index()) {
            continue;
        }
        cases.push((
            format!("{signal_prefix}.{}", wire_expr(wire, wire_info)),
            wire_info.ty.clone(),
        ));
    }

    for (child_index, child) in enumerate(&module_info.children) {
        if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
            let child_signal_prefix = format!("{signal_prefix}.child_{child_index}");
            let child_port_exprs =
                child_port_exprs(modules, module_info, signal_prefix, port_exprs, child_index, instance);
            collect_signal_cases_recursive(modules, instance.module, &child_signal_prefix, &child_port_exprs, cases);
        }
    }
}

fn hidden_parent_bridge_wires(modules: &IrModules, module_info: &IrModuleInfo) -> HashSet<usize> {
    let mut result = HashSet::new();
    for child in &module_info.children {
        let IrModuleChild::ModuleInternalInstance(instance) = &child.inner else {
            continue;
        };
        let child_module_info = &modules[instance.module];
        for (connection_index, connection) in instance.port_connections.iter().enumerate() {
            let IrPortConnection::Input(IrSignal::Wire(wire)) = connection.inner else {
                continue;
            };
            let Some((_, child_port_info)) = child_module_info.ports.get_by_index(connection_index) else {
                continue;
            };
            let parent_wire_info = &module_info.wires[wire];
            if parent_wire_info.debug_info_id.inner.as_deref() == Some(child_port_info.name.as_str()) {
                result.insert(wire.inner().index());
            }
        }
    }
    result
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

fn wrapped_temp_name(prefix: &str, offset: &str) -> String {
    format!("{prefix}_{}", offset_identifier(offset))
}

const I: &str = Indent::I;
