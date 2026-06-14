use crate::lower_process::{lower_process_clocked, lower_process_combinatorial};
use crate::lower_type::lower_ty;
use crate::simulator::LowerResult;
use hwl_language::front::signal::PortOrWire;
use hwl_language::mid::ir::{
    IrModule, IrModuleChild, IrModuleInfo, IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrSignal,
    IrSignals, IrWire,
};
use hwl_language::util::data::IndexMapExt;
use hwl_language::util::iter::IterExt;
use indexmap::IndexMap;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{BasicTypeEnum, StructType};
use itertools::enumerate;

pub struct LoweredModuleInfo<'ctx> {
    pub index: usize,

    /// Module signal type info, including the struct type to store all wires declared in this module
    pub signal_types: ModuleSignalTypes<'ctx>,

    /// Struct type to store values for all child ports that are otherwise not connected to anything,
    ///   they will need a distinct place to point to.
    pub dummy_state_ty: StructType<'ctx>,
    /// Struct type to store the state for all child modules.
    pub children_state_ty: StructType<'ctx>,

    /// Struct type to store all state belonging to this module. This is a struct with fields:
    /// * `wire_state_ty`
    /// * `dummy_state_ty`
    /// * `children_state_ty`
    pub full_state_ty: StructType<'ctx>,

    /// Function name corresponding to each module process.
    pub process_functions: Vec<ProcessKind<String, String>>,

    /// Information for all child module instances.
    pub child_instances: Vec<LoweredChildModuleInfo>,
}

pub enum ProcessKind<O, L> {
    Combinatorial(O),
    Clocked(L),
}

pub struct LoweredChildModuleInfo {
    pub child_module: IrModule,
    pub connections: Vec<SignalOrDummy>,
}

#[derive(Debug, Copy, Clone)]
pub enum SignalOrDummy {
    Signal(PortOrWire<usize, usize>),
    Dummy(usize),
}

pub fn lower_module<'ctx, 'map>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    llvm_target: &TargetData,
    modules: &IrModules,
    map: &'map mut IndexMap<IrModule, LoweredModuleInfo<'ctx>>,
    next_module_index: &mut usize,
    module: IrModule,
) -> LowerResult<&'map LoweredModuleInfo<'ctx>> {
    // check cache
    if let Some(idx) = map.get_index_of(&module) {
        return Ok(map.get_index(idx).unwrap().1);
    }

    // claim index
    let module_index = *next_module_index;
    *next_module_index += 1;

    // get info
    let module_info = &modules[module];
    let IrModuleInfo {
        signals,
        large: _,
        children,
        debug_info_def_file: _,
        debug_info_id: _,
        debug_info_generic_args: _,
    } = module_info;

    // map signal types
    let signal_types = ModuleSignalTypes::new(llvm_ctx, signals)?;

    // visit children, allocating new fields as necessary
    let mut dummy_fields: Vec<BasicTypeEnum> = vec![];
    let mut children_state_fields: Vec<BasicTypeEnum> = vec![];

    let mut process_functions = vec![];
    let mut child_instances = vec![];

    for (child_index, child) in enumerate(children) {
        match &child.inner {
            IrModuleChild::ClockedProcess(proc) => {
                let func_name = format!("module_{module_index}_child_{child_index}_clocked");
                lower_process_clocked(
                    llvm_ctx,
                    llvm_unit,
                    llvm_target,
                    &signal_types,
                    module_info,
                    proc,
                    &func_name,
                )?;
                process_functions.push(ProcessKind::Clocked(func_name));
            }
            IrModuleChild::CombinatorialProcess(proc) => {
                let func_name = format!("module_{module_index}_child_{child_index}_comb");
                lower_process_combinatorial(
                    llvm_ctx,
                    llvm_unit,
                    llvm_target,
                    &signal_types,
                    module_info,
                    proc,
                    &func_name,
                )?;
                process_functions.push(ProcessKind::Combinatorial(func_name));
            }
            IrModuleChild::ModuleInternalInstance(inst) => {
                let &IrModuleInternalInstance {
                    name: _,
                    module: child_module,
                    ref port_connections,
                } = inst;

                let child_module_info = lower_module(
                    llvm_ctx,
                    llvm_unit,
                    llvm_target,
                    modules,
                    map,
                    next_module_index,
                    child_module,
                )?;
                children_state_fields.push(child_module_info.full_state_ty.into());

                let mut connections = Vec::with_capacity(port_connections.len());
                for (conn_index, conn) in enumerate(port_connections) {
                    let result = match conn.inner {
                        IrPortConnection::Input(s) => SignalOrDummy::Signal(signal_types.signal_index(s)),
                        IrPortConnection::Output(s) => match s {
                            Some(s) => SignalOrDummy::Signal(signal_types.signal_index(s)),
                            None => {
                                let port_info = modules[child_module].signals.ports.get_by_index(conn_index).unwrap().1;
                                let ty = &port_info.ty;

                                let dummy_index = dummy_fields.len();
                                dummy_fields.push(lower_ty(llvm_ctx, ty)?);
                                SignalOrDummy::Dummy(dummy_index)
                            }
                        },
                    };
                    connections.push(result);
                }

                child_instances.push(LoweredChildModuleInfo {
                    child_module,
                    connections,
                })
            }
            IrModuleChild::ModuleExternalInstance(_) => todo!(),
        }
    }

    // build result
    let dummy_state_ty = llvm_ctx.struct_type(&dummy_fields, false);
    let children_state_ty = llvm_ctx.struct_type(&children_state_fields, false);
    let full_state_ty = llvm_ctx.struct_type(
        &[
            signal_types.wire_struct_ty.into(),
            dummy_state_ty.into(),
            children_state_ty.into(),
        ],
        false,
    );

    let info = LoweredModuleInfo {
        index: module_index,
        signal_types,
        dummy_state_ty,
        children_state_ty,
        full_state_ty,
        process_functions,
        child_instances,
    };

    // store into cache
    Ok(map.insert_first(module, info))
}

impl LoweredModuleInfo<'_> {
    // TODO cache (or even just hardcode to zero?)
    pub fn state_offset_of_wires(&self, target: &TargetData) -> usize {
        target.offset_of_element(&self.full_state_ty, 0).unwrap() as usize
    }

    pub fn state_offset_of_wire(&self, target: &TargetData, index: usize) -> usize {
        let wire_struct_ty = &self.signal_types.wire_struct_ty;
        let offset_wires = self.state_offset_of_wires(target);
        let offset_wire = target.offset_of_element(wire_struct_ty, index as u32).unwrap() as usize;
        offset_wires + offset_wire
    }

    pub fn state_offset_of_dummy(&self, target: &TargetData, index: usize) -> usize {
        let offset_dummies = target.offset_of_element(&self.full_state_ty, 1).unwrap() as usize;
        let offset_dummy = target.offset_of_element(&self.dummy_state_ty, index as u32).unwrap() as usize;
        offset_dummies + offset_dummy
    }

    pub fn state_offset_of_child(&self, target: &TargetData, index: usize) -> usize {
        let offset_children = target.offset_of_element(&self.full_state_ty, 2).unwrap() as usize;
        let offset_child = target.offset_of_element(&self.children_state_ty, index as u32).unwrap() as usize;
        offset_children + offset_child
    }
}

pub struct ModuleSignalTypes<'ctx> {
    pub port_indices: IndexMap<IrPort, usize>,
    pub wire_indices: IndexMap<IrWire, usize>,
    pub wire_struct_ty: StructType<'ctx>,
}

impl<'ctx> ModuleSignalTypes<'ctx> {
    fn new(llvm_ctx: &'ctx Context, signals: &IrSignals) -> LowerResult<Self> {
        // map ports to indices
        let port_indices: IndexMap<IrPort, usize> = signals.ports.keys().enumerate().map(|(i, p)| (p, i)).collect();

        // map wires to indices
        let wire_indices: IndexMap<IrWire, usize> = signals.wires.keys().enumerate().map(|(i, w)| (w, i)).collect();

        // create wire struct
        let wire_fields = signals
            .wires
            .values()
            .map(|w| lower_ty(llvm_ctx, &w.ty))
            .try_collect_vec()?;
        let wire_struct_ty = llvm_ctx.struct_type(&wire_fields, false);

        Ok(ModuleSignalTypes {
            port_indices,
            wire_indices,
            wire_struct_ty,
        })
    }

    fn signal_index(&self, signal: IrSignal) -> PortOrWire<usize, usize> {
        match signal {
            IrSignal::Port(s) => PortOrWire::Port(*self.port_indices.get(&s).unwrap()),
            IrSignal::Wire(s) => PortOrWire::Wire(*self.wire_indices.get(&s).unwrap()),
        }
    }
}
