use crate::buffer::{Buffer, BufferError};
use crate::lower_process::{FunctionCombinatorialProcess, lower_process_comb};
use crate::lower_types::{lower_ty, read_value_from_buffer, write_value_to_buffer};
use hwl_language::front::check::{TypeContainsReason, check_type_contains_value};
use hwl_language::front::diagnostic::{DiagResult, Diagnostics};
use hwl_language::front::item::ElaborationArenas;
use hwl_language::front::signal::PortOrWire;
use hwl_language::front::value::CompileValue;
use hwl_language::mid::ir::{
    IrModule, IrModuleChild, IrModuleInfo, IrModuleInternalInstance, IrModules, IrPort, IrPortConnection, IrPorts,
    IrSignal, IrSignals, IrWire,
};
use hwl_language::syntax::ast::PortDirection;
use hwl_language::syntax::pos::{Span, Spanned};
use hwl_language::util::big_int::BigUint;
use hwl_language::util::data::IndexMapExt;
use hwl_language::util::iter::IterExt;
use indexmap::IndexMap;
use inkwell::OptimizationLevel;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, FunctionLookupError, JitFunction};
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{BasicTypeEnum, StructType};
use itertools::enumerate;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

/// A fully compiled and prepared simulator module.
/// This does not contain any simulation state and can be reused between multiple [SimulatorInstance] instances.
pub struct SimulatorModule {
    inner: SimulatorModuleInner,
}

/// A simulation instance. This consists of a [SimulatorModule] plus any necessary simulation state.
#[allow(dead_code)]
pub struct SimulatorInstance {
    module: Arc<SimulatorModule>,

    curr_time: u64,
    state_next: Buffer,
}

/// Inner content of [SimulatorModule].
/// Uses [ouroboros] to allow the LLVM state to refer to the 'ctx lifetime, .
#[ouroboros::self_referencing]
struct SimulatorModuleInner {
    llvm_ctx: Context,

    #[borrows(llvm_ctx)]
    #[not_covariant]
    inner: SimulatorCompiledReferencing<'this>,
}

/// Inner content of [SimulatorModule].
/// This contains everything except the [Contex], which needs to be outside of this struct for lifetime reasons.
#[allow(dead_code)]
struct SimulatorCompiledReferencing<'ctx> {
    llvm_unit: Module<'ctx>,
    llvm_engine: ExecutionEngine<'ctx>,

    top_module_ports: IrPorts,
    top_module_ports_named: IndexMap<String, IrPort>,
    top_module_port_indices: IndexMap<IrPort, usize>,

    state_ty: StructType<'ctx>,

    schedule_items: Vec<ScheduleItem<'ctx>>,
    // TODO document, the first n indices are the top ports
    port_offsets: Vec<usize>,
}

/// [SimulatorModule] is an immutable compiled module, should not have any cross-thread synchronization concerns.
unsafe impl Send for SimulatorModule {}
unsafe impl Sync for SimulatorModule {}

impl SimulatorModule {
    pub fn new(modules: &IrModules, top: IrModule) -> LowerResult<SimulatorModule> {
        let builder = SimulatorModuleInnerTryBuilder {
            llvm_ctx: Context::create(),
            inner_builder: |llvm_ctx| compile_simulator_inner(modules, top, llvm_ctx),
        };
        let inner = builder.try_build()?;
        Ok(SimulatorModule { inner })
    }

    pub fn ports(&self) -> &IrPorts {
        self.inner.with_inner(|inner| &inner.top_module_ports)
    }

    pub fn ports_named(&self) -> &IndexMap<String, IrPort> {
        self.inner.with_inner(|inner| &inner.top_module_ports_named)
    }
}

impl SimulatorInstance {
    pub fn new(compiled: Arc<SimulatorModule>) -> LowerResult<SimulatorInstance> {
        let state_next = compiled.inner.with_inner(|inner| {
            // TODO fill buffers with X for each signal and register, to ensure a clean start
            let target = inner.llvm_engine.get_target_data();
            Buffer::new(target, &inner.state_ty)
        })?;

        let inst = SimulatorInstance {
            module: compiled,
            curr_time: 0,
            state_next,
        };
        Ok(inst)
    }

    pub fn module(&self) -> &SimulatorModule {
        &self.module
    }

    pub fn step(&mut self, increment_time: u64) {
        // // TODO careful, we're still assuming there is only one module here
        // // TODO use actual schedule, then we can get rid of this best-effort loop
        const ITER_COUNT: usize = 16;

        let _ = increment_time;

        self.module.inner.with_inner(|inner| {
            for _ in 0..ITER_COUNT {
                for item in &inner.schedule_items {
                    let &ScheduleItem {
                        ref func,
                        start_offsets_instance_ports,
                        offset_instance_wires,
                    } = item;

                    let offsets_instance_ports =
                        unsafe { inner.port_offsets.as_ptr().add(start_offsets_instance_ports) };

                    unsafe { func.call(self.state_next.as_ptr(), offsets_instance_ports, offset_instance_wires) };
                }
            }
        });
    }

    pub fn get_port(&self, port: IrPort) -> CompileValue {
        // get port info
        // TODO extract utility function
        let (port_offset, port_ty) = self.module.inner.with_inner(|inner| {
            let port_info = &inner.top_module_ports[port];
            let port_index = *inner.top_module_port_indices.get(&port).unwrap();
            let port_offset = inner.port_offsets[port_index];
            (port_offset, &port_info.ty)
        });

        // actually get value
        unsafe { read_value_from_buffer(&self.state_next, port_offset, port_ty) }
    }

    pub fn set_port(
        &mut self,
        diags: &Diagnostics,
        elab: &ElaborationArenas,
        port: IrPort,
        value: Spanned<&CompileValue>,
    ) -> DiagResult {
        // get port info
        let (port_offset, port_info) = self.module.inner.with_inner(|inner| {
            let port_info = &inner.top_module_ports[port];
            let port_index = *inner.top_module_port_indices.get(&port).unwrap();
            let port_offset = inner.port_offsets[port_index];
            (port_offset, port_info)
        });

        // check direction and type
        match port_info.direction {
            PortDirection::Input => {}
            PortDirection::Output => todo!("err, cannot assign output port"),
        }

        let reason = TypeContainsReason::Assignment {
            span_target: port_info.debug_span,
            span_target_ty: port_info.debug_info_ty.span,
        };
        check_type_contains_value(diags, elab, reason, &port_info.ty.as_type_hw().as_type(), value)?;

        // actually set value
        unsafe {
            write_value_to_buffer(&mut self.state_next, port_offset, &port_info.ty, value.inner);
        }

        Ok(())
    }
}

fn compile_simulator_inner<'ctx>(
    modules: &IrModules,
    top: IrModule,
    llvm_ctx: &'ctx Context,
) -> LowerResult<SimulatorCompiledReferencing<'ctx>> {
    // create unit and get target data layout
    let llvm_unit = llvm_ctx.create_module("");
    let llvm_engine = llvm_unit
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| LowerError::FailedToCreateExecutionEngine(e.to_string()))?;
    let llvm_target = llvm_engine.get_target_data();

    // lower modules
    let mut lowered_map = IndexMap::new();
    let mut next_module_index = 0;
    let lowered_top_info = lower_module(
        llvm_ctx,
        &llvm_unit,
        &llvm_target,
        modules,
        &mut lowered_map,
        &mut next_module_index,
        top,
    )?;

    // compile functions
    llvm_unit
        .verify()
        .map_err(|e| LowerError::VerificationFailed(e.to_string()))?;

    // create top ports type
    let top_info = &modules[top];
    let mut top_port_fields = vec![];
    for port_info in top_info.signals.ports.values() {
        top_port_fields.push(lower_ty(llvm_ctx, &port_info.ty)?);
    }
    let top_ports_ty = llvm_ctx.struct_type(&top_port_fields, false);

    // create top state ty, with fields:
    //   * top ports
    //   * top instance full state
    let state_ty = llvm_ctx.struct_type(&[top_ports_ty.into(), lowered_top_info.full_state_ty.into()], false);
    let offset_top_ports = llvm_target.offset_of_element(&state_ty, 0).unwrap() as usize;
    let offset_top_instance_state = llvm_target.offset_of_element(&state_ty, 1).unwrap() as usize;

    // fill in top port offsets
    let mut result_port_offsets = vec![];
    let start_offsets_top_instance_ports = result_port_offsets.len();
    for port_index in 0..top_info.signals.ports.len() {
        let offset_port = llvm_target.offset_of_element(&top_ports_ty, port_index as u32).unwrap() as usize;
        result_port_offsets.push(offset_top_ports + offset_port);
    }

    // clone out some information to avoid borrow issues later
    let top_module_port_indices = lowered_top_info.signal_types.port_indices.clone();

    // build schedule
    let mut result_schedule = vec![];
    build_schedule_and_offsets(
        llvm_ctx,
        llvm_target,
        &llvm_engine,
        &lowered_map,
        top,
        &mut result_schedule,
        &mut result_port_offsets,
        offset_top_instance_state,
        start_offsets_top_instance_ports,
    );

    Ok(SimulatorCompiledReferencing {
        llvm_unit,
        llvm_engine,
        top_module_ports: top_info.signals.ports.clone(),
        top_module_ports_named: top_info.signals.ports_named.clone(),
        top_module_port_indices,
        state_ty,
        schedule_items: result_schedule,
        port_offsets: result_port_offsets,
    })
}

struct LoweredModuleInfo<'ctx> {
    index: usize,

    /// Module signal type info, including the struct type to store all wires declared in this module
    signal_types: ModuleSignalTypes<'ctx>,

    /// Struct type to store values for all child ports that are otherwise not connected to anything,
    ///   they will need a distinct place to point to.
    dummy_state_ty: StructType<'ctx>,
    /// Struct type to store the state for all child modules.
    children_state_ty: StructType<'ctx>,

    /// Struct type to store all state belonging to this module. This is a struct with fields:
    /// * `wire_state_ty`
    /// * `dummy_state_ty`
    /// * `children_state_ty`
    full_state_ty: StructType<'ctx>,

    /// Information for all child module instances
    child_instances: Vec<LoweredChildModuleInfo>,

    /// Combinatorial process functions
    functions_comb: Vec<String>,
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

struct LoweredChildModuleInfo {
    child_module: IrModule,
    connections: Vec<SignalOrDummy>,
}

#[derive(Debug, Copy, Clone)]
enum SignalOrDummy {
    Signal(PortOrWire<usize, usize>),
    Dummy(usize),
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

// TODO move this to lower_module, instead of top-level in the simulator
fn lower_module<'ctx, 'map>(
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
        debug_info_id,
        debug_info_generic_args: _,
    } = module_info;

    // map signal types
    let signal_types = ModuleSignalTypes::new(llvm_ctx, signals)?;

    // visit children, allocating new fields as necessary
    let mut dummy_fields: Vec<BasicTypeEnum> = vec![];
    let mut children_state_fields: Vec<BasicTypeEnum> = vec![];
    let mut child_instances = vec![];

    let mut functions_comb = vec![];

    for (child_index, child) in enumerate(children) {
        match &child.inner {
            IrModuleChild::ClockedProcess(_) => {
                // TODO be careful about prev/next,
                //   writes to registers should immediately be visible in the process itself
                //   or do we want to flip IR semantics back again?
                //   Actually, maybe we can just read copy prev to next for all driven registers,
                //      then always read from next for those, and read from prev for other values?
                todo!()
            }
            IrModuleChild::CombinatorialProcess(proc) => {
                let func_name = format!("module_{module_index}_child_{child_index}_comb");
                lower_process_comb(
                    llvm_ctx,
                    llvm_unit,
                    llvm_target,
                    &signal_types,
                    module_info,
                    proc,
                    &func_name,
                )?;
                functions_comb.push(func_name);
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
        child_instances,
        functions_comb,
    };

    // store into cache
    Ok(map.insert_first(module, info))
}

struct ScheduleItem<'ctx> {
    // TODO try replacing this with a raw function pointer, this contains a bunch of redundant arcs
    func: JitFunction<'ctx, FunctionCombinatorialProcess>,
    start_offsets_instance_ports: usize,
    offset_instance_wires: usize,
}

// TODO rename?
fn build_schedule_and_offsets<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_target: &TargetData,
    llvm_engine: &ExecutionEngine<'ctx>,
    map: &IndexMap<IrModule, LoweredModuleInfo<'ctx>>,
    module: IrModule,

    result: &mut Vec<ScheduleItem<'ctx>>,
    result_offsets_instance_ports: &mut Vec<usize>,

    offset_instance_state: usize,
    start_offsets_instance_ports: usize,
) {
    let info = map.get(&module).unwrap();

    // record module processes with the right offsets
    let offset_instance_wires = offset_instance_state + info.state_offset_of_wires(llvm_target);
    for func in &info.functions_comb {
        let func = unsafe { llvm_engine.get_function::<FunctionCombinatorialProcess>(func).unwrap() };

        let item = ScheduleItem {
            func,
            start_offsets_instance_ports,
            offset_instance_wires,
        };
        result.push(item)
    }

    // visit the children, with the right offsets
    for (child_instance_index, child_instance) in enumerate(&info.child_instances) {
        let &LoweredChildModuleInfo {
            child_module,
            ref connections,
        } = child_instance;

        // build child instance port offsets
        let child_start_offsets_instance_ports = result_offsets_instance_ports.len();
        for &conn in connections {
            let conn_offset = match conn {
                SignalOrDummy::Signal(signal) => match signal {
                    PortOrWire::Port(index) => result_offsets_instance_ports[start_offsets_instance_ports + index],
                    PortOrWire::Wire(index) => offset_instance_state + info.state_offset_of_wire(llvm_target, index),
                },
                SignalOrDummy::Dummy(index) => offset_instance_state + info.state_offset_of_dummy(llvm_target, index),
            };
            result_offsets_instance_ports.push(conn_offset);
        }

        // visit child
        let child_offset_instance_state =
            offset_instance_state + info.state_offset_of_child(llvm_target, child_instance_index);
        build_schedule_and_offsets(
            llvm_ctx,
            llvm_target,
            llvm_engine,
            map,
            child_module,
            result,
            result_offsets_instance_ports,
            child_offset_instance_state,
            child_start_offsets_instance_ports,
        );
    }
}

pub type LowerResult<T = ()> = Result<T, LowerError>;

// TODO improve this:
//  * separate into internal errors (programming bugs), environment issues (eg. llvm jit failed)
//    and unsupported errors (eg. integer bitwidth too large)
// TODO check that everything properly propagates errors, no silent casting or unwrapping
// TODO go through and check that we never do unsafe int casts
#[derive(Debug)]
pub enum LowerError {
    BuilderError(BuilderError),
    VerificationFailed(String),
    FailedToCreateExecutionEngine(String),
    FunctionLookupError(FunctionLookupError),
    IntTooLarge(Span, BigUint),
    BufferError(BufferError),
}

impl From<BuilderError> for LowerError {
    fn from(value: BuilderError) -> Self {
        LowerError::BuilderError(value)
    }
}

impl From<FunctionLookupError> for LowerError {
    fn from(value: FunctionLookupError) -> Self {
        LowerError::FunctionLookupError(value)
    }
}

impl From<BufferError> for LowerError {
    fn from(value: BufferError) -> Self {
        LowerError::BufferError(value)
    }
}

impl Display for LowerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::BuilderError(e) => write!(f, "builder error: {e}"),
            LowerError::VerificationFailed(e) => write!(f, "verification failed: {e}"),
            LowerError::FailedToCreateExecutionEngine(e) => write!(f, "failed to create execution engine: {e}"),
            LowerError::FunctionLookupError(e) => {
                write!(f, "function lookup error: {e}")
            }
            LowerError::IntTooLarge(_, i) => write!(f, "int too large: {i}"),
            LowerError::BufferError(e) => write!(f, "buffer error: {e:?}"),
        }
    }
}
