use crate::buffer::{Buffer, BufferError};
use crate::lower_module;
use crate::lower_module::{LoweredChildModuleInfo, LoweredModuleInfo, ProcessKind, SignalOrDummy};
use crate::lower_process::{FunctionProcessClocked, FunctionProcessCombinatorial};
use crate::lower_type::{lower_ty, read_value_from_buffer, write_value_to_buffer};
use hwl_language::front::check::{TypeContainsReason, check_type_contains_value};
use hwl_language::front::diagnostic::{DiagResult, Diagnostics};
use hwl_language::front::item::ElaborationArenas;
use hwl_language::front::signal::PortOrWire;
use hwl_language::front::value::CompileValue;
use hwl_language::mid::ir::{IrModule, IrModules, IrPort, IrPorts};
use hwl_language::syntax::ast::PortDirection;
use hwl_language::syntax::pos::{Span, Spanned};
use hwl_language::util::big_int::BigUint;
use indexmap::IndexMap;
use inkwell::OptimizationLevel;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, FunctionLookupError, JitFunction};
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::StructType;
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
    state_prev: Buffer,
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
        let (target, state_ty) = compiled
            .inner
            .with_inner(|inner| (inner.llvm_engine.get_target_data(), &inner.state_ty));

        // TODO fill buffers with undef for each signal and register, to ensure a clean start
        let state_prev = Buffer::new_zeroed(target, state_ty)?;
        let state_next = Buffer::new_zeroed(target, state_ty)?;

        let inst = SimulatorInstance {
            module: compiled,
            curr_time: 0,
            state_prev,
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

        let callback_print = callback_print as *mut std::ffi::c_void;

        self.module.inner.with_inner(|inner| {
            for _ in 0..ITER_COUNT {
                for item in &inner.schedule_items {
                    let &ScheduleItem {
                        ref func,
                        start_offsets_instance_ports,
                        offset_instance_wires,
                    } = item;

                    unsafe {
                        let offsets_instance_ports = inner.port_offsets.as_ptr().add(start_offsets_instance_ports);
                        match func {
                            ProcessKind::Combinatorial(func) => func.call(
                                callback_print,
                                self.state_next.as_ptr(),
                                offsets_instance_ports,
                                offset_instance_wires,
                            ),
                            ProcessKind::Clocked(func) => func.call(
                                callback_print,
                                self.state_prev.as_ptr(),
                                self.state_next.as_ptr(),
                                offsets_instance_ports,
                                offset_instance_wires,
                            ),
                        }
                    }
                }
            }
        });

        // TODO this copy means that the external user cannot observe signal edges, which is a bit sad?
        //   we do need the copy here to ensure _we_ can see edges, ie. the user modifies and reads next,
        //   and we can compare those
        unsafe {
            self.state_prev.copy_from(&self.state_next);
        }
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

// TODO use print handler
unsafe extern "C" fn callback_print(len: usize, ptr: *const u8) {
    let s = unsafe {
        let buf = std::slice::from_raw_parts(ptr, len);
        std::str::from_utf8_unchecked(buf)
    };
    print!("{s}");
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
    let lowered_top_info = lower_module::lower_module(
        llvm_ctx,
        &llvm_unit,
        &llvm_target,
        modules,
        &mut lowered_map,
        &mut next_module_index,
        top,
    )?;

    println!("{}", llvm_unit.to_string());

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

struct ScheduleItem<'ctx> {
    // TODO try replacing this with a raw function pointer, this contains a bunch of redundant arcs
    func: ProcessKind<JitFunction<'ctx, FunctionProcessCombinatorial>, JitFunction<'ctx, FunctionProcessClocked>>,
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
    for func in &info.process_functions {
        let func = match func {
            ProcessKind::Combinatorial(func) => {
                let func = unsafe { llvm_engine.get_function::<FunctionProcessCombinatorial>(func) };
                ProcessKind::Combinatorial(func.unwrap())
            }
            ProcessKind::Clocked(func) => {
                let func = unsafe { llvm_engine.get_function::<FunctionProcessClocked>(func) };
                ProcessKind::Clocked(func.unwrap())
            }
        };

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
