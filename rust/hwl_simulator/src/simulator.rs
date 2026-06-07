use crate::buffer::{Buffer, BufferError};
use crate::lower_process::{FunctionCombinatorialProcess, lower_process_comb};
use crate::lower_types::{ModuleSignalTypes, read_value_from_buffer, write_value_to_buffer};
use hwl_language::front::check::{TypeContainsReason, check_type_contains_value};
use hwl_language::front::diagnostic::{DiagResult, Diagnostics};
use hwl_language::front::item::ElaborationArenas;
use hwl_language::front::value::CompileValue;
use hwl_language::mid::ir::{IrModule, IrModuleChild, IrModuleInfo, IrModules, IrPort, IrPorts};
use hwl_language::syntax::ast::PortDirection;
use hwl_language::syntax::pos::{Span, Spanned};
use hwl_language::util::big_int::BigUint;
use indexmap::IndexMap;
use inkwell::OptimizationLevel;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, FunctionLookupError, JitFunction};
use inkwell::module::Module;
use itertools::enumerate;
use std::ffi::c_void;
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

    ports_next: Buffer,
    wires_next: Buffer,
    ports_next_ptrs: Buffer,
}

/// Inner content of [SimulatorModule].
/// Uses [ouroboros] to allow the LLVM state to refer to the 'ctx lifetime, .
#[ouroboros::self_referencing]
struct SimulatorModuleInner {
    llvm_context: Context,

    #[borrows(llvm_context)]
    #[not_covariant]
    inner: SimulatorCompiledReferencing<'this>,
}

/// Inner content of [SimulatorModule].
/// This contains everything except the [Contex], which needs to be outside of this struct for lifetime reasons.
#[allow(dead_code)]
struct SimulatorCompiledReferencing<'ctx> {
    llvm_unit: Module<'ctx>,
    llvm_engine: ExecutionEngine<'ctx>,

    top_module_types: ModuleSignalTypes<'ctx>,
    top_module_ports: IrPorts,
    top_module_ports_named: IndexMap<String, IrPort>,

    all_comb_functions: Vec<JitFunction<'ctx, FunctionCombinatorialProcess>>,
}

/// [SimulatorModule] is an immutable compiled module, should not have any cross-thread synchronization concerns.
unsafe impl Send for SimulatorModule {}
unsafe impl Sync for SimulatorModule {}

impl SimulatorModule {
    pub fn new(modules: &IrModules, top: IrModule) -> LowerResult<SimulatorModule> {
        let builder = SimulatorModuleInnerTryBuilder {
            llvm_context: Context::create(),
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
        let (top_module_types, llvm_engine) = compiled
            .inner
            .with_inner(|inner| (&inner.top_module_types, &inner.llvm_engine));

        let target = llvm_engine.get_target_data();
        let ports_next = Buffer::new(target, &top_module_types.ports_struct_ty)?;
        let wires_next = Buffer::new(target, &top_module_types.wires_struct_ty)?;
        let ports_next_ptrs = Buffer::new(target, &top_module_types.ports_array_ty)?;

        // fill ports_next_ptrs with pointers into ports_next
        for port_index in 0..top_module_types.port_indices.len() {
            let port_ptr_offset = port_index * (target.get_pointer_byte_size(None) as usize);
            let port_offset = target
                .offset_of_element(&top_module_types.ports_struct_ty, port_index as u32)
                .unwrap() as usize;
            unsafe {
                ports_next_ptrs.write(port_ptr_offset, ports_next.as_ptr().add(port_offset));
            }
        }

        let inst = SimulatorInstance {
            module: compiled,
            curr_time: 0,
            ports_next,
            wires_next,
            ports_next_ptrs,
        };
        Ok(inst)
    }

    pub fn module(&self) -> &SimulatorModule {
        &self.module
    }

    pub fn step(&mut self, increment_time: u64) {
        // TODO careful, we're still assuming there is only one module here
        // TODO use actual schedule, then we can get rid of this best-effort loop
        const ITER_COUNT: usize = 16;

        let _ = increment_time;

        let all_comb_functions = self.module.inner.with_inner(|inner| &inner.all_comb_functions);
        for _ in 0..ITER_COUNT {
            for f_comb in all_comb_functions {
                unsafe {
                    f_comb.call(
                        self.ports_next_ptrs.as_ptr() as *mut *mut c_void,
                        self.wires_next.as_ptr(),
                    )
                };
            }
        }
    }

    pub fn get_port(&self, port: IrPort) -> CompileValue {
        // extract inner
        let (top_module_ports, top_module_types, llvm_engine) = self
            .module
            .inner
            .with_inner(|inner| (&inner.top_module_ports, &inner.top_module_types, &inner.llvm_engine));

        // gather port info
        let port_info = &top_module_ports[port];
        let port_index = *top_module_types.port_indices.get(&port).unwrap();

        // figure out port offset
        let target = llvm_engine.get_target_data();
        let port_offset = target
            .offset_of_element(&top_module_types.ports_struct_ty, port_index as u32)
            .unwrap() as usize;

        // actually get value
        unsafe { read_value_from_buffer(&self.ports_next, port_offset, &port_info.ty) }
    }

    pub fn set_port(
        &mut self,
        diags: &Diagnostics,
        elab: &ElaborationArenas,
        port: IrPort,
        value: Spanned<&CompileValue>,
    ) -> DiagResult {
        // extract inner
        let (top_module_ports, top_module_types, llvm_engine) = self
            .module
            .inner
            .with_inner(|inner| (&inner.top_module_ports, &inner.top_module_types, &inner.llvm_engine));

        // gather port info
        let port_info = &top_module_ports[port];
        let port_index = *top_module_types.port_indices.get(&port).unwrap();

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

        // figure out port offset
        let target = llvm_engine.get_target_data();
        let port_offset = target
            .offset_of_element(&top_module_types.ports_struct_ty, port_index as u32)
            .unwrap() as usize;

        // actually set value
        unsafe {
            write_value_to_buffer(&mut self.ports_next, port_offset, &port_info.ty, value.inner);
        }

        Ok(())
    }
}

fn compile_simulator_inner<'ctx>(
    modules: &IrModules,
    top: IrModule,
    llvm_ctx: &'ctx Context,
) -> LowerResult<SimulatorCompiledReferencing<'ctx>> {
    let llvm_unit = llvm_ctx.create_module("");

    let top_module_info = &modules[top];
    let (top_module_types, top_module_functions) = lower_module(llvm_ctx, &llvm_unit, top_module_info, 0)?;

    llvm_unit
        .verify()
        .map_err(|e| LowerError::VerificationFailed(e.to_string()))?;

    let execution_engine = llvm_unit
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| LowerError::FailedToCreateExecutionEngine(e.to_string()))?;

    let mut all_comb_functions = vec![];
    for func_name in top_module_functions.comb {
        let func = unsafe { execution_engine.get_function(&func_name)? };
        all_comb_functions.push(func);
    }

    Ok(SimulatorCompiledReferencing {
        llvm_unit,
        llvm_engine: execution_engine,
        top_module_types,
        top_module_ports: top_module_info.signals.ports.clone(),
        top_module_ports_named: top_module_info.signals.ports_named.clone(),
        all_comb_functions,
    })
}

pub struct ModuleFunctions {
    comb: Vec<String>,
}

fn lower_module<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    info: &IrModuleInfo,
    module_index: usize,
) -> LowerResult<(ModuleSignalTypes<'ctx>, ModuleFunctions)> {
    let IrModuleInfo {
        signals,
        large: _,
        children,
        debug_info_def_file: _,
        debug_info_id,
        debug_info_generic_args: _,
    } = info;

    let module_signal_types = ModuleSignalTypes::new(llvm_ctx, debug_info_id.span, signals)?;
    let mut module_functions_comb = vec![];

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
                lower_process_comb(llvm_ctx, llvm_unit, &module_signal_types, info, proc, &func_name)?;
                module_functions_comb.push(func_name);
            }
            IrModuleChild::ModuleInternalInstance(_) => todo!(),
            IrModuleChild::ModuleExternalInstance(_) => todo!(),
        }
    }

    println!("{}", llvm_unit.to_string());

    let module_funcs = ModuleFunctions {
        comb: module_functions_comb,
    };
    Ok((module_signal_types, module_funcs))
}

pub type LowerResult<T = ()> = Result<T, LowerError>;

// TODO improve this:
//  * separate into internal errors (programming bugs), environment issues (eg. llvm jit failed)
//    and unsupported errors (eg. integer bitwidth too large)
// TODO check that everything properly propagates errors, no silent casting or unwrapping
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
