use hwl_language::front::check::{TypeContainsReason, check_type_contains_value};
use hwl_language::front::diagnostic::{DiagResult, Diagnostics};
use hwl_language::front::item::ElaborationArenas;
use hwl_language::front::value::{CompileValue, SimpleCompileValue};
use hwl_language::mid::ir::{
    IrAssignmentTarget, IrBlock, IrBoolBinaryOp, IrCombinatorialProcess, IrExpression, IrExpressionLarge,
    IrIfStatement, IrModule, IrModuleChild, IrModuleInfo, IrModules, IrPort, IrPorts, IrSignal, IrSignalOrVariable,
    IrSignals, IrStatement, IrType, IrVariable, IrVariables, IrWire,
};
use hwl_language::mid::steps::IrTargetSteps;
use hwl_language::syntax::ast::PortDirection;
use hwl_language::syntax::pos::{Span, Spanned};
use hwl_language::util::big_int::BigUint;
use hwl_language::util::data::IndexMapExt;
use indexmap::IndexMap;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, FunctionLookupError, JitFunction};
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{AnyType, ArrayType, BasicTypeEnum, FunctionType, StructType};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use itertools::enumerate;
use std::alloc::{self, Layout};
use std::ffi::c_void;
use std::fmt::{self, Display, Formatter};
use std::ptr::NonNull;
use std::sync::Arc;
use unwrap_match::unwrap_match;

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
        match &port_info.ty {
            IrType::Bool => {
                let value = unsafe { self.ports_next.read::<u8>(port_offset) };

                match value {
                    0 => CompileValue::new_bool(false),
                    1 => CompileValue::new_bool(true),
                    _ => todo!("internal err"),
                }
            }
            IrType::Int(_) => todo!(),
            IrType::Array(_, _) => todo!(),
            IrType::Tuple(_) => todo!(),
            IrType::Struct(_) => todo!(),
            IrType::Enum(_) => todo!(),
        }
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
        match &port_info.ty {
            IrType::Bool => {
                let value = unwrap_match!(value.inner, &CompileValue::Simple(SimpleCompileValue::Bool(value)) => value);
                unsafe { self.ports_next.write::<u8>(port_offset, value as u8) };
            }
            IrType::Int(_) => todo!(),
            IrType::Array(_, _) => todo!(),
            IrType::Tuple(_) => todo!(),
            IrType::Struct(_) => todo!(),
            IrType::Enum(_) => todo!(),
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

    let module_signal_types = build_module_signal_types(llvm_ctx, debug_info_id.span, signals)?;
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

#[derive(Debug)]
struct ModuleSignalTypes<'ctx> {
    port_indices: IndexMap<IrPort, usize>,
    wire_indices: IndexMap<IrWire, usize>,

    /// Array of (untyped) pointers, one per port.
    /// Used for module instances, where each port is a pointer to the right signal in the parent.
    ports_array_ty: ArrayType<'ctx>,
    /// Struct type, one field per port.
    /// Used to store top-level ports.
    ports_struct_ty: StructType<'ctx>,
    /// Struct type, one field per wire.
    /// Used to store module instance wires.
    wires_struct_ty: StructType<'ctx>,
}

fn build_module_signal_types<'ctx>(
    ctx: &'ctx Context,
    span: Span,
    signals: &IrSignals,
) -> Result<ModuleSignalTypes<'ctx>, LowerError> {
    let IrSignals {
        ports,
        wires,
        ports_named: _,
    } = signals;

    // map ports
    let mut port_indices = IndexMap::new();
    let mut port_types = vec![];
    for (port, port_info) in ports {
        port_indices.insert_first(port, port_indices.len());

        let ty = map_ty(ctx, &port_info.ty);
        port_types.push(ty);
    }
    let ports_struct_ty = ctx.struct_type(&port_types, false);
    let ports_array_ty = ctx
        .ptr_type(AddressSpace::default())
        .array_type(usize_to_u31(span, port_indices.len())?);

    // map wires
    let mut wire_indices = IndexMap::new();
    let mut wire_types = vec![];
    for (wire, wire_info) in wires {
        wire_indices.insert_first(wire, wire_indices.len());

        let ty = map_ty(ctx, &wire_info.ty);
        wire_types.push(ty);
    }
    let wires_struct_ty = ctx.struct_type(&wire_types, false);

    Ok(ModuleSignalTypes {
        port_indices,
        wire_indices,
        ports_array_ty,
        ports_struct_ty,
        wires_struct_ty,
    })
}

/// Type of combinatorial process functions
type FunctionCombinatorialProcess = unsafe extern "C" fn(next_ports: *mut *mut c_void, next_wires: *mut c_void) -> ();
fn build_function_type_combinatorial_process(llvm_ctx: &Context) -> FunctionType<'_> {
    let ty_ptr = llvm_ctx.ptr_type(AddressSpace::default());
    llvm_ctx.void_type().fn_type(&[ty_ptr.into(), ty_ptr.into()], false)
}

fn lower_process_comb<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    module_signal_types: &ModuleSignalTypes<'ctx>,
    ir_module: &IrModuleInfo,
    ir_proc: &IrCombinatorialProcess,
    func_name: &str,
) -> LowerResult<FunctionValue<'ctx>> {
    let IrCombinatorialProcess { variables, block } = ir_proc;

    assert!(llvm_unit.get_function(&func_name).is_none());
    let ty_func = build_function_type_combinatorial_process(llvm_ctx);
    let function = llvm_unit.add_function(&func_name, ty_func, None);

    let param_next_ports = function.get_nth_param(0).unwrap();
    param_next_ports.set_name("next_ports");
    let param_next_ports = param_next_ports.into_pointer_value();

    let param_next_wires = function.get_nth_param(1).unwrap();
    param_next_wires.set_name("next_wires");
    let param_next_wires = param_next_wires.into_pointer_value();

    let builder = ProcessBuilder::new(
        ir_module,
        variables,
        llvm_ctx,
        module_signal_types,
        function,
        param_next_ports,
        param_next_wires,
    )?;

    builder.lower_block(block)?;
    builder.llvm_builder.build_return(None)?;

    Ok(function)
}

struct ProcessBuilder<'ir, 'ctx, 't> {
    // ir
    ir_module: &'ir IrModuleInfo,
    ir_vars: &'ir IrVariables,

    // llvm
    llvm_context: &'ctx Context,
    llvm_builder: Builder<'ctx>,

    // state
    function: FunctionValue<'ctx>,
    ir_var_to_alloca: IndexMap<IrVariable, PointerValue<'ctx>>,
    module_signal_types: &'t ModuleSignalTypes<'ctx>,

    param_next_ports: PointerValue<'ctx>,
    param_next_wires: PointerValue<'ctx>,
}

struct MappedSteps {
    // gep: (),
    // slice_len: NonZeroU32,
}

impl<'ir, 'ctx, 't> ProcessBuilder<'ir, 'ctx, 't> {
    fn new(
        ir_module: &'ir IrModuleInfo,
        ir_vars: &'ir IrVariables,
        llvm_ctx: &'ctx Context,
        module_signal_types: &'t ModuleSignalTypes<'ctx>,
        function: FunctionValue<'ctx>,
        param_next_ports: PointerValue<'ctx>,
        param_next_wires: PointerValue<'ctx>,
    ) -> LowerResult<ProcessBuilder<'ir, 'ctx, 't>> {
        // create entry block
        let entry_basic = llvm_ctx.append_basic_block(function, "entry");
        let llvm_builder = llvm_ctx.create_builder();
        llvm_builder.position_at_end(entry_basic);

        // allocate variables
        let mut ir_var_to_alloca = IndexMap::new();
        for (var, var_info) in ir_vars {
            let ty = map_ty(llvm_ctx, &var_info.ty);
            let alloca = llvm_builder.build_alloca(ty, "")?;
            ir_var_to_alloca.insert_first(var, alloca);
        }

        Ok(ProcessBuilder {
            ir_module,
            ir_vars,

            llvm_context: llvm_ctx,
            llvm_builder,

            function,
            ir_var_to_alloca,
            module_signal_types,
            param_next_ports,
            param_next_wires,
        })
    }

    fn lower_block(&self, block: &IrBlock) -> LowerResult<BasicBlock<'ctx>> {
        let IrBlock { statements } = block;
        for stmt in statements {
            let stmt_span = stmt.span;

            match &stmt.inner {
                IrStatement::Assign(target, value) => {
                    self.lower_statement_assign(stmt_span, target, value)?;
                }
                IrStatement::Block(_) => todo!(),
                IrStatement::If(stmt) => self.lower_if_statement(stmt_span, stmt)?,
                IrStatement::For(_) => todo!(),
                IrStatement::Print(_) => {
                    // TODO actually print stuff
                    println!("warning: skipped print")
                }
                IrStatement::AssertFailed => {
                    // TODO actually fail assertion
                    println!("warning: skipped assert failed")
                }
            }
        }

        // block lowering must leave the builder in a valid block that has not yet been terminated
        // we return that block here for caller convenience
        let block_end = self.llvm_builder.get_insert_block().unwrap();
        assert!(block_end.get_terminator().is_none());
        Ok(block_end)
    }

    fn lower_statement_assign(&self, span: Span, target: &IrAssignmentTarget, value: &IrExpression) -> LowerResult {
        let &IrAssignmentTarget { base, ref steps } = target;
        if !steps.is_empty() {
            // TODO implement steps
            //   for final slice, emit for loop or can just a large array type handle it?
            let base_ty = base.ty(&self.ir_module.signals, self.ir_vars);
            let _ = self.eval_steps(base_ty, steps)?;
            todo!()
        }

        let base = self.get_named_ptr(span, base)?;
        let value = self.eval_expr(span, value)?;
        self.llvm_builder.build_store(base, value)?;

        Ok(())
    }

    fn lower_if_statement(&self, span: Span, stmt: &IrIfStatement) -> LowerResult {
        let IrIfStatement {
            condition,
            then_block,
            else_block,
        } = stmt;

        // eval condition
        let condition = self.eval_expr(span, condition)?.into_int_value();

        // remember start block
        let bb_cond_end = self.llvm_builder.get_insert_block().unwrap();
        self.llvm_builder.clear_insertion_position();

        // lower then branch
        let bb_then_start = self.llvm_context.append_basic_block(self.function, "if.then");
        self.llvm_builder.position_at_end(bb_then_start);
        let bb_then_end = self.lower_block(then_block)?;

        // lower else branch
        let bb_else_start_end = if let Some(else_block) = else_block {
            let bb_else_start = self.llvm_context.append_basic_block(self.function, "if.else");
            self.llvm_builder.position_at_end(bb_else_start);
            let bb_else_end = self.lower_block(else_block)?;
            Some((bb_else_start, bb_else_end))
        } else {
            None
        };

        // connect branches
        let bb_end = self.llvm_context.append_basic_block(self.function, "if.end");

        self.llvm_builder.position_at_end(bb_cond_end);
        self.llvm_builder.build_conditional_branch(
            condition,
            bb_then_start,
            bb_else_start_end.map_or(bb_end, |(bb_else_start, _)| bb_else_start),
        )?;

        self.llvm_builder.position_at_end(bb_then_end);
        self.llvm_builder.build_unconditional_branch(bb_end)?;

        if let Some((_, bb_else_end)) = bb_else_start_end {
            self.llvm_builder.position_at_end(bb_else_end);
            self.llvm_builder.build_unconditional_branch(bb_end)?;
        }

        self.llvm_builder.position_at_end(bb_end);
        Ok(())
    }

    fn eval_expr(&self, span: Span, value: &IrExpression) -> LowerResult<BasicValueEnum<'ctx>> {
        let result: BasicValueEnum = match *value {
            IrExpression::Bool(value) => self.llvm_context.bool_type().const_int(value as u64, false).into(),
            IrExpression::Int(_) => todo!(),
            IrExpression::Signal(sig) => {
                let ty = self.map_ty(sig.ty(&self.ir_module.signals));
                let ptr = self.get_named_ptr(span, sig)?;
                self.llvm_builder.build_load(ty, ptr, "")?
            }
            IrExpression::Variable(var) => {
                let ty = self.map_ty(var.ty(self.ir_vars));
                let ptr = self.get_named_ptr(span, var)?;
                self.llvm_builder.build_load(ty, ptr, "")?
            }
            IrExpression::Large(value) => match &self.ir_module.large[value] {
                IrExpressionLarge::Undefined(ty) => {
                    // TODO actually return undef here here
                    if ty != &IrType::Bool {
                        todo!();
                    }

                    self.llvm_context.bool_type().const_zero().into()
                }
                IrExpressionLarge::BoolNot(inner) => {
                    let inner = self.eval_expr(span, inner)?.into_int_value();
                    self.llvm_builder.build_not(inner, "")?.into()
                }
                IrExpressionLarge::BoolBinary(op, lhs, rhs) => {
                    let lhs = self.eval_expr(span, lhs)?.into_int_value();
                    let rhs = self.eval_expr(span, rhs)?.into_int_value();
                    match op {
                        IrBoolBinaryOp::And => self.llvm_builder.build_and(lhs, rhs, "")?.into(),
                        IrBoolBinaryOp::Or => self.llvm_builder.build_or(lhs, rhs, "")?.into(),
                        IrBoolBinaryOp::Xor => self.llvm_builder.build_xor(lhs, rhs, "")?.into(),
                    }
                }
                // TODO for steps, don't just evaluate the base first, immediately apply the steps
                _ => todo!("{value:?}"),
            },
        };

        Ok(result)
    }

    fn eval_steps(&self, ty: &IrType, steps: &IrTargetSteps) -> LowerResult<MappedSteps> {
        let _ = (ty, steps);
        todo!()
    }

    fn get_named_ptr(&self, span: Span, target: impl Into<IrSignalOrVariable>) -> LowerResult<PointerValue<'ctx>> {
        let result = match target.into() {
            IrSignalOrVariable::Signal(sig) => {
                let ty_ptr = self.llvm_context.ptr_type(AddressSpace::default());
                let ty_i32 = self.llvm_context.i32_type();

                match sig {
                    IrSignal::Port(port) => {
                        let port_index = *self.module_signal_types.port_indices.get(&port).unwrap();

                        let gep_indices = &[
                            ty_i32.const_zero(),
                            ty_i32.const_int(usize_to_u31(span, port_index)?.into(), false),
                        ];
                        let port_ptr_ptr = unsafe {
                            self.llvm_builder.build_gep(
                                self.module_signal_types.ports_array_ty,
                                self.param_next_ports,
                                gep_indices,
                                "",
                            )?
                        };

                        self.llvm_builder
                            .build_load(ty_ptr, port_ptr_ptr, "")?
                            .into_pointer_value()
                    }
                    IrSignal::Wire(wire) => {
                        let wire_index = *self.module_signal_types.wire_indices.get(&wire).unwrap();

                        let gep_indices = &[
                            ty_i32.const_zero(),
                            ty_i32.const_int(usize_to_u31(span, wire_index)?.into(), false),
                        ];

                        unsafe {
                            self.llvm_builder.build_gep(
                                self.module_signal_types.wires_struct_ty,
                                self.param_next_wires,
                                gep_indices,
                                "",
                            )?
                        }
                    }
                }
            }
            IrSignalOrVariable::Variable(var) => *self.ir_var_to_alloca.get(&var).unwrap(),
        };
        Ok(result)
    }

    fn map_ty(&self, ty: &IrType) -> BasicTypeEnum<'ctx> {
        map_ty(self.llvm_context, ty)
    }
}

fn map_ty<'ctx>(ctx: &'ctx Context, ty: &IrType) -> BasicTypeEnum<'ctx> {
    // TODO cache these? maybe that's even necessary for struct types
    // TODO optimizations:
    //   for structs and tuples, re-order to minimize size?
    //   for all compound types: bit pack? careful about multithreading!
    match ty {
        IrType::Bool => BasicTypeEnum::IntType(ctx.bool_type()),
        IrType::Int(_) => todo!(),
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}

fn usize_to_u31(span: Span, value: usize) -> LowerResult<u32> {
    if value < i32::MAX as usize {
        Ok(value as u32)
    } else {
        Err(LowerError::IntTooLarge(span, value.into()))
    }
}

pub type LowerResult<T = ()> = Result<T, LowerError>;

#[derive(Debug)]
pub enum LowerError {
    BuilderError(BuilderError),
    VerificationFailed(String),
    FailedToCreateExecutionEngine(String),
    FunctionLookupError(FunctionLookupError),
    IntTooLarge(Span, BigUint),
    BufferError(BufferError),
}

struct Buffer {
    layout: Layout,
    ptr: NonNull<u8>,
}

#[derive(Debug)]
pub enum BufferError {
    InvalidLayout,
    AllocationFailed,
}

impl Buffer {
    fn new(target: &TargetData, ty: &dyn AnyType) -> Result<Buffer, BufferError> {
        let size = target
            .get_store_size(ty)
            .try_into()
            .map_err(|_| BufferError::InvalidLayout)?;
        let align = target
            .get_abi_alignment(ty)
            .try_into()
            .map_err(|_| BufferError::InvalidLayout)?;
        let layout = Layout::from_size_align(size, align).map_err(|_| BufferError::InvalidLayout)?;

        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(BufferError::AllocationFailed)?;

        Ok(Buffer { layout, ptr })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr() as *mut c_void
    }

    unsafe fn read<T>(&self, offset_bytes: usize) -> T {
        unsafe { std::ptr::read::<T>(self.as_ptr().add(offset_bytes) as *const T) }
    }

    unsafe fn write<T>(&self, offset_bytes: usize, value: T) {
        unsafe { std::ptr::write::<T>(self.as_ptr().add(offset_bytes) as *mut T, value) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
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

impl Display for BufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BufferError::InvalidLayout => write!(f, "invalid layout"),
            BufferError::AllocationFailed => write!(f, "allocation failed"),
        }
    }
}
