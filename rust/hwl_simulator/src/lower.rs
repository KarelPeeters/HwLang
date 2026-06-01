use hwl_language::mid::ir::{
    IrAssignmentTarget, IrBlock, IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrIfStatement, IrModule,
    IrModuleChild, IrModuleInfo, IrModules, IrPort, IrSignal, IrSignalOrVariable, IrSignals, IrStatement, IrType,
    IrVariable, IrVariables, IrWire,
};
use hwl_language::mid::steps::IrTargetSteps;
use hwl_language::syntax::pos::Span;
use hwl_language::util::big_int::BigUint;
use hwl_language::util::data::IndexMapExt;
use indexmap::IndexMap;
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::types::{ArrayType, BasicTypeEnum, StructType};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use itertools::enumerate;
use std::sync::Arc;

#[ouroboros::self_referencing]
pub struct SimulatorCompiled {
    llvm_context: Context,

    #[borrows(llvm_context)]
    #[not_covariant]
    inner: SimulatorCompiledInner<'this>,
}

#[allow(dead_code)]
struct SimulatorCompiledInner<'ctx> {
    llvm_unit: Module<'ctx>,
    llvm_execution_engine: ExecutionEngine<'ctx>,
    // TODO store these in a more structured way so we can actually use them
    llvm_functions: Vec<FunctionValue<'ctx>>,
}

#[allow(dead_code)]
pub struct SimulatorInstance {
    compiled: Arc<SimulatorCompiled>,
    // TODO actual state, at least prev/next buffers
}

// TODO basics:
//  * python wrapper for input/output
//  * process sensitivity lists
//  * process scheduling
//  * add global context parameter, containing assertion and print callbacks
//
// TODO optimize:
//  * parallelize module and maybe even process compilation
//  * enable process optimization
//  * fuse sequential processes?
//  * on-disk compilation cache
//  * allow simulator save/restore
pub fn compile_simulator(modules: &IrModules, top: IrModule) -> LowerResult<SimulatorCompiled> {
    // TODO tree walking
    // TODO parallelize compilation
    // TODO optimize
    SimulatorCompiledTryBuilder {
        llvm_context: Context::create(),
        inner_builder: |llvm_ctx| compile_simulator_inner(modules, top, llvm_ctx),
    }
    .try_build()
}

fn compile_simulator_inner<'ctx>(
    modules: &IrModules,
    top: IrModule,
    llvm_ctx: &'ctx Context,
) -> LowerResult<SimulatorCompiledInner<'ctx>> {
    let llvm_unit = llvm_ctx.create_module("");
    lower_module(llvm_ctx, &llvm_unit, &modules[top], 0)?;

    llvm_unit
        .verify()
        .map_err(|e| LowerError::VerificationFailed(e.to_string()))?;

    let execution_engine = llvm_unit
        .create_execution_engine()
        .map_err(|e| LowerError::FailedToCreateExecutionEngine(e.to_string()))?;

    Ok(SimulatorCompiledInner {
        llvm_unit,
        llvm_execution_engine: execution_engine,
        llvm_functions: vec![],
    })
}

fn lower_module<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    info: &IrModuleInfo,
    module_index: usize,
) -> LowerResult {
    let IrModuleInfo {
        signals,
        large: _,
        children,
        debug_info_def_file: _,
        debug_info_id,
        debug_info_generic_args: _,
    } = info;

    let module_signal_types = build_module_signal_types(llvm_ctx, debug_info_id.span, signals)?;

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
                lower_process_comb(
                    llvm_ctx,
                    llvm_unit,
                    &module_signal_types,
                    info,
                    proc,
                    module_index,
                    child_index,
                )?;
            }
            IrModuleChild::ModuleInternalInstance(_) => todo!(),
            IrModuleChild::ModuleExternalInstance(_) => todo!(),
        }
    }

    println!("{}", llvm_unit.to_string());

    Ok(())
}

#[derive(Debug)]
struct ModuleSignalTypes<'ctx> {
    // array of (untyped) pointers, one element per port
    ports_array_ty: ArrayType<'ctx>,
    port_indices: IndexMap<IrPort, usize>,

    // struct, one field per wire
    wires_struct_ty: StructType<'ctx>,
    wire_indices: IndexMap<IrWire, usize>,
}

fn build_module_signal_types<'ctx>(
    ctx: &'ctx Context,
    span: Span,
    signals: &IrSignals,
) -> Result<ModuleSignalTypes<'ctx>, LowerError> {
    let IrSignals { ports, wires, ports_named: _ } = signals;

    // map ports
    let mut port_indices = IndexMap::new();
    for (port, _) in ports {
        port_indices.insert_first(port, port_indices.len());
    }
    let port_array_ty = ctx
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
        ports_array_ty: port_array_ty,
        port_indices,
        wires_struct_ty,
        wire_indices,
    })
}

fn lower_process_comb<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    module_signal_types: &ModuleSignalTypes<'ctx>,
    ir_module: &IrModuleInfo,
    ir_proc: &IrCombinatorialProcess,
    module_index: usize,
    child_index: usize,
) -> LowerResult {
    let IrCombinatorialProcess { variables, block } = ir_proc;

    // signature:
    // * pointer to next ports
    // * pointer to next wires
    let ty_ptr = llvm_ctx.ptr_type(AddressSpace::default());
    let ty_func = llvm_ctx.void_type().fn_type(&[ty_ptr.into(), ty_ptr.into()], false);

    let func_name = format!("module_{module_index}_child_{child_index}_comb");
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

    // TODO actually return something
    Ok(())
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
                        let wire_index = *self.module_signal_types.port_indices.get(&port).unwrap();

                        let gep_indices = &[
                            ty_i32.const_zero(),
                            ty_i32.const_int(usize_to_u31(span, wire_index)?.into(), false),
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
    IntTooLarge(Span, BigUint),
}

impl From<BuilderError> for LowerError {
    fn from(error: BuilderError) -> Self {
        LowerError::BuilderError(error)
    }
}
