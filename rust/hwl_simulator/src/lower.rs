use hwl_language::front::signal::{Port, Wire};
use hwl_language::mid::ir::{
    IrAssignmentTarget, IrBlock, IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrIfStatement, IrModule,
    IrModuleChild, IrModuleInfo, IrModules, IrPort, IrSignalOrVariable, IrSignals, IrStatement, IrType, IrVariable,
    IrVariables, IrWire,
};
use hwl_language::mid::steps::IrTargetSteps;
use hwl_language::syntax::pos::Span;
use hwl_language::util::big_int::BigUint;
use hwl_language::util::data::IndexMapExt;
use indexmap::IndexMap;
use inkwell::AddressSpace;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{ArrayType, BasicTypeEnum, StructType};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use itertools::enumerate;
use std::num::NonZeroU32;

// TODO basics:
//  * python wrapper for input/output
//  * process sensitivity lists
//  * process scheduling
//
// TODO optimize:
//  * parallelize module and maybe even process compilation
//  * enable process optimization
//  * fuse sequential processes?
pub fn lower_simulator(modules: &IrModules, top: IrModule) -> LowerResult {
    // TODO tree walking
    // TODO parallelize compilation
    // TODO optimize
    lower_module(&modules[top], 0)?;

    Ok(())
}

fn lower_module(info: &IrModuleInfo, module_index: usize) -> LowerResult {
    let IrModuleInfo {
        signals,
        large: _,
        children,
        debug_info_def_file: _,
        debug_info_id,
        debug_info_generic_args: _,
    } = info;

    let llvm_ctx = Context::create();
    let llvm_unit = llvm_ctx.create_module("");

    let module_types = create_signal_types(&llvm_ctx, debug_info_id.span, signals)?;

    for (child_index, child) in enumerate(children) {
        match &child.inner {
            IrModuleChild::ClockedProcess(_) => todo!(),
            IrModuleChild::CombinatorialProcess(proc) => {
                lower_process_comb(
                    &llvm_ctx,
                    &llvm_unit,
                    &module_types,
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

fn create_signal_types<'ctx>(
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
    let port_count = port_indices.len();
    let port_array_ty = ctx
        .ptr_type(AddressSpace::default())
        .array_type(u32::try_from(port_count).map_err(|_| LowerError::IntTooLarge(span, port_count.into()))?);

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
        wire_indices: Default::default(),
    })
}

fn lower_process_comb<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    llvm_module_types: &ModuleSignalTypes<'ctx>,
    ir_module: &IrModuleInfo,
    ir_proc: &IrCombinatorialProcess,
    module_index: usize,
    child_index: usize,
) -> BuildResult<()> {
    let IrCombinatorialProcess { variables, block } = ir_proc;

    let func_name = format!("module_{module_index}_child_{child_index}_comb");
    let builder = ProcessBuilder::new(
        llvm_ctx,
        llvm_unit,
        &func_name,
        &llvm_module_types,
        ir_module,
        variables,
    )?;

    builder.lower_block(block)?;
    builder.llvm_builder.build_return(None)?;

    println!("{}", builder.llvm_unit.to_string());

    // let execution_engine = builder
    //     .llvm_module
    //     .create_jit_execution_engine(OptimizationLevel::None)
    //     .unwrap();
    //
    // let func = unsafe {
    //     type Func = unsafe extern "C" fn();
    //     execution_engine.get_function::<Func>(&func_name).unwrap()
    // };
    //
    // // TODO stop calling stuff here
    // unsafe {
    //     func.call();
    // }

    // TODO actually return something
    Ok(())
}

type BuildResult<T> = Result<T, BuilderError>;

struct ProcessBuilder<'ir, 'ctx, 'm> {
    // ir
    ir_module: &'ir IrModuleInfo,
    ir_vars: &'ir IrVariables,

    // llvm
    llvm_context: &'ctx Context,
    llvm_unit: &'m Module<'ctx>,
    llvm_builder: Builder<'ctx>,

    // state
    function: FunctionValue<'ctx>,
    ir_var_to_alloca: IndexMap<IrVariable, PointerValue<'ctx>>,
}

struct MappedSteps {
    gep: (),
    slice_len: NonZeroU32,
}

impl<'ir, 'ctx, 'm> ProcessBuilder<'ir, 'ctx, 'm> {
    fn new(
        llvm_ctx: &'ctx Context,
        llvm_unit: &'m Module<'ctx>,
        llvm_function_name: &str,
        llvm_module_types: &ModuleSignalTypes<'ctx>,
        ir_module: &'ir IrModuleInfo,
        ir_vars: &'ir IrVariables,
    ) -> BuildResult<ProcessBuilder<'ir, 'ctx, 'm>> {
        // create function and entry
        // TODO add some args:
        //   * global context (eg. containing callbacks for print and asserts)
        //   * for clocked blocks: pointers to prev and next state, all including no-alias attributes
        //   * for comb blocks: curr state
        let fn_type = llvm_ctx.void_type().fn_type(&[], false);
        let function = llvm_unit.add_function(llvm_function_name, fn_type, None);
        let entry_block = llvm_ctx.append_basic_block(function, "entry");

        let llvm_builder = llvm_ctx.create_builder();
        llvm_builder.position_at_end(entry_block);

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
            llvm_unit,
            llvm_builder,

            function,
            ir_var_to_alloca,
        })
    }

    fn lower_block(&self, block: &IrBlock) -> BuildResult<()> {
        let IrBlock { statements } = block;
        for stmt in statements {
            match &stmt.inner {
                IrStatement::Assign(target, value) => {
                    self.lower_statement_assign(target, value)?;
                }
                IrStatement::Block(_) => todo!(),
                IrStatement::If(stmt) => self.lower_if_statement(stmt)?,
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
        Ok(())
    }

    fn lower_statement_assign(&self, target: &IrAssignmentTarget, value: &IrExpression) -> BuildResult<()> {
        let &IrAssignmentTarget { base, ref steps } = target;
        if !steps.is_empty() {
            // TODO implement steps
            //   for final slice, emit for loop or can just a large array type handle it?
            let base_ty = base.ty(&self.ir_module.signals, self.ir_vars);
            let _ = self.eval_steps(base_ty, steps)?;
            todo!()
        }

        let base = self.get_named_ptr(base);
        let value = self.eval_expr(value)?;
        self.llvm_builder.build_store(base, value)?;

        Ok(())
    }

    fn lower_if_statement(&self, stmt: &IrIfStatement) -> BuildResult<()> {
        let IrIfStatement {
            condition,
            then_block,
            else_block,
        } = stmt;

        // eval condition
        let condition = self.eval_expr(condition)?.into_int_value();

        // remember start block
        let start_block = self.llvm_builder.get_insert_block().unwrap();

        // lower then branch
        let then_basic = self.llvm_context.append_basic_block(self.function, "if.then");
        self.llvm_builder.position_at_end(then_basic);
        self.lower_block(then_block)?;

        // lower else branch
        let else_basic = if let Some(else_block) = else_block {
            let else_basic = self.llvm_context.append_basic_block(self.function, "if.else");
            self.llvm_builder.position_at_end(else_basic);
            self.lower_block(else_block)?;
            Some(else_basic)
        } else {
            None
        };

        // connect branches
        let end_basic = self.llvm_context.append_basic_block(self.function, "if.end");
        self.llvm_builder.position_at_end(start_block);
        self.llvm_builder
            .build_conditional_branch(condition, then_basic, else_basic.unwrap_or(end_basic))?;

        self.llvm_builder.position_at_end(then_basic);
        self.llvm_builder.build_unconditional_branch(end_basic)?;

        if let Some(else_basic) = else_basic {
            self.llvm_builder.position_at_end(else_basic);
            self.llvm_builder.build_unconditional_branch(end_basic)?;
        }

        self.llvm_builder.position_at_end(end_basic);
        Ok(())
    }

    fn eval_expr(&self, value: &IrExpression) -> BuildResult<BasicValueEnum<'ctx>> {
        let result: BasicValueEnum = match *value {
            IrExpression::Bool(value) => self.llvm_context.bool_type().const_int(value as u64, false).into(),
            IrExpression::Int(_) => todo!(),
            IrExpression::Signal(_) => todo!(),
            IrExpression::Variable(var) => {
                let ty = self.map_ty(var.ty(&self.ir_vars));
                let ptr = self.get_named_ptr(var);
                self.llvm_builder.build_load(ty, ptr.clone(), "")?
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
                    let inner = self.eval_expr(inner)?.into_int_value();
                    self.llvm_builder.build_not(inner, "")?.into()
                }
                _ => todo!("{value:?}"),
            },
        };

        Ok(result)
    }

    fn eval_steps(&self, ty: &IrType, steps: &IrTargetSteps) -> BuildResult<MappedSteps> {
        todo!()
    }

    fn get_named_ptr(&self, target: impl Into<IrSignalOrVariable>) -> PointerValue<'ctx> {
        match target.into() {
            IrSignalOrVariable::Signal(_) => todo!(),
            IrSignalOrVariable::Variable(var) => *self.ir_var_to_alloca.get(&var).unwrap(),
        }
    }

    fn map_ty(&self, ty: &IrType) -> BasicTypeEnum<'ctx> {
        map_ty(self.llvm_context, ty)
    }
}

fn map_ty<'ctx>(ctx: &'ctx Context, ty: &IrType) -> BasicTypeEnum<'ctx> {
    // TODO cache these? maybe that's even necessary for struct types
    match ty {
        IrType::Bool => BasicTypeEnum::IntType(ctx.bool_type()),
        IrType::Int(_) => todo!(),
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}

#[derive(Debug)]
pub enum LowerError {
    BuilderError(BuilderError),
    IntTooLarge(Span, BigUint),
}

pub type LowerResult<T = ()> = Result<T, LowerError>;

impl From<BuilderError> for LowerError {
    fn from(error: BuilderError) -> Self {
        LowerError::BuilderError(error)
    }
}
