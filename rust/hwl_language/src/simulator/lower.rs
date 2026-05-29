use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrIfStatement, IrModule,
    IrModuleChild, IrModuleInfo, IrModules, IrSignalOrVariable, IrStatement, IrType, IrVariable, IrVariables,
};
use crate::mid::steps::IrTargetSteps;
use crate::try_inner;
use crate::util::ResultExt;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use inkwell::OptimizationLevel;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::BasicTypeEnum;
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use std::num::NonZeroU32;

// TODO dedicated error enum
// TODO should be propagate build results or just unwrap them? these are always implementation errors, not user errors
pub fn lower_simulator(modules: &IrModules, top: IrModule) -> BuildResult<()> {
    // TODO tree walking
    // TODO parallelize compilation
    // TODO optimize
    lower_module(&modules[top])?;

    Ok(())
}

fn lower_module(info: &IrModuleInfo) -> BuildResult<()> {
    println!("{:#?}", info);

    let IrModuleInfo {
        signals,
        large,
        children,
        debug_info_def_file: _,
        debug_info_id: _,
        debug_info_generic_args: _,
    } = info;

    for child in children {
        match &child.inner {
            IrModuleChild::ClockedProcess(_) => todo!(),
            IrModuleChild::CombinatorialProcess(proc) => {
                lower_process_comb(info, proc)?;
            }
            IrModuleChild::ModuleInternalInstance(_) => todo!(),
            IrModuleChild::ModuleExternalInstance(_) => todo!(),
        }
    }

    Ok(())
}

fn lower_process_comb(info: &IrModuleInfo, proc: &IrCombinatorialProcess) -> BuildResult<()> {
    let IrCombinatorialProcess { variables, block } = proc;

    // TODO better names (all "NAME" instances)
    // TODO measure context creation cost, does it make sense to share it between multiple processes?
    let ctx = Context::create();

    let func_name = "process_comb_func";
    let builder = ProcessBuilder::new(info, variables, &ctx, "NAME", func_name)?;

    // TODO check event scheduling reason here? or will we keep that higher-level?
    builder.lower_block(block)?;
    builder.llvm_builder.build_return(None)?;

    println!("{}", builder.llvm_module.to_string());

    let execution_engine = builder
        .llvm_module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();

    let func = unsafe {
        type Func = unsafe extern "C" fn();
        execution_engine.get_function::<Func>(func_name).unwrap()
    };

    // TODO stop calling stuff here
    unsafe {
        func.call();
    }

    // TODO actually return something
    Ok(())
}

type BuildResult<T> = Result<T, BuilderError>;

struct ProcessBuilder<'ctx, 'ir> {
    // ir
    ir_module: &'ir IrModuleInfo,
    ir_vars: &'ir IrVariables,

    // llvm
    llvm_context: &'ctx Context,
    llvm_module: Module<'ctx>,
    llvm_builder: Builder<'ctx>,

    // state
    function: FunctionValue<'ctx>,
    ir_var_to_alloca: IndexMap<IrVariable, Result<PointerValue<'ctx>, ZeroSize>>,
}

#[derive(Debug, Copy, Clone)]
struct ZeroSize;

struct MappedSteps {
    gep: (),
    slice_len: NonZeroU32,
}

impl<'ctx, 'ir> ProcessBuilder<'ctx, 'ir> {
    fn new(
        ir_module: &'ir IrModuleInfo,
        ir_vars: &'ir IrVariables,
        llvm_context: &'ctx Context,
        llvm_module_name: &str,
        llvm_function_name: &str,
    ) -> BuildResult<ProcessBuilder<'ctx, 'ir>> {
        let llvm_module = llvm_context.create_module(llvm_module_name);
        let llvm_builder = llvm_context.create_builder();

        // create function and entry
        // TODO add some args:
        //   * global context (eg. containing callbacks for print and asserts)
        //   * for clocked blocks: pointers to prev and next state, all including no-alias attributes
        //   * for comb blocks: curr state
        let fn_type = llvm_context.i8_type().fn_type(&[llvm_context.i8_type().into()], false);
        let function = llvm_module.add_function(llvm_function_name, fn_type, None);
        let entry_block = llvm_context.append_basic_block(function, "entry");
        llvm_builder.position_at_end(entry_block);

        // allocate variables
        let mut ir_var_to_alloca = IndexMap::new();
        for (var, var_info) in ir_vars {
            let alloca = match map_ty(llvm_context, &var_info.ty) {
                Ok(ty) => Ok(llvm_builder.build_alloca(ty, "")?),
                Err(ZeroSize) => Err(ZeroSize),
            };
            ir_var_to_alloca.insert_first(var, alloca);
        }

        Ok(ProcessBuilder {
            ir_module,
            ir_vars,

            llvm_context,
            llvm_module,
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

        let (base, value) = match (base, value) {
            (Ok(base), Ok(value)) => (base, value),
            (Err(ZeroSize), Err(ZeroSize)) => return Ok(()),
            _ => unreachable!(),
        };

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
        let condition = self.eval_expr(condition)?.unwrap().into_int_value();

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

    fn eval_expr(&self, value: &IrExpression) -> BuildResult<Result<BasicValueEnum<'ctx>, ZeroSize>> {
        let result: BasicValueEnum = match *value {
            IrExpression::Bool(value) => self.llvm_context.bool_type().const_int(value as u64, false).into(),
            IrExpression::Int(_) => todo!(),
            IrExpression::Signal(_) => todo!(),
            IrExpression::Variable(var) => {
                let ty = try_inner!(self.map_ty(var.ty(&self.ir_vars)));
                let ptr = try_inner!(self.get_named_ptr(var));
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
                    let inner = try_inner!(self.eval_expr(inner)?).into_int_value();
                    self.llvm_builder.build_not(inner, "")?.into()
                }
                _ => todo!("{value:?}"),
            },
        };

        Ok(Ok(result))
    }

    fn eval_steps(&self, ty: &IrType, steps: &IrTargetSteps) -> BuildResult<Result<MappedSteps, ZeroSize>> {
        todo!()
    }

    fn get_named_ptr(&self, target: impl Into<IrSignalOrVariable>) -> Result<PointerValue<'ctx>, ZeroSize> {
        match target.into() {
            IrSignalOrVariable::Signal(_) => todo!(),
            IrSignalOrVariable::Variable(var) => *self.ir_var_to_alloca.get(&var).unwrap(),
        }
    }

    fn map_ty(&self, ty: &IrType) -> Result<BasicTypeEnum<'ctx>, ZeroSize> {
        map_ty(self.llvm_context, ty)
    }
}

fn map_ty<'ctx>(ctx: &'ctx Context, ty: &IrType) -> Result<BasicTypeEnum<'ctx>, ZeroSize> {
    // TODO cache these? maybe that's even necessary for struct types
    match ty {
        IrType::Bool => Ok(BasicTypeEnum::IntType(ctx.bool_type())),
        IrType::Int(_) => todo!(),
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}

trait ResultZeroExt {
    type T;
    fn unwrap_non_zero_size(self) -> Self::T;
}

impl<T> ResultZeroExt for Result<T, ZeroSize> {
    type T = T;

    fn unwrap_non_zero_size(self) -> T {
        self.unwrap()
    }
}
