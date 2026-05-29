use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrCombinatorialProcess, IrExpression, IrModule, IrModuleInfo, IrModules, IrStatement,
    IrType, IrVariable, IrVariables,
};
use crate::mid::steps::IrTargetSteps;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::BasicTypeEnum;
use inkwell::values::PointerValue;
use std::num::NonZeroU32;

pub fn lower_simulator(modules: &IrModules, top: IrModule) {}

fn lower_module(info: &IrModuleInfo) {
    let IrModuleInfo {
        signals,
        large,
        children,
        debug_info_def_file: _,
        debug_info_id: _,
        debug_info_generic_args: _,
    } = info;

    todo!()
}

fn lower_process_comb(info: &IrModuleInfo, proc: &IrCombinatorialProcess) -> BuildResult<()> {
    let IrCombinatorialProcess { variables, block } = proc;

    // TODO better name
    // TODO measure context creation cost, does it make sense to share it between multiple processes?
    let ctx = Context::create();
    let builder = ProcessBuilder::new(info, variables, &ctx, "name")?;

    // TODO check event scheduling reason here? or will we keep that higher-level?
    builder.lower_block(block)?;

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
    ir_var_to_alloca: IndexMap<IrVariable, PointerValue<'ctx>>,
}

#[derive(Debug)]
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
    ) -> BuildResult<ProcessBuilder<'ctx, 'ir>> {
        let mut slf = ProcessBuilder {
            ir_module,
            ir_vars,

            llvm_context,
            llvm_module: llvm_context.create_module(llvm_module_name),
            llvm_builder: llvm_context.create_builder(),

            ir_var_to_alloca: IndexMap::new(),
        };

        // allocate variables
        for (var, var_info) in ir_vars {
            let ty = slf.map_ty(&var_info.ty)?;
            let alloca = slf.llvm_builder.build_alloca(ty, "")?;
            slf.ir_var_to_alloca.insert_first(var, alloca);
        }

        Ok(slf)
    }

    fn lower_block(&self, block: &IrBlock) -> BuildResult<()> {
        let IrBlock { statements } = block;
        for stmt in statements {
            match &stmt.inner {
                IrStatement::Assign(target, value) => {
                    self.lower_statement_assign(target, value)?;
                }
                IrStatement::Block(_) => todo!(),
                IrStatement::If(_) => todo!(),
                IrStatement::For(_) => todo!(),
                IrStatement::Print(_) => todo!(),
                IrStatement::AssertFailed => todo!(),
            }
        }
        Ok(())
    }

    fn lower_statement_assign(&self, target: &IrAssignmentTarget, value: &IrExpression) -> BuildResult<()> {
        let IrAssignmentTarget { base, steps } = target;

        let base_ty = base.ty(&self.ir_module.signals, self.ir_vars);

        // TODO implement steps. For final slice, emit for loop or can just a large array type handle it?
        if !steps.is_empty() {
            todo!()
        }
        // let steps = self.map_steps(base_ty, steps)?;

        // let value

        todo!()
    }

    fn lower_expr(&self, value: &IrExpression) -> BuildResult<Result<(), ZeroSize>> {
        todo!()
    }

    fn map_steps(&self, ty: &IrType, steps: &IrTargetSteps) -> BuildResult<Result<MappedSteps, ZeroSize>> {
        todo!()
    }

    fn map_ty(&self, ty: &IrType) -> BuildResult<BasicTypeEnum<'ctx>> {
        map_ty(self.llvm_context, ty)
    }
}

fn map_ty<'ctx>(ctx: &'ctx Context, ty: &IrType) -> BuildResult<BasicTypeEnum<'ctx>> {
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
