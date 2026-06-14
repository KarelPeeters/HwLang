use crate::lower_types::{lower_ty, lower_ty_int, usize_to_u31};
use crate::simulator::{LowerResult, ModuleSignalTypes};
use hwl_language::mid::ir::{
    IrAssignmentTarget, IrBlock, IrBoolBinaryOp, IrCombinatorialProcess, IrExpression, IrExpressionLarge,
    IrIfStatement, IrModuleInfo, IrSignal, IrSignalOrVariable, IrStatement, IrType, IrVariable, IrVariables,
};
use hwl_language::mid::steps::IrTargetSteps;
use hwl_language::syntax::pos::Span;
use hwl_language::util::big_int::BigInt;
use hwl_language::util::data::IndexMapExt;
use hwl_language::util::int_repr::IntRepresentation;
use hwl_language::util::range::ClosedNonEmptyRange;
use indexmap::IndexMap;
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{BasicTypeEnum, FunctionType, IntType};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
use itertools::enumerate;
use std::ffi::c_void;

/// Type of combinatorial process functions
pub type FunctionCombinatorialProcess =
    unsafe extern "C" fn(buffer_next: *mut c_void, ports_offsets: *const usize, wires_offset: usize) -> ();

pub fn build_function_type_combinatorial_process<'ctx>(
    llvm_ctx: &'ctx Context,
    target: &TargetData,
) -> FunctionType<'ctx> {
    let ty_ptr = llvm_ctx.ptr_type(AddressSpace::default());
    let ty_usize = llvm_ctx.ptr_sized_int_type(target, None);

    llvm_ctx
        .void_type()
        .fn_type(&[ty_ptr.into(), ty_ptr.into(), ty_usize.into()], false)
}

pub fn lower_process_comb<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    llvm_target: &TargetData,
    module_types: &ModuleSignalTypes<'ctx>,
    ir_module: &IrModuleInfo,
    ir_proc: &IrCombinatorialProcess,
    func_name: &str,
) -> LowerResult<FunctionValue<'ctx>> {
    let IrCombinatorialProcess { variables, block } = ir_proc;

    // create function
    assert!(llvm_unit.get_function(func_name).is_none());
    let ty_func = build_function_type_combinatorial_process(llvm_ctx, llvm_target);
    let function = llvm_unit.add_function(func_name, ty_func, None);

    // get params
    let param_buffer_next = function.get_nth_param(0).unwrap().into_pointer_value();
    param_buffer_next.set_name("buffer_next");
    let param_ports_offsets = function.get_nth_param(1).unwrap().into_pointer_value();
    param_ports_offsets.set_name("ports_offsets");
    let param_wires_offset = function.get_nth_param(2).unwrap().into_int_value();
    param_wires_offset.set_name("wires_offset");

    // lower block
    let builder = ProcessBuilder::new(
        ir_module,
        variables,
        llvm_ctx,
        llvm_target,
        module_types,
        function,
        param_buffer_next,
        param_ports_offsets,
        param_wires_offset,
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
    llvm_ctx: &'ctx Context,
    llvm_target: &'t TargetData,
    llvm_builder: Builder<'ctx>,

    // module mapping
    module_signal_types: &'t ModuleSignalTypes<'ctx>,

    // process mapping
    function: FunctionValue<'ctx>,
    ir_var_to_alloca: IndexMap<IrVariable, PointerValue<'ctx>>,
    param_buffer_next: PointerValue<'ctx>,
    param_ports_offsets: PointerValue<'ctx>,
    param_wires_offset: IntValue<'ctx>,
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
        llvm_target: &'t TargetData,
        module_signal_types: &'t ModuleSignalTypes<'ctx>,
        function: FunctionValue<'ctx>,
        param_buffer_next: PointerValue<'ctx>,
        param_ports_offsets: PointerValue<'ctx>,
        param_wires_offset: IntValue<'ctx>,
    ) -> LowerResult<ProcessBuilder<'ir, 'ctx, 't>> {
        // create entry block
        let entry_basic = llvm_ctx.append_basic_block(function, "entry");
        let llvm_builder = llvm_ctx.create_builder();
        llvm_builder.position_at_end(entry_basic);

        // allocate variables
        let mut ir_var_to_alloca = IndexMap::new();
        for (var, var_info) in ir_vars {
            let ty = lower_ty(llvm_ctx, &var_info.ty)?;
            let alloca = llvm_builder.build_alloca(ty, "")?;
            ir_var_to_alloca.insert_first(var, alloca);
        }

        Ok(ProcessBuilder {
            ir_module,
            ir_vars,

            llvm_ctx,
            llvm_target,
            llvm_builder,

            module_signal_types,

            function,
            ir_var_to_alloca,
            param_buffer_next,
            param_ports_offsets,
            param_wires_offset,
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
                IrStatement::Block(block) => {
                    self.lower_block(block)?;
                }
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
        let bb_then_start = self.llvm_ctx.append_basic_block(self.function, "if.then");
        self.llvm_builder.position_at_end(bb_then_start);
        let bb_then_end = self.lower_block(then_block)?;

        // lower else branch
        let bb_else_start_end = if let Some(else_block) = else_block {
            let bb_else_start = self.llvm_ctx.append_basic_block(self.function, "if.else");
            self.llvm_builder.position_at_end(bb_else_start);
            let bb_else_end = self.lower_block(else_block)?;
            Some((bb_else_start, bb_else_end))
        } else {
            None
        };

        // connect branches
        let bb_end = self.llvm_ctx.append_basic_block(self.function, "if.end");

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
        let result: BasicValueEnum = match value {
            &IrExpression::Bool(value) => {
                let ty_bool = self.llvm_ctx.bool_type();
                ty_bool.const_int(value as u64, false).into()
            }
            IrExpression::Int(value) => {
                let range = ClosedNonEmptyRange::single(value.clone());
                lower_int_constant(self.llvm_ctx, range.as_ref(), value, span)?.into()
            }
            &IrExpression::Signal(sig) => {
                let ty = self.lower_ty(sig.ty(&self.ir_module.signals))?;
                let ptr = self.get_named_ptr(span, sig)?;
                self.llvm_builder.build_load(ty, ptr, "")?
            }
            &IrExpression::Variable(var) => {
                let ty = self.lower_ty(var.ty(self.ir_vars))?;
                let ptr = self.get_named_ptr(span, var)?;
                self.llvm_builder.build_load(ty, ptr, "")?
            }
            &IrExpression::Large(value) => match &self.ir_module.large[value] {
                // TODO match IR definition order
                IrExpressionLarge::Undefined(ty) => {
                    // TODO actually return undef here here
                    if ty != &IrType::Bool {
                        todo!();
                    }

                    self.llvm_ctx.bool_type().const_zero().into()
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
                IrExpressionLarge::ExpandIntRange(new_range, inner) => {
                    let (_, new_ty) = self.lower_ty_int(new_range.as_ref())?;
                    let inner_is_signed = inner
                        .ty(&self.ir_module.large, &self.ir_module.signals, self.ir_vars)
                        .unwrap_int()
                        .start
                        .is_negative();

                    let inner = self.eval_expr(span, inner)?.into_int_value();

                    if inner_is_signed {
                        self.llvm_builder.build_int_s_extend(inner, new_ty, "")?.into()
                    } else {
                        self.llvm_builder.build_int_z_extend(inner, new_ty, "")?.into()
                    }
                }
                // TODO for steps, don't just evaluate the base first, immediately apply the steps
                value => todo!("{value:?}"),
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
                let ty_i32 = self.llvm_ctx.i32_type();

                match sig {
                    IrSignal::Port(port) => {
                        let port_index = *self.module_signal_types.port_indices.get(&port).unwrap();

                        // calculate port offset address
                        let ty_usize = self.llvm_ctx.ptr_sized_int_type(self.llvm_target, None);
                        let port_index = ty_i32.const_int(usize_to_u31(span, port_index)?.into(), false);
                        let port_offset_ptr = unsafe {
                            self.llvm_builder
                                .build_gep(ty_usize, self.param_ports_offsets, &[port_index], "")?
                        };

                        // load port offset
                        let port_offset = self
                            .llvm_builder
                            .build_load(ty_usize, port_offset_ptr, "")?
                            .into_int_value();

                        // calculate port address from base + offset
                        unsafe {
                            self.llvm_builder.build_gep(
                                self.llvm_ctx.i8_type(),
                                self.param_buffer_next,
                                &[port_offset],
                                "",
                            )?
                        }
                    }
                    IrSignal::Wire(wire) => {
                        let wire_index = *self.module_signal_types.wire_indices.get(&wire).unwrap();

                        // calculate base pointer to the wires of this instance
                        let wires_base = unsafe {
                            self.llvm_builder.build_gep(
                                self.llvm_ctx.i8_type(),
                                self.param_buffer_next,
                                &[self.param_wires_offset],
                                "",
                            )?
                        };

                        // calculate pointer to the wire itself
                        let wire_index = ty_i32.const_int(usize_to_u31(span, wire_index)?.into(), false);
                        unsafe {
                            self.llvm_builder.build_gep(
                                self.module_signal_types.wire_struct_ty,
                                wires_base,
                                &[ty_i32.const_zero(), wire_index],
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

    fn lower_ty(&self, ty: &IrType) -> LowerResult<BasicTypeEnum<'ctx>> {
        lower_ty(self.llvm_ctx, ty)
    }

    fn lower_ty_int(&self, range: ClosedNonEmptyRange<&BigInt>) -> LowerResult<(IntRepresentation, IntType<'ctx>)> {
        lower_ty_int(self.llvm_ctx, range)
    }
}

fn lower_int_constant<'ctx>(
    ctx: &'ctx Context,
    range: ClosedNonEmptyRange<&BigInt>,
    value: &BigInt,
    span: Span,
) -> LowerResult<IntValue<'ctx>> {
    let (repr, ty) = lower_ty_int(ctx, range)?;

    // special case zero-width integer, it gets represented as a single-bit int
    if repr.size_bits() == 0 {
        return Ok(ty.const_zero());
    }

    // TODO this is really inefficient, make this block-wise
    // TODO add fast path for small ints?
    // convert to bits
    let mut bits = Vec::new();
    repr.value_to_bits(value, &mut bits);

    // convert to words
    type W = u64;
    let num_words = bits.len().div_ceil(W::BITS as usize);
    let mut words: Vec<W> = vec![0u64; num_words];

    for (i, bit) in enumerate(bits) {
        if bit {
            words[i / W::BITS as usize] |= 1 << (i % W::BITS as usize);
        }
    }

    // convert to llvm
    Ok(ty.const_int_arbitrary_precision(&words))
}
