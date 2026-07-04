use crate::lower_module::ModuleSignalTypes;
use crate::lower_type::{lower_ty, lower_ty_int, usize_to_u31};
use crate::simulator::LowerResult;
use hwl_language::front::signal::Polarized;
use hwl_language::mid::ir::{
    IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrBoolBinaryOp, IrClockedProcess, IrCombinatorialProcess,
    IrExpression, IrExpressionLarge, IrIfStatement, IrModuleInfo, IrSignal, IrSignalOrVariable, IrStatement,
    IrStringPiece, IrType, IrVariable, IrVariables,
};
use hwl_language::syntax::pos::Span;
use hwl_language::util::big_int::BigInt;
use hwl_language::util::data::IndexMapExt;
use hwl_language::util::int_repr::IntRepresentation;
use hwl_language::util::range::ClosedNonEmptyRange;
use indexmap::{IndexMap, IndexSet};
use inkwell::AddressSpace;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{BasicTypeEnum, IntType};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
use itertools::enumerate;
use std::ffi::c_void;

pub type FunctionProcessCombinatorial = unsafe extern "C" fn(
    callback_print: *mut c_void,
    buffer_next: *mut c_void,
    ports_offsets: *const usize,
    wires_offset: usize,
) -> ();

pub fn lower_process_combinatorial<'ctx>(
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
    let ty_ptr = llvm_ctx.ptr_type(AddressSpace::default());
    let ty_usize = llvm_ctx.ptr_sized_int_type(llvm_target, None);
    let ty_func = llvm_ctx
        .void_type()
        .fn_type(&[ty_ptr.into(), ty_ptr.into(), ty_ptr.into(), ty_usize.into()], false);
    let function = llvm_unit.add_function(func_name, ty_func, None);

    // get params
    let param_callback_print = function.get_nth_param(0).unwrap().into_pointer_value();
    param_callback_print.set_name("callback_print");
    let param_buffer_next = function.get_nth_param(1).unwrap().into_pointer_value();
    param_buffer_next.set_name("buffer_next");
    let param_ports_offsets = function.get_nth_param(2).unwrap().into_pointer_value();
    param_ports_offsets.set_name("ports_offsets");
    let param_wires_offset = function.get_nth_param(3).unwrap().into_int_value();
    param_wires_offset.set_name("wires_offset");

    let param_buffers = ProcessBuffers::Combinatorial {
        next: param_buffer_next,
    };

    // create builder
    let builder = ProcessBuilder::new(
        ir_module,
        variables,
        llvm_ctx,
        llvm_target,
        module_types,
        function,
        param_callback_print,
        param_buffers,
        param_ports_offsets,
        param_wires_offset,
    )?;

    // lower process
    builder.lower_block(block)?;
    builder.llvm_builder.build_return(None)?;

    Ok(function)
}

pub type FunctionProcessClocked = unsafe extern "C" fn(
    callback_print: *mut c_void,
    buffer_prev: *mut c_void,
    buffer_next: *mut c_void,
    ports_offsets: *const usize,
    wires_offset: usize,
) -> ();

pub fn lower_process_clocked<'ctx>(
    llvm_ctx: &'ctx Context,
    llvm_unit: &Module<'ctx>,
    llvm_target: &TargetData,
    module_types: &ModuleSignalTypes<'ctx>,
    ir_module: &IrModuleInfo,
    ir_proc: &IrClockedProcess,
    func_name: &str,
) -> LowerResult<FunctionValue<'ctx>> {
    let IrClockedProcess {
        registers,
        variables,
        async_reset,
        clock_signal,
        clock_block,
    } = ir_proc;

    // create function
    assert!(llvm_unit.get_function(func_name).is_none());
    let ty_ptr = llvm_ctx.ptr_type(AddressSpace::default());
    let ty_usize = llvm_ctx.ptr_sized_int_type(llvm_target, None);
    let ty_func = llvm_ctx.void_type().fn_type(
        &[
            ty_ptr.into(),
            ty_ptr.into(),
            ty_ptr.into(),
            ty_ptr.into(),
            ty_usize.into(),
        ],
        false,
    );
    let function = llvm_unit.add_function(func_name, ty_func, None);

    // get params
    let param_callback_print = function.get_nth_param(0).unwrap().into_pointer_value();
    param_callback_print.set_name("callback_print");
    let param_buffer_prev = function.get_nth_param(1).unwrap().into_pointer_value();
    param_buffer_prev.set_name("buffer_prev");
    let param_buffer_next = function.get_nth_param(2).unwrap().into_pointer_value();
    param_buffer_next.set_name("buffer_next");
    let param_ports_offsets = function.get_nth_param(3).unwrap().into_pointer_value();
    param_ports_offsets.set_name("ports_offsets");
    let param_wires_offset = function.get_nth_param(4).unwrap().into_int_value();
    param_wires_offset.set_name("wires_offset");

    let param_buffers = ProcessBuffers::Clocked {
        registers,
        prev: param_buffer_prev,
        next: param_buffer_next,
    };

    // create builder
    let builder = ProcessBuilder::new(
        ir_module,
        variables,
        llvm_ctx,
        llvm_target,
        module_types,
        function,
        param_callback_print,
        param_buffers,
        param_ports_offsets,
        param_wires_offset,
    )?;

    // create reset header, pseudocode:
    //   reset = eval(next, reset_signal)
    //   if (reset) { /*reset*/; return; }
    if let Some(async_reset) = async_reset {
        let IrAsyncResetInfo {
            signal: reset_signal,
            resets,
        } = async_reset;

        // eval reset
        let reset_value = builder.eval_polarized_signal(param_buffer_next, reset_signal.span, reset_signal.inner)?;

        // build reset if statement
        let build_reset_body = || {
            // reset signals
            for reset in resets {
                let (target, ref source) = reset.inner;
                let target_ptr = builder.signal_write_ptr(reset.span, target)?;
                let source_value = builder.eval_expr(reset.span, source)?;
                builder.llvm_builder.build_store(target_ptr, source_value)?;
            }

            // early return to ensure the rest of the process does not run
            builder.llvm_builder.build_return(None)?;

            // create dummy block to keep the builder in a valid state
            let bb_dummy = builder.llvm_ctx.append_basic_block(builder.function, "dummy");
            builder.llvm_builder.position_at_end(bb_dummy);

            Ok(())
        };

        builder.build_if_statement(reset_value, build_reset_body, None::<fn() -> _>)?;
    }

    // create block header, pseudocode:
    //   clock_edge = !eval(prev, clock_signal) & eval(next, clock_signal)
    //   if (!clock_edge) { return; }
    {
        let clock_prev = builder.eval_polarized_signal(param_buffer_prev, clock_signal.span, clock_signal.inner)?;
        let clock_next = builder.eval_polarized_signal(param_buffer_next, clock_signal.span, clock_signal.inner)?;

        let clock_prev_not = builder.llvm_builder.build_not(clock_prev, "")?;
        let clock_edge = builder.llvm_builder.build_and(clock_prev_not, clock_next, "")?;
        let clock_edge_not = builder.llvm_builder.build_not(clock_edge, "")?;

        // build clock if statement
        let build_not_clock_body = || {
            // early return to ensure the rest of the process does not run
            builder.llvm_builder.build_return(None)?;

            // create dummy block to keep the builder in a valid state
            let bb_dummy = builder.llvm_ctx.append_basic_block(builder.function, "dummy");
            builder.llvm_builder.position_at_end(bb_dummy);
            Ok(())
        };
        builder.build_if_statement(clock_edge_not, build_not_clock_body, None::<fn() -> _>)?;
    }

    // lower clocked block itself
    builder.lower_block(clock_block)?;
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

    param_callback_print: PointerValue<'ctx>,
    param_buffers: ProcessBuffers<'ctx, 't>,
    param_ports_offsets: PointerValue<'ctx>,
    param_wires_offset: IntValue<'ctx>,
}

enum ProcessBuffers<'ctx, 't> {
    Combinatorial {
        next: PointerValue<'ctx>,
    },
    Clocked {
        registers: &'t IndexSet<IrSignal>,
        prev: PointerValue<'ctx>,
        next: PointerValue<'ctx>,
    },
}

impl<'ir, 'ctx, 't> ProcessBuilder<'ir, 'ctx, 't> {
    fn new(
        ir_module: &'ir IrModuleInfo,
        ir_vars: &'ir IrVariables,
        llvm_ctx: &'ctx Context,
        llvm_target: &'t TargetData,
        module_signal_types: &'t ModuleSignalTypes<'ctx>,
        function: FunctionValue<'ctx>,
        param_callback_print: PointerValue<'ctx>,
        param_buffers: ProcessBuffers<'ctx, 't>,
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

            param_callback_print,
            param_buffers,
            param_ports_offsets,
            param_wires_offset,
        })
    }

    fn lower_block(&self, block: &IrBlock) -> LowerResult {
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
                IrStatement::Print(s) => {
                    for p in s {
                        match p {
                            IrStringPiece::Literal(p) => {
                                let ty_usize = self.llvm_ctx.ptr_sized_int_type(self.llvm_target, None);
                                let ty_ptr = self.llvm_ctx.ptr_type(AddressSpace::default());
                                let ty_fn = self
                                    .llvm_ctx
                                    .void_type()
                                    .fn_type(&[ty_usize.into(), ty_ptr.into()], false);

                                let arg_len = ty_usize.const_int(p.len() as u64, false);
                                let arg_data = self.llvm_builder.build_global_string_ptr(p, "")?;
                                self.llvm_builder.build_indirect_call(
                                    ty_fn,
                                    self.param_callback_print,
                                    &[arg_len.into(), arg_data.as_pointer_value().into()],
                                    "",
                                )?;
                            }
                            IrStringPiece::Substitute(_) => {
                                todo!("support string substitution")
                            }
                        }
                    }
                }
                IrStatement::AssertFailed => {
                    // TODO actually fail assertion
                    println!("warning: skipped assert failed")
                }
            }
        }

        // block lowering must leave the builder in a valid block that has not yet been terminated
        let block_end = self.llvm_builder.get_insert_block().unwrap();
        assert!(block_end.get_terminator().is_none());
        Ok(())
    }

    fn lower_statement_assign(&self, span: Span, target: &IrAssignmentTarget, value: &IrExpression) -> LowerResult {
        let &IrAssignmentTarget { base, ref steps } = target;
        if !steps.is_empty() {
            // TODO implement steps
            //   for final slice, emit for loop or can just a large array type handle it?
            // let base_ty = base.ty(&self.ir_module.signals, self.ir_vars);
            // let _ = self.eval_steps(base_ty, steps)?;
            todo!()
        }

        let base_ptr = match base {
            IrSignalOrVariable::Signal(signal) => self.signal_write_ptr(span, signal)?,
            IrSignalOrVariable::Variable(var) => self.var_ptr(var),
        };
        let value = self.eval_expr(span, value)?;
        self.llvm_builder.build_store(base_ptr, value)?;

        Ok(())
    }

    fn lower_if_statement(&self, span: Span, stmt: &IrIfStatement) -> LowerResult {
        let IrIfStatement {
            condition,
            then_block,
            else_block,
        } = stmt;

        let cond = self.eval_expr(span, condition)?.into_int_value();

        let build_block = |block| move || self.lower_block(block);
        self.build_if_statement(cond, build_block(then_block), else_block.as_ref().map(build_block))
    }

    /// Build the right basic block layout for an if statement.
    /// Internally we take some care to construct blocks in an order so they appear nicely in the linear block order.
    fn build_if_statement(
        &self,
        cond: IntValue<'ctx>,
        build_then: impl FnOnce() -> LowerResult,
        build_else: Option<impl FnOnce() -> LowerResult>,
    ) -> LowerResult {
        // remember start block
        let bb_cond_end = self.llvm_builder.get_insert_block().unwrap();

        // lower then branch
        let bb_then_start = self.llvm_ctx.append_basic_block(self.function, "if.then");
        self.llvm_builder.position_at_end(bb_then_start);
        build_then()?;
        let bb_then_end = self.llvm_builder.get_insert_block().unwrap();

        // lower else branch
        let bb_else_start_end = if let Some(build_else) = build_else {
            let bb_else_start = self.llvm_ctx.append_basic_block(self.function, "if.else");
            self.llvm_builder.position_at_end(bb_else_start);
            build_else()?;
            let bb_else_end = self.llvm_builder.get_insert_block().unwrap();
            Some((bb_else_start, bb_else_end))
        } else {
            None
        };

        // create end block
        let bb_end = self.llvm_ctx.append_basic_block(self.function, "if.end");

        // create branch instruction
        self.llvm_builder.position_at_end(bb_cond_end);
        self.llvm_builder.build_conditional_branch(
            cond,
            bb_then_start,
            bb_else_start_end.map_or(bb_end, |(bb_else_start, _)| bb_else_start),
        )?;

        // jump from branches to end block
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
                lower_int_constant(self.llvm_ctx, range.as_ref(), value)?.into()
            }
            &IrExpression::Signal(sig) => {
                let ty = self.lower_ty(sig.ty(&self.ir_module.signals))?;
                let ptr = self.signal_read_ptr(span, sig)?;
                self.llvm_builder.build_load(ty, ptr, "")?
            }
            &IrExpression::Variable(var) => {
                let ty = self.lower_ty(var.ty(self.ir_vars))?;
                let ptr = self.var_ptr(var);
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

    fn eval_polarized_signal(
        &self,
        buffer: PointerValue<'ctx>,
        span: Span,
        signal: Polarized<IrSignal>,
    ) -> LowerResult<IntValue<'ctx>> {
        let Polarized { inverted, signal } = signal;

        let signal_ptr = self.signal_ptr(buffer, span, signal.into())?;
        let signal_value = self
            .llvm_builder
            .build_load(self.llvm_ctx.bool_type(), signal_ptr, "")?
            .into_int_value();

        let result = if inverted {
            self.llvm_builder.build_not(signal_value, "")?
        } else {
            signal_value
        };
        Ok(result)
    }

    fn signal_write_ptr(&self, span: Span, signal: IrSignal) -> LowerResult<PointerValue<'ctx>> {
        let buf_next = match self.param_buffers {
            ProcessBuffers::Combinatorial { next } => next,
            ProcessBuffers::Clocked {
                registers: _,
                prev: _,
                next,
            } => next,
        };
        self.signal_ptr(buf_next, span, signal)
    }

    fn signal_read_ptr(&self, span: Span, signal: IrSignal) -> LowerResult<PointerValue<'ctx>> {
        let buf_next = match self.param_buffers {
            ProcessBuffers::Combinatorial { next } => next,
            ProcessBuffers::Clocked { registers, prev, next } => {
                if registers.contains(&signal) {
                    next
                } else {
                    prev
                }
            }
        };
        self.signal_ptr(buf_next, span, signal)
    }

    fn signal_ptr(&self, buffer: PointerValue<'ctx>, span: Span, signal: IrSignal) -> LowerResult<PointerValue<'ctx>> {
        let ty_usize = self.llvm_ctx.ptr_sized_int_type(self.llvm_target, None);
        let ty_i32 = self.llvm_ctx.i32_type();
        let ty_i8 = self.llvm_ctx.i8_type();

        let ptr = match signal {
            IrSignal::Port(port) => {
                let port_index = *self.module_signal_types.port_indices.get(&port).unwrap();

                // calculate port offset address
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
                unsafe { self.llvm_builder.build_gep(ty_i8, buffer, &[port_offset], "")? }
            }
            IrSignal::Wire(wire) => {
                let wire_index = *self.module_signal_types.wire_indices.get(&wire).unwrap();

                // calculate base pointer to the wires of this instance
                let wires_base = unsafe {
                    self.llvm_builder
                        .build_gep(ty_i8, buffer, &[self.param_wires_offset], "")?
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
        };
        Ok(ptr)
    }

    fn var_ptr(&self, var: IrVariable) -> PointerValue<'ctx> {
        *self.ir_var_to_alloca.get(&var).unwrap()
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
