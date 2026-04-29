use crate::back::lower_cpp_wrap::{CppSignalInfo, collect_cpp_signals};
use crate::mid::graph::ir_modules_topological_sort;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrBlock, IrBoolBinaryOp, IrClockedProcess, IrCombinatorialProcess,
    IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrIntArithmeticOp, IrIntCompareOp, IrModule,
    IrModuleChild, IrModuleInfo, IrModuleInternalInstance, IrModules, IrPortConnection, IrSignal, IrSignalOrVariable,
    IrStatement, IrTargetStep, IrType, IrVariables,
};
use crate::syntax::ast::PortDirection;
use crate::util::arena::IndexType;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::int::{IntRepresentation, Signed};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum, IntType, StructType};
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use itertools::enumerate;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

#[derive(Debug)]
pub struct LoweredLlvm {
    pub check_hash: u64,
    pub signals: Vec<CppSignalInfo>,
}

pub fn lower_to_llvm_object(modules: &IrModules, top_module: IrModule, output: &Path) -> Result<LoweredLlvm, String> {
    Target::initialize_native(&InitializationConfig::default())?;

    let context = Context::create();
    let mut codegen = LlvmCodegen::new(&context, modules, top_module);
    codegen.codegen()?;
    codegen.module.verify().map_err(|e| e.to_string())?;

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let target_machine = target
        .create_target_machine(
            &triple,
            TargetMachine::get_host_cpu_name().to_str().unwrap_or("generic"),
            TargetMachine::get_host_cpu_features().to_str().unwrap_or(""),
            OptimizationLevel::None,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| "failed to create LLVM target machine".to_owned())?;
    target_machine
        .write_to_file(&codegen.module, FileType::Object, output)
        .map_err(|e| e.to_string())?;

    Ok(LoweredLlvm {
        check_hash: codegen.check_hash,
        signals: collect_cpp_signals(modules, top_module),
    })
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Stage {
    Prev,
    Next,
}

#[derive(Debug, Copy, Clone)]
struct ModuleTypes<'ctx> {
    signals: StructType<'ctx>,
    ports_val: StructType<'ctx>,
    ports_ptr: StructType<'ctx>,
}

#[derive(Debug, Copy, Clone)]
struct PortPtrs<'ctx> {
    ty: StructType<'ctx>,
    ptr: PointerValue<'ctx>,
}

#[derive(Debug, Copy, Clone)]
struct ValuePtr<'ctx, 'ir> {
    ptr: PointerValue<'ctx>,
    ty: &'ir IrType,
}

struct LlvmCodegen<'ctx, 'ir> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    modules: &'ir IrModules,
    top_module: IrModule,
    check_hash: u64,
    module_types: HashMap<usize, ModuleTypes<'ctx>>,
    module_functions: HashMap<usize, FunctionValue<'ctx>>,
    ptr_type: inkwell::types::PointerType<'ctx>,
    i1_type: IntType<'ctx>,
    i8_type: IntType<'ctx>,
    i64_type: IntType<'ctx>,
}

impl<'ctx, 'ir> LlvmCodegen<'ctx, 'ir> {
    fn new(context: &'ctx Context, modules: &'ir IrModules, top_module: IrModule) -> Self {
        let mut hasher = fnv::FnvHasher::default();
        format!("{modules:#?}").hash(&mut hasher);
        top_module.inner().index().hash(&mut hasher);

        Self {
            context,
            module: context.create_module("hwlang_sim"),
            builder: context.create_builder(),
            modules,
            top_module,
            check_hash: hasher.finish(),
            module_types: HashMap::new(),
            module_functions: HashMap::new(),
            ptr_type: context.ptr_type(AddressSpace::default()),
            i1_type: context.bool_type(),
            i8_type: context.i8_type(),
            i64_type: context.i64_type(),
        }
    }

    fn codegen(&mut self) -> Result<(), String> {
        self.declare_types()?;
        self.declare_module_functions();
        for module in ir_modules_topological_sort(self.modules, [self.top_module]) {
            self.codegen_module_function(module)?;
        }
        self.codegen_exports()?;
        Ok(())
    }

    fn declare_types(&mut self) -> Result<(), String> {
        for module in ir_modules_topological_sort(self.modules, [self.top_module]) {
            let module_index = module.inner().index();
            let signals = self
                .context
                .opaque_struct_type(&format!("ModuleSignals_{module_index}"));
            let ports_val = self
                .context
                .opaque_struct_type(&format!("ModulePortsVal_{module_index}"));
            let ports_ptr = self
                .context
                .opaque_struct_type(&format!("ModulePortsPtr_{module_index}"));
            self.module_types.insert(
                module_index,
                ModuleTypes {
                    signals,
                    ports_val,
                    ports_ptr,
                },
            );
        }

        for module in ir_modules_topological_sort(self.modules, [self.top_module]) {
            let module_info = &self.modules[module];
            let module_index = module.inner().index();
            let types = self.module_types[&module_index];

            let port_fields = module_info
                .ports
                .iter()
                .map(|(_, port)| self.llvm_type(&port.ty).map(BasicTypeEnum::from))
                .collect::<Result<Vec<_>, _>>()?;
            types.ports_val.set_body(&port_fields, false);

            let port_ptr_fields = module_info
                .ports
                .iter()
                .map(|_| BasicTypeEnum::from(self.ptr_type))
                .collect::<Vec<_>>();
            types.ports_ptr.set_body(&port_ptr_fields, false);

            let mut signal_fields = Vec::new();
            for (_, wire_info) in &module_info.wires {
                signal_fields.push(self.llvm_type(&wire_info.ty)?.into());
            }
            for (child_index, child) in enumerate(&module_info.children) {
                if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
                    let child_types = self.module_types[&instance.module.inner().index()];
                    signal_fields.push(child_types.signals.into());
                    for (connection_index, connection) in enumerate(&instance.port_connections) {
                        if matches!(connection.inner, IrPortConnection::Output(None)) {
                            let port_info = self.modules[instance.module]
                                .ports
                                .get_by_index(connection_index)
                                .unwrap()
                                .1;
                            signal_fields.push(self.llvm_type(&port_info.ty)?.into());
                        }
                    }
                } else {
                    let _ = child_index;
                }
            }
            types.signals.set_body(&signal_fields, false);
        }

        Ok(())
    }

    fn declare_module_functions(&mut self) {
        for module in ir_modules_topological_sort(self.modules, [self.top_module]) {
            let module_index = module.inner().index();
            let types = self.module_types[&module_index];
            let fn_type = self.context.void_type().fn_type(
                &[
                    self.ptr_type.into(),
                    self.ptr_type.into(),
                    self.ptr_type.into(),
                    self.ptr_type.into(),
                ],
                false,
            );
            let function =
                self.module
                    .add_function(&format!("module_{module_index}_all"), fn_type, Some(Linkage::Internal));
            let _ = types;
            self.module_functions.insert(module_index, function);
        }
    }

    fn codegen_module_function(&mut self, module: IrModule) -> Result<(), String> {
        let module_index = module.inner().index();
        let function = self.module_functions[&module_index];
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let types = self.module_types[&module_index];
        let prev_signals = function.get_nth_param(0).unwrap().into_pointer_value();
        let prev_ports = PortPtrs {
            ty: types.ports_ptr,
            ptr: function.get_nth_param(1).unwrap().into_pointer_value(),
        };
        let next_signals = function.get_nth_param(2).unwrap().into_pointer_value();
        let next_ports = PortPtrs {
            ty: types.ports_ptr,
            ptr: function.get_nth_param(3).unwrap().into_pointer_value(),
        };

        let module_info = &self.modules[module];
        for (child_index, child) in enumerate(&module_info.children) {
            match &child.inner {
                IrModuleChild::ClockedProcess(proc) => {
                    self.codegen_clocked_process(
                        function,
                        module,
                        child_index,
                        proc,
                        prev_signals,
                        prev_ports,
                        next_signals,
                        next_ports,
                    )?;
                }
                IrModuleChild::CombinatorialProcess(proc) => {
                    self.codegen_comb_process(function, module, child_index, proc, next_signals, next_ports)?;
                }
                IrModuleChild::ModuleInternalInstance(instance) => {
                    self.codegen_internal_instance(
                        module,
                        child_index,
                        instance,
                        prev_signals,
                        prev_ports,
                        next_signals,
                        next_ports,
                    )?;
                }
                IrModuleChild::ModuleExternalInstance(_) => {
                    return Err("external modules are not supported in the LLVM simulator".to_owned());
                }
            }
        }

        self.builder.build_return(None).map_err(|e| e.to_string())?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_clocked_process(
        &self,
        function: FunctionValue<'ctx>,
        module: IrModule,
        child_index: usize,
        proc: &IrClockedProcess,
        prev_signals: PointerValue<'ctx>,
        prev_ports: PortPtrs<'ctx>,
        next_signals: PointerValue<'ctx>,
        next_ports: PortPtrs<'ctx>,
    ) -> Result<(), String> {
        let IrClockedProcess {
            locals,
            clock_signal,
            clock_block,
            async_reset,
        } = proc;
        let module_info = &self.modules[module];

        if let Some(async_reset) = async_reset {
            let reset_body = self
                .context
                .append_basic_block(function, &format!("child_{child_index}_reset"));
            let after = self
                .context
                .append_basic_block(function, &format!("child_{child_index}_after_reset"));
            let reset = self.eval_polarized_signal(
                module_info,
                async_reset.signal.inner,
                Stage::Next,
                next_signals,
                next_ports,
            )?;
            self.builder
                .build_conditional_branch(reset, reset_body, after)
                .map_err(|e| e.to_string())?;
            self.builder.position_at_end(reset_body);
            let mut ctx = BlockCodegen::new(
                self,
                function,
                module_info,
                locals,
                Stage::Next,
                prev_signals,
                prev_ports,
                next_signals,
                next_ports,
            )?;
            for reset in &async_reset.resets {
                let (signal, value) = &reset.inner;
                let target = IrAssignmentTarget::simple(IrSignalOrVariable::Signal(*signal));
                let value = ctx.eval(value)?;
                ctx.codegen_assignment(&target, value)?;
            }
            ctx.branch_if_open(after)?;
            self.builder.position_at_end(after);
        }

        let clock_prev =
            self.eval_polarized_signal(module_info, clock_signal.inner, Stage::Prev, prev_signals, prev_ports)?;
        let clock_next =
            self.eval_polarized_signal(module_info, clock_signal.inner, Stage::Next, next_signals, next_ports)?;
        let not_prev = self
            .builder
            .build_not(clock_prev, "clk_not_prev")
            .map_err(|e| e.to_string())?;
        let is_edge = self
            .builder
            .build_and(not_prev, clock_next, "clk_edge")
            .map_err(|e| e.to_string())?;
        let body = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_clock_body"));
        let after = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_clock_after"));
        self.builder
            .build_conditional_branch(is_edge, body, after)
            .map_err(|e| e.to_string())?;
        self.builder.position_at_end(body);

        let mut ctx = BlockCodegen::new(
            self,
            function,
            module_info,
            locals,
            Stage::Prev,
            prev_signals,
            prev_ports,
            next_signals,
            next_ports,
        )?;
        ctx.codegen_block(clock_block)?;
        ctx.branch_if_open(after)?;
        self.builder.position_at_end(after);
        Ok(())
    }

    fn codegen_comb_process(
        &self,
        function: FunctionValue<'ctx>,
        module: IrModule,
        child_index: usize,
        proc: &IrCombinatorialProcess,
        next_signals: PointerValue<'ctx>,
        next_ports: PortPtrs<'ctx>,
    ) -> Result<(), String> {
        let block = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_comb"));
        self.builder
            .build_unconditional_branch(block)
            .map_err(|e| e.to_string())?;
        self.builder.position_at_end(block);

        let module_info = &self.modules[module];
        let mut ctx = BlockCodegen::new(
            self,
            function,
            module_info,
            &proc.locals,
            Stage::Next,
            next_signals,
            next_ports,
            next_signals,
            next_ports,
        )?;
        ctx.codegen_block(&proc.block)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_internal_instance(
        &self,
        module: IrModule,
        child_index: usize,
        instance: &IrModuleInternalInstance,
        prev_signals: PointerValue<'ctx>,
        prev_ports: PortPtrs<'ctx>,
        next_signals: PointerValue<'ctx>,
        next_ports: PortPtrs<'ctx>,
    ) -> Result<(), String> {
        let module_info = &self.modules[module];
        let child_module = instance.module;
        let child_types = self.module_types[&child_module.inner().index()];
        let child_function = self.module_functions[&child_module.inner().index()];

        let prev_child_signals = self.child_signals_ptr(module_info, prev_signals, child_index)?;
        let next_child_signals = self.child_signals_ptr(module_info, next_signals, child_index)?;
        let prev_child_ports = self.build_child_ports(
            module_info,
            prev_signals,
            prev_ports,
            child_index,
            instance,
            child_types.ports_ptr,
        )?;
        let next_child_ports = self.build_child_ports(
            module_info,
            next_signals,
            next_ports,
            child_index,
            instance,
            child_types.ports_ptr,
        )?;
        self.builder
            .build_call(
                child_function,
                &[
                    prev_child_signals.into(),
                    prev_child_ports.ptr.into(),
                    next_child_signals.into(),
                    next_child_ports.ptr.into(),
                ],
                "",
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_exports(&self) -> Result<(), String> {
        let top_types = self.module_types[&self.top_module.inner().index()];
        let instance_type = self.context.struct_type(
            &[
                top_types.signals.into(),
                top_types.signals.into(),
                top_types.ports_val.into(),
                top_types.ports_val.into(),
            ],
            false,
        );

        self.codegen_check_hash()?;
        self.codegen_create_instance(instance_type)?;
        self.codegen_destroy_instance()?;
        self.codegen_step(instance_type, top_types)?;
        self.codegen_get_port(instance_type, top_types)?;
        self.codegen_set_port(instance_type, top_types)?;
        self.codegen_get_signal(instance_type, top_types)?;
        Ok(())
    }

    fn codegen_check_hash(&self) -> Result<(), String> {
        let function = self
            .module
            .add_function("check_hash", self.i64_type.fn_type(&[], false), None);
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        self.builder
            .build_return(Some(&self.i64_type.const_int(self.check_hash, false)))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_create_instance(&self, instance_type: StructType<'ctx>) -> Result<(), String> {
        let function = self
            .module
            .add_function("create_instance", self.ptr_type.fn_type(&[], false), None);
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let ptr = self
            .builder
            .build_malloc(instance_type, "instance")
            .map_err(|e| e.to_string())?;
        self.builder
            .build_store(ptr, instance_type.const_zero())
            .map_err(|e| e.to_string())?;
        self.builder.build_return(Some(&ptr)).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_destroy_instance(&self) -> Result<(), String> {
        let function = self.module.add_function(
            "destroy_instance",
            self.context
                .void_type()
                .fn_type(&[BasicMetadataTypeEnum::from(self.ptr_type)], false),
            None,
        );
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let ptr = function.get_first_param().unwrap().into_pointer_value();
        self.builder.build_free(ptr).map_err(|e| e.to_string())?;
        self.builder.build_return(None).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_step(&self, instance_type: StructType<'ctx>, top_types: ModuleTypes<'ctx>) -> Result<(), String> {
        let function = self.module.add_function(
            "step",
            self.i8_type
                .fn_type(&[self.ptr_type.into(), self.i64_type.into()], false),
            None,
        );
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let instance = function.get_first_param().unwrap().into_pointer_value();

        let prev_signals = self.struct_field_ptr(instance_type, instance, 0, "prev_signals")?;
        let next_signals = self.struct_field_ptr(instance_type, instance, 1, "next_signals")?;
        let prev_ports = self.struct_field_ptr(instance_type, instance, 2, "prev_ports")?;
        let next_ports = self.struct_field_ptr(instance_type, instance, 3, "next_ports")?;
        let prev_port_ptrs = self.build_top_port_ptrs(top_types, prev_ports)?;
        let next_port_ptrs = self.build_top_port_ptrs(top_types, next_ports)?;
        let top_function = self.module_functions[&self.top_module.inner().index()];
        for _ in 0..32 {
            self.builder
                .build_call(
                    top_function,
                    &[
                        prev_signals.into(),
                        prev_port_ptrs.ptr.into(),
                        next_signals.into(),
                        next_port_ptrs.ptr.into(),
                    ],
                    "",
                )
                .map_err(|e| e.to_string())?;
            let next_signals_value = self
                .builder
                .build_load(top_types.signals, next_signals, "next_signals_value")
                .map_err(|e| e.to_string())?;
            let next_ports_value = self
                .builder
                .build_load(top_types.ports_val, next_ports, "next_ports_value")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(prev_signals, next_signals_value)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(prev_ports, next_ports_value)
                .map_err(|e| e.to_string())?;
        }
        self.builder
            .build_return(Some(&self.i8_type.const_zero()))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_get_port(&self, instance_type: StructType<'ctx>, top_types: ModuleTypes<'ctx>) -> Result<(), String> {
        let function = self.module.add_function(
            "get_port",
            self.i8_type.fn_type(
                &[
                    self.ptr_type.into(),
                    self.i64_type.into(),
                    self.i64_type.into(),
                    self.ptr_type.into(),
                ],
                false,
            ),
            None,
        );
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let instance = function.get_first_param().unwrap().into_pointer_value();
        let data_len = function.get_nth_param(2).unwrap().into_int_value();
        let data = function.get_nth_param(3).unwrap().into_pointer_value();
        let next_ports = self.struct_field_ptr(instance_type, instance, 3, "next_ports")?;
        let port_index = function.get_nth_param(1).unwrap().into_int_value();
        let (cases, bad) =
            self.build_index_dispatch(function, port_index, self.modules[self.top_module].ports.len(), "port")?;
        for (i, ((port, port_info), block)) in self.modules[self.top_module].ports.iter().zip(&cases).enumerate() {
            let _ = port;
            self.builder.position_at_end(*block);
            let field = self.struct_field_ptr(top_types.ports_val, next_ports, i as u32, "port")?;
            let value = self
                .builder
                .build_load(self.llvm_type(&port_info.ty)?, field, "port_value")
                .map_err(|e| e.to_string())?
                .into_int_value();
            self.return_packed_value(function, data_len, data, &port_info.ty, value)?;
        }
        self.builder.position_at_end(bad);
        self.builder
            .build_return(Some(&self.i8_type.const_int(1, false)))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_set_port(&self, instance_type: StructType<'ctx>, top_types: ModuleTypes<'ctx>) -> Result<(), String> {
        let function = self.module.add_function(
            "set_port",
            self.i8_type.fn_type(
                &[
                    self.ptr_type.into(),
                    self.i64_type.into(),
                    self.i64_type.into(),
                    self.ptr_type.into(),
                ],
                false,
            ),
            None,
        );
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let instance = function.get_first_param().unwrap().into_pointer_value();
        let data_len = function.get_nth_param(2).unwrap().into_int_value();
        let data = function.get_nth_param(3).unwrap().into_pointer_value();
        let next_ports = self.struct_field_ptr(instance_type, instance, 3, "next_ports")?;
        let port_index = function.get_nth_param(1).unwrap().into_int_value();
        let (cases, bad) =
            self.build_index_dispatch(function, port_index, self.modules[self.top_module].ports.len(), "port")?;
        for (i, ((_port, port_info), block)) in self.modules[self.top_module].ports.iter().zip(&cases).enumerate() {
            self.builder.position_at_end(*block);
            if port_info.direction == PortDirection::Output {
                self.builder
                    .build_return(Some(&self.i8_type.const_int(3, false)))
                    .map_err(|e| e.to_string())?;
                continue;
            }
            self.check_data_len_or_return(function, data_len, port_info.ty.size_bits())?;
            let field = self.struct_field_ptr(top_types.ports_val, next_ports, i as u32, "port")?;
            let value = self.unpack_value(data, &port_info.ty)?;
            self.builder.build_store(field, value).map_err(|e| e.to_string())?;
            self.builder
                .build_return(Some(&self.i8_type.const_zero()))
                .map_err(|e| e.to_string())?;
        }
        self.builder.position_at_end(bad);
        self.builder
            .build_return(Some(&self.i8_type.const_int(1, false)))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn codegen_get_signal(&self, instance_type: StructType<'ctx>, top_types: ModuleTypes<'ctx>) -> Result<(), String> {
        let function = self.module.add_function(
            "get_signal",
            self.i8_type.fn_type(
                &[
                    self.ptr_type.into(),
                    self.i64_type.into(),
                    self.i64_type.into(),
                    self.ptr_type.into(),
                ],
                false,
            ),
            None,
        );
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let instance = function.get_first_param().unwrap().into_pointer_value();
        let data_len = function.get_nth_param(2).unwrap().into_int_value();
        let data = function.get_nth_param(3).unwrap().into_pointer_value();
        let next_signals = self.struct_field_ptr(instance_type, instance, 1, "next_signals")?;
        let next_ports_val = self.struct_field_ptr(instance_type, instance, 3, "next_ports")?;
        let next_ports = self.build_top_port_ptrs(top_types, next_ports_val)?;
        let mut signal_cases = Vec::new();
        self.collect_signal_ptrs(self.top_module, next_signals, next_ports, &mut signal_cases)?;
        let signal_index = function.get_nth_param(1).unwrap().into_int_value();
        let signal_count = collect_cpp_signals(self.modules, self.top_module).len();
        let (cases, bad) = self.build_index_dispatch(function, signal_index, signal_count, "signal")?;
        for ((value_ptr, ty), block) in signal_cases.into_iter().zip(&cases) {
            self.builder.position_at_end(*block);
            let value = self
                .builder
                .build_load(self.llvm_type(ty)?, value_ptr, "signal_value")
                .map_err(|e| e.to_string())?
                .into_int_value();
            self.return_packed_value(function, data_len, data, ty, value)?;
        }

        self.builder.position_at_end(bad);
        self.builder
            .build_return(Some(&self.i8_type.const_int(1, false)))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn build_index_dispatch(
        &self,
        function: FunctionValue<'ctx>,
        index: IntValue<'ctx>,
        count: usize,
        prefix: &str,
    ) -> Result<(Vec<BasicBlock<'ctx>>, BasicBlock<'ctx>), String> {
        let bad = self.context.append_basic_block(function, "bad_index");
        let cases = (0..count)
            .map(|i| self.context.append_basic_block(function, &format!("{prefix}_{i}")))
            .collect::<Vec<_>>();
        let dispatch = (0..count)
            .map(|i| {
                self.context
                    .append_basic_block(function, &format!("{prefix}_{i}_dispatch"))
            })
            .collect::<Vec<_>>();
        if let Some(first_dispatch) = dispatch.first().copied() {
            let in_range = self
                .builder
                .build_int_compare(
                    IntPredicate::ULT,
                    index,
                    self.i64_type.const_int(count as u64, false),
                    "index_in_range",
                )
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(in_range, first_dispatch, bad)
                .map_err(|e| e.to_string())?;
        } else {
            self.builder
                .build_unconditional_branch(bad)
                .map_err(|e| e.to_string())?;
        }
        for (i, case) in cases.iter().copied().enumerate() {
            self.builder.position_at_end(dispatch[i]);
            if i + 1 == count {
                self.builder
                    .build_unconditional_branch(case)
                    .map_err(|e| e.to_string())?;
                continue;
            }
            let next = dispatch.get(i + 1).copied().unwrap_or(bad);
            let matches = self
                .builder
                .build_int_compare(
                    IntPredicate::EQ,
                    index,
                    self.i64_type.const_int(i as u64, false),
                    "index_matches",
                )
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(matches, case, next)
                .map_err(|e| e.to_string())?;
        }
        Ok((cases, bad))
    }

    fn build_top_port_ptrs(
        &self,
        top_types: ModuleTypes<'ctx>,
        ports_val_ptr: PointerValue<'ctx>,
    ) -> Result<PortPtrs<'ctx>, String> {
        let result = self
            .builder
            .build_alloca(top_types.ports_ptr, "top_ports")
            .map_err(|e| e.to_string())?;
        for (i, _) in enumerate(&self.modules[self.top_module].ports) {
            let value_ptr = self.struct_field_ptr(top_types.ports_val, ports_val_ptr, i as u32, "port_value")?;
            let ptr_field = self.struct_field_ptr(top_types.ports_ptr, result, i as u32, "port_ptr")?;
            self.builder
                .build_store(ptr_field, value_ptr)
                .map_err(|e| e.to_string())?;
        }
        Ok(PortPtrs {
            ty: top_types.ports_ptr,
            ptr: result,
        })
    }

    fn build_child_ports(
        &self,
        module_info: &IrModuleInfo,
        signals: PointerValue<'ctx>,
        ports: PortPtrs<'ctx>,
        child_index: usize,
        instance: &IrModuleInternalInstance,
        child_ports_ty: StructType<'ctx>,
    ) -> Result<PortPtrs<'ctx>, String> {
        let result = self
            .builder
            .build_alloca(child_ports_ty, "child_ports")
            .map_err(|e| e.to_string())?;
        let child_module_info = &self.modules[instance.module];
        let mut dummy_index = 0u32;
        for (connection_index, connection) in enumerate(&instance.port_connections) {
            let ptr = match connection.inner {
                IrPortConnection::Input(signal) | IrPortConnection::Output(Some(signal)) => {
                    self.signal_ptr(module_info, signal, signals, ports)?
                }
                IrPortConnection::Output(None) => {
                    let field_index = self.child_dummy_field_index(module_info, child_index, dummy_index);
                    dummy_index += 1;
                    let _port_info = child_module_info.ports.get_by_index(connection_index).unwrap().1;
                    self.struct_field_ptr(self.module_signal_type(module_info), signals, field_index, "dummy")?
                }
            };
            let field = self.struct_field_ptr(child_ports_ty, result, connection_index as u32, "child_port")?;
            self.builder.build_store(field, ptr).map_err(|e| e.to_string())?;
        }
        Ok(PortPtrs {
            ty: child_ports_ty,
            ptr: result,
        })
    }

    fn collect_signal_ptrs(
        &self,
        module: IrModule,
        signals: PointerValue<'ctx>,
        ports: PortPtrs<'ctx>,
        out: &mut Vec<(PointerValue<'ctx>, &'ir IrType)>,
    ) -> Result<(), String> {
        let module_info = &self.modules[module];
        let hidden = self.hidden_parent_bridge_wires(module_info);
        for (port_index, (_port, port_info)) in enumerate(&module_info.ports) {
            let ptr_field = self.struct_field_ptr(ports.ty, ports.ptr, port_index as u32, "port_ptr")?;
            let ptr = self
                .builder
                .build_load(self.ptr_type, ptr_field, "port_ptr_value")
                .map_err(|e| e.to_string())?
                .into_pointer_value();
            out.push((ptr, &port_info.ty));
        }
        for (wire, wire_info) in &module_info.wires {
            if hidden.contains(&wire.inner().index()) {
                continue;
            }
            out.push((self.wire_ptr(module_info, wire, signals)?, &wire_info.ty));
        }
        for (child_index, child) in enumerate(&module_info.children) {
            if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
                let child_signals = self.child_signals_ptr(module_info, signals, child_index)?;
                let child_ports_ty = self.module_types[&instance.module.inner().index()].ports_ptr;
                let child_ports =
                    self.build_child_ports(module_info, signals, ports, child_index, instance, child_ports_ty)?;
                self.collect_signal_ptrs(instance.module, child_signals, child_ports, out)?;
            }
        }
        Ok(())
    }

    fn return_packed_value(
        &self,
        function: FunctionValue<'ctx>,
        data_len: IntValue<'ctx>,
        data: PointerValue<'ctx>,
        ty: &IrType,
        value: IntValue<'ctx>,
    ) -> Result<(), String> {
        self.check_data_len_or_return(function, data_len, ty.size_bits())?;
        self.clear_data(data, data_len)?;
        self.pack_value(data, ty, value)?;
        self.builder
            .build_return(Some(&self.i8_type.const_zero()))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn check_data_len_or_return(
        &self,
        function: FunctionValue<'ctx>,
        data_len: IntValue<'ctx>,
        size_bits: BigUint,
    ) -> Result<(), String> {
        let min_len = biguint_to_usize(&size_bits)?.div_ceil(8);
        let ok = self
            .builder
            .build_int_compare(
                IntPredicate::UGE,
                data_len,
                self.i64_type.const_int(min_len as u64, false),
                "len_ok",
            )
            .map_err(|e| e.to_string())?;
        let ok_block = self.context.append_basic_block(function, "len_ok");
        let bad_block = self.context.append_basic_block(function, "len_bad");
        self.builder
            .build_conditional_branch(ok, ok_block, bad_block)
            .map_err(|e| e.to_string())?;
        self.builder.position_at_end(bad_block);
        self.builder
            .build_return(Some(&self.i8_type.const_int(1, false)))
            .map_err(|e| e.to_string())?;
        self.builder.position_at_end(ok_block);
        Ok(())
    }

    fn clear_data(&self, data: PointerValue<'ctx>, data_len: IntValue<'ctx>) -> Result<(), String> {
        self.builder
            .build_memset(data, 1, self.i8_type.const_zero(), data_len)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn pack_value(&self, data: PointerValue<'ctx>, ty: &IrType, value: IntValue<'ctx>) -> Result<(), String> {
        let size_bits = biguint_to_usize(&ty.size_bits())?;
        for bit in 0..size_bits {
            let byte_index = self.i64_type.const_int((bit / 8) as u64, false);
            let byte_ptr = unsafe {
                self.builder
                    .build_gep(self.i8_type, data, &[byte_index], "byte_ptr")
                    .map_err(|e| e.to_string())?
            };
            let current = self
                .builder
                .build_load(self.i8_type, byte_ptr, "byte")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let bit_value = if size_bits == 1 {
                value
            } else {
                self.builder
                    .build_right_shift(
                        value,
                        self.llvm_type(ty)?.const_int(bit as u64, false),
                        false,
                        "bit_shift",
                    )
                    .map_err(|e| e.to_string())?
            };
            let bit_i8 = self
                .builder
                .build_int_cast(bit_value, self.i8_type, "bit_i8")
                .map_err(|e| e.to_string())?;
            let masked = self
                .builder
                .build_and(bit_i8, self.i8_type.const_int(1, false), "bit_mask")
                .map_err(|e| e.to_string())?;
            let shifted = self
                .builder
                .build_left_shift(masked, self.i8_type.const_int((bit % 8) as u64, false), "bit_byte")
                .map_err(|e| e.to_string())?;
            let next = self
                .builder
                .build_or(current, shifted, "byte_next")
                .map_err(|e| e.to_string())?;
            self.builder.build_store(byte_ptr, next).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn unpack_value(&self, data: PointerValue<'ctx>, ty: &IrType) -> Result<IntValue<'ctx>, String> {
        let value_type = self.llvm_type(ty)?;
        let size_bits = biguint_to_usize(&ty.size_bits())?;
        let mut result = value_type.const_zero();
        for bit in 0..size_bits {
            let byte_index = self.i64_type.const_int((bit / 8) as u64, false);
            let byte_ptr = unsafe {
                self.builder
                    .build_gep(self.i8_type, data, &[byte_index], "byte_ptr")
                    .map_err(|e| e.to_string())?
            };
            let byte = self
                .builder
                .build_load(self.i8_type, byte_ptr, "byte")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let shifted = self
                .builder
                .build_right_shift(
                    byte,
                    self.i8_type.const_int((bit % 8) as u64, false),
                    false,
                    "byte_shift",
                )
                .map_err(|e| e.to_string())?;
            let masked = self
                .builder
                .build_and(shifted, self.i8_type.const_int(1, false), "byte_bit")
                .map_err(|e| e.to_string())?;
            let widened = self
                .builder
                .build_int_cast(masked, value_type, "bit_value")
                .map_err(|e| e.to_string())?;
            let shifted = if bit == 0 {
                widened
            } else {
                self.builder
                    .build_left_shift(widened, value_type.const_int(bit as u64, false), "value_bit")
                    .map_err(|e| e.to_string())?
            };
            result = self
                .builder
                .build_or(result, shifted, "unpacked")
                .map_err(|e| e.to_string())?;
        }
        Ok(result)
    }

    fn eval_polarized_signal(
        &self,
        module_info: &IrModuleInfo,
        signal: crate::front::signal::Polarized<IrSignal>,
        stage: Stage,
        signals: PointerValue<'ctx>,
        ports: PortPtrs<'ctx>,
    ) -> Result<IntValue<'ctx>, String> {
        let mut value = self.load_signal(module_info, signal.signal, signals, ports)?;
        if signal.inverted {
            value = self.builder.build_not(value, "not_signal").map_err(|e| e.to_string())?;
        }
        let _ = stage;
        Ok(value)
    }

    fn load_signal(
        &self,
        module_info: &IrModuleInfo,
        signal: IrSignal,
        signals: PointerValue<'ctx>,
        ports: PortPtrs<'ctx>,
    ) -> Result<IntValue<'ctx>, String> {
        let value_ptr = self.signal_ptr(module_info, signal, signals, ports)?;
        let ty = match signal {
            IrSignal::Port(port) => &module_info.ports[port].ty,
            IrSignal::Wire(wire) => &module_info.wires[wire].ty,
        };
        Ok(self
            .builder
            .build_load(self.llvm_type(ty)?, value_ptr, "signal")
            .map_err(|e| e.to_string())?
            .into_int_value())
    }

    fn signal_ptr(
        &self,
        module_info: &IrModuleInfo,
        signal: IrSignal,
        signals: PointerValue<'ctx>,
        ports: PortPtrs<'ctx>,
    ) -> Result<PointerValue<'ctx>, String> {
        match signal {
            IrSignal::Port(port) => {
                let field = self.struct_field_ptr(ports.ty, ports.ptr, port.inner().index() as u32, "port_ptr")?;
                Ok(self
                    .builder
                    .build_load(self.ptr_type, field, "port_ptr_value")
                    .map_err(|e| e.to_string())?
                    .into_pointer_value())
            }
            IrSignal::Wire(wire) => self.wire_ptr(module_info, wire, signals),
        }
    }

    fn wire_ptr(
        &self,
        module_info: &IrModuleInfo,
        wire: crate::mid::ir::IrWire,
        signals: PointerValue<'ctx>,
    ) -> Result<PointerValue<'ctx>, String> {
        self.struct_field_ptr(
            self.module_signal_type(module_info),
            signals,
            wire.inner().index() as u32,
            "wire",
        )
    }

    fn child_signals_ptr(
        &self,
        module_info: &IrModuleInfo,
        signals: PointerValue<'ctx>,
        child_index: usize,
    ) -> Result<PointerValue<'ctx>, String> {
        self.struct_field_ptr(
            self.module_signal_type(module_info),
            signals,
            self.child_signal_field_index(module_info, child_index),
            "child_signals",
        )
    }

    fn child_signal_field_index(&self, module_info: &IrModuleInfo, child_index: usize) -> u32 {
        let mut field = module_info.wires.len() as u32;
        for (i, child) in enumerate(&module_info.children) {
            if let IrModuleChild::ModuleInternalInstance(instance) = &child.inner {
                if i == child_index {
                    return field;
                }
                field += 1;
                field += instance
                    .port_connections
                    .iter()
                    .filter(|connection| matches!(connection.inner, IrPortConnection::Output(None)))
                    .count() as u32;
            }
        }
        unreachable!("child index must refer to an internal instance")
    }

    fn child_dummy_field_index(&self, module_info: &IrModuleInfo, child_index: usize, dummy_index: u32) -> u32 {
        self.child_signal_field_index(module_info, child_index) + 1 + dummy_index
    }

    fn module_signal_type(&self, module_info: &IrModuleInfo) -> StructType<'ctx> {
        let module_index = self
            .modules
            .iter()
            .find(|(_, info)| std::ptr::eq(*info, module_info))
            .unwrap()
            .0
            .inner()
            .index();
        self.module_types[&module_index].signals
    }

    fn struct_field_ptr(
        &self,
        ty: StructType<'ctx>,
        ptr: PointerValue<'ctx>,
        index: u32,
        name: &str,
    ) -> Result<PointerValue<'ctx>, String> {
        self.builder
            .build_struct_gep(ty, ptr, index, name)
            .map_err(|e| e.to_string())
    }

    fn hidden_parent_bridge_wires(&self, module_info: &IrModuleInfo) -> HashSet<usize> {
        let mut result = HashSet::new();
        for child in &module_info.children {
            let IrModuleChild::ModuleInternalInstance(instance) = &child.inner else {
                continue;
            };
            let child_module_info = &self.modules[instance.module];
            for (connection_index, connection) in instance.port_connections.iter().enumerate() {
                let IrPortConnection::Input(IrSignal::Wire(wire)) = connection.inner else {
                    continue;
                };
                let Some((_, child_port_info)) = child_module_info.ports.get_by_index(connection_index) else {
                    continue;
                };
                let parent_wire_info = &module_info.wires[wire];
                if parent_wire_info.debug_info_id.inner.as_deref() == Some(child_port_info.name.as_str()) {
                    result.insert(wire.inner().index());
                }
            }
        }
        result
    }

    fn llvm_type(&self, ty: &IrType) -> Result<IntType<'ctx>, String> {
        let width = u32::try_from(&ty.size_bits())
            .map_err(|_| format!("LLVM backend value is too wide: {} bits", ty.size_bits()))?
            .max(1);
        self.context
            .custom_width_int_type(std::num::NonZeroU32::new(width).unwrap())
            .map_err(str::to_owned)
    }

    fn const_int(&self, ty: &IrType, value: &BigInt) -> Result<IntValue<'ctx>, String> {
        let int_type = self.llvm_type(ty)?;
        let v = value
            .to_string()
            .parse::<i64>()
            .map_err(|_| format!("LLVM backend integer literal is too wide: {value}"))?;
        Ok(int_type.const_int(v as u64, v < 0))
    }

    fn cast_value(
        &self,
        value: IntValue<'ctx>,
        ty: &IrType,
        signed: Signed,
        name: &str,
    ) -> Result<IntValue<'ctx>, String> {
        let target = self.llvm_type(ty)?;
        if value.get_type() == target {
            Ok(value)
        } else {
            self.builder
                .build_int_cast_sign_flag(value, target, signed == Signed::Signed, name)
                .map_err(|e| e.to_string())
        }
    }
}

struct BlockCodegen<'a, 'ctx, 'ir> {
    cg: &'a LlvmCodegen<'ctx, 'ir>,
    function: FunctionValue<'ctx>,
    module_info: &'ir IrModuleInfo,
    locals: &'ir IrVariables,
    stage_read: Stage,
    prev_signals: PointerValue<'ctx>,
    prev_ports: PortPtrs<'ctx>,
    next_signals: PointerValue<'ctx>,
    next_ports: PortPtrs<'ctx>,
    local_ptrs: HashMap<usize, ValuePtr<'ctx, 'ir>>,
}

impl<'a, 'ctx, 'ir> BlockCodegen<'a, 'ctx, 'ir> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cg: &'a LlvmCodegen<'ctx, 'ir>,
        function: FunctionValue<'ctx>,
        module_info: &'ir IrModuleInfo,
        locals: &'ir IrVariables,
        stage_read: Stage,
        prev_signals: PointerValue<'ctx>,
        prev_ports: PortPtrs<'ctx>,
        next_signals: PointerValue<'ctx>,
        next_ports: PortPtrs<'ctx>,
    ) -> Result<Self, String> {
        let mut local_ptrs = HashMap::new();
        for (var, info) in locals {
            let ptr = cg
                .builder
                .build_alloca(cg.llvm_type(&info.ty)?, "local")
                .map_err(|e| e.to_string())?;
            local_ptrs.insert(var.inner().index(), ValuePtr { ptr, ty: &info.ty });
        }
        Ok(Self {
            cg,
            function,
            module_info,
            locals,
            stage_read,
            prev_signals,
            prev_ports,
            next_signals,
            next_ports,
            local_ptrs,
        })
    }

    fn codegen_block(&mut self, block: &IrBlock) -> Result<(), String> {
        for stmt in &block.statements {
            self.codegen_statement(&stmt.inner)?;
        }
        Ok(())
    }

    fn codegen_statement(&mut self, stmt: &IrStatement) -> Result<(), String> {
        match stmt {
            IrStatement::Assign(target, expr) => {
                let value = self.eval(expr)?;
                self.codegen_assignment(target, value)?;
            }
            IrStatement::Block(block) => self.codegen_block(block)?,
            IrStatement::If(if_stmt) => self.codegen_if(if_stmt)?,
            IrStatement::For(for_stmt) => self.codegen_for(for_stmt)?,
            IrStatement::Print(_) => {}
            IrStatement::AssertFailed => {
                return Err("assertions are not supported in the LLVM simulator yet".to_owned());
            }
        }
        Ok(())
    }

    fn codegen_if(&mut self, if_stmt: &IrIfStatement) -> Result<(), String> {
        let cond = self.eval(&if_stmt.condition)?;
        let then_block = self.cg.context.append_basic_block(self.function, "if_then");
        let else_block = self.cg.context.append_basic_block(self.function, "if_else");
        let after_block = self.cg.context.append_basic_block(self.function, "if_after");
        self.cg
            .builder
            .build_conditional_branch(cond, then_block, else_block)
            .map_err(|e| e.to_string())?;

        self.cg.builder.position_at_end(then_block);
        self.codegen_block(&if_stmt.then_block)?;
        self.branch_if_open(after_block)?;

        self.cg.builder.position_at_end(else_block);
        if let Some(block) = &if_stmt.else_block {
            self.codegen_block(block)?;
        }
        self.branch_if_open(after_block)?;

        self.cg.builder.position_at_end(after_block);
        Ok(())
    }

    fn codegen_for(&mut self, for_stmt: &IrForStatement) -> Result<(), String> {
        let index_ptr = self.local_ptrs[&for_stmt.index.inner().index()].ptr;
        let index_ty = &self.locals[for_stmt.index].ty;
        let start = self.cg.const_int(index_ty, &for_stmt.range.start)?;
        self.cg
            .builder
            .build_store(index_ptr, start)
            .map_err(|e| e.to_string())?;

        let cond_block = self.cg.context.append_basic_block(self.function, "for_cond");
        let body_block = self.cg.context.append_basic_block(self.function, "for_body");
        let after_block = self.cg.context.append_basic_block(self.function, "for_after");
        self.cg
            .builder
            .build_unconditional_branch(cond_block)
            .map_err(|e| e.to_string())?;

        self.cg.builder.position_at_end(cond_block);
        let current = self
            .cg
            .builder
            .build_load(self.cg.llvm_type(index_ty)?, index_ptr, "for_index")
            .map_err(|e| e.to_string())?
            .into_int_value();
        let end = self.cg.const_int(index_ty, &for_stmt.range.end)?;
        let cond = self
            .cg
            .builder
            .build_int_compare(IntPredicate::SLT, current, end, "for_continue")
            .map_err(|e| e.to_string())?;
        self.cg
            .builder
            .build_conditional_branch(cond, body_block, after_block)
            .map_err(|e| e.to_string())?;

        self.cg.builder.position_at_end(body_block);
        self.codegen_block(&for_stmt.block)?;
        let current = self
            .cg
            .builder
            .build_load(self.cg.llvm_type(index_ty)?, index_ptr, "for_index")
            .map_err(|e| e.to_string())?
            .into_int_value();
        let next = self
            .cg
            .builder
            .build_int_add(current, self.cg.llvm_type(index_ty)?.const_int(1, false), "for_next")
            .map_err(|e| e.to_string())?;
        self.cg
            .builder
            .build_store(index_ptr, next)
            .map_err(|e| e.to_string())?;
        self.branch_if_open(cond_block)?;

        self.cg.builder.position_at_end(after_block);
        Ok(())
    }

    fn codegen_assignment(&mut self, target: &IrAssignmentTarget, value: IntValue<'ctx>) -> Result<(), String> {
        let base = self.assignment_base_ptr(target.base)?;
        let mut offset = self.cg.llvm_type(base.ty)?.const_zero();
        let mut current_ty = base.ty;
        for step in &target.steps {
            match step {
                IrTargetStep::ArrayIndex(index) => {
                    let IrType::Array(inner, _) = current_ty else {
                        return Err("array index assignment target applied to non-array".to_owned());
                    };
                    let index = self
                        .cg
                        .cast_value(self.eval(index)?, current_ty, Signed::Unsigned, "index_cast")?;
                    let inner_bits = self
                        .cg
                        .llvm_type(current_ty)?
                        .const_int(biguint_to_u64(&inner.size_bits())?, false);
                    let scaled = self
                        .cg
                        .builder
                        .build_int_mul(index, inner_bits, "index_offset")
                        .map_err(|e| e.to_string())?;
                    offset = self
                        .cg
                        .builder
                        .build_int_add(offset, scaled, "offset")
                        .map_err(|e| e.to_string())?;
                    current_ty = inner;
                }
                IrTargetStep::ArraySlice { start, len } => {
                    let IrType::Array(inner, _) = current_ty else {
                        return Err("array slice assignment target applied to non-array".to_owned());
                    };
                    let start =
                        self.cg
                            .cast_value(self.eval(start)?, current_ty, Signed::Unsigned, "slice_start_cast")?;
                    let inner_bits = self
                        .cg
                        .llvm_type(current_ty)?
                        .const_int(biguint_to_u64(&inner.size_bits())?, false);
                    let scaled = self
                        .cg
                        .builder
                        .build_int_mul(start, inner_bits, "slice_offset")
                        .map_err(|e| e.to_string())?;
                    offset = self
                        .cg
                        .builder
                        .build_int_add(offset, scaled, "offset")
                        .map_err(|e| e.to_string())?;
                    let slice_ty = IrType::Array(inner.clone(), len.clone());
                    return self.store_partial(base, offset, &slice_ty, value);
                }
            }
        }
        if target.steps.is_empty() {
            let value = self.cg.cast_value(value, base.ty, Signed::Unsigned, "assign_cast")?;
            self.cg
                .builder
                .build_store(base.ptr, value)
                .map_err(|e| e.to_string())?;
        } else {
            self.store_partial(base, offset, current_ty, value)?;
        }
        Ok(())
    }

    fn store_partial(
        &self,
        base: ValuePtr<'ctx, 'ir>,
        offset: IntValue<'ctx>,
        part_ty: &IrType,
        value: IntValue<'ctx>,
    ) -> Result<(), String> {
        let base_type = self.cg.llvm_type(base.ty)?;
        let old = self
            .cg
            .builder
            .build_load(base_type, base.ptr, "old_value")
            .map_err(|e| e.to_string())?
            .into_int_value();
        let part_bits = biguint_to_u64(&part_ty.size_bits())?;
        let base_bits = biguint_to_u64(&base.ty.size_bits())?;
        let ones = if part_bits >= base_bits || part_bits >= 64 {
            base_type.const_all_ones()
        } else {
            base_type.const_int((1u64 << part_bits) - 1, false)
        };
        let offset = self
            .cg
            .builder
            .build_int_cast(offset, base_type, "offset_cast")
            .map_err(|e| e.to_string())?;
        let mask = self
            .cg
            .builder
            .build_left_shift(ones, offset, "part_mask")
            .map_err(|e| e.to_string())?;
        let keep = self
            .cg
            .builder
            .build_and(
                old,
                self.cg.builder.build_not(mask, "not_mask").map_err(|e| e.to_string())?,
                "kept",
            )
            .map_err(|e| e.to_string())?;
        let value = self
            .cg
            .builder
            .build_int_cast(value, base_type, "part_value_cast")
            .map_err(|e| e.to_string())?;
        let shifted = self
            .cg
            .builder
            .build_left_shift(value, offset, "part_shifted")
            .map_err(|e| e.to_string())?;
        let next = self
            .cg
            .builder
            .build_or(keep, shifted, "updated")
            .map_err(|e| e.to_string())?;
        self.cg.builder.build_store(base.ptr, next).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn assignment_base_ptr(&self, base: IrSignalOrVariable) -> Result<ValuePtr<'ctx, 'ir>, String> {
        match base {
            IrSignalOrVariable::Signal(signal) => {
                let ptr = self
                    .cg
                    .signal_ptr(self.module_info, signal, self.next_signals, self.next_ports)?;
                let ty = match signal {
                    IrSignal::Port(port) => &self.module_info.ports[port].ty,
                    IrSignal::Wire(wire) => &self.module_info.wires[wire].ty,
                };
                Ok(ValuePtr { ptr, ty })
            }
            IrSignalOrVariable::Variable(var) => Ok(self.local_ptrs[&var.inner().index()]),
        }
    }

    fn eval(&mut self, expr: &IrExpression) -> Result<IntValue<'ctx>, String> {
        match expr {
            IrExpression::Bool(value) => Ok(self.cg.i1_type.const_int(u64::from(*value), false)),
            IrExpression::Int(value) => self.cg.const_int(&expr.ty(self.module_info, self.locals), value),
            IrExpression::Signal(signal) => {
                let (signals, ports) = self.stage_values(self.stage_read);
                self.cg.load_signal(self.module_info, *signal, signals, ports)
            }
            IrExpression::Variable(var) => {
                let value_ptr = self.local_ptrs[&var.inner().index()];
                Ok(self
                    .cg
                    .builder
                    .build_load(self.cg.llvm_type(value_ptr.ty)?, value_ptr.ptr, "var")
                    .map_err(|e| e.to_string())?
                    .into_int_value())
            }
            IrExpression::Large(index) => self.eval_large(&self.module_info.large[*index]),
        }
    }

    fn eval_large(&mut self, expr: &IrExpressionLarge) -> Result<IntValue<'ctx>, String> {
        match expr {
            IrExpressionLarge::Undefined(ty) => Ok(self.cg.llvm_type(ty)?.const_zero()),
            IrExpressionLarge::BoolNot(inner) => {
                let inner = self.eval(inner)?;
                self.cg.builder.build_not(inner, "not").map_err(|e| e.to_string())
            }
            IrExpressionLarge::BoolBinary(op, left, right) => {
                let left = self.eval(left)?;
                let right = self.eval(right)?;
                match op {
                    IrBoolBinaryOp::And => self.cg.builder.build_and(left, right, "and"),
                    IrBoolBinaryOp::Or => self.cg.builder.build_or(left, right, "or"),
                    IrBoolBinaryOp::Xor => self.cg.builder.build_xor(left, right, "xor"),
                }
                .map_err(|e| e.to_string())
            }
            IrExpressionLarge::IntArithmetic(op, range, left, right) => {
                let result_ty = IrType::Int(range.clone());
                let repr = IntRepresentation::for_range(range.as_ref());
                let left = self.cg.cast_value(self.eval(left)?, &result_ty, repr.signed(), "lhs")?;
                let right = self
                    .cg
                    .cast_value(self.eval(right)?, &result_ty, repr.signed(), "rhs")?;
                match op {
                    IrIntArithmeticOp::Add => self.cg.builder.build_int_add(left, right, "add"),
                    IrIntArithmeticOp::Sub => self.cg.builder.build_int_sub(left, right, "sub"),
                    IrIntArithmeticOp::Mul => self.cg.builder.build_int_mul(left, right, "mul"),
                    IrIntArithmeticOp::Div => match repr.signed() {
                        Signed::Signed => self.cg.builder.build_int_signed_div(left, right, "div"),
                        Signed::Unsigned => self.cg.builder.build_int_unsigned_div(left, right, "div"),
                    },
                    IrIntArithmeticOp::Mod => match repr.signed() {
                        Signed::Signed => self.cg.builder.build_int_signed_rem(left, right, "mod"),
                        Signed::Unsigned => self.cg.builder.build_int_unsigned_rem(left, right, "mod"),
                    },
                    IrIntArithmeticOp::Shr => {
                        self.cg
                            .builder
                            .build_right_shift(left, right, repr.signed() == Signed::Signed, "shr")
                    }
                    IrIntArithmeticOp::Shl => self.cg.builder.build_left_shift(left, right, "shl"),
                    IrIntArithmeticOp::Pow => {
                        return Err("integer power is not supported in the LLVM simulator yet".to_owned());
                    }
                }
                .map_err(|e| e.to_string())
            }
            IrExpressionLarge::IntCompare(op, left, right) => {
                let left_ty = left.ty(self.module_info, self.locals);
                let right_ty = right.ty(self.module_info, self.locals);
                let width_ty = if left_ty.size_bits() >= right_ty.size_bits() {
                    left_ty
                } else {
                    right_ty
                };
                let left = self
                    .cg
                    .cast_value(self.eval(left)?, &width_ty, Signed::Signed, "cmp_lhs")?;
                let right = self
                    .cg
                    .cast_value(self.eval(right)?, &width_ty, Signed::Signed, "cmp_rhs")?;
                let pred = match op {
                    IrIntCompareOp::Eq => IntPredicate::EQ,
                    IrIntCompareOp::Neq => IntPredicate::NE,
                    IrIntCompareOp::Lt => IntPredicate::SLT,
                    IrIntCompareOp::Lte => IntPredicate::SLE,
                    IrIntCompareOp::Gt => IntPredicate::SGT,
                    IrIntCompareOp::Gte => IntPredicate::SGE,
                };
                self.cg
                    .builder
                    .build_int_compare(pred, left, right, "cmp")
                    .map_err(|e| e.to_string())
            }
            IrExpressionLarge::TupleLiteral(elements) => self.concat_values(
                elements
                    .iter()
                    .map(|e| (e, e.ty(self.module_info, self.locals)))
                    .collect(),
            ),
            IrExpressionLarge::StructLiteral(ty, fields) => self
                .concat_values(
                    fields
                        .iter()
                        .map(|e| (e, e.ty(self.module_info, self.locals)))
                        .collect(),
                )
                .map(|v| {
                    self.cg
                        .cast_value(v, &IrType::Struct(ty.clone()), Signed::Unsigned, "struct")
                        .unwrap()
                }),
            IrExpressionLarge::ArrayLiteral(inner_ty, len, elements) => {
                let result_ty = IrType::Array(Box::new(inner_ty.clone()), len.clone());
                let result_type = self.cg.llvm_type(&result_ty)?;
                let mut result = result_type.const_zero();
                let mut offset = BigUint::ZERO;
                for element in elements {
                    match element {
                        IrArrayLiteralElement::Single(value) => {
                            let value =
                                self.cg
                                    .cast_value(self.eval(value)?, &result_ty, Signed::Unsigned, "array_el")?;
                            let shifted = self.shift_left_const(value, &result_ty, biguint_to_u64(&offset)?)?;
                            result = self
                                .cg
                                .builder
                                .build_or(result, shifted, "array")
                                .map_err(|e| e.to_string())?;
                            offset += inner_ty.size_bits();
                        }
                        IrArrayLiteralElement::Spread(value) => {
                            let value_ty = value.ty(self.module_info, self.locals);
                            let value =
                                self.cg
                                    .cast_value(self.eval(value)?, &result_ty, Signed::Unsigned, "array_spread")?;
                            let shifted = self.shift_left_const(value, &result_ty, biguint_to_u64(&offset)?)?;
                            result = self
                                .cg
                                .builder
                                .build_or(result, shifted, "array")
                                .map_err(|e| e.to_string())?;
                            offset += value_ty.size_bits();
                        }
                    }
                }
                Ok(result)
            }
            IrExpressionLarge::EnumLiteral(ty, variant, payload) => {
                let result_ty = IrType::Enum(ty.clone());
                let tag_ty = IrType::Int(ty.tag_range());
                let tag = self.cg.llvm_type(&tag_ty)?.const_int(*variant as u64, false);
                let mut result = self.cg.cast_value(tag, &result_ty, Signed::Unsigned, "tag")?;
                if let Some(payload) = payload {
                    let payload = self
                        .cg
                        .cast_value(self.eval(payload)?, &result_ty, Signed::Unsigned, "payload")?;
                    let payload = self.shift_left_const(payload, &result_ty, biguint_to_u64(&ty.tag_size_bits())?)?;
                    result = self
                        .cg
                        .builder
                        .build_or(result, payload, "enum")
                        .map_err(|e| e.to_string())?;
                }
                Ok(result)
            }
            IrExpressionLarge::ArrayIndex { base, index } => {
                let IrType::Array(inner, _) = base.ty(self.module_info, self.locals) else {
                    return Err("array index on non-array".to_owned());
                };
                let base_ty = base.ty(self.module_info, self.locals);
                let base = self.eval(base)?;
                let index = self
                    .cg
                    .cast_value(self.eval(index)?, &base_ty, Signed::Unsigned, "index")?;
                let shift = self
                    .cg
                    .builder
                    .build_int_mul(
                        index,
                        self.cg
                            .llvm_type(&base_ty)?
                            .const_int(biguint_to_u64(&inner.size_bits())?, false),
                        "index_shift",
                    )
                    .map_err(|e| e.to_string())?;
                let shifted = self
                    .cg
                    .builder
                    .build_right_shift(base, shift, false, "array_index")
                    .map_err(|e| e.to_string())?;
                self.truncate_low_bits(shifted, &inner)
            }
            IrExpressionLarge::ArraySlice { base, start, len } => {
                let IrType::Array(inner, _) = base.ty(self.module_info, self.locals) else {
                    return Err("array slice on non-array".to_owned());
                };
                let base_ty = base.ty(self.module_info, self.locals);
                let result_ty = IrType::Array(inner.clone(), len.clone());
                let base_value = self.eval(base)?;
                let start = self
                    .cg
                    .cast_value(self.eval(start)?, &base_ty, Signed::Unsigned, "start")?;
                let shift = self
                    .cg
                    .builder
                    .build_int_mul(
                        start,
                        self.cg
                            .llvm_type(&base_ty)?
                            .const_int(biguint_to_u64(&inner.size_bits())?, false),
                        "slice_shift",
                    )
                    .map_err(|e| e.to_string())?;
                let shifted = self
                    .cg
                    .builder
                    .build_right_shift(base_value, shift, false, "slice")
                    .map_err(|e| e.to_string())?;
                self.truncate_low_bits(shifted, &result_ty)
            }
            IrExpressionLarge::TupleIndex { base, index } => {
                let base_ty = base.ty(self.module_info, self.locals);
                let IrType::Tuple(elements) = &base_ty else {
                    return Err("tuple index on non-tuple".to_owned());
                };
                let offset: BigUint = elements[..*index].iter().map(IrType::size_bits).sum();
                let base_value = self.eval(base)?;
                self.extract_const_offset(base_value, &base_ty, biguint_to_u64(&offset)?, &elements[*index])
            }
            IrExpressionLarge::StructField { base, field } => {
                let base_ty = base.ty(self.module_info, self.locals);
                let IrType::Struct(info) = &base_ty else {
                    return Err("struct field on non-struct".to_owned());
                };
                let offset = info.field_offset(*field);
                let field_ty = info.fields.get_index(*field).unwrap().1;
                let base_value = self.eval(base)?;
                self.extract_const_offset(base_value, &base_ty, biguint_to_u64(&offset)?, field_ty)
            }
            IrExpressionLarge::EnumTag { base } => {
                let base_ty = base.ty(self.module_info, self.locals);
                let IrType::Enum(info) = &base_ty else {
                    return Err("enum tag on non-enum".to_owned());
                };
                let tag_ty = IrType::Int(info.tag_range());
                let base_value = self.eval(base)?;
                self.truncate_low_bits(base_value, &tag_ty)
            }
            IrExpressionLarge::EnumPayload { base, variant } => {
                let base_ty = base.ty(self.module_info, self.locals);
                let IrType::Enum(info) = &base_ty else {
                    return Err("enum payload on non-enum".to_owned());
                };
                let payload_ty = info.variants.get_index(*variant).unwrap().1.as_ref().unwrap();
                let base_value = self.eval(base)?;
                self.extract_const_offset(base_value, &base_ty, biguint_to_u64(&info.tag_size_bits())?, payload_ty)
            }
            IrExpressionLarge::ToBits(ty, value) | IrExpressionLarge::FromBits(ty, value) => {
                self.cg.cast_value(self.eval(value)?, ty, Signed::Unsigned, "cast_bits")
            }
            IrExpressionLarge::ExpandIntRange(range, value) | IrExpressionLarge::ConstrainIntRange(range, value) => {
                self.cg.cast_value(
                    self.eval(value)?,
                    &IrType::Int(range.clone()),
                    Signed::Unsigned,
                    "cast_int_range",
                )
            }
        }
    }

    fn concat_values(&mut self, elements: Vec<(&IrExpression, IrType)>) -> Result<IntValue<'ctx>, String> {
        let result_ty = IrType::Tuple(elements.iter().map(|(_, ty)| ty.clone()).collect());
        let result_type = self.cg.llvm_type(&result_ty)?;
        let mut result = result_type.const_zero();
        let mut offset = 0u64;
        for (expr, ty) in elements {
            let value = self
                .cg
                .cast_value(self.eval(expr)?, &result_ty, Signed::Unsigned, "concat")?;
            let shifted = self.shift_left_const(value, &result_ty, offset)?;
            result = self
                .cg
                .builder
                .build_or(result, shifted, "concat")
                .map_err(|e| e.to_string())?;
            offset += biguint_to_u64(&ty.size_bits())?;
        }
        Ok(result)
    }

    fn extract_const_offset(
        &self,
        value: IntValue<'ctx>,
        base_ty: &IrType,
        offset: u64,
        result_ty: &IrType,
    ) -> Result<IntValue<'ctx>, String> {
        let shifted = if offset == 0 {
            value
        } else {
            self.cg
                .builder
                .build_right_shift(
                    value,
                    self.cg.llvm_type(base_ty)?.const_int(offset, false),
                    false,
                    "extract_shift",
                )
                .map_err(|e| e.to_string())?
        };
        self.truncate_low_bits(shifted, result_ty)
    }

    fn truncate_low_bits(&self, value: IntValue<'ctx>, result_ty: &IrType) -> Result<IntValue<'ctx>, String> {
        self.cg
            .builder
            .build_int_truncate_or_bit_cast(value, self.cg.llvm_type(result_ty)?, "trunc")
            .map_err(|e| e.to_string())
    }

    fn shift_left_const(&self, value: IntValue<'ctx>, ty: &IrType, offset: u64) -> Result<IntValue<'ctx>, String> {
        if offset == 0 {
            Ok(value)
        } else {
            self.cg
                .builder
                .build_left_shift(value, self.cg.llvm_type(ty)?.const_int(offset, false), "shift")
                .map_err(|e| e.to_string())
        }
    }

    fn stage_values(&self, stage: Stage) -> (PointerValue<'ctx>, PortPtrs<'ctx>) {
        match stage {
            Stage::Prev => (self.prev_signals, self.prev_ports),
            Stage::Next => (self.next_signals, self.next_ports),
        }
    }

    fn branch_if_open(&self, target: BasicBlock<'ctx>) -> Result<(), String> {
        if self
            .cg
            .builder
            .get_insert_block()
            .is_some_and(|block| block.get_terminator().is_none())
        {
            self.cg
                .builder
                .build_unconditional_branch(target)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

fn biguint_to_usize(value: &BigUint) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("value is too large: {value}"))
}

fn biguint_to_u64(value: &BigUint) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| format!("value is too large: {value}"))
}
