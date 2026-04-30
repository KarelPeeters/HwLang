use crate::back::lower_cpp_wrap::{CppSignalInfo, collect_cpp_signals};
use crate::mid::graph::ir_modules_topological_sort;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrBlock, IrBoolBinaryOp, IrClockedProcess, IrCombinatorialProcess,
    IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrIntArithmeticOp, IrIntCompareOp, IrModule,
    IrModuleChild, IrModuleInfo, IrModuleInternalInstance, IrModules, IrPortConnection, IrSignal, IrSignalOrVariable,
    IrStatement, IrStringSubstitution, IrTargetStep, IrType, IrVariables,
};
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::util::arena::IndexType;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::int::{IntRepresentation, Signed};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum, IntType, StructType};
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use itertools::{Itertools, enumerate};
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
    let optimization_level = match std::env::var("HWL_LLVM_OPT").as_deref() {
        Ok("1" | "less") => OptimizationLevel::Less,
        Ok("2" | "default") => OptimizationLevel::Default,
        Ok("3" | "aggressive") => OptimizationLevel::Aggressive,
        Ok("0") | Err(_) => OptimizationLevel::None,
        Ok(value) => {
            return Err(format!(
                "invalid HWL_LLVM_OPT value `{value}`, expected 0, 1, 2, 3, less, default, or aggressive"
            ));
        }
    };
    let target_machine = target
        .create_target_machine(
            &triple,
            TargetMachine::get_host_cpu_name().to_str().unwrap_or("generic"),
            TargetMachine::get_host_cpu_features().to_str().unwrap_or(""),
            optimization_level,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum Stage {
    Prev,
    Next,
}

fn type_signedness(ty: &IrType) -> Signed {
    match ty {
        IrType::Int(range) => IntRepresentation::for_range(range.as_ref()).signed(),
        IrType::Bool | IrType::Array(_, _) | IrType::Tuple(_) | IrType::Struct(_) | IrType::Enum(_) => Signed::Unsigned,
    }
}

#[derive(Debug, Copy, Clone)]
struct ModuleTypes<'ctx> {
    signals: StructType<'ctx>,
    ports_val: StructType<'ctx>,
    ports_ptr: StructType<'ctx>,
}

#[derive(Debug, Clone)]
struct PortPtrs<'ctx> {
    ty: StructType<'ctx>,
    ptr: PointerValue<'ctx>,
    direct: Option<Vec<PointerValue<'ctx>>>,
}

#[derive(Debug, Copy, Clone)]
struct ValuePtr<'ctx, 'ir> {
    ptr: PointerValue<'ctx>,
    ty: &'ir IrType,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct SignalKey {
    path: Vec<usize>,
    kind: SignalKeyKind,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct PolarizedSignalKey {
    signal: SignalKey,
    inverted: bool,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct SignalValueCacheKey {
    signal: PolarizedSignalKey,
    stage: Stage,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum SignalKeyKind {
    Port(usize),
    Wire(usize),
    Dummy { child_index: usize, dummy_index: usize },
}

#[derive(Debug, Clone)]
struct ScheduledNode {
    module: IrModule,
    path: Vec<usize>,
    child_index: usize,
    kind: ScheduledNodeKind,
    read_next: HashSet<SignalKey>,
    partial_read_next: HashSet<SignalKey>,
    write_next: HashSet<SignalKey>,
    clock_key: Option<PolarizedSignalKey>,
    reset_key: Option<PolarizedSignalKey>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum ScheduledNodeKind {
    Comb,
    AsyncReset,
    ClockEdge,
}

#[derive(Debug)]
struct ScheduleScope {
    module: IrModule,
    path: Vec<usize>,
    port_aliases: Vec<SignalKey>,
}

#[derive(Debug, Default)]
struct SignalAccesses {
    reads: HashSet<SignalKey>,
    partial_reads: HashSet<SignalKey>,
    writes: HashSet<SignalKey>,
}

#[derive(Debug, Copy, Clone)]
enum ScheduleReadStage {
    Next,
}

#[derive(Debug, Clone)]
struct CodegenScope<'ctx> {
    module: IrModule,
    prev_signals: PointerValue<'ctx>,
    prev_ports: PortPtrs<'ctx>,
    next_signals: PointerValue<'ctx>,
    next_ports: PortPtrs<'ctx>,
}

struct LlvmCodegen<'ctx, 'ir> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    modules: &'ir IrModules,
    top_module: IrModule,
    check_hash: u64,
    module_types: HashMap<usize, ModuleTypes<'ctx>>,
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
            ptr_type: context.ptr_type(AddressSpace::default()),
            i1_type: context.bool_type(),
            i8_type: context.i8_type(),
            i64_type: context.i64_type(),
        }
    }

    fn codegen(&mut self) -> Result<(), String> {
        self.declare_types()?;
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

    fn build_schedule(&self) -> Result<Vec<ScheduledNode>, String> {
        let top_info = &self.modules[self.top_module];
        let top_aliases = top_info
            .ports
            .iter()
            .map(|(port, _)| SignalKey {
                path: Vec::new(),
                kind: SignalKeyKind::Port(port.inner().index()),
            })
            .collect();
        let scope = ScheduleScope {
            module: self.top_module,
            path: Vec::new(),
            port_aliases: top_aliases,
        };
        let mut nodes = Vec::new();
        self.collect_schedule_nodes(scope, &mut nodes)?;
        self.toposort_schedule(nodes)
    }

    fn collect_schedule_nodes(&self, scope: ScheduleScope, nodes: &mut Vec<ScheduledNode>) -> Result<(), String> {
        let module_info = &self.modules[scope.module];
        for (child_index, child) in enumerate(&module_info.children) {
            match &child.inner {
                IrModuleChild::ClockedProcess(proc) => {
                    if proc.async_reset.is_some() {
                        nodes.push(self.schedule_clocked_node(
                            &scope,
                            child_index,
                            proc,
                            ScheduledNodeKind::AsyncReset,
                        )?);
                    }
                    nodes.push(self.schedule_clocked_node(&scope, child_index, proc, ScheduledNodeKind::ClockEdge)?);
                }
                IrModuleChild::CombinatorialProcess(proc) => {
                    let mut accesses = SignalAccesses::default();
                    self.collect_block_accesses(
                        &scope,
                        &proc.locals,
                        &proc.block,
                        ScheduleReadStage::Next,
                        &mut accesses,
                    );
                    nodes.push(ScheduledNode {
                        module: scope.module,
                        path: scope.path.clone(),
                        child_index,
                        kind: ScheduledNodeKind::Comb,
                        read_next: accesses.reads,
                        partial_read_next: accesses.partial_reads,
                        write_next: accesses.writes,
                        clock_key: None,
                        reset_key: None,
                    });
                }
                IrModuleChild::ModuleInternalInstance(instance) => {
                    let child_scope = self.child_schedule_scope(&scope, child_index, instance)?;
                    self.collect_schedule_nodes(child_scope, nodes)?;
                }
                IrModuleChild::ModuleExternalInstance(_) => {
                    return Err("external modules are not supported in the LLVM simulator".to_owned());
                }
            }
        }
        Ok(())
    }

    fn schedule_clocked_node(
        &self,
        scope: &ScheduleScope,
        child_index: usize,
        proc: &IrClockedProcess,
        kind: ScheduledNodeKind,
    ) -> Result<ScheduledNode, String> {
        let mut accesses = SignalAccesses::default();
        match kind {
            ScheduledNodeKind::AsyncReset => {
                let async_reset = proc
                    .async_reset
                    .as_ref()
                    .ok_or_else(|| "async-reset schedule node without async reset".to_owned())?;
                accesses
                    .reads
                    .insert(self.resolve_signal_key(scope, async_reset.signal.inner.signal));
                let empty_locals = IrVariables::new();
                for reset in &async_reset.resets {
                    let (signal, value) = &reset.inner;
                    accesses.writes.insert(self.resolve_signal_key(scope, *signal));
                    self.collect_expression_accesses(
                        scope,
                        &empty_locals,
                        value,
                        ScheduleReadStage::Next,
                        &mut accesses,
                    );
                }
            }
            ScheduledNodeKind::ClockEdge => {
                accesses
                    .reads
                    .insert(self.resolve_signal_key(scope, proc.clock_signal.inner.signal));
                if let Some(async_reset) = &proc.async_reset {
                    accesses
                        .reads
                        .insert(self.resolve_signal_key(scope, async_reset.signal.inner.signal));
                }
                self.collect_block_accesses(
                    scope,
                    &proc.locals,
                    &proc.clock_block,
                    ScheduleReadStage::Next,
                    &mut accesses,
                );
            }
            ScheduledNodeKind::Comb => unreachable!("clocked node cannot be combinatorial"),
        }
        Ok(ScheduledNode {
            module: scope.module,
            path: scope.path.clone(),
            child_index,
            kind,
            read_next: accesses.reads,
            partial_read_next: accesses.partial_reads,
            write_next: accesses.writes,
            clock_key: matches!(kind, ScheduledNodeKind::ClockEdge)
                .then(|| self.resolve_polarized_signal_key(scope, proc.clock_signal.inner)),
            reset_key: proc
                .async_reset
                .as_ref()
                .filter(|_| matches!(kind, ScheduledNodeKind::AsyncReset | ScheduledNodeKind::ClockEdge))
                .map(|async_reset| self.resolve_polarized_signal_key(scope, async_reset.signal.inner)),
        })
    }

    fn child_schedule_scope(
        &self,
        scope: &ScheduleScope,
        child_index: usize,
        instance: &IrModuleInternalInstance,
    ) -> Result<ScheduleScope, String> {
        let mut child_path = scope.path.clone();
        child_path.push(child_index);

        let mut aliases = Vec::new();
        let mut dummy_index = 0usize;
        for connection in &instance.port_connections {
            let key = match connection.inner {
                IrPortConnection::Input(signal) | IrPortConnection::Output(Some(signal)) => {
                    self.resolve_signal_key(scope, signal)
                }
                IrPortConnection::Output(None) => {
                    let key = SignalKey {
                        path: scope.path.clone(),
                        kind: SignalKeyKind::Dummy {
                            child_index,
                            dummy_index,
                        },
                    };
                    dummy_index += 1;
                    key
                }
            };
            aliases.push(key);
        }

        Ok(ScheduleScope {
            module: instance.module,
            path: child_path,
            port_aliases: aliases,
        })
    }

    fn resolve_signal_key(&self, scope: &ScheduleScope, signal: IrSignal) -> SignalKey {
        match signal {
            IrSignal::Port(port) => scope.port_aliases[port.inner().index()].clone(),
            IrSignal::Wire(wire) => SignalKey {
                path: scope.path.clone(),
                kind: SignalKeyKind::Wire(wire.inner().index()),
            },
        }
    }

    fn resolve_polarized_signal_key(
        &self,
        scope: &ScheduleScope,
        signal: crate::front::signal::Polarized<IrSignal>,
    ) -> PolarizedSignalKey {
        PolarizedSignalKey {
            signal: self.resolve_signal_key(scope, signal.signal),
            inverted: signal.inverted,
        }
    }

    fn collect_block_accesses(
        &self,
        scope: &ScheduleScope,
        locals: &IrVariables,
        block: &IrBlock,
        read_stage: ScheduleReadStage,
        accesses: &mut SignalAccesses,
    ) {
        for stmt in &block.statements {
            self.collect_statement_accesses(scope, locals, &stmt.inner, read_stage, accesses);
        }
    }

    fn collect_statement_accesses(
        &self,
        scope: &ScheduleScope,
        locals: &IrVariables,
        stmt: &IrStatement,
        read_stage: ScheduleReadStage,
        accesses: &mut SignalAccesses,
    ) {
        match stmt {
            IrStatement::Assign(target, expr) => {
                self.collect_assignment_target_accesses(scope, locals, target, read_stage, accesses);
                self.collect_expression_accesses(scope, locals, expr, read_stage, accesses);
            }
            IrStatement::Block(block) => self.collect_block_accesses(scope, locals, block, read_stage, accesses),
            IrStatement::If(if_stmt) => {
                self.collect_expression_accesses(scope, locals, &if_stmt.condition, read_stage, accesses);
                self.collect_block_accesses(scope, locals, &if_stmt.then_block, read_stage, accesses);
                if let Some(else_block) = &if_stmt.else_block {
                    self.collect_block_accesses(scope, locals, else_block, read_stage, accesses);
                }
            }
            IrStatement::For(for_stmt) => {
                self.collect_block_accesses(scope, locals, &for_stmt.block, read_stage, accesses);
            }
            IrStatement::Print(pieces) => {
                for piece in pieces {
                    if let StringPiece::Substitute(subst) = piece {
                        match subst {
                            IrStringSubstitution::Integer(expr, _) => {
                                self.collect_expression_accesses(scope, locals, expr, read_stage, accesses);
                            }
                        }
                    }
                }
            }
            IrStatement::AssertFailed => {}
        }
    }

    fn collect_assignment_target_accesses(
        &self,
        scope: &ScheduleScope,
        locals: &IrVariables,
        target: &IrAssignmentTarget,
        read_stage: ScheduleReadStage,
        accesses: &mut SignalAccesses,
    ) {
        if let IrSignalOrVariable::Signal(signal) = target.base {
            let key = self.resolve_signal_key(scope, signal);
            accesses.writes.insert(key.clone());
            if !target.steps.is_empty() && matches!(read_stage, ScheduleReadStage::Next) {
                accesses.partial_reads.insert(key);
            }
        }
        for step in &target.steps {
            match step {
                IrTargetStep::ArrayIndex(index) => {
                    self.collect_expression_accesses(scope, locals, index, read_stage, accesses);
                }
                IrTargetStep::ArraySlice { start, len: _ } => {
                    self.collect_expression_accesses(scope, locals, start, read_stage, accesses);
                }
            }
        }
    }

    fn collect_expression_accesses(
        &self,
        scope: &ScheduleScope,
        locals: &IrVariables,
        expr: &IrExpression,
        read_stage: ScheduleReadStage,
        accesses: &mut SignalAccesses,
    ) {
        match expr {
            IrExpression::Signal(signal) => {
                if matches!(read_stage, ScheduleReadStage::Next) {
                    accesses.reads.insert(self.resolve_signal_key(scope, *signal));
                }
            }
            IrExpression::Variable(_) | IrExpression::Bool(_) | IrExpression::Int(_) => {}
            IrExpression::Large(index) => match &self.modules[scope.module].large[*index] {
                IrExpressionLarge::Undefined(_) => {}
                IrExpressionLarge::BoolNot(inner) => {
                    self.collect_expression_accesses(scope, locals, inner, read_stage, accesses);
                }
                IrExpressionLarge::BoolBinary(_, left, right)
                | IrExpressionLarge::IntArithmetic(_, _, left, right)
                | IrExpressionLarge::IntCompare(_, left, right) => {
                    self.collect_expression_accesses(scope, locals, left, read_stage, accesses);
                    self.collect_expression_accesses(scope, locals, right, read_stage, accesses);
                }
                IrExpressionLarge::TupleLiteral(elements) | IrExpressionLarge::StructLiteral(_, elements) => {
                    for element in elements {
                        self.collect_expression_accesses(scope, locals, element, read_stage, accesses);
                    }
                }
                IrExpressionLarge::ArrayLiteral(_, _, elements) => {
                    for element in elements {
                        match element {
                            IrArrayLiteralElement::Single(expr) | IrArrayLiteralElement::Spread(expr) => {
                                self.collect_expression_accesses(scope, locals, expr, read_stage, accesses);
                            }
                        }
                    }
                }
                IrExpressionLarge::EnumLiteral(_, _, payload) => {
                    if let Some(payload) = payload {
                        self.collect_expression_accesses(scope, locals, payload, read_stage, accesses);
                    }
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    self.collect_expression_accesses(scope, locals, base, read_stage, accesses);
                    self.collect_expression_accesses(scope, locals, index, read_stage, accesses);
                }
                IrExpressionLarge::ArraySlice { base, start, len: _ } => {
                    self.collect_expression_accesses(scope, locals, base, read_stage, accesses);
                    self.collect_expression_accesses(scope, locals, start, read_stage, accesses);
                }
                IrExpressionLarge::TupleIndex { base, index: _ }
                | IrExpressionLarge::StructField { base, field: _ }
                | IrExpressionLarge::EnumTag { base }
                | IrExpressionLarge::EnumPayload { base, variant: _ }
                | IrExpressionLarge::ToBits(_, base)
                | IrExpressionLarge::FromBits(_, base)
                | IrExpressionLarge::ExpandIntRange(_, base)
                | IrExpressionLarge::ConstrainIntRange(_, base) => {
                    self.collect_expression_accesses(scope, locals, base, read_stage, accesses);
                }
            },
        }
        let _ = locals;
    }

    fn toposort_schedule(&self, nodes: Vec<ScheduledNode>) -> Result<Vec<ScheduledNode>, String> {
        let mut writers: HashMap<SignalKey, Vec<usize>> = HashMap::new();
        for (node_index, node) in enumerate(&nodes) {
            for key in &node.write_next {
                writers.entry(key.clone()).or_default().push(node_index);
            }
        }

        let mut edges: Vec<HashSet<usize>> = vec![HashSet::new(); nodes.len()];
        let mut indegree = vec![0usize; nodes.len()];
        for (reader_index, node) in enumerate(&nodes) {
            for key in node.read_next.iter().chain(&node.partial_read_next) {
                if let Some(signal_writers) = writers.get(key) {
                    for &writer_index in signal_writers {
                        if writer_index != reader_index && edges[writer_index].insert(reader_index) {
                            indegree[reader_index] += 1;
                        }
                    }
                }
            }
        }

        let mut ready = (0..nodes.len()).filter(|&i| indegree[i] == 0).collect::<Vec<_>>();
        let mut order = Vec::with_capacity(nodes.len());
        while let Some(node_index) = ready.first().copied() {
            ready.remove(0);
            order.push(node_index);
            for &to in &edges[node_index] {
                indegree[to] -= 1;
                if indegree[to] == 0 {
                    let insert_at = ready.partition_point(|&existing| existing < to);
                    ready.insert(insert_at, to);
                }
            }
        }

        if order.len() != nodes.len() {
            let cycle_nodes = indegree
                .iter()
                .positions(|&degree| degree != 0)
                .take(12)
                .map(|i| self.schedule_node_name(&nodes[i]))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "combinational cycle not supported by scheduled LLVM backend; involved nodes include: {cycle_nodes}"
            ));
        }

        Ok(order.into_iter().map(|i| nodes[i].clone()).collect())
    }

    fn schedule_node_name(&self, node: &ScheduledNode) -> String {
        let module_info = &self.modules[node.module];
        let module_name = module_info
            .debug_info_id
            .inner
            .clone()
            .unwrap_or_else(|| format!("module_{}", node.module.inner().index()));
        format!(
            "{module_name}@{:?} child {} {:?}",
            node.path, node.child_index, node.kind
        )
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
            next_ports.clone(),
            next_signals,
            next_ports.clone(),
        )?;
        ctx.codegen_block(&proc.block)?;
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

        let top_scope = CodegenScope {
            module: self.top_module,
            prev_signals,
            prev_ports: prev_port_ptrs.clone(),
            next_signals,
            next_ports: next_port_ptrs.clone(),
        };
        let mut scopes = HashMap::new();
        scopes.insert(Vec::new(), top_scope);
        self.codegen_child_scopes(&mut scopes, Vec::new())?;
        let schedule = self.build_schedule()?;
        let written_signals = schedule
            .iter()
            .flat_map(|node| node.write_next.iter().cloned())
            .collect::<HashSet<_>>();
        let mut signal_value_cache = HashMap::new();
        let mut node_index = 0;
        while node_index < schedule.len() {
            let node = &schedule[node_index];
            if matches!(node.kind, ScheduledNodeKind::AsyncReset | ScheduledNodeKind::ClockEdge) {
                let mut end = node_index + 1;
                while end < schedule.len() && self.same_guard_group(node, &schedule[end]) {
                    end += 1;
                }
                if end > node_index + 1 {
                    self.codegen_guard_group(
                        function,
                        &schedule[node_index..end],
                        &scopes,
                        &written_signals,
                        &mut signal_value_cache,
                    )?;
                    node_index = end;
                    continue;
                }
            }

            let scope = scopes
                .get(&node.path)
                .ok_or_else(|| format!("missing codegen scope for scheduled path {:?}", node.path))?
                .clone();
            self.codegen_scheduled_node(function, node, scope, &written_signals, &mut signal_value_cache)?;
            node_index += 1;
        }
        self.copy_memory(prev_signals, next_signals, top_types.signals)?;
        self.copy_memory(prev_ports, next_ports, top_types.ports_val)?;
        self.builder
            .build_return(Some(&self.i8_type.const_zero()))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn same_guard_group(&self, left: &ScheduledNode, right: &ScheduledNode) -> bool {
        left.kind == right.kind && left.clock_key == right.clock_key && left.reset_key == right.reset_key
    }

    fn codegen_child_scopes(
        &self,
        scopes: &mut HashMap<Vec<usize>, CodegenScope<'ctx>>,
        path: Vec<usize>,
    ) -> Result<(), String> {
        let scope = scopes
            .get(&path)
            .ok_or_else(|| format!("missing codegen scope for path {path:?}"))?
            .clone();
        let module_info = &self.modules[scope.module];
        for (child_index, child) in enumerate(&module_info.children) {
            let IrModuleChild::ModuleInternalInstance(instance) = &child.inner else {
                continue;
            };
            let child_types = self.module_types[&instance.module.inner().index()];
            let mut child_path = path.clone();
            child_path.push(child_index);
            let child_scope = CodegenScope {
                module: instance.module,
                prev_signals: self.child_signals_ptr(module_info, scope.prev_signals, child_index)?,
                prev_ports: self.build_child_ports(
                    module_info,
                    scope.prev_signals,
                    scope.prev_ports.clone(),
                    child_index,
                    instance,
                    child_types.ports_ptr,
                )?,
                next_signals: self.child_signals_ptr(module_info, scope.next_signals, child_index)?,
                next_ports: self.build_child_ports(
                    module_info,
                    scope.next_signals,
                    scope.next_ports.clone(),
                    child_index,
                    instance,
                    child_types.ports_ptr,
                )?,
            };
            scopes.insert(child_path.clone(), child_scope);
            self.codegen_child_scopes(scopes, child_path)?;
        }
        Ok(())
    }

    fn codegen_scheduled_node(
        &self,
        function: FunctionValue<'ctx>,
        node: &ScheduledNode,
        scope: CodegenScope<'ctx>,
        written_signals: &HashSet<SignalKey>,
        signal_value_cache: &mut HashMap<SignalValueCacheKey, IntValue<'ctx>>,
    ) -> Result<(), String> {
        let module_info = &self.modules[node.module];
        match (&module_info.children[node.child_index].inner, node.kind) {
            (IrModuleChild::CombinatorialProcess(proc), ScheduledNodeKind::Comb) => {
                self.codegen_comb_process(
                    function,
                    node.module,
                    node.child_index,
                    proc,
                    scope.next_signals,
                    scope.next_ports,
                )?;
            }
            (IrModuleChild::ClockedProcess(proc), ScheduledNodeKind::AsyncReset) => {
                self.codegen_async_reset_node(function, node, proc, scope, written_signals, signal_value_cache)?;
            }
            (IrModuleChild::ClockedProcess(proc), ScheduledNodeKind::ClockEdge) => {
                self.codegen_clock_edge_node(function, node, proc, scope, written_signals, signal_value_cache)?;
            }
            _ => return Err("scheduled node kind did not match IR child".to_owned()),
        }
        Ok(())
    }

    fn codegen_guard_group(
        &self,
        function: FunctionValue<'ctx>,
        nodes: &[ScheduledNode],
        scopes: &HashMap<Vec<usize>, CodegenScope<'ctx>>,
        written_signals: &HashSet<SignalKey>,
        signal_value_cache: &mut HashMap<SignalValueCacheKey, IntValue<'ctx>>,
    ) -> Result<(), String> {
        let first = &nodes[0];
        let first_scope = scopes
            .get(&first.path)
            .ok_or_else(|| format!("missing codegen scope for scheduled path {:?}", first.path))?
            .clone();
        let first_module_info = &self.modules[first.module];
        match first.kind {
            ScheduledNodeKind::AsyncReset => {
                let IrModuleChild::ClockedProcess(first_proc) = &first_module_info.children[first.child_index].inner
                else {
                    return Err("async-reset group contained non-clocked process".to_owned());
                };
                let async_reset = first_proc
                    .async_reset
                    .as_ref()
                    .ok_or_else(|| "async-reset group node without async reset".to_owned())?;
                let reset_body = self.context.append_basic_block(function, "reset_group");
                let after = self.context.append_basic_block(function, "after_reset_group");
                let reset = self.eval_polarized_signal_cached(
                    first_module_info,
                    async_reset.signal.inner,
                    first.reset_key.as_ref(),
                    Stage::Next,
                    first_scope.next_signals,
                    first_scope.next_ports.clone(),
                    written_signals,
                    signal_value_cache,
                )?;
                self.builder
                    .build_conditional_branch(reset, reset_body, after)
                    .map_err(|e| e.to_string())?;
                self.builder.position_at_end(reset_body);
                for node in nodes {
                    let module_info = &self.modules[node.module];
                    let IrModuleChild::ClockedProcess(proc) = &module_info.children[node.child_index].inner else {
                        return Err("async-reset group contained non-clocked process".to_owned());
                    };
                    let scope = scopes
                        .get(&node.path)
                        .ok_or_else(|| format!("missing codegen scope for scheduled path {:?}", node.path))?
                        .clone();
                    self.codegen_async_reset_body(function, proc, scope)?;
                }
                self.branch_if_open(after)?;
                self.builder.position_at_end(after);
            }
            ScheduledNodeKind::ClockEdge => {
                let IrModuleChild::ClockedProcess(first_proc) = &first_module_info.children[first.child_index].inner
                else {
                    return Err("clock group contained non-clocked process".to_owned());
                };
                let should_run = self.codegen_clock_guard(
                    first_module_info,
                    first,
                    first_proc,
                    first_scope,
                    written_signals,
                    signal_value_cache,
                )?;
                let body = self.context.append_basic_block(function, "clock_group");
                let after = self.context.append_basic_block(function, "after_clock_group");
                self.builder
                    .build_conditional_branch(should_run, body, after)
                    .map_err(|e| e.to_string())?;
                self.builder.position_at_end(body);
                for node in nodes {
                    let module_info = &self.modules[node.module];
                    let IrModuleChild::ClockedProcess(proc) = &module_info.children[node.child_index].inner else {
                        return Err("clock group contained non-clocked process".to_owned());
                    };
                    let scope = scopes
                        .get(&node.path)
                        .ok_or_else(|| format!("missing codegen scope for scheduled path {:?}", node.path))?
                        .clone();
                    self.codegen_clock_edge_body(function, module_info, proc, scope)?;
                }
                self.branch_if_open(after)?;
                self.builder.position_at_end(after);
            }
            ScheduledNodeKind::Comb => unreachable!("combinatorial nodes are not guard grouped"),
        }
        Ok(())
    }

    fn codegen_async_reset_node(
        &self,
        function: FunctionValue<'ctx>,
        node: &ScheduledNode,
        proc: &IrClockedProcess,
        scope: CodegenScope<'ctx>,
        written_signals: &HashSet<SignalKey>,
        signal_value_cache: &mut HashMap<SignalValueCacheKey, IntValue<'ctx>>,
    ) -> Result<(), String> {
        let module_info = &self.modules[scope.module];
        let child_index = node.child_index;
        let async_reset = proc
            .async_reset
            .as_ref()
            .ok_or_else(|| "async-reset schedule node without async reset".to_owned())?;
        let reset_body = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_reset"));
        let after = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_after_reset"));
        let reset = self.eval_polarized_signal_cached(
            module_info,
            async_reset.signal.inner,
            node.reset_key.as_ref(),
            Stage::Next,
            scope.next_signals,
            scope.next_ports.clone(),
            written_signals,
            signal_value_cache,
        )?;
        self.builder
            .build_conditional_branch(reset, reset_body, after)
            .map_err(|e| e.to_string())?;

        self.builder.position_at_end(reset_body);
        self.codegen_async_reset_body(function, proc, scope)?;
        self.branch_if_open(after)?;
        self.builder.position_at_end(after);
        Ok(())
    }

    fn codegen_async_reset_body(
        &self,
        function: FunctionValue<'ctx>,
        proc: &IrClockedProcess,
        scope: CodegenScope<'ctx>,
    ) -> Result<(), String> {
        let module_info = &self.modules[scope.module];
        let async_reset = proc
            .async_reset
            .as_ref()
            .ok_or_else(|| "async-reset body without async reset".to_owned())?;
        let mut ctx = BlockCodegen::new(
            self,
            function,
            module_info,
            &proc.locals,
            Stage::Next,
            scope.prev_signals,
            scope.prev_ports.clone(),
            scope.next_signals,
            scope.next_ports.clone(),
        )?;
        for reset in &async_reset.resets {
            let (signal, value) = &reset.inner;
            let target = IrAssignmentTarget::simple(IrSignalOrVariable::Signal(*signal));
            let value_ty = value.ty(module_info, &proc.locals);
            let value = ctx.eval(value)?;
            ctx.codegen_assignment(&target, value, &value_ty)?;
        }
        Ok(())
    }

    fn codegen_clock_edge_node(
        &self,
        function: FunctionValue<'ctx>,
        node: &ScheduledNode,
        proc: &IrClockedProcess,
        scope: CodegenScope<'ctx>,
        written_signals: &HashSet<SignalKey>,
        signal_value_cache: &mut HashMap<SignalValueCacheKey, IntValue<'ctx>>,
    ) -> Result<(), String> {
        let module_info = &self.modules[scope.module];
        let child_index = node.child_index;
        let should_run = self.codegen_clock_guard(
            module_info,
            node,
            proc,
            scope.clone(),
            written_signals,
            signal_value_cache,
        )?;

        let body = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_clock_body"));
        let after = self
            .context
            .append_basic_block(function, &format!("child_{child_index}_clock_after"));
        self.builder
            .build_conditional_branch(should_run, body, after)
            .map_err(|e| e.to_string())?;
        self.builder.position_at_end(body);

        self.codegen_clock_edge_body(function, module_info, proc, scope)?;
        self.branch_if_open(after)?;
        self.builder.position_at_end(after);
        Ok(())
    }

    fn codegen_clock_guard(
        &self,
        module_info: &IrModuleInfo,
        node: &ScheduledNode,
        proc: &IrClockedProcess,
        scope: CodegenScope<'ctx>,
        written_signals: &HashSet<SignalKey>,
        signal_value_cache: &mut HashMap<SignalValueCacheKey, IntValue<'ctx>>,
    ) -> Result<IntValue<'ctx>, String> {
        let clock_prev = self.eval_polarized_signal_cached(
            module_info,
            proc.clock_signal.inner,
            node.clock_key.as_ref(),
            Stage::Prev,
            scope.prev_signals,
            scope.prev_ports.clone(),
            written_signals,
            signal_value_cache,
        )?;
        let clock_next = self.eval_polarized_signal_cached(
            module_info,
            proc.clock_signal.inner,
            node.clock_key.as_ref(),
            Stage::Next,
            scope.next_signals,
            scope.next_ports.clone(),
            written_signals,
            signal_value_cache,
        )?;
        let not_prev = self
            .builder
            .build_not(clock_prev, "clk_not_prev")
            .map_err(|e| e.to_string())?;
        let mut should_run = self
            .builder
            .build_and(not_prev, clock_next, "clk_edge")
            .map_err(|e| e.to_string())?;

        if let Some(async_reset) = &proc.async_reset {
            let reset = self.eval_polarized_signal_cached(
                module_info,
                async_reset.signal.inner,
                node.reset_key.as_ref(),
                Stage::Next,
                scope.next_signals,
                scope.next_ports.clone(),
                written_signals,
                signal_value_cache,
            )?;
            let not_reset = self
                .builder
                .build_not(reset, "clk_not_reset")
                .map_err(|e| e.to_string())?;
            should_run = self
                .builder
                .build_and(should_run, not_reset, "clk_edge_without_reset")
                .map_err(|e| e.to_string())?;
        }

        Ok(should_run)
    }

    fn codegen_clock_edge_body(
        &self,
        function: FunctionValue<'ctx>,
        module_info: &IrModuleInfo,
        proc: &IrClockedProcess,
        scope: CodegenScope<'ctx>,
    ) -> Result<(), String> {
        let mut ctx = BlockCodegen::new(
            self,
            function,
            module_info,
            &proc.locals,
            Stage::Next,
            scope.prev_signals,
            scope.prev_ports.clone(),
            scope.next_signals,
            scope.next_ports.clone(),
        )?;
        ctx.codegen_block(&proc.clock_block)?;
        Ok(())
    }

    fn copy_memory(
        &self,
        dest: PointerValue<'ctx>,
        src: PointerValue<'ctx>,
        ty: StructType<'ctx>,
    ) -> Result<(), String> {
        let size = self.struct_size_i64(ty)?;
        self.builder
            .build_memcpy(dest, 1, src, 1, size)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn branch_if_open(&self, target: BasicBlock<'ctx>) -> Result<(), String> {
        if self
            .builder
            .get_insert_block()
            .is_some_and(|block| block.get_terminator().is_none())
        {
            self.builder
                .build_unconditional_branch(target)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn struct_size_i64(&self, ty: StructType<'ctx>) -> Result<IntValue<'ctx>, String> {
        let size = ty
            .size_of()
            .ok_or_else(|| "cannot compute size of opaque struct".to_owned())?;
        if size.get_type() == self.i64_type {
            Ok(size)
        } else {
            self.builder
                .build_int_cast_sign_flag(size, self.i64_type, false, "struct_size")
                .map_err(|e| e.to_string())
        }
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
        let mut direct = Vec::new();
        for (i, _) in enumerate(&self.modules[self.top_module].ports) {
            let value_ptr = self.struct_field_ptr(top_types.ports_val, ports_val_ptr, i as u32, "port_value")?;
            direct.push(value_ptr);
        }
        Ok(PortPtrs {
            ty: top_types.ports_ptr,
            ptr: self.ptr_type.const_null(),
            direct: Some(direct),
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
        let mut dummy_index = 0u32;
        let mut direct = Vec::new();
        for connection in &instance.port_connections {
            let ptr = match connection.inner {
                IrPortConnection::Input(signal) | IrPortConnection::Output(Some(signal)) => {
                    self.signal_ptr(module_info, signal, signals, ports.clone())?
                }
                IrPortConnection::Output(None) => {
                    let field_index = self.child_dummy_field_index(module_info, child_index, dummy_index);
                    dummy_index += 1;
                    self.struct_field_ptr(self.module_signal_type(module_info), signals, field_index, "dummy")?
                }
            };
            direct.push(ptr);
        }
        Ok(PortPtrs {
            ty: child_ports_ty,
            ptr: self.ptr_type.const_null(),
            direct: Some(direct),
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
            out.push((self.port_ptr(&ports, port_index)?, &port_info.ty));
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
                let child_ports = self.build_child_ports(
                    module_info,
                    signals,
                    ports.clone(),
                    child_index,
                    instance,
                    child_ports_ty,
                )?;
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

    #[allow(clippy::too_many_arguments)]
    fn eval_polarized_signal_cached(
        &self,
        module_info: &IrModuleInfo,
        signal: crate::front::signal::Polarized<IrSignal>,
        key: Option<&PolarizedSignalKey>,
        stage: Stage,
        signals: PointerValue<'ctx>,
        ports: PortPtrs<'ctx>,
        written_signals: &HashSet<SignalKey>,
        signal_value_cache: &mut HashMap<SignalValueCacheKey, IntValue<'ctx>>,
    ) -> Result<IntValue<'ctx>, String> {
        let _ = (key, written_signals, signal_value_cache);
        self.eval_polarized_signal(module_info, signal, stage, signals, ports)
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
            IrSignal::Port(port) => self.port_ptr(&ports, port.inner().index()),
            IrSignal::Wire(wire) => self.wire_ptr(module_info, wire, signals),
        }
    }

    fn port_ptr(&self, ports: &PortPtrs<'ctx>, port_index: usize) -> Result<PointerValue<'ctx>, String> {
        if let Some(direct) = &ports.direct {
            return Ok(direct[port_index]);
        }
        let field = self.struct_field_ptr(ports.ty, ports.ptr, port_index as u32, "port_ptr")?;
        Ok(self
            .builder
            .build_load(self.ptr_type, field, "port_ptr_value")
            .map_err(|e| e.to_string())?
            .into_pointer_value())
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
                let value_ty = expr.ty(self.module_info, self.locals);
                let value = self.eval(expr)?;
                self.codegen_assignment(target, value, &value_ty)?;
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

    fn codegen_assignment(
        &mut self,
        target: &IrAssignmentTarget,
        value: IntValue<'ctx>,
        value_ty: &IrType,
    ) -> Result<(), String> {
        let base = self.assignment_base_ptr(target.base)?;
        let mut offset = self.cg.llvm_type(base.ty)?.const_zero();
        let mut current_ty = base.ty.clone();
        for (step_index, step) in target.steps.iter().enumerate() {
            match step {
                IrTargetStep::ArrayIndex(index) => {
                    let IrType::Array(inner, _) = &current_ty else {
                        return Err("array index assignment target applied to non-array".to_owned());
                    };
                    let index = self
                        .cg
                        .cast_value(self.eval(index)?, base.ty, Signed::Unsigned, "index_cast")?;
                    let inner_bits = self
                        .cg
                        .llvm_type(base.ty)?
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
                    current_ty = inner.as_ref().clone();
                }
                IrTargetStep::ArraySlice { start, len } => {
                    let IrType::Array(inner, _) = &current_ty else {
                        return Err("array slice assignment target applied to non-array".to_owned());
                    };
                    let start = self
                        .cg
                        .cast_value(self.eval(start)?, base.ty, Signed::Unsigned, "slice_start_cast")?;
                    let inner_bits = self
                        .cg
                        .llvm_type(base.ty)?
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
                    if step_index == target.steps.len() - 1 {
                        return self.store_partial(base, offset, &slice_ty, value);
                    }
                    current_ty = slice_ty;
                }
            }
        }
        if target.steps.is_empty() {
            let value = self
                .cg
                .cast_value(value, base.ty, type_signedness(value_ty), "assign_cast")?;
            self.cg
                .builder
                .build_store(base.ptr, value)
                .map_err(|e| e.to_string())?;
        } else {
            self.store_partial(base, offset, &current_ty, value)?;
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
            .build_int_cast_sign_flag(offset, base_type, false, "offset_cast")
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
            .build_int_cast_sign_flag(value, base_type, false, "part_value_cast")
            .map_err(|e| e.to_string())?;
        let value = self
            .cg
            .builder
            .build_and(value, ones, "part_value_masked")
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
                    .signal_ptr(self.module_info, signal, self.next_signals, self.next_ports.clone())?;
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
                let left_ty = left.ty(self.module_info, self.locals);
                let right_ty = right.ty(self.module_info, self.locals);
                let result_width = u32::try_from(&result_ty.size_bits())
                    .map_err(|_| format!("LLVM backend value is too wide: {} bits", result_ty.size_bits()))?
                    .max(1);
                let left_width = u32::try_from(&left_ty.size_bits())
                    .map_err(|_| format!("LLVM backend value is too wide: {} bits", left_ty.size_bits()))?
                    .max(1);
                let right_width = u32::try_from(&right_ty.size_bits())
                    .map_err(|_| format!("LLVM backend value is too wide: {} bits", right_ty.size_bits()))?
                    .max(1);
                let op_signed =
                    type_signedness(&left_ty) == Signed::Signed || type_signedness(&right_ty) == Signed::Signed;
                let left_op_width = if op_signed && type_signedness(&left_ty) == Signed::Unsigned {
                    left_width + 1
                } else {
                    left_width
                };
                let right_op_width = if op_signed && type_signedness(&right_ty) == Signed::Unsigned {
                    right_width + 1
                } else {
                    right_width
                };
                let op_width = result_width.max(left_op_width).max(right_op_width);
                let op_type = self
                    .cg
                    .context
                    .custom_width_int_type(std::num::NonZeroU32::new(op_width).unwrap())
                    .map_err(str::to_owned)?;
                let left = self
                    .cg
                    .builder
                    .build_int_cast_sign_flag(
                        self.eval(left)?,
                        op_type,
                        type_signedness(&left_ty) == Signed::Signed,
                        "lhs",
                    )
                    .map_err(|e| e.to_string())?;
                let right = self
                    .cg
                    .builder
                    .build_int_cast_sign_flag(
                        self.eval(right)?,
                        op_type,
                        type_signedness(&right_ty) == Signed::Signed,
                        "rhs",
                    )
                    .map_err(|e| e.to_string())?;
                let result = match op {
                    IrIntArithmeticOp::Add => self.cg.builder.build_int_add(left, right, "add"),
                    IrIntArithmeticOp::Sub => self.cg.builder.build_int_sub(left, right, "sub"),
                    IrIntArithmeticOp::Mul => self.cg.builder.build_int_mul(left, right, "mul"),
                    IrIntArithmeticOp::Div => match op_signed {
                        true => self.signed_floor_div_rem(left, right, true),
                        false => self.cg.builder.build_int_unsigned_div(left, right, "div"),
                    },
                    IrIntArithmeticOp::Mod => match op_signed {
                        true => self.signed_floor_div_rem(left, right, false),
                        false => self.cg.builder.build_int_unsigned_rem(left, right, "mod"),
                    },
                    IrIntArithmeticOp::Shr => self.cg.builder.build_right_shift(left, right, op_signed, "shr"),
                    IrIntArithmeticOp::Shl => self.cg.builder.build_left_shift(left, right, "shl"),
                    IrIntArithmeticOp::Pow => self.codegen_pow(left, right),
                }
                .map_err(|e| e.to_string())?;
                self.cg.cast_value(
                    result,
                    &result_ty,
                    if op_signed { Signed::Signed } else { Signed::Unsigned },
                    "arith_result",
                )
            }
            IrExpressionLarge::IntCompare(op, left, right) => {
                let left_ty = left.ty(self.module_info, self.locals);
                let right_ty = right.ty(self.module_info, self.locals);
                let width_ty = if left_ty.size_bits() >= right_ty.size_bits() {
                    left_ty.clone()
                } else {
                    right_ty.clone()
                };
                let left = self
                    .cg
                    .cast_value(self.eval(left)?, &width_ty, type_signedness(&left_ty), "cmp_lhs")?;
                let right = self
                    .cg
                    .cast_value(self.eval(right)?, &width_ty, type_signedness(&right_ty), "cmp_rhs")?;
                let signed =
                    type_signedness(&left_ty) == Signed::Signed || type_signedness(&right_ty) == Signed::Signed;
                let pred = match (op, signed) {
                    (IrIntCompareOp::Eq, _) => IntPredicate::EQ,
                    (IrIntCompareOp::Neq, _) => IntPredicate::NE,
                    (IrIntCompareOp::Lt, true) => IntPredicate::SLT,
                    (IrIntCompareOp::Lte, true) => IntPredicate::SLE,
                    (IrIntCompareOp::Gt, true) => IntPredicate::SGT,
                    (IrIntCompareOp::Gte, true) => IntPredicate::SGE,
                    (IrIntCompareOp::Lt, false) => IntPredicate::ULT,
                    (IrIntCompareOp::Lte, false) => IntPredicate::ULE,
                    (IrIntCompareOp::Gt, false) => IntPredicate::UGT,
                    (IrIntCompareOp::Gte, false) => IntPredicate::UGE,
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
                let value_ty = value.ty(self.module_info, self.locals);
                self.cg.cast_value(
                    self.eval(value)?,
                    &IrType::Int(range.clone()),
                    type_signedness(&value_ty),
                    "cast_int_range",
                )
            }
        }
    }

    fn signed_floor_div_rem(
        &self,
        left: IntValue<'ctx>,
        right: IntValue<'ctx>,
        want_div: bool,
    ) -> Result<IntValue<'ctx>, inkwell::builder::BuilderError> {
        let zero = left.get_type().const_zero();
        let one = left.get_type().const_int(1, false);
        let trunc_div = self.cg.builder.build_int_signed_div(left, right, "trunc_div")?;
        let trunc_rem = self.cg.builder.build_int_signed_rem(left, right, "trunc_rem")?;
        let rem_nonzero = self
            .cg
            .builder
            .build_int_compare(IntPredicate::NE, trunc_rem, zero, "rem_nonzero")?;
        let left_neg = self
            .cg
            .builder
            .build_int_compare(IntPredicate::SLT, left, zero, "lhs_neg")?;
        let right_neg = self
            .cg
            .builder
            .build_int_compare(IntPredicate::SLT, right, zero, "rhs_neg")?;
        let signs_differ = self.cg.builder.build_xor(left_neg, right_neg, "signs_differ")?;
        let adjust = self.cg.builder.build_and(rem_nonzero, signs_differ, "floor_adjust")?;
        let div_adjusted = self.cg.builder.build_int_sub(trunc_div, one, "floor_div")?;
        let rem_adjusted = self.cg.builder.build_int_add(trunc_rem, right, "floor_rem")?;
        let result = if want_div {
            self.cg.builder.build_select(adjust, div_adjusted, trunc_div, "div")?
        } else {
            self.cg.builder.build_select(adjust, rem_adjusted, trunc_rem, "mod")?
        };
        Ok(result.into_int_value())
    }

    fn codegen_pow(
        &self,
        base: IntValue<'ctx>,
        exponent: IntValue<'ctx>,
    ) -> Result<IntValue<'ctx>, inkwell::builder::BuilderError> {
        let ty = base.get_type();
        let acc_ptr = self.cg.builder.build_alloca(ty, "pow_acc")?;
        let exp_ptr = self.cg.builder.build_alloca(ty, "pow_exp")?;
        self.cg.builder.build_store(acc_ptr, ty.const_int(1, false))?;
        self.cg.builder.build_store(exp_ptr, exponent)?;

        let cond_block = self.cg.context.append_basic_block(self.function, "pow_cond");
        let body_block = self.cg.context.append_basic_block(self.function, "pow_body");
        let after_block = self.cg.context.append_basic_block(self.function, "pow_after");
        self.cg.builder.build_unconditional_branch(cond_block)?;

        self.cg.builder.position_at_end(cond_block);
        let exp = self
            .cg
            .builder
            .build_load(ty, exp_ptr, "pow_exp_value")?
            .into_int_value();
        let cond = self
            .cg
            .builder
            .build_int_compare(IntPredicate::NE, exp, ty.const_zero(), "pow_continue")?;
        self.cg
            .builder
            .build_conditional_branch(cond, body_block, after_block)?;

        self.cg.builder.position_at_end(body_block);
        let acc = self
            .cg
            .builder
            .build_load(ty, acc_ptr, "pow_acc_value")?
            .into_int_value();
        let acc = self.cg.builder.build_int_mul(acc, base, "pow_acc_next")?;
        self.cg.builder.build_store(acc_ptr, acc)?;
        let exp = self
            .cg
            .builder
            .build_load(ty, exp_ptr, "pow_exp_value")?
            .into_int_value();
        let exp = self
            .cg
            .builder
            .build_int_sub(exp, ty.const_int(1, false), "pow_exp_next")?;
        self.cg.builder.build_store(exp_ptr, exp)?;
        self.cg.builder.build_unconditional_branch(cond_block)?;

        self.cg.builder.position_at_end(after_block);
        Ok(self.cg.builder.build_load(ty, acc_ptr, "pow_result")?.into_int_value())
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
            Stage::Prev => (self.prev_signals, self.prev_ports.clone()),
            Stage::Next => (self.next_signals, self.next_ports.clone()),
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
