use crate::front::diagnostic::{DiagError, DiagResult, DiagnosticError, DiagnosticWarning, Diagnostics};
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrExpressionLarge,
    IrForStatement, IrIfStatement, IrModuleChild, IrModuleInfo, IrPortConnection, IrSignal, IrSignalOrVariable,
    IrSignals, IrStatement, IrStringPiece, IrStringSubstitution, IrStructType, IrType, IrVariables,
};
use crate::mid::steps::{IrTargetStepScalar, IrTargetStepSlice, IrTargetSteps};
use crate::syntax::pos::Span;
use crate::util::big_int::{AnyInt, BigUint, IsZero};
use crate::util::data::{IndexMapExt, chain_keys};
use crate::util::iter::IterExt;
use crate::util::range::{ClosedNonEmptyRange, ClosedRange};
use crate::util::sparse_change_array::SparseChangeArray;
use crate::util::{Never, ResultNeverExt};
use hwl_util::constants::MAX_DRIVER_INFO_PATHS;
use hwl_util::swrite;
use indexmap::{IndexMap, IndexSet};
use itertools::{Itertools, enumerate};
use std::fmt::Debug;
use std::ops::ControlFlow;
use unwrap_match::unwrap_match;

// TODO check for cycles somewhere, maybe even here
// TODO record instance cones
pub fn compute_module_cones(diags: &Diagnostics, module: &IrModuleInfo) -> DiagResult {
    let signals = &module.signals;
    let mut combined_drivers: IndexMap<IrSignal, IrMask<Vec<Span>>> = IndexMap::new();

    let mut report_signal_drive = |signal: IrSignal, mask: IrMask<bool>, span: Span| {
        let combined_drive = combined_drivers
            .entry(signal)
            .or_insert_with(|| IrMask::new(signal.ty(signals), vec![]));
        IrMask::zip2_for_each_mut(combined_drive, &mask, &mut |combined, &curr_drives| {
            if curr_drives {
                combined.push(span);
            }
        })
    };

    let mut any_err = Ok(());
    for child in &module.children {
        match &child.inner {
            IrModuleChild::ClockedProcess(proc) => {
                let cones = match compute_clocked_process_cones(diags, module, proc) {
                    Ok(d) => d,
                    Err(e) => {
                        any_err = Err(e);
                        continue;
                    }
                };
                for (signal, mask) in cones.drives {
                    report_signal_drive(signal, mask, child.span);
                }
            }
            IrModuleChild::CombinatorialProcess(proc) => {
                let cones = match compute_combinatorial_process_cones(diags, module, proc) {
                    Ok(d) => d,
                    Err(e) => {
                        any_err = Err(e);
                        continue;
                    }
                };

                // check self-loops
                match check_combinatorial_self_loops(diags, signals, child.span, &cones) {
                    Ok(()) => {}
                    Err(e) => {
                        any_err = Err(e);
                        continue;
                    }
                }

                for (signal, mask) in cones.drives {
                    report_signal_drive(signal, mask, child.span);
                }
            }
            IrModuleChild::ModuleInternalInstance(inst) => {
                for conn in &inst.port_connections {
                    match conn.inner {
                        IrPortConnection::Input(_) => {}
                        IrPortConnection::Output(signal) => {
                            if let Some(signal) = signal {
                                report_signal_drive(signal, IrMask::new(signal.ty(signals), true), conn.span);
                            }
                        }
                    }
                }
            }
            IrModuleChild::ModuleExternalInstance(inst) => {
                for (_, (_, conn)) in &inst.port_connections {
                    match conn.inner {
                        IrPortConnection::Input(_) => {}
                        IrPortConnection::Output(signal) => {
                            if let Some(signal) = signal {
                                report_signal_drive(signal, IrMask::new(signal.ty(signals), true), conn.span);
                            }
                        }
                    }
                }
            }
        }
    }
    any_err?;

    for signal in signals.all_signals_except_inputs() {
        // track driver status
        let mut any_undriven = false;
        let mut all_undriven = true;
        let mut any_overlapping = IndexSet::new();
        let mut all_overlapping = true;

        if let Some(mask) = combined_drivers.get(&signal) {
            mask.for_each_leaf(|v| {
                match v.len() {
                    0 => {
                        any_undriven = true;
                        all_overlapping = false;
                    }
                    1 => {
                        all_undriven = false;
                        all_overlapping = false;
                    }
                    _ => {
                        all_undriven = false;
                        any_overlapping.extend(v.iter().copied())
                    }
                }
                ControlFlow::Continue(())
            })
            .remove_never();
        } else {
            any_undriven = true;
        }

        // get some debug info
        let signal_kind = signal.debug_info_kind();
        let signal_name = signal.debug_info_name(signals);

        // maybe report multiple drivers
        if !any_overlapping.is_empty() {
            let (title_suffix, msg_info) = if all_overlapping {
                ("has multiple drivers", None)
            } else {
                let mask = combined_drivers.get(&signal).unwrap().clone();
                let msg_info = build_driver_info_message(
                    "parts with multiple drivers",
                    signal.ty(signals),
                    signal_name,
                    mask,
                    |v| v.len() > 1,
                );
                ("has multiple overlapping drivers", Some(msg_info))
            };

            let mut diag = DiagnosticError::new(
                format!("{signal_kind} `{signal_name}` {title_suffix}"),
                signal.span(signals),
                format!("{signal_kind} defined here"),
            );
            for drive_span in any_overlapping {
                diag = diag.add_info(drive_span, "driven here");
            }
            if let Some(msg_info) = msg_info {
                diag = diag.add_footer_info(msg_info);
            }

            any_err = Err(diag.report(diags));
        }

        // report warning for (partially) undriven signals
        if any_undriven {
            let (title_suffix, msg_info) = if all_undriven {
                ("has no driver", None)
            } else {
                let mask = combined_drivers.get(&signal).unwrap().clone();
                let msg_info = build_driver_info_message(
                    "parts without a driver",
                    signal.ty(signals),
                    signal_name,
                    mask,
                    Vec::is_empty,
                );
                ("is not fully driven", Some(msg_info))
            };

            let mut diag = DiagnosticWarning::new(
                format!("{signal_kind} `{signal_name}` {title_suffix}"),
                signal.span(signals),
                format!("{signal_kind} defined here"),
            );
            if let Some(msg_info) = msg_info {
                diag = diag.add_footer_info(msg_info);
            }
            diag.report(diags);
        }
    }

    any_err
}

fn check_combinatorial_self_loops(
    diags: &Diagnostics,
    signals: &IrSignals,
    proc_span: Span,
    cones: &Cones,
) -> DiagResult {
    // find any signals with self loops
    let mut loop_signals = IndexSet::new();
    for (&signal, read) in &cones.reads {
        let Some(write) = cones.drives.get(&signal) else {
            continue;
        };

        let mut any_overlap = false;
        IrMask::zip2_for_each(read, write, &mut |r, w| any_overlap |= *r & *w);

        if any_overlap {
            loop_signals.insert(signal);
        }
    }

    // report error if necessary
    if !loop_signals.is_empty() {
        let mut diag = DiagnosticError::new("combinatorial self-loop", proc_span, "for this combinatorial process");

        // visit signals in module declaration order instead of whatever order we collected them in
        for signal in signals.all_signals() {
            let signal_kind = signal.debug_info_kind();
            let signal_name = signal.debug_info_name(signals);

            if loop_signals.contains(&signal) {
                diag = diag.add_info(signal.span(signals), format!("for {signal_kind} `{signal_name}`"));
            }
        }

        return Err(diag
            .add_footer_hint("This does not create valid combinatorial logic.")
            .report(diags));
    }

    Ok(())
}

fn build_driver_info_message<T: Eq + Debug>(
    msg: &str,
    signal_ty: &IrType,
    signal_name: &str,
    mut mask: IrMask<T>,
    mut cond: impl FnMut(&T) -> bool,
) -> String {
    mask.canonicalize();

    let mut result = format!("{msg}:");
    let mut path_count: usize = 0;

    let _: ControlFlow<()> = mask.for_each_path(signal_ty, |prefix, scalar| {
        if cond(scalar) {
            swrite!(&mut result, "\n    {signal_name}{prefix}");
            path_count += 1;

            if path_count >= MAX_DRIVER_INFO_PATHS {
                swrite!(&mut result, "\n    {signal_name}...");
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    });

    result
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum IrMask<T> {
    Scalar(T),
    Array(Box<SparseChangeArray<IrMask<T>>>),
    TupleOrStruct(Vec<IrMask<T>>),
}

#[derive(Debug)]
pub struct Drives<'p> {
    parent: Option<&'p Drives<'p>>,
    map: IndexMap<IrSignal, IrMask<bool>>,
}

fn compute_clocked_process_cones(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    proc: &IrClockedProcess,
) -> DiagResult<Cones> {
    let IrClockedProcess {
        registers,
        variables,
        async_reset: _,
        clock_signal: _,
        clock_block,
    } = proc;

    let cones_raw = compute_block_cones(diags, module, ProcessKind::Clocked, variables, clock_block)?;

    // the actual drives are just all the registers declared by this process
    // TODO should we track more specific drives anyway?
    let drives = registers
        .iter()
        .map(|&signal| (signal, IrMask::new(signal.ty(&module.signals), true)))
        .collect();

    Ok(Cones {
        reads: cones_raw.reads,
        drives,
    })
}

#[derive(Debug)]
pub struct Cones {
    // Reads only count if they read values that have not yet been written yet.
    reads: IndexMap<IrSignal, IrMask<bool>>,
    drives: IndexMap<IrSignal, IrMask<bool>>,
}

fn compute_combinatorial_process_cones(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    proc: &IrCombinatorialProcess,
) -> DiagResult<Cones> {
    let IrCombinatorialProcess { variables, block } = proc;
    compute_block_cones(diags, module, ProcessKind::Combinatorial, variables, block)
}

#[derive(Debug, Copy, Clone)]
enum ProcessKind {
    Clocked,
    Combinatorial,
}

#[derive(Debug, Copy, Clone)]
enum DynamicDriveError {
    ArrayIndex,
    ArraySlice,
}

fn compute_block_cones(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    process_kind: ProcessKind,
    vars: &IrVariables,
    block: &IrBlock,
) -> DiagResult<Cones> {
    let mut signal_errors = IndexMap::new();
    let mut reads = IndexMap::new();
    let mut drives = Drives::root();

    compute_block_cones_impl(
        diags,
        module,
        process_kind,
        vars,
        block,
        &mut signal_errors,
        &mut reads,
        &mut drives,
    );

    if let Some(&e) = signal_errors.values().next() {
        Err(e)
    } else {
        Ok(Cones {
            reads,
            drives: drives.map,
        })
    }
}

fn compute_block_cones_impl(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    process_kind: ProcessKind,
    vars: &IrVariables,
    block: &IrBlock,
    signal_errors: &mut IndexMap<IrSignal, DiagError>,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
    drives: &mut Drives,
) {
    let IrBlock { statements } = block;
    for stmt in statements {
        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                let &IrAssignmentTarget { base, ref steps } = target;
                let IrTargetSteps {
                    steps_scalar,
                    step_slice,
                } = steps;

                // record reads
                for step in steps_scalar {
                    match step {
                        IrTargetStepScalar::ArrayIndex(index) => {
                            record_expression_reads(module, vars, drives, reads, index)
                        }
                        &IrTargetStepScalar::TupleIndex(_) | &IrTargetStepScalar::StructField(_) => {}
                    }
                }
                if let Some(step_slice) = step_slice {
                    let IrTargetStepSlice { start, len: _ } = step_slice;
                    record_expression_reads(module, vars, drives, reads, start);
                }
                record_expression_reads(module, vars, drives, reads, source);

                // record signal writes
                // ignore if this signal already has an error, to avoid noise
                let base = match base {
                    IrSignalOrVariable::Signal(base) => {
                        if signal_errors.contains_key(&base) {
                            continue;
                        }
                        base
                    }
                    IrSignalOrVariable::Variable(_) => continue,
                };

                let curr = drives.curr_mut(&module.signals, base);
                match curr.write_steps(module, process_kind, vars, steps) {
                    Ok(()) => {}
                    Err(e) => {
                        // TODO expand this report to include much more information,
                        //   in particular which step is causing the issue and which parts were not yet driven
                        // TODO include examples in .[]. format?
                        let kind_str = match e {
                            DynamicDriveError::ArrayIndex => "index",
                            DynamicDriveError::ArraySlice => "slice",
                        };

                        let e = DiagnosticError::new(
                            format!("dynamic array {kind_str} assignment to not yet driven signal"),
                            stmt.span,
                            format!("dynamic array {kind_str} assignment here"),
                        )
                            .add_footer_hint("This does not create valid combinatorial logic.\nUse a constant target or do a default full assignment beforehand.")
                            .report(diags);
                        signal_errors.insert_first(base, e);
                    }
                }
            }
            IrStatement::Block(stmt_inner) => {
                compute_block_cones_impl(
                    diags,
                    module,
                    process_kind,
                    vars,
                    stmt_inner,
                    signal_errors,
                    reads,
                    drives,
                );
            }
            IrStatement::If(stmt_inner) => {
                let IrIfStatement {
                    condition,
                    then_block,
                    else_block,
                } = stmt_inner;

                // record reads
                record_expression_reads(module, vars, drives, reads, condition);

                // visit then
                let mut then_drivers = drives.new_child();
                compute_block_cones_impl(
                    diags,
                    module,
                    process_kind,
                    vars,
                    then_block,
                    signal_errors,
                    reads,
                    &mut then_drivers,
                );
                let then_map = then_drivers.map;

                // visit else
                let mut else_drivers = drives.new_child();
                if let Some(else_block) = else_block {
                    compute_block_cones_impl(
                        diags,
                        module,
                        process_kind,
                        vars,
                        else_block,
                        signal_errors,
                        reads,
                        &mut else_drivers,
                    );
                }
                let else_map = else_drivers.map;

                // merge
                for &signal in chain_keys(&then_map, &else_map) {
                    // skip signals that already have errors
                    if signal_errors.contains_key(&signal) {
                        continue;
                    }

                    let mut any_mismatch = false;
                    IrMask::zip3_for_each_mut(
                        drives.curr_mut(&module.signals, signal),
                        then_map.get(&signal),
                        else_map.get(&signal),
                        &mut |parent, then_value, else_value| {
                            // if parent already drives, children don't matter
                            if *parent {
                                return;
                            }

                            // check that children match
                            let then_value = then_value.copied().unwrap_or(false);
                            let else_value = else_value.copied().unwrap_or(false);
                            any_mismatch |= then_value != else_value;

                            // store union, assume driven because:
                            // * for clocked blocks, this is correct
                            // * for combinatorial blocks, this avoids false negatives in driver tracking
                            *parent |= then_value | else_value;
                        },
                    );

                    let allow_driver_mismatch = match process_kind {
                        ProcessKind::Clocked => true,
                        ProcessKind::Combinatorial => false,
                    };
                    if any_mismatch && !allow_driver_mismatch {
                        let e = DiagnosticError::new(
                            "driver mismatch between conditional branches",
                            stmt.span,
                            "conditional branch here",
                        )
                            .add_info(signal.span(&module.signals), "for this signal")
                            .add_footer_hint("This does not create valid combinatorial logic.\nEnsure both branches drive the same signal (subset) or do a default full assignment beforehand.")
                            .report(diags);
                        signal_errors.insert_first(signal, e);
                    }
                }
            }
            IrStatement::For(stmt_inner) => {
                let IrForStatement { index: _, range, block } = stmt_inner;
                if range.is_empty() {
                    // block will never execute, so it also can't drive or read anything
                } else {
                    // this is pessimistic, but safe:
                    //   we do not use the fact that the index variable is known in each iteration,
                    //   which considers both reads and writes to potentially access too much
                    compute_block_cones_impl(diags, module, process_kind, vars, block, signal_errors, reads, drives);
                }
            }
            IrStatement::Print(pieces) => {
                // record reads
                for piece in pieces {
                    match piece {
                        IrStringPiece::Literal(_) => {}
                        IrStringPiece::Substitute(sub) => match sub {
                            IrStringSubstitution::Integer(expr, _radix) => {
                                record_expression_reads(module, vars, drives, reads, expr);
                            }
                        },
                    }
                }

                // does not drive anything
            }
            IrStatement::AssertFailed => {
                // does not read or drive anything
            }
        }
    }
}

fn record_expression_reads(
    module: &IrModuleInfo,
    vars: &IrVariables,
    drivers: &Drives,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
    expr: &IrExpression,
) {
    match expr {
        // constants, no reads
        IrExpression::Bool(_) | IrExpression::Int(_) => {}
        // we don't care about variable reads
        IrExpression::Variable(_) => {}

        // read the entire signal
        &IrExpression::Signal(signal) => {
            record_signal_reads(module, vars, drivers, reads, signal, &IrTargetSteps::new())
        }

        // For steps, read all step operands.
        // For the base signal only read the pieces that can actually be read.
        &IrExpression::Large(expr_large) => match &module.large[expr_large] {
            IrExpressionLarge::Steps { base, steps } => {
                // read step operands
                let IrTargetSteps {
                    steps_scalar,
                    step_slice,
                } = steps;
                for step in steps_scalar {
                    match step {
                        IrTargetStepScalar::ArrayIndex(index) => {
                            record_expression_reads(module, vars, drivers, reads, index)
                        }
                        &IrTargetStepScalar::TupleIndex(index) | &IrTargetStepScalar::StructField(index) => {
                            let _: usize = index;
                        }
                    }
                }
                if let Some(step) = step_slice {
                    let IrTargetStepSlice { start, len } = step;
                    record_expression_reads(module, vars, drivers, reads, start);
                    let _: &BigUint = len;
                }

                // read base
                if let &IrExpression::Signal(base) = base {
                    // for signals, only read the pieces that can actually be read
                    record_signal_reads(module, vars, drivers, reads, base, steps);
                } else {
                    // for other expressions, just read everything
                    record_expression_reads(module, vars, drivers, reads, base);
                }
            }

            // just read operands
            IrExpressionLarge::Undefined(_)
            | IrExpressionLarge::BoolNot(_)
            | IrExpressionLarge::BoolBinary(_, _, _)
            | IrExpressionLarge::IntArithmetic(_, _, _, _)
            | IrExpressionLarge::IntCompare(_, _, _)
            | IrExpressionLarge::TupleLiteral(_)
            | IrExpressionLarge::ArrayLiteral(_, _, _)
            | IrExpressionLarge::StructLiteral(_, _)
            | IrExpressionLarge::EnumLiteral(_, _, _)
            | IrExpressionLarge::EnumTag { .. }
            | IrExpressionLarge::EnumPayload { .. }
            | IrExpressionLarge::ToBits(_, _)
            | IrExpressionLarge::FromBits(_, _)
            | IrExpressionLarge::ExpandIntRange(_, _)
            | IrExpressionLarge::ConstrainIntRange(_, _) => {
                expr.for_each_operand(&module.large, &mut |operand| {
                    record_expression_reads(module, vars, drivers, reads, operand)
                });
            }
        },
    }
}

fn record_signal_reads(
    module: &IrModuleInfo,
    vars: &IrVariables,
    drivers: &Drives,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
    signal: IrSignal,
    steps: &IrTargetSteps,
) {
    let IrTargetSteps {
        steps_scalar,
        step_slice,
    } = steps;

    let signal_reads = reads
        .entry(signal)
        .or_insert_with(|| IrMask::new(signal.ty(&module.signals), false));
    let signal_driven = drivers.curr(signal);

    // TODO this is cursed, build proper zip iterators, including simple "constant" iterators that can be mixed?
    let mut pieces_driven = signal_driven.map(|signal_driven| {
        let mut pieces_driven = vec![];
        signal_driven.for_each_possible_leaf_after_steps(module, vars, steps_scalar, step_slice.as_ref(), &mut |m| {
            pieces_driven.push(m)
        });
        pieces_driven.into_iter()
    });
    signal_reads.for_each_possible_leaf_after_steps_mut(
        module,
        vars,
        steps_scalar,
        step_slice.as_ref(),
        &mut |piece_read| {
            if let Some(pieces_driven) = &mut pieces_driven {
                // only count reads from pieces that have not yet been driven
                let piece_driven = pieces_driven.next().unwrap();
                IrMask::zip2_for_each_mut(piece_read, piece_driven, &mut |scalar_read, scalar_driven| {
                    *scalar_read |= !scalar_driven;
                })
            } else {
                // not driven at all, count as read
                piece_read.fill(true);
            }
        },
    );

    assert!(pieces_driven.is_none_or(IterExt::is_empty));
}

impl Drives<'_> {
    fn root() -> Drives<'static> {
        Drives {
            parent: None,
            map: IndexMap::new(),
        }
    }

    fn new_child(&self) -> Drives<'_> {
        Drives {
            parent: Some(self),
            map: IndexMap::new(),
        }
    }

    fn curr(&self, signal: IrSignal) -> Option<&IrMask<bool>> {
        let mut curr = self;
        loop {
            if let Some(mask) = curr.map.get(&signal) {
                return Some(mask);
            }
            curr = curr.parent?;
        }
    }

    fn curr_mut(&mut self, signals: &IrSignals, signal: IrSignal) -> &mut IrMask<bool> {
        self.map.entry(signal).or_insert_with(|| {
            let mut curr = self.parent;
            loop {
                curr = match curr {
                    None => {
                        break IrMask::new(signal.ty(signals), false);
                    }
                    Some(curr) => {
                        if let Some(d) = curr.map.get(&signal) {
                            break d.clone();
                        } else {
                            curr.parent
                        }
                    }
                };
            }
        })
    }
}

// TODO rework/refactor this to be more elegant
// TODO move to separate module
impl<T> IrMask<T> {
    fn new(ty: &IrType, init: T) -> IrMask<T>
    where
        T: Clone,
    {
        match ty {
            IrType::Bool | IrType::Int(_) | IrType::Enum(_) => IrMask::Scalar(init),
            IrType::Array(ty_inner, len) => {
                let array = if len.is_zero() {
                    SparseChangeArray::new_empty()
                } else {
                    let mask_inner = IrMask::new(ty_inner, init);
                    SparseChangeArray::new(len.clone(), mask_inner)
                };
                IrMask::Array(Box::new(array))
            }
            IrType::Tuple(ty_fields) => {
                let mask_fields = ty_fields
                    .iter()
                    .map(|ty_field| IrMask::new(ty_field, init.clone()))
                    .collect_vec();
                IrMask::TupleOrStruct(mask_fields)
            }
            IrType::Struct(ty_info) => {
                let IrStructType {
                    ty: _,
                    debug_info_name: _,
                    fields,
                } = ty_info;
                let mask_fields = fields
                    .values()
                    .map(|ty_field| IrMask::new(ty_field, init.clone()))
                    .collect_vec();
                IrMask::TupleOrStruct(mask_fields)
            }
        }
    }

    fn canonicalize(&mut self)
    where
        T: Eq,
    {
        match self {
            IrMask::Scalar(_) => {}
            IrMask::Array(slf) => {
                slf.for_each_block_mut(|_, inner| {
                    inner.canonicalize();
                    ControlFlow::Continue(())
                })
                .remove_never();
                slf.canonicalize()
            }
            IrMask::TupleOrStruct(slf) => slf.iter_mut().for_each(Self::canonicalize),
        }
    }

    fn for_each_path<B>(&self, ty: &IrType, mut f: impl FnMut(&str, &T) -> ControlFlow<B>) -> ControlFlow<B>
    where
        T: Debug,
    {
        fn g<T: Debug, B>(
            slf: &IrMask<T>,
            ty: &IrType,
            f: &mut impl FnMut(&str, &T) -> ControlFlow<B>,
            prefix: &mut String,
        ) -> ControlFlow<B> {
            let prefix_len = prefix.len();
            match ty {
                IrType::Bool | IrType::Int(_) | IrType::Enum(_) => {
                    let slf = unwrap_match!(slf, IrMask::Scalar(slf) => slf);
                    f(prefix, slf)?;
                }
                IrType::Array(ty_inner, _) => {
                    let slf = unwrap_match!(slf, IrMask::Array(slf) => slf);
                    slf.for_each_block(|range, mask| {
                        if let Some(single) = range.as_single() {
                            swrite!(prefix, "[{single}]")
                        } else {
                            swrite!(prefix, "[{range}]")
                        };
                        g(mask, ty_inner, f, prefix)?;
                        prefix.truncate(prefix_len);
                        ControlFlow::Continue(())
                    })?;
                }
                IrType::Tuple(ty_fields) => {
                    let slf = unwrap_match!(slf, IrMask::TupleOrStruct(slf) => slf);
                    assert_eq!(slf.len(), ty_fields.len());

                    for (field_index, field_ty) in enumerate(ty_fields) {
                        swrite!(prefix, ".{field_index}");
                        g(&slf[field_index], field_ty, f, prefix)?;
                        prefix.truncate(prefix_len);
                    }
                }
                IrType::Struct(ty_info) => {
                    let slf = unwrap_match!(slf, IrMask::TupleOrStruct(slf) => slf);
                    assert_eq!(slf.len(), ty_info.fields.len());

                    for (field_index, (field_name, field_ty)) in enumerate(&ty_info.fields) {
                        swrite!(prefix, ".{field_name}");
                        g(&slf[field_index], field_ty, f, prefix)?;
                        prefix.truncate(prefix_len);
                    }
                }
            }

            ControlFlow::Continue(())
        }

        let mut prefix = String::new();
        g(self, ty, &mut f, &mut prefix)
    }

    fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.for_each_leaf_mut(|leaf| {
            *leaf = value.clone();
            ControlFlow::Continue(())
        })
        .remove_never();
    }

    fn for_each_leaf<B>(&self, mut f: impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
        fn for_each_leaf_impl<T, B>(slf: &IrMask<T>, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
            match slf {
                IrMask::Scalar(slf) => f(slf),
                IrMask::Array(slf) => slf.for_each_block(|_, m| for_each_leaf_impl(m, f)),
                IrMask::TupleOrStruct(values) => values.iter().try_for_each(|m| for_each_leaf_impl(m, f)),
            }
        }
        for_each_leaf_impl(self, &mut f)
    }

    fn for_each_leaf_mut<B>(&mut self, mut f: impl FnMut(&mut T) -> ControlFlow<B>) -> ControlFlow<B> {
        fn for_each_leaf_mut_impl<T, B>(
            slf: &mut IrMask<T>,
            f: &mut impl FnMut(&mut T) -> ControlFlow<B>,
        ) -> ControlFlow<B> {
            match slf {
                IrMask::Scalar(slf) => f(slf),
                IrMask::Array(slf) => slf.for_each_block_mut(|_, m| for_each_leaf_mut_impl(m, f)),
                IrMask::TupleOrStruct(values) => values.iter_mut().try_for_each(|m| for_each_leaf_mut_impl(m, f)),
            }
        }
        for_each_leaf_mut_impl(self, &mut f)
    }

    fn any(&self, mut f: impl FnMut(&T) -> bool) -> bool {
        self.for_each_leaf(|v| {
            if f(v) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .is_break()
    }

    fn all(&self, mut f: impl FnMut(&T) -> bool) -> bool {
        !self.any(|v| !f(v))
    }

    fn zip2_for_each<U: Debug>(a: &IrMask<T>, b: &IrMask<U>, f: &mut impl FnMut(&T, &U)) {
        match a {
            IrMask::Scalar(a) => {
                let b = unwrap_match!(b, IrMask::Scalar(b) => b);
                f(a, b)
            }
            IrMask::Array(a) => {
                let b = unwrap_match!(b, IrMask::Array(b) => b);
                SparseChangeArray::zip2_for_each_block::<_, Never>(a, b, |_, a, b| {
                    Self::zip2_for_each(a, b, f);
                    ControlFlow::Continue(())
                })
                .remove_never();
            }
            IrMask::TupleOrStruct(a) => {
                let b = unwrap_match!(b, IrMask::TupleOrStruct(b) => b);
                for i in 0..a.len() {
                    Self::zip2_for_each(&a[i], &b[i], f);
                }
            }
        }
    }

    fn zip2_for_each_mut<U: Debug>(a: &mut IrMask<T>, b: &IrMask<U>, f: &mut impl FnMut(&mut T, &U))
    where
        T: Clone,
    {
        IrMask::zip3_for_each_mut::<U, Never>(a, Some(b), None, &mut |a, b, c| {
            let b = b.unwrap();
            match c {
                None => {}
                Some(never) => never.unreachable(),
            }
            f(a, b)
        })
    }

    fn zip3_for_each_mut<U: Debug, V: Debug>(
        a: &mut IrMask<T>,
        b: Option<&IrMask<U>>,
        c: Option<&IrMask<V>>,
        f: &mut impl FnMut(&mut T, Option<&U>, Option<&V>),
    ) where
        T: Clone,
    {
        match a {
            IrMask::Scalar(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::Scalar(b) => b));
                let c = c.map(|c| unwrap_match!(c, IrMask::Scalar(c) => c));
                f(a, b, c)
            }
            IrMask::Array(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::Array(b) => &**b));
                let c = c.map(|c| unwrap_match!(c, IrMask::Array(c) => &**c));
                SparseChangeArray::zip3_for_each_block_mut::<_, _, Never>(a, b, c, |_, a, b, c| {
                    Self::zip3_for_each_mut(a, b, c, f);
                    ControlFlow::Continue(())
                })
                .remove_never();
            }
            IrMask::TupleOrStruct(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::TupleOrStruct(b) => b));
                let c = c.map(|c| unwrap_match!(c, IrMask::TupleOrStruct(c) => c));

                for i in 0..a.len() {
                    let b = b.map(|b| &b[i]);
                    let c = c.map(|c| &c[i]);
                    Self::zip3_for_each_mut(&mut a[i], b, c, f);
                }
            }
        }
    }
}

impl IrMask<bool> {
    fn write_steps(
        &mut self,
        module: &IrModuleInfo,
        process_kind: ProcessKind,
        vars: &IrVariables,
        steps: &IrTargetSteps,
    ) -> Result<(), DynamicDriveError> {
        let IrTargetSteps {
            steps_scalar,
            step_slice,
        } = steps;
        self.write_steps_impl(module, process_kind, vars, steps_scalar, step_slice.as_ref())
    }

    fn write_steps_impl(
        &mut self,
        module: &IrModuleInfo,
        process_kind: ProcessKind,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
    ) -> Result<(), DynamicDriveError> {
        let Some((step_curr, steps_scalar)) = steps_scalar.split_first() else {
            // no remaining scalar steps, only maybe the final scalar step left
            match step_slice {
                None => {
                    self.fill(true);
                }
                Some(step_slice) => {
                    let IrTargetStepSlice { start, len } = step_slice;
                    let len_1 = match len.sub_1() {
                        Ok(len_1) => len_1,
                        Err(IsZero) => return Ok(()),
                    };

                    let slf = unwrap_match!(self, IrMask::Array(slf) => slf);
                    let start_int = unwrap_uint(module, vars, start);

                    match start_int {
                        IntKind::Single(start) => {
                            let range_full = ClosedRange {
                                start: &start,
                                end: &(&start + len),
                            };
                            slf.for_each_block_in_range_mut(range_full, |_, v| {
                                v.fill(true);
                                ControlFlow::Continue(())
                            })
                            .remove_never();
                        }
                        IntKind::Range(start_range) => {
                            let full_range = ClosedRange {
                                start: &start_range.start,
                                end: &(&start_range.end + len_1),
                            };
                            IrMask::write_steps_impl_dyn(slf, full_range, module, process_kind, vars, &[], None)
                                .map_err(|()| DynamicDriveError::ArraySlice)?;
                        }
                    }
                }
            }

            return Ok(());
        };

        match step_curr {
            IrTargetStepScalar::ArrayIndex(index) => {
                let slf = unwrap_match!(self, IrMask::Array(slf) => slf);
                let index_int = unwrap_uint(module, vars, index);

                match index_int {
                    IntKind::Single(index) => {
                        let end = index.next();
                        let range = ClosedRange {
                            start: &index,
                            end: &end,
                        };
                        let mut result = Ok(());
                        slf.for_each_block_in_range_mut(range, |_, m| {
                            result = result.and_then(|()| {
                                m.write_steps_impl(module, process_kind, vars, steps_scalar, step_slice)
                            });
                            ControlFlow::Continue(())
                        })
                        .remove_never();
                        result?;
                    }
                    IntKind::Range(index_range) => {
                        let full_range = ClosedRange::from(index_range.as_ref());
                        IrMask::write_steps_impl_dyn(
                            slf,
                            full_range,
                            module,
                            process_kind,
                            vars,
                            steps_scalar,
                            step_slice,
                        )
                        .map_err(|()| DynamicDriveError::ArrayIndex)?;
                    }
                }
            }
            &IrTargetStepScalar::TupleIndex(index) | &IrTargetStepScalar::StructField(index) => {
                let slf = unwrap_match!(self, IrMask::TupleOrStruct(slf) => slf);
                slf[index].write_steps_impl(module, process_kind, vars, steps_scalar, step_slice)?;
            }
        }

        Ok(())
    }

    fn write_steps_impl_dyn(
        full_array: &mut SparseChangeArray<IrMask<bool>>,
        full_range: ClosedRange<&BigUint>,
        module: &IrModuleInfo,
        process_kind: ProcessKind,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
    ) -> Result<(), ()> {
        match process_kind {
            ProcessKind::Clocked => {
                // always allowed, report everything that is maybe driven
                full_array
                    .for_each_block_in_range_mut(full_range, |_, m| {
                        m.for_each_possible_leaf_after_steps_mut(module, vars, steps_scalar, step_slice, &mut |s| {
                            s.fill(true)
                        });
                        ControlFlow::Continue(())
                    })
                    .remove_never();
            }
            ProcessKind::Combinatorial => {
                // only allowed if everything that could be driven is already driven
                let fully_driven = full_array
                    .for_each_block_in_range(full_range, |_, m| {
                        let fully_driven = m.all_after_steps(module, vars, steps_scalar, step_slice);
                        if fully_driven {
                            ControlFlow::Continue(())
                        } else {
                            ControlFlow::Break(())
                        }
                    })
                    .is_continue();

                if !fully_driven {
                    return Err(());
                }

                // we don't need to set anything,
                //   we've just checked that everything that could be written was already written
            }
        }

        Ok(())
    }

    fn all_after_steps(
        &self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
    ) -> bool {
        let mut all = true;
        self.for_each_possible_leaf_after_steps(module, vars, steps_scalar, step_slice, &mut |m| all &= m.all(|&b| b));
        all
    }

    fn for_each_possible_leaf_after_steps_mut(
        &mut self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
        f: &mut impl FnMut(&mut IrMask<bool>),
    ) {
        let Some((step_curr, steps_scalar)) = steps_scalar.split_first() else {
            match step_slice {
                None => {
                    f(self);
                }
                Some(step_slice) => {
                    let IrTargetStepSlice { start, len } = step_slice;
                    let len_1 = match len.sub_1() {
                        Ok(len_1) => len_1,
                        Err(IsZero) => return,
                    };

                    let slf = unwrap_match!(self, IrMask::Array(slf) => slf);
                    let start_range = unwrap_uint_range(module, vars, start);

                    let full_range = ClosedRange {
                        start: &start_range.start,
                        end: &(start_range.end + len_1),
                    };
                    slf.for_each_block_in_range_mut(full_range, |_, m| {
                        f(m);
                        ControlFlow::Continue(())
                    })
                    .remove_never();
                }
            }
            return;
        };

        match step_curr {
            IrTargetStepScalar::ArrayIndex(index) => {
                let slf = unwrap_match!(self, IrMask::Array(slf) => slf);
                let index_range = unwrap_uint_range(module, vars, index);

                slf.for_each_block_in_range_mut(ClosedRange::from(index_range.as_ref()), |_, m| {
                    m.for_each_possible_leaf_after_steps_mut(module, vars, steps_scalar, step_slice, f);
                    ControlFlow::Continue(())
                })
                .remove_never();
            }
            &IrTargetStepScalar::TupleIndex(index) | &IrTargetStepScalar::StructField(index) => {
                let slf = unwrap_match!(self, IrMask::TupleOrStruct(slf) => slf);
                slf[index].for_each_possible_leaf_after_steps_mut(module, vars, steps_scalar, step_slice, f)
            }
        }
    }

    fn for_each_possible_leaf_after_steps<'a>(
        &'a self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
        f: &mut impl FnMut(&'a IrMask<bool>),
    ) {
        let Some((step_curr, steps_scalar)) = steps_scalar.split_first() else {
            match step_slice {
                None => {
                    f(self);
                }
                Some(step_slice) => {
                    let IrTargetStepSlice { start, len } = step_slice;
                    let len_1 = match len.sub_1() {
                        Ok(len_1) => len_1,
                        Err(IsZero) => return,
                    };

                    let slf = unwrap_match!(self, IrMask::Array(slf) => slf);
                    let start_range = unwrap_uint_range(module, vars, start);

                    let full_range = ClosedRange {
                        start: &start_range.start,
                        end: &(start_range.end + len_1),
                    };
                    slf.for_each_block_in_range(full_range, |_, m| {
                        f(m);
                        ControlFlow::Continue(())
                    })
                    .remove_never();
                }
            }
            return;
        };

        match step_curr {
            IrTargetStepScalar::ArrayIndex(index) => {
                let slf = unwrap_match!(self, IrMask::Array(slf) => slf);

                let index_range = unwrap_uint_range(module, vars, index);

                slf.for_each_block_in_range(ClosedRange::from(index_range.as_ref()), |_, m| {
                    m.for_each_possible_leaf_after_steps(module, vars, steps_scalar, step_slice, f);
                    ControlFlow::Continue(())
                })
                .remove_never();
            }
            &IrTargetStepScalar::TupleIndex(index) | &IrTargetStepScalar::StructField(index) => {
                let slf = unwrap_match!(self, IrMask::TupleOrStruct(slf) => slf);
                slf[index].for_each_possible_leaf_after_steps(module, vars, steps_scalar, step_slice, f)
            }
        }
    }
}

enum IntKind {
    Single(BigUint),
    Range(ClosedNonEmptyRange<BigUint>),
}

fn unwrap_uint(module: &IrModuleInfo, vars: &IrVariables, value: &IrExpression) -> IntKind {
    if let IrExpression::Int(value) = value {
        IntKind::Single(BigUint::try_from(value).unwrap())
    } else {
        IntKind::Range(unwrap_uint_range(module, vars, value))
    }
}

fn unwrap_uint_range(module: &IrModuleInfo, vars: &IrVariables, value: &IrExpression) -> ClosedNonEmptyRange<BigUint> {
    value
        .ty(&module.large, &module.signals, vars)
        .unwrap_int()
        .map(|v| BigUint::try_from(v).unwrap())
}
