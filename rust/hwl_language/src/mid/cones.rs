use crate::front::diagnostic::{DiagResult, DiagnosticError, DiagnosticWarning, Diagnostics};
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrExpressionLarge,
    IrForStatement, IrIfStatement, IrModuleChild, IrModuleInfo, IrPortConnection, IrSignal, IrSignalOrVariable,
    IrStatement, IrStringPiece, IrStringSubstitution, IrStructType, IrTargetStepScalar, IrTargetStepSlice,
    IrTargetSteps, IrType, IrVariables,
};
use crate::syntax::ast::PortDirection;
use crate::syntax::pos::Span;
use crate::util::Never;
use crate::util::data::chain_keys;
use crate::util::iter::IterExt;
use indexmap::{IndexMap, IndexSet};
use itertools::{Either, Itertools, rev};
use std::fmt::Debug;
use std::iter::chain;
use std::ops::ControlFlow;
use unwrap_match::unwrap_match;

// TODO actually record info
// TODO check for cycles somewhere, maybe even here
// TODO record instance cones
pub fn compute_module_cones(diags: &Diagnostics, module: &IrModuleInfo) -> DiagResult {
    let mut combined_drivers: IndexMap<IrSignal, IrMask<Vec<Span>>> = IndexMap::new();

    let mut report_signal_drive = |signal: IrSignal, mask: IrMask<bool>, span: Span| {
        let combined_drive = combined_drivers
            .entry(signal)
            .or_insert_with(|| IrMask::new(signal.ty(module), vec![]));
        IrMask::zip2_for_each(combined_drive, &mask, &mut |combined, &curr_drives| {
            if curr_drives {
                combined.push(span);
            }
        })
    };

    let mut any_err = Ok(());
    for child in &module.children {
        match &child.inner {
            IrModuleChild::ClockedProcess(proc) => {
                let cones = compute_clocked_process_cones(module, proc);
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
                                report_signal_drive(signal, IrMask::new(signal.ty(module), true), conn.span);
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
                                report_signal_drive(signal, IrMask::new(signal.ty(module), true), conn.span);
                            }
                        }
                    }
                }
            }
        }
    }
    any_err?;

    let signals_that_need_drivers = chain(
        module
            .ports
            .iter()
            .filter(|(_, info)| match info.direction {
                PortDirection::Input => false,
                PortDirection::Output => true,
            })
            .map(|(p, _)| IrSignal::Port(p)),
        module.wires.keys().map(IrSignal::Wire),
    );

    for signal in signals_that_need_drivers {
        let mut any_driven = false;
        let mut any_undriven = false;
        let mut overlapping_drivers = IndexSet::new();

        if let Some(mask) = combined_drivers.get(&signal) {
            let _ = mask.for_each_leaf::<Never>(|v| {
                match v.len() {
                    0 => {
                        any_undriven = true;
                    }
                    1 => {
                        any_driven = true;
                    }
                    _ => {
                        any_driven = true;
                        overlapping_drivers.extend(v.iter().copied())
                    }
                }
                ControlFlow::Continue(())
            });
        } else {
            any_undriven = true;
        }

        let (signal_kind, signal_name) = match signal {
            IrSignal::Port(port) => ("port", module.ports[port].name.as_str()),
            IrSignal::Wire(wire) => (
                "wire",
                module.wires[wire]
                    .debug_info_id
                    .inner
                    .as_ref()
                    .map_or("_", String::as_str),
            ),
        };

        // report multiple drivers
        if !overlapping_drivers.is_empty() {
            let mut diag = DiagnosticError::new(
                format!("{signal_kind} `{signal_name}` has multiple overlapping drivers"),
                signal.span(module),
                "{signal_kind} defined here",
            );
            for drive_span in overlapping_drivers {
                diag = diag.add_info(drive_span, "driven here");
            }
            any_err = Err(diag.report(diags));
        }

        // report (partially) undriven
        if any_undriven {
            let title_suffix = if !any_driven {
                "has no driver"
            } else {
                "is not fully driven"
            };
            DiagnosticWarning::new(
                format!("{signal_kind} `{signal_name}` {title_suffix}"),
                signal.span(module),
                "signal defined here",
            )
            .report(diags);
        }
    }

    any_err
}

// TODO optimize for large arrays
#[derive(Debug, Clone)]
pub enum IrMask<T> {
    Scalar(T),
    Compound(Vec<IrMask<T>>),
}

#[derive(Debug)]
pub struct Drivers<'p> {
    parent: Option<&'p Drivers<'p>>,
    map: IndexMap<IrSignal, IrMask<bool>>,
}

fn compute_clocked_process_cones(module: &IrModuleInfo, proc: &IrClockedProcess) -> Cones {
    let IrClockedProcess {
        registers,
        variables: _,
        async_reset: _,
        clock_signal: _,
        clock_block: _,
    } = proc;

    // TODO actually compute cones
    //   note: supress read before write errors when visiting blocks, we actually don't care about driving at all
    // registers
    //     .iter()
    //     .map(|&signal| (signal, IrMask::new(signal.ty(module), true)))
    //     .collect();

    todo!()
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

    let mut drivers = Drivers::root();
    let mut read = IndexMap::new();
    compute_combinatorial_block_cones(diags, module, variables, block, &mut drivers, &mut read)?;

    Ok(Cones {
        reads: read,
        drives: drivers.map,
    })
}

struct UndrivenConditionalDrive;

fn compute_combinatorial_block_cones(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    vars: &IrVariables,
    block: &IrBlock,
    drivers: &mut Drivers,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
) -> DiagResult {
    let IrBlock { statements } = block;
    for stmt in statements {
        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                let IrAssignmentTarget { base, steps } = target;
                let IrTargetSteps {
                    steps_scalar,
                    step_slice,
                } = steps;

                // record reads
                for step in steps_scalar {
                    match step {
                        IrTargetStepScalar::ArrayIndex(index) => {
                            record_expression_reads(module, vars, drivers, reads, index)
                        }
                        &IrTargetStepScalar::TupleIndex(_) | &IrTargetStepScalar::StructField(_) => {}
                    }
                }
                if let Some(step_slice) = step_slice {
                    let IrTargetStepSlice { start, len: _ } = step_slice;
                    record_expression_reads(module, vars, drivers, reads, start);
                }
                record_expression_reads(module, vars, drivers, reads, source);

                // record writes
                let base = match base {
                    IrSignalOrVariable::Signal(base) => *base,
                    IrSignalOrVariable::Variable(_) => {
                        // we don't care about variables
                        continue;
                    }
                };

                let curr = drivers.curr_mut(module, base);
                match curr.write_steps(module, vars, steps) {
                    Ok(()) => {}
                    Err(UndrivenConditionalDrive) => {
                        // TODO continue on after whitelisting this signal (in this branch) to report multiple errors at once?
                        // TODO expand this report to include much more information,
                        //   in particular which step is causing the issue and which parts were not yet driven
                        // TODO include whether this was due to dynamic indexing, dynamic slicing or a true conditional
                        // TODO include examples in .[]. format?
                        // TODO add hints: assign default beforehand

                        let e = DiagnosticError::new(
                            "dynamic array assignment to not yet driven signal",
                            stmt.span,
                            "dynamic array assignment here",
                        )
                            .add_footer_hint("This does not create valid combinatorial logic.\nUse a constant target or do a default full assignment beforehand.")
                            .report(diags);
                        return Err(e);
                    }
                }
            }
            IrStatement::Block(stmt_inner) => {
                compute_combinatorial_block_cones(diags, module, vars, stmt_inner, drivers, reads)?;
            }
            IrStatement::If(stmt_inner) => {
                let IrIfStatement {
                    condition,
                    then_block,
                    else_block,
                } = stmt_inner;

                // record reads
                record_expression_reads(module, vars, drivers, reads, condition);

                // visit then
                let mut then_drivers = drivers.new_child();
                compute_combinatorial_block_cones(diags, module, vars, then_block, &mut then_drivers, reads)?;
                let then_map = then_drivers.map;

                // visit else
                let mut else_drivers = drivers.new_child();
                if let Some(else_block) = else_block {
                    compute_combinatorial_block_cones(diags, module, vars, else_block, &mut else_drivers, reads)?;
                }
                let else_map = else_drivers.map;

                // merge
                for &signal in chain_keys(&then_map, &else_map) {
                    let mut any_err = false;
                    IrMask::zip3_maybe_for_each(
                        drivers.curr_mut(module, signal),
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
                            any_err |= then_value != else_value;

                            // store union, assume driven to avoid future false positives
                            *parent |= then_value | else_value;
                        },
                    );
                    if any_err {
                        // TODO include info where signal was written and which parts were written
                        let e = DiagnosticError::new(
                            "driver mismatch between conditional branches",
                            stmt.span,
                            "conditional branch here",
                        )
                            .add_info(signal.span(module), "for this signal")
                            .add_footer_hint("This does not create valid combinatorial logic.\nEnsure both branches drive the same signal (subset) or do a default full assignment beforehand.")
                            .report(diags);
                        return Err(e);
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
                    compute_combinatorial_block_cones(diags, module, vars, block, drivers, reads)?;
                }
            }
            IrStatement::Print(pieces) => {
                // record reads
                for piece in pieces {
                    match piece {
                        IrStringPiece::Literal(_) => {}
                        IrStringPiece::Substitute(sub) => match sub {
                            IrStringSubstitution::Integer(expr, _radix) => {
                                record_expression_reads(module, vars, drivers, reads, expr);
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

    Ok(())
}

fn record_expression_reads(
    module: &IrModuleInfo,
    vars: &IrVariables,
    drivers: &Drivers,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
    expr: &IrExpression,
) {
    let steps = ReadSteps {
        steps_scalar_rev: vec![],
        step_slice: None,
    };
    record_expression_reads_impl(module, vars, drivers, reads, expr, steps);
}

struct ReadSteps {
    steps_scalar_rev: Vec<IrTargetStepScalar>,
    step_slice: Option<IrTargetStepSlice>,
}

fn record_expression_reads_impl(
    module: &IrModuleInfo,
    vars: &IrVariables,
    drivers: &Drivers,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
    expr: &IrExpression,
    mut steps: ReadSteps,
) {
    match expr {
        // constants, no reads
        IrExpression::Bool(_) | IrExpression::Int(_) => {}
        // we only care about signal reads
        IrExpression::Variable(_) => {}

        &IrExpression::Large(expr_large) => match &module.large[expr_large] {
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

            // participate in step collection
            IrExpressionLarge::ArrayIndex { base, index } => {
                let step = IrTargetStepScalar::ArrayIndex(index.clone());
                record_expression_reads_base(module, vars, drivers, reads, base, steps, Either::Left(step));
                record_expression_reads(module, vars, drivers, reads, index);
            }
            IrExpressionLarge::ArraySlice { base, start, len } => {
                let step = IrTargetStepSlice {
                    start: start.clone(),
                    len: len.clone(),
                };
                record_expression_reads_base(module, vars, drivers, reads, base, steps, Either::Right(step));
                record_expression_reads(module, vars, drivers, reads, start);
            }
            IrExpressionLarge::TupleIndex { base, index } => {
                let step = IrTargetStepScalar::TupleIndex(*index);
                record_expression_reads_base(module, vars, drivers, reads, base, steps, Either::Left(step));
            }
            IrExpressionLarge::StructField { base, field } => {
                let step = IrTargetStepScalar::StructField(*field);
                record_expression_reads_base(module, vars, drivers, reads, base, steps, Either::Left(step));
            }
        },

        // finish step collection and record actual reads
        &IrExpression::Signal(signal) => {
            let ReadSteps {
                steps_scalar_rev,
                step_slice,
            } = steps;
            let mut steps_scalar = steps_scalar_rev;
            steps_scalar.reverse();

            let signal_reads = reads
                .entry(signal)
                .or_insert_with(|| IrMask::new(signal.ty(module), false));
            let signal_driven = drivers.curr(signal);

            // TODO this is cursed, build proper zip iterators, including simple "constant" iterators that can be mixed?
            let mut pieces_driven = signal_driven.map(|signal_driven| {
                let mut pieces_driven = vec![];
                signal_driven.for_each_possible_leaf_after_steps(
                    module,
                    vars,
                    &steps_scalar,
                    step_slice.as_ref(),
                    &mut |m| pieces_driven.push(m),
                );
                pieces_driven.into_iter()
            });

            signal_reads.for_each_possible_leaf_after_steps_mut(
                module,
                vars,
                &steps_scalar,
                step_slice.as_ref(),
                &mut |piece_read| {
                    if let Some(pieces_driven) = &mut pieces_driven {
                        // only count reads from pieces that have not yet been driven
                        let piece_driven = pieces_driven.next().unwrap();
                        IrMask::zip2_for_each(piece_read, piece_driven, &mut |scalar_read, scalar_driven| {
                            *scalar_read |= !scalar_driven;
                        })
                    } else {
                        // not driven at all, count as read
                        piece_read.fill(true);
                    }
                },
            );

            assert!(pieces_driven.is_none_or(|it| it.is_empty()));
        }
    }
}

fn record_expression_reads_base(
    module: &IrModuleInfo,
    vars: &IrVariables,
    drivers: &Drivers,
    reads: &mut IndexMap<IrSignal, IrMask<bool>>,
    base: &IrExpression,
    mut steps: ReadSteps,
    step: Either<IrTargetStepScalar, IrTargetStepSlice>,
) {
    if steps.step_slice.is_some() {
        // TODO support fusion, like we did for writes by enforcing it in the IR?
        // fallback to fully reading the base
        record_expression_reads(module, vars, drivers, reads, base)
    } else {
        // continue collecting steps
        match step {
            Either::Left(step) => {
                steps.steps_scalar_rev.push(step);
            }
            Either::Right(step) => {
                steps.step_slice = Some(step);
            }
        }
        record_expression_reads_impl(module, vars, drivers, reads, base, steps);
    }
}

impl Drivers<'_> {
    fn root() -> Drivers<'static> {
        Drivers {
            parent: None,
            map: IndexMap::new(),
        }
    }

    fn new_child(&self) -> Drivers<'_> {
        Drivers {
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
            curr = match curr.parent {
                None => return None,
                Some(parent) => parent,
            };
        }
    }

    fn curr_mut(&mut self, module: &IrModuleInfo, signal: IrSignal) -> &mut IrMask<bool> {
        self.map.entry(signal).or_insert_with(|| {
            let mut curr = self.parent;
            loop {
                curr = match curr {
                    None => {
                        break IrMask::new(signal.ty(module), false);
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
                let len = usize::try_from(len).unwrap_or_else(|_| todo!());

                let mask_inner = IrMask::new(ty_inner, init);
                IrMask::Compound(vec![mask_inner; len])
            }
            IrType::Tuple(ty_fields) => IrMask::Compound(
                ty_fields
                    .iter()
                    .map(|ty_field| IrMask::new(ty_field, init.clone()))
                    .collect_vec(),
            ),
            IrType::Struct(ty_info) => {
                let IrStructType {
                    ty: _,
                    debug_info_name: _,
                    fields,
                } = ty_info;
                IrMask::Compound(
                    fields
                        .values()
                        .map(|ty_field| IrMask::new(ty_field, init.clone()))
                        .collect_vec(),
                )
            }
        }
    }

    fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        match self {
            IrMask::Scalar(slf) => *slf = value,
            IrMask::Compound(values) => values.iter_mut().for_each(|v| v.fill(value.clone())),
        }
    }

    fn for_each_leaf<B>(&self, mut f: impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
        fn for_each_leaf_impl<T, B>(slf: &IrMask<T>, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
            match slf {
                IrMask::Scalar(slf) => f(slf),
                IrMask::Compound(values) => values.iter().try_for_each(|m| for_each_leaf_impl(m, f)),
            }
        }
        for_each_leaf_impl(self, &mut f)
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

    fn zip2_for_each<U: Debug>(a: &mut IrMask<T>, b: &IrMask<U>, f: &mut impl FnMut(&mut T, &U)) {
        match a {
            IrMask::Scalar(a) => {
                let b = unwrap_match!(b, IrMask::Scalar(b) => b);
                f(a, b)
            }
            IrMask::Compound(a) => {
                let b = unwrap_match!(b, IrMask::Compound(b) => b);
                for i in 0..a.len() {
                    Self::zip2_for_each(&mut a[i], &b[i], f);
                }
            }
        }
    }

    fn zip3_maybe_for_each<U: Debug, V: Debug>(
        a: &mut IrMask<T>,
        b: Option<&IrMask<U>>,
        c: Option<&IrMask<V>>,
        f: &mut impl FnMut(&mut T, Option<&U>, Option<&V>),
    ) {
        match a {
            IrMask::Scalar(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::Scalar(b) => b));
                let c = c.map(|c| unwrap_match!(c, IrMask::Scalar(c) => c));
                f(a, b, c)
            }
            IrMask::Compound(a) => {
                let b = b.map(|b| unwrap_match!(b, IrMask::Compound(b) => b));
                let c = c.map(|c| unwrap_match!(c, IrMask::Compound(c) => c));

                for i in 0..a.len() {
                    let b = b.map(|b| &b[i]);
                    let c = c.map(|c| &c[i]);
                    Self::zip3_maybe_for_each(&mut a[i], b, c, f);
                }
            }
        }
    }
}

impl IrMask<bool> {
    fn write_steps(
        &mut self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps: &IrTargetSteps,
    ) -> Result<(), UndrivenConditionalDrive> {
        let IrTargetSteps {
            steps_scalar,
            step_slice,
        } = steps;
        self.write_steps_impl(module, vars, steps_scalar, step_slice.as_ref())
    }

    fn write_steps_impl(
        &mut self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
    ) -> Result<(), UndrivenConditionalDrive> {
        let Some((step_curr, steps_scalar)) = steps_scalar.split_first() else {
            match step_slice {
                None => {
                    self.fill(true);
                }
                Some(step_slice) => {
                    let IrTargetStepSlice { start, len } = step_slice;

                    let slf = unwrap_match!(self, IrMask::Compound(slf) => slf);
                    let len = usize::try_from(len).unwrap_or_else(|_| todo!());

                    match unwrap_int(module, vars, start) {
                        IntKind::Single(start) => {
                            let slice = &mut slf[start..][..len];
                            slice.iter_mut().for_each(|v| v.fill(true));
                        }
                        IntKind::Range(start_range) => {
                            let index_range = start_range.start..start_range.end + len;

                            // check no partially written elements in range
                            let fully_driven = slf[index_range].iter().all(|m| m.all(|&b| b));
                            if !fully_driven {
                                return Err(UndrivenConditionalDrive);
                            }

                            // we don't need to set anything,
                            //   we've just checked that everything that could be written was already written
                        }
                    }
                }
            }

            return Ok(());
        };

        match step_curr {
            IrTargetStepScalar::ArrayIndex(index) => {
                let slf = unwrap_match!(self, IrMask::Compound(slf) => slf);

                match unwrap_int(module, vars, index) {
                    IntKind::Single(index) => {
                        slf[index].write_steps_impl(module, vars, steps_scalar, step_slice)?;
                    }
                    IntKind::Range(index_range) => {
                        // check no partially written elements in range
                        let fully_driven = slf[index_range]
                            .iter()
                            .all(|m| m.all_after_steps(module, vars, steps_scalar, step_slice));
                        if !fully_driven {
                            return Err(UndrivenConditionalDrive);
                        }

                        // we don't need to set anything,
                        //   we've just checked that everything that could be written was already written
                    }
                }
            }
            IrTargetStepScalar::TupleIndex(_) => todo!(),
            IrTargetStepScalar::StructField(_) => todo!(),
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

                    let slf = unwrap_match!(self, IrMask::Compound(slf) => slf);

                    let start_range = unwrap_int_unified(module, vars, start);
                    let len = usize::try_from(len).unwrap_or_else(|_| todo!());

                    let full_range = start_range.start..start_range.end + len;
                    slf[full_range].iter_mut().for_each(f);
                }
            }
            return;
        };

        match step_curr {
            IrTargetStepScalar::ArrayIndex(index) => {
                let slf = unwrap_match!(self, IrMask::Compound(slf) => slf);

                let index_range = unwrap_int_unified(module, vars, index);
                slf[index_range]
                    .iter_mut()
                    .for_each(|x| x.for_each_possible_leaf_after_steps_mut(module, vars, steps_scalar, step_slice, f))
            }
            IrTargetStepScalar::TupleIndex(_) => todo!(),
            IrTargetStepScalar::StructField(_) => todo!(),
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

                    let slf = unwrap_match!(self, IrMask::Compound(slf) => slf);

                    let start_range = unwrap_int_unified(module, vars, start);
                    let len = usize::try_from(len).unwrap_or_else(|_| todo!());

                    let full_range = start_range.start..start_range.end + len;
                    slf[full_range].iter().for_each(f);
                }
            }
            return;
        };

        match step_curr {
            IrTargetStepScalar::ArrayIndex(index) => {
                let slf = unwrap_match!(self, IrMask::Compound(slf) => slf);

                let index_range = unwrap_int_unified(module, vars, index);
                slf[index_range]
                    .iter()
                    .for_each(|x| x.for_each_possible_leaf_after_steps(module, vars, steps_scalar, step_slice, f))
            }
            IrTargetStepScalar::TupleIndex(_) => todo!(),
            IrTargetStepScalar::StructField(_) => todo!(),
        }
    }
}

enum IntKind {
    Single(usize),
    Range(std::ops::Range<usize>),
}

fn unwrap_int(module: &IrModuleInfo, vars: &IrVariables, value: &IrExpression) -> IntKind {
    if let IrExpression::Int(value) = value {
        IntKind::Single(usize::try_from(value).unwrap_or_else(|_| todo!()))
    } else {
        let range = value.ty(module, vars).unwrap_int();
        let start = usize::try_from(&range.start).unwrap_or_else(|_| todo!());
        let end = usize::try_from(&range.end).unwrap_or_else(|_| todo!());
        IntKind::Range(start..end)
    }
}

fn unwrap_int_unified(module: &IrModuleInfo, vars: &IrVariables, value: &IrExpression) -> std::ops::Range<usize> {
    let range = value.ty(module, vars).unwrap_int();
    let start = usize::try_from(&range.start).unwrap_or_else(|_| todo!());
    let end = usize::try_from(&range.end).unwrap_or_else(|_| todo!());
    start..end
}
