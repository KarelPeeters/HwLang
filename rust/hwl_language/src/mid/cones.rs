use crate::front::diagnostic::{DiagResult, DiagnosticError, DiagnosticWarning, Diagnostics};
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrForStatement, IrIfStatement,
    IrModuleChild, IrModuleInfo, IrSignal, IrSignalOrVariable, IrStatement, IrStructType, IrTargetStepScalar,
    IrTargetStepSlice, IrTargetSteps, IrType, IrVariables,
};
use crate::util::Never;
use crate::util::data::chain_keys;
use indexmap::{IndexMap, IndexSet};
use itertools::{Itertools, chain, enumerate};
use std::fmt::Debug;
use std::ops::ControlFlow;
use unwrap_match::unwrap_match;

pub fn compute_module_drivers(diags: &Diagnostics, module: &IrModuleInfo) -> DiagResult {
    let mut combined_drivers: IndexMap<IrSignal, IrMask<Vec<usize>>> = IndexMap::new();

    for (child_index, child) in enumerate(&module.children) {
        let child_masks = match &child.inner {
            IrModuleChild::ClockedProcess(proc) => compute_clocked_process_drivers(module, proc)?,
            IrModuleChild::CombinatorialProcess(proc) => compute_combinatorial_process_drivers(diags, module, proc)?,
            IrModuleChild::ModuleInternalInstance(_) => {
                todo!()
            }
            IrModuleChild::ModuleExternalInstance(_) => {
                todo!()
            }
        };

        for (signal, child_mask) in child_masks {
            let combined_drive = combined_drivers
                .entry(signal)
                .or_insert_with(|| IrMask::new(signal.ty(module), vec![]));
            IrMask::zip2_for_each(combined_drive, &child_mask, &mut |combined, &curr_drives| {
                if curr_drives {
                    combined.push(child_index);
                }
            })
        }
    }

    // TODO improve error messages
    let mut any_err = Ok(());

    let all_signals = chain(
        module.ports.keys().map(IrSignal::Port),
        module.wires.keys().map(IrSignal::Wire),
    );

    for signal in all_signals {
        let mut any_driven = false;
        let mut any_undriven = false;
        let mut overlapping_children = IndexSet::new();

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
                        overlapping_children.extend(v.iter().copied())
                    }
                }
                ControlFlow::Continue(())
            });
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
        if !overlapping_children.is_empty() {
            let mut diag = DiagnosticError::new(
                format!("{signal_kind} `{signal_name}` has multiple overlapping drivers"),
                signal.span(module),
                "{signal_kind} defined here",
            );
            for child in overlapping_children {
                diag = diag.add_info(module.children[child].span, "driven here");
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

fn compute_clocked_process_drivers(
    module: &IrModuleInfo,
    proc: &IrClockedProcess,
) -> DiagResult<IndexMap<IrSignal, IrMask<bool>>> {
    let IrClockedProcess {
        registers,
        variables: _,
        async_reset: _,
        clock_signal: _,
        clock_block: _,
    } = proc;

    let drivers = registers
        .iter()
        .map(|&signal| (signal, IrMask::new(signal.ty(module), true)))
        .collect();
    Ok(drivers)
}

fn compute_combinatorial_process_drivers(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    proc: &IrCombinatorialProcess,
) -> DiagResult<IndexMap<IrSignal, IrMask<bool>>> {
    let IrCombinatorialProcess { variables, block } = proc;

    let mut drivers = Drivers::root();
    compute_combinatorial_block_drivers(diags, module, variables, block, &mut drivers)?;
    Ok(drivers.map)
}

struct UndrivenConditionalDrive;

fn compute_combinatorial_block_drivers(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    vars: &IrVariables,
    block: &IrBlock,
    drivers: &mut Drivers,
) -> DiagResult {
    let IrBlock { statements } = block;
    for stmt in statements {
        match &stmt.inner {
            IrStatement::Assign(target, source) => {
                let _ = source;

                let IrAssignmentTarget { base, steps } = target;
                let base = match base {
                    IrSignalOrVariable::Signal(base) => *base,
                    IrSignalOrVariable::Variable(_) => {
                        // we don't care about variables
                        continue;
                    }
                };

                let curr = drivers.curr(module, base);
                match curr.set_steps(module, vars, steps) {
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
                compute_combinatorial_block_drivers(diags, module, vars, stmt_inner, drivers)?;
            }
            IrStatement::If(stmt_inner) => {
                let IrIfStatement {
                    condition: _,
                    then_block,
                    else_block,
                } = stmt_inner;

                // visit then
                let mut then_drivers = drivers.new_child();
                compute_combinatorial_block_drivers(diags, module, vars, then_block, &mut then_drivers)?;
                let then_map = then_drivers.map;

                // visit else
                let mut else_drivers = drivers.new_child();
                if let Some(else_block) = else_block {
                    compute_combinatorial_block_drivers(diags, module, vars, else_block, &mut else_drivers)?;
                }
                let else_map = else_drivers.map;

                // merge
                for &signal in chain_keys(&then_map, &else_map) {
                    let mut any_err = false;
                    IrMask::zip3_maybe_for_each(
                        drivers.curr(module, signal),
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
                    // block will never execute, so it also can't drive anything
                } else {
                    // this is pessimistic, but safe:
                    //   we do not use the fact that the index variable is known in each iteration
                    compute_combinatorial_block_drivers(diags, module, vars, block, drivers)?;
                }
            }
            IrStatement::Print(_) | IrStatement::AssertFailed => {
                // cannot drive anything
            }
        }
    }

    Ok(())
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

    fn curr(&mut self, module: &IrModuleInfo, signal: IrSignal) -> &mut IrMask<bool> {
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

// TODO rework this
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
    fn set_steps(
        &mut self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps: &IrTargetSteps,
    ) -> Result<(), UndrivenConditionalDrive> {
        let IrTargetSteps {
            steps_scalar,
            step_slice,
        } = steps;
        self.set_steps_impl(module, vars, steps_scalar, step_slice.as_ref())
    }

    // TODO proper error handling
    // TODO should we check for single-element ranges or only accept constants?
    //   (careful: backends should emit simple enough code that downstream tools agree)
    fn set_steps_impl(
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
                        slf[index].set_steps_impl(module, vars, steps_scalar, step_slice)?;
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

    fn for_each_possible_leaf_after_steps(
        &self,
        module: &IrModuleInfo,
        vars: &IrVariables,
        steps_scalar: &[IrTargetStepScalar],
        step_slice: Option<&IrTargetStepSlice>,
        f: &mut impl FnMut(&IrMask<bool>),
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
