use crate::front::compile::{ArenaVariables, CompileRefs, Variable, VariableInfo};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::domain::ValueDomain;
use crate::front::signal::{Signal, SignalOrVariable};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::{IrAssignmentTarget, IrLargeArena, IrStatement, IrVariable, IrVariableInfo};
use crate::syntax::ast::{MaybeIdentifier, Spanned};
use crate::syntax::pos::Span;
use crate::util::arena::{IndexType, RandomCheck};
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
use indexmap::{IndexMap, IndexSet};
use itertools::{zip_eq, Itertools};
use std::cell::Cell;

// TODO this is really not just the variable values any more, it also tracks value version states
//   and will probably track combinatorial coverage in the future. This is more like a SSA-style "flow" state.
// TODO merge this into scopes and/or into flow?
#[derive(Debug)]
pub struct VariableValues<'p> {
    kind: ParentKind<'p>,
    check: RandomCheck,
    var_values: IndexMap<Variable, MaybeAssignedValue>,
    signal_versions: IndexMap<Signal, ValueVersion>,
}

#[derive(Debug)]
pub struct VariableValuesContent {
    check: RandomCheck,
    next_version_if_root: Option<u64>,
    var_values: IndexMap<Variable, MaybeAssignedValue>,
    signal_versions: IndexMap<Signal, ValueVersion>,
}

#[derive(Debug, Clone)]
pub enum MaybeAssignedValue<A = AssignedValue> {
    Assigned(A),
    NotYetAssigned,
    PartiallyAssigned,
    Error(ErrorGuaranteed),
}

#[derive(Debug, Clone)]
pub struct AssignedValue {
    pub last_assignment_span: Span,
    pub value_and_version: Value<CompileValue, (HardwareValue, ValueVersion)>,
}

impl AssignedValue {
    pub fn value(&self) -> Value {
        match &self.value_and_version {
            Value::Compile(value) => Value::Compile(value.clone()),
            Value::Hardware((value, _)) => Value::Hardware(value.clone()),
        }
    }
}

#[derive(Debug)]
enum ParentKind<'p> {
    Root {
        next_version: Cell<u64>,
    },
    Child {
        parent: &'p VariableValues<'p>,
        // technically redundant, but prevents constant pointer chasing
        next_version: &'p Cell<u64>,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ValueVersion(u64);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ValueVersioned {
    pub value: SignalOrVariable,
    pub version: ValueVersion,
}

impl VariableValues<'static> {
    pub fn new_root(variables: &ArenaVariables) -> Self {
        Self {
            check: variables.check(),
            kind: ParentKind::Root {
                next_version: Cell::new(0),
            },
            var_values: Default::default(),
            signal_versions: Default::default(),
        }
    }
}

impl<'p> VariableValues<'p> {
    pub fn new_child(parent: &'p VariableValues) -> Self {
        Self {
            check: parent.check,
            kind: ParentKind::Child {
                parent,
                next_version: parent.kind.next_version(),
            },
            var_values: Default::default(),
            signal_versions: Default::default(),
        }
    }

    pub fn into_content(self) -> VariableValuesContent {
        let next_version_if_root = match self.kind {
            ParentKind::Root { next_version } => Some(next_version.get()),
            ParentKind::Child { .. } => None,
        };

        VariableValuesContent {
            check: self.check,
            next_version_if_root,
            var_values: self.var_values,
            signal_versions: self.signal_versions,
        }
    }

    pub fn restore_root_from_content(arena: &ArenaVariables, content: VariableValuesContent) -> Self {
        let VariableValuesContent {
            check,
            next_version_if_root,
            var_values,
            signal_versions,
        } = content;

        assert_eq!(arena.check(), check);
        let next_version = next_version_if_root.expect("expected root content");
        let kind = ParentKind::Root {
            next_version: Cell::new(next_version),
        };

        Self {
            kind,
            check,
            var_values,
            signal_versions,
        }
    }

    pub fn restore_child_from_content(
        arena: &ArenaVariables,
        parent: &'p VariableValues,
        content: VariableValuesContent,
    ) -> Self {
        let VariableValuesContent {
            check,
            next_version_if_root,
            var_values,
            signal_versions,
        } = content;

        assert_eq!(arena.check(), check);
        assert_eq!(parent.check(), check);
        assert!(next_version_if_root.is_none());

        let kind = ParentKind::Child {
            parent,
            next_version: parent.kind.next_version(),
        };

        Self {
            kind,
            check,
            var_values,
            signal_versions,
        }
    }

    pub fn check(&self) -> RandomCheck {
        self.check
    }

    pub fn var_new(&mut self, variables: &mut ArenaVariables, info: VariableInfo) -> Variable {
        assert_eq!(self.check, variables.check());

        let var = variables.push(info);
        self.var_values.insert_first(var, MaybeAssignedValue::NotYetAssigned);
        var
    }

    pub fn var_new_immutable_init(
        &mut self,
        variables: &mut ArenaVariables,
        id: MaybeIdentifier,
        assign_span: Span,
        value: Result<Value, ErrorGuaranteed>,
    ) -> Variable {
        let info = VariableInfo {
            id,
            mutable: false,
            ty: None,
        };
        let var = self.var_new(variables, info);

        let assigned = match value {
            Ok(value) => {
                let assigned = AssignedValue {
                    last_assignment_span: assign_span,
                    value_and_version: match value {
                        Value::Compile(value) => Value::Compile(value),
                        Value::Hardware(value) => Value::Hardware((value, self.kind.increment_version())),
                    },
                };
                MaybeAssignedValue::Assigned(assigned)
            }
            Err(e) => MaybeAssignedValue::Error(e),
        };

        self.var_values.insert(var, assigned);
        var
    }

    pub fn var_set(
        &mut self,
        diags: &Diagnostics,
        var: Variable,
        assignment_span: Span,
        value: Value,
    ) -> Result<(), ErrorGuaranteed> {
        assert_eq!(self.check, var.inner().check());
        let known_var = self.iter_up().any(|curr| curr.var_values.contains_key(&var));
        if !known_var {
            return Err(diags.report_internal_error(assignment_span, "set unknown variable"));
        }

        let value_and_version = match value {
            Value::Compile(value) => Value::Compile(value),
            Value::Hardware(value) => Value::Hardware((value, self.kind.increment_version())),
        };

        let assigned = MaybeAssignedValue::Assigned(AssignedValue {
            last_assignment_span: assignment_span,
            value_and_version,
        });
        self.var_values.insert(var, assigned);
        Ok(())
    }

    // TODO create a variant of this that immediately applies the implications
    pub fn var_get(
        &self,
        diags: &Diagnostics,
        span_use: Span,
        var: Variable,
    ) -> Result<&AssignedValue, ErrorGuaranteed> {
        match self.var_get_maybe(diags, span_use, var)? {
            MaybeAssignedValue::Assigned(value) => Ok(value),
            MaybeAssignedValue::NotYetAssigned => Err(diags.report_simple(
                "variable has not yet been assigned a value",
                span_use,
                "variable used here",
            )),
            // TODO point to examples of assignment and non-assignment
            MaybeAssignedValue::PartiallyAssigned => Err(diags.report_simple(
                "variable has not yet been assigned a value in all preceding branches",
                span_use,
                "variable used here",
            )),
            &MaybeAssignedValue::Error(e) => Err(e),
        }
    }

    pub fn var_get_maybe(
        &self,
        diags: &Diagnostics,
        span_use: Span,
        var: Variable,
    ) -> Result<&MaybeAssignedValue, ErrorGuaranteed> {
        self.var_find(var)
            .ok_or_else(|| diags.report_internal_error(span_use, "get unknown variable"))
    }

    fn var_find(&self, var: Variable) -> Option<&MaybeAssignedValue> {
        assert_eq!(self.check, var.inner().check());
        self.iter_up().find_map(|curr| curr.var_values.get(&var))
    }

    pub fn signal_new(&mut self, diags: &Diagnostics, signal: Spanned<Signal>) -> Result<(), ErrorGuaranteed> {
        let known_signal = self
            .iter_up()
            .any(|curr| curr.signal_versions.contains_key(&signal.inner));
        if known_signal {
            return Err(diags.report_internal_error(signal.span, "redefining signal"));
        }

        self.signal_versions
            .insert_first(signal.inner, self.kind.increment_version());
        Ok(())
    }

    pub fn signal_report_write(&mut self, diags: &Diagnostics, signal: Spanned<Signal>) -> Result<(), ErrorGuaranteed> {
        // TODO checking only the root is enough, signals will always be declared there
        let known_signal = self
            .iter_up()
            .any(|curr| curr.signal_versions.contains_key(&signal.inner));
        if !known_signal {
            return Err(diags.report_internal_error(signal.span, "write to unknown signal"));
        }

        self.signal_versions.insert(signal.inner, self.kind.increment_version());
        Ok(())
    }

    // TODO turn signals into variables in the frontend at the first load,
    //   that way they automatically participate in implications, versioning,
    //   const value reasoning, ...
    //   This should also simplify the backends.
    pub fn signal_get_version(&self, signal: Signal) -> Option<ValueVersion> {
        self.iter_up()
            .find_map(|curr| curr.signal_versions.get(&signal).copied())
    }

    pub fn signal_versioned(&self, signal: Signal) -> Option<ValueVersioned> {
        self.signal_get_version(signal).map(|version| ValueVersioned {
            value: SignalOrVariable::Signal(signal),
            version,
        })
    }

    fn iter_up(&self) -> impl Iterator<Item = &VariableValues> + '_ {
        let mut next = Some(self);
        std::iter::from_fn(move || {
            let curr = next;
            next = curr.and_then(|c| c.kind.parent());
            curr
        })
    }
}

impl ParentKind<'_> {
    fn next_version(&self) -> &Cell<u64> {
        match self {
            ParentKind::Root { next_version } => next_version,
            ParentKind::Child {
                parent: _,
                next_version,
            } => next_version,
        }
    }

    fn increment_version(&self) -> ValueVersion {
        let next_version = self.next_version();
        let version = next_version.get();
        next_version.set(version + 1);
        ValueVersion(version)
    }

    fn parent(&self) -> Option<&VariableValues<'_>> {
        match self {
            ParentKind::Root { .. } => None,
            ParentKind::Child {
                parent,
                next_version: _,
            } => Some(parent),
        }
    }
}

struct NeedsHardwareMerge;

// TODO take implications from before the merge into account while merging, requires implications to be in the flow
pub fn merge_variable_branches<C: ExpressionContext>(
    refs: CompileRefs,
    ctx: &mut C,
    large: &mut IrLargeArena,
    variables: &ArenaVariables,
    parent: &mut VariableValues,
    span_merge: Span,
    mut children: Vec<(&mut C::Block, VariableValuesContent)>,
) -> Result<(), ErrorGuaranteed> {
    // TODO do we need to handle the empty children case separately?

    // collect the interesting vars and signals:
    //   * items that are not in any child didn't change
    //   * items that are not in the parent will go out of scope and can be ignored
    let mut merged_vars = IndexSet::new();
    let mut merged_signals = IndexSet::new();
    for (_, child) in &children {
        assert_eq!(parent.check, child.check);
        for &var in child.var_values.keys() {
            if parent.var_find(var).is_some() {
                merged_vars.insert(var);
            }
        }
        for &signal in child.signal_versions.keys() {
            if parent.signal_get_version(signal).is_some() {
                merged_signals.insert(signal);
            }
        }
    }

    // TODO avoid cloning values if possible
    for var in merged_vars {
        let value_parent = parent.var_find(var).unwrap();

        let mut merged: Option<MaybeAssignedValue<Result<AssignedValue, NeedsHardwareMerge>>> = None;
        for (_, child) in &children {
            let child_value = child.var_values.get(&var).unwrap_or(value_parent);

            // first branch, just set it
            let Some(merged_value) = merged else {
                merged = Some(match child_value {
                    MaybeAssignedValue::Assigned(v) => MaybeAssignedValue::Assigned(Ok(v.clone())),
                    MaybeAssignedValue::NotYetAssigned => MaybeAssignedValue::NotYetAssigned,
                    MaybeAssignedValue::PartiallyAssigned => MaybeAssignedValue::PartiallyAssigned,
                    &MaybeAssignedValue::Error(e) => MaybeAssignedValue::Error(e),
                });
                continue;
            };

            // check if we need to do a merge merge
            //   we don't stop once we know we need to merge, because later unassigned values might remove that requirement again
            let merged_new = match (merged_value, child_value) {
                // once we need a hardware merge that stays true
                (MaybeAssignedValue::Assigned(Err(e)), MaybeAssignedValue::Assigned(_)) => {
                    MaybeAssignedValue::Assigned(Err(e))
                }
                // actual merge check
                (MaybeAssignedValue::Assigned(Ok(merged_value)), MaybeAssignedValue::Assigned(child_value)) => {
                    let same_value_and_version = match (&merged_value.value_and_version, &child_value.value_and_version)
                    {
                        (Value::Compile(merged_value), Value::Compile(child_value)) => merged_value == child_value,
                        (Value::Hardware((_, merged_value)), Value::Hardware((_, child_value))) => {
                            merged_value == child_value
                        }
                        _ => false,
                    };

                    if same_value_and_version {
                        let span_combined = if merged_value.last_assignment_span == child_value.last_assignment_span {
                            merged_value.last_assignment_span
                        } else {
                            span_merge
                        };

                        MaybeAssignedValue::Assigned(Ok(AssignedValue {
                            last_assignment_span: span_combined,
                            value_and_version: merged_value.value_and_version,
                        }))
                    } else {
                        MaybeAssignedValue::Assigned(Err(NeedsHardwareMerge))
                    }
                }
                // unassigned cases, these are in some sense nice because we can avoid doing the merge
                (MaybeAssignedValue::NotYetAssigned, MaybeAssignedValue::NotYetAssigned) => {
                    MaybeAssignedValue::NotYetAssigned
                }
                (
                    MaybeAssignedValue::NotYetAssigned
                    | MaybeAssignedValue::PartiallyAssigned
                    | MaybeAssignedValue::Assigned(_),
                    MaybeAssignedValue::NotYetAssigned
                    | MaybeAssignedValue::PartiallyAssigned
                    | MaybeAssignedValue::Assigned(_),
                ) => MaybeAssignedValue::PartiallyAssigned,
                (MaybeAssignedValue::Error(e), _) | (_, &MaybeAssignedValue::Error(e)) => MaybeAssignedValue::Error(e),
            };
            merged = Some(merged_new);
        }

        let result_value = match merged {
            // no branches, nothing to do
            None => continue,
            // hardware merge needed
            Some(MaybeAssignedValue::Assigned(Err(NeedsHardwareMerge))) => {
                // TODO avoid vec allocation
                let children = children
                    .iter_mut()
                    .map(|(block, child)| {
                        let value = child.var_values.get(&var).unwrap_or(value_parent);
                        let value = match value {
                            MaybeAssignedValue::Assigned(v) => v,
                            _ => unreachable!(),
                        };
                        (&mut **block, Spanned::new(span_merge, value.value()))
                    })
                    .collect_vec();
                let merged = merge_hardware_values(refs, ctx, large, span_merge, variables[var].id, children)?;

                let value_and_version = (merged.to_general_expression(), parent.kind.increment_version());
                MaybeAssignedValue::Assigned(AssignedValue {
                    last_assignment_span: span_merge,
                    value_and_version: Value::Hardware(value_and_version),
                })
            }
            // simple passthrough
            Some(MaybeAssignedValue::Assigned(Ok(value))) => MaybeAssignedValue::Assigned(value),
            Some(MaybeAssignedValue::NotYetAssigned) => MaybeAssignedValue::NotYetAssigned,
            Some(MaybeAssignedValue::PartiallyAssigned) => MaybeAssignedValue::PartiallyAssigned,
            Some(MaybeAssignedValue::Error(e)) => MaybeAssignedValue::Error(e),
        };

        parent.var_values.insert(var, result_value);
    }

    for signal in merged_signals {
        let version_parent = match parent.signal_get_version(signal) {
            // context without versioning, give up
            // TODO why do contexts like this exist? don't we want versioning everywhere, eg. even in port connections?
            None => continue,
            Some(version_parent) => version_parent,
        };

        let mut merged: Option<Result<ValueVersion, NeedsHardwareMerge>> = None;
        for (_, child) in &children {
            let version_child = child.signal_versions.get(&signal).copied().unwrap_or(version_parent);

            merged = match merged {
                None => Some(Ok(version_child)),
                Some(Ok(curr_version)) => {
                    if curr_version == version_child {
                        Some(Ok(curr_version))
                    } else {
                        Some(Err(NeedsHardwareMerge))
                    }
                }
                Some(Err(NeedsHardwareMerge)) => Some(Err(NeedsHardwareMerge)),
            }
        }

        let merged_version = match merged {
            None => continue,
            Some(Ok(version)) => version,
            Some(Err(NeedsHardwareMerge)) => parent.kind.increment_version(),
        };
        parent.signal_versions.insert(signal, merged_version);
    }

    Ok(())
}

// TODO at this point we should know that this is an ir context already
fn merge_hardware_values<'a, C: ExpressionContext>(
    refs: CompileRefs,
    ctx: &mut C,
    large: &mut IrLargeArena,
    span_merge: Span,
    debug_info_id: MaybeIdentifier,
    children: Vec<(&mut C::Block, Spanned<Value>)>,
) -> Result<HardwareValue<HardwareType, IrVariable>, ErrorGuaranteed>
where
    C::Block: 'a,
{
    let diags = refs.diags;

    // check that all types are hardware
    // (we do this before finding the common type to get nicer error messages)
    let value_ty_hw = |value: Spanned<&Value>| {
        value.inner.ty().as_hardware_type(refs).map_err(|_| {
            let ty_str = value.inner.ty().diagnostic_string();
            let diag = Diagnostic::new("merging if assignments needs hardware type")
                .add_info(debug_info_id.span(), "for this variable")
                .add_info(
                    value.span,
                    format!(
                        "value assigned here has type `{}` which cannot be represented in hardware",
                        ty_str
                    ),
                )
                .add_error(span_merge, "merging happens here")
                .finish();
            diags.report(diag)
        })
    };
    let tys = children
        .iter()
        .map(|(_, v)| value_ty_hw(v.as_ref()))
        .try_collect_all_vec()?;

    // find common type
    let ty = tys.iter().fold(Type::Undefined, |a, t| a.union(&t.as_type(), false));

    // convert common to hardware too
    let ty = ty.as_hardware_type(refs).map_err(|_| {
        let ty_str = ty.diagnostic_string();

        let mut diag = Diagnostic::new("merging if assignments needs hardware type")
            .add_info(debug_info_id.span(), "for this variable")
            .add_error(
                span_merge,
                format!(
                    "merging happens here, combined type `{}` cannot be represented in hardware",
                    ty_str
                ),
            );
        for ((_, v), ty) in zip_eq(&children, tys) {
            diag = diag.add_info(
                v.span,
                format!("value assigned here has type `{}`", ty.diagnostic_string()),
            )
        }
        diags.report(diag.finish())
    })?;

    // create result variable
    let var_ir_info = IrVariableInfo {
        ty: ty.as_ir(refs),
        debug_info_id: debug_info_id.spanned_string(refs.fixed.source),
    };
    let var_ir = ctx.new_ir_variable(diags, span_merge, var_ir_info)?;

    // store values into that variable
    let mut domain = ValueDomain::CompileTime;
    for (block, value) in children {
        let value = value.inner.as_hardware_value(refs, large, value.span, &ty)?;
        let store = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), value.expr);
        ctx.push_ir_statement(diags, block, Spanned::new(span_merge, store))?;
        domain = domain.join(value.domain);
    }

    Ok(HardwareValue {
        ty,
        domain,
        expr: var_ir,
    })
}
