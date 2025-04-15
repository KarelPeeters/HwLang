use crate::front::compile::{ArenaVariables, Variable, VariableInfo};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::signal::{Signal, SignalOrVariable};
use crate::front::types::{HardwareType, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::{IrAssignmentTarget, IrLargeArena, IrStatement, IrVariable, IrVariableInfo};
use crate::syntax::ast::{MaybeIdentifier, Spanned};
use crate::syntax::pos::Span;
use crate::util::arena::{IndexType, RandomCheck};
use crate::util::data::IndexMapExt;
use crate::util::result_pair;
use indexmap::IndexMap;
use itertools::chain;
use std::cell::Cell;
use std::hash::Hash;

// TODO this is really not just the variable values any more, it also tracks value version states
//   and will probably track combinatorial coverage in the future. This is more like a SSA-style "flow" state.
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
pub enum MaybeAssignedValue {
    Assigned(AssignedValue),
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

    pub fn restore_root_from_content(variables: &ArenaVariables, content: VariableValuesContent) -> Self {
        let VariableValuesContent {
            check,
            next_version_if_root,
            var_values,
            signal_versions,
        } = content;

        assert_eq!(variables.check(), check);
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
        value: Value,
    ) -> Variable {
        let info = VariableInfo {
            id,
            mutable: false,
            ty: None,
        };
        let var = self.var_new(variables, info);
        let assigned = AssignedValue {
            last_assignment_span: assign_span,
            value_and_version: match value {
                Value::Compile(value) => Value::Compile(value),
                Value::Hardware(value) => Value::Hardware((value, self.kind.increment_version())),
            },
        };
        self.var_values.insert(var, MaybeAssignedValue::Assigned(assigned));
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

// TODO expand to n-way merge (for match statements)
// TODO take implications from before the merge into account while merging, requires implications to be in the flow
pub fn merge_variable_branches<C: ExpressionContext>(
    diags: &Diagnostics,
    ctx: &mut C,
    large: &mut IrLargeArena,
    variables: &ArenaVariables,
    parent: &mut VariableValues,
    span_merge: Span,
    ctx_block_0: &mut C::Block,
    child_0: VariableValuesContent,
    ctx_block_1: &mut C::Block,
    child_1: VariableValuesContent,
) -> Result<(), ErrorGuaranteed> {
    let VariableValuesContent {
        check: check_0,
        next_version_if_root: _,
        var_values: var_values_0,
        signal_versions: signal_versions_0,
    } = child_0;
    let VariableValuesContent {
        check: check_1,
        next_version_if_root: _,
        var_values: var_values_1,
        signal_versions: signal_versions_1,
    } = child_1;
    assert_eq!(parent.check, check_0);
    assert_eq!(parent.check, check_1);

    // TODO avoid cloning values, but that probably requires us to clone map keys
    for &var in combined_keys(&var_values_0, &var_values_1) {
        let value_parent = match parent.var_find(var) {
            // this variable did not yet exist in the parent scope, so it would not be observable after this merge
            None => continue,
            Some(v) => v,
        };

        let value_0 = var_values_0.get(&var).map(Ok).unwrap_or_else(|| Ok(value_parent))?;
        let value_1 = var_values_1.get(&var).map(Ok).unwrap_or_else(|| Ok(value_parent))?;

        let value_combined = match (value_0, value_1) {
            (MaybeAssignedValue::Assigned(value_0), MaybeAssignedValue::Assigned(value_1)) => {
                let same_value_and_version = match (&value_0.value_and_version, &value_1.value_and_version) {
                    (Value::Compile(value_0), Value::Compile(value_1)) => value_0 == value_1,
                    (Value::Hardware((_, version_0)), Value::Hardware((_, version_1))) => version_0 == version_1,
                    _ => false,
                };

                if same_value_and_version {
                    let span_combined = if value_0.last_assignment_span == value_1.last_assignment_span {
                        value_0.last_assignment_span
                    } else {
                        span_merge
                    };

                    MaybeAssignedValue::Assigned(AssignedValue {
                        last_assignment_span: span_combined,
                        value_and_version: value_0.value_and_version.clone(),
                    })
                } else {
                    let value_0 = Spanned::new(value_0.last_assignment_span, value_0.value());
                    let value_1 = Spanned::new(value_1.last_assignment_span, value_1.value());

                    // TODO if the merge fails, we don't need to immediately report an error,
                    //   we could also delay that until someone actually tries to use it.
                    //   That's adds complexity and is probably not very useful though.
                    let value_combined = merge_values(
                        diags,
                        ctx,
                        large,
                        span_merge,
                        &variables[var].id,
                        value_0.as_ref(),
                        value_1.as_ref(),
                    );
                    match value_combined {
                        Ok(MergedValue {
                            value,
                            store_0,
                            store_1,
                        }) => {
                            ctx.push_ir_statement(diags, ctx_block_0, store_0)?;
                            ctx.push_ir_statement(diags, ctx_block_1, store_1)?;

                            let version = parent.kind.increment_version();
                            MaybeAssignedValue::Assigned(AssignedValue {
                                last_assignment_span: span_merge,
                                value_and_version: Value::Hardware((value.to_general_expression(), version)),
                            })
                        }
                        Err(e) => MaybeAssignedValue::Error(e),
                    }
                }
            }
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
            (&MaybeAssignedValue::Error(e), _) | (_, &MaybeAssignedValue::Error(e)) => MaybeAssignedValue::Error(e),
        };

        parent.var_values.insert(var, value_combined);
    }

    for &signal in combined_keys(&signal_versions_0, &signal_versions_1) {
        let version_parent = match parent.signal_get_version(signal) {
            // context without versioning, give up
            // TODO why do contexts like this exist? don't we want versioning everywhere, eg. even in port connections?
            None => continue,
            Some(version_parent) => version_parent,
        };
        let version_0 = signal_versions_0.get(&signal).copied().unwrap_or(version_parent);
        let version_1 = signal_versions_1.get(&signal).copied().unwrap_or(version_parent);

        let version_combined = if version_0 == version_1 {
            version_0
        } else {
            parent.kind.increment_version()
        };
        parent.signal_versions.insert(signal, version_combined);
    }

    Ok(())
}

struct MergedValue {
    value: HardwareValue<HardwareType, IrVariable>,
    store_0: Spanned<IrStatement>,
    store_1: Spanned<IrStatement>,
}

fn merge_values<C: ExpressionContext>(
    diags: &Diagnostics,
    ctx: &mut C,
    large: &mut IrLargeArena,
    span_merge: Span,
    debug_info_id: &MaybeIdentifier,
    value_0: Spanned<&Value>,
    value_1: Spanned<&Value>,
) -> Result<MergedValue, ErrorGuaranteed> {
    // check that both types are hardware
    // (we do this before finding the common type to get nicer error messages)
    let value_ty_hw = |value: Spanned<&Value>| {
        value.inner.ty().as_hardware_type().ok_or_else(|| {
            let ty_str = value.inner.ty().to_diagnostic_string();
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
    let ty_0 = value_ty_hw(value_0);
    let ty_1 = value_ty_hw(value_1);
    let (ty_0, ty_1) = result_pair(ty_0, ty_1)?;

    // figure out the common type
    let ty = ty_0.as_type().union(&ty_1.as_type(), false);
    let ty = ty.as_hardware_type().ok_or_else(|| {
        let ty_str = ty.to_diagnostic_string();
        let ty_0_str = ty_0.to_diagnostic_string();
        let ty_1_str = ty_1.to_diagnostic_string();

        let diag = Diagnostic::new("merging if assignments needs hardware type")
            .add_info(debug_info_id.span(), "for this variable")
            .add_info(value_0.span, format!("value assigned here has type `{}`", ty_0_str))
            .add_info(value_1.span, format!("value assigned here has type `{}`", ty_1_str))
            .add_error(
                span_merge,
                format!(
                    "merging happens here, combined type `{}` cannot be represented in hardware",
                    ty_str
                ),
            )
            .finish();
        diags.report(diag)
    })?;

    // convert values to common type
    let value_0 = value_0.inner.as_hardware_value(diags, large, value_0.span, &ty);
    let value_1 = value_1.inner.as_hardware_value(diags, large, value_1.span, &ty);
    let (value_0, value_1) = result_pair(value_0, value_1)?;

    // create result variable
    let var_ir_info = IrVariableInfo {
        ty: ty.as_ir(),
        debug_info_id: debug_info_id.clone(),
    };
    let var_ir = ctx.new_ir_variable(diags, span_merge, var_ir_info)?;

    // store values into that variable
    let store_0 = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), value_0.expr);
    let store_1 = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), value_1.expr);

    Ok(MergedValue {
        store_0: Spanned::new(span_merge, store_0),
        store_1: Spanned::new(span_merge, store_1),
        value: HardwareValue {
            ty,
            domain: value_0.domain.join(value_1.domain),
            expr: var_ir,
        },
    })
}

fn combined_keys<'a, K: Eq + Hash, V>(
    map_0: &'a IndexMap<K, V>,
    map_1: &'a IndexMap<K, V>,
) -> impl Iterator<Item = &'a K> + 'a {
    chain(map_0.keys(), map_1.keys().filter(|&k| !map_0.contains_key(k)))
}
