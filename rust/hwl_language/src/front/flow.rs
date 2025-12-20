use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagError, DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::{DomainSignal, ValueDomain};
use crate::front::implication::{HardwareValueWithVersion, Implication, ImplicationKind, ValueWithVersion};
use crate::front::module::ExtraRegisterInit;
use crate::front::signal::{Port, Register, Signal, SignalOrVariable, Wire};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, HardwareValue, MaybeUndefined, MixedCompoundValue, NotCompile,
    SimpleCompileValue, Value, ValueCommon,
};
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrExpression, IrExpressionLarge, IrLargeArena, IrRegisters, IrSignal,
    IrSignalOrVariable, IrStatement, IrTargetStep, IrVariable, IrVariableInfo, IrVariables, IrWires,
};
use crate::syntax::ast::{MaybeIdentifier, SyncDomain};
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::source::SourceDatabase;
use crate::try_inner;
use crate::util::arena::RandomCheck;
use crate::util::big_int::BigInt;
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
use crate::util::range::RangeEmpty;
use crate::util::range_multi::{ClosedMultiRange, ClosedNonEmptyMultiRange};
use crate::util::{NON_ZERO_USIZE_ONE, NON_ZERO_USIZE_TWO};
use indexmap::{IndexMap, IndexSet};
use itertools::{Either, Itertools, zip_eq};
use std::cell::Cell;
use std::num::NonZeroUsize;
use unwrap_match::unwrap_match;

// TODO find all points where scopes are nested, and also create flow children to save memory
// TODO store constants into scopes instead of in flow, so we can stop using flow top-level entirely

// TODO check domain for reads and writes to all signals
//   eg. currently we can still read async signals in clocked blocks

// TODO signals should behave more like variables, where the compiler remembers which things it has assigned,
//   eg. if smaller range then it knows that for future reads, if constant they behave like constants

// TODO cleanup
// TODO order struct, impls, and impl functions
// TODO instead of looping everywhere, can we just recurse?
//   try both and benchmark, create some deeply stacked flow chains

trait FlowPrivate: Sized {
    fn root(&self) -> &FlowRoot<'_>;

    fn implied_info(&self, value: ValueVersion) -> Option<&ImpliedInfo>;

    fn var_set_maybe(&mut self, var: Variable, assignment_span: Span, value: VariableValue);

    fn var_get_maybe(&self, var: Spanned<Variable>) -> DiagResult<VariableValueRef<'_>>;

    fn try_var_info(&self, var: VariableIndex) -> Option<&VariableInfo>;
}

#[allow(private_bounds)]
pub trait Flow: FlowPrivate {
    #[allow(clippy::needless_lifetimes)]
    fn new_child_compile<'s>(&'s mut self, span: Span, reason: &'static str) -> FlowCompile<'s>;

    // TODO find a better name
    fn check_hardware(&mut self, span: Span, reason: &str) -> DiagResult<&mut FlowHardware<'_>>;

    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>>;

    fn var_new(&mut self, info: VariableInfo) -> Variable;

    // TODO maybe we should only allow setting IR variables, so they're more clearly owned by flow and we're free to reuse them during merging
    fn var_set(&mut self, var: Variable, assignment_span: Span, value: DiagResult<Value>) {
        let assigned = match value {
            Ok(value) => {
                let value = value.map_hardware(|value| {
                    let version = self.root().next_version();
                    HardwareValueWithVersion { value, version }
                });
                MaybeAssignedValue::Assigned(MaybeUndefined::Defined(AssignedValue {
                    last_assignment_span: assignment_span,
                    value,
                }))
            }
            Err(e) => MaybeAssignedValue::Error(e),
        };
        self.var_set_maybe(var, assignment_span, assigned);
    }

    fn var_set_undefined(&mut self, var: Variable, assignment_span: Span) {
        let assigned = MaybeAssignedValue::Assigned(MaybeUndefined::Undefined);
        self.var_set_maybe(var, assignment_span, assigned);
    }

    fn var_info(&self, var: Spanned<Variable>) -> DiagResult<&VariableInfo> {
        assert_eq!(var.inner.check, self.root().check);
        self.try_var_info(var.inner.index).ok_or_else(|| {
            self.root()
                .diags
                .report_internal_error(var.span, "failed to find variable")
        })
    }

    fn var_eval(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        var: Spanned<Variable>,
    ) -> DiagResult<ValueWithVersion> {
        match self.var_get_maybe(var)? {
            MaybeAssignedValue::Assigned(value) => match value {
                MaybeUndefined::Undefined => {
                    // should not be possible,
                    //   undefined here means the caller is responsible for ensuring this is never observable
                    Err(diags.report_internal_error(var.span, "variable is undefined"))
                }
                MaybeUndefined::Defined(assigned) => match &assigned.value {
                    Value::Simple(v) => Ok(Value::Simple(v.clone())),
                    Value::Compound(v) => Ok(Value::Compound(v.clone())),
                    Value::Hardware(v) => {
                        let v = v.clone().map_version(|index| ValueVersion {
                            signal: SignalOrVariable::Variable(var.inner),
                            index,
                        });
                        Ok(apply_info_to_value(large, self.implied_info(v.version), v))
                    }
                },
            },
            MaybeAssignedValue::NotYetAssigned => {
                let var_info = self.var_info(var)?;
                let diag = Diagnostic::new("variable has not yet been assigned a value")
                    .add_error(var.span, "variable used here")
                    .add_info(var_info.span_decl, "variable declared here")
                    .finish();
                Err(diags.report(diag))
            }
            MaybeAssignedValue::PartiallyAssigned => {
                let var_info = self.var_info(var)?;
                let diag = Diagnostic::new("variable has not yet been assigned a value in all preceding branches")
                    .add_error(var.span, "variable used here")
                    .add_info(var_info.span_decl, "variable declared here")
                    .finish();
                Err(diags.report(diag))
            }
            MaybeAssignedValue::Error(e) => Err(e),
        }
    }

    fn var_new_immutable_init(
        &mut self,
        span_decl: Span,
        id: VariableId,
        assign_span: Span,
        value: DiagResult<Value>,
    ) -> Variable {
        let info = VariableInfo {
            span_decl,
            id,
            mutable: false,
            ty: None,
            use_ir_variable: None,
        };
        let var = self.var_new(info);
        self.var_set(var, assign_span, value);
        var
    }

    // TODO allow capturing hardware values, at least within process?
    //   now that we have composite values that should be doable,
    //   we can then discard the hardware values on conversion to compile time
    fn var_capture(&self, var: Spanned<Variable>) -> DiagResult<CapturedValue> {
        match self.var_get_maybe(var)? {
            MaybeAssignedValue::Assigned(value) => match value {
                MaybeUndefined::Undefined => Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotFullyInitialized)),
                MaybeUndefined::Defined(assigned) => match CompileValue::try_from(assigned.value) {
                    Ok(v) => Ok(CapturedValue::Value(v)),
                    Err(NotCompile) => Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotCompile)),
                },
            },
            MaybeAssignedValue::NotYetAssigned => {
                Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotFullyInitialized))
            }
            MaybeAssignedValue::PartiallyAssigned => {
                Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotFullyInitialized))
            }
            MaybeAssignedValue::Error(e) => Err(e),
        }
    }

    fn var_is_not_yet_assigned(&self, var: Spanned<Variable>) -> DiagResult<bool> {
        match self.var_get_maybe(var)? {
            MaybeAssignedValue::NotYetAssigned => Ok(true),
            MaybeAssignedValue::Assigned(_) | MaybeAssignedValue::PartiallyAssigned => Ok(false),
            MaybeAssignedValue::Error(e) => Err(e),
        }
    }

    fn signal_eval(&self, ctx: &mut CompileItemContext, signal: Spanned<Signal>) -> DiagResult<ValueWithVersion>;
}

#[derive(Debug)]
pub struct FlowRoot<'d> {
    diags: &'d Diagnostics,
    check: RandomCheck,
    next_var_index: Cell<NonZeroUsize>,
    next_version: Cell<NonZeroUsize>,
}

#[derive(Debug)]
pub struct FlowRootContent {
    check: RandomCheck,
    next_var_index: Cell<NonZeroUsize>,
    next_version: Cell<NonZeroUsize>,
}

impl FlowRoot<'_> {
    pub fn new(diags: &Diagnostics) -> FlowRoot<'_> {
        FlowRoot {
            diags,
            check: RandomCheck::new(),
            next_var_index: Cell::new(NON_ZERO_USIZE_ONE),
            // start versions at 2,
            //   so we can always safely use 1 as the first version for values that have not yet been assigned
            next_version: Cell::new(NON_ZERO_USIZE_TWO),
        }
    }

    pub fn into_content(self) -> FlowRootContent {
        FlowRootContent {
            check: self.check,
            next_var_index: self.next_var_index,
            next_version: self.next_version,
        }
    }

    pub fn restore(diags: &Diagnostics, content: FlowRootContent) -> FlowRoot<'_> {
        let FlowRootContent {
            check,
            next_var_index,
            next_version,
        } = content;
        FlowRoot {
            diags,
            check,
            next_var_index,
            next_version,
        }
    }

    fn next_variable(&self) -> Variable {
        let index = self.next_var_index.get();
        self.next_var_index.set(index.checked_add(1).expect("overflow"));
        Variable {
            check: self.check,
            index: VariableIndex(index),
        }
    }

    fn next_version(&self) -> ValueVersionIndex {
        let version = self.next_version.get();
        self.next_version.set(version.checked_add(1).expect("overflow"));
        ValueVersionIndex(version)
    }
}

pub enum FlowKind<C, H> {
    Compile(C),
    Hardware(H),
}

pub struct FlowCompile<'p> {
    root: &'p FlowRoot<'p>,
    compile_span: Span,
    compile_reason: &'static str,

    kind: FlowCompileKind<'p>,

    variable_slots: IndexMap<VariableIndex, VariableSlot>,
}

#[derive(Debug)]
struct VariableSlot {
    info: VariableInfo,
    value: VariableValue,
}

pub enum FlowCompileKind<'p> {
    Root,
    IsolatedCompile(&'p FlowCompile<'p>),
    ScopedCompile(&'p mut FlowCompile<'p>),
    ScopedHardware(&'p mut FlowHardware<'p>),
}

impl FlowPrivate for FlowCompile<'_> {
    fn root(&self) -> &FlowRoot<'_> {
        self.root
    }

    fn implied_info(&self, value: ValueVersion) -> Option<&ImpliedInfo> {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowCompileKind::Root => return None,
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    return parent.implied_info(value);
                }
            }
        }
    }

    fn var_set_maybe(&mut self, var: Variable, assignment_span: Span, value: VariableValue) {
        // checks
        assert_eq!(self.root.check, var.check);
        let diags = self.root.diags;

        // TODO this conversion might be expensive, maybe only do this when assertions are enabled
        // TODO maybe only store unwrapped compile values in compile flows?
        let value = if let MaybeAssignedValue::Assigned(assigned) = &value
            && let MaybeUndefined::Defined(assigned) = &assigned
            && CompileValue::try_from(&assigned.value).is_err()
        {
            let e = diags.report_internal_error(assignment_span, "cannot assign hardware value in compile context");
            MaybeAssignedValue::Error(e)
        } else {
            value
        };

        // find which slot to store the value in
        let mut curr = self;
        let slot = loop {
            if let Some(slot) = curr.variable_slots.get_mut(&var.index) {
                break slot;
            }
            curr = match &mut curr.kind {
                FlowCompileKind::Root => {
                    let _ = diags.report_internal_error(assignment_span, "hit root before finding variable slot");
                    return;
                }
                FlowCompileKind::IsolatedCompile(_) => {
                    // TODO should this actually be an internal error?
                    let _ = diags
                        .report_internal_error(assignment_span, "hit isolated parent before finding variable slot");
                    return;
                }
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    parent.var_set_maybe(var, assignment_span, value);
                    return;
                }
            }
        };

        slot.value = value;
    }

    fn var_get_maybe(&self, var: Spanned<Variable>) -> DiagResult<VariableValueRef<'_>> {
        // TODO block evaluation of hardware values in compile context?
        assert_eq!(self.root.check, var.inner.check);
        let diags = self.root.diags;

        let mut curr = self;
        loop {
            if let Some(entry) = curr.variable_slots.get(&var.inner.index) {
                return Ok(entry.value.as_ref());
            }
            curr = match &curr.kind {
                FlowCompileKind::Root => {
                    return Err(diags.report_internal_error(var.span, "hit root before finding variable slot"));
                }
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    return parent.var_get_maybe(var);
                }
            }
        }
    }

    fn try_var_info(&self, var: VariableIndex) -> Option<&VariableInfo> {
        let mut curr = self;
        loop {
            if let Some(slot) = curr.variable_slots.get(&var) {
                return Some(&slot.info);
            }
            curr = match &curr.kind {
                FlowCompileKind::Root => return None,
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => return parent.try_var_info(var),
            }
        }
    }
}

impl Flow for FlowCompile<'_> {
    fn new_child_compile(&mut self, span: Span, reason: &'static str) -> FlowCompile<'_> {
        let root = self.root;
        let slf = unsafe { lifetime_cast::compile_mut(self) };
        FlowCompile {
            root,
            compile_span: span,
            compile_reason: reason,
            kind: FlowCompileKind::ScopedCompile(slf),
            variable_slots: IndexMap::new(),
        }
    }

    fn check_hardware(&mut self, span: Span, reason: &str) -> DiagResult<&mut FlowHardware<'_>> {
        Err(self.err_not_hardware(span, reason))
    }

    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>> {
        FlowKind::Compile(unsafe { lifetime_cast::compile_mut(self) })
    }

    fn var_new(&mut self, info: VariableInfo) -> Variable {
        let var = self.root.next_variable();

        let slot = VariableSlot {
            info,
            value: MaybeAssignedValue::NotYetAssigned,
        };
        self.variable_slots.insert_first(var.index, slot);

        var
    }

    fn signal_eval(&self, _: &mut CompileItemContext, signal: Spanned<Signal>) -> DiagResult<ValueWithVersion> {
        // get first hardware parent flow
        let mut curr = self;
        let parent_hw = loop {
            curr = match &curr.kind {
                FlowCompileKind::Root => break None,
                &FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => break Some(parent),
            }
        };

        // try evaluating the signal as a compile-time implied value
        if let Some(parent_hw) = parent_hw {
            let signal_version = ValueVersion {
                signal: SignalOrVariable::Signal(signal.inner),
                index: parent_hw.signal_get_version(signal.inner),
            };
            let implied_info = parent_hw.implied_info(signal_version);

            if let Some(ImpliedInfo::Compile(value)) = implied_info {
                return Ok(Value::from(value.clone()));
            }
        }

        // otherwise
        Err(self.err_not_hardware(signal.span, "signal evaluation"))
    }
}

impl<'p> FlowCompile<'p> {
    pub fn new_root(root: &'p FlowRoot, span: Span, reason: &'static str) -> FlowCompile<'p> {
        FlowCompile {
            root,
            compile_span: span,
            compile_reason: reason,
            kind: FlowCompileKind::Root,
            variable_slots: IndexMap::new(),
        }
    }

    // TODO think about a better name
    pub fn new_child_isolated(&self) -> FlowCompile<'_> {
        let slf = unsafe { lifetime_cast::compile_ref(self) };
        FlowCompile {
            root: self.root,
            compile_span: self.compile_span,
            compile_reason: self.compile_reason,
            kind: FlowCompileKind::IsolatedCompile(slf),
            variable_slots: IndexMap::default(),
        }
    }

    pub fn new_child_scoped(&mut self) -> FlowCompile<'_> {
        let root = self.root;
        let compile_span = self.compile_span;
        let compile_reason = self.compile_reason;
        let slf = unsafe { lifetime_cast::compile_mut(self) };
        FlowCompile {
            root,
            compile_span,
            compile_reason,
            kind: FlowCompileKind::ScopedCompile(slf),
            variable_slots: IndexMap::default(),
        }
    }

    pub fn into_content(self) -> FlowCompileContent {
        let FlowCompile {
            root,
            compile_span,
            compile_reason,
            kind: _,
            variable_slots,
        } = self;
        FlowCompileContent {
            check: root.check,
            compile_span,
            compile_reason,
            variable_slots,
        }
    }

    pub fn restore_root(root: &'p FlowRoot, content: FlowCompileContent) -> FlowCompile<'p> {
        let FlowCompileContent {
            check,
            compile_span,
            compile_reason,
            variable_slots,
        } = content;
        assert_eq!(check, root.check);

        FlowCompile {
            root,
            compile_span,
            compile_reason,
            kind: FlowCompileKind::Root,
            variable_slots,
        }
    }

    pub fn restore_child_isolated<'s>(parent: &'s FlowCompile, content: FlowCompileContent) -> FlowCompile<'s> {
        let FlowCompileContent {
            check,
            compile_span,
            compile_reason,
            variable_slots,
        } = content;
        assert_eq!(parent.root.check, check);

        let parent = unsafe { lifetime_cast::compile_ref(parent) };
        FlowCompile {
            root: parent.root,
            compile_span,
            compile_reason,
            kind: FlowCompileKind::IsolatedCompile(parent),
            variable_slots,
        }
    }

    fn err_not_hardware(&self, span: Span, reason: &str) -> DiagError {
        let diag = Diagnostic::new(format!("{reason} is only allowed in a hardware context"))
            .add_error(span, format!("{reason} here"))
            .add_info(
                self.compile_span,
                format!("context is compile-time because of this {}", self.compile_reason),
            )
            .finish();
        self.root.diags.report(diag)
    }
}

pub struct FlowHardwareRoot<'p> {
    root: &'p FlowRoot<'p>,
    parent: &'p FlowCompile<'p>,

    span: Span,
    process_kind: HardwareProcessKind<'p>,

    ir_wires: &'p mut IrWires,
    ir_registers: &'p mut IrRegisters,
    ir_variables: IrVariables,
    signal_shadow_vars: IndexMap<Signal, ShadowSignalInfo>,

    common: FlowHardwareCommon,
}

pub struct ShadowSignalInfo {
    signal_ir: IrSignal,
    var_ir: IrVariable,
}

pub struct FlowHardwareBranch<'p> {
    root: &'p FlowRoot<'p>,
    parent: &'p mut FlowHardware<'p>,

    cond_domain: Spanned<ValueDomain>,
    common: FlowHardwareCommon,
}

struct FlowHardwareCommon {
    variables: FlowHardwareVariables,
    signal_versions: IndexMap<Signal, ValueVersionIndex>,
    statements: Vec<Spanned<IrStatement>>,
    implied_info: IndexMap<ValueVersion, ImpliedInfo>,
}

#[derive(Debug)]
enum ImpliedInfo {
    Compile(CompileValue),
    IntRange(ClosedNonEmptyMultiRange<BigInt>),
}

fn apply_info_to_value(
    large: &mut IrLargeArena,
    info: Option<&ImpliedInfo>,
    value: HardwareValueWithVersion,
) -> ValueWithVersion {
    match info {
        None => ValueWithVersion::Hardware(value),
        Some(info) => match info {
            ImpliedInfo::Compile(value) => ValueWithVersion::from(value.clone()),
            ImpliedInfo::IntRange(implied_range) => {
                let HardwareValueWithVersion { value, version } = value;
                let HardwareValue { ty, domain, expr } = value;
                assert!(matches!(ty, HardwareType::Int(_)));

                let expr_constrained = large.push_expr(IrExpressionLarge::ConstrainIntRange(
                    implied_range.enclosing_range().cloned(),
                    expr,
                ));
                Value::Hardware(HardwareValueWithVersion {
                    value: HardwareValue {
                        ty: HardwareType::Int(implied_range.clone()),
                        domain,
                        expr: expr_constrained,
                    },
                    version,
                })
            }
        },
    }
}

struct FlowHardwareVariables {
    // We store both the info and the value as a single combined entry, to avoid duplicate map lookups and insertions.
    combined: IndexMap<VariableIndex, VariableInfoAndValue>,
}

#[derive(Default)]
struct VariableInfoAndValue {
    info: Option<VariableInfo>,
    value: Option<VariableValue>,
}

// TODO track signals very similarly to variables, certainly once written to at least once
//   (then we can also stop shadowing signals in the verilog backend, and handle it in the frontend)
pub struct FlowHardware<'p> {
    root: &'p FlowRoot<'p>,
    enable_domain_checks: bool,
    kind: FlowHardwareKind<'p>,
}

enum FlowHardwareKind<'p> {
    Root(&'p mut FlowHardwareRoot<'p>),
    Branch(&'p mut FlowHardwareBranch<'p>),
    Scoped(FlowHardwareScoped<'p>),
}

struct FlowHardwareScoped<'p> {
    parent: &'p mut FlowHardware<'p>,
    variables: FlowHardwareVariables,
}

impl FlowHardwareCommon {
    fn new(implied_types: IndexMap<ValueVersion, ImpliedInfo>) -> FlowHardwareCommon {
        FlowHardwareCommon {
            variables: FlowHardwareVariables {
                combined: IndexMap::new(),
            },
            signal_versions: IndexMap::new(),
            statements: vec![],
            implied_info: implied_types,
        }
    }
}

impl FlowPrivate for FlowHardware<'_> {
    fn root(&self) -> &FlowRoot<'_> {
        self.root
    }

    fn implied_info(&self, value: ValueVersion) -> Option<&ImpliedInfo> {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => {
                    return root.common.implied_info.get(&value);
                }
                FlowHardwareKind::Branch(branch) => {
                    if let Some(ty) = branch.common.implied_info.get(&value) {
                        return Some(ty);
                    }
                    branch.parent
                }
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
            }
        }
    }

    fn var_set_maybe(&mut self, var: Variable, _: Span, value: VariableValue) {
        assert_eq!(self.root.check, var.check);
        // TODO don't allow setting variables declared outside of the root hardware flow
        //   (nvm for now, this will all become simpler once constants are no longer in the flow)

        let mut curr = self;
        let variables = loop {
            curr = match &mut curr.kind {
                FlowHardwareKind::Root(root) => break &mut root.common.variables,
                FlowHardwareKind::Branch(branch) => break &mut branch.common.variables,
                FlowHardwareKind::Scoped(scoped) => {
                    // if the value was declared in this scope we can keep it here,
                    //   otherwise fallthrough to the parent
                    if scoped.variables.combined.contains_key(&var.index) {
                        break &mut scoped.variables;
                    }
                    scoped.parent
                }
            }
        };

        variables.combined.entry(var.index).or_default().value = Some(value);
    }

    fn var_get_maybe(&self, var: Spanned<Variable>) -> DiagResult<VariableValueRef<'_>> {
        assert_eq!(self.root.check, var.inner.check);

        let mut curr = self;
        loop {
            let (variables, next) = match &curr.kind {
                FlowHardwareKind::Root(root) => (&root.common.variables, Either::Left(root)),
                FlowHardwareKind::Branch(branch) => (&branch.common.variables, Either::Right(&branch.parent)),
                FlowHardwareKind::Scoped(scoped) => (&scoped.variables, Either::Right(&scoped.parent)),
            };
            if let Some(combined) = variables.combined.get(&var.inner.index)
                && let Some(value) = &combined.value
            {
                return Ok(value.as_ref());
            }
            curr = match next {
                Either::Left(root) => return root.parent.var_get_maybe(var),
                Either::Right(next) => next,
            };
        }
    }

    fn try_var_info(&self, var: VariableIndex) -> Option<&VariableInfo> {
        // TODO fix code duplication
        let mut curr = self;
        loop {
            let (variables, next) = match &curr.kind {
                FlowHardwareKind::Root(root) => (&root.common.variables, Either::Left(root)),
                FlowHardwareKind::Branch(branch) => (&branch.common.variables, Either::Right(&branch.parent)),
                FlowHardwareKind::Scoped(scoped) => (&scoped.variables, Either::Right(&scoped.parent)),
            };
            if let Some(info) = variables.combined.get(&var)
                && let Some(info) = &info.info
            {
                return Some(info);
            }
            curr = match next {
                Either::Left(root) => return root.parent.try_var_info(var),
                Either::Right(next) => next,
            };
        }
    }
}

impl Flow for FlowHardware<'_> {
    fn new_child_compile(&mut self, span: Span, reason: &'static str) -> FlowCompile<'_> {
        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        FlowCompile {
            root,
            compile_span: span,
            compile_reason: reason,
            kind: FlowCompileKind::ScopedHardware(slf),
            variable_slots: IndexMap::new(),
        }
    }

    fn check_hardware(&mut self, _: Span, _: &str) -> DiagResult<&mut FlowHardware<'_>> {
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        Ok(slf)
    }

    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>> {
        FlowKind::Hardware(unsafe { lifetime_cast::hardware_mut(self) })
    }

    fn var_new(&mut self, info: VariableInfo) -> Variable {
        let variables = match &mut self.kind {
            FlowHardwareKind::Root(root) => &mut root.common.variables,
            FlowHardwareKind::Branch(branch) => &mut branch.common.variables,
            FlowHardwareKind::Scoped(scoped) => &mut scoped.variables,
        };

        let var = self.root.next_variable();

        let combined = VariableInfoAndValue {
            info: Some(info),
            value: Some(MaybeAssignedValue::NotYetAssigned),
        };
        variables.combined.insert_first(var.index, combined);

        var
    }

    fn signal_eval(&self, ctx: &mut CompileItemContext, signal: Spanned<Signal>) -> DiagResult<ValueWithVersion> {
        // try compile-time evaluation first to skip any checking
        let version = ValueVersion {
            signal: SignalOrVariable::Signal(signal.inner),
            index: self.signal_get_version(signal.inner),
        };
        let info = self.implied_info(version);
        if let Some(ImpliedInfo::Compile(value)) = info {
            return Ok(Value::from(value.clone()));
        }

        // otherwise, do hardware evaluation
        let value = signal.inner.as_hardware_value(ctx, signal.span)?;

        // For clocked blocks, check read domain validness.
        // This is technically not needed for correctness (assignments also check domain validness),
        //   but it makes error messages a bit clearer.
        // TODO this probably disables too many things, eg. nested blocks
        if self.enable_domain_checks {
            match &self.root_hw().process_kind {
                &HardwareProcessKind::ClockedBlockBody { domain, .. } => {
                    ctx.check_valid_domain_crossing(
                        signal.span,
                        domain.map_inner(ValueDomain::Sync),
                        Spanned::new(signal.span, value.domain),
                        "signal read in clocked block",
                    )?;
                }
                HardwareProcessKind::CombinatorialBlockBody { .. }
                | HardwareProcessKind::WireExpression { .. }
                | HardwareProcessKind::InstancePortConnection { .. } => {}
            }
        }

        // apply implications
        let value_with_version = HardwareValueWithVersion { value, version };
        Ok(apply_info_to_value(&mut ctx.large, info, value_with_version))
    }
}

impl<'p> FlowHardwareRoot<'p> {
    pub fn new(
        parent: &'p FlowCompile,
        span: Span,
        kind: HardwareProcessKind<'p>,
        ir_wires: &'p mut IrWires,
        ir_registers: &'p mut IrRegisters,
    ) -> FlowHardwareRoot<'p> {
        let parent = unsafe { lifetime_cast::compile_ref(parent) };
        FlowHardwareRoot {
            root: parent.root,
            parent,
            span,
            process_kind: kind,
            ir_wires,
            ir_registers,
            ir_variables: IrVariables::new(),
            signal_shadow_vars: IndexMap::new(),
            common: FlowHardwareCommon::new(IndexMap::new()),
        }
    }

    pub fn as_flow(&mut self) -> FlowHardware<'_> {
        let slf = unsafe { lifetime_cast::hardware_root_mut(self) };
        FlowHardware {
            root: slf.root,
            enable_domain_checks: true,
            kind: FlowHardwareKind::Root(slf),
        }
    }

    pub fn finish(self) -> (IrVariables, IrBlock) {
        // TODO for combinatorial blocks, check that signals are _fully_ driven
        let FlowHardwareRoot {
            root: _,
            parent: _,
            span,
            process_kind: _,
            ir_wires: _,
            ir_registers: _,
            ir_variables,
            signal_shadow_vars,
            common,
        } = self;
        let FlowHardwareCommon {
            variables: _,
            signal_versions: _,
            statements: mut ir_statements,
            implied_info: _,
        } = common;

        // insert shadow copies at the start of the block
        let mut all_statements = vec![];
        for (_, info) in signal_shadow_vars {
            let stmt = IrStatement::Assign(
                IrAssignmentTarget::simple(info.var_ir.into()),
                IrExpression::Signal(info.signal_ir),
            );
            all_statements.push(Spanned::new(span, stmt));
        }
        all_statements.append(&mut ir_statements);

        let block = IrBlock {
            statements: all_statements,
        };

        (ir_variables, block)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ImplicationContradiction;

impl<'p> FlowHardware<'p> {
    fn root_hw<'s>(&'s self) -> &'s FlowHardwareRoot<'p> {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => return root,
                FlowHardwareKind::Branch(branch) => branch.parent,
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
            }
        }
    }

    fn root_hw_mut<'s>(&'s mut self) -> &'s mut FlowHardwareRoot<'p> {
        let mut curr = self;
        loop {
            curr = match &mut curr.kind {
                FlowHardwareKind::Root(root) => return root,
                FlowHardwareKind::Branch(branch) => branch.parent,
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
            }
        }
    }

    pub fn new_child_branch(
        &mut self,
        ctx: &mut CompileItemContext,
        span: Span,
        cond_domain: Spanned<ValueDomain>,
        cond_implications: Vec<Implication>,
    ) -> DiagResult<Result<FlowHardwareBranch<'_>, ImplicationContradiction>> {
        let implied_types = try_inner!(self.implications_to_implied_info(ctx, span, cond_implications)?);

        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        Ok(Ok(FlowHardwareBranch {
            root,
            parent: slf,
            cond_domain,
            common: FlowHardwareCommon::new(implied_types),
        }))
    }

    fn implications_to_implied_info(
        &self,
        ctx: &mut CompileItemContext,
        span: Span,
        implications: Vec<Implication>,
    ) -> DiagResult<Result<IndexMap<ValueVersion, ImpliedInfo>, ImplicationContradiction>> {
        let mut implied_info = IndexMap::new();

        for implication in implications {
            let Implication { version, kind } = implication;
            let curr_info = implied_info.get(&version).or_else(|| self.implied_info(version));

            match kind {
                ImplicationKind::BoolEq(new_value) => {
                    // check for contradiction
                    if let Some(&ImpliedInfo::Compile(CompileValue::Simple(SimpleCompileValue::Bool(curr_value)))) =
                        curr_info
                    {
                        if curr_value != new_value {
                            return Ok(Err(ImplicationContradiction));
                        }
                    }

                    // set new implication
                    implied_info.insert(
                        version,
                        ImpliedInfo::Compile(CompileValue::Simple(SimpleCompileValue::Bool(new_value))),
                    );
                }
                ImplicationKind::IntIn(new_range) => {
                    let curr_range = match curr_info {
                        // existing implied info, use that
                        Some(curr_info) => unwrap_match!(curr_info, ImpliedInfo::IntRange(curr_range) => curr_range),
                        // no implications yet, get range from the base type
                        None => {
                            let curr_ty = match version.signal {
                                SignalOrVariable::Signal(signal) => {
                                    // According to the signature this can fail due to the type not yet being inferred,
                                    //  but here that isn't actually be possible: the signal is present in an implication,
                                    //  so it has been evaluated already.
                                    signal.expect_ty(ctx, span)?.inner
                                }
                                SignalOrVariable::Variable(var) => {
                                    // We only care about variables that are assigned, defined and store a hardware value.
                                    let var_value = self.var_get_maybe(Spanned::new(span, var))?;
                                    if let VariableValueRef::Assigned(var_info) = var_value
                                        && let MaybeUndefined::Defined(var_info) = var_info
                                        && let Value::Hardware(var_value) = var_info.value
                                    {
                                        &var_value.value.ty
                                    } else {
                                        continue;
                                    }
                                }
                            };

                            unwrap_match!(curr_ty, HardwareType::Int(curr_range) => curr_range)
                        }
                    };

                    // intersect with existing range to get contradiction or new range
                    let new_range = ClosedMultiRange::from(curr_range.clone()).intersect(&new_range);
                    match ClosedNonEmptyMultiRange::try_from(new_range) {
                        Ok(new_range) => {
                            implied_info.insert(version, ImpliedInfo::IntRange(new_range));
                        }
                        Err(RangeEmpty) => {
                            return Ok(Err(ImplicationContradiction));
                        }
                    }
                }
            }
        }

        Ok(Ok(implied_info))
    }

    pub fn new_child_scoped(&mut self) -> FlowHardware<'_> {
        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        FlowHardware {
            root,
            enable_domain_checks: true,
            kind: FlowHardwareKind::Scoped(FlowHardwareScoped {
                parent: slf,
                variables: FlowHardwareVariables {
                    combined: IndexMap::new(),
                },
            }),
        }
    }

    pub fn new_child_scoped_without_domain_checks(&mut self) -> FlowHardware<'_> {
        let mut result = self.new_child_scoped();
        result.enable_domain_checks = false;
        result
    }

    pub fn join_child_branches_pair(
        &mut self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span_merge: Span,
        branches: (FlowHardwareBranchContent, FlowHardwareBranchContent),
    ) -> DiagResult<(IrBlock, IrBlock)> {
        let (branch_0, branch_1) = branches;
        let branches = vec![branch_0, branch_1];

        let result = self.join_child_branches(refs, large, span_merge, branches)?;

        assert_eq!(result.len(), 2);
        let mut result = result.into_iter();
        let block_0 = result.next().unwrap();
        let block_1 = result.next().unwrap();
        Ok((block_0, block_1))
    }

    pub fn join_child_branches(
        &mut self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span_merge: Span,
        mut branches: Vec<FlowHardwareBranchContent>,
    ) -> DiagResult<Vec<IrBlock>> {
        // TODO merge implications too
        // TODO do something else if children are empty, eg. push an instruction for `assert(false)`

        // collect the interesting vars and signals
        // things we can skip:
        //   * items that are not in any child, they didn't change
        //   * items that don't exist in the parent, they won't be alive afterwards
        let mut merged_vars = IndexSet::new();
        let mut merged_signals = IndexSet::new();
        for branch in &branches {
            assert_eq!(self.root.check, branch.check);
            for &var in branch.common.variables.combined.keys() {
                if self.try_var_info(var).is_some() {
                    merged_vars.insert(var);
                }
            }
            for &signal in branch.common.signal_versions.keys() {
                merged_signals.insert(signal);
            }
        }

        // merge variables
        for var in merged_vars {
            let var = Variable {
                check: self.root.check,
                index: var,
            };
            let var_info = self.var_info(Spanned::new(span_merge, var))?;
            let var_span_decl = var_info.span_decl;

            let value_merged = merge_branch_values(refs, large, self, span_merge, var, &mut branches)
                .unwrap_or_else(VariableValue::Error);

            self.var_set_maybe(var, var_span_decl, value_merged);
        }

        // merge signals
        for signal in merged_signals {
            let version_parent = self.signal_get_version(signal);

            let mut merged: Option<Result<ValueVersionIndex, NeedsHardwareMerge>> = None;
            for branch in &branches {
                let version_child = branch
                    .common
                    .signal_versions
                    .get(&signal)
                    .copied()
                    .unwrap_or(version_parent);

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
                Some(Err(NeedsHardwareMerge)) => self.root.next_version(),
            };
            self.signal_set_version(signal, merged_version);
        }

        // extract blocks
        let mut branch_implied_types = vec![];
        let branch_blocks = branches
            .into_iter()
            .map(|branch| {
                let FlowHardwareBranchContent {
                    check: _,
                    cond_domain: _,
                    common,
                } = branch;
                let FlowHardwareCommon {
                    variables: _,
                    signal_versions: _,
                    statements,
                    implied_info: implied_types,
                } = common;
                branch_implied_types.push(implied_types);
                IrBlock { statements }
            })
            .collect_vec();

        merge_branch_implied_types(&mut self.first_common_mut().implied_info, &branch_implied_types);

        // TODO merge condition domains too?
        Ok(branch_blocks)
    }

    // TODO can we remove this again?
    pub fn get_ir_wires(&mut self) -> &mut IrWires {
        self.root_hw_mut().ir_wires
    }

    // TODO extract more of these?
    fn first_common_mut(&mut self) -> &mut FlowHardwareCommon {
        let mut curr = self;
        loop {
            curr = match &mut curr.kind {
                FlowHardwareKind::Root(root) => return &mut root.common,
                FlowHardwareKind::Branch(branch) => return &mut branch.common,
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
            }
        }
    }

    pub fn push_ir_statement(&mut self, statement: Spanned<IrStatement>) {
        self.first_common_mut().statements.push(statement);
    }

    pub fn new_ir_variable(&mut self, info: IrVariableInfo) -> IrVariable {
        self.root_hw_mut().ir_variables.push(info)
    }

    pub fn check_clocked_block(
        &mut self,
        span: Span,
        reason: &str,
    ) -> DiagResult<(
        Spanned<SyncDomain<DomainSignal>>,
        &mut ExtraRegisters<'p>,
        &mut IrRegisters,
    )> {
        let root_hw = self.root_hw_mut();
        let diags = root_hw.root.diags;

        let report_err = |span_curr: Span, kind_curr: &str| {
            let diag = Diagnostic::new(format!("{reason} is only allowed in a clocked block"))
                .add_error(span, format!("{reason} used here"))
                .add_info(span_curr, format!("currently inside this {kind_curr}"))
                .finish();
            diags.report(diag)
        };

        match &mut root_hw.process_kind {
            HardwareProcessKind::ClockedBlockBody {
                span_keyword: _,
                domain,
                registers_driven: _,
                extra_registers,
            } => Ok((*domain, extra_registers, root_hw.ir_registers)),

            &mut HardwareProcessKind::CombinatorialBlockBody { span_keyword, .. } => {
                Err(report_err(span_keyword, "combinatorial block"))
            }
            &mut HardwareProcessKind::WireExpression { span_keyword, .. } => {
                Err(report_err(span_keyword, "wire expression"))
            }
            &mut HardwareProcessKind::InstancePortConnection { span_connection } => {
                Err(report_err(span_connection, "instance port connection"))
            }
        }
    }

    pub fn block_kind(&mut self) -> &mut HardwareProcessKind<'p> {
        &mut self.root_hw_mut().process_kind
    }

    pub fn condition_domains(&self) -> impl Iterator<Item = Spanned<ValueDomain>> + use<'_, 'p> {
        std::iter::successors(Some(self), |curr| match &curr.kind {
            FlowHardwareKind::Root(_) => None,
            FlowHardwareKind::Branch(branch) => Some(branch.parent),
            FlowHardwareKind::Scoped(scoped) => Some(scoped.parent),
        })
        .filter_map(|curr| match &curr.kind {
            FlowHardwareKind::Root(_) => None,
            FlowHardwareKind::Scoped(_) => None,
            FlowHardwareKind::Branch(branch) => Some(branch.cond_domain),
        })
    }

    fn signal_get_version(&self, signal: Signal) -> ValueVersionIndex {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => {
                    if let Some(&version) = root.common.signal_versions.get(&signal) {
                        break version;
                    }

                    // we've reached the root without finding a version for this signal,
                    //   this means it has not yet been written and we can safely assign it the first possible version
                    break ValueVersionIndex(NON_ZERO_USIZE_ONE);
                }
                FlowHardwareKind::Branch(branch) => {
                    if let Some(&version) = branch.common.signal_versions.get(&signal) {
                        break version;
                    }
                    branch.parent
                }
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
            }
        }
    }

    fn signal_set_version(&mut self, signal: Signal, version: ValueVersionIndex) {
        self.first_common_mut().signal_versions.insert(signal, version);
    }

    pub fn signal_assign(
        &mut self,
        ctx: &mut CompileItemContext,
        assign_span: Span,
        signal: Spanned<Signal>,
        steps: Vec<IrTargetStep>,
        value_hardware: HardwareValue,
        value_compile: Option<CompileValue>,
    ) -> DiagResult {
        // TODO track partial drivers to avoid clashes and to generate extra concat blocks
        // TODO comb blocks: check that all written signals are _always_ written
        // TODO ban signal assignments in wire expressions and port connections

        // get signal info
        let (signal_ty, signal_ir) = signal.inner.expect_ty_and_ir(ctx, signal.span)?;
        let signal_ty = signal_ty.cloned();
        let signal_id = signal.inner.diagnostic_string(ctx).to_owned();

        // get or create shadow variable
        let root_hw = self.root_hw_mut();
        let shadow_var_ir = root_hw
            .signal_shadow_vars
            .entry(signal.inner)
            .or_insert_with(|| {
                let var_info = IrVariableInfo {
                    ty: signal_ty.inner.as_ir(ctx.refs),
                    debug_info_span: root_hw.span,
                    debug_info_id: Some(signal_id),
                };
                let var_ir = root_hw.ir_variables.push(var_info);
                ShadowSignalInfo { signal_ir, var_ir }
            })
            .var_ir;

        // expand value to target type
        let value_hardware_expanded =
            value_hardware.as_hardware_value_unchecked(ctx.refs, &mut ctx.large, assign_span, signal_ty.inner)?;

        // assign to both signal and the shadow variable
        let mut push_stmt = |target: IrSignalOrVariable, steps: Vec<IrTargetStep>, value: IrExpression| {
            let stmt = IrStatement::Assign(IrAssignmentTarget { base: target, steps }, value);
            let stmt = Spanned::new(assign_span, stmt);
            self.push_ir_statement(stmt);
        };
        push_stmt(signal_ir.into(), steps.clone(), value_hardware_expanded.expr.clone());
        push_stmt(shadow_var_ir.into(), steps, value_hardware_expanded.expr);

        // bump version
        let version = self.root.next_version();
        self.signal_set_version(signal.inner, version);

        // store implied info if any
        let implied_info = &mut self.first_common_mut().implied_info;
        let value_version = ValueVersion {
            signal: SignalOrVariable::Signal(signal.inner),
            index: version,
        };
        if let Some(value_compile) = value_compile {
            implied_info.insert(value_version, ImpliedInfo::Compile(value_compile));
        } else if let HardwareType::Int(ty) = value_hardware.ty {
            implied_info.insert(value_version, ImpliedInfo::IntRange(ty));
        } else {
            implied_info.swap_remove(&value_version);
        }

        Ok(())
    }
}

pub struct FlowHardwareBranchContent {
    check: RandomCheck,
    // TODO think about how this should be propagated, eg. if two branches break and the third one doesn't,
    //   the value of all merged values and even implications might depend on the condition domain
    cond_domain: Spanned<ValueDomain>,
    common: FlowHardwareCommon,
}

impl FlowHardwareBranch<'_> {
    pub fn as_flow(&mut self) -> FlowHardware<'_> {
        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_branch_mut(self) };
        FlowHardware {
            root,
            enable_domain_checks: true,
            kind: FlowHardwareKind::Branch(slf),
        }
    }

    pub fn finish(self) -> FlowHardwareBranchContent {
        let FlowHardwareBranch {
            root,
            parent: _,
            cond_domain,
            common,
        } = self;
        FlowHardwareBranchContent {
            check: root.check,
            cond_domain,
            common,
        }
    }
}

#[derive(Debug)]
pub struct FlowCompileContent {
    check: RandomCheck,
    compile_span: Span,
    compile_reason: &'static str,

    variable_slots: IndexMap<VariableIndex, VariableSlot>,
}

// TODO rename to BlockKind? at least make sure everything is consistent
pub enum HardwareProcessKind<'e> {
    CombinatorialBlockBody {
        span_keyword: Span,
        wires_driven: &'e mut IndexMap<Wire, Span>,
        ports_driven: &'e mut IndexMap<Port, Span>,
    },
    ClockedBlockBody {
        span_keyword: Span,
        domain: Spanned<SyncDomain<DomainSignal>>,
        registers_driven: &'e mut IndexMap<Register, Span>,
        extra_registers: ExtraRegisters<'e>,
    },
    WireExpression {
        span_keyword: Span,
        span_init: Span,
    },
    InstancePortConnection {
        span_connection: Span,
    },
}

pub enum ExtraRegisters<'e> {
    NoReset,
    WithReset(&'e mut Vec<ExtraRegisterInit>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Variable {
    check: RandomCheck,
    index: VariableIndex,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct VariableIndex(NonZeroUsize);

#[derive(Debug)]
pub struct VariableInfo {
    pub span_decl: Span,
    pub id: VariableId,
    pub mutable: bool,
    pub ty: Option<Spanned<Type>>,
    pub use_ir_variable: Option<IrVariable>,
}

#[derive(Debug, Copy, Clone)]
pub enum VariableId {
    Id(MaybeIdentifier),
    Custom(&'static str),
}

impl VariableId {
    pub fn str(self, source: &SourceDatabase) -> Option<&str> {
        match self {
            VariableId::Id(MaybeIdentifier::Identifier(id)) => Some(id.str(source)),
            VariableId::Id(MaybeIdentifier::Dummy { span: _ }) => None,
            VariableId::Custom(s) => Some(s),
        }
    }
}

type FlowValue = Value<SimpleCompileValue, MixedCompoundValue, HardwareValueWithVersion<ValueVersionIndex>>;
type VariableValue = MaybeAssignedValue<FlowValue>;
type VariableValueRef<'a> = MaybeAssignedValue<&'a FlowValue>;

#[derive(Debug, Copy, Clone)]
enum MaybeAssignedValue<V> {
    /// The undefined case acts as if the variable has been assigned a value, without that actually being true.
    /// This should only be used if the caller can somehow guarantee that
    /// the variable will _actually_ be assigned before any use, but [Flow] does not realize this by itself.
    Assigned(MaybeUndefined<AssignedValue<V>>),
    NotYetAssigned,
    PartiallyAssigned,
    Error(DiagError),
}

#[derive(Debug, Copy, Clone)]
struct AssignedValue<V> {
    last_assignment_span: Span,
    value: V,
}

impl<V> MaybeAssignedValue<V> {
    pub fn as_ref(&self) -> MaybeAssignedValue<&V> {
        match self {
            MaybeAssignedValue::Assigned(v) => {
                MaybeAssignedValue::Assigned(v.as_ref().map_defined(|v| AssignedValue {
                    last_assignment_span: v.last_assignment_span,
                    value: &v.value,
                }))
            }
            MaybeAssignedValue::NotYetAssigned => MaybeAssignedValue::NotYetAssigned,
            MaybeAssignedValue::PartiallyAssigned => MaybeAssignedValue::PartiallyAssigned,
            &MaybeAssignedValue::Error(e) => MaybeAssignedValue::Error(e),
        }
    }
}

// TODO find a better name for this
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ValueVersion {
    signal: SignalOrVariable,
    index: ValueVersionIndex,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct ValueVersionIndex(NonZeroUsize);

struct NeedsHardwareMerge;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum CapturedValue {
    Item(AstRefItem),
    Value(CompileValue),
    FailedCapture(FailedCaptureReason),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum FailedCaptureReason {
    NotCompile,
    NotFullyInitialized,
}

// TODO re-use IR variables where possible to reduce noise in the generated RTL
//   conditions where this is possible:
//   * the previous value is already in a IR variable for the corresponding high-level variable
//   * the IRVariable is not itself still used elsewhere
//       Can we somehow guarantee that? not really, if we hand out ir variables during variable eval,
//       callers might hold on to them for a while.
//       Maybe we should just make the convention that variables are not valid access calls to flow.join.
//   Once we add this we can remove the hardcoded special case for exit flags,
//       and maybe even remove the "unused variable" optimization from lower_verilog
fn merge_branch_values(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    parent_flow: &mut FlowHardware,
    span_merge: Span,
    var: Variable,
    branches: &mut [FlowHardwareBranchContent],
) -> DiagResult<VariableValue> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    let var_spanned = Spanned::new(span_merge, var);
    let var_info = parent_flow.var_info(var_spanned)?;
    let parent_value = parent_flow.var_get_maybe(var_spanned)?;

    // visit all branch values to check if we need to do a hardware merge
    #[derive(Debug)]
    enum Merged<'a> {
        AllMatch(AssignedValue<&'a FlowValue>),
        NotFullyAssigned,
        NeedsHardwareMerge,
    }
    let mut merged: Option<Merged> = None;

    let mut used_parent_value = false;

    for branch in &mut *branches {
        // TODO apply implications here too
        let branch_value = match branch.common.variables.combined.get(&var.index) {
            Some(branch_combined) => {
                assert!(branch_combined.info.is_none());
                branch_combined.value.as_ref().unwrap().as_ref()
            }
            None => {
                used_parent_value = true;
                parent_value
            }
        };

        // first (non-undefined) branch, just set it
        let Some(merged_value) = merged else {
            merged = match branch_value {
                MaybeAssignedValue::Assigned(value) => match value {
                    MaybeUndefined::Undefined => None,
                    MaybeUndefined::Defined(value) => Some(Merged::AllMatch(value)),
                },
                MaybeAssignedValue::NotYetAssigned | MaybeAssignedValue::PartiallyAssigned => {
                    Some(Merged::NotFullyAssigned)
                }
                MaybeAssignedValue::Error(e) => return Err(e),
            };
            continue;
        };

        // join multiple values
        let merged_new = match (merged_value, branch_value) {
            (Merged::NotFullyAssigned, _)
            | (_, MaybeAssignedValue::NotYetAssigned | MaybeAssignedValue::PartiallyAssigned) => {
                // if anything is not fully assigned, the result is not fully assigned
                Merged::NotFullyAssigned
            }
            (_, MaybeAssignedValue::Error(e)) => {
                // short-circuit on error
                return Err(e);
            }
            (Merged::NeedsHardwareMerge, MaybeAssignedValue::Assigned(_)) => {
                // if we need a hardware merge and we get more normal values, we will still need a hardware merge
                Merged::NeedsHardwareMerge
            }
            (Merged::AllMatch(merged), MaybeAssignedValue::Assigned(MaybeUndefined::Undefined)) => {
                // undefined does not care about the result value, so keep whatever we have so far
                Merged::AllMatch(merged)
            }
            (Merged::AllMatch(merged), MaybeAssignedValue::Assigned(MaybeUndefined::Defined(branch))) => {
                // actual merge check between two assigned values
                let value_matches = match (merged.value, branch.value) {
                    (FlowValue::Simple(merged), FlowValue::Simple(branch)) => merged == branch,
                    (FlowValue::Hardware(merged), FlowValue::Hardware(branch)) => merged.version == branch.version,

                    (FlowValue::Compound(merged), FlowValue::Compound(branch)) => {
                        if let Ok(merged) = CompileCompoundValue::try_from(merged)
                            && let Ok(branch) = CompileCompoundValue::try_from(branch)
                        {
                            merged == branch
                        } else {
                            false
                        }
                    }

                    (FlowValue::Simple(_), FlowValue::Hardware(_) | FlowValue::Compound(_)) => false,
                    (FlowValue::Hardware(_), FlowValue::Simple(_) | FlowValue::Compound(_)) => false,
                    (FlowValue::Compound(_), FlowValue::Simple(_) | FlowValue::Hardware(_)) => false,
                };

                if value_matches {
                    let span_new = if merged.last_assignment_span == branch.last_assignment_span {
                        merged.last_assignment_span
                    } else {
                        span_merge
                    };
                    Merged::AllMatch(AssignedValue {
                        last_assignment_span: span_new,
                        value: merged.value,
                    })
                } else {
                    Merged::NeedsHardwareMerge
                }
            }
        };
        merged = Some(merged_new);
    }

    // handle the merge result
    match merged {
        None => {
            // There were no (non-undefined) branches, this implies that control flow diverges or that all branches
            //   have an undefined value.
            return Ok(MaybeAssignedValue::Assigned(MaybeUndefined::Undefined));
        }
        Some(Merged::AllMatch(value)) => {
            // all branches agree on the assigned value, we don't need to do a hardware merge
            let assigned = AssignedValue {
                last_assignment_span: value.last_assignment_span,
                value: value.value.clone(),
            };
            return Ok(MaybeAssignedValue::Assigned(MaybeUndefined::Defined(assigned)));
        }
        Some(Merged::NotFullyAssigned) => {
            // fully un-assigned would have exited earlier, so the result is partially assigned
            return Ok(MaybeAssignedValue::PartiallyAssigned);
        }
        Some(Merged::NeedsHardwareMerge) => {
            // fall through into hardware merge
        }
    }

    // at this point we know that there are no un-assigned branches, and that not all branches have the same value:
    //   this means we actually need to do a hardware merge
    fn unwrap_branch_value(v: VariableValueRef<'_>) -> MaybeUndefined<AssignedValue<&FlowValue>> {
        unwrap_match!(v, MaybeAssignedValue::Assigned(value) => value)
    }

    // check that all types are hardware
    // (we do this before finding the common type to get nicer error messages)
    let branch_tys = branches
        .iter()
        .map(|branch| {
            let branch_value = match branch.common.variables.combined.get(&var.index) {
                Some(branch_combined) => {
                    assert!(branch_combined.info.is_none());
                    branch_combined.value.as_ref().unwrap().as_ref()
                }
                None => parent_value,
            };

            let branch_value = unwrap_branch_value(branch_value);

            let branch_value = match branch_value {
                MaybeUndefined::Undefined => return Ok(HardwareType::Undefined),
                MaybeUndefined::Defined(branch_value) => branch_value,
            };
            let branch_ty = match &branch_value.value {
                Value::Simple(v) => v.ty(),
                Value::Compound(v) => v.ty(),
                Value::Hardware(v) => v.value.ty.as_type(),
            };

            branch_ty.as_hardware_type(elab).map_err(|_| {
                let ty_str = branch_ty.value_string(elab);
                let diag = Diagnostic::new("merging if assignments needs hardware type")
                    .add_info(var_info.span_decl, "for this variable")
                    .add_info(
                        branch_value.last_assignment_span,
                        format!("value assigned here has type `{ty_str}` which cannot be represented in hardware"),
                    )
                    .add_error(span_merge, "merging happens here")
                    .finish();
                diags.report(diag)
            })
        })
        .try_collect_all_vec()?;

    // find common type
    let ty = branch_tys.iter().fold(Type::Undefined, |a, t| a.union(&t.as_type()));

    // convert common to hardware too
    let ty = ty.as_hardware_type(elab).map_err(|_| {
        let ty_str = ty.value_string(elab);

        let mut diag = Diagnostic::new("merging if assignments needs hardware type")
            .add_info(var_info.span_decl, "for this variable")
            .add_error(
                span_merge,
                format!("merging happens here, combined type `{ty_str}` cannot be represented in hardware"),
            );

        for (branch, ty) in zip_eq(&*branches, branch_tys) {
            if let Some(branch_combined) = branch.common.variables.combined.get(&var.index) {
                assert!(branch_combined.info.is_none());
                let branch_combined = branch_combined.value.as_ref().unwrap().as_ref();

                let branch_value = unwrap_branch_value(branch_combined);

                match branch_value {
                    MaybeUndefined::Undefined => {}
                    MaybeUndefined::Defined(branch_value) => {
                        diag = diag.add_info(
                            branch_value.last_assignment_span,
                            format!("value in branch assigned here has type `{}`", ty.value_string(elab)),
                        )
                    }
                }
            }
        }
        if used_parent_value {
            let parent_value = unwrap_branch_value(parent_value);

            match parent_value {
                MaybeUndefined::Undefined => {}
                MaybeUndefined::Defined(parent_value) => {
                    diag = diag.add_info(
                        parent_value.last_assignment_span,
                        format!("value before branch assigned here has type `{}`", ty.value_string(elab)),
                    );
                }
            }
        }
        diags.report(diag.finish())
    })?;

    // create result variable
    let var_ir = match var_info.use_ir_variable {
        Some(var_ir) => var_ir,
        None => {
            let var_ir_info = IrVariableInfo {
                ty: ty.as_ir(refs),
                debug_info_span: var_info.span_decl,
                debug_info_id: var_info.id.str(refs.fixed.source).map(str::to_owned),
            };
            parent_flow.new_ir_variable(var_ir_info)
        }
    };

    // store values into that variable
    let mut domain = ValueDomain::CompileTime;
    let mut domain_cond = ValueDomain::CompileTime;

    let mut build_store = |value: VariableValueRef| {
        let value = unwrap_branch_value(value);

        match value {
            MaybeUndefined::Undefined => {
                // undefined, skip store
                Ok(None)
            }
            MaybeUndefined::Defined(assigned) => {
                let (assigned_domain, assigned_expr) = match &assigned.value {
                    Value::Simple(v) => (
                        v.domain(),
                        v.as_ir_expression_unchecked(refs, large, assigned.last_assignment_span, &ty)?,
                    ),
                    Value::Compound(v) => (
                        v.domain(),
                        v.as_ir_expression_unchecked(refs, large, assigned.last_assignment_span, &ty)?,
                    ),
                    Value::Hardware(v) => (
                        v.value.domain(),
                        v.value
                            .as_ir_expression_unchecked(refs, large, assigned.last_assignment_span, &ty)?,
                    ),
                };

                domain = domain.join(assigned_domain);

                if matches!(assigned_expr, IrExpression::Variable(value_var) if value_var == var_ir) {
                    // copy from variable to itself, skip store
                    Ok(None)
                } else {
                    let store = IrStatement::Assign(IrAssignmentTarget::simple(var_ir.into()), assigned_expr);
                    Ok(Some(Spanned::new(span_merge, store)))
                }
            }
        }
    };

    for branch in &mut *branches {
        if let Some(branch_combined) = branch.common.variables.combined.get(&var.index) {
            assert!(branch_combined.info.is_none());
            let branch_value = branch_combined.value.as_ref().unwrap();
            if let Some(store) = build_store(branch_value.as_ref())? {
                branch.common.statements.push(store);
            }
        }
        domain_cond = domain_cond.join(branch.cond_domain.inner);
    }
    if used_parent_value {
        // re-borrow parent_value
        let parent_value = parent_flow.var_get_maybe(var_spanned)?;
        if let Some(store) = build_store(parent_value)? {
            parent_flow.push_ir_statement(store);
        }
    }
    domain = domain.join(domain_cond);

    // wrap result
    let result_value = HardwareValue {
        ty,
        domain,
        expr: IrExpression::Variable(var_ir),
    };
    let result_version = parent_flow.root.next_version();

    let result_assigned = AssignedValue {
        last_assignment_span: span_merge,
        value: Value::Hardware(HardwareValueWithVersion {
            value: result_value,
            version: result_version,
        }),
    };
    Ok(MaybeAssignedValue::Assigned(MaybeUndefined::Defined(result_assigned)))
}

fn merge_branch_implied_types(
    parent: &mut IndexMap<ValueVersion, ImpliedInfo>,
    branch_implications: &[IndexMap<ValueVersion, ImpliedInfo>],
) {
    // TODO do something more interesting here, this placeholder implementation is correct but a missed opportunity
    // TODO we might need to merge this with the actual values, all in merge_branch_values
    let _ = (parent, branch_implications);
}

/// The borrow checking has trouble with nested chains of parent references,
/// but they're not actually unsafe in the way we want to use them.
// Supress false positive https://github.com/rust-lang/rust-clippy/issues/12860.
#[allow(clippy::unnecessary_cast)]
mod lifetime_cast {
    use crate::front::flow::{FlowCompile, FlowHardware, FlowHardwareBranch, FlowHardwareRoot};

    pub unsafe fn compile_ref<'s>(flow: &'s FlowCompile) -> &'s FlowCompile<'s> {
        unsafe { &*(flow as *const FlowCompile<'_> as *const FlowCompile<'s>) }
    }
    pub unsafe fn compile_mut<'s>(flow: &'s mut FlowCompile) -> &'s mut FlowCompile<'s> {
        unsafe { &mut *(flow as *mut FlowCompile<'_> as *mut FlowCompile<'s>) }
    }
    pub unsafe fn hardware_mut<'s>(flow: &'s mut FlowHardware) -> &'s mut FlowHardware<'s> {
        unsafe { &mut *(flow as *mut FlowHardware<'_> as *mut FlowHardware<'s>) }
    }
    pub unsafe fn hardware_root_mut<'s>(flow: &'s mut FlowHardwareRoot) -> &'s mut FlowHardwareRoot<'s> {
        unsafe { &mut *(flow as *mut FlowHardwareRoot<'_> as *mut FlowHardwareRoot<'s>) }
    }
    pub unsafe fn hardware_branch_mut<'s>(flow: &'s mut FlowHardwareBranch) -> &'s mut FlowHardwareBranch<'s> {
        unsafe { &mut *(flow as *mut FlowHardwareBranch<'_> as *mut FlowHardwareBranch<'s>) }
    }
}
