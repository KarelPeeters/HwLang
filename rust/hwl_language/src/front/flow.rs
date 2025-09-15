use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagError, DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::{DomainSignal, ValueDomain};
use crate::front::implication::{
    ClosedIncRangeMulti, HardwareValueWithVersion, Implication, ValueWithVersion, join_implications,
};
use crate::front::module::ExtraRegisterInit;
use crate::front::signal::{Port, Register, Signal, SignalOrVariable, Wire};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrExpression, IrExpressionLarge, IrLargeArena, IrRegisters, IrStatement, IrVariable,
    IrVariableInfo, IrVariables, IrWires,
};
use crate::syntax::ast::{MaybeIdentifier, SyncDomain};
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::util::arena::RandomCheck;
use crate::util::data::IndexMapExt;
use crate::util::iter::IterExt;
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

    fn for_each_implication(&self, value: ValueVersion, f: impl FnMut(&Implication));

    fn apply_implications(
        &self,
        large: &mut IrLargeArena,
        value: HardwareValueWithVersion,
    ) -> HardwareValueWithVersion {
        if let HardwareType::Int(ty) = &value.value.ty {
            let mut range = ClosedIncRangeMulti::from_range(ty.clone());
            self.for_each_implication(value.version, |implication| {
                let &Implication { version, op, ref right } = implication;
                assert_eq!(value.version, version);
                range.apply_implication(op, right);
            });

            match range.to_range() {
                // TODO support never type or maybe specifically empty ranges
                // TODO or better, once implications discover there's a contradiction we can stop evaluating the block
                None => value,
                Some(range) => {
                    if &range == ty {
                        value
                    } else {
                        let expr_constr = IrExpressionLarge::ConstrainIntRange(range.clone(), value.value.expr);
                        let value_constr = HardwareValue {
                            ty: HardwareType::Int(range),
                            domain: value.value.domain,
                            expr: large.push_expr(expr_constr),
                        };
                        HardwareValueWithVersion {
                            value: value_constr,
                            version: value.version,
                        }
                    }
                }
            }
        } else {
            value
        }
    }

    fn var_set_maybe(&mut self, var: Variable, assignment_span: Span, value: VariableValue);

    fn var_get_maybe(&self, var: Spanned<Variable>) -> DiagResult<VariableValueRef<'_>>;
}

#[allow(private_bounds)]
pub trait Flow: FlowPrivate {
    fn new_child_compile<'s>(&'s mut self, span: Span, reason: &'static str) -> FlowCompile<'s>;

    // TODO find a better name
    fn check_hardware(&mut self, span: Span, reason: &str) -> DiagResult<&mut FlowHardware<'_>>;

    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>>;

    fn var_new(&mut self, info: VariableInfo) -> Variable;

    fn var_set(&mut self, var: Variable, assignment_span: Span, value: DiagResult<Value>) {
        let assigned = match value {
            Ok(value) => {
                let value_with_version = match value {
                    Value::Compile(value) => Value::Compile(value),
                    Value::Hardware(value) => Value::Hardware(HardwareValueWithVersion {
                        value,
                        version: self.root().next_version(),
                    }),
                };
                MaybeAssignedValue::Assigned(AssignedValue {
                    last_assignment_span: assignment_span,
                    value_with_version,
                })
            }
            Err(e) => MaybeAssignedValue::Error(e),
        };
        self.var_set_maybe(var, assignment_span, assigned);
    }

    fn var_info(&self, var: Spanned<Variable>) -> DiagResult<&VariableInfo>;

    fn var_eval(&self, ctx: &mut CompileItemContext, var: Spanned<Variable>) -> DiagResult<ValueWithVersion> {
        let diags = ctx.refs.diags;

        match self.var_get_maybe(var)? {
            MaybeAssignedValue::Assigned(value) => Ok(value.value_with_version.clone().map_hardware(|v| {
                let v = v.map_version(|index| ValueVersion {
                    signal: SignalOrVariable::Variable(var.inner),
                    index,
                });
                self.apply_implications(&mut ctx.large, v)
            })),
            MaybeAssignedValue::NotYetAssigned => {
                let var_info = self.var_info(var)?;
                let diag = Diagnostic::new("variable has not yet been assigned a value")
                    .add_error(var.span, "variable used here")
                    .add_info(var_info.id.span(), "variable declared here")
                    .finish();
                Err(diags.report(diag))
            }
            MaybeAssignedValue::PartiallyAssigned => {
                let var_info = self.var_info(var)?;
                let diag = Diagnostic::new("variable has not yet been assigned a value in all preceding branches")
                    .add_error(var.span, "variable used here")
                    .add_info(var_info.id.span(), "variable declared here")
                    .finish();
                Err(diags.report(diag))
            }
            MaybeAssignedValue::Error(e) => Err(e),
        }
    }

    fn var_new_immutable_init(&mut self, id: MaybeIdentifier, assign_span: Span, value: DiagResult<Value>) -> Variable {
        let info = VariableInfo {
            id,
            mutable: false,
            ty: None,
        };
        let var = self.var_new(info);
        self.var_set(var, assign_span, value);
        var
    }

    fn var_capture(&self, var: Spanned<Variable>) -> DiagResult<CapturedValue> {
        match self.var_get_maybe(var)? {
            MaybeAssignedValue::Assigned(value) => match &value.value_with_version {
                Value::Compile(v) => Ok(CapturedValue::Value(v.clone())),
                Value::Hardware(_) => Ok(CapturedValue::FailedCapture(FailedCaptureReason::Hardware)),
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

    fn for_each_implication(&self, value: ValueVersion, f: impl FnMut(&Implication)) {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowCompileKind::Root => break,
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    parent.for_each_implication(value, f);
                    break;
                }
            }
        }
    }

    fn var_set_maybe(&mut self, var: Variable, assignment_span: Span, value: VariableValue) {
        // checks
        assert_eq!(self.root.check, var.check);
        let diags = self.root.diags;

        let mut value = value;
        if let MaybeAssignedValue::Assigned(assigned) = &value {
            if let Value::Hardware(_) = assigned.value_with_version {
                let e = diags.report_internal_error(assignment_span, "cannot assign hardware value in compile context");
                value = MaybeAssignedValue::Error(e);
            }
        }

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
        let diag = Diagnostic::new(format!("{reason} is only allowed in a hardware context"))
            .add_error(span, format!("{reason} here"))
            .add_info(
                self.compile_span,
                format!("context is compile-time because of this {}", self.compile_reason),
            )
            .finish();
        Err(self.root.diags.report(diag))
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

    fn var_info(&self, var: Spanned<Variable>) -> DiagResult<&VariableInfo> {
        let mut curr = self;
        loop {
            if let Some(slot) = curr.variable_slots.get(&var.inner.index) {
                return Ok(&slot.info);
            }
            curr = match &curr.kind {
                FlowCompileKind::Root => {
                    let diags = self.root.diags;
                    return Err(diags.report_internal_error(var.span, "failed to find variable info"));
                }
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => return parent.var_info(var),
            }
        }
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
}

pub struct FlowHardwareRoot<'p> {
    root: &'p FlowRoot<'p>,
    parent: &'p FlowCompile<'p>,
    process_kind: HardwareProcessKind<'p>,

    ir_wires: &'p mut IrWires,
    ir_registers: &'p mut IrRegisters,
    ir_variables: IrVariables,

    common: FlowHardwareCommon,
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
    // TODO store these as a map(version) -> type instead of this, so pre-applied
    //   this should be faster and allow us to more easily cut dead branches
    implications: Vec<Implication>,
}

struct FlowHardwareVariables {
    // We store both the info and the value as a signle combined entry, to avoid duplicate map lookups and insertions.
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
    fn new(implications: Vec<Implication>) -> FlowHardwareCommon {
        FlowHardwareCommon {
            variables: FlowHardwareVariables {
                combined: IndexMap::new(),
            },
            signal_versions: IndexMap::new(),
            statements: vec![],
            implications,
        }
    }
}

impl FlowPrivate for FlowHardware<'_> {
    fn root(&self) -> &FlowRoot<'_> {
        self.root
    }

    fn for_each_implication(&self, value: ValueVersion, mut f: impl FnMut(&Implication)) {
        let mut visit = |common: &FlowHardwareCommon| {
            common
                .implications
                .iter()
                .filter(|imp| imp.version == value)
                .for_each(&mut f);
        };

        // TODO extract self.for_each_common
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => {
                    visit(&root.common);
                    break;
                }
                FlowHardwareKind::Branch(branch) => {
                    visit(&branch.common);
                    branch.parent
                }
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
            };
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
            if let Some(combined) = variables.combined.get(&var.inner.index) {
                if let Some(value) = &combined.value {
                    return Ok(value.as_ref());
                }
            }
            curr = match next {
                Either::Left(root) => return root.parent.var_get_maybe(var),
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

    fn check_hardware<'s>(&'s mut self, _: Span, _: &str) -> DiagResult<&'s mut FlowHardware<'s>> {
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

    fn var_info(&self, var: Spanned<Variable>) -> DiagResult<&VariableInfo> {
        // TODO fix code duplication
        let mut curr = self;
        loop {
            let (variables, next) = match &curr.kind {
                FlowHardwareKind::Root(root) => (&root.common.variables, Either::Left(root)),
                FlowHardwareKind::Branch(branch) => (&branch.common.variables, Either::Right(&branch.parent)),
                FlowHardwareKind::Scoped(scoped) => (&scoped.variables, Either::Right(&scoped.parent)),
            };
            if let Some(info) = variables.combined.get(&var.inner.index) {
                if let Some(info) = &info.info {
                    return Ok(info);
                }
            }
            curr = match next {
                Either::Left(root) => return root.parent.var_info(var),
                Either::Right(next) => next,
            };
        }
    }
}

impl<'p> FlowHardwareRoot<'p> {
    pub fn new(
        parent: &'p FlowCompile,
        kind: HardwareProcessKind<'p>,
        ir_wires: &'p mut IrWires,
        ir_registers: &'p mut IrRegisters,
    ) -> FlowHardwareRoot<'p> {
        let parent = unsafe { lifetime_cast::compile_ref(parent) };
        FlowHardwareRoot {
            root: parent.root,
            parent,
            process_kind: kind,
            ir_wires,
            ir_registers,
            ir_variables: IrVariables::new(),
            common: FlowHardwareCommon::new(vec![]),
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
            process_kind: _,
            ir_wires: _,
            ir_registers: _,
            ir_variables,
            common,
        } = self;
        let FlowHardwareCommon {
            variables: _,
            signal_versions: _,
            statements: ir_statements,
            implications: _,
        } = common;

        let block = IrBlock {
            statements: ir_statements,
        };
        (ir_variables, block)
    }
}

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
        cond_domain: Spanned<ValueDomain>,
        cond_implications: Vec<Implication>,
    ) -> FlowHardwareBranch<'_> {
        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        FlowHardwareBranch {
            root,
            parent: slf,
            cond_domain,
            common: FlowHardwareCommon::new(cond_implications),
        }
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

    pub fn join_child_branches(
        &mut self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span_merge: Span,
        mut branches: Vec<FlowHardwareBranchContent>,
    ) -> DiagResult<Vec<IrBlock>> {
        // TODO merge implications too
        // TODO do something else if childen are empty, eg. push an instruction for `assert(false)`

        // collect the interesting vars and signals
        // things we can skip:
        //   * items that are not in any child, they didn't change
        //   * items that don't exist in the parent, they won't be alive afterwards
        let mut merged_vars = IndexSet::new();
        let mut merged_signals = IndexSet::new();
        for branch in &branches {
            assert_eq!(self.root.check, branch.check);
            for &var in branch.common.variables.combined.keys() {
                if branch.common.variables.combined.contains_key(&var) {
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
            let var_id_span = self.var_info(Spanned::new(span_merge, var))?.id.span();

            let value_merged = merge_branch_values(refs, large, self, span_merge, var, &mut branches)
                .unwrap_or_else(VariableValue::Error);
            self.var_set_maybe(var, var_id_span, value_merged);
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
        let mut branch_implications = vec![];
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
                    implications,
                } = common;
                branch_implications.push(implications);
                IrBlock { statements }
            })
            .collect_vec();

        let merged_implications = join_implications(&branch_implications);
        self.first_common_mut().implications.extend(merged_implications);

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

    pub fn signal_eval(
        &self,
        ctx: &mut CompileItemContext,
        signal: Spanned<Signal>,
    ) -> DiagResult<HardwareValueWithVersion> {
        let value_raw = signal.inner.as_hardware_value(ctx, signal.span)?;

        // For clocked blocks, check read domain validness.
        // This is technically not needed for correctness (assignments also check domain validness),
        //   but it makes error messages a bit clearer.
        if self.enable_domain_checks {
            match &self.root_hw().process_kind {
                &HardwareProcessKind::ClockedBlockBody { domain, .. } => {
                    ctx.check_valid_domain_crossing(
                        signal.span,
                        domain.map_inner(ValueDomain::Sync),
                        Spanned::new(signal.span, value_raw.domain),
                        "signal read in clocked block",
                    )?;
                }
                HardwareProcessKind::CombinatorialBlockBody { .. }
                | HardwareProcessKind::WireExpression { .. }
                | HardwareProcessKind::InstancePortConnection { .. } => {}
            }
        }

        // wrap and apply implications
        let version = ValueVersion {
            signal: SignalOrVariable::Signal(signal.inner),
            index: self.signal_get_version(signal.inner),
        };
        let value = HardwareValueWithVersion {
            value: value_raw,
            version,
        };
        Ok(self.apply_implications(&mut ctx.large, value))
    }

    fn signal_set_version(&mut self, signal: Signal, version: ValueVersionIndex) {
        self.first_common_mut().signal_versions.insert(signal, version);
    }

    pub fn signal_assign(&mut self, signal: Spanned<Signal>, full: bool) {
        // TODO use this to check for accidental latches
        // TODO record full/partial write for combinatorial block driver checking
        // TODO for combinatorial block codegen, can we rely on the verilog tools to be smart enough
        //   or should we initialize fully driven signals anyway?
        let _ = full;

        // bump version
        let version = self.root.next_version();
        self.signal_set_version(signal.inner, version);
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
    pub id: MaybeIdentifier,
    pub mutable: bool,
    pub ty: Option<Spanned<Type>>,
}

type VariableValue =
    MaybeAssignedValue<AssignedValue<Value<CompileValue, HardwareValueWithVersion<ValueVersionIndex>>>>;
type VariableValueRef<'a> =
    MaybeAssignedValue<&'a AssignedValue<Value<CompileValue, HardwareValueWithVersion<ValueVersionIndex>>>>;

#[derive(Debug, Copy, Clone)]
pub enum MaybeAssignedValue<A> {
    Assigned(A),
    NotYetAssigned,
    PartiallyAssigned,
    Error(DiagError),
}

impl<A> MaybeAssignedValue<A> {
    pub fn as_ref(&self) -> MaybeAssignedValue<&A> {
        match self {
            MaybeAssignedValue::Assigned(v) => MaybeAssignedValue::Assigned(v),
            MaybeAssignedValue::NotYetAssigned => MaybeAssignedValue::NotYetAssigned,
            MaybeAssignedValue::PartiallyAssigned => MaybeAssignedValue::PartiallyAssigned,
            &MaybeAssignedValue::Error(e) => MaybeAssignedValue::Error(e),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssignedValue<V> {
    pub last_assignment_span: Span,
    pub value_with_version: V,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ValueVersion {
    // TODO we could also _only_ use an index and ensure they're unique per signal, but that's sketchy to guarantee
    signal: SignalOrVariable,
    index: ValueVersionIndex,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
    Hardware,
    NotFullyInitialized,
}

fn merge_branch_values(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    parent_flow: &mut FlowHardware,
    span_merge: Span,
    var: Variable,
    branches: &mut [FlowHardwareBranchContent],
) -> DiagResult<VariableValue> {
    let diags = refs.diags;

    let var_spanned = Spanned::new(span_merge, var);
    let var_info = parent_flow.var_info(var_spanned)?;
    let parent_value = parent_flow.var_get_maybe(var_spanned)?;

    // visit all branch values to check if we can skip the merge
    let mut merged: Option<MaybeAssignedValue<Result<AssignedValue<_>, NeedsHardwareMerge>>> = None;
    let mut used_value_parent = false;

    for branch in &mut *branches {
        // TODO apply implications here too
        let branch_value = match branch.common.variables.combined.get(&var.index) {
            Some(branch_combined) => {
                assert!(branch_combined.info.is_none());
                branch_combined.value.as_ref().unwrap().as_ref()
            }
            None => {
                used_value_parent = true;
                parent_value
            }
        };

        // first branch, just set it
        let Some(merged_value) = merged else {
            merged = Some(match branch_value {
                MaybeAssignedValue::Assigned(v) => MaybeAssignedValue::Assigned(Ok(v.clone())),
                MaybeAssignedValue::NotYetAssigned => MaybeAssignedValue::NotYetAssigned,
                MaybeAssignedValue::PartiallyAssigned => MaybeAssignedValue::PartiallyAssigned,
                MaybeAssignedValue::Error(e) => MaybeAssignedValue::Error(e),
            });
            continue;
        };

        // check if we need to do a merge
        //   we don't stop once we know we need to merge, because later unassigned values might remove that requirement again
        let merged_new = match (merged_value, branch_value) {
            // once we need a hardware merge and we get more normal values, we will steeds need a hardware merge
            (MaybeAssignedValue::Assigned(Err(e)), MaybeAssignedValue::Assigned(_)) => {
                MaybeAssignedValue::Assigned(Err(e))
            }
            // actual merge check
            (MaybeAssignedValue::Assigned(Ok(merged_value)), MaybeAssignedValue::Assigned(child_value)) => {
                let same_value_and_version = match (&merged_value.value_with_version, &child_value.value_with_version) {
                    (Value::Compile(merged_value), Value::Compile(child_value)) => merged_value == child_value,
                    (Value::Hardware(merged_value), Value::Hardware(child_value)) => {
                        merged_value.version == child_value.version
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
                        value_with_version: merged_value.value_with_version,
                    }))
                } else {
                    MaybeAssignedValue::Assigned(Err(NeedsHardwareMerge))
                }
            }
            // unassigned and error cases, these are in some sense nice because we can avoid doing the merge
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
            (MaybeAssignedValue::Error(e), _) | (_, MaybeAssignedValue::Error(e)) => MaybeAssignedValue::Error(e),
        };
        merged = Some(merged_new);
    }

    // check if we're done
    let merged = match merged {
        None => {
            // There were no branches, this implies that control flow diverges.
            // It doesn't really matter what value we use, so we'll just use an unassigned one.
            return Ok(MaybeAssignedValue::NotYetAssigned);
        }
        Some(MaybeAssignedValue::Assigned(Ok(value))) => Ok(MaybeAssignedValue::Assigned(value)),
        Some(MaybeAssignedValue::NotYetAssigned) => Ok(MaybeAssignedValue::NotYetAssigned),
        Some(MaybeAssignedValue::PartiallyAssigned) => Ok(MaybeAssignedValue::PartiallyAssigned),
        Some(MaybeAssignedValue::Error(e)) => Ok(MaybeAssignedValue::Error(e)),
        Some(MaybeAssignedValue::Assigned(Err(NeedsHardwareMerge))) => Err(NeedsHardwareMerge),
    };
    match merged {
        Ok(merged) => return Ok(merged),
        Err(NeedsHardwareMerge) => {}
    }

    // at this point we know that there are no un-assigned branches, and that not all branches have the same value:
    //   this means we actually need to do a hardware merge
    fn constrain_lifetime<B, F: for<'a> Fn(VariableValueRef<'a>) -> &'a B>(f: F) -> F {
        f
    }
    let unwrap_branch_value =
        constrain_lifetime(|v: VariableValueRef| unwrap_match!(v, MaybeAssignedValue::Assigned(value) => value));

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

            match &branch_value.value_with_version {
                Value::Compile(v) => {
                    let ty = v.ty();
                    ty.as_hardware_type(refs).map_err(|_| {
                        let ty_str = ty.diagnostic_string();
                        let diag = Diagnostic::new("merging if assignments needs hardware type")
                            .add_info(var_info.id.span(), "for this variable")
                            .add_info(
                                branch_value.last_assignment_span,
                                format!(
                                    "value assigned here has type `{ty_str}` which cannot be represented in hardware"
                                ),
                            )
                            .add_error(span_merge, "merging happens here")
                            .finish();
                        diags.report(diag)
                    })
                }
                Value::Hardware(v) => Ok(v.value.ty.clone()),
            }
        })
        .try_collect_all_vec()?;

    // find common type
    let ty = branch_tys
        .iter()
        .fold(Type::Undefined, |a, t| a.union(&t.as_type(), false));

    // convert common to hardware too
    let ty = ty.as_hardware_type(refs).map_err(|_| {
        let ty_str = ty.diagnostic_string();

        let mut diag = Diagnostic::new("merging if assignments needs hardware type")
            .add_info(var_info.id.span(), "for this variable")
            .add_error(
                span_merge,
                format!("merging happens here, combined type `{ty_str}` cannot be represented in hardware"),
            );

        for (branch, ty) in zip_eq(&*branches, branch_tys) {
            if let Some(branch_combined) = branch.common.variables.combined.get(&var.index) {
                assert!(branch_combined.info.is_none());
                let branch_combined = branch_combined.value.as_ref().unwrap().as_ref();

                let branch_value = unwrap_branch_value(branch_combined);
                diag = diag.add_info(
                    branch_value.last_assignment_span,
                    format!("value in branch assigned here has type `{}`", ty.diagnostic_string()),
                )
            }
        }
        if used_value_parent {
            let value_parent = unwrap_branch_value(parent_value);
            diag = diag.add_info(
                value_parent.last_assignment_span,
                format!(
                    "value before branch assigned here has type `{}`",
                    ty.diagnostic_string()
                ),
            );
        }
        diags.report(diag.finish())
    })?;

    // create result variable
    let var_ir_info = IrVariableInfo {
        ty: ty.as_ir(refs),
        debug_info_id: var_info.id.spanned_string(refs.fixed.source),
    };
    let var_ir = parent_flow.new_ir_variable(var_ir_info);

    // store values into that variable
    let mut domain = ValueDomain::CompileTime;
    let mut domain_cond = ValueDomain::CompileTime;

    let mut build_store = |value: VariableValueRef| {
        let value = unwrap_branch_value(value);
        let value = match &value.value_with_version {
            Value::Compile(v) => v.as_hardware_value(refs, large, value.last_assignment_span, &ty)?,
            Value::Hardware(v) => v.value.clone(),
        };

        domain = domain.join(value.domain);

        let store = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), value.expr);
        Ok(Spanned::new(span_merge, store))
    };

    for branch in &mut *branches {
        if let Some(branch_combined) = branch.common.variables.combined.get(&var.index) {
            assert!(branch_combined.info.is_none());
            let branch_value = branch_combined.value.as_ref().unwrap();
            branch.common.statements.push(build_store(branch_value.as_ref())?);
        }
        domain_cond = domain_cond.join(branch.cond_domain.inner);
    }
    if used_value_parent {
        // re-borrow parent_value
        let parent_value = parent_flow.var_get_maybe(var_spanned)?;
        parent_flow.push_ir_statement(build_store(parent_value)?);
    }
    domain = domain.join(domain_cond);

    // wrap result
    let result_value = HardwareValue {
        ty,
        domain,
        expr: IrExpression::Variable(var_ir),
    };
    let result_version = parent_flow.root.next_version();

    Ok(MaybeAssignedValue::Assigned(AssignedValue {
        last_assignment_span: span_merge,
        value_with_version: Value::Hardware(HardwareValueWithVersion {
            value: result_value,
            version: result_version,
        }),
    }))
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
