use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagError, DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::{DomainSignal, ValueDomain};
use crate::front::implication::{HardwareValueWithVersion, Implication, ImplicationKind, ValueWithVersion};
use crate::front::module::ExtraRegisterInit;
use crate::front::signal::{Port, Register, Signal, SignalOrVariable, Wire};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, HardwareValue, MaybeCompile, MaybeUndefined, MixedCompoundValue, NotCompile,
    SimpleCompileValue, Value, ValueCommon,
};
use crate::mid::ir::{
    IrAssignmentTarget, IrBlock, IrExpression, IrExpressionLarge, IrLargeArena, IrRegisters, IrStatement, IrVariable,
    IrVariableInfo, IrVariables, IrWires,
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
use crate::util::range_multi::{AnyMultiRange, ClosedMultiRange, ClosedNonEmptyMultiRange};
use crate::util::{NON_ZERO_USIZE_ONE, NON_ZERO_USIZE_TWO, ResultExt};
use indexmap::{IndexMap, IndexSet};
use itertools::{Itertools, zip_eq};
use std::cell::Cell;
use std::num::NonZeroUsize;
use unwrap_match::unwrap_match;

pub enum FlowKind<C, H> {
    Compile(C),
    Hardware(H),
}

#[derive(Debug)]
pub struct FlowRoot<'d> {
    // TODO maybe this should be refs, instead having an extra parameter for a couple of functions
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

pub struct FlowCompile<'p> {
    root: &'p FlowRoot<'p>,

    compile_span: Span,
    compile_reason: &'static str,

    parent: FlowCompileKind<'p>,
    variables: IndexMap<VariableIndex, VariableSlot>,
}

pub enum FlowCompileKind<'p> {
    /// Top level flow, created for each elaborated item.
    Root,
    /// A scoped block within another compile parent.
    /// Cannot modify parent variables.
    IsolatedCompile(&'p FlowCompile<'p>),
    /// A scoped block within another compile parent.
    /// Can modify parent variables,
    ///   but also declare new variables locally which are then dropped at the end of the scope.
    ScopedCompile(&'p mut FlowCompile<'p>),
    /// A compile flow within a hardware parent.
    ///   Can still modify parent variables (by setting them to constant variables),
    ///   and also declare new variables locally which are then dropped at the end of the scope.
    ScopedHardware(&'p mut FlowHardware<'p>),
}

#[derive(Debug)]
pub struct FlowCompileContent {
    check: RandomCheck,

    compile_span: Span,
    compile_reason: &'static str,

    variables: IndexMap<VariableIndex, VariableSlot>,
}

pub struct FlowHardware<'p> {
    root: &'p FlowRoot<'p>,
    enable_domain_checks: bool,
    kind: FlowHardwareKind<'p>,
}

enum FlowHardwareKind<'p> {
    /// Top level hardware flow, created for each process.
    /// Cannot modify parent variables.
    Root(&'p mut FlowHardwareRoot<'p>),
    /// A conditional branch within a hardware flow.
    /// Wires to signals and variables should not propagate to the parent immediately,
    ///   they will be merged when all branches have ended.
    Branch(&'p mut FlowHardwareBranch<'p>),
    /// A scoped block within a hardware parent.
    /// Can modify parent variables,
    ///   and also declare new variables locally which are then dropped at the end of the scope
    Scoped(FlowHardwareScoped<'p>),
}

pub struct FlowHardwareRoot<'p> {
    root: &'p FlowRoot<'p>,
    parent: &'p FlowCompile<'p>,

    #[allow(dead_code)]
    span: Span,
    process_kind: HardwareProcessKind<'p>,

    ir_wires: &'p mut IrWires,
    ir_registers: &'p mut IrRegisters,
    ir_variables: IrVariables,

    common: FlowHardwareCommon,
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

pub struct FlowHardwareBranch<'p> {
    root: &'p FlowRoot<'p>,
    parent: &'p mut FlowHardware<'p>,

    cond_domain: Spanned<ValueDomain>,
    common: FlowHardwareCommon,
}

pub struct FlowHardwareBranchContent {
    check: RandomCheck,
    // TODO think about how this should be propagated, eg. if two branches break and the third one doesn't,
    //   the value of all merged values and even implications might depend on the condition domain
    cond_domain: Spanned<ValueDomain>,
    common: FlowHardwareCommon,
}

struct FlowHardwareScoped<'p> {
    parent: &'p mut FlowHardware<'p>,
    variables: IndexMap<VariableIndex, VariableSlot>,
}

struct FlowHardwareCommon {
    // TODO document why single map
    variables: IndexMap<VariableIndex, VariableSlotOption>,
    signals: IndexMap<Signal, SignalContent>,
    statements: Vec<Spanned<IrStatement>>,
}

#[derive(Debug)]
struct VariableSlot {
    info: VariableInfo,
    content: VariableContent,
}

#[derive(Default)]
struct VariableSlotOption {
    info: Option<VariableInfo>,
    content: Option<VariableContent>,
}

#[derive(Debug, Copy, Clone)]
pub struct ImplicationContradiction;

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

    /// If set, branch joining will always use this IR variable as the result.
    /// This only works if all values are guaranteed to have the exact right type for this variable,
    ///   and nothing else will use this variable at any point.
    pub join_ir_variable: Option<IrVariable>,
}

#[derive(Debug)]
pub enum VariableId {
    Id(MaybeIdentifier),
    Custom(&'static str),
}

#[derive(Debug, Clone)]
enum VariableContent {
    Assigned(Spanned<VariableValue>),
    NotFullyAssigned(VariableNotFullyAssigned),
    Error(DiagError),
}

#[derive(Debug, Copy, Clone)]
enum VariableNotFullyAssigned {
    NotYetAssigned,
    PartiallyAssigned,

    /// Undefined means no value has been assigned yet, but merging should act like there was an assigned value.
    /// This is useful when there is some external mechanism to enforce that by the time the variable is read it will
    /// actually have a value, but Flow does not understand that.
    /// This is used for function return values.
    Undefined,
}

type VariableValue = Value<SimpleCompileValue, MixedCompoundValue, VariableValueHardware>;

#[derive(Debug, Clone)]
struct VariableValueHardware {
    value_raw: HardwareValue<HardwareType, IrVariable>,
    version: VersionIndex,
    implied_int_range: Option<ClosedNonEmptyMultiRange<BigInt>>,
}

pub type VarSetValue = Value<SimpleCompileValue, MixedCompoundValue, HardwareValue<HardwareType, IrVariable>>;

enum SignalContent {
    Compile(CompileValue),
    Hardware(SignalContentHardware),
}

#[derive(Debug, Clone)]
struct SignalContentHardware {
    version: VersionIndex,
    implied_int_range: Option<ClosedNonEmptyMultiRange<BigInt>>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ValueVersion {
    signal: SignalOrVariable,
    index: VersionIndex,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct VersionIndex(NonZeroUsize);

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

trait FlowPrivate: Sized {
    fn root(&self) -> &FlowRoot<'_>;

    fn var_info_option(&self, var: VariableIndex) -> Option<&VariableInfo>;
    fn var_set_content(&mut self, var: Variable, assignment_span: Span, content: VariableContent) -> DiagResult;
    fn var_get_content(&self, var: Spanned<Variable>) -> DiagResult<&VariableContent>;

    fn signal_get_content(&self, signal: Signal) -> DiagResult<&SignalContent>;
}

const VAR_EVAL_HW_REASON: &str = "accessing a hardware variable";

#[allow(private_bounds)]
pub trait Flow: FlowPrivate {
    #[allow(clippy::needless_lifetimes)]
    fn new_child_compile<'s>(&'s mut self, span: Span, reason: &'static str) -> FlowCompile<'s>;

    fn require_hardware(&mut self, span: Span, reason: &str) -> DiagResult<&mut FlowHardware<'_>>;

    fn kind(&self) -> FlowKind<&FlowCompile<'_>, &FlowHardware<'_>>;
    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>>;

    fn var_new(&mut self, info: VariableInfo) -> Variable;

    fn var_info(&self, var: Spanned<Variable>) -> DiagResult<&VariableInfo> {
        assert_eq!(var.inner.check, self.root().check);
        self.var_info_option(var.inner.index).ok_or_else(|| {
            self.root()
                .diags
                .report_internal_error(var.span, "failed to find variable")
        })
    }

    fn var_set(
        &mut self,
        refs: CompileRefs,
        var: Variable,
        assignment_span: Span,
        value: DiagResult<Value>,
    ) -> DiagResult {
        // store value in IrVariable if it's a hardware value
        let value = value.and_then(|value| {
            value.try_map_hardware(|value| {
                let var_info = self.var_info(Spanned::new(assignment_span, var))?;
                let debug_info_id = var_info.id.as_str(refs.fixed.source).map(str::to_owned);

                let flow = self.require_hardware(assignment_span, "assigning hardware value")?;

                Ok(flow.store_hardware_value_in_new_ir_variable(refs, assignment_span, debug_info_id, value))
            })
        });

        self.var_set_without_copy(var, assignment_span, value)
    }

    /// More specific form of [var_set] that only works for compile-time value.
    fn var_set_compile(&mut self, var: Variable, assignment_span: Span, value: DiagResult<CompileValue>) -> DiagResult {
        self.var_set_without_copy(var, assignment_span, value.map(VarSetValue::from))
    }

    /// More specific form of [var_set] if the value is already in an IR variable,
    /// and transferring the ownership of that variable to the flow is desired.
    fn var_set_without_copy(
        &mut self,
        var: Variable,
        assignment_span: Span,
        value: DiagResult<VarSetValue>,
    ) -> DiagResult {
        // TODO stop accepting DiagError here, if something fails we should stop immediately,
        //   there are too many much type inference and side effects that can cause secondary issues
        // TODO double-check that we're not setting a hardware value in a compile context, that should be impossible
        let assigned = match value {
            Ok(value) => {
                let value = value.map_hardware(|value| VariableValueHardware {
                    value_raw: value,
                    version: self.root().next_version(),
                    implied_int_range: None,
                });
                VariableContent::Assigned(Spanned::new(assignment_span, value))
            }
            Err(e) => VariableContent::Error(e),
        };
        self.var_set_content(var, assignment_span, assigned)
    }

    fn var_set_undefined(&mut self, var: Variable, span: Span) -> DiagResult {
        let content = VariableContent::NotFullyAssigned(VariableNotFullyAssigned::Undefined);
        self.var_set_content(var, span, content)
    }

    /// Evaluate the given variable.
    ///
    /// The [IrVariable] in the hardware case is already a defensive copy, and can be used by the caller freely.
    /// The flow will not modify it later.
    fn var_eval(
        &mut self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        var: Spanned<Variable>,
    ) -> DiagResult<ValueWithVersion> {
        self.var_eval_without_copy(large, var)?
            .try_map_hardware(|value_uncopied| {
                // store into intermediate variable (copy-on-read)
                let var_info = self.var_info(var)?;
                let debug_info_id = var_info.id.as_str(refs.fixed.source).map(str::to_owned);
                let flow = self.require_hardware(var.span, VAR_EVAL_HW_REASON)?;

                let value =
                    flow.store_hardware_value_in_new_ir_variable(refs, var.span, debug_info_id, value_uncopied.value);

                Ok(HardwareValueWithVersion {
                    value: value.map_expression(IrExpression::Variable),
                    version: value_uncopied.version,
                })
            })
    }

    /// Variant of [var_eval] that does not create a defensive copy of the hardware value.
    fn var_eval_without_copy(
        &mut self,
        large: &mut IrLargeArena,
        var: Spanned<Variable>,
    ) -> DiagResult<ValueWithVersion> {
        let diags = self.root().diags;

        let result = match self.var_get_content(var)? {
            VariableContent::Assigned(value) => match &value.inner {
                Value::Simple(v) => Value::Simple(v.clone()),
                Value::Compound(v) => Value::Compound(v.clone()),
                Value::Hardware(v) => Value::Hardware(v.as_hardware_value_without_copy(large, var.inner)),
            },
            VariableContent::NotFullyAssigned(kind) => {
                let var_info = self.var_info(var)?;
                return Err(kind.report_diag(diags, var.span, var_info));
            }
            &VariableContent::Error(e) => return Err(e),
        };

        // check that we're not returning a hardware value in a compile context
        //   (this check also catches compound values with hardware parts)
        match self.kind() {
            FlowKind::Hardware(_) => {}
            FlowKind::Compile(slf) => {
                if CompileValue::try_from(&result).is_err() {
                    return Err(slf.err_not_hardware(var.span, VAR_EVAL_HW_REASON));
                }
            }
        }

        Ok(result)
    }

    /// Get the type of the given value without actually evaluating it.
    /// This does not require that the value can be evaluated in the current flow,
    ///   for example asking the type of a hardware value in a compile-time flow is allowed.
    /// This takes into account any active implications.
    fn type_of(&self, ctx: &mut CompileItemContext, value: Spanned<SignalOrVariable>) -> DiagResult<Type> {
        match value.inner {
            SignalOrVariable::Signal(signal) => match self.signal_get_content(signal)? {
                SignalContent::Compile(value) => Ok(value.ty()),
                SignalContent::Hardware(signal_hw) => signal_hw
                    .ty_hw(ctx, signal, value.span)
                    .as_ref_ok()
                    .map(HardwareType::as_type),
            },
            SignalOrVariable::Variable(var) => {
                let content = self.var_get_content(Spanned::new(value.span, var))?;

                match content {
                    VariableContent::Assigned(value) => Ok(value.inner.ty()),
                    VariableContent::NotFullyAssigned(kind) => {
                        let var_info = self.var_info(Spanned::new(value.span, var))?;
                        Err(kind.report_diag(ctx.refs.diags, value.span, var_info))
                    }
                    &VariableContent::Error(e) => Err(e),
                }
            }
        }
    }

    fn var_new_immutable_init(
        &mut self,
        refs: CompileRefs,
        span_decl: Span,
        id: VariableId,
        assign_span: Span,
        value: DiagResult<Value>,
    ) -> DiagResult<Variable> {
        let info = VariableInfo {
            span_decl,
            id,
            mutable: false,
            ty: None,
            join_ir_variable: None,
        };
        let var = self.var_new(info);
        self.var_set(refs, var, assign_span, value)?;
        Ok(var)
    }

    fn var_capture(&self, var: Spanned<Variable>) -> DiagResult<CapturedValue> {
        match self.var_get_content(var)? {
            VariableContent::Assigned(value) => match CompileValue::try_from(&value.inner) {
                Ok(v) => Ok(CapturedValue::Value(v)),
                Err(NotCompile) => Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotCompile)),
            },
            VariableContent::NotFullyAssigned(_) => {
                Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotFullyInitialized))
            }
            &VariableContent::Error(e) => Err(e),
        }
    }

    fn var_is_not_yet_assigned(&self, var: Spanned<Variable>) -> DiagResult<bool> {
        match self.var_get_content(var)? {
            VariableContent::NotFullyAssigned(kind) => match kind {
                VariableNotFullyAssigned::NotYetAssigned => Ok(true),
                VariableNotFullyAssigned::PartiallyAssigned => Ok(false),
                VariableNotFullyAssigned::Undefined => Ok(true),
            },
            VariableContent::Assigned(_) => Ok(false),
            &VariableContent::Error(e) => Err(e),
        }
    }

    fn signal_eval_if_compile(&mut self, signal: Spanned<Signal>) -> DiagResult<Option<&CompileValue>> {
        match self.signal_get_content(signal.inner)? {
            SignalContent::Compile(value) => Ok(Some(value)),
            SignalContent::Hardware(_) => Ok(None),
        }
    }

    fn signal_eval(&mut self, ctx: &mut CompileItemContext, signal: Spanned<Signal>) -> DiagResult<ValueWithVersion> {
        match self.signal_get_content(signal.inner)? {
            SignalContent::Compile(value) => {
                // compile-time evaluation can skips any hardware checking
                Ok(Value::from(value.clone()))
            }
            SignalContent::Hardware(content) => {
                // check that we're in a hardware flow
                let slf = match self.kind() {
                    FlowKind::Hardware(slf) => slf,
                    FlowKind::Compile(slf) => {
                        return Err(slf.err_not_hardware(signal.span, "signal evaluation"));
                    }
                };

                // evaluate as a value
                let value = content.as_hardware_value(ctx, signal.inner, signal.span)?;

                // check that we can access the domain in the current block
                // TODO this probably disables too many things, eg. nested blocks
                if slf.enable_domain_checks {
                    match &slf.root_hw().process_kind {
                        &HardwareProcessKind::ClockedBlockBody { domain, .. } => {
                            ctx.check_valid_domain_crossing(
                                signal.span,
                                domain.map_inner(ValueDomain::Sync),
                                Spanned::new(signal.span, value.value.domain),
                                "signal read in clocked block",
                            )?;
                        }
                        HardwareProcessKind::CombinatorialBlockBody { .. }
                        | HardwareProcessKind::WireExpression { .. }
                        | HardwareProcessKind::InstancePortConnection { .. } => {}
                    }
                }

                // copy the value into a temporary variable to avoid later writes from leaking through
                let slf = match self.kind_mut() {
                    FlowKind::Compile(_) => unreachable!(),
                    FlowKind::Hardware(slf) => slf,
                };

                let debug_info_id = signal.inner.diagnostic_string(ctx).to_owned();
                let value_var = slf.store_hardware_value_in_new_ir_variable(
                    ctx.refs,
                    signal.span,
                    Some(debug_info_id),
                    value.value,
                );
                let value = HardwareValueWithVersion {
                    value: value_var.map_expression(IrExpression::Variable),
                    version: value.version,
                };

                Ok(ValueWithVersion::Hardware(value))
            }
        }
    }
}

/// We start assigning new versions at at 2,
///   so we can always safely use 1 as the first version for values that have not yet been assigned.
const VERSION_INITIAL: VersionIndex = VersionIndex(NON_ZERO_USIZE_ONE);

impl FlowRoot<'_> {
    pub fn new(diags: &Diagnostics) -> FlowRoot<'_> {
        FlowRoot {
            diags,
            check: RandomCheck::new(),
            next_var_index: Cell::new(NON_ZERO_USIZE_ONE),
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

    fn next_version(&self) -> VersionIndex {
        let version = self.next_version.get();
        self.next_version.set(version.checked_add(1).expect("overflow"));
        VersionIndex(version)
    }
}

impl FlowPrivate for FlowCompile<'_> {
    fn root(&self) -> &FlowRoot<'_> {
        self.root
    }

    fn var_info_option(&self, var: VariableIndex) -> Option<&VariableInfo> {
        let mut curr = self;
        loop {
            if let Some(slot) = curr.variables.get(&var) {
                return Some(&slot.info);
            }
            curr = match &curr.parent {
                FlowCompileKind::Root => return None,
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => return parent.var_info_option(var),
            }
        }
    }

    fn var_set_content(&mut self, var: Variable, assignment_span: Span, content: VariableContent) -> DiagResult {
        // checks
        assert_eq!(self.root.check, var.check);
        let diags = self.root.diags;

        // TODO maybe only store unwrapped compile values in compile flows?
        let value = if let VariableContent::Assigned(assigned) = &content
            && CompileValue::try_from(&assigned.inner).is_err()
        {
            let e = diags.report_internal_error(
                assignment_span,
                "cannot assign hardware value to variable in compile context",
            );
            VariableContent::Error(e)
        } else {
            content
        };

        // find which slot to store the value in
        let mut curr = self;
        let slot = loop {
            if let Some(slot) = curr.variables.get_mut(&var.index) {
                break slot;
            }
            curr = match &mut curr.parent {
                FlowCompileKind::Root => {
                    let e = diags.report_internal_error(assignment_span, "hit root before finding variable slot");
                    return Err(e);
                }
                FlowCompileKind::IsolatedCompile(_) => {
                    let e = diags
                        .report_internal_error(assignment_span, "hit isolated parent before finding variable slot");
                    return Err(e);
                }
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    return parent.var_set_content(var, assignment_span, value);
                }
            }
        };

        slot.content = value;
        Ok(())
    }

    fn var_get_content(&self, var: Spanned<Variable>) -> DiagResult<&VariableContent> {
        assert_eq!(self.root.check, var.inner.check);
        let diags = self.root.diags;

        let mut curr = self;
        loop {
            if let Some(entry) = curr.variables.get(&var.inner.index) {
                return Ok(&entry.content);
            }
            curr = match &curr.parent {
                FlowCompileKind::Root => {
                    let e = diags.report_internal_error(var.span, "hit root before finding variable slot");
                    return Ok(VariableContent::err_ref(e));
                }
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    return parent.var_get_content(var);
                }
            }
        }
    }

    fn signal_get_content(&self, signal: Signal) -> DiagResult<&SignalContent> {
        let mut curr = self;
        loop {
            curr = match &curr.parent {
                FlowCompileKind::Root => return Ok(&SignalContent::INITIAL),
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => {
                    return parent.signal_get_content(signal);
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
            parent: FlowCompileKind::ScopedCompile(slf),
            variables: IndexMap::new(),
        }
    }

    fn require_hardware(&mut self, span: Span, reason: &str) -> DiagResult<&mut FlowHardware<'_>> {
        Err(self.err_not_hardware(span, reason))
    }

    fn kind(&self) -> FlowKind<&FlowCompile<'_>, &FlowHardware<'_>> {
        FlowKind::Compile(unsafe { lifetime_cast::compile_ref(self) })
    }

    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>> {
        FlowKind::Compile(unsafe { lifetime_cast::compile_mut(self) })
    }

    fn var_new(&mut self, info: VariableInfo) -> Variable {
        let var = self.root.next_variable();

        let slot = VariableSlot {
            info,
            content: VariableContent::NotFullyAssigned(VariableNotFullyAssigned::NotYetAssigned),
        };
        self.variables.insert_first(var.index, slot);

        var
    }

    fn signal_eval(&mut self, _: &mut CompileItemContext, signal: Spanned<Signal>) -> DiagResult<ValueWithVersion> {
        // This is a compile-time flow, so normally signals can't be evaluated.
        // They might have an implied compile-time value through, so check that first.

        // walk up until we hit a hardware parent
        let mut curr = &*self;
        let parent_hw = loop {
            curr = match &curr.parent {
                FlowCompileKind::Root => break None,
                FlowCompileKind::IsolatedCompile(parent) => parent,
                FlowCompileKind::ScopedCompile(parent) => parent,
                FlowCompileKind::ScopedHardware(parent) => break Some(parent),
            }
        };

        // walk up through the chain of hardware parents
        let mut curr = parent_hw;
        loop {
            curr = match curr {
                None => break,
                Some(curr) => {
                    let (common, parent) = match &curr.kind {
                        FlowHardwareKind::Root(curr) => (Some(&curr.common), None),
                        FlowHardwareKind::Branch(curr) => (Some(&curr.common), Some(&curr.parent)),
                        FlowHardwareKind::Scoped(curr) => (None, Some(&curr.parent)),
                    };

                    // check if this hardware flow has some information about the signal
                    if let Some(common) = common {
                        if let Some(signal_content) = common.signals.get(&signal.inner) {
                            match signal_content {
                                SignalContent::Compile(value) => {
                                    // we've found a compile-time value, we can return that
                                    return Ok(Value::from(value.clone()));
                                }
                                SignalContent::Hardware(_) => {
                                    // we've found proof that the signal is hardware, so we need to stop
                                    // (this is important to allow children to override the parent)
                                    break;
                                }
                            }
                        }
                    }

                    parent
                }
            };
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
            parent: FlowCompileKind::Root,
            variables: IndexMap::new(),
        }
    }

    // TODO think about a better name
    pub fn new_child_isolated(&self) -> FlowCompile<'_> {
        let slf = unsafe { lifetime_cast::compile_ref(self) };
        FlowCompile {
            root: self.root,
            compile_span: self.compile_span,
            compile_reason: self.compile_reason,
            parent: FlowCompileKind::IsolatedCompile(slf),
            variables: IndexMap::default(),
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
            parent: FlowCompileKind::ScopedCompile(slf),
            variables: IndexMap::default(),
        }
    }

    pub fn into_content(self) -> FlowCompileContent {
        let FlowCompile {
            root,
            compile_span,
            compile_reason,
            parent: _,
            variables: variable_slots,
        } = self;
        FlowCompileContent {
            check: root.check,
            compile_span,
            compile_reason,
            variables: variable_slots,
        }
    }

    pub fn restore_root(root: &'p FlowRoot, content: FlowCompileContent) -> FlowCompile<'p> {
        let FlowCompileContent {
            check,
            compile_span,
            compile_reason,
            variables: variable_slots,
        } = content;
        assert_eq!(check, root.check);

        FlowCompile {
            root,
            compile_span,
            compile_reason,
            parent: FlowCompileKind::Root,
            variables: variable_slots,
        }
    }

    pub fn restore_child_isolated<'s>(parent: &'s FlowCompile, content: FlowCompileContent) -> FlowCompile<'s> {
        let FlowCompileContent {
            check,
            compile_span,
            compile_reason,
            variables: variable_slots,
        } = content;
        assert_eq!(parent.root.check, check);

        let parent = unsafe { lifetime_cast::compile_ref(parent) };
        FlowCompile {
            root: parent.root,
            compile_span,
            compile_reason,
            parent: FlowCompileKind::IsolatedCompile(parent),
            variables: variable_slots,
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
        // build new flow
        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_mut(self) };

        let mut result = FlowHardwareBranch {
            root,
            parent: slf,
            cond_domain,
            common: FlowHardwareCommon {
                variables: IndexMap::new(),
                signals: IndexMap::new(),
                statements: vec![],
            },
        };

        // apply implications
        let mut result_hw = result.as_flow();
        for implication in cond_implications {
            try_inner!(result_hw.add_implication(ctx, span, implication)?);
        }

        Ok(Ok(result))
    }

    pub fn add_implication(
        &mut self,
        ctx: &mut CompileItemContext,
        span: Span,
        implication: Implication,
    ) -> DiagResult<Result<(), ImplicationContradiction>> {
        let Implication { version, kind } = implication;
        let ValueVersion { signal, index } = version;

        match signal {
            SignalOrVariable::Signal(signal) => {
                let prev_content = self.signal_get_content(signal)?;

                // only apply implications to the latest version
                match prev_content {
                    SignalContent::Compile(_) => {}
                    SignalContent::Hardware(prev_content) => {
                        if prev_content.version != index {
                            return Ok(Ok(()));
                        }
                    }
                }

                match kind {
                    ImplicationKind::BoolEq(impl_value) => {
                        if let &SignalContent::Compile(CompileValue::Simple(SimpleCompileValue::Bool(prev_value))) =
                            prev_content
                        {
                            // already compile, just check
                            if prev_value != impl_value {
                                return Ok(Err(ImplicationContradiction));
                            }
                        } else {
                            // not yet compile, remember value
                            self.first_common_mut()
                                .signals
                                .insert(signal, SignalContent::Compile(CompileValue::new_bool(impl_value)));
                        }
                    }
                    ImplicationKind::IntIn(impl_range) => {
                        match prev_content {
                            SignalContent::Compile(prev_value) => {
                                // already compile, just check
                                let prev_value =
                                    unwrap_match!(prev_value, CompileValue::Simple(SimpleCompileValue::Int(v)) => v);
                                if !impl_range.contains(prev_value) {
                                    return Ok(Err(ImplicationContradiction));
                                }
                            }
                            SignalContent::Hardware(prev_value) => {
                                // not yet compile, intersect ranges
                                let prev_ty = prev_value.ty_hw(ctx, signal, span)?;
                                let prev_range = unwrap_match!(prev_ty, HardwareType::Int(range) => range);

                                let new_range = ClosedMultiRange::from(prev_range).intersect(&impl_range);
                                let new_range = try_inner!(
                                    ClosedNonEmptyMultiRange::try_from(new_range)
                                        .map_err(|_: RangeEmpty| ImplicationContradiction)
                                );

                                let new_content = SignalContent::Hardware(SignalContentHardware {
                                    version: prev_value.version,
                                    implied_int_range: Some(new_range),
                                });
                                self.first_common_mut().signals.insert(signal, new_content);
                            }
                        }
                    }
                }
            }
            SignalOrVariable::Variable(var) => {
                let prev_content = self.var_get_content(Spanned::new(span, var))?;

                // we don't need to track implications for error cases
                let prev_value = match prev_content {
                    VariableContent::Assigned(prev_value) => prev_value,
                    VariableContent::NotFullyAssigned(_) | VariableContent::Error(_) => return Ok(Ok(())),
                };

                // only apply implications to the latest version
                match &prev_value.inner {
                    VariableValue::Simple(_) => {}
                    VariableValue::Compound(_) => {}
                    VariableValue::Hardware(prev_value) => {
                        if prev_value.version != index {
                            return Ok(Ok(()));
                        }
                    }
                }

                match kind {
                    ImplicationKind::BoolEq(impl_value) => {
                        if let Value::Simple(SimpleCompileValue::Bool(prev_value)) = prev_value.inner {
                            // already compile, just check
                            if prev_value != impl_value {
                                return Ok(Err(ImplicationContradiction));
                            }
                        } else {
                            // not yet compile, remember value
                            let new_content = VariableContent::Assigned(Spanned::new(
                                prev_value.span,
                                Value::Simple(SimpleCompileValue::Bool(impl_value)),
                            ));
                            let new_slot = VariableSlotOption {
                                info: None,
                                content: Some(new_content),
                            };
                            self.first_common_mut().variables.insert(var.index, new_slot);
                        }
                    }
                    ImplicationKind::IntIn(impl_range) => {
                        match &prev_value.inner {
                            VariableValue::Simple(SimpleCompileValue::Int(prev_value)) => {
                                // already compile, just check
                                if !impl_range.contains(prev_value) {
                                    return Ok(Err(ImplicationContradiction));
                                }
                            }
                            VariableValue::Hardware(prev_value_hw) => {
                                let prev_range =
                                    unwrap_match!(prev_value_hw.ty_hw(), HardwareType::Int(range) => range);

                                let new_range = ClosedMultiRange::from(prev_range).intersect(&impl_range);
                                let new_range = try_inner!(
                                    ClosedNonEmptyMultiRange::try_from(new_range)
                                        .map_err(|_: RangeEmpty| ImplicationContradiction)
                                );

                                let new_value = VariableValueHardware {
                                    value_raw: prev_value_hw.value_raw.clone(),
                                    version: prev_value_hw.version,
                                    implied_int_range: Some(new_range),
                                };
                                let new_content = VariableContent::Assigned(Spanned::new(
                                    prev_value.span,
                                    VariableValue::Hardware(new_value),
                                ));
                                let new_slot = VariableSlotOption {
                                    info: None,
                                    content: Some(new_content),
                                };
                                self.first_common_mut().variables.insert(var.index, new_slot);
                            }
                            _ => unreachable!("trying to imply int range for non-int var"),
                        }
                    }
                }
            }
        }

        Ok(Ok(()))
    }

    pub fn new_child_scoped(&mut self) -> FlowHardware<'_> {
        let root = self.root;
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        FlowHardware {
            root,
            enable_domain_checks: true,
            kind: FlowHardwareKind::Scoped(FlowHardwareScoped {
                parent: slf,
                variables: IndexMap::new(),
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
        // TODO do something else if children are empty, eg. push an instruction for `assert(false)`
        //    or maybe even compile-time error?

        // collect the interesting vars and signals
        // things we can skip:
        //   * items that are not in any child, they didn't change
        //   * items that don't exist in the parent, they won't be alive afterwards
        let mut merged_vars = IndexSet::new();
        let mut merged_signals = IndexSet::new();
        for branch in &branches {
            assert_eq!(self.root.check, branch.check);
            for &var in branch.common.variables.keys() {
                if self.var_info_option(var).is_some() {
                    merged_vars.insert(var);
                }
            }
            for &signal in branch.common.signals.keys() {
                merged_signals.insert(signal);
            }
        }

        // merge variables
        for var in merged_vars {
            let var = Variable {
                check: self.root.check,
                index: var,
            };

            let value_merged = merge_branch_variable(refs, large, self, span_merge, var, &mut branches)
                .unwrap_or_else(VariableContent::Error);

            self.var_set_content(var, span_merge, value_merged)?;
        }

        // merge signals
        for signal in merged_signals {
            let signal_merged = merge_branch_signal(self, signal, &mut branches)?;
            self.signal_set_content(signal, signal_merged);
        }

        // extract blocks
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
                    signals: _,
                    statements,
                } = common;
                IrBlock { statements }
            })
            .collect_vec();

        // TODO merge condition domains too?
        Ok(branch_blocks)
    }

    pub fn get_ir_wires(&mut self) -> &mut IrWires {
        self.root_hw_mut().ir_wires
    }

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

    fn signal_set_content(&mut self, signal: Signal, content: SignalContent) {
        self.first_common_mut().signals.insert(signal, content);
    }

    /// Report that `signal` has been assigned a new value.
    ///
    /// `extra_info` contains any extra information about the assigned value,
    /// either the full compile-time value or a more specific type of the assigned value.
    /// This must apply to the entire new value of the signal, not just the assigned part.
    pub fn signal_report_assignment(
        &mut self,
        signal: Spanned<Signal>,
        extra_info: Option<MaybeCompile<CompileValue, HardwareType>>,
    ) {
        // TODO track partial drivers to avoid clashes and to generate extra concat blocks
        // TODO comb blocks: check that all written signals are _always_ written
        // TODO ban signal assignments in wire expressions and port connections

        let new_hardware_content = |implied_int_range| {
            SignalContent::Hardware(SignalContentHardware {
                version: self.root.next_version(),
                implied_int_range,
            })
        };

        let content = match extra_info {
            Some(extra_info) => match extra_info {
                MaybeCompile::Compile(value) => SignalContent::Compile(value),
                MaybeCompile::Hardware(ty) => match ty {
                    HardwareType::Int(range) => new_hardware_content(Some(range)),
                    _ => new_hardware_content(None),
                },
            },
            None => new_hardware_content(None),
        };
        self.signal_set_content(signal.inner, content);
    }

    pub fn store_hardware_value_in_new_ir_variable(
        &mut self,
        refs: CompileRefs,
        span: Span,
        debug_info_id: Option<String>,
        value: HardwareValue,
    ) -> HardwareValue<HardwareType, IrVariable> {
        let var_ir_info = IrVariableInfo {
            ty: value.ty.as_ir(refs),
            debug_info_span: span,
            debug_info_id,
        };

        let var_ir = self.new_ir_variable(var_ir_info);

        let target = IrAssignmentTarget::simple(var_ir.into());
        let stmt_store = IrStatement::Assign(target, value.expr);
        self.push_ir_statement(Spanned::new(span, stmt_store));

        HardwareValue {
            ty: value.ty,
            domain: value.domain,
            expr: var_ir,
        }
    }
}

impl FlowPrivate for FlowHardware<'_> {
    fn root(&self) -> &FlowRoot<'_> {
        self.root
    }

    fn var_info_option(&self, var: VariableIndex) -> Option<&VariableInfo> {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => {
                    if let Some(slot) = root.common.variables.get(&var) {
                        if let Some(info) = &slot.info {
                            return Some(info);
                        }
                    }
                    return root.parent.var_info_option(var);
                }
                FlowHardwareKind::Branch(branch) => {
                    if let Some(slot) = branch.common.variables.get(&var) {
                        if let Some(info) = &slot.info {
                            return Some(info);
                        }
                    }
                    branch.parent
                }
                FlowHardwareKind::Scoped(scoped) => {
                    if let Some(slot) = scoped.variables.get(&var) {
                        return Some(&slot.info);
                    }
                    scoped.parent
                }
            };
        }
    }

    fn var_set_content(&mut self, var: Variable, _: Span, content: VariableContent) -> DiagResult {
        assert_eq!(self.root.check, var.check);

        // find the right flow to declare this variable in
        let mut curr = self;
        let variables = loop {
            curr = match &mut curr.kind {
                FlowHardwareKind::Root(root) => break &mut root.common.variables,
                FlowHardwareKind::Branch(branch) => break &mut branch.common.variables,
                FlowHardwareKind::Scoped(scoped) => {
                    // if the value was declared in this scope we can store it here,
                    //   otherwise fallthrough to the parent
                    if let Some(entry) = scoped.variables.get_mut(&var.index) {
                        entry.content = content;
                        return Ok(());
                    }

                    scoped.parent
                }
            }
        };

        variables.entry(var.index).or_default().content = Some(content);
        Ok(())
    }

    fn var_get_content(&self, var: Spanned<Variable>) -> DiagResult<&VariableContent> {
        assert_eq!(self.root.check, var.inner.check);

        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => {
                    if let Some(slot) = root.common.variables.get(&var.inner.index) {
                        if let Some(content) = &slot.content {
                            return Ok(content);
                        }
                    }
                    return root.parent.var_get_content(var);
                }
                FlowHardwareKind::Branch(branch) => {
                    if let Some(slot) = branch.common.variables.get(&var.inner.index) {
                        if let Some(content) = &slot.content {
                            return Ok(content);
                        }
                    }
                    branch.parent
                }
                FlowHardwareKind::Scoped(scoped) => {
                    if let Some(slot) = scoped.variables.get(&var.inner.index) {
                        return Ok(&slot.content);
                    }
                    scoped.parent
                }
            };
        }
    }

    fn signal_get_content(&self, signal: Signal) -> DiagResult<&SignalContent> {
        let mut curr = self;
        loop {
            curr = match &curr.kind {
                FlowHardwareKind::Root(root) => {
                    if let Some(content) = root.common.signals.get(&signal) {
                        break Ok(content);
                    }
                    break Ok(&SignalContent::INITIAL);
                }
                FlowHardwareKind::Branch(branch) => {
                    if let Some(content) = branch.common.signals.get(&signal) {
                        break Ok(content);
                    }
                    branch.parent
                }
                FlowHardwareKind::Scoped(scoped) => scoped.parent,
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
            parent: FlowCompileKind::ScopedHardware(slf),
            variables: IndexMap::new(),
        }
    }

    fn require_hardware(&mut self, _: Span, _: &str) -> DiagResult<&mut FlowHardware<'_>> {
        let slf = unsafe { lifetime_cast::hardware_mut(self) };
        Ok(slf)
    }

    fn kind(&self) -> FlowKind<&FlowCompile<'_>, &FlowHardware<'_>> {
        FlowKind::Hardware(unsafe { lifetime_cast::hardware_ref(self) })
    }

    fn kind_mut(&mut self) -> FlowKind<&mut FlowCompile<'_>, &mut FlowHardware<'_>> {
        FlowKind::Hardware(unsafe { lifetime_cast::hardware_mut(self) })
    }

    fn var_new(&mut self, info: VariableInfo) -> Variable {
        let var = self.root.next_variable();
        let content = VariableContent::NotFullyAssigned(VariableNotFullyAssigned::NotYetAssigned);

        match &mut self.kind {
            FlowHardwareKind::Root(root) => {
                let slot = VariableSlotOption {
                    info: Some(info),
                    content: Some(content),
                };
                root.common.variables.insert(var.index, slot);
            }
            FlowHardwareKind::Branch(branch) => {
                let slot = VariableSlotOption {
                    info: Some(info),
                    content: Some(content),
                };
                branch.common.variables.insert(var.index, slot);
            }
            FlowHardwareKind::Scoped(scoped) => {
                let slot = VariableSlot { info, content };
                scoped.variables.insert(var.index, slot);
            }
        }

        var
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
            common: FlowHardwareCommon {
                variables: Default::default(),
                signals: Default::default(),
                statements: vec![],
            },
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
            span: _,
            process_kind: _,
            ir_wires: _,
            ir_registers: _,
            ir_variables,
            common,
        } = self;
        let FlowHardwareCommon {
            variables: _,
            signals: _,
            statements,
        } = common;

        let block = IrBlock { statements };
        (ir_variables, block)
    }
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
enum MergedVariable {
    /// Not all branches have assigned something to the variable./
    NotFullyAssigned,
    /// All branches have assigned the same value to the variable.
    /// This applies even if the spans and implied ranges are different,
    ///   in which case this will store the already merged result.
    AllMatch(Spanned<VariableValue>),
    /// Different branches have assigned different values to the variable, so a hardware merge is needed.
    NeedsHardwareMerge,
}

fn merge_branch_variable(
    refs: CompileRefs,
    large: &mut IrLargeArena,
    parent_flow: &mut FlowHardware,
    span_merge: Span,
    var: Variable,
    branches: &mut [FlowHardwareBranchContent],
) -> DiagResult<VariableContent> {
    let diags = refs.diags;
    let elab = &refs.shared.elaboration_arenas;

    let var_spanned = Spanned::new(span_merge, var);
    let var_info = parent_flow.var_info(var_spanned)?;
    let parent_content = parent_flow.var_get_content(var_spanned)?;

    // visit all branch values to check if we need to do a hardware merge
    let mut merged: Option<MergedVariable> = None;
    let mut used_parent_value = false;

    for branch in &mut *branches {
        let branch_content = match branch.common.variables.get(&var.index) {
            Some(branch_slot) => {
                // we can assert/unwrap here:
                // * we know merged variables must exist in the parent, so they cannot be declared in branches
                // * slots cannot be fully empty, so if there's no info there must be content
                assert!(branch_slot.info.is_none());
                branch_slot.content.as_ref().unwrap()
            }
            None => {
                used_parent_value = true;
                parent_content
            }
        };

        // first (maybe non-undefined) branch, just set it
        let Some(merged_value) = merged else {
            merged = match branch_content {
                VariableContent::Assigned(value) => Some(MergedVariable::AllMatch(value.clone())),
                VariableContent::NotFullyAssigned(kind) => match kind {
                    VariableNotFullyAssigned::NotYetAssigned | VariableNotFullyAssigned::PartiallyAssigned => {
                        Some(MergedVariable::NotFullyAssigned)
                    }
                    VariableNotFullyAssigned::Undefined => None,
                },
                &VariableContent::Error(e) => return Err(e),
            };
            continue;
        };

        // join multiple values
        let merged_new = match (merged_value, branch_content) {
            (MergedVariable::NotFullyAssigned, _)
            | (
                _,
                VariableContent::NotFullyAssigned(VariableNotFullyAssigned::NotYetAssigned)
                | VariableContent::NotFullyAssigned(VariableNotFullyAssigned::PartiallyAssigned),
            ) => {
                // if anything is not fully assigned, the result is not fully assigned
                MergedVariable::NotFullyAssigned
            }
            (_, &VariableContent::Error(e)) => {
                // short-circuit on error
                return Err(e);
            }
            (merged_value, VariableContent::NotFullyAssigned(VariableNotFullyAssigned::Undefined)) => {
                // undefined branches do not affect the merge result
                merged_value
            }
            (MergedVariable::NeedsHardwareMerge, VariableContent::Assigned(_)) => {
                // if we need a hardware merge and we get more normal values, we will still need a hardware merge
                MergedVariable::NeedsHardwareMerge
            }
            (MergedVariable::AllMatch(merged), VariableContent::Assigned(branch)) => {
                // actual merge check between two assigned values
                let value_matches = match (&merged.inner, &branch.inner) {
                    (VariableValue::Simple(merged), VariableValue::Simple(branch)) => merged == branch,
                    (VariableValue::Compound(merged), VariableValue::Compound(branch)) => {
                        if let Ok(merged) = CompileCompoundValue::try_from(merged)
                            && let Ok(branch) = CompileCompoundValue::try_from(branch)
                        {
                            merged == branch
                        } else {
                            false
                        }
                    }
                    // TODO should we count a match here if the versions are equal? can that ever even happen?
                    (VariableValue::Hardware(_), VariableValue::Hardware(_)) => false,

                    (VariableValue::Simple(_), VariableValue::Hardware(_) | VariableValue::Compound(_)) => false,
                    (VariableValue::Hardware(_), VariableValue::Simple(_) | VariableValue::Compound(_)) => false,
                    (VariableValue::Compound(_), VariableValue::Simple(_) | VariableValue::Hardware(_)) => false,
                };

                if value_matches {
                    let span_new = if merged.span == branch.span {
                        // TODO is this case actually possible?
                        merged.span
                    } else {
                        span_merge
                    };
                    MergedVariable::AllMatch(Spanned::new(span_new, merged.inner))
                } else {
                    MergedVariable::NeedsHardwareMerge
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
            return Ok(VariableContent::NotFullyAssigned(VariableNotFullyAssigned::Undefined));
        }
        Some(MergedVariable::AllMatch(value)) => {
            // all branches agree on the assigned value, we don't need to do a hardware merge
            return Ok(VariableContent::Assigned(value));
        }
        Some(MergedVariable::NotFullyAssigned) => {
            // fully un-assigned would have exited earlier, so the result is partially assigned
            return Ok(VariableContent::NotFullyAssigned(
                VariableNotFullyAssigned::PartiallyAssigned,
            ));
        }
        Some(MergedVariable::NeedsHardwareMerge) => {
            // fall through into hardware merge
        }
    }

    // at this point we know that there are no un-assigned branches, and that not all branches have the same value:
    //   this means we actually need to do a hardware merge
    fn unwrap_branch_value(v: &VariableContent) -> MaybeUndefined<Spanned<&VariableValue>> {
        match v {
            VariableContent::Assigned(v) => MaybeUndefined::Defined(v.as_ref()),
            VariableContent::NotFullyAssigned(kind) => match kind {
                VariableNotFullyAssigned::NotYetAssigned | VariableNotFullyAssigned::PartiallyAssigned => {
                    unreachable!("expected variable content")
                }
                VariableNotFullyAssigned::Undefined => MaybeUndefined::Undefined,
            },
            VariableContent::Error(_) => unreachable!("expected variable content"),
        }
    }
    fn unwrap_branch_slot(slot: &VariableSlotOption) -> &VariableContent {
        assert!(slot.info.is_none());
        slot.content.as_ref().unwrap()
    }

    // check that all types are hardware
    // (we do this before finding the common type to get nicer error messages)
    let branch_tys = branches
        .iter()
        .map(|branch| {
            let branch_content = match branch.common.variables.get(&var.index) {
                Some(branch_slot) => unwrap_branch_slot(branch_slot),
                None => parent_content,
            };
            let branch_value = unwrap_branch_value(branch_content);

            let branch_value = match branch_value {
                MaybeUndefined::Undefined => return Ok(HardwareType::Undefined),
                MaybeUndefined::Defined(branch_value) => branch_value,
            };
            let branch_ty = branch_value.inner.ty();

            branch_ty.as_hardware_type(elab).map_err(|_| {
                let ty_str = branch_ty.value_string(elab);
                let diag = Diagnostic::new("merging if assignments needs hardware type")
                    .add_info(var_info.span_decl, "for this variable")
                    .add_info(
                        branch_value.span,
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
            if let Some(branch_slot) = branch.common.variables.get(&var.index) {
                let branch_content = unwrap_branch_slot(branch_slot);
                let branch_value = unwrap_branch_value(branch_content);

                match branch_value {
                    MaybeUndefined::Undefined => {}
                    MaybeUndefined::Defined(branch_value) => {
                        diag = diag.add_info(
                            branch_value.span,
                            format!("value in branch assigned here has type `{}`", ty.value_string(elab)),
                        )
                    }
                }
            }
        }
        if used_parent_value {
            let parent_value = unwrap_branch_value(parent_content);

            match parent_value {
                MaybeUndefined::Undefined => {}
                MaybeUndefined::Defined(parent_value) => {
                    diag = diag.add_info(
                        parent_value.span,
                        format!("value before branch assigned here has type `{}`", ty.value_string(elab)),
                    );
                }
            }
        }
        diags.report(diag.finish())
    })?;

    // create result variable
    let var_ir = match var_info.join_ir_variable {
        None => {
            let var_ir_info = IrVariableInfo {
                ty: ty.as_ir(refs),
                debug_info_span: var_info.span_decl,
                debug_info_id: var_info.id.as_str(refs.fixed.source).map(str::to_owned),
            };
            parent_flow.new_ir_variable(var_ir_info)
        }
        Some(var_ir) => var_ir,
    };

    // store values into that variable
    let mut domain = ValueDomain::CompileTime;
    let mut domain_cond = ValueDomain::CompileTime;

    let mut build_store = |value: MaybeUndefined<Spanned<&VariableValue>>| {
        match value {
            MaybeUndefined::Undefined => {
                // undefined, skip store
                Ok(None)
            }
            MaybeUndefined::Defined(assigned) => {
                let (assigned_domain, assigned_expr) = match &assigned.inner {
                    Value::Simple(v) => (
                        v.domain(),
                        v.as_ir_expression_unchecked(refs, large, assigned.span, &ty)?,
                    ),
                    Value::Compound(v) => (
                        v.domain(),
                        v.as_ir_expression_unchecked(refs, large, assigned.span, &ty)?,
                    ),
                    Value::Hardware(v) => {
                        // no need to take a copy here,
                        //   we're immediately using this value in a store operation and then discarding it
                        let value = v.as_hardware_value_without_copy(large, var);
                        let expr = value
                            .value
                            .as_ir_expression_unchecked(refs, large, assigned.span, &ty)?;
                        (value.value.domain, expr)
                    }
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

    for branch in branches {
        if let Some(branch_slot) = branch.common.variables.get(&var.index) {
            let branch_content = unwrap_branch_slot(branch_slot);
            let branch_value = unwrap_branch_value(branch_content);
            if let Some(store) = build_store(branch_value)? {
                branch.common.statements.push(store);
            }
        }
        domain_cond = domain_cond.join(branch.cond_domain.inner);
    }
    if used_parent_value {
        let parent_content = parent_flow.var_get_content(var_spanned)?;
        let parent_value = unwrap_branch_value(parent_content);
        if let Some(store) = build_store(parent_value)? {
            parent_flow.push_ir_statement(store);
        }
    }
    domain = domain.join(domain_cond);

    // wrap result
    // (we don't need to set an implied range here, we've just created a new variable with the right type)
    let result_value = HardwareValue {
        ty,
        domain,
        expr: var_ir,
    };
    let result_version = parent_flow.root.next_version();
    let result_var_value = VariableValueHardware {
        value_raw: result_value,
        version: result_version,
        implied_int_range: None,
    };
    Ok(VariableContent::Assigned(Spanned::new(
        span_merge,
        VariableValue::Hardware(result_var_value),
    )))
}

fn merge_branch_signal(
    parent_flow: &mut FlowHardware,
    signal: Signal,
    branches: &mut [FlowHardwareBranchContent],
) -> DiagResult<SignalContent> {
    // TODO extract general TriLattice type for this?
    enum Merged<T> {
        None,
        AllMatch(T),
        Different,
    }

    // we don't need to merge versions, if we get to this point we know there are differences between the branches
    let mut merged_compile: Merged<&CompileValue> = Merged::None;
    let mut merged_int_range: Merged<ClosedNonEmptyMultiRange<_>> = Merged::None;

    let parent_content = parent_flow.signal_get_content(signal)?;

    for branch in branches {
        let branch_content = branch.common.signals.get(&signal).unwrap_or(parent_content);

        match branch_content {
            SignalContent::Compile(branch_value) => {
                merged_compile = match merged_compile {
                    Merged::None => Merged::AllMatch(branch_value),
                    Merged::AllMatch(prev_value) => {
                        if prev_value == branch_value {
                            Merged::AllMatch(prev_value)
                        } else {
                            Merged::Different
                        }
                    }
                    Merged::Different => Merged::Different,
                };

                merged_int_range = if let CompileValue::Simple(SimpleCompileValue::Int(branch_value)) = branch_value {
                    match merged_int_range {
                        Merged::None => Merged::AllMatch(ClosedNonEmptyMultiRange::single(branch_value.clone())),
                        Merged::AllMatch(range) => {
                            Merged::AllMatch(range.union(&ClosedMultiRange::single(branch_value.clone())))
                        }
                        Merged::Different => Merged::Different,
                    }
                } else {
                    Merged::Different
                }
            }
            SignalContent::Hardware(branch_content) => {
                let SignalContentHardware {
                    version: _,
                    implied_int_range,
                } = branch_content;

                merged_compile = Merged::Different;

                merged_int_range = if let Some(implied_int_range) = implied_int_range {
                    match merged_int_range {
                        Merged::None => Merged::AllMatch(implied_int_range.clone()),
                        Merged::AllMatch(range) => {
                            Merged::AllMatch(range.union(&ClosedMultiRange::from(implied_int_range.clone())))
                        }
                        Merged::Different => Merged::Different,
                    }
                } else {
                    Merged::Different
                };
            }
        }
    }

    let merged_content = match merged_compile {
        Merged::AllMatch(merged_compile) => SignalContent::Compile(merged_compile.clone()),
        Merged::None | Merged::Different => {
            let version = parent_flow.root.next_version();

            let implied_int_range = match merged_int_range {
                Merged::AllMatch(range) => Some(range),
                Merged::None | Merged::Different => None,
            };

            SignalContent::Hardware(SignalContentHardware {
                version,
                implied_int_range,
            })
        }
    };
    Ok(merged_content)
}

impl VariableId {
    pub fn as_str<'s>(&self, source: &'s SourceDatabase) -> Option<&'s str> {
        match self {
            VariableId::Id(id) => match id {
                MaybeIdentifier::Dummy { .. } => None,
                MaybeIdentifier::Identifier(id) => Some(id.spanned_str(source).inner),
            },
            VariableId::Custom(s) => Some(s),
        }
    }
}

impl VariableContent {
    fn err_ref(e: DiagError) -> &'static VariableContent {
        let _ = e;
        static ERR: VariableContent = VariableContent::Error(DiagError::promise_error_has_been_reported());
        &ERR
    }
}

impl VariableValueHardware {
    pub fn ty_hw(&self) -> HardwareType {
        match &self.implied_int_range {
            Some(range) => HardwareType::Int(range.clone()),
            None => self.value_raw.ty.clone(),
        }
    }

    pub fn as_hardware_value_without_copy(&self, large: &mut IrLargeArena, var: Variable) -> HardwareValueWithVersion {
        let VariableValueHardware {
            value_raw,
            version,
            implied_int_range,
        } = self;

        let value = if let Some(implied_int_range) = implied_int_range {
            let expr = large.push_expr(IrExpressionLarge::ConstrainIntRange(
                implied_int_range.enclosing_range().cloned(),
                IrExpression::Variable(value_raw.expr),
            ));
            HardwareValue {
                ty: HardwareType::Int(implied_int_range.clone()),
                domain: value_raw.domain,
                expr,
            }
        } else {
            HardwareValue {
                ty: value_raw.ty.clone(),
                domain: value_raw.domain,
                expr: IrExpression::Variable(value_raw.expr),
            }
        };

        HardwareValueWithVersion {
            value,
            version: ValueVersion {
                signal: SignalOrVariable::Variable(var),
                index: *version,
            },
        }
    }
}

impl Typed for VariableValueHardware {
    fn ty(&self) -> Type {
        self.ty_hw().as_type()
    }
}

impl SignalContent {
    pub const INITIAL: Self = SignalContent::Hardware(SignalContentHardware::INITIAL);
}

impl SignalContentHardware {
    pub const INITIAL: Self = SignalContentHardware {
        version: VERSION_INITIAL,
        implied_int_range: None,
    };

    pub fn ty_hw(&self, ctx: &mut CompileItemContext, signal: Signal, span: Span) -> DiagResult<HardwareType> {
        match &self.implied_int_range {
            Some(range) => Ok(HardwareType::Int(range.clone())),
            None => Ok(signal.expect_ty(ctx, span)?.inner.clone()),
        }
    }

    pub fn as_hardware_value(
        &self,
        ctx: &mut CompileItemContext,
        signal: Signal,
        span: Span,
    ) -> DiagResult<HardwareValueWithVersion> {
        let &SignalContentHardware {
            version,
            ref implied_int_range,
        } = self;

        // read from signal (or shadow variable)
        let value_raw = signal.as_hardware_value(ctx, span)?;

        // constrain range if needed
        let value = match implied_int_range {
            None => value_raw,
            Some(range) => {
                let expr = ctx.large.push_expr(IrExpressionLarge::ConstrainIntRange(
                    range.enclosing_range().cloned(),
                    value_raw.expr,
                ));
                HardwareValue {
                    ty: HardwareType::Int(range.clone()),
                    domain: value_raw.domain,
                    expr,
                }
            }
        };

        // wrap result
        let version = ValueVersion {
            signal: SignalOrVariable::Signal(signal),
            index: version,
        };
        Ok(HardwareValueWithVersion { value, version })
    }
}

impl VariableNotFullyAssigned {
    pub fn report_diag(&self, diags: &Diagnostics, span: Span, var_info: &VariableInfo) -> DiagError {
        match self {
            VariableNotFullyAssigned::NotYetAssigned => {
                let diag = Diagnostic::new("variable has not yet been assigned a value")
                    .add_error(span, "variable used here")
                    .add_info(var_info.span_decl, "variable declared here")
                    .finish();
                diags.report(diag)
            }
            VariableNotFullyAssigned::PartiallyAssigned => {
                let diag = Diagnostic::new("variable has not yet been assigned a value in all preceding branches")
                    .add_error(span, "variable used here")
                    .add_info(var_info.span_decl, "variable declared here")
                    .finish();
                diags.report(diag)
            }
            VariableNotFullyAssigned::Undefined => diags.report_internal_error(span, "evaluating undefined variable"),
        }
    }
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
    pub unsafe fn hardware_ref<'s>(flow: &'s FlowHardware) -> &'s FlowHardware<'s> {
        unsafe { &*(flow as *const FlowHardware<'_> as *const FlowHardware<'s>) }
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
