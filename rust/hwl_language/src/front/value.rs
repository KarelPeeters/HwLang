use crate::data::compiled::{CompiledDatabase, CompiledStage, Item, ModulePort, Register, Variable, Wire};
use crate::data::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::common::{GenericContainer, GenericMap};
use crate::front::solver::{Solver, SolverBool, SolverInt, SolverOpaque};
use crate::front::types::{NominalTypeUnique, Type};
use crate::impl_enum_from_err_guaranteed;
use crate::syntax::ast::{DomainKind, SyncDomain};
use crate::syntax::pos::Span;
use itertools::Itertools;

// TODO how does this represent generic parameters that should still be replaced later?
//  eg. in `type bits(n: uint) = bool[n];`, what is the resulting type? `Type::Array(Type::Bool, Value { ... })`?
#[derive(Debug, Clone)]
pub struct Value<C = ValueContent, T = Type> {
    /// The value content.
    pub content: C,
    /// The type of this value for the purposes of declaration type inference.
    // TODO pick a better name for this, and document the type/content separation somewhere
    pub ty_default: T,
    /// The domain of this value.
    pub domain: ValueDomain,
    /// Whether this value is writable, and if it is, what the final target of the write becomes.
    pub access: ValueAccess,
    /// The origin of this value.
    /// Typically the span of the expression that produced it.
    pub origin: Span,
}

pub type ValueInt = Value<SolverInt, Result<RangeInfo<SolverInt>, ErrorGuaranteed>>;
pub type ValueBool = Value<SolverBool, Result<(), ErrorGuaranteed>>;

#[derive(Debug, Clone)]
pub enum ValueContent {
    Error(ErrorGuaranteed),
    Undefined,
    Bool(SolverBool),
    Int(SolverInt),
    Opaque(SolverOpaque),
    Range(RangeInfo<ValueInt>),
    Module(ModuleValueInfo),
}

impl_enum_from_err_guaranteed!(ValueContent);

#[derive(Debug, Clone)]
pub enum ValueAccess {
    ReadOnly,
    WriteOnlyPort(ModulePort),
    WriteReadWire(Wire),
    WriteReadRegister(Register),
    WriteReadVariable(Variable),
    Error(ErrorGuaranteed),
}

impl<C: From<ErrorGuaranteed>> Value<C, Type> {
    pub fn error(e: ErrorGuaranteed, origin: Span) -> Self {
        Value {
            content: C::from(e),
            ty_default: Type::Error(e),
            domain: ValueDomain::Error(e),
            access: ValueAccess::Error(e),
            origin,
        }
    }
}

impl ValueInt {
    pub fn error_int(solver: &mut Solver, e: ErrorGuaranteed, origin: Span) -> ValueInt {
        ValueInt {
            content: solver.error_int(e),
            ty_default: Err(e),
            domain: ValueDomain::Error(e),
            access: ValueAccess::Error(e),
            origin,
        }
    }
}

impl ValueContent {
    pub fn unwrap_bool(&self, span: Span, diags: &Diagnostics, solver: &mut Solver) -> SolverBool {
        match self {
            &ValueContent::Bool(b) => b,
            _ => solver.error_bool(diags.report_internal_error(span, "expected boolean content")),
        }
    }

    pub fn unwrap_int(&self, span: Span, diags: &Diagnostics, solver: &mut Solver) -> SolverInt {
        match self {
            &ValueContent::Int(i) => i,
            _ => solver.error_int(diags.report_internal_error(span, "expected integer content")),
        }
    }

    pub fn opaque_of_type(ty: &Type, solver: &mut Solver) -> ValueContent {
        // TODO unchecked propagation (also check everything else that matches on type)
        match ty {
            &Type::Error(e) => ValueContent::Error(e),
            Type::Boolean => ValueContent::Bool(solver.arbitrary_bool()),
            &Type::Integer(RangeInfo { start_inc, end_inc }) => {
                let result = solver.arbitrary_int();
                if let Some(start) = start_inc {
                    let axiom = solver.compare_lte(start, result);
                    solver.add_axiom(axiom);
                }
                if let Some(end) = end_inc {
                    let axiom = solver.compare_lte(result, end);
                    solver.add_axiom(axiom);
                }
                ValueContent::Int(result)
            }
            _ => ValueContent::Opaque(solver.new_opaque()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValueDomain<V = DomainSignal> {
    Error(ErrorGuaranteed),
    // TODO rename to compile-time, this is not necessarily constant
    //   (eg. in functions, certain variables might be constant) 
    CompileTime,
    Clock,
    // TODO allow separate sync/async per edge, necessary for "async" reset
    Async,
    Sync(SyncDomain<V>),
    FunctionBody(Item),
}

// TODO expand to all possible values again
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DomainSignal {
    Error(ErrorGuaranteed),
    Port(ModulePort),
    Wire(Wire),
    Register(Register),
    Invert(Box<DomainSignal>),
}

impl_enum_from_err_guaranteed!(DomainSignal);

impl ValueDomain {
    pub fn from_domain_kind(domain: DomainKind<DomainSignal>) -> Self {
        match domain {
            DomainKind::Async => ValueDomain::Async,
            DomainKind::Sync(sync) => ValueDomain::Sync(SyncDomain {
                clock: sync.clock,
                reset: sync.reset,
            }),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ArrayAccessIndex<V> {
    Error(ErrorGuaranteed),
    Single(V),
    Range(BoundedRangeInfo<V>),
}

/// Both start and end are inclusive.
/// This is convenient for arithmetic range calculations.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RangeInfo<V> {
    pub start_inc: Option<V>,
    pub end_inc: Option<V>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct BoundedRangeInfo<V> {
    pub start_inc: V,
    pub end_inc: V,
}

impl<V> BoundedRangeInfo<V> {
    pub fn into_general_range(self) -> RangeInfo<V> {
        RangeInfo {
            start_inc: Some(self.start_inc),
            end_inc: Some(self.end_inc),
        }
    }
}

// TODO double check which fields should be used for eq and hash
#[derive(Debug, Clone)]
pub struct FunctionReturnValue {
    pub item: Item,
    pub ret_ty: Type,
}

#[derive(Debug, Clone)]
pub struct ModuleValueInfo {
    // TODO should this be here or not?
    pub nominal_type_unique: NominalTypeUnique,
    // TODO don't include this, make this a cheaply copyable type
    pub ports: Vec<ModulePort>,
}

impl<V> RangeInfo<V> {
    pub const UNBOUNDED: RangeInfo<V> = RangeInfo {
        start_inc: None,
        end_inc: None,
    };

    pub fn as_ref(&self) -> RangeInfo<&V> {
        RangeInfo {
            start_inc: self.start_inc.as_ref(),
            end_inc: self.end_inc.as_ref(),
        }
    }

    pub fn map_inner<U>(self, mut f: impl FnMut(V) -> U) -> RangeInfo<U> {
        RangeInfo {
            start_inc: self.start_inc.map(&mut f),
            end_inc: self.end_inc.map(&mut f),
        }
    }
}

impl GenericContainer for Value {
    type Result = Value;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> Value {
        Value {
            origin: self.origin,
            ty_default: self.ty_default.replace_generics(compiled, map),
            domain: self.domain.replace_generics(compiled, map),
            access: self.access.clone(),
            content: self.content.replace_generics(compiled, map),
        }
    }
}

impl GenericContainer for ValueContent {
    type Result = ValueContent;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> ValueContent {
        match self {
            &ValueContent::Error(e) => ValueContent::Error(e),
            &ValueContent::Bool(b) => ValueContent::Bool(b),
            &ValueContent::Int(i) => ValueContent::Int(i),
            ValueContent::Opaque(o) => ValueContent::Opaque(*o),
            ValueContent::Range(r) => ValueContent::Range(r.clone()),
            ValueContent::Module(m) => ValueContent::Module(m.replace_generics(compiled, map)),
            ValueContent::Undefined => ValueContent::Undefined,
        }
    }
}

impl GenericContainer for ValueDomain {
    type Result = ValueDomain;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> ValueDomain {
        match self {
            ValueDomain::Error(e) => ValueDomain::Error(e.clone()),
            ValueDomain::CompileTime => ValueDomain::CompileTime,
            ValueDomain::Clock => ValueDomain::Clock,
            ValueDomain::Async => ValueDomain::Async,
            ValueDomain::Sync(sync) => ValueDomain::Sync(sync.replace_generics(compiled, map)),
            ValueDomain::FunctionBody(i) => ValueDomain::FunctionBody(i.clone()),
        }
    }
}

impl GenericContainer for SyncDomain<DomainSignal> {
    type Result = SyncDomain<DomainSignal>;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> SyncDomain<DomainSignal> {
        SyncDomain {
            clock: self.clock.replace_generics(compiled, map),
            reset: self.reset.replace_generics(compiled, map),
        }
    }
}

impl GenericContainer for DomainSignal {
    type Result = DomainSignal;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> DomainSignal {
        match self {
            &DomainSignal::Error(e) => DomainSignal::Error(e),
            DomainSignal::Invert(v) => DomainSignal::Invert(Box::new(v.replace_generics(compiled, map))),
            DomainSignal::Port(p) => DomainSignal::Port(p.replace_generics(compiled, map)),
            &DomainSignal::Wire(w) => DomainSignal::Wire(w),
            &DomainSignal::Register(r) => DomainSignal::Register(r),
        }
    }
}

impl GenericContainer for ArrayAccessIndex<Box<Value>> {
    type Result = ArrayAccessIndex<Box<Value>>;

    fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
        match self {
            ArrayAccessIndex::Error(e) => ArrayAccessIndex::Error(e.clone()),
            ArrayAccessIndex::Single(v) => ArrayAccessIndex::Single(Box::new(v.replace_generics(compiled, map))),
            ArrayAccessIndex::Range(r) => ArrayAccessIndex::Range(r.replace_generics(compiled, map)),
        }
    }
}

impl GenericContainer for RangeInfo<Box<Value>> {
    type Result = RangeInfo<Box<Value>>;

    fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
        RangeInfo {
            start_inc: self.start_inc.as_ref().map(|v| Box::new(v.replace_generics(compiled, map))),
            end_inc: self.end_inc.as_ref().map(|v| Box::new(v.replace_generics(compiled, map))),
        }
    }
}

impl GenericContainer for BoundedRangeInfo<Box<Value>> {
    type Result = BoundedRangeInfo<Box<Value>>;

    fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
        BoundedRangeInfo {
            start_inc: Box::new(self.start_inc.replace_generics(compiled, map)),
            end_inc: Box::new(self.end_inc.replace_generics(compiled, map)),
        }
    }
}

impl GenericContainer for ModuleValueInfo {
    type Result = ModuleValueInfo;

    fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
        ModuleValueInfo {
            nominal_type_unique: self.nominal_type_unique.clone(),
            ports: self.ports.iter().map(|p| p.replace_generics(compiled, map)).collect_vec(),
        }
    }
}