use crate::data::compiled::{CompiledDatabase, CompiledStage, GenericTypeParameter, GenericTypeParameterInfo, GenericValueParameter, GenericValueParameterInfo, Item, ModulePort, ModulePortInfo};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::syntax::ast::{DomainKind, PortKind, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use indexmap::IndexMap;
// TODO move all common stuff to the data module, since multiple stages depend on it

/// The context in which this expression or block is used.
/// This is used to determine which expression kinds are allowed and how they should behave,
/// and to pass through the necessary metadata.
///
/// For example:
/// * return is only allowed in functions
/// * break/continue are only allowed in loops
/// TODO maybe allow break to be used in clocked blocks too?
/// TODO create a wrapper type for (ExpressionContext, Scope), they're very often used together
#[derive(Debug)]
pub enum ExpressionContext<'a> {
    Type(Span),
    Const(Span),
    /// Used in function body as part of normal statements or expressions.
    // TODO just get the return type from the function item as-needed
    FunctionBody { func_item: Item, ret_ty_span: Span, ret_ty: Type },
    /// Used for wire expressions or instance port connections.
    ModuleBody(Span),
    CombinatorialBlock(Span),
    ClockedBlock(Spanned<&'a SyncDomain<Value>>),
}

#[derive(Debug, Clone)]
pub enum ContextDomainKind<V = ValueDomainKind> {
    Specific(V),
    Passthrough,
}

impl ExpressionContext<'_> {
    pub fn span(&self) -> Span {
        match self {
            &ExpressionContext::Type(span) => span,
            &ExpressionContext::Const(span) => span,
            &ExpressionContext::FunctionBody { ret_ty_span, .. } => ret_ty_span,
            &ExpressionContext::ModuleBody(span) => span,
            &ExpressionContext::CombinatorialBlock(span) => span,
            ExpressionContext::ClockedBlock(spanned) => spanned.span,
        }
    }

    pub fn domain(&self) -> ContextDomainKind {
        match self {
            ExpressionContext::Type(_) => ContextDomainKind::Specific(ValueDomainKind::Const),
            ExpressionContext::Const(_) => ContextDomainKind::Specific(ValueDomainKind::Const),
            &ExpressionContext::FunctionBody { func_item, .. } => ContextDomainKind::Specific(ValueDomainKind::FunctionBody(func_item)),
            ExpressionContext::ModuleBody(_) => ContextDomainKind::Passthrough,
            ExpressionContext::CombinatorialBlock(_) => ContextDomainKind::Passthrough,
            ExpressionContext::ClockedBlock(domain) => ContextDomainKind::Specific(ValueDomainKind::Sync(domain.inner.clone())),
        }
    }
}

impl<V> ContextDomainKind<V> {
    pub fn as_ref(&self) -> ContextDomainKind<&V> {
        match self {
            ContextDomainKind::Specific(domain) => ContextDomainKind::Specific(domain),
            ContextDomainKind::Passthrough => ContextDomainKind::Passthrough,
        }
    }
}


#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(Item),
    Direct(ScopedEntryDirect),
}

pub type ScopedEntryDirect = MaybeConstructor<TypeOrValue>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValue<T = Type, V = Value> {
    Type(T),
    Value(V),
    /// This error case means we don't know whether this is a type or value.
    Error(ErrorGuaranteed),
}

impl<T, V> TypeOrValue<T, V> {
    pub fn as_ref(&self) -> TypeOrValue<&T, &V> {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t),
            TypeOrValue::Value(v) => TypeOrValue::Value(v),
            &TypeOrValue::Error(e) => TypeOrValue::Error(e),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValueDomainKind<V = Value> {
    Error(ErrorGuaranteed),
    Const,
    Clock,
    // TODO allow separate sync/async per edge, necessary for "async" reset
    Async,
    Sync(SyncDomain<V>),
    FunctionBody(Item),
}

impl ValueDomainKind {
    pub fn from_domain_kind(domain: DomainKind<Value>) -> Self {
        match domain {
            DomainKind::Async => ValueDomainKind::Async,
            DomainKind::Sync(sync) => ValueDomainKind::Sync(SyncDomain {
                clock: sync.clock,
                reset: sync.reset,
            }),
        }
    }
}

/// A Monad trait, specifically for replacing generic parameters in a type or value with more concrete arguments.
// TODO this while concept is pretty tricky, maybe it's better to switch to something rust-like
//   where generics don't get deep-replaced, but stay as at least one level of "parameter"
pub trait GenericContainer {
    type Result;

    /// The implementation can assume the replacement has already been kind- and type-checked.
    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> Self::Result;
}

pub struct GenericMap {
    pub generic_ty: IndexMap<GenericTypeParameter, Type>,
    pub generic_value: IndexMap<GenericValueParameter, Value>,
    pub module_port: IndexMap<ModulePort, Value>,
}

impl GenericMap {
    pub fn empty() -> Self {
        Self {
            generic_ty: IndexMap::new(),
            generic_value: IndexMap::new(),
            module_port: IndexMap::new(),
        }
    }
}

impl GenericContainer for TypeOrValue {
    type Result = TypeOrValue;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> Self {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t.replace_generics(compiled, map)),
            TypeOrValue::Value(v) => TypeOrValue::Value(v.replace_generics(compiled, map)),
            &TypeOrValue::Error(e) => TypeOrValue::Error(e),
        }
    }
}

// TODO creating new generic parameters for replacements a good idea?
//   this breaks parameter identity-ness
impl GenericContainer for GenericTypeParameter {
    type Result = Type;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> Type {
        let param = *self;

        if let Some(replacement) = map.generic_ty.get(&param) {
            // replace the entire parameter
            replacement.clone()
        } else {
            // check if bounds of the parameter need to be replaced
            // (for now this doesn't do anything, but this is a compile-type check for added fields, eg. bounds)
            let GenericTypeParameterInfo { defining_item: _, defining_id: _ } = &compiled[param];
            Type::GenericParameter(param)
        }
    }
}

impl GenericContainer for GenericValueParameter {
    type Result = Value;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
        map: &GenericMap,
    ) -> Value {
        let param = *self;

        if let Some(replacement) = map.generic_value.get(&param) {
            // replace the entire parameter
            replacement.clone()
        } else {
            // check if bounds of the parameter need to be replaced
            // (for now this doesn't do anything, but this is a compile-type check for added fields, eg. bounds)
            let ty_new = compiled[param].ty.clone().replace_generics(compiled, map);
            let GenericValueParameterInfo { defining_item, defining_id, ty, ty_span } = &compiled[param];

            if &ty_new != ty {
                let param_new = compiled.generic_value_params.push(GenericValueParameterInfo {
                    defining_item: *defining_item,
                    defining_id: defining_id.clone(),
                    ty: ty_new,
                    ty_span: ty_span.clone(),
                });
                Value::GenericParameter(param_new)
            } else {
                Value::GenericParameter(param)
            }
        }
    }
}

impl GenericContainer for ModulePort {
    type Result = ModulePort;

    fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
        // we intentionally don't replace the port itself, we might need it to stay a port during module port mapping
        // in the [Value] enum the port will still potentially be replaced by an arbitrary value

        let port = *self;

        let kind_new = match compiled[port].kind.clone() {
            PortKind::Clock => PortKind::Clock,
            PortKind::Normal { domain: sync, ty } => PortKind::Normal {
                domain: match sync {
                    DomainKind::Async => DomainKind::Async,
                    DomainKind::Sync(domain) => DomainKind::Sync(SyncDomain {
                        clock: domain.clock.replace_generics(compiled, map),
                        reset: domain.reset.replace_generics(compiled, map),
                    }),
                },
                ty: ty.replace_generics(compiled, map),
            },
        };

        let &ModulePortInfo { ast, direction, ref kind } = &compiled[port];
        if &kind_new != kind {
            let port_new = compiled.module_ports.push(ModulePortInfo {
                ast,
                direction,
                kind: kind_new,
            });
            port_new
        } else {
            port
        }
    }
}