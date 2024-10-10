use crate::data::compiled::{CompiledDatabasePartial, GenericTypeParameter, GenericTypeParameterInfo, GenericValueParameter, GenericValueParameterInfo, Item, ModulePort, ModulePortInfo};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::syntax::ast::{DomainKind, PortKind, SyncDomain};
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
#[derive(Debug)]
pub enum ExpressionContext {
    /// Used in function body as part of normal statements or expressions.
    FunctionBody { ret_ty_span: Span, ret_ty: Type },
    /// Anything else, even including type definitions inside of function bodies.
    NotFunctionBody,
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
pub enum ValueDomainKind {
    Error(ErrorGuaranteed),
    Const,
    Clock,
    Async,
    Sync(SyncDomain<Value>),
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
    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map: &GenericMap,
    ) -> Self::Result;
}

pub struct GenericMap<'a> {
    pub generic_ty: &'a IndexMap<GenericTypeParameter, Type>,
    pub generic_value: &'a IndexMap<GenericValueParameter, Value>,
    pub module_port: &'a IndexMap<ModulePort, Value>,
}

impl GenericContainer for TypeOrValue {
    type Result = TypeOrValue;

    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
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

    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
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

    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
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

    fn replace_generics(&self, compiled: &mut CompiledDatabasePartial, map: &GenericMap) -> Self::Result {
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

        let ModulePortInfo { defining_item, defining_id, direction, kind } = &compiled[port];
        if &kind_new != kind {
            let port_new = compiled.module_ports.push(ModulePortInfo {
                defining_item: *defining_item,
                defining_id: defining_id.clone(),
                direction: *direction,
                kind: kind_new,
            });
            port_new
        } else {
            port
        }
    }
}