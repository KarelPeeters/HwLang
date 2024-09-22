use crate::data::compiled::{CompiledDatabasePartial, FunctionParameter, FunctionTypeParameter, FunctionTypeParameterInfo, FunctionValueParameter, FunctionValueParameterInfo, GenericTypeParameter, GenericTypeParameterInfo, GenericValueParameter, GenericValueParameterInfo, Item, ModulePort, ModulePortInfo};
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::syntax::ast::{PortKind, SyncDomain, SyncKind};
use indexmap::IndexMap;
// TODO move all common stuff to the data module, since multiple stages depend on it

// TODO pick a better name for this
// TODO is this still necessary? can't items also be sorted into types or values immediately? 
#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(Item),
    Direct(ScopedEntryDirect),
}

// TODO transpose or not?
// TODO find a better name for this
pub type ScopedEntryDirect = MaybeConstructor<TypeOrValue>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValue<T = Type, V = Value> {
    Type(T),
    Value(V),
}

impl<T, V> TypeOrValue<T, V> {
    pub fn as_ref(&self) -> TypeOrValue<&T, &V> {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t),
            TypeOrValue::Value(v) => TypeOrValue::Value(v),
        }
    }

    pub fn unwrap_type(self) -> T {
        match self {
            TypeOrValue::Type(t) => t,
            TypeOrValue::Value(_) => panic!("Expected type, got value"),
        }
    }

    pub fn unwrap_value(self) -> V {
        match self {
            TypeOrValue::Type(_) => panic!("Expected value, got type"),
            TypeOrValue::Value(v) => v,
        }
    }

    pub fn unit(&self) -> TypeOrValue<(), ()> {
        match self {
            TypeOrValue::Type(_) => TypeOrValue::Type(()),
            TypeOrValue::Value(_) => TypeOrValue::Value(()),
        }
    }
}

/// A Monad trait, specifically for replacing generic parameters in a type or value with more concrete arguments.
// TODO replace this a more general "map" trait, where the user can supply their own closure
pub trait GenericContainer {
    type Result;

    /// The implementation can assume the replacement has already been kind- and type-checked.
    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Self::Result;
}

impl GenericContainer for TypeOrValue {
    type Result = TypeOrValue;

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Self {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t.replace_generic_params(compiled, map_ty, map_value)),
            TypeOrValue::Value(v) => TypeOrValue::Value(v.replace_generic_params(compiled, map_ty, map_value)),
        }
    }
}

// TODO creating new generic parameters for replacements a good idea?
//   this breaks parameter identity-ness
impl GenericContainer for GenericTypeParameter {
    type Result = Type;

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        _map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Type {
        let param = *self;

        if let Some(replacement) = map_ty.get(&param) {
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

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Value {
        let param = *self;

        if let Some(replacement) = map_value.get(&param) {
            // replace the entire parameter
            replacement.clone()
        } else {
            // check if bounds of the parameter need to be replaced
            // (for now this doesn't do anything, but this is a compile-type check for added fields, eg. bounds)
            let ty_new = compiled[param].ty.clone().replace_generic_params(compiled, map_ty, map_value);
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

impl GenericContainer for FunctionParameter {
    type Result = FunctionParameter;

    fn replace_generic_params(&self, compiled: &mut CompiledDatabasePartial, map_ty: &IndexMap<GenericTypeParameter, Type>, map_value: &IndexMap<GenericValueParameter, Value>) -> Self::Result {
        match *self {
            FunctionParameter::Type(param) =>
                FunctionParameter::Type(param.replace_generic_params(compiled, map_ty, map_value)),
            FunctionParameter::Value(param) =>
                FunctionParameter::Value(param.replace_generic_params(compiled, map_ty, map_value)),
        }
    }
}

impl GenericContainer for FunctionTypeParameter {
    type Result = FunctionTypeParameter;

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        _map_ty: &IndexMap<GenericTypeParameter, Type>,
        _map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> FunctionTypeParameter {
        let param = *self;

        // check if bounds of the parameter need to be replaced
        // (for now this doesn't do anything, but this is a compile-type check for added fields, eg. bounds)
        let FunctionTypeParameterInfo { defining_item: _, defining_id: _ } = &compiled[param];
        param
    }
}

impl GenericContainer for FunctionValueParameter {
    type Result = FunctionValueParameter;

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> FunctionValueParameter {
        let param = *self;

        // check if bounds of the parameter need to be replaced
        // (for now this doesn't do anything, but this is a compile-type check for added fields, eg. bounds)
        let ty_new = compiled[param].ty.clone().replace_generic_params(compiled, map_ty, map_value);
        let FunctionValueParameterInfo { defining_item, defining_id, ty, ty_span } = &compiled[param];

        if &ty_new != ty {
            let param_new = compiled.function_value_params.push(FunctionValueParameterInfo {
                defining_item: *defining_item,
                defining_id: defining_id.clone(),
                ty: ty_new,
                ty_span: ty_span.clone(),
            });
            param_new
        } else {
            param
        }
    }
}

impl GenericContainer for ModulePort {
    type Result = ModulePort;

    fn replace_generic_params(&self, compiled: &mut CompiledDatabasePartial, map_ty: &IndexMap<GenericTypeParameter, Type>, map_value: &IndexMap<GenericValueParameter, Value>) -> Self::Result {
        let port = *self;

        let kind_new = match compiled[port].kind.clone() {
            PortKind::Clock => PortKind::Clock,
            PortKind::Normal { sync, ty } => PortKind::Normal {
                sync: match sync {
                    SyncKind::Async => SyncKind::Async,
                    SyncKind::Sync(domain) => SyncKind::Sync(SyncDomain {
                        clock: domain.clock.replace_generic_params(compiled, map_ty, map_value),
                        reset: domain.reset.replace_generic_params(compiled, map_ty, map_value),
                    }),
                },
                ty: ty.replace_generic_params(compiled, map_ty, map_value),
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