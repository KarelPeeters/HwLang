use indexmap::IndexMap;

use crate::front::driver::ItemReference;
use crate::front::param::{GenericArgs, GenericContainer, GenericParameterUniqueId, GenericParams, GenericTypeParameter};
use crate::front::values::Value;
use crate::front::TypeOrValue;
use crate::syntax::ast::{PortDirection, PortKind, SyncKind};

#[derive(Debug, Clone)]
pub enum MaybeConstructor<T> {
    Constructor(Constructor<T>),
    Immediate(T),
}

#[derive(Debug, Clone)]
pub struct Constructor<T> {
    pub parameters: GenericParams,
    pub inner: T,
}

// TODO push type constructor args into struct, enum, ... types, like Rust?
#[derive(Debug, Clone)]
pub enum Type {
    Generic(GenericTypeParameter),

    Boolean,
    Bits(Box<Value>),
    // TODO range of what inner type? and how do int ranges with mixed types work exactly?
    Range,
    Array(Box<Type>, Box<Value>),
    Integer(IntegerTypeInfo),
    Function(FunctionTypeInfo),
    Tuple(Vec<Type>),
    Struct(StructTypeInfo),
    Enum(EnumTypeInfo),
    Module(ModuleTypeInfo),
}

#[derive(Debug, Clone)]
pub struct IntegerTypeInfo {
    pub range: Box<Value>,
}

#[derive(Debug, Clone)]
pub struct FunctionTypeInfo {
    pub params: Vec<Type>,
    pub ret: Box<Type>,
}

/// Two nominal types are considered equal iff their unique ids are equal.
/// This is a stronger requirement than just requiring the type structures to be the same in two ways:
/// * Both types need to be defined by the same exact item
/// * The generic parameters need to match.
/// This is an additional requirement if some of the parameters don't effect the structure of the type.  
#[derive(Debug, Clone)]
pub struct NominalTypeUnique {
    pub item_reference: ItemReference,
    pub args: GenericArgs,
}

#[derive(Debug, Clone)]
pub struct StructTypeInfo {
    pub nominal_type_unique: NominalTypeUnique,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone)]
pub struct EnumTypeInfo {
    pub nominal_type_unique: NominalTypeUnique,
    pub variants: Vec<(String, Option<Type>)>,
}

// TODO should modules be structural types instead? or are interfaces already the structural variant of modules?
//  the end use case would be passing a module constructor as a parameter to another module
#[derive(Debug, Clone)]
pub struct ModuleTypeInfo {
    pub item_reference: ItemReference,
    pub ports: Vec<(String, PortTypeInfo)>,
}

#[derive(Debug, Clone)]
pub struct PortTypeInfo {
    pub direction: PortDirection,
    pub kind: PortKind<SyncKind<usize>, Type>,
}

impl<T> MaybeConstructor<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> MaybeConstructor<U> {
        match self {
            MaybeConstructor::Constructor(c) => MaybeConstructor::Constructor(Constructor {
                parameters: c.parameters,
                inner: f(c.inner),
            }),
            MaybeConstructor::Immediate(t) => MaybeConstructor::Immediate(f(t)),
        }
    }
}

impl GenericContainer for Type {
    fn replace_generic_params(&self, map: &IndexMap<GenericParameterUniqueId, TypeOrValue<Type, Value>>) -> Self {
        match *self {
            Type::Generic(ref generic) => match map.get(&generic.unique_id) {
                None => Type::Generic(generic.clone()),
                Some(new) => new.as_ref().unwrap_type().clone(),
            },
            Type::Boolean => Type::Boolean,
            Type::Bits(ref width) => Type::Bits(Box::new(width.replace_generic_params(map))),
            Type::Array(ref inner, ref len) => {
                Type::Array(
                    Box::new(inner.replace_generic_params(map)),
                    Box::new(len.replace_generic_params(map)),
                )
            }
            Type::Range => Type::Range,
            Type::Integer(ref info) => {
                Type::Integer(IntegerTypeInfo {
                    range: Box::new(info.range.replace_generic_params(map)),
                })
            }
            Type::Function(ref info) => {
                Type::Function(FunctionTypeInfo {
                    params: info.params.iter().map(|t| t.replace_generic_params(map)).collect(),
                    ret: Box::new(info.ret.replace_generic_params(map)),
                })
            }
            Type::Tuple(ref types) => {
                Type::Tuple(types.iter().map(|t| t.replace_generic_params(map)).collect())
            },
            Type::Struct(ref info) => {
                Type::Struct(StructTypeInfo {
                    nominal_type_unique: info.nominal_type_unique.replace_generic_params(map),
                    fields: info.fields.iter()
                        .map(|(name, ty)| (name.clone(), ty.replace_generic_params(map)))
                        .collect(),
                })
            },
            Type::Enum(ref info) => {
                Type::Enum(EnumTypeInfo {
                    nominal_type_unique: info.nominal_type_unique.replace_generic_params(map),
                    variants: info.variants.iter()
                        .map(|(name, ty)| {
                            (name.clone(), ty.as_ref().map(|t| t.replace_generic_params(map)))
                        })
                        .collect(),
                })
            },
            Type::Module(_) => todo!(),
        }
    }
}

impl GenericContainer for NominalTypeUnique {
    fn replace_generic_params(&self, map: &IndexMap<GenericParameterUniqueId, TypeOrValue<Type, Value>>) -> Self {
        NominalTypeUnique {
            item_reference: self.item_reference.clone(),
            args: GenericArgs {
                vec: self.args.vec.iter().map(|t| t.replace_generic_params(map)).collect(),
            },
        }
    }
}