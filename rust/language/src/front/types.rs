use indexmap::IndexMap;

use crate::front::driver::ItemReference;
use crate::front::param::{GenericArgs, GenericContainer, GenericParameterUniqueId, GenericParams, GenericTypeParameter};
use crate::front::TypeOrValue;
use crate::front::values::Value;
use crate::syntax::ast::{PortDirection, PortKind, SyncKind};

#[derive(Debug, Clone)]
pub enum MaybeConstructor<T> {
    Constructor(Generic<T>),
    Immediate(T),
}

// TODO remove? this is just what the struct/enum thing already is
#[derive(Debug, Clone)]
pub struct Generic<T> {
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

#[derive(Debug, Clone)]
pub struct StructTypeInfo {
    pub generic_struct: Generic<StructTypeInfoInner>,
    pub args: GenericArgs,
}

#[derive(Debug, Clone)]
pub struct StructTypeInfoInner {
    pub item_reference: ItemReference,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone)]
pub struct EnumTypeInfo {
    pub generic_enum: Generic<EnumTypeInfoInner>,
    pub args: GenericArgs,
}

#[derive(Debug, Clone)]
pub struct EnumTypeInfoInner {
    pub item_reference: ItemReference,
    // TODO use identifier instead of string?
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
            MaybeConstructor::Constructor(c) => MaybeConstructor::Constructor(Generic {
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
            Type::Tuple(ref types) => Type::Tuple(types.iter().map(|t| t.replace_generic_params(map)).collect()),
            // TODO carefully think about this: once we allow defining local structs that use scoped generic parameters,
            //   we need to replace them deep inside the fields.
            //   Do we also want to keep nominal type-ness? How exactly does that interact with the above?
            Type::Struct(_) => todo!(),
            Type::Enum(_) => todo!(),
            Type::Module(_) => todo!(),
        }
    }
}