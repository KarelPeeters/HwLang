use crate::data::compiled::{CompiledDatabasePartial, GenericParameter, GenericTypeParameter, Item};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::common::TypeOrValue;
use crate::front::common::{GenericContainer, GenericMap};
use crate::front::values::Value;
use derivative::Derivative;
use indexmap::IndexMap;

// TODO find a better name for this
#[derive(Debug, Clone)]
pub enum MaybeConstructor<T> {
    Immediate(T),
    Constructor(Constructor<T>),
    /// This error case means we don't know whether this is a constructor or not.
    Error(ErrorGuaranteed),
}

#[derive(Debug, Clone)]
pub struct Constructor<T> {
    pub parameters: GenericParameters,
    pub inner: T,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericParameters {
    pub vec: Vec<GenericParameter>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArguments {
    pub vec: Vec<TypeOrValue>,
}

// TODO function arguments?

// TODO push type constructor args into struct, enum, ... types, like Rust?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    // error
    // TODO how should this behave under equality?
    Error(ErrorGuaranteed),

    // parameters
    GenericParameter(GenericTypeParameter),

    // basic
    Any,
    Unchecked,
    Unit,
    // TODO make unit just a special case of tuple
    Never,
    Boolean,
    Clock,
    Bits(Option<Box<Value>>),
    String,
    // TODO range of what inner type? and how do int ranges with mixed types work exactly?
    Range,
    Array(Box<Type>, Box<Value>),
    Integer(IntegerTypeInfo),
    Function(FunctionTypeInfo),
    Tuple(Vec<Type>),
    Struct(StructTypeInfo),
    Enum(EnumTypeInfo),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct IntegerTypeInfo {
    pub range: Box<Value>,
}

/// This is only used for higher-order functions, which are restricted to only take values as parameters.
/// Functions themselves don't really define a type, they define a value constructor.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionTypeInfo {
    pub params: Vec<Type>,
    pub ret: Box<Type>,
}

/// Two nominal types are considered equal iff their unique ids are equal.
/// This is a stronger requirement than just requiring the type structures to be the same in two ways:
/// * Both types need to be defined by the same exact item
/// * The generic parameters need to match.
/// This is an additional requirement if some of the parameters don't effect the structure of the type.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NominalTypeUnique {
    pub item: Item,
    pub args: GenericArguments,
    // TODO think about how captured values and types should work
}

#[derive(Debug, Clone)]
#[derive(Derivative)]
#[derivative(Eq, PartialEq, Hash)]
pub struct StructTypeInfo {
    pub nominal_type_unique: NominalTypeUnique,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub fields: IndexMap<String, Type>,
}

#[derive(Debug, Clone)]
#[derive(Derivative)]
#[derivative(Eq, PartialEq, Hash)]
pub struct EnumTypeInfo {
    pub nominal_type_unique: NominalTypeUnique,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub variants: IndexMap<String, Option<Type>>,
}

impl<T> MaybeConstructor<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> MaybeConstructor<U> {
        match self {
            MaybeConstructor::Constructor(c) => MaybeConstructor::Constructor(Constructor {
                parameters: c.parameters,
                inner: f(c.inner),
            }),
            MaybeConstructor::Immediate(t) => MaybeConstructor::Immediate(f(t)),
            MaybeConstructor::Error(e) => MaybeConstructor::Error(e),
        }
    }
}

impl GenericContainer for Type {
    type Result = Type;

    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map: &GenericMap,
    ) -> Self {
        match *self {
            Type::Error(e) => Type::Error(e),

            Type::GenericParameter(param) =>
                param.replace_generics(compiled, map),

            Type::Any => Type::Any,
            Type::Unchecked => Type::Unchecked,
            Type::Never => Type::Never,
            Type::Unit => Type::Unit,
            Type::Boolean => Type::Boolean,
            Type::Clock => Type::Clock,
            Type::String => Type::String,
            Type::Bits(ref width) => {
                Type::Bits(width.as_ref().map(|width| Box::new(width.replace_generics(compiled, map))))
            }
            Type::Array(ref inner, ref len) => {
                Type::Array(
                    Box::new(inner.replace_generics(compiled, map)),
                    Box::new(len.replace_generics(compiled, map)),
                )
            }
            Type::Range => Type::Range,
            Type::Integer(ref info) => {
                Type::Integer(IntegerTypeInfo {
                    range: Box::new(info.range.replace_generics(compiled, map)),
                })
            }
            Type::Function(ref info) => {
                Type::Function(FunctionTypeInfo {
                    params: info.params.iter()
                        .map(|p| p.replace_generics(compiled, map))
                        .collect(),
                    ret: Box::new(info.ret.replace_generics(compiled, map)),
                })
            }
            Type::Tuple(ref types) => {
                Type::Tuple(types.iter().map(|t| t.replace_generics(compiled, map)).collect())
            }
            Type::Struct(ref info) => {
                Type::Struct(StructTypeInfo {
                    nominal_type_unique: info.nominal_type_unique.replace_generics(compiled, map),
                    fields: info.fields.iter()
                        .map(|(name, ty)| (name.clone(), ty.replace_generics(compiled, map)))
                        .collect(),
                })
            }
            Type::Enum(ref info) => {
                Type::Enum(EnumTypeInfo {
                    nominal_type_unique: info.nominal_type_unique.replace_generics(compiled, map),
                    variants: info.variants.iter()
                        .map(|(name, ty)| {
                            (name.clone(), ty.as_ref().map(|t| t.replace_generics(compiled, map)))
                        })
                        .collect(),
                })
            }
        }
    }
}

impl GenericContainer for NominalTypeUnique {
    type Result = NominalTypeUnique;

    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map: &GenericMap,
    ) -> Self {
        NominalTypeUnique {
            item: self.item,
            args: GenericArguments {
                vec: self.args.vec.iter().map(|t| t.replace_generics(compiled, map)).collect(),
            },
        }
    }
}
