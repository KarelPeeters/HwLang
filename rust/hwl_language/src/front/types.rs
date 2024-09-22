use crate::data::compiled::{CompiledDatabasePartial, FunctionParameter, FunctionTypeParameter, GenericParameter, GenericTypeParameter, GenericValueParameter, Item, ModulePort};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::common::GenericContainer;
use crate::front::common::TypeOrValue;
use crate::front::values::Value;
use derivative::Derivative;
use indexmap::IndexMap;

#[derive(Debug, Clone)]
pub enum MaybeConstructor<T> {
    Constructor(Constructor<T>),
    Immediate(T),
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionParameters {
    pub vec: Vec<FunctionParameter>,
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
    FunctionParameter(FunctionTypeParameter),

    // basic
    Any,
    // TODO make unit just a special case of tuple
    Unit,
    Boolean,
    Bits(Option<Box<Value>>),
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct IntegerTypeInfo {
    pub range: Box<Value>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionTypeInfo {
    pub params: FunctionParameters,
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

// TODO should modules be structural types instead? or are interfaces already the structural variant of modules?
//  the end use case would be passing a module constructor as a parameter to another module
#[derive(Debug, Clone)]
#[derive(Derivative)]
#[derivative(Eq, PartialEq, Hash)]
pub struct ModuleTypeInfo {
    pub nominal_type_unique: NominalTypeUnique,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub ports: Vec<ModulePort>,
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

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Self {
        match *self {
            Type::Error(e) => Type::Error(e),

            Type::GenericParameter(param) =>
                param.replace_generic_params(compiled, map_ty, map_value),
            Type::FunctionParameter(param) =>
                Type::FunctionParameter(param.replace_generic_params(compiled, map_ty, map_value)),

            Type::Any => Type::Any,
            Type::Unit => Type::Unit,
            Type::Boolean => Type::Boolean,
            Type::Bits(ref width) => {
                Type::Bits(width.as_ref().map(|width| Box::new(width.replace_generic_params(compiled, map_ty, map_value))))
            }
            Type::Array(ref inner, ref len) => {
                Type::Array(
                    Box::new(inner.replace_generic_params(compiled, map_ty, map_value)),
                    Box::new(len.replace_generic_params(compiled, map_ty, map_value)),
                )
            }
            Type::Range => Type::Range,
            Type::Integer(ref info) => {
                Type::Integer(IntegerTypeInfo {
                    range: Box::new(info.range.replace_generic_params(compiled, map_ty, map_value)),
                })
            }
            Type::Function(ref info) => {
                Type::Function(FunctionTypeInfo {
                    params: FunctionParameters {
                        vec: info.params.vec.iter()
                            .map(|p| p.replace_generic_params(compiled, map_ty, map_value))
                            .collect(),
                    },
                    ret: Box::new(info.ret.replace_generic_params(compiled, map_ty, map_value)),
                })
            }
            Type::Tuple(ref types) => {
                Type::Tuple(types.iter().map(|t| t.replace_generic_params(compiled, map_ty, map_value)).collect())
            }
            Type::Struct(ref info) => {
                Type::Struct(StructTypeInfo {
                    nominal_type_unique: info.nominal_type_unique.replace_generic_params(compiled, map_ty, map_value),
                    fields: info.fields.iter()
                        .map(|(name, ty)| (name.clone(), ty.replace_generic_params(compiled, map_ty, map_value)))
                        .collect(),
                })
            }
            Type::Enum(ref info) => {
                Type::Enum(EnumTypeInfo {
                    nominal_type_unique: info.nominal_type_unique.replace_generic_params(compiled, map_ty, map_value),
                    variants: info.variants.iter()
                        .map(|(name, ty)| {
                            (name.clone(), ty.as_ref().map(|t| t.replace_generic_params(compiled, map_ty, map_value)))
                        })
                        .collect(),
                })
            }
            Type::Module(_) => todo!(),
        }
    }
}

impl GenericContainer for NominalTypeUnique {
    type Result = NominalTypeUnique;

    fn replace_generic_params(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map_ty: &IndexMap<GenericTypeParameter, Type>,
        map_value: &IndexMap<GenericValueParameter, Value>,
    ) -> Self {
        NominalTypeUnique {
            item: self.item,
            args: GenericArguments {
                vec: self.args.vec.iter().map(|t| t.replace_generic_params(compiled, map_ty, map_value)).collect(),
            },
        }
    }
}
