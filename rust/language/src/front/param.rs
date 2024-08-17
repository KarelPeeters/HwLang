use crate::front::common::ItemReference;
use crate::front::common::TypeOrValue;
use crate::front::types::Type;
use crate::syntax::ast::Identifier;
use crate::syntax::pos::Span;
use derivative::Derivative;
use indexmap::IndexMap;

#[derive(Debug, Clone)]
pub struct GenericParams {
    pub vec: Vec<GenericParameter>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericArgs {
    pub vec: Vec<TypeOrValue>
}

#[derive(Debug, Clone)]
pub enum GenericParameter {
    Type(GenericTypeParameter),
    Value(GenericValueParameter),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct GenericParameterUniqueId {
    pub defining_item: ItemReference,
    pub param_index: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ModulePortUniqueId {
    pub defining_item: ItemReference,
    pub param_index: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct FunctionParameterUniqueId {
    pub defining_item: ItemReference,
    pub param_index: usize,
}

// TODO keep only the unique id, store other info as shared auxiliary somewhere
#[derive(Debug, Clone)]
#[derive(Derivative)]
#[derivative(Eq, PartialEq, Hash)]
pub struct GenericTypeParameter {
    pub unique_id: GenericParameterUniqueId,

    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub id: Identifier,

    // TODO add constraints
}

// TODO keep only the unique id, store other info as shared auxiliary somewhere
#[derive(Debug, Clone)]
#[derive(Derivative)]
#[derivative(Eq, PartialEq, Hash)]
pub struct GenericValueParameter {
    pub unique_id: GenericParameterUniqueId,

    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub id: Identifier,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub ty: Type,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub ty_span: Span
}

// TODO keep only the unique id, store other info as shared auxiliary somewhere
#[derive(Debug, Clone)]
#[derive(Derivative)]
#[derivative(Eq, PartialEq, Hash)]
pub struct ValueParameter {
    pub unique_id: FunctionParameterUniqueId,

    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub id: Identifier,
    #[derivative(PartialEq = "ignore")]
    #[derivative(Hash = "ignore")]
    pub ty: Type,
}

impl GenericParameter {
    pub fn unique_id(&self) -> GenericParameterUniqueId {
        match self {
            GenericParameter::Type(param) => param.unique_id,
            GenericParameter::Value(param) => param.unique_id,
        }
    }
}

/// A Monad trait, specifically for replacing generic parameters in a type or value with more concrete arguments.
// TODO replace this a more general "map" trait, where the user can supply their own closure
pub trait GenericContainer {
    /// The implementation can assume the replacement has already been kind- and type-checked.
    fn replace_generic_params(&self, map: &IndexMap<GenericParameterUniqueId, TypeOrValue>) -> Self;
}
