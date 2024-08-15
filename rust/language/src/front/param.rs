use indexmap::IndexMap;

use crate::front::driver::ItemReference;
use crate::front::TypeOrValue;
use crate::front::types::Type;
use crate::syntax::ast::Identifier;

#[derive(Debug, Clone)]
pub struct GenericParams {
    pub vec: Vec<GenericParameter>,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct GenericTypeParameter {
    pub unique_id: GenericParameterUniqueId,
    pub id: Identifier,
    // TODO add constraints
}

// TODO unify generics and other parameters?
//  in practise this mostly just simplifies the language and AST and keeps the IR code the same
// TODO remove non-clone fields, they're a bit misplaced here anyway, it will just become a typevar
#[derive(Debug, Clone)]
pub struct GenericValueParameter {
    pub unique_id: GenericParameterUniqueId,
    pub id: Identifier,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct ValueParameter {
    pub defining_item: ItemReference,
    pub id: Identifier,
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
