use crate::front::driver::{ItemReference};
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

#[derive(Debug, Clone)]
pub struct GenericTypeParameter {
    pub defining_item: ItemReference,
    pub id: Identifier,
    // TODO add constraints
}

// TODO unify generics and other parameters?
//  in practise this mostly just simplifies the language and AST and keeps the IR code the same
#[derive(Debug, Clone)]
pub struct GenericValueParameter {
    pub defining_item: ItemReference,
    pub id: Identifier,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct ValueParameter {
    pub defining_item: ItemReference,
    pub id: Identifier,
    pub ty: Type,
}