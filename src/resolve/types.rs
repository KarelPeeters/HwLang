use num_bigint::BigInt;
use crate::new_index_type;
use crate::util::arena::ArenaSet;

new_index_type!(pub Type);

pub type TypeArena = ArenaSet<Type, TypeInfo>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeInfo {
    Void, // TODO the same as the empty tuple?
    Integer(Option<TypeInteger>),
    Function(TypeFunction),
    Tuple(Vec<Type>),
    Module(TypeModule),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInteger {
    pub min: Option<BigInt>,
    pub max: Option<BigInt>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeFunction {
    pub params: Vec<Type>,
    pub ret: Box<Type>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeModule {
    // TODO
}