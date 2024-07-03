use std::hash::{Hash, Hasher};

use num_bigint::{BigInt, BigUint};
use num_traits::identities::Zero;

use crate::new_index_type;
use crate::resolve::compile::ItemReference;
use crate::resolve::values::Value;
use crate::syntax::ast::{PortDirection, PortKind, SyncKind};
use crate::util::arena::ArenaSet;

new_index_type!(pub Type);

pub struct Types {
    arena: ArenaSet<Type, TypeInfo>,
    basic: BasicTypes<Type>,
}

#[derive(Debug)]
pub struct BasicTypes<T> {
    pub ty_type: T,
    pub ty_void: T,
    pub ty_bool: T,
    pub ty_int: T,
    pub ty_uint: T,
    pub ty_range: T,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeInfo {
    Type,
    Boolean,
    Range,
    Bits(BigUint),
    Integer(TypeInfoInteger),
    Function(TypeInfoFunction),
    Tuple(Vec<Type>),
    Struct(TypeInfoStruct),
    Enum(TypeInfoEnum),
    Module(TypeInfoModule),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInfoInteger {
    // TODO assert min <= max
    pub min: Option<BigInt>,
    pub max: Option<BigInt>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInfoFunction {
    pub params: Vec<Type>,
    pub ret: Type,
}

#[derive(Debug, Clone)]
pub struct TypeInfoStruct {
    pub unique: TypeUnique,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone)]
pub struct TypeInfoEnum {
    pub unique: TypeUnique,
    // TODO refer to identifiers or nothing here instead?
    pub variants: Vec<(String, Option<Type>)>,
}

// TODO should modules be structural types instead? or are interfaces already the structural variant of modules?
//  the end use case would be passing a module constructor as a parameter to another module
#[derive(Debug, Clone)]
pub struct TypeInfoModule {
    pub unique: TypeUnique,
    pub ports: Vec<(String, PortTypeInfo)>,
}

#[derive(Debug, Clone)]
pub struct PortTypeInfo {
    pub direction: PortDirection,
    pub kind: PortKind<SyncKind<usize>, Type>,
}

/// Used to deduplicate [nominative types](https://en.wikipedia.org/wiki/Nominal_type_system) like structs or enums.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeUnique {
    pub item_reference: ItemReference,
    pub params: Vec<Value>,
}

impl Default for Types {
    fn default() -> Self {
        let mut arena = ArenaSet::default();

        let basic = BasicTypes {
            ty_type: arena.push(TypeInfo::Type),
            ty_void: arena.push(TypeInfo::Tuple(vec![])),
            ty_bool: arena.push(TypeInfo::Boolean),
            ty_int: arena.push(TypeInfo::Integer(TypeInfoInteger { min: None, max: None })),
            ty_uint: arena.push(TypeInfo::Integer(TypeInfoInteger { min: Some(BigInt::zero()), max: None })),
            ty_range: arena.push(TypeInfo::Range),
        };

        Types {
            arena,
            basic,
        }
    }
}

impl Types {
    pub fn push(&mut self, info: TypeInfo) -> Type {
        self.arena.push(info)
    }

    pub fn basic(&self) -> &BasicTypes<Type> {
        // this is an accessor function to prevent mutable access to the inner types
        &self.basic
    }
}

impl<T> BasicTypes<T> {
    pub fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> BasicTypes<U> {
        BasicTypes {
            ty_type: f(&self.ty_type),
            ty_void: f(&self.ty_void),
            ty_bool: f(&self.ty_bool),
            ty_int: f(&self.ty_int),
            ty_uint: f(&self.ty_uint),
            ty_range: f(&self.ty_range),
        }
    }
}

impl Hash for TypeInfoStruct {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique.hash(state);
    }
}

impl PartialEq for TypeInfoStruct {
    fn eq(&self, other: &Self) -> bool {
        &self.unique == &other.unique
    }
}

impl Eq for TypeInfoStruct {}

impl Hash for TypeInfoEnum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique.hash(state);
    }
}

impl PartialEq for TypeInfoEnum {
    fn eq(&self, other: &Self) -> bool {
        &self.unique == &other.unique
    }
}

impl Eq for TypeInfoEnum {}

impl Hash for TypeInfoModule {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique.hash(state);
    }
}

impl PartialEq for TypeInfoModule {
    fn eq(&self, other: &Self) -> bool {
        &self.unique == &other.unique
    }
}

impl Eq for TypeInfoModule {}
