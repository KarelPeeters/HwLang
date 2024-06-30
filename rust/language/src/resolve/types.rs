use std::hash::{Hash, Hasher};
use num_bigint::BigInt;
use num_traits::identities::Zero;

use crate::new_index_type;
use crate::resolve::values::Value;
use crate::syntax::pos::FileId;
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
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeInfo {
    Type,
    Boolean,
    Integer(TypeInfoInteger),
    Function(TypeInfoFunction),
    Tuple(Vec<Type>),
    Struct(TypeInfoStruct),
    Enum(TypeInfoEnum),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInfoInteger {
    pub min: Option<BigInt>,
    pub max: Option<BigInt>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInfoFunction {
    pub params: Vec<Type>,
    pub ret: Type,
}

#[derive(Debug, Clone, Eq)]
pub struct TypeInfoStruct {
    // only the reference and params are included in hash and eq
    pub item_reference: ItemReference,
    pub params: Vec<Value>,
    
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, Eq)]
pub struct TypeInfoEnum {
    // only the reference and params are included in hash and eq
    pub item_reference: ItemReference,
    pub params: Vec<Value>,

    pub variants: Vec<(String, Option<Type>)>,
}

/// Utility type to refer to a specific item in a specific file.
/// Used to deduplicate [nominative types](https://en.wikipedia.org/wiki/Nominal_type_system) like structs or enums.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ItemReference {
    pub file: FileId,
    pub item_index: usize,
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
        }
    }
}

impl Hash for TypeInfoStruct {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.item_reference, &self.params).hash(state);
    }
}

impl PartialEq for TypeInfoStruct {
    fn eq(&self, other: &Self) -> bool {
        (self.item_reference, &self.params) == (other.item_reference, &other.params)
    }
}

impl Hash for TypeInfoEnum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.item_reference, &self.params).hash(state);
    }
}

impl PartialEq for TypeInfoEnum {
    fn eq(&self, other: &Self) -> bool {
        (self.item_reference, &self.params) == (other.item_reference, &other.params)
    }
}