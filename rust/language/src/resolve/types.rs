use num_bigint::BigInt;
use crate::new_index_type;
use crate::syntax::pos::FileId;
use crate::util::arena::ArenaSet;

new_index_type!(pub Type);

pub struct Types {
    arena: ArenaSet<Type, TypeInfo>,
    ty_type: Type,
    ty_void: Type,
    ty_int: Type,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeInfo {
    Type,
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInfoStruct {
    // TODO fields
    // TODO exclude everything except the reference from hash and eq?
    pub item_reference: ItemReference,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeInfoEnum {
    // TODO options
    // TODO exclude everything except the reference from hash and eq?
    pub item_reference: ItemReference,
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
        Self {
            ty_type: arena.push(TypeInfo::Type),
            ty_void: arena.push(TypeInfo::Tuple(vec![])),
            ty_int: arena.push(TypeInfo::Integer(TypeInfoInteger { min: None, max: None })),
            arena,
        }
    }
}

impl Types {
    pub fn push(&mut self, info: TypeInfo) -> Type {
        self.arena.push(info)
    }

    pub fn ty_type(&self) -> Type {
        self.ty_type
    }

    pub fn ty_void(&self) -> Type {
        self.ty_void
    }

    pub fn ty_int(&self) -> Type {
        self.ty_int
    }
}