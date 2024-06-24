use num_bigint::BigInt;
use crate::new_index_type;
use crate::syntax::pos::FileId;
use crate::util::arena::ArenaSet;

new_index_type!(pub Type);

pub struct Types {
    arena: ArenaSet<Type, TypeInfo>,
    ty_void: Type,
    ty_int: Type,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeInfo {
    Integer(TypeInteger),
    Function(TypeFunction),
    Tuple(Vec<Type>),
    Struct(TypeStruct),
    Enum(TypeEnum),
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
pub struct TypeStruct {
    // TODO fields
    // TODO exclude everything except the reference from hash and eq?
    pub item_reference: ItemReference,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct TypeEnum {
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
            ty_void: arena.push(TypeInfo::Tuple(vec![])),
            ty_int: arena.push(TypeInfo::Integer(TypeInteger { min: None, max: None })),
            arena,
        }
    }
}

impl Types {
    pub fn push(&mut self, info: TypeInfo) -> Type {
        self.arena.push(info)
    }

    pub fn ty_void(&self) -> Type {
        self.ty_void
    }

    pub fn ty_int(&self) -> Type {
        self.ty_int
    }
}