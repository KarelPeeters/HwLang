use crate::front::common::{ItemReference, TypeOrValue};
use crate::front::param::{GenericContainer, GenericParameterUniqueId, GenericValueParameter, ModulePortUniqueId, ValueParameter};
use crate::front::types::ModuleTypeInfo;
use crate::syntax::ast::BinaryOp;
use indexmap::IndexMap;
use num_bigint::BigInt;

// TODO should all values have types? or can eg. ints just be free abstract objects?
// TODO during compilation, have a "value" wrapper that lazily computes the content and type to break up cycles
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Value {
    // parameters
    // TODO replace all of these with just their unique ID, this is not the place to store type info
    GenericParameter(GenericValueParameter),
    FunctionParameter(ValueParameter),
    ModulePort(ModulePortUniqueId),

    // basic
    Int(BigInt),
    // TODO long-term this should become a standard struct instead of compiler magic
    Range(ValueRangeInfo),
    // TODO this BinaryOp should probably be separate from the ast one
    Binary(BinaryOp, Box<Value>, Box<Value>),

    // structures
    Function(FunctionValue),
    Module(ModuleValueInfo),
    // Struct(StructValue),
    // Tuple(TupleValue),
    // Enum(EnumValue),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ValueRangeInfo {
    pub start: Option<Box<Value>>,
    pub end: Option<Box<Value>>,
    pub end_inclusive: bool,
}

// TODO reduce this to just the defining item and generic args
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionValue {
    // only this field is used in hash and eq
    // TODO also include captured values once those exist
    pub item_reference: ItemReference,

    // pub ty: Type,
    // pub params: Vec<Identifier>,
    // pub body: FunctionBody,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleValueInfo {
    pub ty: ModuleTypeInfo,
    // TODO body
}

impl ValueRangeInfo {
    pub fn new(start: Option<Box<Value>>, end: Option<Box<Value>>, end_inclusive: bool) -> Self {
        let result = Self { start, end, end_inclusive };
        result.assert_valid();
        result
    }

    pub fn unbounded() -> Self {
        Self {
            start: None,
            end: None,
            end_inclusive: false,
        }
    }

    pub fn assert_valid(&self) {
        if self.end.is_none() {
            assert!(!self.end_inclusive);
        }
        // TODO typecheck?
    }
}

impl GenericContainer for Value {
    fn replace_generic_params(&self, map: &IndexMap<GenericParameterUniqueId, TypeOrValue>) -> Self {
        match *self {
            Value::GenericParameter(ref value) => match map.get(&value.unique_id) {
                None => Value::GenericParameter(value.clone()),
                Some(new) => new.as_ref().unwrap_value().clone(),
            }
            Value::FunctionParameter(_) => todo!(),
            Value::ModulePort(_) => todo!(),

            Value::Int(_) => todo!(),
            Value::Range(_) => todo!(),
            Value::Binary(op, ref left, ref right) => {
                Value::Binary(
                    op,
                    Box::new(left.replace_generic_params(map)),
                    Box::new(right.replace_generic_params(map)),
                )
            }

            Value::Function(_) => todo!(),
            Value::Module(_) => todo!(),
        }
    }
}