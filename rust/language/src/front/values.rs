use indexmap::IndexMap;
use num_bigint::BigInt;

use crate::front::driver::{FunctionBody, ItemReference};
use crate::front::param::{GenericContainer, GenericParameterUniqueId, GenericValueParameter, ValueParameter};
use crate::front::TypeOrValue;
use crate::front::types::Type;
use crate::syntax::ast::Identifier;

// TODO should all values have types? or can eg. ints just be free abstract objects?
// TODO during compilation, have a "value" wrapper that lazily computes the content and type to break up cycles
#[derive(Debug, Clone)]
pub enum Value {
    Generic(GenericValueParameter),
    Parameter(ValueParameter),
    
    Int(BigInt),
    Function(FunctionValue),
    Module(ModuleValue),
    // Struct(StructValue),
    // Tuple(TupleValue),
    // Enum(EnumValue),

    // TODO should this be a dedicated type or just an instance of the normal range struct?
    Range(ValueRangeInfo),
}

#[derive(Debug, Clone)]
pub struct ValueRangeInfo {
    pub start: Option<Box<Value>>,
    pub end: Option<Box<Value>>,
    pub end_inclusive: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionValue {
    // only this field is used in hash and eq
    // TODO also include captured values once those exist
    pub item_reference: ItemReference,
    
    pub ty: Type,
    pub params: Vec<Identifier>,
    pub body: FunctionBody,
}

#[derive(Debug, Clone)]
pub struct ModuleValue {
    // TODO include real content?
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
            Value::Generic(ref value) => match map.get(&value.unique_id) {
                None => Value::Generic(value.clone()),
                Some(new) => new.as_ref().unwrap_value().clone(),
            }
            Value::Parameter(_) => todo!(),
            Value::Int(_) => todo!(),
            Value::Function(_) => todo!(),
            Value::Module(_) => todo!(),
            Value::Range(_) => todo!(),
        }
    }
}