use crate::data::compiled::{FunctionParameter, GenericTypeParameter, GenericValueParameter, Item, ModulePort};
use crate::front::common::GenericContainer;
use crate::front::types::{ModuleTypeInfo, Type};
use crate::syntax::ast::BinaryOp;
use indexmap::IndexMap;
use num_bigint::BigInt;

// TODO should all values have types? or can eg. ints just be free abstract objects?
// TODO during compilation, have a "value" wrapper that lazily computes the content and type to break up cycles
// TODO should all values (and types) have (optional) origin spans for easier error messages?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Value {
    // parameters
    GenericParameter(GenericValueParameter),
    FunctionParameter(FunctionParameter),
    ModulePort(ModulePort),

    // basic
    Int(BigInt),
    // TODO long-term this should become a standard struct instead of compiler magic
    Range(RangeInfo<Box<Value>>),
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
pub struct RangeInfo<V> {
    pub start: Option<V>,
    pub end: Option<V>,
    pub end_inclusive: bool,
}

// TODO reduce this to just the defining item and generic args
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionValue {
    // only this field is used in hash and eq
    // TODO also include captured values once those exist
    pub item: Item,

    // pub ty: Type,
    // pub params: Vec<Identifier>,
    // pub body: FunctionBody,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleValueInfo {
    pub ty: ModuleTypeInfo,
    // TODO body
}

impl<V> RangeInfo<V> {
    pub fn new(start: Option<V>, end: Option<V>, end_inclusive: bool) -> Self {
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
        // TODO typecheck and assert that start <= end?
    }
}

impl GenericContainer for Value {
    fn replace_generic_params(&self, map_ty: &IndexMap<GenericTypeParameter, Type>, map_value: &IndexMap<GenericValueParameter, Value>) -> Self {
        match *self {
            Value::GenericParameter(param) =>
                map_value.get(&param).cloned().unwrap_or(Value::GenericParameter(param)),
            Value::FunctionParameter(unique_id) => Value::FunctionParameter(unique_id),
            Value::ModulePort(unique_id) => Value::ModulePort(unique_id),

            Value::Int(_) => todo!(),
            Value::Range(_) => todo!(),
            Value::Binary(op, ref left, ref right) => {
                Value::Binary(
                    op,
                    Box::new(left.replace_generic_params(map_ty, map_value)),
                    Box::new(right.replace_generic_params(map_ty, map_value)),
                )
            }

            Value::Function(_) => todo!(),
            Value::Module(_) => todo!(),
        }
    }
}