use crate::data::compiled::{CompiledDatabasePartial, GenericValueParameter, Item, ModulePort, Register, Variable};
use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::common::{GenericContainer, GenericMap};
use crate::front::types::{NominalTypeUnique, Type};
use crate::syntax::ast::BinaryOp;
use itertools::Itertools;
use num_bigint::BigInt;

// TODO should all values have types? or can eg. ints just be free abstract objects?
// TODO during compilation, have a "value" wrapper that lazily computes the content and type to break up cycles
// TODO should all values (and types) have (optional) origin spans for easier error messages?
// TODO Eq impl is a bit suspicious, remove it and replace it by named functions, eg. is_same_value
// TODO attach a span to each value? that kills both interning and compiler-value-constructing though
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Value {
    // error
    Error(ErrorGuaranteed),

    // parameters
    GenericParameter(GenericValueParameter),
    ModulePort(ModulePort),

    // basic
    Never,
    Unit,
    InstConstant(BigInt),
    // TODO long-term this should become a standard struct instead of compiler magic
    Range(RangeInfo<Box<Value>>),
    // TODO this BinaryOp should probably be separate from the ast one
    Binary(BinaryOp, Box<Value>, Box<Value>),
    UnaryNot(Box<Value>),

    // structures
    // TODO functions are represented very strangely, double-check if this makes sense
    FunctionReturn(FunctionReturnValue),
    Module(ModuleValueInfo),
    // Struct(StructValue),
    // Tuple(TupleValue),
    // Enum(EnumValue),

    // variables
    // TODO what should these contain? an id? a type?
    //  how should these behave under generic substitution?
    Wire,
    Register(Register),
    Variable(Variable),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RangeInfo<V> {
    pub start: Option<V>,
    pub end: Option<V>,
    pub end_inclusive: bool,
}

// TODO double check which fields should be used for eq and hash
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionReturnValue {
    pub item: Item,
    pub ret_ty: Type,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ModuleValueInfo {
    // TODO should this be here or not?
    pub nominal_type_unique: NominalTypeUnique,
    pub ports: Vec<ModulePort>,
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
    type Result = Value;

    fn replace_generics(
        &self,
        compiled: &mut CompiledDatabasePartial,
        map: &GenericMap,
    ) -> Value {
        match *self {
            Value::Error(e) => Value::Error(e),

            Value::GenericParameter(param) =>
                param.replace_generics(compiled, map),
            Value::ModulePort(module_port) => {
                match map.module_port.get(&module_port) {
                    Some(value) => value.clone(),
                    None => Value::ModulePort(module_port),
                }
            }

            Value::Unit => Value::Unit,
            Value::Never => Value::Never,

            Value::InstConstant(ref info) => Value::InstConstant(info.clone()),
            Value::Range(ref info) => Value::Range(RangeInfo {
                start: info.start.as_ref()
                    .map(|v| Box::new(v.replace_generics(compiled, map))),
                end: info.end.as_ref()
                    .map(|v| Box::new(v.replace_generics(compiled, map))),
                end_inclusive: info.end_inclusive,
            }),
            Value::Binary(op, ref left, ref right) => {
                Value::Binary(
                    op,
                    Box::new(left.replace_generics(compiled, map)),
                    Box::new(right.replace_generics(compiled, map)),
                )
            }
            Value::UnaryNot(ref inner) =>
                Value::UnaryNot(Box::new(inner.replace_generics(compiled, map))),

            Value::FunctionReturn(ref func) => {
                Value::FunctionReturn(FunctionReturnValue {
                    item: func.item,
                    ret_ty: func.ret_ty.replace_generics(compiled, map),
                })
            }
            Value::Module(ref info) => Value::Module(info.replace_generics(compiled, map)),

            Value::Wire => Value::Wire,
            Value::Register(r) => Value::Register(r),
            Value::Variable(var) => Value::Variable(var),
        }
    }
}

impl GenericContainer for ModuleValueInfo {
    type Result = ModuleValueInfo;

    fn replace_generics(&self, compiled: &mut CompiledDatabasePartial, map: &GenericMap) -> Self::Result {
        ModuleValueInfo {
            nominal_type_unique: self.nominal_type_unique.clone(),
            ports: self.ports.iter().map(|p| p.replace_generics(compiled, map)).collect_vec(),
        }
    }
}