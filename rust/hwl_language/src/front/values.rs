use crate::data::compiled::{CompiledDatabase, CompiledStage, GenericValueParameter, Item, ModulePort, Register, Variable};
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
    BoolConstant(bool),
    IntConstant(BigInt),
    StringConstant(String),
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
    pub const UNBOUNDED: RangeInfo<V> = RangeInfo {
        start: None,
        end: None,
    };

    pub fn map_inner<U>(self, mut f: impl FnMut(V) -> U) -> RangeInfo<U> {
        RangeInfo {
            start: self.start.map(&mut f),
            end: self.end.map(&mut f),
        }
    }
}

impl GenericContainer for Value {
    type Result = Value;

    fn replace_generics<S: CompiledStage>(
        &self,
        compiled: &mut CompiledDatabase<S>,
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

            Value::BoolConstant(b) => Value::BoolConstant(b),
            Value::IntConstant(ref info) => Value::IntConstant(info.clone()),
            Value::StringConstant(ref info) => Value::StringConstant(info.clone()),
            Value::Range(ref info) => Value::Range(RangeInfo {
                start: info.start.as_ref()
                    .map(|v| Box::new(v.replace_generics(compiled, map))),
                end: info.end.as_ref()
                    .map(|v| Box::new(v.replace_generics(compiled, map))),
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

    fn replace_generics<S: CompiledStage>(&self, compiled: &mut CompiledDatabase<S>, map: &GenericMap) -> Self::Result {
        ModuleValueInfo {
            nominal_type_unique: self.nominal_type_unique.clone(),
            ports: self.ports.iter().map(|p| p.replace_generics(compiled, map)).collect_vec(),
        }
    }
}