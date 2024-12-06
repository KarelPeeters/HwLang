use crate::new::compile::{Constant, Parameter, Port, Register, Variable, Wire};
use crate::new::function::FunctionValue;
use crate::new::ir::{IrExpression, IrModule};
use crate::new::types::{IntRange, Type};
use num_bigint::{BigInt, BigUint};

// TODO rename
#[derive(Debug, Clone)]
pub enum MaybeCompile<T> {
    Compile(CompileValue),
    // TODO rename
    Other(T),
}

#[derive(Debug, Copy, Clone)]
pub enum NamedValue {
    Constant(Constant),
    Parameter(Parameter),
    Variable(Variable),

    Port(Port),
    Wire(Wire),
    Register(Register),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum CompileValue {
    Undefined,
    Type(Type),

    Bool(bool),
    Int(BigInt),
    String(String),
    Array(Vec<CompileValue>),
    IntRange(IntRange),
    Module(IrModule),
    Function(FunctionValue),
    // TODO list, tuple, struct, function, module (once we allow passing modules as generics)
}

#[derive(Debug, Clone)]
pub enum AssignmentTarget {
    Port(Port),
    Wire(Wire),
    Register(Register),
    Variable(Variable),
}

#[derive(Debug, Clone)]
pub enum HardwareValueResult {
    Success(IrExpression),
    Undefined,
    PartiallyUndefined,
    Unrepresentable,
}

impl CompileValue {
    pub fn ty(&self) -> Type {
        match self {
            CompileValue::Undefined => Type::Undefined,
            CompileValue::Type(_) => Type::Type,
            CompileValue::Bool(_) => Type::Bool,
            CompileValue::Int(value) => Type::Int(IntRange {
                start_inc: Some(value.clone()),
                end_inc: Some(value.clone()),
            }),
            CompileValue::String(_) => Type::String,
            CompileValue::Array(values) => {
                let inner = values.iter()
                    .fold(Type::Undefined, |acc, v| acc.union(&v.ty()));
                Type::Array(Box::new(inner), BigUint::from(values.len()))
            }
            CompileValue::IntRange(_) => Type::Range,
            CompileValue::Module(_) => Type::Module,
            CompileValue::Function(_) => Type::Function,
        }
    }

    pub fn as_hardware_value(&self) -> HardwareValueResult {
        match self {
            CompileValue::Undefined => HardwareValueResult::Undefined,
            &CompileValue::Bool(value) => HardwareValueResult::Success(IrExpression::Bool(value)),
            CompileValue::Int(value) => HardwareValueResult::Success(IrExpression::Int(value.clone())),
            CompileValue::Array(values) => {
                let mut hardware_values = vec![];
                let mut all_undefined = true;
                let mut any_undefined = false;

                for value in values {
                    match value.as_hardware_value() {
                        HardwareValueResult::Unrepresentable => return HardwareValueResult::Unrepresentable,
                        HardwareValueResult::Success(v) => {
                            all_undefined = false;
                            hardware_values.push(v)
                        },
                        HardwareValueResult::Undefined => {
                            any_undefined = true;
                        }
                        HardwareValueResult::PartiallyUndefined => return HardwareValueResult::PartiallyUndefined,
                    }
                }

                match (any_undefined, all_undefined) {
                    (true, true) => HardwareValueResult::Undefined,
                    (true, false) => HardwareValueResult::PartiallyUndefined,
                    (false, false) => {
                        assert_eq!(hardware_values.len(), values.len());
                        HardwareValueResult::Success(IrExpression::Array(hardware_values))
                    }
                    (false, true) => {
                        assert!(hardware_values.is_empty());
                        HardwareValueResult::Success(IrExpression::Array(vec![]))
                    }
                }
            }
            CompileValue::Type(_) |
            CompileValue::String(_) | CompileValue::IntRange(_) |
            CompileValue::Module(_) | CompileValue::Function(_) => HardwareValueResult::Unrepresentable,
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        match self {
            CompileValue::Undefined => "undefined".to_string(),
            CompileValue::Type(ty) => ty.to_diagnostic_string(),
            CompileValue::Bool(value) => value.to_string(),
            CompileValue::Int(value) => value.to_string(),
            CompileValue::String(value) => value.clone(),
            CompileValue::Array(values) => {
                let values = values.iter()
                    .map(|value| value.to_diagnostic_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", values)
            }
            CompileValue::IntRange(range) => format!("({})", range),
            // TODO module item name and generic args?
            CompileValue::Module(_) => "module".to_string(),
            CompileValue::Function(_) => "function".to_string(),
        }
    }
}
