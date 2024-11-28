use crate::new::compile::{Port, Register, Variable, Wire};
use crate::new::function::FunctionValue;
use crate::new::ir::IrModule;
use crate::new::misc::ValueDomain;
use crate::new::types::{IntRange, Type};
use num_bigint::BigInt;

#[derive(Debug, Clone)]
pub enum ExpressionValue {
    Compile(CompileValue),

    // TODO unique SSA ID for type constraining based on ifs?
    Port(Port),
    Wire(Wire),
    Register(Register),
    // TODO separate compile-time and runtime variables in syntax?
    //   maybe rename runtime variables to be wires?
    Variable(Variable),

    // TODO how to represent read/write? should write targets just be a special case distinct from all other expression evaluation?
    RuntimeExpression {
        ty: Type,
        domain: ValueDomain,
    },
}

#[derive(Debug, Clone)]
pub enum ScopedValue {
    Compile(CompileValue),
    Port(Port),
    Wire(Wire),
    Register(Register),
    Variable(Variable),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum CompileValue {
    Type(Type),

    // TODO undefined should not be allowed for compile-time values, for register initialization
    Undefined,
    Bool(bool),
    Int(BigInt),
    String(String),
    Array(Vec<CompileValue>),
    IntRange(IntRange),
    Module(IrModule),
    Function(FunctionValue),
    // TODO list, tuple, struct, function, module (once we allow passing modules as generics)
}

impl ExpressionValue {
    pub fn from_scoped(scoped: ScopedValue) -> Self {
        match scoped {
            ScopedValue::Compile(value) => ExpressionValue::Compile(value),
            ScopedValue::Port(port) => ExpressionValue::Port(port),
            ScopedValue::Wire(wire) => ExpressionValue::Wire(wire),
            ScopedValue::Register(register) => ExpressionValue::Register(register),
            ScopedValue::Variable(variable) => ExpressionValue::Variable(variable),
        }
    }
}

impl CompileValue {
    pub fn to_diagnostic_string(&self) -> String {
        match self {
            CompileValue::Type(ty) => ty.to_diagnostic_string(),
            CompileValue::Undefined => "undefined".to_string(),
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
