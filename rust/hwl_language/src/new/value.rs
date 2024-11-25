use crate::data::diagnostic::ErrorGuaranteed;
use crate::impl_from_error_guaranteed;
use crate::new::compile::{Port, Register, Variable, Wire};
use crate::new::ir::IrModule;
use crate::new::misc::ValueDomain;
use crate::new::types::{IntRange, Type};
use num_bigint::BigInt;

pub enum ExpressionValue {
    Error(ErrorGuaranteed),
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

impl_from_error_guaranteed!(ExpressionValue);

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
    Undefined,
    Bool(bool),
    Int(BigInt),
    String(String),
    Array(Vec<CompileValue>),
    IntRange(IntRange),
    Module(IrModule),
    // TODO list, tuple, struct, function, module (once we allow passing modules as generics)
}

impl CompileValue {
    pub fn to_diagnostic_string(&self) -> String {
        match self {
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
        }
    }
}
