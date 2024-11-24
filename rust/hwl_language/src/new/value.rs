use crate::new::misc::ValueDomain;
use crate::new::types::Type;
use crate::syntax::ast::PortDirection;
use num_bigint::BigInt;

pub enum ExpressionValue {
    Compile(CompileValue),
    // TODO add ports, wires, regs, variables, _including_ a unique SSA id?

    General {
        ty: Type,
        domain: ValueDomain,
    },
}

#[derive(Debug, Clone)]
pub enum ScopedValue {
    Compile(CompileValue),
    // TODO replace ports, vars, signals, regs, ... with arenas?
    Port { direction: PortDirection, domain: ValueDomain, ty: Type },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum CompileValue {
    Bool(bool),
    Int(BigInt),
    String(String),
    Array(Vec<CompileValue>),
    // TODO list, tuple, struct, function, module (once we allow passing modules as generics)
}

impl CompileValue {
    pub fn to_diagnostic_string(&self) -> String {
        match self {
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
        }
    }
}