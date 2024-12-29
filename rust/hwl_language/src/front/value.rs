use crate::front::block::TypedIrExpression;
use crate::front::compile::{Constant, Parameter, Port, Register, Variable, Wire};
use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::function::FunctionValue;
use crate::front::ir::IrExpression;
use crate::front::misc::ValueDomain;
use crate::front::types::{ClosedIncRange, IncRange, Type, Typed};
use crate::syntax::parsed::AstRefModule;
use crate::syntax::pos::Span;
use num_bigint::{BigInt, BigUint};

// TODO rename
// TODO just provide both as default args
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MaybeCompile<T, C = CompileValue> {
    Compile(C),
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
    Tuple(Vec<CompileValue>),
    Array(Vec<CompileValue>),
    IntRange(IncRange<BigInt>),
    Module(AstRefModule),
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

impl Typed for CompileValue {
    fn ty(&self) -> Type {
        match self {
            CompileValue::Undefined => Type::Undefined,
            CompileValue::Type(_) => Type::Type,
            CompileValue::Bool(_) => Type::Bool,
            CompileValue::Int(value) => Type::Int(IncRange {
                start_inc: Some(value.clone()),
                end_inc: Some(value.clone()),
            }),
            CompileValue::String(_) => Type::String,
            CompileValue::Tuple(values) => Type::Tuple(values.iter().map(|v| v.ty()).collect()),
            CompileValue::Array(values) => {
                let inner = values.iter().fold(Type::Undefined, |acc, v| acc.union(&v.ty()));
                Type::Array(Box::new(inner), BigUint::from(values.len()))
            }
            CompileValue::IntRange(_) => Type::Range,
            CompileValue::Module(_) => Type::Module,
            CompileValue::Function(_) => Type::Function,
        }
    }
}

impl CompileValue {
    pub const UNIT: CompileValue = CompileValue::Tuple(vec![]);

    pub fn contains_undefined(&self) -> bool {
        match self {
            CompileValue::Undefined => true,
            CompileValue::Tuple(values) => values.iter().any(|v| v.contains_undefined()),
            CompileValue::Array(values) => values.iter().any(|v| v.contains_undefined()),
            CompileValue::Type(_)
            | CompileValue::Bool(_)
            | CompileValue::Int(_)
            | CompileValue::String(_)
            | CompileValue::IntRange(_)
            | CompileValue::Module(_)
            | CompileValue::Function(_) => false,
        }
    }

    pub fn as_hardware_value(&self) -> HardwareValueResult {
        fn map_array(
            values: &[CompileValue],
            f: impl FnOnce(Vec<IrExpression>) -> IrExpression,
        ) -> HardwareValueResult {
            let mut hardware_values = vec![];
            let mut all_undefined = true;
            let mut any_undefined = false;

            for value in values {
                match value.as_hardware_value() {
                    HardwareValueResult::Unrepresentable => return HardwareValueResult::Unrepresentable,
                    HardwareValueResult::Success(v) => {
                        all_undefined = false;
                        hardware_values.push(v)
                    }
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
                    HardwareValueResult::Success(f(hardware_values))
                }
                (false, true) => {
                    assert!(hardware_values.is_empty());
                    HardwareValueResult::Success(f(vec![]))
                }
            }
        }

        match self {
            CompileValue::Undefined => HardwareValueResult::Undefined,
            &CompileValue::Bool(value) => HardwareValueResult::Success(IrExpression::Bool(value)),
            CompileValue::Int(value) => HardwareValueResult::Success(IrExpression::Int(value.clone())),
            CompileValue::Tuple(values) => map_array(&values, IrExpression::TupleLiteral),
            CompileValue::Array(values) => map_array(&values, IrExpression::ArrayLiteral),
            CompileValue::Type(_)
            | CompileValue::String(_)
            | CompileValue::IntRange(_)
            | CompileValue::Module(_)
            | CompileValue::Function(_) => HardwareValueResult::Unrepresentable,
        }
    }

    pub fn to_ir_expression(&self, diags: &Diagnostics, span: Span) -> Result<TypedIrExpression, ErrorGuaranteed> {
        let ty = self.ty();
        let ty_hw = ty.as_hardware_type().ok_or_else(|| {
            diags.report_simple(
                "value type must be representable in hardware",
                span,
                format!(
                    "value `{}` with type `{}` is not representable in hardware",
                    self.to_diagnostic_string(),
                    ty.to_diagnostic_string()
                ),
            )
        })?;

        match self.as_hardware_value() {
            HardwareValueResult::Success(v) => Ok(TypedIrExpression {
                ty: ty_hw,
                domain: ValueDomain::CompileTime,
                expr: v,
            }),
            HardwareValueResult::Undefined | HardwareValueResult::PartiallyUndefined => Err(diags.report_simple(
                "undefined can only be used for register initialization",
                span,
                "value is undefined",
            )),
            HardwareValueResult::Unrepresentable => {
                let reason = "compile time value must be representable in hardware";
                Err(diags.report_simple(
                    reason,
                    span,
                    format!(
                        "value `{}` is not representable in hardware",
                        self.to_diagnostic_string()
                    ),
                ))
            }
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        // TODO avoid printing diagnostics strings that are very long (eg. large strings, arrays, structs, ...)
        match self {
            CompileValue::Undefined => "undefined".to_string(),
            CompileValue::Type(ty) => ty.to_diagnostic_string(),
            CompileValue::Bool(value) => value.to_string(),
            CompileValue::Int(value) => value.to_string(),
            CompileValue::String(value) => value.clone(),
            CompileValue::Tuple(values) => {
                let values = values
                    .iter()
                    .map(|value| value.to_diagnostic_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", values)
            }
            CompileValue::Array(values) => {
                let values = values
                    .iter()
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

impl MaybeCompile<TypedIrExpression> {
    pub fn domain(&self) -> &ValueDomain {
        match self {
            MaybeCompile::Compile(_) => &ValueDomain::CompileTime,
            MaybeCompile::Other(value) => &value.domain,
        }
    }

    pub fn to_ir_expression(&self, diags: &Diagnostics, span: Span) -> Result<TypedIrExpression, ErrorGuaranteed> {
        match self {
            MaybeCompile::Compile(v) => v.to_ir_expression(diags, span),
            MaybeCompile::Other(v) => Ok(v.clone()),
        }
    }
}

impl MaybeCompile<TypedIrExpression<ClosedIncRange<BigInt>>, BigInt> {
    pub fn range(&self) -> ClosedIncRange<&BigInt> {
        match self {
            MaybeCompile::Compile(v) => ClosedIncRange::single(v),
            MaybeCompile::Other(v) => v.ty.as_ref(),
        }
    }
}

impl<T: Typed, C: Typed> Typed for MaybeCompile<T, C> {
    fn ty(&self) -> Type {
        match self {
            MaybeCompile::Compile(v) => v.ty(),
            MaybeCompile::Other(v) => v.ty(),
        }
    }
}

impl<T> From<CompileValue> for MaybeCompile<T, CompileValue> {
    fn from(value: CompileValue) -> Self {
        MaybeCompile::Compile(value)
    }
}
