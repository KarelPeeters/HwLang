use crate::front::block::TypedIrExpression;
use crate::front::compile::{CompileState, Constant, Parameter, Port, Register, Variable, Wire};
use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::function::FunctionValue;
use crate::front::ir::IrExpression;
use crate::front::misc::ValueDomain;
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::syntax::ast::{ArrayLiteralElement, Spanned};
use crate::syntax::parsed::AstRefModule;
use crate::syntax::pos::Span;
use itertools::enumerate;
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use std::convert::identity;

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

#[derive(Debug)]
pub enum HardwareValueResult {
    Success(IrExpression),
    Undefined,
    PartiallyUndefined,
    Unrepresentable,
    InvalidType,
}

#[derive(Debug, Copy, Clone)]
pub enum HardwareReason {
    AssignmentToPort(Span),
    AssignmentToWire(Span),
    AssignmentToRegister(Span),

    InstancePortConnection(Span),
    RuntimeArrayIndexing(Span),

    IfMerge { span_stmt: Span },

    TupleWithOtherHardwareValue { span_tuple: Span, span_other_value: Span },
}

impl AssignmentTarget {
    pub fn ty(&self, state: &CompileState) -> Option<Spanned<Type>> {
        match *self {
            AssignmentTarget::Port(port) => Some(state.ports[port].ty.as_ref().map_inner(|ty| ty.as_type())),
            AssignmentTarget::Wire(wire) => Some(state.wires[wire].ty.as_ref().map_inner(|ty| ty.as_type())),
            AssignmentTarget::Register(register) => {
                Some(state.registers[register].ty.as_ref().map_inner(|ty| ty.as_type()))
            }
            AssignmentTarget::Variable(variable) => state.variables[variable].ty.clone(),
        }
    }
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
                let inner = values.iter().fold(Type::Undefined, |acc, v| acc.union(&v.ty(), true));
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

    pub fn as_hardware_value(&self, ty: &HardwareType) -> HardwareValueResult {
        fn map_array<'t, E>(
            values: &[CompileValue],
            t: impl Fn(usize) -> &'t HardwareType,
            e: impl Fn(IrExpression) -> E,
            f: impl FnOnce(Vec<E>) -> IrExpression,
        ) -> HardwareValueResult {
            let mut hardware_values = vec![];
            let mut all_undefined = true;
            let mut any_undefined = false;

            for (i, value) in enumerate(values) {
                match value.as_hardware_value(t(i)) {
                    HardwareValueResult::Unrepresentable => return HardwareValueResult::Unrepresentable,
                    HardwareValueResult::Success(v) => {
                        all_undefined = false;
                        hardware_values.push(e(v))
                    }
                    HardwareValueResult::Undefined => {
                        any_undefined = true;
                    }
                    HardwareValueResult::PartiallyUndefined => return HardwareValueResult::PartiallyUndefined,
                    HardwareValueResult::InvalidType => return HardwareValueResult::InvalidType,
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
            CompileValue::Int(value) => match ty {
                HardwareType::Int(ty) if ty.contains(value) => {
                    let expr = IrExpression::Int(value.clone());
                    let expr = if ty.start_inc == ty.end_inc {
                        expr
                    } else {
                        IrExpression::ExpandIntRange(ty.clone(), Box::new(expr))
                    };
                    HardwareValueResult::Success(expr)
                }
                _ => HardwareValueResult::InvalidType,
            },
            CompileValue::Tuple(values) => match ty {
                HardwareType::Tuple(tys) if tys.len() == values.len() => {
                    map_array(values, |i| &tys[i], identity, IrExpression::TupleLiteral)
                }
                _ => HardwareValueResult::InvalidType,
            },
            CompileValue::Array(values) => match ty {
                HardwareType::Array(inner_ty, len) if len.to_usize() == Some(values.len()) => map_array(
                    values,
                    |_i| inner_ty,
                    |e| ArrayLiteralElement { spread: None, value: e },
                    |e| IrExpression::ArrayLiteral(inner_ty.to_ir(), e),
                ),
                _ => HardwareValueResult::InvalidType,
            },
            CompileValue::Type(_)
            | CompileValue::String(_)
            | CompileValue::IntRange(_)
            | CompileValue::Module(_)
            | CompileValue::Function(_) => HardwareValueResult::Unrepresentable,
        }
    }

    pub fn as_ir_expression(
        &self,
        diags: &Diagnostics,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<TypedIrExpression, ErrorGuaranteed> {
        match self.as_hardware_value(ty_hw) {
            HardwareValueResult::Success(v) => Ok(TypedIrExpression {
                ty: ty_hw.clone(),
                domain: ValueDomain::CompileTime,
                expr: v,
            }),
            HardwareValueResult::Undefined | HardwareValueResult::PartiallyUndefined => Err(diags.report_simple(
                "undefined can only be used for register initialization",
                span,
                "value is undefined",
            )),
            HardwareValueResult::Unrepresentable => Err(diags.report_internal_error(
                span,
                format!(
                    "value `{}` has hardware type but is itself not representable",
                    self.to_diagnostic_string()
                ),
            )),
            // type checking should have already happened before this, so this is an internal error
            HardwareValueResult::InvalidType => Err(diags.report_internal_error(
                span,
                format!(
                    "value `{}` with type `{}` cannot be represented as hardware type `{}`",
                    self.to_diagnostic_string(),
                    self.ty().to_diagnostic_string(),
                    ty_hw.as_type().to_diagnostic_string()
                ),
            )),
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

impl<T, E> MaybeCompile<TypedIrExpression<T, E>> {
    pub fn domain(&self) -> &ValueDomain {
        match self {
            MaybeCompile::Compile(_) => &ValueDomain::CompileTime,
            MaybeCompile::Other(value) => &value.domain,
        }
    }
}

impl MaybeCompile<TypedIrExpression> {
    /// Turn this expression into an [IrExpression].
    ///
    /// Also tries to expand the expression to the given type, but this can fail.
    /// It it still the responsibility of the caller to typecheck the resulting expression.
    pub fn as_ir_expression(
        &self,
        diags: &Diagnostics,
        span: Span,
        ty: &HardwareType,
    ) -> Result<TypedIrExpression, ErrorGuaranteed> {
        match self {
            MaybeCompile::Compile(v) => v.as_ir_expression(diags, span, ty),
            MaybeCompile::Other(v) => expand_expression_to_type(v.clone(), ty),
        }
    }
}

fn expand_expression_to_type(
    expr: TypedIrExpression,
    target: &HardwareType,
) -> Result<TypedIrExpression, ErrorGuaranteed> {
    let result = match (&expr.ty, target) {
        (HardwareType::Int(expr_ty), HardwareType::Int(target)) => {
            if target != expr_ty && target.contains_range(expr_ty) {
                TypedIrExpression {
                    ty: HardwareType::Int(target.clone()),
                    domain: expr.domain,
                    expr: IrExpression::ExpandIntRange(target.clone(), Box::new(expr.expr)),
                }
            } else {
                expr
            }
        }
        _ => expr,
    };

    Ok(result)
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
