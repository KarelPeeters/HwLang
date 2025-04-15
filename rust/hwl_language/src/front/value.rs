use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::domain::ValueDomain;
use crate::front::function::FunctionValue;
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::mid::ir::{IrArrayLiteralElement, IrExpression, IrExpressionLarge, IrLargeArena, IrVariable};
use crate::syntax::parsed::AstRefModule;
use crate::syntax::pos::Span;
use crate::util::big_int::{BigInt, BigUint};
use itertools::enumerate;
use std::convert::identity;

// TODO just provide both as default args, by now it's pretty clear that this uses HardwareValue almost always
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Value<C = CompileValue, T = HardwareValue> {
    Compile(C),
    Hardware(T),
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HardwareValue<T = HardwareType, E = IrExpression> {
    pub ty: T,
    pub domain: ValueDomain,
    pub expr: E,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum MaybeUndefined<T> {
    Undefined,
    Defined(T),
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

    pub fn try_as_hardware_value(&self, large: &mut IrLargeArena, ty: &HardwareType) -> HardwareValueResult {
        fn map_array<'t, E>(
            large: &mut IrLargeArena,
            values: &[CompileValue],
            t: impl Fn(usize) -> &'t HardwareType,
            e: impl Fn(IrExpression) -> E,
            f: impl FnOnce(Vec<E>) -> IrExpressionLarge,
        ) -> HardwareValueResult {
            let mut hardware_values = vec![];
            let mut all_undefined = true;
            let mut any_undefined = false;

            for (i, value) in enumerate(values) {
                match value.try_as_hardware_value(large, t(i)) {
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
                    HardwareValueResult::Success(large.push_expr(f(hardware_values)))
                }
                (false, true) => {
                    assert!(hardware_values.is_empty());
                    HardwareValueResult::Success(large.push_expr(f(vec![])))
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
                        large.push_expr(IrExpressionLarge::ExpandIntRange(ty.clone(), expr))
                    };
                    HardwareValueResult::Success(expr)
                }
                _ => HardwareValueResult::InvalidType,
            },
            CompileValue::Tuple(values) => match ty {
                HardwareType::Tuple(tys) if tys.len() == values.len() => {
                    map_array(large, values, |i| &tys[i], identity, IrExpressionLarge::TupleLiteral)
                }
                _ => HardwareValueResult::InvalidType,
            },
            CompileValue::Array(values) => match ty {
                HardwareType::Array(inner_ty, len) if len == &BigUint::from(values.len()) => map_array(
                    large,
                    values,
                    |_i| inner_ty,
                    IrArrayLiteralElement::Single,
                    |e| IrExpressionLarge::ArrayLiteral(inner_ty.as_ir(), len.clone(), e),
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

    pub fn as_hardware_value_or_undefined(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<MaybeUndefined<IrExpression>, ErrorGuaranteed> {
        match self.try_as_hardware_value(large, ty_hw) {
            HardwareValueResult::Success(v) => Ok(MaybeUndefined::Defined(v)),
            HardwareValueResult::Undefined => Ok(MaybeUndefined::Undefined),
            HardwareValueResult::PartiallyUndefined => Err(diags.report_todo(span, "partially undefined values")),
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

    pub fn as_ir_expression_or_undefined(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<MaybeUndefined<HardwareValue>, ErrorGuaranteed> {
        let hw_value = self.as_hardware_value_or_undefined(diags, large, span, ty_hw)?;
        let typed_expr = hw_value.map_inner(|expr| HardwareValue {
            ty: ty_hw.clone(),
            domain: ValueDomain::CompileTime,
            expr,
        });
        Ok(typed_expr)
    }

    pub fn as_ir_expression(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<HardwareValue, ErrorGuaranteed> {
        match self.as_ir_expression_or_undefined(diags, large, span, ty_hw)? {
            MaybeUndefined::Defined(ir_expr) => Ok(ir_expr),
            MaybeUndefined::Undefined => Err(diags.report_simple(
                "undefined values are not allowed here",
                span,
                format!("undefined value `{}` here", self.to_diagnostic_string()),
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
            CompileValue::String(value) => format!("{:?}", value),
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

impl<E> Typed for HardwareValue<HardwareType, E> {
    fn ty(&self) -> Type {
        self.ty.as_type()
    }
}

impl<T> HardwareValue<T, IrVariable> {
    pub fn to_general_expression(self) -> HardwareValue<T, IrExpression> {
        HardwareValue {
            ty: self.ty,
            domain: self.domain,
            expr: IrExpression::Variable(self.expr),
        }
    }
}

impl HardwareValue {
    pub fn soft_expand_to_type(self, large: &mut IrLargeArena, ty: &HardwareType) -> Self {
        match (&self.ty, ty) {
            (HardwareType::Int(expr_ty), HardwareType::Int(target)) => {
                if target != expr_ty && target.contains_range(expr_ty) {
                    HardwareValue {
                        ty: HardwareType::Int(target.clone()),
                        domain: self.domain,
                        expr: large.push_expr(IrExpressionLarge::ExpandIntRange(target.clone(), self.expr)),
                    }
                } else {
                    self
                }
            }
            _ => self,
        }
    }
}

impl<C, T> Value<C, T> {
    pub fn unwrap_compile(self) -> C {
        match self {
            Value::Compile(v) => v,
            Value::Hardware(_) => panic!("expected compile value"),
        }
    }

    pub fn as_ref(&self) -> Value<&C, &T> {
        match self {
            Value::Compile(v) => Value::Compile(v),
            Value::Hardware(v) => Value::Hardware(v),
        }
    }

    pub fn try_map_other<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Value<C, U>, E> {
        match self {
            Value::Compile(v) => Ok(Value::Compile(v)),
            Value::Hardware(v) => Ok(Value::Hardware(f(v)?)),
        }
    }
}

impl<T, E> Value<CompileValue, HardwareValue<T, E>> {
    pub fn domain(&self) -> &ValueDomain {
        match self {
            Value::Compile(_) => &ValueDomain::CompileTime,
            Value::Hardware(value) => &value.domain,
        }
    }
}

impl Value {
    /// Turn this expression into an [IrExpression].
    ///
    /// Also tries to expand the expression to the given type, but this can fail.
    /// It it still the responsibility of the caller to typecheck the resulting expression.
    pub fn as_ir_expression(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> Result<HardwareValue, ErrorGuaranteed> {
        match self {
            Value::Compile(v) => v.as_ir_expression(diags, large, span, ty),
            Value::Hardware(v) => Ok(v.clone().soft_expand_to_type(large, ty)),
        }
    }
}

impl Value<CompileValue, HardwareValue<HardwareType, IrExpressionLarge>> {
    pub fn to_maybe_compile(self, large: &mut IrLargeArena) -> Value {
        match self {
            Value::Compile(v) => Value::Compile(v),
            Value::Hardware(v) => Value::Hardware(HardwareValue {
                ty: v.ty,
                domain: v.domain,
                expr: large.push_expr(v.expr),
            }),
        }
    }
}

impl Value<BigInt, HardwareValue<ClosedIncRange<BigInt>>> {
    pub fn range(&self) -> ClosedIncRange<&BigInt> {
        match self {
            Value::Compile(v) => ClosedIncRange::single(v),
            Value::Hardware(v) => v.ty.as_ref(),
        }
    }
}

impl<C: Typed, T: Typed> Typed for Value<C, T> {
    fn ty(&self) -> Type {
        match self {
            Value::Compile(v) => v.ty(),
            Value::Hardware(v) => v.ty(),
        }
    }
}

impl<T> From<CompileValue> for Value<CompileValue, T> {
    fn from(value: CompileValue) -> Self {
        Value::Compile(value)
    }
}

impl<T> MaybeUndefined<T> {
    pub fn map_inner<U>(self, f: impl FnOnce(T) -> U) -> MaybeUndefined<U> {
        match self {
            MaybeUndefined::Undefined => MaybeUndefined::Undefined,
            MaybeUndefined::Defined(v) => MaybeUndefined::Defined(f(v)),
        }
    }
}
