use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::domain::ValueDomain;
use crate::front::function::FunctionValue;
use crate::front::item::{ElaboratedEnum, ElaboratedInterface, ElaboratedModule, ElaboratedStruct};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::mid::ir::{IrArrayLiteralElement, IrExpression, IrExpressionLarge, IrLargeArena, IrType, IrVariable};
use crate::syntax::pos::Span;
use crate::util::big_int::{BigInt, BigUint};
use itertools::{enumerate, Itertools};
use std::convert::identity;
use std::sync::Arc;

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
    String(Arc<String>),

    IntRange(IncRange<BigInt>),
    Tuple(Arc<Vec<CompileValue>>),
    Array(Arc<Vec<CompileValue>>),
    // TODO avoid storing copy of field types
    Struct(ElaboratedStruct, Vec<Type>, Vec<CompileValue>),
    Enum(ElaboratedEnum, Vec<Option<Type>>, (usize, Option<Box<CompileValue>>)),

    Function(FunctionValue),
    Module(ElaboratedModule),
    Interface(ElaboratedInterface),
    InterfaceView(ElaboratedInterfaceView),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedInterfaceView {
    pub interface: ElaboratedInterface,
    pub view: String,
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
    Defined(IrExpression),
    Undefined,
    PartiallyUndefined,
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
            CompileValue::Tuple(values) => Type::Tuple(Arc::new(values.iter().map(|v| v.ty()).collect())),
            CompileValue::Array(values) => {
                let inner = values.iter().fold(Type::Undefined, |acc, v| acc.union(&v.ty(), true));
                Type::Array(Arc::new(inner), BigUint::from(values.len()))
            }
            CompileValue::Struct(item, fields, _) => Type::Struct(*item, fields.clone()),
            CompileValue::Enum(item, types, _) => Type::Enum(*item, types.clone()),
            CompileValue::IntRange(_) => Type::Range,
            CompileValue::Function(_) => Type::Function,
            CompileValue::Module(_) => Type::Module,
            CompileValue::Interface(_) => Type::Interface,
            CompileValue::InterfaceView(_) => Type::InterfaceView,
        }
    }
}

impl CompileValue {
    pub fn unit() -> Self {
        // Empty tuples are considered types, since all their inner values are types.
        CompileValue::Type(Type::unit())
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, CompileValue::Type(ty) if ty.is_unit())
    }

    pub fn contains_undefined(&self) -> bool {
        match self {
            CompileValue::Undefined => true,
            CompileValue::Tuple(values) => values.iter().any(CompileValue::contains_undefined),
            CompileValue::Array(values) => values.iter().any(CompileValue::contains_undefined),
            CompileValue::Struct(_, _, values) => values.iter().any(CompileValue::contains_undefined),
            CompileValue::Enum(_, _, (_, value)) => value.as_ref().is_some_and(|v| v.contains_undefined()),
            CompileValue::Type(_)
            | CompileValue::Bool(_)
            | CompileValue::Int(_)
            | CompileValue::String(_)
            | CompileValue::IntRange(_)
            | CompileValue::Function(_)
            | CompileValue::Module(_)
            | CompileValue::Interface(_)
            | CompileValue::InterfaceView(_) => false,
        }
    }

    pub fn try_as_hardware_value(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> Result<HardwareValueResult, ErrorGuaranteed> {
        fn map_array<'t, E>(
            diags: &Diagnostics,
            large: &mut IrLargeArena,
            span: Span,
            values: &[CompileValue],
            t: impl Fn(usize) -> &'t HardwareType,
            e: impl Fn(IrExpression) -> E,
            f: impl FnOnce(Vec<E>) -> IrExpressionLarge,
        ) -> Result<HardwareValueResult, ErrorGuaranteed> {
            let mut hardware_values = Vec::with_capacity(values.len());
            let mut all_undefined = true;
            let mut any_undefined = false;

            for (i, value) in enumerate(values) {
                match value.try_as_hardware_value(diags, large, span, t(i))? {
                    HardwareValueResult::Defined(v) => {
                        all_undefined = false;
                        hardware_values.push(e(v))
                    }
                    HardwareValueResult::Undefined => {
                        any_undefined = true;
                    }
                    HardwareValueResult::PartiallyUndefined => return Ok(HardwareValueResult::PartiallyUndefined),
                }
            }

            let result = match (any_undefined, all_undefined) {
                (true, true) => HardwareValueResult::Undefined,
                (true, false) => HardwareValueResult::PartiallyUndefined,
                (false, false) => {
                    assert_eq!(hardware_values.len(), values.len());
                    HardwareValueResult::Defined(large.push_expr(f(hardware_values)))
                }
                (false, true) => {
                    assert!(hardware_values.is_empty());
                    HardwareValueResult::Defined(large.push_expr(f(vec![])))
                }
            };
            Ok(result)
        }

        let err_type = || diags.report_internal_error(span, "type mismatch while converting to hardware value");

        match self {
            CompileValue::Undefined => Ok(HardwareValueResult::Undefined),
            &CompileValue::Bool(value) => Ok(HardwareValueResult::Defined(IrExpression::Bool(value))),
            CompileValue::Int(value) => match ty {
                HardwareType::Int(ty) if ty.contains(value) => {
                    let expr = IrExpression::Int(value.clone());
                    let expr = if ty.start_inc == ty.end_inc {
                        expr
                    } else {
                        large.push_expr(IrExpressionLarge::ExpandIntRange(ty.clone(), expr))
                    };
                    Ok(HardwareValueResult::Defined(expr))
                }
                _ => Err(err_type()),
            },
            CompileValue::Tuple(values) => match ty {
                HardwareType::Tuple(tys) if tys.len() == values.len() => map_array(
                    diags,
                    large,
                    span,
                    values,
                    |i| &tys[i],
                    identity,
                    IrExpressionLarge::TupleLiteral,
                ),
                _ => Err(err_type()),
            },
            CompileValue::Array(values) => match ty {
                HardwareType::Array(inner_ty, len) if len == &BigUint::from(values.len()) => map_array(
                    diags,
                    large,
                    span,
                    values,
                    |_i| inner_ty,
                    IrArrayLiteralElement::Single,
                    |e| IrExpressionLarge::ArrayLiteral(inner_ty.as_ir(), len.clone(), e),
                ),
                _ => Err(err_type()),
            },
            CompileValue::Struct(item_value, _, values) => match ty {
                HardwareType::Struct(item, fields) if item == item_value => {
                    assert_eq!(fields.len(), values.len());
                    map_array(
                        diags,
                        large,
                        span,
                        values,
                        |i| &fields[i],
                        identity,
                        IrExpressionLarge::TupleLiteral,
                    )
                }
                _ => Err(err_type()),
            },
            &CompileValue::Enum(item_value, _, (variant_index, ref content_value)) => match ty {
                HardwareType::Enum(hw_enum) if hw_enum.elab == item_value => {
                    // convert content to bits and then ir expression
                    let content_ir = content_value
                        .as_ref()
                        .map(|content_value| {
                            let content_ty = hw_enum.variants[variant_index].as_ref().unwrap();
                            let content_bits = content_ty.value_to_bits(diags, span, content_value)?;

                            let content_elements = content_bits
                                .into_iter()
                                .map(|b| IrArrayLiteralElement::Single(IrExpression::Bool(b)))
                                .collect_vec();
                            let content_ir =
                                IrExpressionLarge::ArrayLiteral(IrType::Bool, content_ty.size_bits(), content_elements);
                            Ok(large.push_expr(content_ir))
                        })
                        .transpose()?;

                    // build the entire ir expression
                    let result = hw_enum.build_ir_expression(large, variant_index, content_ir)?;
                    Ok(HardwareValueResult::Defined(result))
                }
                _ => Err(err_type()),
            },
            CompileValue::Type(_)
            | CompileValue::String(_)
            | CompileValue::IntRange(_)
            | CompileValue::Function(_)
            | CompileValue::Module(_)
            | CompileValue::Interface(_)
            | CompileValue::InterfaceView(_) => {
                Err(diags.report_internal_error(span, "this value kind is not representable as hardware"))
            }
        }
    }

    pub fn as_ir_expression_or_undefined(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<MaybeUndefined<IrExpression>, ErrorGuaranteed> {
        match self.try_as_hardware_value(diags, large, span, ty_hw)? {
            HardwareValueResult::Defined(v) => Ok(MaybeUndefined::Defined(v)),
            HardwareValueResult::Undefined => Ok(MaybeUndefined::Undefined),
            HardwareValueResult::PartiallyUndefined => Err(diags.report_todo(span, "partially undefined values")),
        }
    }

    pub fn as_hardware_value_or_undefined(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<MaybeUndefined<HardwareValue>, ErrorGuaranteed> {
        let hw_value = self.as_ir_expression_or_undefined(diags, large, span, ty_hw)?;
        let typed_expr = hw_value.map_inner(|expr| HardwareValue {
            ty: ty_hw.clone(),
            domain: ValueDomain::CompileTime,
            expr,
        });
        Ok(typed_expr)
    }

    pub fn as_hardware_value(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty_hw: &HardwareType,
    ) -> Result<HardwareValue, ErrorGuaranteed> {
        match self.as_hardware_value_or_undefined(diags, large, span, ty_hw)? {
            MaybeUndefined::Defined(ir_expr) => Ok(ir_expr),
            MaybeUndefined::Undefined => Err(diags.report_simple(
                "undefined values are not allowed here",
                span,
                format!("undefined value `{}` here", self.diagnostic_string()),
            )),
        }
    }

    pub fn diagnostic_string(&self) -> String {
        // TODO avoid printing diagnostics strings that are very long (eg. large strings, arrays, structs, ...)
        match self {
            CompileValue::Undefined => "undefined".to_string(),
            CompileValue::Type(ty) => ty.diagnostic_string(),
            CompileValue::Bool(value) => value.to_string(),
            CompileValue::Int(value) => value.to_string(),
            CompileValue::String(value) => format!("{:?}", value),
            CompileValue::Tuple(values) => {
                let values = values
                    .iter()
                    .map(|value| value.diagnostic_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({})", values)
            }
            CompileValue::Array(values) => {
                let values = values
                    .iter()
                    .map(|value| value.diagnostic_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", values)
            }
            CompileValue::Struct(_, _, values) => {
                let values = values.iter().map(CompileValue::diagnostic_string).join(", ");
                format!("struct({})", values)
            }
            CompileValue::Enum(_, _, (index, value)) => match value {
                None => format!("enum({})", index),
                Some(value) => format!("enum({}, {})", index, value.diagnostic_string()),
            },
            CompileValue::IntRange(range) => format!("({})", range),
            // TODO include item name and generic args
            CompileValue::Function(_) => "function".to_string(),
            CompileValue::Module(_) => "module".to_string(),
            CompileValue::Interface(_) => "interface".to_string(),
            CompileValue::InterfaceView(_) => "interface_view".to_string(),
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
    #[track_caller]
    pub fn unwrap_compile(self) -> C {
        match self {
            Value::Compile(v) => v,
            Value::Hardware(_) => panic!("expected compile value"),
        }
    }

    #[track_caller]
    pub fn unwrap_hardware(self) -> T {
        match self {
            Value::Compile(_) => panic!("expected hardware value"),
            Value::Hardware(v) => v,
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

    pub fn map_compile<U>(self, f: impl FnOnce(C) -> U) -> Value<U, T> {
        match self {
            Value::Compile(v) => Value::Compile(f(v)),
            Value::Hardware(v) => Value::Hardware(v),
        }
    }

    pub fn map_hardware<U>(self, f: impl FnOnce(T) -> U) -> Value<C, U> {
        match self {
            Value::Compile(v) => Value::Compile(v),
            Value::Hardware(v) => Value::Hardware(f(v)),
        }
    }
}

impl<C: Clone, T: Clone> Value<&C, &T> {
    pub fn cloned(self) -> Value<C, T> {
        match self {
            Value::Compile(v) => Value::Compile(v.clone()),
            Value::Hardware(v) => Value::Hardware(v.clone()),
        }
    }
}

impl<T, E> Value<CompileValue, HardwareValue<T, E>> {
    pub fn domain(&self) -> ValueDomain {
        match self {
            Value::Compile(_) => ValueDomain::CompileTime,
            Value::Hardware(value) => value.domain,
        }
    }
}

impl Value {
    /// Turn this expression into an [IrExpression].
    ///
    /// Also tries to expand the expression to the given type, but this can fail.
    /// It it still the responsibility of the caller to typecheck the resulting expression.
    pub fn as_hardware_value(
        &self,
        diags: &Diagnostics,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> Result<HardwareValue, ErrorGuaranteed> {
        match self {
            Value::Compile(v) => v.as_hardware_value(diags, large, span, ty),
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
