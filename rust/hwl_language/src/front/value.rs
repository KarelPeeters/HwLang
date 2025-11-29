use crate::front::compile::CompileRefs;
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable};
use crate::front::domain::ValueDomain;
use crate::front::function::FunctionValue;
use crate::front::item::{
    ElaboratedEnum, ElaboratedInterface, ElaboratedInterfaceView, ElaboratedModule, ElaboratedStruct,
};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::mid::ir::{IrArrayLiteralElement, IrExpression, IrExpressionLarge, IrLargeArena};
use crate::syntax::ast::StringPiece;
use crate::syntax::pos::Span;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::data::NonEmptyVec;
use crate::util::iter::IterExt;
use crate::util::{Never, ResultNeverExt};
use itertools::zip_eq;
use std::sync::Arc;

pub type CompileValue = Value<SimpleCompileValue, CompileCompoundValue, Never>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Value<S = SimpleCompileValue, C = MixedCompoundValue, H = HardwareValue> {
    // TODO should simple and hardware share one branch?
    // TODO rename simple to SimpleCompile?
    Simple(S),
    Compound(C),
    Hardware(H),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum SimpleCompileValue {
    Type(Type),
    Bool(bool),
    Int(BigInt),
    Array(Arc<Vec<CompileValue>>),
    Function(FunctionValue),
    Module(ElaboratedModule),
    Interface(ElaboratedInterface),
    InterfaceView(ElaboratedInterfaceView),
}

#[derive(Debug, Clone)]
pub enum MixedCompoundValue {
    String(Arc<Vec<StringPiece<String, HardwareValue>>>),
    Range(
        RangeValue<
            MaybeCompile<BigInt, HardwareValue<ClosedIncRange<BigInt>>>,
            MaybeCompile<BigUint, HardwareValue<ClosedIncRange<BigUint>>>,
        >,
    ),
    Tuple(NonEmptyVec<Value>),
    Struct(StructValue<Value>),
    Enum(EnumValue<Box<Value>>),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum CompileCompoundValue {
    String(Arc<String>),
    Range(RangeValue<BigInt, BigUint>),
    Tuple(NonEmptyVec<CompileValue>),
    Struct(StructValue<CompileValue>),
    Enum(EnumValue<Box<CompileValue>>),
}

/// This type intentionally does not implement Eq/PartialEq, comparing hardware values only makes sense at runtime,
/// not during compilation.
#[derive(Debug, Clone)]
pub struct HardwareValue<T = HardwareType, E = IrExpression> {
    pub ty: T,
    pub domain: ValueDomain,
    pub expr: E,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum RangeValue<B, L> {
    // TODO remove inclusive/exclude distinction, it's messy especially for eq/hash
    //   is that even enough? start/length can also equal the same range!
    StartEnd { start: Option<B>, end: RangeEnd<B> },
    StartLength { start: B, length: L },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum RangeEnd<B> {
    Exclusive(Option<B>),
    Inclusive(B),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StructValue<V> {
    pub ty: ElaboratedStruct,
    pub fields: Vec<V>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct EnumValue<V> {
    pub ty: ElaboratedEnum,
    pub variant: usize,
    pub payload: Option<V>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum MaybeCompile<C, H> {
    Compile(C),
    Hardware(H),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum MaybeUndefined<T> {
    Undefined,
    Defined(T),
}

#[derive(Debug, Copy, Clone)]
pub struct NotCompile;

pub trait ValueCommon {
    /// Convert this value to a hardware value with the exact type `ty`.
    ///
    /// This fails if the value cannot be represented as a hardware value of the given type.
    /// The caller should have checked for type compatibility already, a type mismatch will result in an internal error.
    fn as_hardware_value_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: HardwareType,
    ) -> DiagResult<HardwareValue> {
        // TODO avoid walking the hierarchy twice?
        let domain = self.domain();
        let expr = self.as_ir_expression_unchecked(refs, large, span, &ty)?;
        Ok(HardwareValue { ty, domain, expr })
    }

    fn domain(&self) -> ValueDomain;

    fn as_ir_expression_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> DiagResult<IrExpression>;
}

impl ValueCommon for Never {
    fn domain(&self) -> ValueDomain {
        self.unreachable()
    }
    fn as_ir_expression_unchecked(
        &self,
        _: CompileRefs,
        _: &mut IrLargeArena,
        _: Span,
        _: &HardwareType,
    ) -> DiagResult<IrExpression> {
        self.unreachable()
    }
}

impl<S: ValueCommon, C: ValueCommon, H: ValueCommon> ValueCommon for Value<S, C, H> {
    fn domain(&self) -> ValueDomain {
        match self {
            Value::Simple(v) => v.domain(),
            Value::Compound(v) => v.domain(),
            Value::Hardware(v) => v.domain(),
        }
    }

    fn as_ir_expression_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> DiagResult<IrExpression> {
        match self {
            Value::Simple(v) => v.as_ir_expression_unchecked(refs, large, span, ty),
            Value::Compound(v) => v.as_ir_expression_unchecked(refs, large, span, ty),
            Value::Hardware(v) => v.as_ir_expression_unchecked(refs, large, span, ty),
        }
    }
}

impl ValueCommon for SimpleCompileValue {
    fn domain(&self) -> ValueDomain {
        ValueDomain::CompileTime
    }

    fn as_ir_expression_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> DiagResult<IrExpression> {
        let err_type = || refs.diags.report(err_hw_type_mismatch(span, ty));

        match self {
            SimpleCompileValue::Type(_) => Err(err_type()),
            &SimpleCompileValue::Bool(v) => match ty {
                HardwareType::Bool => Ok(IrExpression::Bool(v)),
                _ => Err(err_type()),
            },
            SimpleCompileValue::Int(v) => match &ty {
                HardwareType::Int(ty) if ty.contains(v) => {
                    let expr = IrExpressionLarge::ExpandIntRange(ty.clone(), IrExpression::Int(v.clone()));
                    Ok(large.push_expr(expr))
                }
                _ => Err(err_type()),
            },
            SimpleCompileValue::Array(v) => match &ty {
                HardwareType::Array(e_ty, len) if &BigUint::from(v.len()) == len => {
                    // TODO if all values are the same, replace with repeat?
                    let result = v
                        .iter()
                        .map(|e| {
                            e.as_ir_expression_unchecked(refs, large, span, e_ty)
                                .map(IrArrayLiteralElement::Single)
                        })
                        .try_collect_vec()?;
                    let expr = large.push_expr(IrExpressionLarge::ArrayLiteral(e_ty.as_ir(refs), len.clone(), result));
                    Ok(expr)
                }
                _ => Err(err_type()),
            },
            SimpleCompileValue::Function(_)
            | SimpleCompileValue::Module(_)
            | SimpleCompileValue::Interface(_)
            | SimpleCompileValue::InterfaceView(_) => Err(err_type()),
        }
    }
}

impl ValueCommon for MixedCompoundValue {
    fn domain(&self) -> ValueDomain {
        match self {
            MixedCompoundValue::String(v) => ValueDomain::fold(v.iter().map(|p| match p {
                StringPiece::Literal(_) => ValueDomain::CompileTime,
                StringPiece::Substitute(v) => v.domain,
            })),
            MixedCompoundValue::Range(v) => {
                fn opt_domain<C, H>(x: Option<&MaybeCompile<C, HardwareValue<H>>>) -> ValueDomain {
                    match x {
                        None => ValueDomain::CompileTime,
                        Some(start) => match start {
                            MaybeCompile::Compile(_) => ValueDomain::CompileTime,
                            MaybeCompile::Hardware(x) => x.domain,
                        },
                    }
                }

                let mut domain = ValueDomain::CompileTime;
                match v {
                    RangeValue::StartEnd { start, end } => {
                        let end_domain = match end {
                            RangeEnd::Exclusive(end) => opt_domain(end.as_ref()),
                            RangeEnd::Inclusive(end) => opt_domain(Some(end)),
                        };
                        domain = domain.join(opt_domain(start.as_ref()));
                        domain = domain.join(end_domain);
                    }
                    RangeValue::StartLength { start, length } => {
                        domain = domain.join(opt_domain(Some(start)));
                        domain = domain.join(opt_domain(Some(length)));
                    }
                }

                domain
            }
            MixedCompoundValue::Tuple(v) => ValueDomain::fold(v.iter().map(Value::domain)),
            MixedCompoundValue::Struct(v) => {
                let StructValue { ty: _, fields } = v;
                ValueDomain::fold(fields.iter().map(Value::domain))
            }
            MixedCompoundValue::Enum(v) => {
                let EnumValue {
                    ty: _,
                    variant: _,
                    payload,
                } = v;
                match payload {
                    None => ValueDomain::CompileTime,
                    Some(payload) => payload.domain(),
                }
            }
        }
    }

    fn as_ir_expression_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> DiagResult<IrExpression> {
        let err_type = || refs.diags.report(err_hw_type_mismatch(span, ty));

        match self {
            MixedCompoundValue::String(_) => Err(err_type()),
            MixedCompoundValue::Range(_) => Err(err_type()),
            MixedCompoundValue::Tuple(v) => match &ty {
                HardwareType::Tuple(ty) if v.len() == ty.len() => {
                    let result = zip_eq(v.iter(), ty.iter())
                        .map(|(e, e_ty)| e.as_ir_expression_unchecked(refs, large, span, e_ty))
                        .try_collect_vec()?;
                    Ok(large.push_expr(IrExpressionLarge::TupleLiteral(result)))
                }
                _ => Err(err_type()),
            },
            MixedCompoundValue::Struct(v) => match &ty {
                HardwareType::Struct(ty_hw) if ty_hw.inner() == v.ty => {
                    let info = refs.shared.elaboration_arenas.struct_info(ty_hw.inner());
                    let fields_hw = info.fields_hw.as_ref().expect("hardware struct");

                    let result = zip_eq(v.fields.iter(), fields_hw.iter())
                        .map(|(e, e_ty)| e.as_ir_expression_unchecked(refs, large, span, e_ty))
                        .try_collect_vec()?;
                    Ok(large.push_expr(IrExpressionLarge::TupleLiteral(result)))
                }
                _ => Err(err_type()),
            },
            MixedCompoundValue::Enum(v) => match &ty {
                HardwareType::Enum(ty_hw) if ty_hw.inner() == v.ty => {
                    let &EnumValue {
                        ty: _,
                        variant,
                        ref payload,
                    } = v;
                    let info = refs.shared.elaboration_arenas.enum_info(ty_hw.inner());
                    let info_hw = info.hw.as_ref().unwrap();

                    // convert content to bits
                    let payload_bits = payload
                        .as_ref()
                        .map(|payload| {
                            let (payload_ty, _) = info_hw.payload_types[variant].as_ref().unwrap();
                            let payload_expr = payload.as_ir_expression_unchecked(refs, large, span, payload_ty)?;
                            Ok(large.push_expr(IrExpressionLarge::ToBits(payload_ty.as_ir(refs), payload_expr)))
                        })
                        .transpose()?;

                    // build the entire ir expression
                    info_hw.build_ir_expression(large, variant, payload_bits)
                }
                _ => Err(err_type()),
            },
        }
    }
}

impl ValueCommon for CompileCompoundValue {
    fn domain(&self) -> ValueDomain {
        ValueDomain::CompileTime
    }

    fn as_ir_expression_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> DiagResult<IrExpression> {
        MixedCompoundValue::from(self.clone()).as_ir_expression_unchecked(refs, large, span, ty)
    }
}

impl ValueCommon for HardwareValue {
    fn domain(&self) -> ValueDomain {
        self.domain
    }

    fn as_ir_expression_unchecked(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        span: Span,
        ty: &HardwareType,
    ) -> DiagResult<IrExpression> {
        let err_type = || refs.diags.report(err_hw_type_mismatch(span, ty));

        if &self.ty == ty {
            return Ok(self.expr.clone());
        }

        match (&self.ty, &ty) {
            (HardwareType::Bool, HardwareType::Bool) => Ok(self.expr.clone()),
            (HardwareType::Int(ty_curr), HardwareType::Int(ty)) => {
                if ty.contains_range(ty_curr.as_ref()) {
                    Ok(large.push_expr(IrExpressionLarge::ExpandIntRange(ty.clone(), self.expr.clone())))
                } else {
                    Err(err_type())
                }
            }
            (HardwareType::Tuple(ty_curr), HardwareType::Tuple(ty)) => {
                if ty_curr.len() != ty.len() {
                    return Err(err_type());
                }

                let result = (0..ty.len())
                    .map(|i| {
                        let curr_elem_expr = large.push_expr(IrExpressionLarge::TupleIndex {
                            base: self.expr.clone(),
                            index: BigUint::from(i),
                        });
                        let curr_elem_hw = HardwareValue {
                            ty: ty_curr[i].clone(),
                            domain: self.domain,
                            expr: curr_elem_expr,
                        };
                        curr_elem_hw.as_ir_expression_unchecked(refs, large, span, &ty[i])
                    })
                    .try_collect_vec()?;

                Ok(large.push_expr(IrExpressionLarge::TupleLiteral(result)))
            }
            (HardwareType::Array(ty_inner_curr, len_curr), HardwareType::Array(ty_inner, len)) => {
                if len_curr != len {
                    return Err(err_type());
                }

                // TODO express this as a for loop in hardware? this can result in a lot of generated code
                // TODO clean this up once we have non-exclusive integer range
                let mut next = BigUint::ZERO;
                let iter = std::iter::from_fn(|| {
                    if &next < len {
                        let curr = next.clone();
                        next += 1u8;
                        Some(curr)
                    } else {
                        None
                    }
                });
                let result = iter
                    .map(|index| {
                        let curr_elem_expr = large.push_expr(IrExpressionLarge::ArrayIndex {
                            base: self.expr.clone(),
                            index: IrExpression::Int(index.into()),
                        });
                        let curr_elem_hw = HardwareValue {
                            ty: ty_inner_curr.as_ref().clone(),
                            domain: self.domain,
                            expr: curr_elem_expr,
                        };
                        let elem_expr = curr_elem_hw.as_ir_expression_unchecked(refs, large, span, ty_inner)?;
                        Ok(IrArrayLiteralElement::Single(elem_expr))
                    })
                    .try_collect_vec()?;

                Ok(large.push_expr(IrExpressionLarge::ArrayLiteral(
                    ty_inner.as_ir(refs),
                    len.clone(),
                    result,
                )))
            }
            (HardwareType::Struct(_), HardwareType::Struct(_)) | (HardwareType::Enum(_), HardwareType::Enum(_)) => {
                // there's no struct or enum subtyping yet, so this is always an error for now
                Err(err_type())
            }
            (
                HardwareType::Undefined
                | HardwareType::Bool
                | HardwareType::Int(_)
                | HardwareType::Tuple(_)
                | HardwareType::Array(_, _)
                | HardwareType::Struct(_)
                | HardwareType::Enum(_),
                _,
            ) => Err(err_type()),
        }
    }
}

fn err_hw_type_mismatch(span: Span, ty: &HardwareType) -> Diagnostic {
    let msg = format!(
        "wrong type when converting to hardware value with type {}",
        ty.diagnostic_string()
    );
    Diagnostic::new_internal_error(msg).add_error(span, "here").finish()
}

impl<C, H> Value<SimpleCompileValue, C, H> {
    pub fn unit() -> Self {
        // Empty tuples are considered types, since (vacuously) all of their inner values are types.
        //   `is_unit` can't be implemented here yet, hardware values can also be unit
        Self::new_ty(Type::unit())
    }

    pub fn new_ty(ty: Type) -> Self {
        Value::Simple(SimpleCompileValue::Type(ty))
    }

    pub fn new_bool(v: bool) -> Self {
        Value::Simple(SimpleCompileValue::Bool(v))
    }

    pub fn new_int(v: BigInt) -> Self {
        Value::Simple(SimpleCompileValue::Int(v))
    }
}

impl<T, E> HardwareValue<T, E> {
    pub fn map_type<U>(self, f: impl FnOnce(T) -> U) -> HardwareValue<U, E> {
        HardwareValue {
            ty: f(self.ty),
            domain: self.domain,
            expr: self.expr,
        }
    }

    pub fn map_expression<U>(self, f: impl FnOnce(E) -> U) -> HardwareValue<T, U> {
        HardwareValue {
            ty: self.ty,
            domain: self.domain,
            expr: f(self.expr),
        }
    }
}

impl RangeValue<BigInt, BigUint> {
    pub fn as_range(&self) -> IncRange<BigInt> {
        match self {
            RangeValue::StartEnd { start, end } => {
                let end_inc = match end {
                    RangeEnd::Exclusive(end) => end.as_ref().map(|end| end - 1),
                    RangeEnd::Inclusive(end) => Some(end.clone()),
                };
                IncRange {
                    start_inc: start.clone(),
                    end_inc,
                }
            }
            RangeValue::StartLength { start, length } => {
                let end_inc = start + length - 1;
                IncRange {
                    start_inc: Some(start.clone()),
                    end_inc: Some(end_inc),
                }
            }
        }
    }
}

impl<H> From<CompileValue> for Value<SimpleCompileValue, MixedCompoundValue, H> {
    fn from(v: CompileValue) -> Self {
        match v {
            CompileValue::Simple(v) => Value::Simple(v),
            CompileValue::Compound(v) => Value::Compound(MixedCompoundValue::from(v)),
            CompileValue::Hardware(never) => never.unreachable(),
        }
    }
}
impl From<CompileCompoundValue> for MixedCompoundValue {
    fn from(v: CompileCompoundValue) -> Self {
        match v {
            CompileCompoundValue::String(v) => {
                MixedCompoundValue::String(Arc::new(vec![StringPiece::Literal(Arc::unwrap_or_clone(v))]))
            }
            CompileCompoundValue::Range(v) => match v {
                RangeValue::StartEnd { start, end } => MixedCompoundValue::Range(RangeValue::StartEnd {
                    start: start.map(MaybeCompile::Compile),
                    end: match end {
                        RangeEnd::Exclusive(end) => RangeEnd::Exclusive(end.map(MaybeCompile::Compile)),
                        RangeEnd::Inclusive(end) => RangeEnd::Inclusive(MaybeCompile::Compile(end)),
                    },
                }),
                RangeValue::StartLength { start, length } => MixedCompoundValue::Range(RangeValue::StartLength {
                    start: MaybeCompile::Compile(start),
                    length: MaybeCompile::Compile(length),
                }),
            },
            CompileCompoundValue::Tuple(v) => MixedCompoundValue::Tuple(v.map(Value::from)),
            CompileCompoundValue::Struct(StructValue { ty, fields }) => {
                let fields = fields.into_iter().map(Value::from).collect();
                MixedCompoundValue::Struct(StructValue { ty, fields })
            }
            CompileCompoundValue::Enum(EnumValue { ty, variant, payload }) => {
                let payload = payload.map(|e| Box::new(Value::from(*e)));
                MixedCompoundValue::Enum(EnumValue { ty, variant, payload })
            }
        }
    }
}
impl<C> From<MaybeCompile<BigInt, HardwareValue<ClosedIncRange<BigInt>>>>
    for Value<SimpleCompileValue, C, HardwareValue>
{
    fn from(value: MaybeCompile<BigInt, HardwareValue<ClosedIncRange<BigInt>>>) -> Self {
        match value {
            MaybeCompile::Compile(v) => Value::Simple(SimpleCompileValue::Int(v)),
            MaybeCompile::Hardware(v) => Value::Hardware(v.map_type(HardwareType::Int)),
        }
    }
}

impl<C> From<MaybeCompile<BigUint, HardwareValue<ClosedIncRange<BigUint>>>>
    for Value<SimpleCompileValue, C, HardwareValue>
{
    fn from(value: MaybeCompile<BigUint, HardwareValue<ClosedIncRange<BigUint>>>) -> Self {
        match value {
            MaybeCompile::Compile(v) => Value::Simple(SimpleCompileValue::Int(v.into())),
            MaybeCompile::Hardware(v) => Value::Hardware(v.map_type(|r| HardwareType::Int(r.map(BigInt::from)))),
        }
    }
}

impl<H: Clone> TryFrom<&Value<SimpleCompileValue, MixedCompoundValue, H>> for CompileValue {
    type Error = NotCompile;
    fn try_from(v: &Value<SimpleCompileValue, MixedCompoundValue, H>) -> Result<Self, Self::Error> {
        match v {
            Value::Simple(v) => Ok(CompileValue::Simple(v.clone())),
            Value::Compound(v) => Ok(CompileValue::Compound(CompileCompoundValue::try_from(v)?)),
            Value::Hardware(_) => Err(NotCompile),
        }
    }
}
impl TryFrom<&MixedCompoundValue> for CompileCompoundValue {
    type Error = NotCompile;
    fn try_from(v: &MixedCompoundValue) -> Result<Self, Self::Error> {
        match v {
            MixedCompoundValue::String(v) => {
                // TODO simplify this by forcing banning consecutive literals
                let mut s = String::new();
                for p in v.iter() {
                    match p {
                        StringPiece::Literal(p) => s.push_str(p),
                        StringPiece::Substitute(p) => {
                            let _: &HardwareValue = p;
                            return Err(NotCompile);
                        }
                    }
                }
                Ok(CompileCompoundValue::String(Arc::new(s)))
            }
            MixedCompoundValue::Range(v) => {
                fn try_map_bound<I: Clone>(
                    b: &MaybeCompile<I, HardwareValue<ClosedIncRange<I>>>,
                ) -> Result<I, NotCompile> {
                    match b {
                        MaybeCompile::Compile(b) => Ok(b.clone()),
                        MaybeCompile::Hardware(_) => Err(NotCompile),
                    }
                }
                match v {
                    RangeValue::StartEnd { start, end } => Ok(CompileCompoundValue::Range(RangeValue::StartEnd {
                        start: start.as_ref().map(try_map_bound).transpose()?,
                        end: match end {
                            RangeEnd::Exclusive(end) => {
                                RangeEnd::Exclusive(end.as_ref().map(try_map_bound).transpose()?)
                            }
                            RangeEnd::Inclusive(end) => RangeEnd::Inclusive(try_map_bound(end)?),
                        },
                    })),
                    RangeValue::StartLength { start, length } => {
                        Ok(CompileCompoundValue::Range(RangeValue::StartLength {
                            start: try_map_bound(start)?,
                            length: try_map_bound(length)?,
                        }))
                    }
                }
            }
            MixedCompoundValue::Tuple(v) => {
                let v = v.try_map_ref(|e| CompileValue::try_from(e))?;
                Ok(CompileCompoundValue::Tuple(v))
            }
            MixedCompoundValue::Struct(v) => {
                let fields = v.fields.iter().map(CompileValue::try_from).try_collect_vec()?;
                Ok(CompileCompoundValue::Struct(StructValue { ty: v.ty, fields }))
            }
            MixedCompoundValue::Enum(v) => {
                let payload = v
                    .payload
                    .as_ref()
                    .map(|e| CompileValue::try_from(&**e).map(Box::new))
                    .transpose()?;
                Ok(CompileCompoundValue::Enum(EnumValue {
                    ty: v.ty,
                    variant: v.variant,
                    payload,
                }))
            }
        }
    }
}

impl<S, C, H> Value<S, C, H> {
    pub fn map_compile<U>(self, f: impl FnOnce(S) -> U) -> Value<U, C, H> {
        self.try_map_compile::<U, Never>(|v| Ok(f(v))).remove_never()
    }
    pub fn map_mixed<U>(self, f: impl FnOnce(C) -> U) -> Value<S, U, H> {
        self.try_map_mixed::<U, Never>(|v| Ok(f(v))).remove_never()
    }
    pub fn map_hardware<U>(self, f: impl FnOnce(H) -> U) -> Value<S, C, U> {
        self.try_map_hardware::<U, Never>(|v| Ok(f(v))).remove_never()
    }

    pub fn try_map_compile<T, E>(self, f: impl FnOnce(S) -> Result<T, E>) -> Result<Value<T, C, H>, E> {
        match self {
            Value::Simple(v) => Ok(Value::Simple(f(v)?)),
            Value::Compound(v) => Ok(Value::Compound(v)),
            Value::Hardware(v) => Ok(Value::Hardware(v)),
        }
    }
    pub fn try_map_mixed<T, E>(self, f: impl FnOnce(C) -> Result<T, E>) -> Result<Value<S, T, H>, E> {
        match self {
            Value::Simple(v) => Ok(Value::Simple(v)),
            Value::Compound(v) => Ok(Value::Compound(f(v)?)),
            Value::Hardware(v) => Ok(Value::Hardware(v)),
        }
    }
    pub fn try_map_hardware<T, E>(self, f: impl FnOnce(H) -> Result<T, E>) -> Result<Value<S, C, T>, E> {
        match self {
            Value::Simple(v) => Ok(Value::Simple(v)),
            Value::Compound(v) => Ok(Value::Compound(v)),
            Value::Hardware(v) => Ok(Value::Hardware(f(v)?)),
        }
    }
}

impl<S: Typed, C: Typed, H: Typed> Typed for Value<S, C, H> {
    fn ty(&self) -> Type {
        match self {
            Value::Simple(v) => v.ty(),
            Value::Compound(v) => v.ty(),
            Value::Hardware(v) => v.ty(),
        }
    }
}
impl Typed for MixedCompoundValue {
    fn ty(&self) -> Type {
        match self {
            MixedCompoundValue::String(_) => Type::String,
            MixedCompoundValue::Range(_) => Type::Range,
            MixedCompoundValue::Tuple(values) => Type::Tuple(Arc::new(values.iter().map(Value::ty).collect())),
            MixedCompoundValue::Struct(value) => Type::Struct(value.ty),
            MixedCompoundValue::Enum(value) => Type::Enum(value.ty),
        }
    }
}
impl Typed for CompileCompoundValue {
    fn ty(&self) -> Type {
        match self {
            CompileCompoundValue::String(_) => Type::String,
            CompileCompoundValue::Range(_) => Type::Range,
            CompileCompoundValue::Tuple(values) => Type::Tuple(Arc::new(values.iter().map(Value::ty).collect())),
            CompileCompoundValue::Struct(value) => Type::Struct(value.ty),
            CompileCompoundValue::Enum(value) => Type::Enum(value.ty),
        }
    }
}
impl Typed for SimpleCompileValue {
    fn ty(&self) -> Type {
        match self {
            SimpleCompileValue::Type(_) => Type::Type,
            SimpleCompileValue::Bool(_) => Type::Bool,
            SimpleCompileValue::Int(v) => Type::Int(ClosedIncRange::single(v.clone()).into_range()),
            SimpleCompileValue::Array(values) => {
                // TODO precompute this once? this can get slow for large arrays
                let ty_inner = Type::union_all(values.iter().map(CompileValue::ty));
                Type::Array(Arc::new(ty_inner), BigUint::from(values.len()))
            }
            SimpleCompileValue::Function(_) => Type::Function,
            SimpleCompileValue::Module(_) => Type::Module,
            SimpleCompileValue::Interface(_) => Type::Interface,
            SimpleCompileValue::InterfaceView(_) => Type::InterfaceView,
        }
    }
}
impl Typed for HardwareValue<HardwareType> {
    fn ty(&self) -> Type {
        self.ty.as_type()
    }
}
impl<D: Typed> Typed for MaybeUndefined<D> {
    fn ty(&self) -> Type {
        match self {
            MaybeUndefined::Undefined => Type::Undefined,
            MaybeUndefined::Defined(v) => v.ty(),
        }
    }
}

impl<T> MaybeUndefined<T> {
    pub fn map_defined<U>(self, f: impl FnOnce(T) -> U) -> MaybeUndefined<U> {
        match self {
            MaybeUndefined::Undefined => MaybeUndefined::Undefined,
            MaybeUndefined::Defined(v) => MaybeUndefined::Defined(f(v)),
        }
    }

    pub fn as_ref(&self) -> MaybeUndefined<&T> {
        match self {
            MaybeUndefined::Undefined => MaybeUndefined::Undefined,
            MaybeUndefined::Defined(v) => MaybeUndefined::Defined(v),
        }
    }
}

impl<T: Clone> MaybeUndefined<&T> {
    pub fn cloned(self) -> MaybeUndefined<T> {
        match self {
            MaybeUndefined::Undefined => MaybeUndefined::Undefined,
            MaybeUndefined::Defined(v) => MaybeUndefined::Defined(v.clone()),
        }
    }
}

impl<C, H> MaybeCompile<C, H> {
    pub fn map_compile<U>(self, f: impl FnOnce(C) -> U) -> MaybeCompile<U, H> {
        match self {
            MaybeCompile::Compile(v) => MaybeCompile::Compile(f(v)),
            MaybeCompile::Hardware(v) => MaybeCompile::Hardware(v),
        }
    }

    pub fn map_hardware<U>(self, f: impl FnOnce(H) -> U) -> MaybeCompile<C, U> {
        match self {
            MaybeCompile::Compile(v) => MaybeCompile::Compile(v),
            MaybeCompile::Hardware(v) => MaybeCompile::Hardware(f(v)),
        }
    }

    pub fn unwrap_compile(self) -> Result<C, NotCompile> {
        match self {
            MaybeCompile::Compile(v) => Ok(v),
            MaybeCompile::Hardware(_) => Err(NotCompile),
        }
    }
}
