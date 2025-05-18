use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::item::{ElaboratedEnum, ElaboratedStruct};
use crate::front::value::CompileValue;
use crate::mid::ir::{IrArrayLiteralElement, IrExpression, IrExpressionLarge, IrLargeArena, IrType};
use crate::swrite;
use crate::syntax::pos::Span;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::int::IntRepresentation;
use crate::util::iter::IterExt;
use itertools::{zip_eq, Itertools};
use std::collections::Bound;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, RangeBounds};
use unwrap_match::unwrap_match;

// TODO add an arena for types?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    // Higher order type, containing other types (including type itself!).
    Type,
    // Lattice top type (including type!)
    Any,
    // Lattice bottom type
    Undefined,
    Clock,
    Bool,
    String,
    Int(IncRange<BigInt>),
    Tuple(Vec<Type>),
    Array(Box<Type>, BigUint),
    // TODO avoid storing copy of field types
    Struct(ElaboratedStruct, Vec<Type>),
    Enum(ElaboratedEnum, Vec<Option<Type>>),
    Range,
    // TODO maybe maybe these (optionally) more specific
    Function,
    Module,
    Interface,
    InterfaceView,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum HardwareType {
    Clock,
    Bool,
    Int(ClosedIncRange<BigInt>),
    Tuple(Vec<HardwareType>),
    Array(Box<HardwareType>, BigUint),
    Struct(ElaboratedStruct, Vec<HardwareType>),
    // TODO allow configuring whether this is stored compactly or more like a tuple,
    //   the first is better for memory but the second may be better for timing
    Enum(HardwareEnum),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct HardwareEnum {
    pub elab: ElaboratedEnum,
    pub variants: Vec<Option<HardwareType>>,
    pub data_size: usize,
}

impl HardwareEnum {
    pub fn tag_range(&self) -> ClosedIncRange<BigInt> {
        ClosedIncRange {
            start_inc: BigInt::ZERO,
            end_inc: BigInt::from(self.variants.len()) - 1,
        }
    }

    pub fn padding_for_variant(&self, variant: usize) -> usize {
        let content_size = match &self.variants[variant] {
            None => 0,
            Some(variant) => usize::try_from(variant.size_bits()).unwrap(),
        };

        assert!(content_size <= self.data_size);
        self.data_size - content_size
    }

    pub fn build_ir_expression(
        &self,
        large: &mut IrLargeArena,
        variant: usize,
        content_bits: Option<IrExpression>,
    ) -> Result<IrExpression, ErrorGuaranteed> {
        assert_eq!(self.variants[variant].is_some(), content_bits.is_some());

        // tag
        let tag_range = self.tag_range();
        let ir_tag = IrExpressionLarge::ExpandIntRange(tag_range, IrExpression::Int(BigInt::from(variant)));

        // content
        let mut ir_elements = vec![];
        if let Some(content_bits) = content_bits {
            ir_elements.push(IrArrayLiteralElement::Spread(content_bits));
        }

        // padding
        for _ in 0..self.padding_for_variant(variant) {
            ir_elements.push(IrArrayLiteralElement::Single(IrExpression::Bool(false)));
        }

        // build final expression
        let ir_content = IrExpressionLarge::ArrayLiteral(IrType::Bool, BigUint::from(self.data_size), ir_elements);
        let ir_expr = IrExpressionLarge::TupleLiteral(vec![large.push_expr(ir_tag), large.push_expr(ir_content)]);
        Ok(large.push_expr(ir_expr))
    }
}

// TODO rename to min/max? more intuitive than start/end, min and max are clearly inclusive
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct IncRange<T> {
    pub start_inc: Option<T>,
    pub end_inc: Option<T>,
}

// TODO can this represent the empty range? maybe exclusive is better after all...
//   we don't really want empty ranges for int types, but for for loops and slices we do
// TODO switch to exclusive ranges, much more intuitive to program with, especially for arrays and loops
//   match code becomes harder, but that's fine
// TODO transition this to multi-range as the int type
// TODO make sure that people can only construct non-decreasing ranges,
//   there are still some panics in the compiler because of this
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ClosedIncRange<T> {
    pub start_inc: T,
    pub end_inc: T,
}

pub trait Typed {
    fn ty(&self) -> Type;
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NonHardwareType;

impl Type {
    pub const UNIT: Type = Type::Tuple(Vec::new());

    pub fn union(&self, other: &Type, allow_compound_subtype: bool) -> Type {
        match (self, other) {
            // top and bottom
            (Type::Any, _) | (_, Type::Any) => Type::Any,
            (Type::Undefined, other) | (other, Type::Undefined) => other.clone(),

            // simple matches
            (Type::Type, Type::Type) => Type::Type,
            (Type::Clock, Type::Clock) => Type::Clock,
            (Type::Bool, Type::Bool) => Type::Bool,
            (Type::String, Type::String) => Type::String,
            // TODO should we even allow unions for these?
            (Type::Range, Type::Range) => Type::Range,
            (Type::Function, Type::Function) => Type::Function,
            (Type::Module, Type::Module) => Type::Module,
            (Type::Interface, Type::Interface) => Type::Interface,
            (Type::InterfaceView, Type::InterfaceView) => Type::InterfaceView,

            (Type::Int(a), Type::Int(b)) => {
                let IncRange {
                    start_inc: a_start,
                    end_inc: a_end,
                } = a;
                let IncRange {
                    start_inc: b_start,
                    end_inc: b_end,
                } = b;

                let start = match (a_start, b_start) {
                    (Some(a_start), Some(b_start)) => Some(a_start.min(b_start).clone()),
                    (None, _) | (_, None) => None,
                };
                let end = match (a_end, b_end) {
                    (Some(a_end), Some(b_end)) => Some(a_end.max(b_end).clone()),
                    (None, _) | (_, None) => None,
                };

                Type::Int(IncRange {
                    start_inc: start,
                    end_inc: end,
                })
            }

            (Type::Tuple(a), Type::Tuple(b)) => {
                if a.len() == b.len() {
                    Type::Tuple(
                        zip_eq(a, b)
                            .map(|(a, b)| {
                                if allow_compound_subtype {
                                    a.union(b, allow_compound_subtype)
                                } else if a == b {
                                    a.clone()
                                } else {
                                    Type::Any
                                }
                            })
                            .collect_vec(),
                    )
                } else {
                    Type::Any
                }
            }
            (Type::Array(a_inner, a_len), Type::Array(b_inner, b_len)) => {
                if a_len == b_len {
                    let inner = if allow_compound_subtype {
                        a_inner.union(b_inner, allow_compound_subtype)
                    } else if a_inner == b_inner {
                        *a_inner.clone()
                    } else {
                        Type::Any
                    };
                    Type::Array(Box::new(inner), a_len.clone())
                } else {
                    // TODO into list once that exists?
                    Type::Any
                }
            }
            (Type::Struct(a_item, a_fields), Type::Struct(b_item, b_fields)) => {
                if a_item == b_item {
                    debug_assert_eq!(a_fields, b_fields);
                    Type::Struct(*a_item, a_fields.clone())
                } else {
                    Type::Any
                }
            }
            (&Type::Enum(a_item, ref a_variants), &Type::Enum(b_item, ref b_variants)) => {
                if a_item == b_item {
                    debug_assert_eq!(a_variants, b_variants);
                    Type::Enum(a_item, a_variants.clone())
                } else {
                    Type::Any
                }
            }

            // simple mismatches
            (
                Type::Type
                | Type::Clock
                | Type::Bool
                | Type::String
                | Type::Range
                | Type::Function
                | Type::Module
                | Type::Interface
                | Type::InterfaceView
                | Type::Int(_)
                | Type::Tuple(_)
                | Type::Array(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _),
                Type::Type
                | Type::Clock
                | Type::Bool
                | Type::String
                | Type::Range
                | Type::Function
                | Type::Module
                | Type::Interface
                | Type::InterfaceView
                | Type::Int(_)
                | Type::Tuple(_)
                | Type::Array(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _),
            ) => Type::Any,
        }
    }

    pub fn contains_type(&self, ty: &Type, allow_compound_subtype: bool) -> bool {
        self == &self.union(ty, allow_compound_subtype)
    }

    // TODO centralize error messages for this, everyone is just doing them manually for now
    pub fn as_hardware_type(&self) -> Result<HardwareType, NonHardwareType> {
        match self {
            Type::Clock => Ok(HardwareType::Clock),
            Type::Bool => Ok(HardwareType::Bool),
            Type::Int(range) => match range.clone().try_into_closed() {
                Ok(closed_range) => Ok(HardwareType::Int(closed_range)),
                Err(_) => Err(NonHardwareType),
            },
            Type::Tuple(inner) => inner
                .iter()
                .map(Type::as_hardware_type)
                .try_collect()
                .map(HardwareType::Tuple),
            Type::Array(inner, len) => inner
                .as_hardware_type()
                .map(|inner| HardwareType::Array(Box::new(inner), len.clone())),
            Type::Struct(item, fields) => fields
                .iter()
                .map(Type::as_hardware_type)
                .try_collect()
                .map(|fields| HardwareType::Struct(*item, fields)),
            &Type::Enum(item, ref variants) => {
                let variants = variants
                    .iter()
                    .map(|v| v.as_ref().map(Type::as_hardware_type).transpose())
                    .try_collect_vec()?;

                let data_size = variants
                    .iter()
                    .filter_map(|ty| ty.as_ref().map(HardwareType::size_bits))
                    .max()
                    .unwrap_or(BigUint::ZERO);
                let data_size = usize::try_from(data_size).map_err(|_| NonHardwareType)?;

                Ok(HardwareType::Enum(HardwareEnum {
                    elab: item,
                    variants,
                    data_size,
                }))
            }
            Type::Type
            | Type::Any
            | Type::Undefined
            | Type::String
            | Type::Range
            | Type::Function
            | Type::Module
            | Type::Interface
            | Type::InterfaceView => Err(NonHardwareType),
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        match self {
            Type::Type => "type".to_string(),
            Type::Any => "any".to_string(),
            Type::Undefined => "undefined".to_string(),

            Type::Clock => "clock".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Int(range) => format!("int({})", range),
            Type::Tuple(inner) => {
                let inner_str = inner.iter().map(Type::to_diagnostic_string).join(", ");
                format!("({})", inner_str)
            }
            Type::Array(first_inner, first_len) => {
                let mut dims = String::new();

                swrite!(&mut dims, "{}", first_len);
                let mut inner = first_inner;
                while let Type::Array(curr_inner, curr_len) = &**inner {
                    swrite!(&mut dims, ", {}", curr_len);
                    inner = curr_inner;
                }

                let inner_str = inner.to_diagnostic_string();
                format!("{inner_str}[{dims}]")
            }
            Type::Struct(_, _) => "struct".to_string(),
            Type::Enum(_, _) => "enum".to_string(),
            Type::Range => "range".to_string(),
            Type::Function => "function".to_string(),
            Type::Module => "module".to_string(),
            Type::Interface => "interface".to_string(),
            Type::InterfaceView => "interface_view".to_string(),
        }
    }
}

impl HardwareType {
    pub fn as_type(&self) -> Type {
        match self {
            HardwareType::Clock => Type::Clock,
            HardwareType::Bool => Type::Bool,
            HardwareType::Int(range) => Type::Int(range.clone().into_range()),
            HardwareType::Tuple(inner) => Type::Tuple(inner.iter().map(HardwareType::as_type).collect_vec()),
            HardwareType::Array(inner, len) => Type::Array(Box::new(inner.as_type()), len.clone()),
            HardwareType::Struct(item, fields) => {
                Type::Struct(*item, fields.iter().map(HardwareType::as_type).collect_vec())
            }
            HardwareType::Enum(hw_enum) => Type::Enum(
                hw_enum.elab,
                hw_enum
                    .variants
                    .iter()
                    .map(|v| v.as_ref().map(HardwareType::as_type))
                    .collect_vec(),
            ),
        }
    }

    pub fn as_ir(&self) -> IrType {
        match self {
            HardwareType::Clock => IrType::Bool,
            HardwareType::Bool => IrType::Bool,
            HardwareType::Int(range) => IrType::Int(range.clone()),
            HardwareType::Tuple(inner) => IrType::Tuple(inner.iter().map(HardwareType::as_ir).collect_vec()),
            HardwareType::Array(inner, len) => IrType::Array(Box::new(inner.as_ir()), len.clone()),
            HardwareType::Struct(_, fields) => IrType::Tuple(fields.iter().map(HardwareType::as_ir).collect_vec()),
            HardwareType::Enum(hw_enum) => {
                let tag_ty = IrType::Int(ClosedIncRange {
                    start_inc: BigInt::ZERO,
                    end_inc: BigInt::from(hw_enum.variants.len()) - 1,
                });
                let data_ty = IrType::Array(Box::new(IrType::Bool), BigUint::from(hw_enum.data_size));
                IrType::Tuple(vec![tag_ty, data_ty])
            }
        }
    }

    pub fn size_bits(&self) -> BigUint {
        self.as_ir().size_bits()
    }

    pub fn every_bit_pattern_is_valid(&self) -> bool {
        match self {
            HardwareType::Clock => true,
            HardwareType::Bool => true,
            HardwareType::Int(range) => {
                let repr = IntRepresentation::for_range(range);
                &repr.range() == range
            }
            HardwareType::Tuple(inner) => inner.iter().all(HardwareType::every_bit_pattern_is_valid),
            HardwareType::Array(inner, _len) => inner.every_bit_pattern_is_valid(),
            HardwareType::Struct(_, fields) => fields.iter().all(HardwareType::every_bit_pattern_is_valid),
            HardwareType::Enum(hw_enum) => {
                // The tag needs to be fully valid.
                let tag_range = ClosedIncRange {
                    start_inc: BigInt::ZERO,
                    end_inc: BigInt::from(hw_enum.variants.len()) - 1,
                };
                let tag_repr = IntRepresentation::for_range(&tag_range);
                if tag_repr.range() != tag_range {
                    return false;
                }

                // We don't need all variants to be the same size:
                //   the bits they don't cover will never be used, since the tag should be checked first.
                // Each variant being valid individually is enough.
                hw_enum
                    .variants
                    .iter()
                    .filter_map(Option::as_ref)
                    .all(HardwareType::every_bit_pattern_is_valid)
            }
        }
    }

    pub fn value_to_bits(
        &self,
        diags: &Diagnostics,
        span: Span,
        value: &CompileValue,
    ) -> Result<Vec<bool>, ErrorGuaranteed> {
        let mut result = Vec::new();
        self.value_to_bits_impl(diags, span, value, &mut result)?;
        Ok(result)
    }

    fn value_to_bits_impl(
        &self,
        diags: &Diagnostics,
        span: Span,
        value: &CompileValue,
        result: &mut Vec<bool>,
    ) -> Result<(), ErrorGuaranteed> {
        let err_internal = || {
            diags.report_internal_error(
                span,
                format!(
                    "failed to convert value `{}` to bits of type `{}`",
                    value.to_diagnostic_string(),
                    self.to_diagnostic_string()
                ),
            )
        };
        match (self, value) {
            (HardwareType::Bool, &CompileValue::Bool(v)) => {
                result.push(v);
                Ok(())
            }
            (HardwareType::Int(range), CompileValue::Int(v)) => {
                let repr = IntRepresentation::for_range(range);
                repr.value_to_bits(v, result).map_err(|_| err_internal())
            }
            (HardwareType::Tuple(ty_inners), CompileValue::Tuple(v_inners)) => {
                assert_eq!(ty_inners.len(), v_inners.len());
                for (ty_inner, v_inner) in zip_eq(ty_inners, v_inners) {
                    ty_inner.value_to_bits_impl(diags, span, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Array(ty_inner, ty_len), CompileValue::Array(v_inner)) => {
                assert_eq!(ty_len, &BigUint::from(v_inner.len()));
                for v_inner in v_inner {
                    ty_inner.value_to_bits_impl(diags, span, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Struct(_, ty_fields), CompileValue::Struct(_, _, value_fields)) => {
                assert_eq!(ty_fields.len(), ty_fields.len());
                for (ty_inner, v_inner) in zip_eq(ty_fields, value_fields) {
                    ty_inner.value_to_bits_impl(diags, span, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Enum(ty_enum), &CompileValue::Enum(_, _, (variant, ref content))) => {
                // tag
                let tag_range = ty_enum.tag_range();
                HardwareType::Int(tag_range).value_to_bits_impl(
                    diags,
                    span,
                    &CompileValue::Int(BigInt::from(variant)),
                    result,
                )?;

                // content
                if let Some(content) = content {
                    let content_ty = ty_enum.variants[variant].as_ref().unwrap();
                    content_ty.value_to_bits_impl(diags, span, content, result)?;
                }

                // padding
                for _ in 0..ty_enum.padding_for_variant(variant) {
                    result.push(false);
                }
                Ok(())
            }

            // clock cannot be converted
            (HardwareType::Clock, _) => Err(err_internal()),
            // type mismatches
            (
                HardwareType::Bool
                | HardwareType::Int(_)
                | HardwareType::Tuple(_)
                | HardwareType::Array(_, _)
                | HardwareType::Struct(_, _)
                | HardwareType::Enum(_),
                _,
            ) => Err(err_internal()),
        }
    }

    pub fn value_from_bits(
        &self,
        diags: &Diagnostics,
        span: Span,
        bits: &[bool],
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let mut iter = bits.iter().copied();
        let result = self.value_from_bits_impl(diags, span, &mut iter)?;
        match iter.next() {
            None => Ok(result),
            Some(_) => Err(diags.report_internal_error(span, "leftover bits when converting to value")),
        }
    }

    pub fn value_from_bits_impl(
        &self,
        diags: &Diagnostics,
        span: Span,
        bits: &mut impl Iterator<Item = bool>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let err_internal = || {
            diags.report_internal_error(
                span,
                format!(
                    "failed to convert bits to value of type `{}`",
                    self.to_diagnostic_string()
                ),
            )
        };

        match self {
            HardwareType::Clock => Err(err_internal()),
            HardwareType::Bool => Ok(CompileValue::Bool(bits.next().ok_or(err_internal())?)),
            HardwareType::Int(range) => {
                let repr = IntRepresentation::for_range(range);
                let bits: Vec<bool> = (0..repr.size_bits())
                    .map(|_| bits.next().ok_or_else(err_internal))
                    .try_collect()?;

                let result = repr.value_from_bits(&bits).map_err(|_| err_internal())?;

                if range.contains(&result) {
                    Ok(CompileValue::Int(result))
                } else {
                    Err(err_internal())
                }
            }
            HardwareType::Tuple(inners) => Ok(CompileValue::Array(
                inners
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(diags, span, bits))
                    .try_collect()?,
            )),
            HardwareType::Array(inner, len) => {
                let len = usize::try_from(len).map_err(|_| err_internal())?;
                Ok(CompileValue::Array(
                    (0..len)
                        .map(|_| inner.value_from_bits_impl(diags, span, bits))
                        .try_collect()?,
                ))
            }
            HardwareType::Struct(item, fields) => {
                let inners = fields
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(diags, span, bits))
                    .try_collect()?;
                let fields = fields.iter().map(HardwareType::as_type).collect_vec();
                Ok(CompileValue::Struct(*item, fields, inners))
            }
            HardwareType::Enum(item) => {
                // tag
                let tag_ty = HardwareType::Int(item.tag_range());
                let tag_value = tag_ty.value_from_bits_impl(diags, span, bits)?;
                let tag_value = unwrap_match!(tag_value, CompileValue::Int(v) => v);
                let tag_value = usize::try_from(tag_value).unwrap();

                // content
                let content_ty = item.variants[tag_value].as_ref();
                let content_value = if let Some(content_ty) = content_ty {
                    let content_value = content_ty.value_from_bits_impl(diags, span, bits)?;
                    Some(Box::new(content_value))
                } else {
                    None
                };

                // discard padding
                for _ in 0..item.padding_for_variant(tag_value) {
                    bits.next().ok_or_else(err_internal)?;
                }

                Ok(CompileValue::Enum(
                    item.elab,
                    item.variants
                        .iter()
                        .map(|v| v.as_ref().map(HardwareType::as_type))
                        .collect_vec(),
                    (tag_value, content_value),
                ))
            }
        }
    }

    pub fn to_diagnostic_string(&self) -> String {
        self.as_type().to_diagnostic_string()
    }
}

impl<T> IncRange<T> {
    pub const OPEN: IncRange<T> = IncRange {
        start_inc: None,
        end_inc: None,
    };

    pub fn try_into_closed(self) -> Result<ClosedIncRange<T>, Self> {
        let IncRange { start_inc, end_inc } = self;

        let start_inc = match start_inc {
            Some(start_inc) => start_inc,
            None => {
                return Err(IncRange {
                    start_inc: None,
                    end_inc,
                })
            }
        };
        let end_inc = match end_inc {
            Some(end_inc) => end_inc,
            None => {
                return Err(IncRange {
                    start_inc: Some(start_inc),
                    end_inc: None,
                })
            }
        };

        Ok(ClosedIncRange { start_inc, end_inc })
    }

    pub fn contains(&self, other: &T) -> bool
    where
        T: Ord,
    {
        let IncRange { start_inc, end_inc } = self;
        match (start_inc, end_inc) {
            (None, None) => true,
            (Some(start_inc), None) => start_inc <= other,
            (None, Some(end_inc)) => other <= end_inc,
            (Some(start_inc), Some(end_inc)) => start_inc <= other && other <= end_inc,
        }
    }

    pub fn contains_range(&self, other: &IncRange<T>) -> bool
    where
        T: Ord,
    {
        let IncRange {
            start_inc: self_start_inc,
            end_inc: self_end_inc,
        } = self;
        let IncRange {
            start_inc: other_start_inc,
            end_inc: other_end_inc,
        } = other;

        let start_contains = match (self_start_inc, other_start_inc) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(self_start_inc), Some(other_start_inc)) => self_start_inc <= other_start_inc,
        };
        let end_contains = match (self_end_inc, other_end_inc) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(self_end_inc), Some(other_end_inc)) => self_end_inc >= other_end_inc,
        };

        start_contains && end_contains
    }
}

impl<T> ClosedIncRange<T> {
    pub fn single(value: T) -> ClosedIncRange<T>
    where
        T: Clone,
    {
        ClosedIncRange {
            start_inc: value.clone(),
            end_inc: value,
        }
    }

    pub fn into_range(self) -> IncRange<T> {
        let ClosedIncRange { start_inc, end_inc } = self;
        IncRange {
            start_inc: Some(start_inc),
            end_inc: Some(end_inc),
        }
    }

    pub fn as_ref(&self) -> ClosedIncRange<&T> {
        ClosedIncRange {
            start_inc: &self.start_inc,
            end_inc: &self.end_inc,
        }
    }

    pub fn map(self, mut f: impl FnMut(T) -> T) -> Self {
        let ClosedIncRange { start_inc, end_inc } = self;
        ClosedIncRange {
            start_inc: f(start_inc),
            end_inc: f(end_inc),
        }
    }

    pub fn contains(&self, value: &T) -> bool
    where
        T: Ord,
    {
        let ClosedIncRange { start_inc, end_inc } = self;
        start_inc <= value && value <= end_inc
    }

    pub fn contains_range(&self, other: &ClosedIncRange<T>) -> bool
    where
        T: Ord,
    {
        let ClosedIncRange { start_inc, end_inc } = self;
        let ClosedIncRange {
            start_inc: other_start_inc,
            end_inc: other_end_inc,
        } = other;
        start_inc <= other_start_inc && other_end_inc <= end_inc
    }

    pub fn as_single(&self) -> Option<&T>
    where
        T: Eq,
    {
        if self.start_inc == self.end_inc {
            Some(&self.start_inc)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_
    where
        T: Clone + AddAssign<u32> + Ord,
    {
        let mut next = self.start_inc.clone();
        std::iter::from_fn(move || {
            if next <= self.end_inc {
                let curr = next.clone();
                next += 1;
                Some(curr)
            } else {
                None
            }
        })
    }
}

impl<T: Display> Display for IncRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let IncRange { start_inc, end_inc } = self;
        match (start_inc, end_inc) {
            (None, None) => write!(f, ".."),
            (Some(start_inc), None) => write!(f, "{}..", start_inc),
            (None, Some(end_inc)) => write!(f, "..={}", end_inc),
            (Some(start_inc), Some(end_inc)) => write!(f, "{}..={}", start_inc, end_inc),
        }
    }
}

impl<T: Display> Display for ClosedIncRange<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ClosedIncRange { start_inc, end_inc } = self;
        write!(f, "{}..={}", start_inc, end_inc)
    }
}

impl<T> RangeBounds<T> for IncRange<T> {
    fn start_bound(&self) -> Bound<&T> {
        match &self.start_inc {
            None => Bound::Unbounded,
            Some(start_inc) => Bound::Included(start_inc),
        }
    }

    fn end_bound(&self) -> Bound<&T> {
        match &self.end_inc {
            None => Bound::Unbounded,
            Some(end_inc) => Bound::Included(end_inc),
        }
    }
}

impl<T> RangeBounds<T> for ClosedIncRange<T> {
    fn start_bound(&self) -> Bound<&T> {
        Bound::Included(&self.start_inc)
    }

    fn end_bound(&self) -> Bound<&T> {
        Bound::Included(&self.end_inc)
    }
}

pub struct ClosedIncRangeIterator<T> {
    next: T,
    end_inc: T,
}

impl IntoIterator for ClosedIncRange<BigUint> {
    type Item = BigUint;
    type IntoIter = ClosedIncRangeIterator<BigUint>;

    fn into_iter(self) -> Self::IntoIter {
        ClosedIncRangeIterator {
            next: self.start_inc,
            end_inc: self.end_inc,
        }
    }
}

impl Iterator for ClosedIncRangeIterator<BigUint> {
    type Item = BigUint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next <= self.end_inc {
            let curr = self.next.clone();
            self.next += 1u8;
            Some(curr)
        } else {
            None
        }
    }
}

impl IntoIterator for ClosedIncRange<BigInt> {
    type Item = BigInt;
    type IntoIter = ClosedIncRangeIterator<BigInt>;

    fn into_iter(self) -> Self::IntoIter {
        ClosedIncRangeIterator {
            next: self.start_inc,
            end_inc: self.end_inc,
        }
    }
}

impl Iterator for ClosedIncRangeIterator<BigInt> {
    type Item = BigInt;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next <= self.end_inc {
            let curr = self.next.clone();
            self.next += 1u8;
            Some(curr)
        } else {
            None
        }
    }
}
