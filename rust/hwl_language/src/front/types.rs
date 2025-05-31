use crate::front::compile::CompileRefs;
use crate::front::diagnostic::ErrorGuaranteed;
use crate::front::item::{ElaboratedEnum, ElaboratedStruct, HardwareChecked, HardwareEnumInfo};
use crate::mid::ir::{IrArrayLiteralElement, IrExpression, IrExpressionLarge, IrLargeArena, IrType};
use crate::swrite;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::ResultExt;
use itertools::{zip_eq, Itertools};
use std::collections::Bound;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, RangeBounds};
use std::sync::Arc;

// TODO add an arena for types?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    // Higher order type, containing other types (including type itself!).
    Type,
    // Lattice top type (including type!)
    Any,
    // Lattice bottom type
    Undefined,
    Bool,
    String,
    Int(IncRange<BigInt>),
    Tuple(Arc<Vec<Type>>),
    Array(Arc<Type>, BigUint),
    Struct(ElaboratedStruct),
    Enum(ElaboratedEnum),
    Range,
    // TODO maybe maybe these (optionally) more specific
    Function,
    Module,
    Interface,
    InterfaceView,
}

// TODO change this to be a struct with some properties (size, ir, all valid, ...) plus a kind enum
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum HardwareType {
    Bool,
    Int(ClosedIncRange<BigInt>),
    Tuple(Arc<Vec<HardwareType>>),
    Array(Arc<HardwareType>, BigUint),
    Struct(HardwareChecked<ElaboratedStruct>),
    Enum(HardwareChecked<ElaboratedEnum>),
}

impl HardwareEnumInfo {
    pub fn tag_range(&self) -> ClosedIncRange<BigInt> {
        ClosedIncRange {
            start_inc: BigInt::ZERO,
            end_inc: BigInt::from(self.content_types.len()) - 1,
        }
    }

    pub fn padding_for_variant(&self, refs: CompileRefs, variant: usize) -> usize {
        let content_size = match &self.content_types[variant] {
            None => 0,
            Some(variant) => usize::try_from(variant.size_bits(refs)).unwrap(),
        };

        assert!(content_size <= self.max_content_size);
        self.max_content_size - content_size
    }

    pub fn build_ir_expression(
        &self,
        refs: CompileRefs,
        large: &mut IrLargeArena,
        variant: usize,
        content_bits: Option<IrExpression>,
    ) -> Result<IrExpression, ErrorGuaranteed> {
        assert_eq!(self.content_types[variant].is_some(), content_bits.is_some());

        // tag
        let tag_range = self.tag_range();
        let ir_tag = IrExpressionLarge::ExpandIntRange(tag_range, IrExpression::Int(BigInt::from(variant)));

        // content
        let mut ir_elements = vec![];
        if let Some(content_bits) = content_bits {
            ir_elements.push(IrArrayLiteralElement::Spread(content_bits));
        }

        // padding
        for _ in 0..self.padding_for_variant(refs, variant) {
            ir_elements.push(IrArrayLiteralElement::Single(IrExpression::Bool(false)));
        }

        // build final expression
        let ir_content =
            IrExpressionLarge::ArrayLiteral(IrType::Bool, BigUint::from(self.max_content_size), ir_elements);
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
    pub fn unit() -> Type {
        Type::Tuple(Arc::new(vec![]))
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, Type::Tuple(inner) if inner.is_empty())
    }

    pub fn union(&self, other: &Type, allow_compound_subtype: bool) -> Type {
        match (self, other) {
            // top and bottom
            (Type::Any, _) | (_, Type::Any) => Type::Any,
            (Type::Undefined, other) | (other, Type::Undefined) => other.clone(),

            // simple matches
            (Type::Type, Type::Type) => Type::Type,
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
                    Type::Tuple(Arc::new(
                        zip_eq(a.iter(), b.iter())
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
                    ))
                } else {
                    Type::Any
                }
            }
            (Type::Array(a_inner, a_len), Type::Array(b_inner, b_len)) => {
                if a_len == b_len {
                    let inner = if allow_compound_subtype {
                        Arc::new(a_inner.union(b_inner, allow_compound_subtype))
                    } else if a_inner == b_inner {
                        a_inner.clone()
                    } else {
                        Arc::new(Type::Any)
                    };
                    Type::Array(inner, a_len.clone())
                } else {
                    // TODO into list once that exists?
                    Type::Any
                }
            }
            (&Type::Struct(a_elab), &Type::Struct(b_elab)) => {
                if a_elab == b_elab {
                    Type::Struct(a_elab)
                } else {
                    Type::Any
                }
            }
            (&Type::Enum(a_elab), &Type::Enum(b_elab)) => {
                if a_elab == b_elab {
                    Type::Enum(a_elab)
                } else {
                    Type::Any
                }
            }

            // simple mismatches
            (
                Type::Type
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
                | Type::Struct(_)
                | Type::Enum(_),
                Type::Type
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
                | Type::Struct(_)
                | Type::Enum(_),
            ) => Type::Any,
        }
    }

    pub fn contains_type(&self, ty: &Type, allow_compound_subtype: bool) -> bool {
        self == &self.union(ty, allow_compound_subtype)
    }

    // TODO centralize error messages for this, everyone is just doing them manually for now
    pub fn as_hardware_type(&self, refs: CompileRefs) -> Result<HardwareType, NonHardwareType> {
        match self {
            Type::Bool => Ok(HardwareType::Bool),
            Type::Int(range) => match range.clone().try_into_closed() {
                Ok(closed_range) => Ok(HardwareType::Int(closed_range)),
                Err(_) => Err(NonHardwareType),
            },
            Type::Tuple(inner) => inner
                .iter()
                .map(|ty| ty.as_hardware_type(refs))
                .try_collect()
                .map(|v| HardwareType::Tuple(Arc::new(v))),
            Type::Array(inner, len) => inner
                .as_hardware_type(refs)
                .map(|inner| HardwareType::Array(Arc::new(inner), len.clone())),
            &Type::Struct(elab) => {
                let info = refs.shared.elaboration_arenas.struct_info(elab);
                match info.fields_hw {
                    Ok(_) => Ok(HardwareType::Struct(HardwareChecked::new_unchecked(elab))),
                    Err(_) => Err(NonHardwareType),
                }
            }
            &Type::Enum(elab) => {
                let info = refs.shared.elaboration_arenas.enum_info(elab);
                match info.hw {
                    Ok(_) => Ok(HardwareType::Enum(HardwareChecked::new_unchecked(elab))),
                    Err(_) => Err(NonHardwareType),
                }
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

    pub fn diagnostic_string(&self) -> String {
        match self {
            Type::Type => "type".to_string(),
            Type::Any => "any".to_string(),
            Type::Undefined => "undefined".to_string(),

            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Int(range) => format!("int({})", range),
            Type::Tuple(inner) => {
                let inner_str = inner.iter().map(Type::diagnostic_string).join(", ");
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

                let inner_str = inner.diagnostic_string();
                format!("{inner_str}[{dims}]")
            }
            // TODO better names here, including the definition name/loc/params
            Type::Struct(_) => "struct".to_string(),
            Type::Enum(_) => "enum".to_string(),
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
            HardwareType::Bool => Type::Bool,
            HardwareType::Int(range) => Type::Int(range.clone().into_range()),
            HardwareType::Tuple(inner) => Type::Tuple(Arc::new(inner.iter().map(HardwareType::as_type).collect_vec())),
            HardwareType::Array(inner, len) => Type::Array(Arc::new(inner.as_type()), len.clone()),
            HardwareType::Struct(elab) => Type::Struct(elab.inner()),
            HardwareType::Enum(elab) => Type::Enum(elab.inner()),
        }
    }

    pub fn as_ir(&self, refs: CompileRefs) -> IrType {
        match self {
            HardwareType::Bool => IrType::Bool,
            HardwareType::Int(range) => IrType::Int(range.clone()),
            HardwareType::Tuple(inner) => IrType::Tuple(inner.iter().map(|ty| ty.as_ir(refs)).collect_vec()),
            HardwareType::Array(inner, len) => IrType::Array(Box::new(inner.as_ir(refs)), len.clone()),
            &HardwareType::Struct(elab) => {
                let info = refs.shared.elaboration_arenas.struct_info(elab.inner());
                let fields_hw = info.fields_hw.as_ref_ok().unwrap();
                IrType::Tuple(fields_hw.iter().map(|ty| ty.as_ir(refs)).collect_vec())
            }
            HardwareType::Enum(elab) => {
                let info = refs.shared.elaboration_arenas.enum_info(elab.inner());
                let info_hw = info.hw.as_ref_ok().unwrap();

                let tag_ty = IrType::Int(ClosedIncRange {
                    start_inc: BigInt::ZERO,
                    end_inc: BigInt::from(info.variants.len()) - 1,
                });
                let data_ty = IrType::Array(Box::new(IrType::Bool), BigUint::from(info_hw.max_content_size));
                IrType::Tuple(vec![tag_ty, data_ty])
            }
        }
    }

    pub fn diagnostic_string(&self) -> String {
        self.as_type().diagnostic_string()
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
