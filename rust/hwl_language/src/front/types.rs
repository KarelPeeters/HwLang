use crate::front::compile::CompileRefs;
use crate::front::diagnostic::DiagResult;
use crate::front::item::{ElaboratedEnum, ElaboratedStruct, ElaborationArenas, HardwareChecked, HardwareEnumInfo};
use crate::front::value::HardwareValue;
use crate::mid::ir::{IrArrayLiteralElement, IrExpression, IrExpressionLarge, IrIntCompareOp, IrLargeArena, IrType};
use crate::util::big_int::{BigInt, BigUint};
use crate::util::range::{ClosedNonEmptyRange, Range};
use crate::util::{Never, ResultExt};
use itertools::{Itertools, zip_eq};
use std::sync::Arc;

// TODO add an arena for types?
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type {
    // Lattice top type (including type!)
    Any,
    // Lattice bottom type
    Undefined,
    // Higher order type, containing other types (including type itself!).
    Type,

    Bool,
    String,
    Int(Range<BigInt>),
    // TODO empty tuples should be convertible to hardware
    Tuple(Arc<Vec<Type>>),
    Array(Arc<Type>, BigUint),
    // TODO make user type covariant? or allow users to define variance?
    Struct(ElaboratedStruct),
    Enum(ElaboratedEnum),
    // TODO include bound ranges as part of the type?
    Range,
    // TODO maybe maybe these (optionally) more specific
    Function,
    Module,
    Interface,
    InterfaceView,
    Builtin,
}

// TODO change this to be a struct with some properties (size, ir, all valid, ...) plus a kind enum
// TODO add range
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum HardwareType {
    Undefined,
    Bool,
    Int(ClosedNonEmptyRange<BigInt>),
    Tuple(Arc<Vec<HardwareType>>),
    Array(Arc<HardwareType>, BigUint),
    Struct(HardwareChecked<ElaboratedStruct>),
    Enum(HardwareChecked<ElaboratedEnum>),
}

#[derive(Debug, Copy, Clone)]
pub struct TypeBool;

impl HardwareEnumInfo {
    pub fn padding_for_variant(&self, variant: usize) -> usize {
        let content_size = match &self.payload_types[variant] {
            None => 0,
            Some((_, ty_ir)) => usize::try_from(ty_ir.size_bits()).unwrap(),
        };

        assert!(content_size <= self.max_payload_size);
        self.max_payload_size - content_size
    }

    pub fn build_ir_expression(
        &self,
        large: &mut IrLargeArena,
        variant: usize,
        content_bits: Option<IrExpression>,
    ) -> DiagResult<IrExpression> {
        assert_eq!(self.payload_types[variant].is_some(), content_bits.is_some());

        // tag
        let ir_tag =
            IrExpressionLarge::ExpandIntRange(self.tag_range.clone(), IrExpression::Int(BigInt::from(variant)));

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
        let ir_content =
            IrExpressionLarge::ArrayLiteral(IrType::Bool, BigUint::from(self.max_payload_size), ir_elements);
        let ir_expr = IrExpressionLarge::TupleLiteral(vec![large.push_expr(ir_tag), large.push_expr(ir_content)]);
        Ok(large.push_expr(ir_expr))
    }

    pub fn check_tag_matches(&self, large: &mut IrLargeArena, value: IrExpression, variant: usize) -> IrExpression {
        let tag = large.push_expr(IrExpressionLarge::TupleIndex {
            base: value,
            index: BigUint::ZERO,
        });

        large.push_expr(IrExpressionLarge::IntCompare(
            IrIntCompareOp::Eq,
            tag,
            IrExpression::Int(BigInt::from(variant)),
        ))
    }

    pub fn extract_payload(
        &self,
        large: &mut IrLargeArena,
        value: &HardwareValue,
        variant: usize,
    ) -> Option<HardwareValue> {
        let (payload_ty, payload_ty_ir) = self.payload_types[variant].as_ref()?;

        let payload_bits_all = large.push_expr(IrExpressionLarge::TupleIndex {
            base: value.expr.clone(),
            index: BigUint::ONE,
        });
        let payload_bits = large.push_expr(IrExpressionLarge::ArraySlice {
            base: payload_bits_all,
            start: IrExpression::Int(BigInt::ZERO),
            len: payload_ty_ir.size_bits(),
        });
        let payload = large.push_expr(IrExpressionLarge::FromBits(payload_ty_ir.clone(), payload_bits));

        Some(HardwareValue {
            ty: payload_ty.clone(),
            domain: value.domain,
            expr: payload,
        })
    }
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

    pub fn union_all(types: impl IntoIterator<Item = Type>) -> Type {
        types.into_iter().fold(Type::Undefined, |acc, ty| acc.union(&ty))
    }

    pub fn union(&self, other: &Type) -> Type {
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
            (Type::Builtin, Type::Builtin) => Type::Builtin,

            (Type::Int(a), Type::Int(b)) => Type::Int(a.union(b.as_ref()).cloned()),

            (Type::Tuple(a), Type::Tuple(b)) => {
                if a.len() == b.len() {
                    Type::Tuple(Arc::new(
                        zip_eq(a.iter(), b.iter()).map(|(a, b)| a.union(b)).collect_vec(),
                    ))
                } else {
                    Type::Any
                }
            }
            (Type::Array(a_inner, a_len), Type::Array(b_inner, b_len)) => {
                if a_len == b_len {
                    Type::Array(Arc::new(a_inner.union(b_inner)), a_len.clone())
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

            // simple mismatches (we list all of them out manually here to force exhaustiveness checking)
            (
                Type::Type
                | Type::Bool
                | Type::String
                | Type::Range
                | Type::Function
                | Type::Module
                | Type::Interface
                | Type::InterfaceView
                | Type::Builtin
                | Type::Int(_)
                | Type::Tuple(_)
                | Type::Array(_, _)
                | Type::Struct(_)
                | Type::Enum(_),
                _,
            ) => Type::Any,
        }
    }

    pub fn contains_type(&self, ty: &Type) -> bool {
        self == &self.union(ty)
    }

    // TODO centralize error messages for this, everyone is just doing them manually for now
    // TODO accept empty tuples here, maybe those need to be normal values instead of types,
    //   and then cast to type where needed
    pub fn as_hardware_type(&self, elab: &ElaborationArenas) -> Result<HardwareType, NonHardwareType> {
        match self {
            Type::Undefined => Ok(HardwareType::Undefined),
            Type::Bool => Ok(HardwareType::Bool),
            Type::Int(range) => match ClosedNonEmptyRange::try_from(range.as_ref()) {
                Ok(closed_range) => Ok(HardwareType::Int(closed_range.cloned())),
                Err(_) => Err(NonHardwareType),
            },
            Type::Tuple(inner) => inner
                .iter()
                .map(|ty| ty.as_hardware_type(elab))
                .try_collect()
                .map(|v| HardwareType::Tuple(Arc::new(v))),
            Type::Array(inner, len) => inner
                .as_hardware_type(elab)
                .map(|inner| HardwareType::Array(Arc::new(inner), len.clone())),
            &Type::Struct(ty_struct) => {
                let info = elab.struct_info(ty_struct);
                match info.fields_hw {
                    Ok(_) => Ok(HardwareType::Struct(HardwareChecked::new_unchecked(ty_struct))),
                    Err(_) => Err(NonHardwareType),
                }
            }
            &Type::Enum(ty_enum) => {
                let info = elab.enum_info(ty_enum);
                match info.hw {
                    Ok(_) => Ok(HardwareType::Enum(HardwareChecked::new_unchecked(ty_enum))),
                    Err(_) => Err(NonHardwareType),
                }
            }
            Type::Type
            | Type::Any
            | Type::String
            | Type::Range
            | Type::Function
            | Type::Module
            | Type::Interface
            | Type::InterfaceView
            | Type::Builtin => Err(NonHardwareType),
        }
    }
}

impl HardwareType {
    pub fn as_type(&self) -> Type {
        match self {
            HardwareType::Undefined => Type::Undefined,
            HardwareType::Bool => Type::Bool,
            HardwareType::Int(range) => Type::Int(Range::from(range.clone())),
            HardwareType::Tuple(inner) => Type::Tuple(Arc::new(inner.iter().map(HardwareType::as_type).collect_vec())),
            HardwareType::Array(inner, len) => Type::Array(Arc::new(inner.as_type()), len.clone()),
            HardwareType::Struct(elab) => Type::Struct(elab.inner()),
            HardwareType::Enum(elab) => Type::Enum(elab.inner()),
        }
    }

    pub fn as_ir(&self, refs: CompileRefs) -> IrType {
        match self {
            HardwareType::Undefined => IrType::Tuple(vec![]),
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
                let info_hw = info.hw.as_ref().unwrap();

                let tag_ty = IrType::Int(info_hw.tag_range.clone());
                let data_ty = IrType::Array(Box::new(IrType::Bool), BigUint::from(info_hw.max_payload_size));
                IrType::Tuple(vec![tag_ty, data_ty])
            }
        }
    }
}

impl Typed for Never {
    fn ty(&self) -> Type {
        self.unreachable()
    }
}
