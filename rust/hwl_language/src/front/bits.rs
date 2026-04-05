use crate::front::compile::CompileRefs;
use crate::front::item::ElaborationArenas;
use crate::front::types::HardwareType;
use crate::front::value::{CompileCompoundValue, CompileValue, EnumValue, SimpleCompileValue, StructValue};
use hwl_common::util::ResultExt;
use hwl_common::util::big_int::{BigInt, BigUint};
use hwl_common::util::int::IntRepresentation;
use hwl_common::util::iter::IterExt;
use hwl_common::util::range_multi::{AnyMultiRange, ClosedNonEmptyMultiRange};
use itertools::{Either, Itertools, zip_eq};
use std::sync::Arc;
use unwrap_match::unwrap_match;

#[derive(Debug, Copy, Clone)]
pub struct ToBitsWrongType;

#[derive(Debug, Copy, Clone)]
pub struct FromBitsWrongLength;

/// Error returned when a bit pattern is invalid.
/// This can happen when using `unsafe_from_bits` at compile time.
#[derive(Debug, Copy, Clone)]
pub struct FromBitsInvalidValue;

// TODO optimize all of this, treating bit vectors as arrays of booleans is very slow and wasteful
impl HardwareType {
    pub fn size_bits(&self, refs: CompileRefs) -> BigUint {
        // TODO cache or precompute this value for structs and enums
        self.as_ir(refs).size_bits()
    }

    pub fn every_bit_pattern_is_valid(&self, elab: &ElaborationArenas) -> bool {
        match self {
            HardwareType::Undefined => true,
            HardwareType::Bool => true,
            HardwareType::Int(range) => {
                let repr = IntRepresentation::for_range(range.enclosing_range());
                &ClosedNonEmptyMultiRange::from(repr.range()) == range
            }
            HardwareType::Tuple(inner) => inner.iter().all(|ty| ty.every_bit_pattern_is_valid(elab)),
            HardwareType::Array(inner, _len) => inner.every_bit_pattern_is_valid(elab),
            &HardwareType::Struct(slf) => {
                let info = elab.struct_info(slf.inner());
                let fields_hw = info.fields_hw.as_ref_ok().unwrap();
                fields_hw.iter().all(|ty| ty.every_bit_pattern_is_valid(elab))
            }
            &HardwareType::Enum(slf) => {
                let info = elab.enum_info(slf.inner());
                let info_hw = info.hw.as_ref().unwrap();

                // The tag needs to be fully valid.
                let tag_repr = IntRepresentation::for_range(info_hw.tag_range.as_ref());
                if tag_repr.range() != info_hw.tag_range {
                    return false;
                }

                // TODO rethink this, this makes enum comparisons trickier,
                //   it would be nicer to enforce zero bits for padding.
                // We don't need all variants to be the same size:
                //   the bits they don't cover will never be used, since the tag should be checked first.
                // Each variant being valid individually is enough.
                info_hw
                    .payload_types
                    .iter()
                    .filter_map(Option::as_ref)
                    .all(|(ty, _)| ty.every_bit_pattern_is_valid(elab))
            }
        }
    }

    pub fn value_to_bits(&self, elab: &ElaborationArenas, value: &CompileValue) -> Result<Vec<bool>, ToBitsWrongType> {
        let mut result = Vec::new();
        self.value_to_bits_impl(elab, value, &mut result)?;
        Ok(result)
    }

    fn value_to_bits_impl(
        &self,
        elab: &ElaborationArenas,
        value: &CompileValue,
        result: &mut Vec<bool>,
    ) -> Result<(), ToBitsWrongType> {
        match (self, value) {
            (HardwareType::Bool, &CompileValue::Simple(SimpleCompileValue::Bool(value))) => {
                result.push(value);
                Ok(())
            }
            (HardwareType::Int(range), CompileValue::Simple(SimpleCompileValue::Int(value))) => {
                let repr = IntRepresentation::for_range(range.enclosing_range());
                repr.value_to_bits(value, result);
                Ok(())
            }
            (HardwareType::Tuple(ty_inners), CompileValue::Compound(CompileCompoundValue::Tuple(value))) => {
                if ty_inners.len() != value.len() {
                    return Err(ToBitsWrongType);
                }
                for (ty_inner, v_inner) in zip_eq(ty_inners.iter(), value.iter()) {
                    ty_inner.value_to_bits_impl(elab, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Array(ty_inner, ty_len), CompileValue::Simple(SimpleCompileValue::Array(value))) => {
                if ty_len != &BigUint::from(value.len()) {
                    return Err(ToBitsWrongType);
                }
                for v_inner in value.iter() {
                    ty_inner.value_to_bits_impl(elab, v_inner, result)?;
                }
                Ok(())
            }
            (&HardwareType::Struct(slf), CompileValue::Compound(CompileCompoundValue::Struct(value))) => {
                let &StructValue { ty, ref fields } = value;
                if slf.inner() != ty {
                    return Err(ToBitsWrongType);
                }
                let info = elab.struct_info(slf.inner());
                let fields_hw = info.fields_hw.as_ref_ok().unwrap();

                for (ty_inner, v_inner) in zip_eq(fields_hw, fields.iter()) {
                    ty_inner.value_to_bits_impl(elab, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Enum(slf), CompileValue::Compound(CompileCompoundValue::Enum(value))) => {
                let &EnumValue {
                    ty,
                    variant,
                    ref payload,
                } = value;
                if slf.inner() != ty {
                    return Err(ToBitsWrongType);
                }
                let info = elab.enum_info(slf.inner());
                let info_hw = info.hw.as_ref().unwrap();

                // tag
                let tag_range = info_hw.tag_range.clone();
                HardwareType::Int(ClosedNonEmptyMultiRange::from(tag_range)).value_to_bits_impl(
                    elab,
                    &CompileValue::new_int(BigInt::from(variant)),
                    result,
                )?;

                // content
                if let Some(payload) = payload {
                    let (payload_ty, _) = info_hw.payload_types[variant].as_ref().unwrap();
                    payload_ty.value_to_bits_impl(elab, payload, result)?;
                }

                // padding
                for _ in 0..info_hw.padding_for_variant(variant) {
                    result.push(false);
                }
                Ok(())
            }

            // type mismatches
            (
                HardwareType::Undefined
                | HardwareType::Bool
                | HardwareType::Int(_)
                | HardwareType::Tuple(_)
                | HardwareType::Array(_, _)
                | HardwareType::Struct(_)
                | HardwareType::Enum(_),
                _,
            ) => Err(ToBitsWrongType),
        }
    }

    pub fn value_from_bits(
        &self,
        elab: &ElaborationArenas,
        bits: &[bool],
    ) -> Result<CompileValue, Either<FromBitsInvalidValue, FromBitsWrongLength>> {
        let mut iter = bits.iter().copied();
        let result = self.value_from_bits_impl(elab, &mut iter)?;
        if iter.next().is_some() {
            return Err(Either::Right(FromBitsWrongLength));
        }
        Ok(result)
    }

    fn value_from_bits_impl(
        &self,
        elab: &ElaborationArenas,
        bits: &mut impl Iterator<Item = bool>,
    ) -> Result<CompileValue, Either<FromBitsInvalidValue, FromBitsWrongLength>> {
        match self {
            // TODO what should this do? can this ever happen?
            HardwareType::Undefined => Err(Either::Left(FromBitsInvalidValue)),

            HardwareType::Bool => {
                let bit = bits.next().ok_or(Either::Right(FromBitsWrongLength))?;
                Ok(CompileValue::new_bool(bit))
            }
            HardwareType::Int(range) => {
                let repr = IntRepresentation::for_range(range.enclosing_range());
                let bits: Vec<bool> = (0..repr.size_bits())
                    .map(|_| bits.next().ok_or(Either::Right(FromBitsWrongLength)))
                    .try_collect()?;
                let result = repr.value_from_bits(&bits);

                if range.contains(&result) {
                    Ok(CompileValue::new_int(result))
                } else {
                    Err(Either::Left(FromBitsInvalidValue))
                }
            }
            HardwareType::Tuple(inners) => {
                let result = inners
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(elab, bits))
                    .try_collect_vec()?;
                Ok(CompileValue::Compound(CompileCompoundValue::Tuple(result)))
            }
            HardwareType::Array(inner, len) => {
                let len = usize::try_from(len.clone()).map_err(|_| Either::Right(FromBitsWrongLength))?;
                let result = (0..len)
                    .map(|_| inner.value_from_bits_impl(elab, bits))
                    .try_collect_vec()?;
                Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(result))))
            }
            HardwareType::Struct(slf) => {
                let info = elab.struct_info(slf.inner());
                let fields = info.fields_hw.as_ref_ok().unwrap();

                let result_fields = fields
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(elab, bits))
                    .try_collect()?;

                let result = StructValue {
                    ty: slf.inner(),
                    fields: result_fields,
                };
                Ok(CompileValue::Compound(CompileCompoundValue::Struct(result)))
            }
            HardwareType::Enum(slf) => {
                let info = elab.enum_info(slf.inner());
                let info_hw = info.hw.as_ref().unwrap();

                // tag
                let tag_ty = HardwareType::Int(ClosedNonEmptyMultiRange::from(info_hw.tag_range.clone()));
                let tag_value = tag_ty.value_from_bits_impl(elab, bits)?;
                let tag_value = unwrap_match!(tag_value, CompileValue::Simple(SimpleCompileValue::Int(v)) => v);
                let tag_value = usize::try_from(tag_value).unwrap();

                // content
                let payload_ty = info_hw.payload_types[tag_value].as_ref();
                let payload_value = if let Some((content_ty, _)) = payload_ty {
                    let content_value = content_ty.value_from_bits_impl(elab, bits)?;
                    Some(Box::new(content_value))
                } else {
                    None
                };

                // discard padding
                for _ in 0..info_hw.padding_for_variant(tag_value) {
                    bits.next().ok_or(Either::Right(FromBitsWrongLength))?;
                }

                let result = EnumValue {
                    ty: slf.inner(),
                    variant: tag_value,
                    payload: payload_value,
                };
                Ok(CompileValue::Compound(CompileCompoundValue::Enum(result)))
            }
        }
    }
}
