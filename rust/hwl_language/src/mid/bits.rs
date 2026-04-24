use crate::front::value::{CompileCompoundValue, CompileValue, EnumValue, SimpleCompileValue, StructValue};
use crate::mid::ir::IrType;
use crate::util::big_int::BigUint;
use crate::util::int::IntRepresentation;
use crate::util::iter::IterExt;
use itertools::{Either, Itertools, zip_eq};
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
pub struct ToBitsWrongType;

#[derive(Debug, Copy, Clone)]
pub struct FromBitsWrongLength;

/// Error returned when a bit pattern is invalid.
/// This can happen when using `unsafe_from_bits` at compile time.
#[derive(Debug, Copy, Clone)]
pub struct FromBitsInvalidValue;

impl IrType {
    pub fn size_bits(&self) -> BigUint {
        match self {
            IrType::Bool => BigUint::ONE,
            IrType::Int(range) => BigUint::from(IntRepresentation::for_range(range.as_ref()).size_bits()),
            IrType::Array(inner, len) => inner.size_bits() * len,
            IrType::Tuple(inner) => inner.iter().map(IrType::size_bits).sum(),
            IrType::Struct(info) => info.size_bits(),
            IrType::Enum(info) => info.size_bits(),
        }
    }

    pub fn value_to_bits(&self, value: &CompileValue) -> Result<Vec<bool>, ToBitsWrongType> {
        let mut result = Vec::new();
        self.value_to_bits_impl(value, &mut result)?;
        Ok(result)
    }

    fn value_to_bits_impl(&self, value: &CompileValue, result: &mut Vec<bool>) -> Result<(), ToBitsWrongType> {
        match (self, value) {
            (IrType::Bool, CompileValue::Simple(SimpleCompileValue::Bool(v))) => {
                result.push(*v);
                Ok(())
            }
            (IrType::Int(range), CompileValue::Simple(SimpleCompileValue::Int(v))) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                repr.value_to_bits(v, result);
                Ok(())
            }
            (IrType::Array(ty_inner, ty_len), CompileValue::Simple(SimpleCompileValue::Array(v_inner))) => {
                if ty_len != &BigUint::from(v_inner.len()) {
                    return Err(ToBitsWrongType);
                }
                for v_inner in v_inner.iter() {
                    ty_inner.value_to_bits_impl(v_inner, result)?;
                }
                Ok(())
            }
            (IrType::Tuple(ty_inners), CompileValue::Compound(CompileCompoundValue::Tuple(v_inners))) => {
                if ty_inners.len() != v_inners.len() {
                    return Err(ToBitsWrongType);
                }
                for (ty_inner, v_inner) in zip_eq(ty_inners.iter(), v_inners.iter()) {
                    ty_inner.value_to_bits_impl(v_inner, result)?;
                }
                Ok(())
            }
            (IrType::Struct(ty_info), CompileValue::Compound(CompileCompoundValue::Struct(v_struct))) => {
                if ty_info.ty.inner() != v_struct.ty || ty_info.fields.len() != v_struct.fields.len() {
                    return Err(ToBitsWrongType);
                }
                for (ty_inner, v_inner) in zip_eq(ty_info.fields.values(), &v_struct.fields) {
                    ty_inner.value_to_bits_impl(v_inner, result)?;
                }
                Ok(())
            }
            (IrType::Enum(ty_info), CompileValue::Compound(CompileCompoundValue::Enum(v_enum))) => {
                if ty_info.ty.inner() != v_enum.ty || v_enum.variant >= ty_info.variants.len() {
                    return Err(ToBitsWrongType);
                }

                // tag
                IrType::Int(ty_info.tag_range())
                    .value_to_bits_impl(&CompileValue::new_int(v_enum.variant.into()), result)?;

                // payload
                let payload_ty = &ty_info.variants[v_enum.variant];
                match (payload_ty, &v_enum.payload) {
                    (Some(payload_ty), Some(payload)) => payload_ty.value_to_bits_impl(payload, result)?,
                    (None, None) => {}
                    _ => return Err(ToBitsWrongType),
                }

                // padding
                let payload_size = payload_ty.as_ref().map_or(BigUint::ZERO, IrType::size_bits);
                let padding_size = ty_info.max_payload_size_bits() - payload_size;
                result.extend(std::iter::repeat_n(
                    false,
                    usize::try_from(padding_size).map_err(|_| ToBitsWrongType)?,
                ));
                Ok(())
            }

            // type mismatches
            (_, _) => Err(ToBitsWrongType),
        }
    }

    pub fn value_from_bits(
        &self,
        bits: &[bool],
    ) -> Result<CompileValue, Either<FromBitsInvalidValue, FromBitsWrongLength>> {
        let mut iter = bits.iter().copied();
        let result = self.value_from_bits_impl(&mut iter)?;
        match iter.next() {
            None => Ok(result),
            Some(_) => Err(Either::Right(FromBitsWrongLength)),
        }
    }

    fn value_from_bits_impl(
        &self,
        bits: &mut impl Iterator<Item = bool>,
    ) -> Result<CompileValue, Either<FromBitsInvalidValue, FromBitsWrongLength>> {
        match self {
            IrType::Bool => Ok(CompileValue::new_bool(
                bits.next().ok_or(Either::Right(FromBitsWrongLength))?,
            )),
            IrType::Int(range) => {
                let repr = IntRepresentation::for_range(range.as_ref());
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
            IrType::Array(inner, len) => {
                let len = usize::try_from(len.clone()).map_err(|_| Either::Right(FromBitsWrongLength))?;
                let result = (0..len).map(|_| inner.value_from_bits_impl(bits)).try_collect()?;
                Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(result))))
            }
            IrType::Tuple(inners) => {
                let result = inners
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(bits))
                    .try_collect_vec()?;

                Ok(CompileValue::Compound(CompileCompoundValue::Tuple(result)))
            }
            IrType::Struct(info) => {
                let fields = info
                    .fields
                    .values()
                    .map(|field_ty| field_ty.value_from_bits_impl(bits))
                    .try_collect_vec()?;
                Ok(CompileValue::Compound(CompileCompoundValue::Struct(StructValue {
                    ty: info.ty.inner(),
                    fields,
                })))
            }
            IrType::Enum(info) => {
                // tag
                let tag = IrType::Int(info.tag_range()).value_from_bits_impl(bits)?;
                let tag = match tag {
                    CompileValue::Simple(SimpleCompileValue::Int(tag)) => tag,
                    _ => unreachable!("int decoding should always return a compile-time int"),
                };
                let variant = usize::try_from(tag).map_err(|_| Either::Left(FromBitsInvalidValue))?;

                // payload
                let payload_ty = info
                    .variants
                    .get_index(variant)
                    .ok_or(Either::Left(FromBitsInvalidValue))?
                    .1;
                let payload = match payload_ty {
                    Some(payload_ty) => Some(Box::new(payload_ty.value_from_bits_impl(bits)?)),
                    None => None,
                };

                // padding
                let payload_size = payload_ty.as_ref().map_or(BigUint::ZERO, IrType::size_bits);
                let padding_size = info.max_payload_size_bits() - payload_size;
                for _ in 0..usize::try_from(padding_size).map_err(|_| Either::Right(FromBitsWrongLength))? {
                    bits.next().ok_or(Either::Right(FromBitsWrongLength))?;
                }

                Ok(CompileValue::Compound(CompileCompoundValue::Enum(EnumValue {
                    ty: info.ty.inner(),
                    variant,
                    payload,
                })))
            }
        }
    }
}
