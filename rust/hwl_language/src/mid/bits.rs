use crate::front::value::{CompileCompoundValue, CompileValue, SimpleCompileValue};
use crate::mid::ir::IrType;
use crate::util::big_int::BigUint;
use crate::util::int::IntRepresentation;
use crate::util::iter::IterExt;
use itertools::{zip_eq, Either, Itertools};
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
            IrType::Tuple(inner) => inner.iter().map(IrType::size_bits).sum(),
            IrType::Array(inner, len) => inner.size_bits() * len,
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
            (IrType::Tuple(ty_inners), CompileValue::Compound(CompileCompoundValue::Tuple(v_inners))) => {
                if ty_inners.len() != v_inners.len() {
                    return Err(ToBitsWrongType);
                }
                for (ty_inner, v_inner) in zip_eq(ty_inners.iter(), v_inners.iter()) {
                    ty_inner.value_to_bits_impl(v_inner, result)?;
                }
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
            IrType::Tuple(inners) => {
                let result = inners
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(bits))
                    .try_collect_vec()?;

                Ok(CompileValue::Compound(CompileCompoundValue::Tuple(result)))
            }
            IrType::Array(inner, len) => {
                let len = usize::try_from(len.clone()).map_err(|_| Either::Right(FromBitsWrongLength))?;
                let result = (0..len).map(|_| inner.value_from_bits_impl(bits)).try_collect()?;
                Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(result))))
            }
            IrType::Struct(_) => todo!(),
            IrType::Enum(_) => todo!(),
        }
    }
}
