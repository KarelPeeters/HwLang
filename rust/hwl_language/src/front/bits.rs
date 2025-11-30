use crate::front::compile::CompileRefs;
use crate::front::diagnostic::DiagResult;
use crate::front::types::{ClosedIncRange, HardwareType};
use crate::front::value::{CompileCompoundValue, CompileValue, EnumValue, SimpleCompileValue, StructValue};
use crate::mid::ir::IrType;
use crate::syntax::pos::Span;
use crate::util::ResultExt;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::data::{EmptyVec, NonEmptyVec};
use crate::util::int::{IntRepresentation, InvalidRange};
use crate::util::iter::IterExt;
use itertools::{Itertools, zip_eq};
use std::sync::Arc;
use unwrap_match::unwrap_match;

// TODO optimize all of this, treating bit vectors as arrays of booleans is very slow and wasteful
impl HardwareType {
    pub fn size_bits(&self, refs: CompileRefs) -> BigUint {
        // TODO cache or precompute this value for structs and enums
        self.as_ir(refs).size_bits()
    }

    pub fn every_bit_pattern_is_valid(&self, refs: CompileRefs) -> bool {
        match self {
            HardwareType::Undefined => true,
            HardwareType::Bool => true,
            HardwareType::Int(range) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                &repr.range() == range
            }
            HardwareType::Tuple(inner) => inner.iter().all(|ty| ty.every_bit_pattern_is_valid(refs)),
            HardwareType::Array(inner, _len) => inner.every_bit_pattern_is_valid(refs),
            &HardwareType::Struct(elab) => {
                let info = refs.shared.elaboration_arenas.struct_info(elab.inner());
                let fields_hw = info.fields_hw.as_ref_ok().unwrap();
                fields_hw.iter().all(|ty| ty.every_bit_pattern_is_valid(refs))
            }
            &HardwareType::Enum(elab) => {
                let info = refs.shared.elaboration_arenas.enum_info(elab.inner());
                let info_hw = info.hw.as_ref().unwrap();

                // The tag needs to be fully valid.
                let tag_range = ClosedIncRange {
                    start_inc: BigInt::ZERO,
                    end_inc: BigInt::from(info.variants.len()) - 1,
                };
                let tag_repr = IntRepresentation::for_range(tag_range.as_ref());
                if tag_repr.range() != tag_range {
                    return false;
                }

                // We don't need all variants to be the same size:
                //   the bits they don't cover will never be used, since the tag should be checked first.
                // Each variant being valid individually is enough.
                info_hw
                    .payload_types
                    .iter()
                    .filter_map(Option::as_ref)
                    .all(|(ty, _)| ty.every_bit_pattern_is_valid(refs))
            }
        }
    }

    pub fn value_to_bits(&self, refs: CompileRefs, span: Span, value: &CompileValue) -> DiagResult<Vec<bool>> {
        let mut result = Vec::new();
        self.value_to_bits_impl(refs, span, value, &mut result)?;
        Ok(result)
    }

    fn value_to_bits_impl(
        &self,
        refs: CompileRefs,
        span: Span,
        value: &CompileValue,
        result: &mut Vec<bool>,
    ) -> DiagResult {
        let diags = refs.diags;
        let err_internal = || {
            diags.report_internal_error(
                span,
                format!(
                    "failed to convert value `{}` to bits of type `{}`",
                    value.value_string(&refs.shared.elaboration_arenas),
                    self.diagnostic_string()
                ),
            )
        };
        match (self, value) {
            (HardwareType::Bool, &CompileValue::Simple(SimpleCompileValue::Bool(value))) => {
                result.push(value);
                Ok(())
            }
            (HardwareType::Int(range), CompileValue::Simple(SimpleCompileValue::Int(value))) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                repr.value_to_bits(value, result).map_err(|_| err_internal())
            }
            (HardwareType::Tuple(ty_inners), CompileValue::Compound(CompileCompoundValue::Tuple(value))) => {
                assert_eq!(ty_inners.len(), value.len());
                for (ty_inner, v_inner) in zip_eq(ty_inners.iter(), value.iter()) {
                    ty_inner.value_to_bits_impl(refs, span, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Array(ty_inner, ty_len), CompileValue::Simple(SimpleCompileValue::Array(value))) => {
                assert_eq!(ty_len, &BigUint::from(value.len()));
                for v_inner in value.iter() {
                    ty_inner.value_to_bits_impl(refs, span, v_inner, result)?;
                }
                Ok(())
            }
            (&HardwareType::Struct(elab_ty), CompileValue::Compound(CompileCompoundValue::Struct(value))) => {
                let &StructValue { ty, ref fields } = value;
                if elab_ty.inner() != ty {
                    return Err(err_internal());
                }
                let info = refs.shared.elaboration_arenas.struct_info(elab_ty.inner());
                let fields_hw = info.fields_hw.as_ref_ok().unwrap();

                for (ty_inner, v_inner) in zip_eq(fields_hw, fields.iter()) {
                    ty_inner.value_to_bits_impl(refs, span, v_inner, result)?;
                }
                Ok(())
            }
            (HardwareType::Enum(elab_ty), CompileValue::Compound(CompileCompoundValue::Enum(value))) => {
                let &EnumValue {
                    ty,
                    variant,
                    ref payload,
                } = value;
                if elab_ty.inner() != ty {
                    return Err(err_internal());
                }
                let info = refs.shared.elaboration_arenas.enum_info(elab_ty.inner());
                let info_hw = info.hw.as_ref().unwrap();

                // tag
                let tag_range = info_hw.tag_range();
                HardwareType::Int(tag_range).value_to_bits_impl(
                    refs,
                    span,
                    &CompileValue::new_int(BigInt::from(variant)),
                    result,
                )?;

                // content
                if let Some(payload) = payload {
                    let (payload_ty, _) = info_hw.payload_types[variant].as_ref().unwrap();
                    payload_ty.value_to_bits_impl(refs, span, payload, result)?;
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
            ) => Err(err_internal()),
        }
    }

    pub fn value_from_bits(&self, refs: CompileRefs, span: Span, bits: &[bool]) -> DiagResult<CompileValue> {
        let diags = refs.diags;

        let mut iter = bits.iter().copied();
        let result = self.value_from_bits_impl(refs, span, &mut iter)?;
        match iter.next() {
            None => Ok(result),
            Some(_) => Err(diags.report_internal_error(span, "leftover bits when converting to value")),
        }
    }

    fn value_from_bits_impl(
        &self,
        refs: CompileRefs,
        span: Span,
        bits: &mut impl Iterator<Item = bool>,
    ) -> DiagResult<CompileValue> {
        let diags = refs.diags;

        // TODO this is not always an internal error, eg. if the bits don't form a valid value this should just be
        //   a normal error
        let err_internal = || {
            diags.report_internal_error(
                span,
                format!("failed to convert bits to value of type `{}`", self.diagnostic_string()),
            )
        };

        match self {
            HardwareType::Undefined => Err(err_internal()),
            HardwareType::Bool => {
                let bit = bits.next().ok_or_else(err_internal)?;
                Ok(CompileValue::new_bool(bit))
            }
            HardwareType::Int(range) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                let bits: Vec<bool> = (0..repr.size_bits())
                    .map(|_| bits.next().ok_or_else(err_internal))
                    .try_collect()?;

                let result = repr.value_from_bits(&bits).map_err(|_| err_internal())?;

                if range.contains(&result) {
                    Ok(CompileValue::new_int(result))
                } else {
                    Err(err_internal())
                }
            }
            HardwareType::Tuple(inners) => {
                let result = inners
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(refs, span, bits))
                    .try_collect_vec()?;
                match NonEmptyVec::try_from(result) {
                    Ok(result) => Ok(CompileValue::Compound(CompileCompoundValue::Tuple(result))),
                    Err(EmptyVec) => Ok(CompileValue::unit()),
                }
            }
            HardwareType::Array(inner, len) => {
                let len = usize::try_from(len).map_err(|_| err_internal())?;
                let result = (0..len)
                    .map(|_| inner.value_from_bits_impl(refs, span, bits))
                    .try_collect_vec()?;
                Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(result))))
            }
            HardwareType::Struct(elab) => {
                let info = refs.shared.elaboration_arenas.struct_info(elab.inner());
                let fields = info.fields_hw.as_ref_ok().unwrap();

                let result_fields = fields
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(refs, span, bits))
                    .try_collect()?;

                let result = StructValue {
                    ty: elab.inner(),
                    fields: result_fields,
                };
                Ok(CompileValue::Compound(CompileCompoundValue::Struct(result)))
            }
            HardwareType::Enum(elab) => {
                let info = refs.shared.elaboration_arenas.enum_info(elab.inner());
                let info_hw = info.hw.as_ref().unwrap();

                // tag
                let tag_ty = HardwareType::Int(info_hw.tag_range());
                let tag_value = tag_ty.value_from_bits_impl(refs, span, bits)?;
                let tag_value = unwrap_match!(tag_value, CompileValue::Simple(SimpleCompileValue::Int(v)) => v);
                let tag_value = usize::try_from(tag_value).unwrap();

                // content
                let payload_ty = info_hw.payload_types[tag_value].as_ref();
                let payload_value = if let Some((content_ty, _)) = payload_ty {
                    let content_value = content_ty.value_from_bits_impl(refs, span, bits)?;
                    Some(Box::new(content_value))
                } else {
                    None
                };

                // discard padding
                for _ in 0..info_hw.padding_for_variant(tag_value) {
                    bits.next().ok_or_else(err_internal)?;
                }

                let result = EnumValue {
                    ty: elab.inner(),
                    variant: tag_value,
                    payload: payload_value,
                };
                Ok(CompileValue::Compound(CompileCompoundValue::Enum(result)))
            }
        }
    }
}

#[derive(Debug)]
pub struct WrongType;

impl IrType {
    pub fn size_bits(&self) -> BigUint {
        match self {
            IrType::Bool => BigUint::ONE,
            IrType::Int(range) => BigUint::from(IntRepresentation::for_range(range.as_ref()).size_bits()),
            IrType::Tuple(inner) => inner.iter().map(IrType::size_bits).sum(),
            IrType::Array(inner, len) => inner.size_bits() * len,
        }
    }

    // TODO is there a way to avoid all of this code duplication with HardwareType?
    //   Maybe we just need to move some more hardware type logic into the IR backend?
    pub fn value_to_bits(&self, value: &CompileValue) -> Result<Vec<bool>, WrongType> {
        let mut result = Vec::new();
        self.value_to_bits_impl(value, &mut result)?;
        Ok(result)
    }

    fn value_to_bits_impl(&self, value: &CompileValue, result: &mut Vec<bool>) -> Result<(), WrongType> {
        match (self, value) {
            (IrType::Bool, CompileValue::Simple(SimpleCompileValue::Bool(v))) => {
                result.push(*v);
                Ok(())
            }
            (IrType::Int(range), CompileValue::Simple(SimpleCompileValue::Int(v))) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                repr.value_to_bits(v, result).map_err(|_: InvalidRange| WrongType)
            }
            (IrType::Tuple(ty_inners), CompileValue::Compound(CompileCompoundValue::Tuple(v_inners))) => {
                if ty_inners.len() != v_inners.len() {
                    return Err(WrongType);
                }
                for (ty_inner, v_inner) in zip_eq(ty_inners.iter(), v_inners.iter()) {
                    ty_inner.value_to_bits_impl(v_inner, result)?;
                }
                Ok(())
            }
            (IrType::Array(ty_inner, ty_len), CompileValue::Simple(SimpleCompileValue::Array(v_inner))) => {
                if ty_len != &BigUint::from(v_inner.len()) {
                    return Err(WrongType);
                }
                for v_inner in v_inner.iter() {
                    ty_inner.value_to_bits_impl(v_inner, result)?;
                }
                Ok(())
            }

            // type mismatches
            (_, _) => Err(WrongType),
        }
    }

    pub fn value_from_bits(&self, bits: &[bool]) -> Result<CompileValue, WrongType> {
        let mut iter = bits.iter().copied();
        let result = self.value_from_bits_impl(&mut iter)?;
        match iter.next() {
            None => Ok(result),
            Some(_) => Err(WrongType),
        }
    }

    fn value_from_bits_impl(&self, bits: &mut impl Iterator<Item = bool>) -> Result<CompileValue, WrongType> {
        match self {
            IrType::Bool => Ok(CompileValue::new_bool(bits.next().ok_or(WrongType)?)),
            IrType::Int(range) => {
                let repr = IntRepresentation::for_range(range.as_ref());
                let bits: Vec<bool> = (0..repr.size_bits())
                    .map(|_| bits.next().ok_or(WrongType))
                    .try_collect()?;

                let result = repr.value_from_bits(&bits).map_err(|_| WrongType)?;

                if range.contains(&result) {
                    Ok(CompileValue::new_int(result))
                } else {
                    Err(WrongType)
                }
            }
            IrType::Tuple(inners) => {
                let result = inners
                    .iter()
                    .map(|inner| inner.value_from_bits_impl(bits))
                    .try_collect_vec()?;

                match NonEmptyVec::try_from(result) {
                    Ok(result) => Ok(CompileValue::Compound(CompileCompoundValue::Tuple(result))),
                    Err(EmptyVec) => Ok(CompileValue::unit()),
                }
            }
            IrType::Array(inner, len) => {
                let len = usize::try_from(len).map_err(|_| WrongType)?;
                let result = (0..len).map(|_| inner.value_from_bits_impl(bits)).try_collect()?;
                Ok(CompileValue::Simple(SimpleCompileValue::Array(Arc::new(result))))
            }
        }
    }
}
