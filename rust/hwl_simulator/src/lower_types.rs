use crate::buffer::Buffer;
use crate::simulator::{LowerError, LowerResult};
use hwl_language::front::value::{CompileValue, SimpleCompileValue};
use hwl_language::mid::ir::IrType;
use hwl_language::syntax::pos::Span;
use hwl_language::util::big_int::BigInt;
use hwl_language::util::int_repr::IntRepresentation;
use hwl_language::util::range::ClosedNonEmptyRange;
use inkwell::context::Context;
use inkwell::types::{BasicTypeEnum, IntType};
use std::num::NonZeroU32;
use unwrap_match::unwrap_match;

pub fn lower_ty<'ctx>(ctx: &'ctx Context, ty: &IrType) -> LowerResult<BasicTypeEnum<'ctx>> {
    // TODO cache these? maybe that's even necessary for struct types
    // TODO optimizations:
    //   for structs and tuples, re-order to minimize size?
    //   for all compound types: bit pack? careful about multithreading!
    match ty {
        IrType::Bool => Ok(ctx.bool_type().into()),
        IrType::Int(range) => Ok(lower_ty_int(ctx, range.as_ref())?.1.into()),
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}

/// Integer types are represented as the LLVM type of the same width,
/// except for zero-sized integers, which are represented as single-bit integers with value zero.
///
/// The LLVM memory representation is has as its size the minimum number of bytes necessary to store the number of bits,
/// any extra high bits must be zero.
/// LLVM store operations do this automatically and LLVM load operations require that this invariant is respected.
pub fn lower_ty_int<'ctx>(
    ctx: &'ctx Context,
    range: ClosedNonEmptyRange<&BigInt>,
) -> LowerResult<(IntRepresentation, IntType<'ctx>)> {
    let repr = IntRepresentation::for_range(range);
    let size_bits = repr.size_bits();

    let size_bits = u32::try_from(size_bits).map_err::<LowerError, _>(|_| todo!())?;
    let result = match NonZeroU32::new(size_bits) {
        // zero size: pad to one bit, to allow this type to still be represented as normal
        None => ctx.bool_type(),
        // nonzero size
        Some(size_bits) => ctx
            .custom_width_int_type(size_bits)
            .map_err::<LowerError, _>(|_| todo!())?,
    };

    Ok((repr, result))
}

pub fn usize_to_u31(span: Span, value: usize) -> LowerResult<u32> {
    if value < i32::MAX as usize {
        Ok(value as u32)
    } else {
        Err(LowerError::IntTooLarge(span, value.into()))
    }
}

pub unsafe fn read_value_from_buffer(buffer: &Buffer, base_offset: usize, ty: &IrType) -> CompileValue {
    match ty {
        IrType::Bool => {
            let value = unsafe { buffer.read::<u8>(base_offset) };

            match value {
                0 => CompileValue::new_bool(false),
                1 => CompileValue::new_bool(true),
                _ => unreachable!("invalid boolean value"),
            }
        }
        IrType::Int(range) => {
            // TODO this is really inefficient, make this block-wise
            // TODO endianness?
            let repr = IntRepresentation::for_range(range.as_ref());

            // read bits
            let mut bits = vec![];
            for i_bit in 0..repr.size_bits() {
                let i_byte = (i_bit / 8) as usize;
                let i_bit_in_byte = i_bit % 8;

                let byte = unsafe { buffer.read::<u8>(base_offset + i_byte) };
                let bit = (byte >> i_bit_in_byte) & 1 != 0;
                bits.push(bit);
            }

            // convert from bits
            CompileValue::new_int(repr.value_from_bits(&bits))
        }
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}

pub unsafe fn write_value_to_buffer(buffer: &mut Buffer, base_offset: usize, ty: &IrType, value: &CompileValue) {
    match ty {
        IrType::Bool => {
            let value = unwrap_match!(value, &CompileValue::Simple(SimpleCompileValue::Bool(value)) => value);
            unsafe { buffer.write::<u8>(base_offset, value as u8) };
        }
        IrType::Int(range) => {
            // TODO this is really inefficient, make this block-wise
            // TODO endianness?
            let repr = IntRepresentation::for_range(range.as_ref());
            let value = unwrap_match!(value, &CompileValue::Simple(SimpleCompileValue::Int(ref value)) => value);

            // convert to bits
            let mut bits = Vec::new();
            repr.value_to_bits(value, &mut bits);

            // write bytes
            let byte_count = repr.size_bits().div_ceil(8) as usize;
            for i_byte in 0..byte_count {
                let mut byte: u8 = 0;
                for i_bit in 0..8 {
                    let bit = *bits.get(i_byte * 8 + i_bit).unwrap_or(&false);
                    byte |= (bit as u8) << i_bit;
                }
                unsafe { buffer.write::<u8>(base_offset + i_byte, byte) };
            }
        }
        IrType::Array(_, _) => todo!(),
        IrType::Tuple(_) => todo!(),
        IrType::Struct(_) => todo!(),
        IrType::Enum(_) => todo!(),
    }
}
