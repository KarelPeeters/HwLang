use hwl_language::sim::recorder::WaveSignalType;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum WaveRadix {
    Bin,
    Hex,
    Dec,
}

pub fn enum_tag_width(variant_count: usize) -> usize {
    if variant_count <= 1 {
        0
    } else {
        usize::BITS as usize - (variant_count - 1).leading_zeros() as usize
    }
}

pub fn format_value_for_type_with_radix(
    bits: &[u8],
    bit_offset: usize,
    ty: &WaveSignalType,
    radix: WaveRadix,
) -> String {
    match ty {
        WaveSignalType::Bool => {
            if get_bit(bits, bit_offset) {
                "true".to_owned()
            } else {
                "false".to_owned()
            }
        }
        &WaveSignalType::Int { signed, width } => format_int_value(bits, bit_offset, width, signed, radix),
        WaveSignalType::Array { len, element } => {
            let stride = element.bit_len();
            let elements = (0..*len)
                .map(|index| format_value_for_type_with_radix(bits, bit_offset + index * stride, element, radix))
                .collect::<Vec<_>>();
            format!("[{}]", elements.join(", "))
        }
        WaveSignalType::Tuple(elements) => {
            let mut offset = bit_offset;
            let values = elements
                .iter()
                .map(|element| {
                    let value = format_value_for_type_with_radix(bits, offset, element, radix);
                    offset += element.bit_len();
                    value
                })
                .collect::<Vec<_>>();
            if values.len() == 1 {
                format!("({},)", values[0])
            } else {
                format!("({})", values.join(", "))
            }
        }
        WaveSignalType::Struct { name, fields } => {
            let mut offset = bit_offset;
            let values = fields
                .iter()
                .map(|(field_name, field_ty)| {
                    let value = format_value_for_type_with_radix(bits, offset, field_ty, radix);
                    offset += field_ty.bit_len();
                    format!("{field_name}={value}")
                })
                .collect::<Vec<_>>();
            format!("{name}.new({})", values.join(", "))
        }
        WaveSignalType::Enum { name, variants } => {
            let tag_width = enum_tag_width(variants.len());
            let tag = get_unsigned(bits, bit_offset, tag_width) as usize;
            let Some((variant_name, payload_ty)) = variants.get(tag) else {
                return format!("{name}.<invalid {tag}>");
            };
            match payload_ty {
                Some(payload_ty) => {
                    let payload = format_value_for_type_with_radix(bits, bit_offset + tag_width, payload_ty, radix);
                    format!("{name}.{variant_name}({payload})")
                }
                None => format!("{name}.{variant_name}"),
            }
        }
    }
}

fn format_int_value(bits: &[u8], bit_offset: usize, width: usize, signed: bool, radix: WaveRadix) -> String {
    if width == 0 {
        return "0".to_owned();
    }
    match radix {
        WaveRadix::Bin => return format!("0b{}", bit_string(bits, bit_offset, width)),
        WaveRadix::Hex => return format_hex_value(bits, bit_offset, width),
        WaveRadix::Dec => {}
    }
    if width > 128 {
        return format!("0x{:x}...", get_unsigned(bits, bit_offset, 128));
    }
    let value = get_unsigned(bits, bit_offset, width);
    if signed && width < 128 && get_bit(bits, bit_offset + width - 1) {
        let signed_value = value as i128 - (1i128 << width);
        signed_value.to_string()
    } else {
        value.to_string()
    }
}

fn bit_string(bits: &[u8], bit_offset: usize, width: usize) -> String {
    (0..width)
        .rev()
        .map(|index| if get_bit(bits, bit_offset + index) { '1' } else { '0' })
        .collect()
}

fn format_hex_value(bits: &[u8], bit_offset: usize, width: usize) -> String {
    if width > 128 {
        return format!("0x{:x}...", get_unsigned(bits, bit_offset, 128));
    }
    let value = get_unsigned(bits, bit_offset, width);
    let digits = width.div_ceil(4).max(1);
    format!("0x{value:0digits$x}")
}

pub fn get_unsigned(bits: &[u8], bit_offset: usize, bit_len: usize) -> u128 {
    let mut value = 0u128;
    for i in 0..bit_len.min(128) {
        if get_bit(bits, bit_offset + i) {
            value |= 1u128 << i;
        }
    }
    value
}

pub fn get_bit(bits: &[u8], bit: usize) -> bool {
    bits.get(bit / 8).is_some_and(|byte| ((byte >> (bit % 8)) & 1) != 0)
}

#[cfg(test)]
mod tests {
    use super::{WaveRadix, format_value_for_type_with_radix};
    use hwl_language::sim::recorder::WaveSignalType;

    #[test]
    fn radix_formatting_changes_integer_display() {
        let ty = WaveSignalType::Int {
            signed: false,
            width: 8,
        };
        let bits = [0xab];

        assert_eq!(format_value_for_type_with_radix(&bits, 0, &ty, WaveRadix::Dec), "171");
        assert_eq!(format_value_for_type_with_radix(&bits, 0, &ty, WaveRadix::Hex), "0xab");
        assert_eq!(
            format_value_for_type_with_radix(&bits, 0, &ty, WaveRadix::Bin),
            "0b10101011"
        );
    }
}
