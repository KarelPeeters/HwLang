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

pub fn bits_equal(a: &[u8], b: &[u8], bit_offset: usize, bit_len: usize) -> bool {
    (0..bit_len).all(|index| get_bit(a, bit_offset + index) == get_bit(b, bit_offset + index))
}
