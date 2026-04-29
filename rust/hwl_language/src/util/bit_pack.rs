use num_integer::div_ceil;

pub fn bit_buffer_size_bytes(size_bits: usize) -> usize {
    match size_bits {
        0 => 0,
        1..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        33..=64 => 8,
        65.. => div_ceil(size_bits, 32) * 4,
    }
}

pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut buffer = vec![0u8; bit_buffer_size_bytes(bits.len())];
    pack_bits_into(bits, &mut buffer);
    buffer
}

pub fn pack_bits_into(bits: &[bool], buffer: &mut [u8]) {
    buffer.fill(0);
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            buffer[i / 8] |= 1 << (i % 8);
        }
    }
}

pub fn unpack_bits(buffer: &[u8], size_bits: usize) -> Vec<bool> {
    (0..size_bits).map(|i| packed_bit(buffer, i)).collect()
}

pub fn packed_unsigned(bits: &[u8], bit_offset: usize, bit_len: usize) -> u128 {
    let mut value = 0u128;
    for i in 0..bit_len.min(128) {
        if packed_bit(bits, bit_offset + i) {
            value |= 1u128 << i;
        }
    }
    value
}

pub fn packed_bit(bits: &[u8], bit: usize) -> bool {
    bits.get(bit / 8).is_some_and(|byte| ((byte >> (bit % 8)) & 1) != 0)
}

pub fn packed_bits_equal(a: &[u8], b: &[u8], bit_offset: usize, bit_len: usize) -> bool {
    (0..bit_len).all(|index| packed_bit(a, bit_offset + index) == packed_bit(b, bit_offset + index))
}

#[cfg(test)]
mod tests {
    use crate::util::bit_pack::{
        bit_buffer_size_bytes, pack_bits, pack_bits_into, packed_bits_equal, packed_unsigned, unpack_bits,
    };

    #[test]
    fn bit_buffer_size_matches_simulator_abi_chunks() {
        assert_eq!(bit_buffer_size_bytes(0), 0);
        assert_eq!(bit_buffer_size_bytes(1), 1);
        assert_eq!(bit_buffer_size_bytes(8), 1);
        assert_eq!(bit_buffer_size_bytes(9), 2);
        assert_eq!(bit_buffer_size_bytes(16), 2);
        assert_eq!(bit_buffer_size_bytes(17), 4);
        assert_eq!(bit_buffer_size_bytes(32), 4);
        assert_eq!(bit_buffer_size_bytes(33), 8);
        assert_eq!(bit_buffer_size_bytes(64), 8);
        assert_eq!(bit_buffer_size_bytes(65), 12);
    }

    #[test]
    fn pack_and_unpack_round_trip_lsb_first() {
        let bits = vec![true, false, true, true, false, false, false, true, true, false];
        let packed = pack_bits(&bits);

        assert_eq!(packed, vec![0b1000_1101, 0b0000_0001]);
        assert_eq!(unpack_bits(&packed, bits.len()), bits);
    }

    #[test]
    fn pack_bits_into_clears_existing_buffer() {
        let mut buffer = vec![0xff; 4];

        pack_bits_into(&[true, false, true], &mut buffer);

        assert_eq!(buffer, vec![0b0000_0101, 0, 0, 0]);
    }

    #[test]
    fn packed_unsigned_and_equal_support_offsets() {
        let bits = pack_bits(&[false, true, false, true, true, false]);
        let same = pack_bits(&[true, true, false, true, true, true]);
        let different = pack_bits(&[false, true, false, false, true, false]);

        assert_eq!(packed_unsigned(&bits, 1, 4), 0b1101);
        assert!(packed_bits_equal(&bits, &same, 1, 4));
        assert!(!packed_bits_equal(&bits, &different, 1, 4));
    }
}
