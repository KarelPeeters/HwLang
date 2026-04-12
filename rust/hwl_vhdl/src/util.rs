/// Convert bytes encoded in ISO8859-1 (also called Latin-1) to a normal Rust UTF-8 string.
pub fn string_from_bytes_iso8859_1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}
