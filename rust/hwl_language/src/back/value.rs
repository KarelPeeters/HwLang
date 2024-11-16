use num_bigint::BigInt;

#[derive(Debug, Clone)]
pub enum BackValue {
    Undefined,
    Bool(bool),
    Int(BigInt),
    String(String),
}
