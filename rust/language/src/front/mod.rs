use crate::front::types::Type;
use crate::front::values::Value;

pub mod types;
pub mod values;
pub mod param;
pub mod scope;
pub mod driver;
pub mod error;

#[derive(Debug, Clone)]
pub enum TypeOrValue<T = Type, V = Value> {
    Type(T),
    Value(V),
}