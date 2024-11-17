#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Type<V> {
    Bool,
    Int(RangeInfo<V>),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RangeInfo<V> {
    start_inc: Option<V>,
    end_inc: Option<V>,
}