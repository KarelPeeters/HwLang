use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::parsed::AstRefItem;
use crate::new::types::Type;
use crate::new::value::Value;
use crate::util::Never;

// TODO move everything in this file to a better place
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum TypeOrValue<V> {
    Type(Type),
    Value(V),
    Error(ErrorGuaranteed),
}

#[derive(Debug, Clone)]
pub enum ScopedEntry {
    Item(AstRefItem),
    Direct(TypeOrValue<Value>),
}

impl<V> TypeOrValue<V> {
    pub fn map_value<W>(self, mut f: impl FnMut(V) -> W) -> TypeOrValue<W> {
        match self {
            TypeOrValue::Type(t) => TypeOrValue::Type(t),
            TypeOrValue::Value(v) => TypeOrValue::Value(f(v)),
            TypeOrValue::Error(e) => TypeOrValue::Error(e),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Unchecked(());

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum MaybeUnchecked<T, U = Unchecked> {
    Checked(T),
    Unchecked(U),
}

impl<T, U: Copy> MaybeUnchecked<T, U> {
    pub fn as_ref(&self) -> MaybeUnchecked<&T, U> {
        match self {
            MaybeUnchecked::Checked(t) => MaybeUnchecked::Checked(t),
            &MaybeUnchecked::Unchecked(u) => MaybeUnchecked::Unchecked(u),
        }
    }
}

impl<T> MaybeUnchecked<T, Never> {
    pub fn inner(self) -> T {
        match self {
            MaybeUnchecked::Checked(t) => t,
            MaybeUnchecked::Unchecked(u) => match u {}
        }
    }
}

impl<T> MaybeUnchecked<T, Unchecked> {
    pub fn require_checked(self) -> Result<MaybeUnchecked<T, Never>, Unchecked> {
        match self {
            MaybeUnchecked::Checked(t) => Ok(MaybeUnchecked::Checked(t)),
            MaybeUnchecked::Unchecked(u) => Err(u),
        }
    }
}

impl Unchecked {
    pub fn from_err(e: ErrorGuaranteed) -> Self {
        let _ = e;
        Unchecked(())
    }

    pub fn new_unchecked_param() -> Self {
        Unchecked(())
    }
}
