use std::borrow::Borrow;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum ArcOrRef<'a, R: ?Sized + ToOwned + 'a> {
    Arc(Arc<R::Owned>),
    Ref(&'a R),
}

impl<'a, R: ?Sized + ToOwned + 'a> AsRef<R> for ArcOrRef<'a, R> {
    fn as_ref(&self) -> &R {
        match self {
            ArcOrRef::Arc(arc) => (&**arc).borrow(),
            ArcOrRef::Ref(r) => r,
        }
    }
}

impl<'a, R: ?Sized + ToOwned + 'a> Borrow<R> for ArcOrRef<'a, R> {
    fn borrow(&self) -> &R {
        self.as_ref()
    }
}
