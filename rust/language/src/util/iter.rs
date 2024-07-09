pub trait IterExt: Iterator {
    fn single(self) -> Option<Self::Item>;
}

impl<I: Iterator> IterExt for I {
    fn single(mut self) -> Option<Self::Item> {
        self.try_fold(None, |a, x| {
            match a {
                None => Ok(Some(x)),
                Some(_) => Err(()),
            }
        }).ok().flatten()
    }
}
