pub trait IterExt: Iterator {
    /// Get the single element of the iterator, if it has exactly one element.
    fn single(self) -> Option<Self::Item>;

    /// Variant of [Iterator::try_collect] or [itertools::Itertools::try_collect] without short-shortcutting.
    /// The entire iterator is always finished, even if an error is encountered.
    fn try_collect_all<T, F: FromIterator<T>, E>(self) -> Result<F, E>
    where
        Self: Iterator<Item = Result<T, E>>;

    fn try_collect_all_vec<T, E>(self) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
    {
        self.try_collect_all()
    }

    fn try_collect_vec<T, E>(self) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
    {
        self.collect()
    }

    fn with_first(self) -> impl Iterator<Item = (Self::Item, bool)>
    where
        Self: Sized,
    {
        let mut first = true;
        self.map(move |item| (item, std::mem::replace(&mut first, false)))
    }

    fn with_last(self) -> impl Iterator<Item = (Self::Item, bool)>
    where
        Self: Sized,
    {
        let mut iter = self.peekable();
        std::iter::from_fn(move || {
            if let Some(item) = iter.next() {
                let is_last = iter.peek().is_none();
                Some((item, is_last))
            } else {
                None
            }
        })
    }
}

impl<I: Iterator> IterExt for I {
    fn single(mut self) -> Option<Self::Item> {
        let first = self.next()?;
        match self.next() {
            None => Some(first),
            Some(_) => None,
        }
    }

    fn try_collect_all<T, F: FromIterator<T>, E>(self) -> Result<F, E>
    where
        Self: Iterator<Item = Result<T, E>>,
    {
        let mut any_err = Ok(());
        let result: F = self
            .filter_map(|x| {
                match x {
                    Ok(x) => {
                        if any_err.is_ok() {
                            // collect
                            Some(x)
                        } else {
                            // stop collecting (and allocating) once an error is encountered,
                            //  but keep iterating
                            None
                        }
                    }
                    Err(e) => {
                        // record error and don't collect
                        any_err = Err(e);
                        None
                    }
                }
            })
            .collect();

        any_err?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::util::iter::IterExt;

    #[test]
    fn iter_single() {
        assert_eq!(Option::<&i32>::None, [].iter().single());
        assert_eq!(Some(&0), [0].iter().single());
        assert_eq!(None, [0, 1].iter().single());
        assert_eq!(None, [0, 1, 2].iter().single());
    }
}
