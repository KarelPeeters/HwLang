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
}

impl<I: Iterator> IterExt for I {
    fn single(mut self) -> Option<Self::Item> {
        self.try_fold(None, |a, x| match a {
            None => Ok(Some(x)),
            Some(_) => Err(()),
        })
        .ok()
        .flatten()
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
