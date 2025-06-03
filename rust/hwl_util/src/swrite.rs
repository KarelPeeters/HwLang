pub trait StringMut {
    fn as_mut_string(&mut self) -> &mut String;
}

impl StringMut for String {
    fn as_mut_string(&mut self) -> &mut String {
        self
    }
}

impl<T: StringMut> StringMut for &mut T {
    fn as_mut_string(&mut self) -> &mut String {
        (*self).as_mut_string()
    }
}

/// Variant of write! that only works for strings, and doesn't return a spurious error.
#[macro_export]
macro_rules! swrite {
    ($dst:expr, $($arg:tt)*) => {{
        use std::fmt::Write;
        use $crate::swrite::StringMut;
        let dst = $dst.as_mut_string();
        write!(dst, $($arg)*).unwrap();
    }};
}

/// Variant of writeln! that only works for strings, and doesn't return a spurious error.
#[macro_export]
macro_rules! swriteln {
    ($dst:expr $(,)?) => {{
        use std::fmt::Write;
        use $crate::swrite::StringMut;
        let dst = $dst.as_mut_string();
        writeln!(dst).unwrap();
    }};
    ($dst:expr, $($arg:tt)*) => {{
        use std::fmt::Write;
        use $crate::swrite::StringMut;
        let dst = $dst.as_mut_string();
        writeln!(dst, $($arg)*).unwrap();
    }};
}
