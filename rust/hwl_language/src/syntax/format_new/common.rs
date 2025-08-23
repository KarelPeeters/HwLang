use hwl_util::swrite;

#[derive(Debug, Copy, Clone)]
pub struct SourceTokenIndex(pub usize);

pub fn swrite_indent(f: &mut String, indent: usize) {
    swrite!(f, "    ");
    for _ in 0..indent {
        swrite!(f, " |  ")
    }
}
