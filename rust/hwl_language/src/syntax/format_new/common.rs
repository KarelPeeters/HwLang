use hwl_util::swrite;

pub fn swrite_indent(f: &mut String, indent: usize) {
    swrite!(f, "    ");
    for _ in 0..indent {
        swrite!(f, " |  ")
    }
}
