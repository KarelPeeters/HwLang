use std::fmt::{Debug, Formatter};

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct FileId(pub usize);

impl Debug for FileId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("[{}]", self.0))
    }
}

#[derive(Copy, Clone)]
pub struct Pos {
    pub file: FileId,
    pub line: usize,
    pub col: usize,
}

impl Debug for Pos {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}{}:{}", self.file, self.line, self.col))
    }
}

#[derive(Copy, Clone)]
pub struct Span {
    //inclusive
    pub start: Pos,
    //exclusive
    pub end: Pos,
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        assert!(self.start.file == self.end.file);
        write!(f, "{:?}{}:{}..{}:{}",
               self.start.file,
               self.start.line, self.start.col,
               self.end.line, self.end.col
        )
    }
}

impl Span {
    pub fn new(start: Pos, end: Pos) -> Self {
        Self { start, end }
    }

    pub fn empty_at(at: Pos) -> Self {
        Self::new(at, at)
    }

    pub fn dummy() -> Self {
        Span::empty_at(Pos {
            file: FileId(usize::MAX),
            line: 0,
            col: 0,
        })
    }
}

pub fn byte_offset_to_pos(src: &str, offset: usize, file: FileId) -> Option<Pos> {
    let mut line = 0;
    let mut col = 0;
    let mut bytes = 0;

    for c in src.chars() {
        let mut buf = [0; 4];
        bytes += c.encode_utf8(&mut buf).len();

        if c == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }

        if bytes == offset {
            return Some(Pos { file, line: line + 1, col: col + 1 });
        }
        if bytes > offset {
            return None;
        }
    }

    None
}
