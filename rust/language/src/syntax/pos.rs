use std::fmt::{Debug, Formatter};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct FileId(pub usize);

impl Debug for FileId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("[{}]", self.0))
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
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

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Span {
    //inclusive
    pub start: Pos,
    //exclusive
    pub end: Pos,
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        assert_eq!(self.start.file, self.end.file);
        write!(
            f,
            "{:?}{}:{}..{}:{}",
            self.start.file, self.start.line, self.start.col, self.end.line, self.end.col
        )
    }
}

impl FileId {
    pub const SINGLE: FileId = FileId(0);
    pub const DUMMY: FileId = FileId(usize::MAX);
}

impl Pos {
    #[must_use]
    pub fn step_over(self, s: &str) -> Pos {
        // TODO does this handle \r correctly?
        let mut result = self;

        for c in s.chars() {
            if c == '\n' {
                result.col = 1;
                result.line += 1;
            } else {
                result.col += 1;
            }
        }

        result
    }
}

impl Span {
    pub fn new(start: Pos, end: Pos) -> Self {
        Self { start, end }
    }

    pub fn empty_at(at: Pos) -> Self {
        Self::new(at, at)
    }
}

pub struct FileOffsets {
    file: FileId,
    total_bytes: usize,
    line_to_start_byte: Vec<usize>,
}

impl FileOffsets {
    pub fn new(file: FileId, src: &str) -> Self {
        let mut line_to_start_byte = vec![0];

        // iterating over bytes here is fine: we only care about the ascii newline
        for (i, b) in src.as_bytes().iter().copied().enumerate() {
            if b == b'\n' {
                // the next line starts after this byte
                line_to_start_byte.push(i + 1);
            }
        }

        FileOffsets {
            file,
            total_bytes: src.len(),
            line_to_start_byte
        }
    }

    pub fn byte_to_pos(&self, byte: usize) -> Pos {
        assert!(
            byte < self.total_bytes,
            "Byte {} out of range in file {:?} containing {} bytes", byte, self.file, self.total_bytes
        );
        let line_0 = self.line_to_start_byte.binary_search(&byte)
            .unwrap_or_else(|next_line_0| next_line_0 - 1);
        let col_0 = byte - self.line_to_start_byte[line_0];
        Pos {
            file: self.file,
            line: line_0 + 1,
            col: col_0 + 1,
        }
    }

    // short name, this is used a lot in the grammar
    pub fn span(&self, start_byte: usize, end_byte: usize) -> Span {
        let start = self.byte_to_pos(start_byte);
        let end = self.byte_to_pos(end_byte);
        Span { start, end }
    }
}
