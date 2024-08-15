use std::fmt::{Debug, Formatter};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FileId(pub usize);

/// Minimal source code position.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Pos {
    pub file: FileId,
    pub byte: usize,
}

/// Expanded source code position.
///
/// The line and column are stored as zero-based,
/// whenever visible to the end user they should be displayed as one-based
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PosFull {
    pub file: FileId,
    pub byte: usize,
    pub line_0: usize,
    pub col_0: usize,
}

// TODO make this more compact, sharing the file?
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Span {
    /// inclusive
    pub start: Pos,
    /// exclusive
    pub end: Pos,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SpanFull {
    /// inclusive
    pub start: PosFull,
    /// exclusive
    pub end: PosFull,
}

impl FileId {
    pub const SINGLE: FileId = FileId(0);
    pub const DUMMY: FileId = FileId(usize::MAX);
}

impl Span {
    pub fn new(start: Pos, end: Pos) -> Self {
        Self { start, end }
    }

    pub fn empty_at(at: Pos) -> Self {
        Self::new(at, at)
    }
}

// Short debug implementations, these can appear a lot in AST debug outputs.
impl Debug for Pos {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Pos([{}]:{})", self.file.0, self.byte))
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        assert_eq!(self.start.file, self.end.file);
        f.write_fmt(format_args!("Span([{}]:{}..{})", self.start.file.0, self.start.byte, self.end.byte))
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

    pub fn file(&self) -> FileId {
        self.file
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn line_count(&self) -> usize {
        self.line_to_start_byte.len()
    }

    pub fn expand_pos(&self, pos: Pos) -> PosFull {
        assert_eq!(pos.file, self.file);
        let line_0 = self.line_to_start_byte.binary_search(&pos.byte)
            .unwrap_or_else(|next_line_0| next_line_0 - 1);
        let col_0 = pos.byte - self.line_to_start_byte[line_0];
        PosFull {
            file: pos.file,
            byte: pos.byte,
            line_0,
            col_0,
        }
    }

    pub fn expand_span(&self, span: Span) -> SpanFull {
        SpanFull {
            start: self.expand_pos(span.start),
            end: self.expand_pos(span.end),
        }
    }

    pub fn line_start_byte(&self, line_0: usize) -> usize {
        self.line_to_start_byte[line_0]
    }
}
