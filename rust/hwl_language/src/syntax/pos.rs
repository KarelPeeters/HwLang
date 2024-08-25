use std::cmp::{max, min};
use std::fmt::{Debug, Formatter};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FileId(pub usize);

/// Minimal source code position.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
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
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
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
        assert_eq!(start.file, end.file);
        Self { start, end }
    }

    pub fn empty_at(at: Pos) -> Self {
        Self::new(at, at)
    }

    pub fn contains(self, other: Span) -> bool {
        assert_eq!(self.start.file, other.start.file);
        self.start.byte <= other.start.byte && other.end.byte <= self.end.byte
    }

    pub fn join(self, other: Span) -> Span {
        assert_eq!(self.start.file, other.start.file);
        let file = self.start.file;
        Span {
            start: Pos { file, byte: min(self.start.byte, other.start.byte) },
            end: Pos { file, byte: max(self.end.byte, other.end.byte) },
        }
    }

    pub fn len_bytes(self) -> usize {
        self.end.byte - self.start.byte
    }

    pub fn range_bytes(self) -> std::ops::Range<usize> {
        self.start.byte..self.end.byte
    }
}

impl PosFull {
    pub fn pos(self) -> Pos {
        Pos {
            file: self.file,
            byte: self.byte,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DifferentFile;

impl SpanFull {
    pub fn span(self) -> Span {
        Span {
            start: self.start.pos(),
            end: self.end.pos(),
        }
    }

    pub fn distance_lines(self, other: SpanFull) -> Result<usize, DifferentFile> {
        if self.start.file == other.start.file {
            // get proper end-exclusive line ranges
            let self_end_line_0 = self.end.line_0 + ((self.end.col_0 > 0) as usize);
            let other_end_line_0 = other.end.line_0 + ((other.end.col_0 > 0) as usize);

            if self_end_line_0 < other.start.line_0 {
                // self is fully before other
                Ok(other.start.line_0 - self_end_line_0)
            } else if self.start.line_0 > other.end.line_0 {
                // self is fully after other
                Ok(self.start.line_0 - other_end_line_0)
            } else {
                // overlapping
                Ok(0)
            }
        } else {
            Err(DifferentFile)
        }
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

pub struct LineOffsets {
    total_bytes: usize,
    line_to_start_byte: Vec<usize>,
}

impl LineOffsets {
    // This set of line endings was chosen because it matches what the LSP protocol wants,
    // and we don't really care that much about the specifics.
    pub const LINE_ENDINGS: &'static [&'static str] = &["\r\n", "\n", "\r"];

    pub fn new(src: &str) -> Self {
        let mut line_to_start_byte = vec![0];

        for (i, b) in src.as_bytes().iter().copied().enumerate() {
            let b_next = src.as_bytes().get(i + 1).copied();
            if b == b'\n' || (b == b'\r' && b_next != Some(b'\n')) {
                // the next line starts _after_ this byte
                line_to_start_byte.push(i + 1);
            }
        }

        LineOffsets {
            total_bytes: src.len(),
            line_to_start_byte
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn line_count(&self) -> usize {
        self.line_to_start_byte.len()
    }

    pub fn full_span(&self, file: FileId) -> Span {
        Span {
            start: Pos { file, byte: 0 },
            end: Pos { file, byte: self.total_bytes },
        }
    }

    pub fn expand_pos(&self, pos: Pos) -> PosFull {
        // OPTIMIZE: maybe cache the last lookup and check its neighborhood first
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
        // OPTIMIZE: the second position must come after the first and is probably close
        SpanFull {
            start: self.expand_pos(span.start),
            end: self.expand_pos(span.end),
        }
    }

    pub fn line_start_byte(&self, line_0: usize) -> usize {
        self.line_to_start_byte[line_0]
    }
}
