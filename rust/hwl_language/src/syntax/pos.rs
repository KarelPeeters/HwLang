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
#[derive(Copy, Clone, Eq, PartialEq)]
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

impl Debug for PosFull {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("PosFull([{}]:{}:{})", self.file.0, self.line_0 + 1, self.col_0 + 1))
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        assert_eq!(self.start.file, self.end.file);
        f.write_fmt(format_args!("Span([{}]:{}..{})", self.start.file.0, self.start.byte, self.end.byte))
    }
}

#[derive(Clone)]
pub struct LineOffsets {
    total_bytes: usize,
    line_info: Vec<LineInfo>,
}

// TODO get this more compact by using u32?
//   we can even get rid of the the 2_char bit and store a single default per file plus a set of exceptions
//   or maybe simpler, a separate bitset
#[derive(Debug, Copy, Clone)]
struct LineInfo(usize);

impl LineInfo {
    pub fn new(start_byte: usize, follows_2_char_ending: bool) -> Self {
        assert_eq!(start_byte & (1 << (usize::BITS - 1)), 0);
        LineInfo(start_byte | (follows_2_char_ending as usize) << (usize::BITS - 1))
    }

    /// The byte index at which this line starts.
    pub fn start_byte(&self) -> usize {
        self.0 & !(1 << (usize::BITS - 1))
    }

    /// Whether the previous line ends with "\r\n".
    pub fn follows_2_char_ending(&self) -> bool {
        self.0 & (1 << (usize::BITS - 1)) != 0
    }
}

impl LineOffsets {
    // This set of line endings was chosen because it matches what the LSP protocol wants,
    // and we don't really care that much about the specifics.
    pub const LINE_ENDINGS: &'static [&'static str] = &["\r\n", "\n", "\r"];

    // TODO switch to memchr, benchmark the difference
    pub fn new(src: &str) -> Self {
        let mut line_info = vec![LineInfo::new(0, false)];
        let mut follows_2_char_ending = false;

        for (i, b) in src.bytes().enumerate() {
            match b {
                b'\r' => {
                    let b_next = src.as_bytes().get(i + 1).copied();
                    if b_next == Some(b'\n') {
                        assert!(!follows_2_char_ending);
                        follows_2_char_ending = true;
                    } else {
                        line_info.push(LineInfo::new(i + 1, follows_2_char_ending));
                        follows_2_char_ending = false;
                    }
                }
                b'\n' => {
                    line_info.push(LineInfo::new(i + 1, follows_2_char_ending));
                    follows_2_char_ending = false;
                }
                _ => {}
            }
        }

        LineOffsets {
            total_bytes: src.len(),
            line_info
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn line_count(&self) -> usize {
        self.line_info.len()
    }

    pub fn full_span(&self, file: FileId) -> Span {
        Span {
            start: Pos { file, byte: 0 },
            end: Pos { file, byte: self.total_bytes },
        }
    }

    pub fn expand_pos(&self, pos: Pos) -> PosFull {
        // OPTIMIZE: maybe cache the last lookup and check its neighborhood first
        let line_0 = self.line_info.binary_search_by_key(&pos.byte, |info| info.start_byte())
            .unwrap_or_else(|next_line_0| next_line_0 - 1);
        let col_0 = pos.byte - self.line_info[line_0].start_byte();
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

    pub fn line_start(&self, line_0: usize) -> usize {
        self.line_info[line_0].start_byte()
    }

    pub fn line_end(&self, line_0: usize, include_terminator: bool) -> usize {
        self.line_info.get(line_0 + 1)
            .map_or(self.total_bytes, |info| {
                if include_terminator {
                    // range runs until the next line starts
                    info.start_byte()
                } else if info.follows_2_char_ending() {
                    // exclude the two character line ending
                    info.start_byte() - 2
                } else {
                    // exclude the single character line ending
                    info.start_byte() - 1
                }
            })
    }

    pub fn line_range(&self, line_0: usize, include_terminator: bool) -> std::ops::Range<usize> {
        self.line_start(line_0)..self.line_end(line_0, include_terminator)
    }

    pub fn split_lines(&self, span: SpanFull, include_terminator: bool) -> impl Iterator<Item=SpanFull> + '_ {
        assert_eq!(span.start.file, span.end.file);
        let file = span.start.file;

        (span.start.line_0..=span.end.line_0).filter_map(move |line_0| {
            // start from the entire line
            let range = self.line_range(line_0, include_terminator);
            let mut start = range.start;
            let mut end = range.end;

            // limit the range to the span
            if line_0 == span.start.line_0 {
                start = max(start, span.start.byte);
            }
            if line_0 == span.end.line_0 {
                end = min(end, span.end.byte);
            }

            // skip less-than-empty ranges
            // they can happen when the span starts in the middle of the newline terminator
            if start > end {
                return None;
            }

            // emit the now clean, single-line span
            let span = SpanFull {
                start: PosFull {
                    file,
                    byte: start,
                    line_0,
                    col_0: start - self.line_info[line_0].start_byte(),
                },
                end: PosFull {
                    file,
                    byte: end,
                    line_0,
                    col_0: end - self.line_info[line_0].start_byte(),
                },
            };
            Some(span)
        })
    }
}
