use hwl_language::syntax::pos::{PosFull, SpanFull};
use tower_lsp::lsp_types::{Position, Range};

pub trait ToLsp {
    type T;
    fn to_lsp(self) -> Self::T;
}

impl ToLsp for PosFull {
    type T = Position;
    fn to_lsp(self) -> Position {
        let PosFull { file: _, byte: _, line_0, col_0 } = self;
        Position {
            line: line_0 as u32,
            character: col_0 as u32,
        }
    }
}

impl ToLsp for SpanFull {
    type T = Range;
    fn to_lsp(self) -> Range {
        let SpanFull { start, end } = self;
        Range {
            start: start.to_lsp(),
            end: end.to_lsp(),
        }
    }
}
