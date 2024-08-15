use language::syntax::pos::{Pos, Span};
use tower_lsp::lsp_types::{Position, Range};

pub trait ToLsp {
    type T;
    fn to_lsp(self) -> Self::T;
}

impl ToLsp for Pos {
    type T = Position;
    fn to_lsp(self) -> Position {
        let Pos { file: _, line, col } = self;
        Position {
            line: (line - 1) as u32,
            character: (col - 1) as u32,
        }
    }
}

impl ToLsp for Span {
    type T = Range;
    fn to_lsp(self) -> Range {
        let Span { start, end } = self;
        Range {
            start: start.to_lsp(),
            end: end.to_lsp(),
        }
    }
}
