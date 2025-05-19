use crate::server::settings::PositionEncoding;
use hwl_language::syntax::pos::{LineOffsets, Pos, Span};
use hwl_language::syntax::source::FileId;
use hwl_language::util::iter::IterExt;

pub fn pos_to_lsp(encoding: PositionEncoding, offsets: &LineOffsets, src: &str, pos: Pos) -> lsp_types::Position {
    let pos = offsets.expand_pos(pos);

    let col_0_encoded = match encoding {
        PositionEncoding::Utf8 => pos.col_0,
        PositionEncoding::Utf16 => {
            let start_line_byte = offsets.line_start(pos.line_0);
            src[start_line_byte..][..pos.col_0].encode_utf16().count()
        }
    };

    // TODO check overflow
    lsp_types::Position {
        line: pos.line_0 as u32,
        character: col_0_encoded as u32,
    }
}

pub fn lsp_to_pos(
    encoding: PositionEncoding,
    offsets: &LineOffsets,
    src: &str,
    file: FileId,
    pos: lsp_types::Position,
) -> Pos {
    let lsp_types::Position { line, character } = pos;

    let line_0 = line as usize;
    let col_0 = match encoding {
        PositionEncoding::Utf8 => character as usize,
        PositionEncoding::Utf16 => {
            let line_str = &src[offsets.line_range(line_0, true)];
            line_str
                .encode_utf16()
                .take(character as usize)
                .map(|c| {
                    let char = char::decode_utf16(std::iter::once(c)).single().unwrap().unwrap();
                    char.len_utf8()
                })
                .sum()
        }
    };

    let byte = offsets.line_start(line_0) + col_0;
    Pos { file, byte }
}

pub fn span_to_lsp(encoding: PositionEncoding, offsets: &LineOffsets, src: &str, span: Span) -> lsp_types::Range {
    lsp_types::Range {
        start: pos_to_lsp(encoding, offsets, src, span.start()),
        end: pos_to_lsp(encoding, offsets, src, span.end()),
    }
}
