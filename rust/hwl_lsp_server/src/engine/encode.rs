use crate::server::settings::PositionEncoding;
use hwl_language::syntax::pos::{LineOffsets, Pos, Span};

pub fn encode_pos_to_lsp(
    encoding: PositionEncoding,
    offsets: &LineOffsets,
    source: &str,
    pos: Pos,
) -> lsp_types::Position {
    let pos = offsets.expand_pos(pos);

    let col_0_encoded = match encoding {
        PositionEncoding::Utf8 => pos.col_0,
        PositionEncoding::Utf16 => {
            let start_line_byte = offsets.line_start(pos.line_0);
            source[start_line_byte..][..pos.col_0].encode_utf16().count()
        }
    };

    // TODO check overflow
    lsp_types::Position {
        line: pos.line_0 as u32,
        character: col_0_encoded as u32,
    }
}

pub fn encode_span_to_lsp(
    encoding: PositionEncoding,
    offsets: &LineOffsets,
    source: &str,
    span: Span,
) -> lsp_types::Range {
    lsp_types::Range {
        start: encode_pos_to_lsp(encoding, offsets, source, span.start),
        end: encode_pos_to_lsp(encoding, offsets, source, span.end),
    }
}