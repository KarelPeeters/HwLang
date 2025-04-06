use crate::engine::encode::encode_span_to_lsp;
use crate::server::dispatch::RequestHandler;
use crate::server::settings::PositionEncoding;
use crate::server::state::{RequestResult, ServerState};
use hwl_language::syntax::pos::{LineOffsets, Span};
use hwl_language::syntax::source::FileId;
use hwl_language::syntax::token::{TokenCategory, Tokenizer};
use itertools::Itertools;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::{
    SemanticToken, SemanticTokenType, SemanticTokens, SemanticTokensLegend, SemanticTokensParams, SemanticTokensResult,
    TextDocumentIdentifier,
};
use strum::IntoEnumIterator;

impl RequestHandler<SemanticTokensFullRequest> for ServerState {
    fn handle_request(&mut self, params: SemanticTokensParams) -> RequestResult<Option<SemanticTokensResult>> {
        let SemanticTokensParams {
            work_done_progress_params: _,
            partial_result_params: _,
            text_document,
        } = params;
        let TextDocumentIdentifier { uri } = text_document;

        let source = self.vfs.inner()?.get_text(&uri)?;
        // TODO cache offsets somewhere
        let offsets = LineOffsets::new(source);

        let mut semantic_tokens = vec![];
        let mut prev_start_simple = lsp_types::Position { line: 0, character: 0 };

        for token in Tokenizer::new(FileId::dummy(), source) {
            let token = match token {
                Ok(token) => token,
                // TODO support error recovery in the tokenizer?
                // TODO make tokenization error visible to user?
                Err(e) => {
                    eprintln!("tokenization failed: {e:?}");
                    break;
                }
            };

            if let Some(semantic_index) = semantic_token_index(token.ty.category()) {
                if self.settings.supports_multi_line_semantic_tokens {
                    semantic_tokens.push(to_semantic_token(
                        self.settings.position_encoding,
                        source,
                        &offsets,
                        &mut prev_start_simple,
                        token.span,
                        semantic_index,
                    ));
                } else {
                    for span in offsets.split_lines(offsets.expand_span(token.span), false) {
                        semantic_tokens.push(to_semantic_token(
                            self.settings.position_encoding,
                            source,
                            &offsets,
                            &mut prev_start_simple,
                            span.span(),
                            semantic_index,
                        ));
                    }
                }
            }
        }

        let result = SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: semantic_tokens,
        });
        Ok(Some(result))
    }
}

fn to_semantic_token(
    encoding: PositionEncoding,
    source: &str,
    offsets: &LineOffsets,
    prev_start: &mut lsp_types::Position,
    span: Span,
    semantic_index: u32,
) -> SemanticToken {
    let range = encode_span_to_lsp(encoding, offsets, source, span);

    // TODO extract position encoding to a common location
    let delta_line = range.start.line - prev_start.line;
    let delta_start = if range.start.line == prev_start.line {
        range.start.character - prev_start.character
    } else {
        range.start.character
    };

    let encoded_length = match encoding {
        PositionEncoding::Utf8 => span.len_bytes(),
        PositionEncoding::Utf16 => source[span.range_bytes()].encode_utf16().count(),
    };

    *prev_start = range.start;

    // TODO check int overflow
    SemanticToken {
        delta_line,
        delta_start,
        length: encoded_length as u32,
        token_type: semantic_index,
        token_modifiers_bitset: 0,
    }
}

// TODO get supported tokens from client capabilities
pub fn semantic_token_map(category: TokenCategory) -> Option<SemanticTokenType> {
    match category {
        TokenCategory::WhiteSpace => None,
        TokenCategory::Comment => Some(SemanticTokenType::COMMENT),
        // TODO better categorization of ids?
        TokenCategory::Identifier => Some(SemanticTokenType::VARIABLE),
        TokenCategory::IntegerLiteral => Some(SemanticTokenType::NUMBER),
        TokenCategory::StringLiteral => Some(SemanticTokenType::STRING),
        TokenCategory::Keyword => Some(SemanticTokenType::KEYWORD),
        // TODO is operator right or should this be None?
        TokenCategory::Symbol => Some(SemanticTokenType::OPERATOR),
    }
}

pub fn semantic_token_legend() -> SemanticTokensLegend {
    let padding = || SemanticTokenType::STRING;
    let token_types = TokenCategory::iter()
        .map(move |c| semantic_token_map(c).unwrap_or(padding()))
        .collect_vec();
    SemanticTokensLegend {
        token_types,
        token_modifiers: vec![],
    }
}

pub fn semantic_token_index(category: TokenCategory) -> Option<u32> {
    // TODO this is super cursed, (why) is this correct?
    semantic_token_map(category).map(|_| category as u32)
}
