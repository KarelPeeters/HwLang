use crate::server::settings::PositionEncoding;
use crate::server::state::{RequestHandler, ServerState};
use hwl_language::syntax::pos::FileId;
use hwl_language::syntax::token::{TokenCategory, Tokenizer};
use itertools::Itertools;
use lsp_types::request::SemanticTokensFullRequest;
use lsp_types::{SemanticToken, SemanticTokenType, SemanticTokens, SemanticTokensLegend, SemanticTokensParams, SemanticTokensResult, TextDocumentIdentifier};
use strum::IntoEnumIterator;

impl RequestHandler<SemanticTokensFullRequest> for ServerState {
    fn handle_request(&mut self, params: SemanticTokensParams) -> Result<Option<SemanticTokensResult>, String> {
        let SemanticTokensParams {
            work_done_progress_params: _,
            partial_result_params: _,
            text_document,
        } = params;

        let TextDocumentIdentifier { uri } = text_document;

        let info = match self.virtual_file_system.get_full(&uri) {
            Some(source) => source,
            None => return Err(format!("file not open {uri:?}")),
        };

        let mut semantic_tokens = vec![];
        let mut prev_start_simple = info.offsets.full_span(FileId::SINGLE).start;

        for token in Tokenizer::new(FileId::SINGLE, &info.text) {
            let token = match token {
                Ok(token) => token,
                // TODO support error recovery in the tokenizer?
                // TODO make tokenization error visible to user?
                Err(e) => {
                    eprintln!("tokenization failed: {e:?}");
                    break;
                },
            };

            if let Some(semantic_index) = semantic_token_index(token.ty.category()) {
                let start = info.offsets.expand_pos(token.span.start);
                let prev_start = info.offsets.expand_pos(prev_start_simple);

                // TODO check client multi-line token capability
                // TODO extract position encoding to a common location
                let delta_line = start.line_0 - prev_start.line_0;
                let delta_col = if start.line_0 == prev_start.line_0 {
                    prev_start.col_0..start.col_0
                } else {
                    0..start.col_0
                };
                let start_line_byte = info.offsets.line_start_byte(start.line_0);

                let (encoded_delta_start, encoded_length) = match self.settings.position_encoding {
                    PositionEncoding::Utf8 => (
                        delta_col.end - delta_col.start,
                        token.span.len_bytes()
                    ),
                    PositionEncoding::Utf16 => (
                        info.text[start_line_byte..][delta_col].encode_utf16().count(),
                        info.text[token.span.start.byte..token.span.end.byte].encode_utf16().count(),
                    ),
                };

                // TODO check int overflow
                let semantic_token = SemanticToken {
                    delta_line: delta_line as u32,
                    delta_start: encoded_delta_start as u32,
                    length: encoded_length as u32,
                    token_type: semantic_index as u32,
                    token_modifiers_bitset: 0,
                };
                semantic_tokens.push(semantic_token);

                // only update start if the token was actually included
                prev_start_simple = token.span.start;
            }
        }

        let result = SemanticTokensResult::Tokens(SemanticTokens { result_id: None, data: semantic_tokens });
        Ok(Some(result))
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

pub fn semantic_token_index(category: TokenCategory) -> Option<usize> {
    semantic_token_map(category).map(|_| category as usize)
}