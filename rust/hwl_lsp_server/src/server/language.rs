use crate::server::state::{RequestHandler, ServerState};
use hwl_language::syntax::pos::{FileId, FileOffsets};
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

        let source = match self.open_files.get(&uri) {
            Some(source) => source,
            None => return Err(format!("file not open {uri:?}")),
        };

        let mut semantic_tokens = vec![];

        let offsets = FileOffsets::new(FileId::SINGLE, source);
        let mut prev_start = offsets.full_span().start;

        for token in Tokenizer::new(FileId::SINGLE, source) {
            let token = match token {
                Ok(token) => token,
                // TODO support error recovery in the tokenizer?
                Err(_) => break,
            };

            if let Some(semantic_index) = semantic_token_index(token.ty.category()) {
                let start_full = offsets.expand_pos(token.span.start);
                let prev_start_full = offsets.expand_pos(prev_start);

                // TODO convert to utf-16 offsets
                // TODO check client multi-line token capability
                let delta_line = start_full.line_0 - prev_start_full.line_0;
                let delta_start = if start_full.line_0 == prev_start_full.line_0 {
                    start_full.col_0 - prev_start_full.col_0
                } else {
                    start_full.col_0
                };

                let semantic_token = SemanticToken {
                    delta_line: delta_line as u32,
                    delta_start: delta_start as u32,
                    length: token.span.len_bytes() as u32,
                    token_type: semantic_index as u32,
                    token_modifiers_bitset: 0,
                };
                semantic_tokens.push(semantic_token);

                // only update start if the token was actually included
                prev_start = token.span.start;
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