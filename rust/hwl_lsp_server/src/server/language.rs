use crate::server::state::ServerState;
use hwl_language::syntax::token::TokenCategory;
use itertools::Itertools;
use lsp_types::{CompletionParams, CompletionResponse, DocumentHighlight, DocumentHighlightParams, SemanticTokenType, SemanticTokensParams, SemanticTokensResult};
use strum::IntoEnumIterator;

impl ServerState {
    pub fn document_highlight(&mut self, params: &DocumentHighlightParams) -> Option<Vec<DocumentHighlight>> {
        // self.log_params("document_highlight", &params);
        // 
        // let DocumentHighlightParams {
        //     text_document_position_params,
        //     work_done_progress_params: _,
        //     partial_result_params: _,
        // } = params;
        // let TextDocumentPositionParams {
        //     text_document,
        //     position: _,
        // } = text_document_position_params;
        // let TextDocumentIdentifier { uri } = text_document;
        // 
        // let state = self.state.lock();
        // let source = match state.documents.get(&uri) {
        //     Some(source) => source,
        //     None => {
        //         self.log_error("failed to find file in DB");
        //         return Ok(None);
        //     }
        // };
        // 
        // let offsets = FileOffsets::new(FileId::SINGLE, source);
        // let ast = match parse_file_content(source, &offsets) {
        //     Ok(ast) => ast,
        //     Err(e) => {
        //         self.log_error(format!("failed to parse file: {e:?}"));
        //         return Ok(None);
        //     },
        // };
        // 
        // let mut result = vec![];
        // for item in ast.items {
        //     let range = offsets.expand_span(item.common_info().span_full).to_lsp();
        //     self.log_info(format!("sending range {range:?}"));
        //     result.push(DocumentHighlight {
        //         range,
        //         kind: Some(DocumentHighlightKind::TEXT),
        //     });
        // }
        // 
        // Ok(Some(result))

        todo!()
    }

    pub fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Option<SemanticTokensResult> {
        // self.log_params("semantic_tokens_full", &params);
        // 
        // let SemanticTokensParams {
        //     work_done_progress_params: _,
        //     partial_result_params: _,
        //     text_document,
        // } = params;
        // 
        // let TextDocumentIdentifier { uri } = text_document;
        // 
        // // TODO proper error reporting
        // let state = self.state.lock();
        // let source = match state.documents.get(&uri) {
        //     Some(source) => source,
        //     None => {
        //         self.log_error("failed to find file in DB");
        //         return Ok(None);
        //     }
        // };
        // 
        // let mut semantic_tokens = vec![];
        // 
        // let offsets = FileOffsets::new(FileId::SINGLE, source);
        // let mut prev_start = offsets.full_span().start;
        // 
        // // TODO check client multi-line token capability
        // // TODO check that both client and server support utf8-encoding
        // for token in Tokenizer::new(FileId::SINGLE, source) {
        //     let token = match token {
        //         Ok(token) => token,
        //         Err(e) => {
        //             // TODO error recovery
        //             self.log_error(format!("failed to tokenize file: {e:?}"));
        //             break;
        //         }
        //     };
        // 
        //     if let Some(semantic_index) = semantic_token_index(token.ty.category()) {
        //         let start_full = offsets.expand_pos(token.span.start);
        //         let prev_start_full = offsets.expand_pos(prev_start);
        // 
        //         let delta_line = start_full.line_0 - prev_start_full.line_0;
        //         let delta_start = if start_full.line_0 == prev_start_full.line_0 {
        //             start_full.col_0 - prev_start_full.col_0
        //         } else {
        //             start_full.col_0
        //         };
        // 
        //         let semantic_token = SemanticToken {
        //             delta_line: delta_line as u32,
        //             delta_start: delta_start as u32,
        //             length: token.span.len_bytes() as u32,
        //             token_type: semantic_index as u32,
        //             token_modifiers_bitset: 0,
        //         };
        //         semantic_tokens.push(semantic_token);
        // 
        //         // only update start if the token was actually included
        //         prev_start = token.span.start;
        //     }
        // }
        // 
        // let result = SemanticTokensResult::Tokens(SemanticTokens { result_id: None, data: semantic_tokens });
        // Ok(Some(result))

        todo!()
    }

    pub fn completion(&self, params: CompletionParams) -> Option<CompletionResponse> {
        // let _ = params;
        // self.client
        //     .log_message(MessageType::INFO, format!("completion({:?})", params))
        //     ;
        // 
        // // TODO actually populate, and check if there are more useful fields to populate
        // let items = vec![
        //     CompletionItem {
        //         label: "custom_completion_text".to_string(),
        //         label_details: None,
        //         kind: Some(CompletionItemKind::TEXT),
        //         ..CompletionItem::default()
        //     },
        //     CompletionItem {
        //         label: "custom_completion_text_method".to_string(),
        //         label_details: None,
        //         kind: Some(CompletionItemKind::METHOD),
        //         ..CompletionItem::default()
        //     },
        // ];
        // 
        // Ok(Some(CompletionResponse::Array(items)))

        todo!()
    }
}

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

pub fn semantic_token_legend() -> Vec<SemanticTokenType> {
    let padding = || SemanticTokenType::STRING;
    TokenCategory::iter()
        .map(move |c| semantic_token_map(c).unwrap_or(padding()))
        .collect_vec()
}

pub fn semantic_token_index(category: TokenCategory) -> Option<usize> {
    semantic_token_map(category).map(|_| category as usize)
}
