#![no_main]

use hwl_language::syntax::source::FileId;
use hwl_language::syntax::token::{
    TokenType, parse_token_int_literal_binary, parse_token_int_literal_decimal, parse_token_int_literal_hexadecimal,
    parse_token_string_middle, tokenize,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (&str, bool)| target(data));

fn target(data: (&str, bool)) {
    let (source, emit_incomplete_token) = data;

    // try tokenizing
    let tokens = tokenize(FileId::dummy(), source, emit_incomplete_token);

    // try parsing str/int literals
    if let Ok(tokens) = tokens {
        for token in tokens {
            let token_str = &source[token.span.start_byte..token.span.end_byte];
            match token.ty {
                TokenType::StringMiddle => {
                    parse_token_string_middle(token_str).unwrap();
                }
                TokenType::IntLiteralBinary => {
                    parse_token_int_literal_binary(token_str).unwrap();
                }
                TokenType::IntLiteralDecimal => {
                    parse_token_int_literal_decimal(token_str).unwrap();
                }
                TokenType::IntLiteralHexadecimal => {
                    parse_token_int_literal_hexadecimal(token_str).unwrap();
                }
                _ => {}
            }
        }
    }
}
