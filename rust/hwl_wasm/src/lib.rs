use hwl_language::back::lower;
use hwl_language::data::diagnostic::{DiagnosticStringSettings, Diagnostics};
use hwl_language::data::lowered::LoweredDatabase;
use hwl_language::data::source::FilePath;
use hwl_language::data::source::SourceDatabase;
use hwl_language::front::driver::compile;
use hwl_language::syntax::pos::FileId;
use hwl_language::syntax::token::{TokenCategory, Tokenizer};
use itertools::Itertools;
use strum::IntoEnumIterator;
use wasm_bindgen::prelude::wasm_bindgen;

/// This function automatically runs when the module gets initialized.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(getter_with_clone)]
pub struct CompileAndLowerResult {
    pub diagnostics_ansi: String,
    pub lowered_verilog: String,
}

#[wasm_bindgen]
pub fn initial_source() -> String {
    include_str!("initial_source.kh").to_owned()
}

#[wasm_bindgen]
pub fn compile_and_lower(src: String) -> CompileAndLowerResult {
    let mut source = SourceDatabase::new();
    const STD_TYPES_SRC: &str = include_str!("../../../design/project/std/types.kh");
    source.add_file(
        FilePath(vec!["std".to_owned(), "types".to_owned()]),
        "srd/types.kh".to_owned(),
        STD_TYPES_SRC.to_owned(),
    ).unwrap();
    source.add_file(
        FilePath(vec!["top".to_owned()]),
        "top.kh".to_owned(),
        src,
    ).unwrap();

    let diag = Diagnostics::new();
    let (parsed, compiled) = compile(&diag, &source);
    let LoweredDatabase { top_module_name, verilog_source } = lower(&diag, &source, &parsed, &compiled);

    // TODO lower directly to html?
    let diag_settings = DiagnosticStringSettings::default();
    let diagnostics_ansi = diag.finish().into_iter()
        .map(|d| d.to_string(&source, diag_settings))
        .join("\n");

    let _ = top_module_name;
    CompileAndLowerResult {
        diagnostics_ansi,
        lowered_verilog: verilog_source,
    }
}

/// See <https://lezer.codemirror.net/docs/ref/#common.Tree^build>
/// for the expected format of the returned array.
///
/// Node type indices refer to [codemirror_node_types].
/// The result also includes the final top node, with should have an index of `codemirror_node_types().length`.
///
/// The offsets are
/// > the number of characters (UTF16 code units) from the start of the document,
/// > counting line breaks as one character
#[wasm_bindgen]
pub fn codemirror_tokenize_to_tree(src: &str) -> Vec<u32> {
    let mut result = vec![];

    // TODO correct offsets: count newlines as one, and utf16 all the way

    let token_category_to_index = token_category_to_index();
    let top_node_index = codemirror_node_types().len();

    for token in Tokenizer::new(FileId::SINGLE, src) {
        match token {
            Ok(token) => {
                if let Some(category_index) = token_category_to_index[token.ty.category().index()] {
                    result.extend_from_slice(&[
                        category_index,
                        token.span.start.byte as u32,
                        token.span.end.byte as u32,
                        4,
                    ]);
                }
            }
            Err(_) => {
                // just stop, the error will be reported by the following real compiler flow
                break;
            }
        }
    }

    // push final top token
    result.extend_from_slice(&[
        top_node_index as u32,
        0,
        src.len() as u32,
        (4 + result.len()) as u32,
    ]);

    result
}

#[wasm_bindgen]
pub fn codemirror_node_types() -> Vec<String> {
    TokenCategory::iter()
        .filter_map(|tc| token_category_to_tag(tc))
        .map(str::to_owned)
        .collect_vec()
}

/// Mapping to <https://lezer.codemirror.net/docs/ref/#highlight.tags>.
/// This is implemented on the Rust side to check at compile time whether all categories are covered.
fn token_category_to_tag(tc: TokenCategory) -> Option<&'static str> {
    match tc {
        TokenCategory::WhiteSpace => None,
        TokenCategory::Comment => Some("comment"),
        TokenCategory::Identifier => Some("name"),
        TokenCategory::IntegerLiteral => Some("number"),
        TokenCategory::StringLiteral => Some("string"),
        TokenCategory::Keyword => Some("keyword"),
        TokenCategory::Symbol => Some("punctuation")
    }
}

fn token_category_to_index() -> Vec<Option<u32>> {
    let mut next_index = 0;
    let mut result = vec![];

    for tc in TokenCategory::iter() {
        let index = match token_category_to_tag(tc) {
            None => None,
            Some(_) => {
                let index = next_index;
                next_index += 1;
                Some(index)
            }
        };
        result.push(index);
    }

    result
}