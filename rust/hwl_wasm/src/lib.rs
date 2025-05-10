use hwl_language::back::lower_cpp::lower_to_cpp;
use hwl_language::back::lower_verilog::lower_to_verilog;
use hwl_language::front::compile::{compile, CollectPrintHandler, ElaborationSet};
use hwl_language::front::diagnostic::{DiagnosticStringSettings, Diagnostics};
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FileId, FilePath, SourceDatabase, SourceDatabaseBuilder};
use hwl_language::syntax::token::{TokenCategory, Tokenizer};
use hwl_language::util::{ResultExt, NON_ZERO_USIZE_ONE};
use itertools::Itertools;
use std::time::Duration;
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
    pub prints: Vec<String>,
    pub lowered_verilog: String,
    pub lowered_cpp: String,
}

const TIMEOUT: Duration = Duration::from_millis(500);

#[wasm_bindgen]
pub fn compile_and_lower(top_src: String) -> CompileAndLowerResult {
    let source = build_source(top_src);

    let diags = Diagnostics::new();
    let parsed = ParsedDatabase::new(&diags, &source);

    let mut print_handler = CollectPrintHandler::new();
    let start = wasm_timer::Instant::now();
    let should_stop = || start.elapsed() >= TIMEOUT;
    let compiled = compile(
        &diags,
        &source,
        &parsed,
        ElaborationSet::AsMuchAsPossible,
        &mut print_handler,
        &should_stop,
        NON_ZERO_USIZE_ONE,
    );

    let lowered = compiled
        .as_ref_ok()
        .and_then(|c| lower_to_verilog(&diags, &source, &parsed, &c.modules, c.top_module));
    let sim = compiled
        .as_ref_ok()
        .and_then(|c| lower_to_cpp(&diags, &c.modules, c.top_module));

    // TODO lower directly to html?
    let diag_settings = DiagnosticStringSettings::default();
    let diagnostics_ansi = diags
        .finish()
        .into_iter()
        .map(|d| d.to_string(&source, diag_settings))
        .join("\n\n");

    let lowered_verilog = lowered.map_or_else(|_| "/* error */".to_string(), |lowered| lowered.verilog_source);
    let lowered_cpp = sim.unwrap_or_else(|_| "/* error */".to_string());

    CompileAndLowerResult {
        diagnostics_ansi,
        prints: print_handler.finish(),
        lowered_verilog,
        lowered_cpp,
    }
}

mod included_sources {
    pub const SRC_INITIAL_TOP: &str = include_str!("../../../design/top_webdemo.kh");
    include!(concat!(env!("OUT_DIR"), "/std_sources.rs"));
}

#[wasm_bindgen]
pub fn initial_source() -> String {
    included_sources::SRC_INITIAL_TOP.to_owned()
}

fn build_source(top_src: String) -> SourceDatabase {
    let mut source = SourceDatabaseBuilder::new();
    for &(steps, path, content) in included_sources::STD_SOURCES {
        let steps = FilePath(steps.iter().map(|&s| s.to_owned()).collect_vec());
        source.add_file(steps, path.to_owned(), content.to_owned()).unwrap();
    }
    source
        .add_file(FilePath(vec!["top".to_owned()]), "top.kh".to_owned(), top_src)
        .unwrap();
    source.finish()
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

    for token in Tokenizer::new(FileId::dummy(), src, true) {
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
    result.extend_from_slice(&[top_node_index as u32, 0, src.len() as u32, (4 + result.len()) as u32]);

    result
}

#[wasm_bindgen]
pub fn codemirror_node_types() -> Vec<String> {
    TokenCategory::iter()
        .filter_map(token_category_to_tag)
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
        TokenCategory::Symbol => Some("punctuation"),
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
