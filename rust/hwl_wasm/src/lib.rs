use hwl_language::back::lower_cpp::lower_to_cpp;
use hwl_language::back::lower_verilog::lower_to_verilog;
use hwl_language::front::compile::{ElaborationSet, compile};
use hwl_language::front::diagnostic::{DiagResult, Diagnostics, diags_to_string};
use hwl_language::front::print::CollectPrintHandler;
use hwl_language::syntax::format::{FormatError, FormatSettings, format_file};
use hwl_language::syntax::parsed::ParsedDatabase;
use hwl_language::syntax::source::{FileId, SourceDatabase};
use hwl_language::syntax::token::{TokenCategory, Tokenizer};
use hwl_language::util::{NON_ZERO_USIZE_ONE, ResultExt};
use itertools::Itertools;
use std::time::Duration;
use strum::IntoEnumIterator;
use wasm_bindgen::prelude::wasm_bindgen;

// Suppress the "unused crate" warning for `getrandom`.
// It's in the dependency list for its side effects, not for direct use.
#[allow(unused_imports)]
use getrandom as _;
use hwl_language::syntax::hierarchy::SourceHierarchy;

/// This function automatically runs when the module gets initialized.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(getter_with_clone)]
pub struct RunAllResult {
    pub compile_diags_ansi: String,
    pub compile_prints: Vec<String>,

    pub lowered_verilog: String,
    pub lowered_cpp: String,

    pub format_diags_ansi: String,
    pub format_debug_str: String,
}

const TIMEOUT: Duration = Duration::from_millis(500);

#[wasm_bindgen]
pub fn run_all(top_src: String, include_format: bool) -> RunAllResult {
    let diags = Diagnostics::new();
    let mut source = SourceDatabase::new();
    let hierarchy_file = build_source(&diags, &mut source, top_src);

    // compile
    let mut print_handler = CollectPrintHandler::new();
    let compiled = hierarchy_file.as_ref_ok().and_then(|&(ref hierarchy, top_file)| {
        let parsed = ParsedDatabase::new(&diags, &source, hierarchy);

        let start = wasm_timer::Instant::now();
        let should_stop = || start.elapsed() >= TIMEOUT;
        let dummy_span = source.full_span(top_file);

        compile(
            &diags,
            &source,
            hierarchy,
            &parsed,
            ElaborationSet::AsMuchAsPossible,
            &mut print_handler,
            &should_stop,
            NON_ZERO_USIZE_ONE,
            dummy_span,
        )
    });

    // lower
    let lowered = compiled
        .as_ref_ok()
        .and_then(|c| lower_to_verilog(&diags, &c.modules, &c.external_modules, c.top_module));
    let sim = compiled
        .as_ref_ok()
        .and_then(|c| lower_to_cpp(&diags, &c.modules, c.top_module));

    // format
    let diags_format = Diagnostics::new();
    let formatted = if include_format {
        Some(hierarchy_file.as_ref_ok().and_then(|&(_, top_file)| {
            format_file(&diags_format, &source, &FormatSettings::default(), top_file)
                .map(|f| f.debug_str())
                .map_err(FormatError::to_diag_error)
        }))
    } else {
        None
    };

    // package results
    // TODO lower diagnostics directly to html instead of through ansi first?
    let compile_diags_ansi = diags_to_string(&source, diags.finish(), true);
    let lowered_verilog = lowered.map_or_else(|_| "/* error */".to_string(), |lowered| lowered.source);
    let lowered_cpp = sim.unwrap_or_else(|_| "/* error */".to_string());

    let format_diags_ansi = diags_to_string(&source, diags_format.finish(), true);
    let format_debug_str = formatted.map_or_else(String::new, |f| f.unwrap_or_else(|_| "/* error */".to_string()));

    RunAllResult {
        compile_diags_ansi,
        compile_prints: print_handler.finish(),
        lowered_verilog,
        lowered_cpp,
        format_diags_ansi,
        format_debug_str,
    }
}

#[wasm_bindgen]
pub fn format_source(source: String) -> Option<String> {
    let diags = Diagnostics::new();
    let mut src_db = SourceDatabase::new();
    let file = src_db.add_file("dummy.kh".to_owned(), source);
    match format_file(&diags, &src_db, &FormatSettings::default(), file) {
        Ok(result) => Some(result.new_content),
        Err(_) => None,
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

fn build_source(
    diags: &Diagnostics,
    source: &mut SourceDatabase,
    top_src: String,
) -> DiagResult<(SourceHierarchy, FileId)> {
    let mut hierarchy = SourceHierarchy::new();

    for &(steps, path, content) in included_sources::STD_SOURCES {
        let file = source.add_file(path.to_owned(), content.to_owned());
        let dummy_span = source.full_span(file);
        let steps = steps.iter().map(|&s| s.to_owned()).collect_vec();
        hierarchy.add_file(diags, source, dummy_span, &steps, file)?;
    }

    let file = source.add_file("top.kh".to_owned(), top_src);
    let dummy_span = source.full_span(file);
    hierarchy.add_file(diags, source, dummy_span, &["top".to_owned()], file)?;

    Ok((hierarchy, file))
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
                        token.span.start_byte as u32,
                        token.span.end_byte as u32,
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
