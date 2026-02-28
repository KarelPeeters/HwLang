use crate::syntax::token::REGEX_ID;
use crate::syntax::token::TokenType;
use hwl_util::constants::HWL_LANGUAGE_NAME;
use indexmap::IndexMap;
use serde_json::{Value, json};

// TODO if there a good way to test that this matches the tokenizer?
pub fn generate_textmate_language_json() -> String {
    let mut builder = Builder {
        patterns: vec![],
        repository: IndexMap::new(),
    };

    // add patterns
    add_comments(&mut builder);
    add_literals(&mut builder);
    add_keywords_and_symbols(&mut builder);
    add_identifier(&mut builder);

    // wrap in top-level
    let lang = json!({
        "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
        "name": HWL_LANGUAGE_NAME,
        "scopeName": name("source"),
        "patterns": builder.patterns,
        "repository": builder.repository,
    });

    // return as json string
    serde_json::to_string_pretty(&lang).unwrap()
}

struct Builder {
    patterns: Vec<Value>,
    repository: IndexMap<String, Value>,
}

/// Add the per-language suffix to a pattern name.
fn name(base: &str) -> String {
    format!("{base}.{}", HWL_LANGUAGE_NAME.to_ascii_lowercase())
}

fn add_identifier(builder: &mut Builder) {
    builder.patterns.push(json!({
        "name": name("identifier"),
        "match": format!("\\b{REGEX_ID}\\b"),
    }));
}

fn add_comments(builder: &mut Builder) {
    // single-line comments are simple matches
    builder.patterns.push(json!({
        "name": name("comment.line.double-slash"),
        "match": "//.*$",
    }));

    // block comments can nest, so need a layer of indirection though the repository
    let repo_key_block_comment = "block_comment";
    builder
        .patterns
        .push(json!({"include": format!("#{repo_key_block_comment}")}));
    builder.repository.insert(
        repo_key_block_comment.to_owned(),
        json!({
            "patterns": [
                {
                    "name": name("comment.block"),
                    "begin": "/\\*",
                    "end": "\\*/",
                    "patterns": [{"include": format!("#{repo_key_block_comment}")}],
                }
            ]
        }),
    );
}

fn add_literals(builder: &mut Builder) {
    // raw string literals, no substitution
    // (use begin/end instead of match to allow multi-line string literals)
    builder.patterns.push(json!({
        "name": name("string.quoted.double"),
        "begin": "\"",
        "end": "\"",
    }));

    // standard string literals, with substitution
    builder.patterns.push(json!({
        "name": name("string.quoted.double"),
        "begin": "\"",
        "end": "\"",
        "patterns": [{
            "name": name("meta.string_substitution"),
            "begin": "\\{",
            "end": "\\}",
            "patterns": [{ "include": "$self" }]
        }]
    }));

    // int literals
    builder.patterns.push(json!({
        "name": name("constant.numeric"),
        "match": "[0-9][0-9_bxa-f]*",
    }));
}

// TODO combine this with the semantic token stuff in the tokenizer itself?
fn add_keywords_and_symbols(builder: &mut Builder) {
    for info in TokenType::FIXED_TOKENS {
        if let Some(category) = token_category(info.ty) {
            let name = name(&format!("{}.{}", category.name_base(), info.name.to_lowercase()));
            let pattern = format!("\\b{}\\b", escape_textmate_regex(info.literal));
            builder.patterns.push(json!({
                "name": name,
                "match": pattern,
            }))
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum Category {
    KeywordControl,
    KeywordOther,
    KeywordOperator,
    StorageType,
    StorageModifier,
    ConstantLanguage,
    Punctuation,
}

impl Category {
    fn name_base(self) -> &'static str {
        match self {
            Category::KeywordControl => "keyword.control",
            Category::KeywordOther => "keyword.other",
            Category::KeywordOperator => "keyword.operator",
            Category::StorageType => "storage.type",
            Category::StorageModifier => "storage.modifier",
            Category::ConstantLanguage => "constant.language",
            Category::Punctuation => "punctuation",
        }
    }
}

fn token_category(ty: TokenType) -> Option<Category> {
    match ty {
        // custom tokens are handled elsewhere
        TokenType::BlockComment
        | TokenType::LineComment
        | TokenType::Identifier
        | TokenType::IntLiteralBinary
        | TokenType::IntLiteralDecimal
        | TokenType::IntLiteralHexadecimal
        | TokenType::StringStart
        | TokenType::StringEnd
        | TokenType::StringSubStart
        | TokenType::StringSubEnd
        | TokenType::StringMiddle => None,
        // control
        TokenType::Import
        | TokenType::Return
        | TokenType::Break
        | TokenType::Continue
        | TokenType::If
        | TokenType::Else
        | TokenType::Loop
        | TokenType::Match
        | TokenType::For
        | TokenType::While => Some(Category::KeywordControl),
        // type and variable defs
        TokenType::Type
        | TokenType::Struct
        | TokenType::Enum
        | TokenType::Module
        | TokenType::Const
        | TokenType::Val
        | TokenType::Var
        | TokenType::Wire
        | TokenType::Reg
        | TokenType::Ref
        | TokenType::Deref => Some(Category::StorageType),
        // storage modifiers
        TokenType::External => Some(Category::StorageModifier),
        // literals
        TokenType::True | TokenType::False | TokenType::Undef => Some(Category::ConstantLanguage),
        // other keywords
        TokenType::Interface
        | TokenType::Ports
        | TokenType::Port
        | TokenType::Instance
        | TokenType::Fn
        | TokenType::Comb
        | TokenType::Clock
        | TokenType::Clocked
        | TokenType::In
        | TokenType::Out
        | TokenType::Async
        | TokenType::Sync
        | TokenType::Pub
        | TokenType::As
        | TokenType::Builtin
        | TokenType::UnsafeValueWithDomain
        | TokenType::IdFromStr => Some(Category::KeywordOther),
        // punctuation
        TokenType::Semi
        | TokenType::Colon
        | TokenType::Comma
        | TokenType::Arrow
        | TokenType::DoubleArrow
        | TokenType::Underscore
        | TokenType::ColonColon
        | TokenType::OpenC
        | TokenType::CloseC
        | TokenType::OpenR
        | TokenType::CloseR
        | TokenType::OpenS
        | TokenType::CloseS => Some(Category::Punctuation),
        // operators
        TokenType::Dot
        | TokenType::DotDot
        | TokenType::DotDotEq
        | TokenType::PlusDotDot
        | TokenType::AmperAmper
        | TokenType::PipePipe
        | TokenType::CaretCaret
        | TokenType::EqEq
        | TokenType::Neq
        | TokenType::Gte
        | TokenType::Gt
        | TokenType::Lte
        | TokenType::Lt
        | TokenType::Amper
        | TokenType::Pipe
        | TokenType::Caret
        | TokenType::LtLt
        | TokenType::GtGt
        | TokenType::Plus
        | TokenType::Minus
        | TokenType::Star
        | TokenType::Slash
        | TokenType::Percent
        | TokenType::Bang
        | TokenType::StarStar
        | TokenType::Eq
        | TokenType::PlusEq
        | TokenType::MinusEq
        | TokenType::StarEq
        | TokenType::SlashEq
        | TokenType::PercentEq
        | TokenType::AmperEq
        | TokenType::PipeEq
        | TokenType::CaretEq => Some(Category::KeywordOperator),
    }
}

/// Escape a literal string.
/// The result, when used as a regex in a TextMate language, will only match exactly the given string.
fn escape_textmate_regex(literal: &str) -> String {
    let mut f = String::new();
    for c in literal.chars() {
        if is_textmate_regex_special_char(c) {
            f.push('\\');
        }
        f.push(c);
    }
    f
}

fn is_textmate_regex_special_char(c: char) -> bool {
    ".*+?[](){}|^$\\".contains(c)
}

#[cfg(test)]
mod tests {
    use crate::syntax::external::textmate::generate_textmate_language_json;
    use hwl_util::io::IoErrorExt;

    #[test]
    fn matches_textmate_grammar() {
        let expected = generate_textmate_language_json();

        let path_rel = "../../lsp_client/syntaxes/hwlang.tmLanguage.json";
        let path_abs = std::env::current_dir().unwrap().join(path_rel);

        // TODO find a batter way to update the actual grammar
        // std::fs::write(&path_abs, &expected).unwrap();

        let actual = std::fs::read_to_string(&path_abs)
            .map_err(|e| e.with_path(path_abs))
            .unwrap();

        assert_eq!(expected, actual);
    }
}
