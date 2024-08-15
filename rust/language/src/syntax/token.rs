use std::cmp::min;

use itertools::Itertools;
use lazy_static::lazy_static;
use regex::{Regex, RegexSet, SetMatches};
use strum::EnumIter;

use crate::syntax::pos::{FileId, Pos, Span};

#[derive(Debug, Eq, PartialEq)]
pub struct Token<S> {
    pub ty: TokenType,
    pub string: S,
    pub span: Span,
}

#[derive(Debug, Eq, PartialEq)]
pub struct InvalidToken {
    pub pos: Pos,
    pub prefix: String,
}

const ERROR_CONTEXT_LENGTH: usize = 16;

pub fn tokenize(file: FileId, source: &str) -> Result<Vec<Token<&str>>, InvalidToken> {
    Tokenizer::new(file, source).try_collect()
}

// TODO implement recovery by matching without start anchor?
// TODO error check regex overlap in advance at test-time using
//   https://users.rust-lang.org/t/detect-regex-conflict/57184/13
// TODO use lazy_static to compile the regexes only once?
pub struct Tokenizer<'s> {
    compiled: &'static CompiledRegex,
    file: FileId,
    curr_byte: usize,
    left: &'s str,
    errored: bool,
}

impl<'s> Tokenizer<'s> {
    pub fn new(file: FileId, source: &'s str) -> Self {
        Tokenizer {
            compiled: CompiledRegex::instance(),
            file,
            curr_byte: 0,
            left: source,
            errored: false,
        }
    }
}

impl<'s> Iterator for Tokenizer<'s> {
    type Item = Result<Token<&'s str>, InvalidToken>;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(!self.errored, "Cannot continue calling next on tokenizer that returned an error");
        if self.left.is_empty() {
            return None;
        }
        
        let start = Pos { file: self.file, byte: self.curr_byte };
        let matches = self.compiled.set.matches(self.left);
        
        let m = match pick_match(matches, &self.compiled.vec, self.left) {
            None => {
                self.errored = true;
                let left_context = &self.left[..min(self.left.len(), ERROR_CONTEXT_LENGTH)];
                return Some(Err(InvalidToken {
                    pos: Pos { file: self.file, byte: self.curr_byte },
                    prefix: left_context.to_owned(),
                }));
            }
            Some(match_index) => match_index,
        };

        let match_str = &self.left[..m.len];
        self.left = &self.left[m.len..];
        
        self.curr_byte += match_str.len();
        let end = Pos { file: self.file, byte: self.curr_byte };
        let span = Span::new(start, end);

        Some(Ok(Token {
            ty: TOKEN_PATTERNS[m.index].0,
            string: match_str,
            span,
        }))
    }
}

#[derive(Clone)]
struct CompiledRegex {
    set: RegexSet,
    vec: Vec<Regex>,
}

impl CompiledRegex {
    fn new() -> Self {
        let patterns = TOKEN_PATTERNS
            .iter()
            .map(|(_, pattern, kind, _)| {
                let bare = match kind {
                    PK::Regex => pattern.to_string(),
                    PK::Literal => regex::escape(pattern),
                };
                // surround in non-capturing group to make sure that the start-of-string "^" binds correctly
                format!("^(?:{bare})")
            })
            .collect_vec();

        let set = RegexSet::new(&patterns).unwrap();
        let vec = patterns.iter().map(|p| Regex::new(p).unwrap()).collect_vec();

        CompiledRegex { set, vec }
    }

    fn instance() -> &'static Self {
        // TODO https://docs.rs/regex/latest/regex/#sharing-a-regex-across-threads-can-result-in-contention.
        lazy_static! {
            static ref INSTANCE: CompiledRegex = CompiledRegex::new();
        }
        &*INSTANCE
    }
}

struct PickedMatch {
    index: usize,
    len: usize,
    prio: TokenPriority,
}

fn pick_match(matches: SetMatches, regex_vec: &[Regex], left: &str) -> Option<PickedMatch> {
    let mut unique = false;
    let mut result: Option<PickedMatch> = None;
    let mut match_count = 0;

    for index in matches.iter() {
        match_count += 1;

        let range = regex_vec[index].find(left).unwrap().range();
        assert_eq!(range.start, 0);
        let len = range.end;

        let (_, _, _, prio) = TOKEN_PATTERNS[index];
        match prio {
            TP::Unique => unique = true,
            TP::Normal | TP::Low => {}
        }

        let better = match result {
            Some(PickedMatch {
                index: prev_index,
                len: prev_len,
                prio: prev_prio,
            }) => {
                let key = (len, prio);
                let prev_key = (prev_len, prev_prio);
                assert_ne!(
                    key,
                    prev_key,
                    "tokens {:?} and {:?} both with priority {:?} match same prefix {:?}",
                    TOKEN_PATTERNS[prev_index].0,
                    TOKEN_PATTERNS[index].0,
                    prio,
                    &left[..len],
                );
                key > prev_key
            }
            None => true,
        };
        if better {
            result = Some(PickedMatch { index, len, prio });
        }
    }

    if unique {
        assert!(match_count == 1);
    }
    result
}

macro_rules! declare_tokens {
    ($($token:ident($string:literal, $kind:expr, $cat:expr, $prio:expr),)*) => {
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TokenType {
            $($token,)*
        }

        const TOKEN_PATTERNS: &[(TokenType, &'static str, PatternKind, TokenPriority)] = &[
            $((TokenType::$token, $string, $kind, $prio),)*
        ];

        // TODO function vs array?
        impl TokenType {
            pub fn category(self) -> TokenCategory {
                match self {
                    $(TokenType::$token => $cat,)*
                }
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum PatternKind {
    Regex,
    Literal,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, EnumIter)]
pub enum TokenCategory {
    WhiteSpace,
    Comment,
    Identifier,
    IntegerLiteral,
    StringLiteral,
    Keyword,
    Symbol,
}

/// Priority is only used to tiebreak equal-length matches.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenPriority {
    Unique = 2,
    Normal = 1,
    Low = 0,
}

use PatternKind as PK;
use TokenCategory as TC;
use TokenPriority as TP;

// TODO move to separate file
declare_tokens! {
    // ignored
    WhiteSpace(r"\s+", PK::Regex, TC::WhiteSpace, TP::Unique),
    LineComment(r"//[^\n\r]*", PK::Regex, TC::Comment, TP::Normal),
    BlockComment(r"/\*([^\*]*\*+[^\*/])*([^\*]*\*+|[^\*])*\*/", PK::Regex, TC::Comment, TP::Normal),

    // patterns
    Identifier(r"(_[a-zA-Z_0-9]+)|([a-zA-Z][a-zA-Z_0-9]*)", PK::Regex, TC::Identifier, TP::Low),
    IntLiteralDecimal(r"[0-9]+", PK::Regex, TC::IntegerLiteral, TP::Unique),
    IntPatternHexadecimal(r"0x[0-9a-fA-F_?]+", PK::Regex, TC::IntegerLiteral, TP::Unique),
    IntPatternBinary(r"0b[0-9a-fA-F_?]+", PK::Regex, TC::IntegerLiteral, TP::Unique),

    // TODO better string literal pattern with escape codes and string formatting expressions
    StringLiteral(r#""[^"]*""#, PK::Regex, TC::StringLiteral, TP::Unique),

    // keywords
    Use("use", PK::Literal, TC::Keyword, TP::Normal),
    As("as", PK::Literal, TC::Keyword, TP::Normal),
    Type("type", PK::Literal, TC::Keyword, TP::Normal),
    Struct("struct", PK::Literal, TC::Keyword, TP::Normal),
    Enum("enum", PK::Literal, TC::Keyword, TP::Normal),
    Ports("ports", PK::Literal, TC::Keyword, TP::Normal),
    Module("module", PK::Literal, TC::Keyword, TP::Normal),
    Function("function", PK::Literal, TC::Keyword, TP::Normal),
    Combinatorial("combinatorial", PK::Literal, TC::Keyword, TP::Normal),
    Clocked("clocked", PK::Literal, TC::Keyword, TP::Normal),
    Const("const", PK::Literal, TC::Keyword, TP::Normal),
    Val("val", PK::Literal, TC::Keyword, TP::Normal),
    Var("var", PK::Literal, TC::Keyword, TP::Normal),
    Input("input", PK::Literal, TC::Keyword, TP::Normal),
    Output("output", PK::Literal, TC::Keyword, TP::Normal),
    Async("async", PK::Literal, TC::Keyword, TP::Normal),
    Sync("sync", PK::Literal, TC::Keyword, TP::Normal),
    Return("return", PK::Literal, TC::Keyword, TP::Normal),
    Break("break", PK::Literal, TC::Keyword, TP::Normal),
    Continue("continue", PK::Literal, TC::Keyword, TP::Normal),
    True("true", PK::Literal, TC::Keyword, TP::Normal),
    False("false", PK::Literal, TC::Keyword, TP::Normal),
    If("if", PK::Literal, TC::Keyword, TP::Normal),
    Else("else", PK::Literal, TC::Keyword, TP::Normal),
    Loop("loop", PK::Literal, TC::Keyword, TP::Normal),
    For("for", PK::Literal, TC::Keyword, TP::Normal),
    In("in", PK::Literal, TC::Keyword, TP::Normal),
    While("while", PK::Literal, TC::Keyword, TP::Normal),
    Public("pub", PK::Literal, TC::Keyword, TP::Normal),

    // misc symbols
    Semi(";", PK::Literal, TC::Symbol, TP::Normal),
    Colon(":", PK::Literal, TC::Symbol, TP::Normal),
    Comma(",", PK::Literal, TC::Symbol, TP::Normal),
    Arrow("->", PK::Literal, TC::Symbol, TP::Normal),
    Underscore("_", PK::Literal, TC::Symbol, TP::Normal),
    ColonColon("::", PK::Literal, TC::Symbol, TP::Normal),

    // braces
    OpenC("{", PK::Literal, TC::Symbol, TP::Normal),
    CloseC("}", PK::Literal, TC::Symbol, TP::Normal),
    OpenR("(", PK::Literal, TC::Symbol, TP::Normal),
    CloseR(")", PK::Literal, TC::Symbol, TP::Normal),
    OpenS("[", PK::Literal, TC::Symbol, TP::Normal),
    CloseS("]", PK::Literal, TC::Symbol, TP::Normal),

    // operators
    Dot(".", PK::Literal, TC::Symbol, TP::Normal),
    Dots("..", PK::Literal, TC::Symbol, TP::Normal),
    DotsEq("..=", PK::Literal, TC::Symbol, TP::Normal),
    AmperAmper("&&", PK::Literal, TC::Symbol, TP::Normal),
    PipePipe("||", PK::Literal, TC::Symbol, TP::Normal),
    EqEq("==", PK::Literal, TC::Symbol, TP::Normal),
    Neq("!=", PK::Literal, TC::Symbol, TP::Normal),
    Gte(">=", PK::Literal, TC::Symbol, TP::Normal),
    Gt(">", PK::Literal, TC::Symbol, TP::Normal),
    Lte("<=", PK::Literal, TC::Symbol, TP::Normal),
    Lt("<", PK::Literal, TC::Symbol, TP::Normal),
    Amper("&", PK::Literal, TC::Symbol, TP::Normal),
    Circumflex("^", PK::Literal, TC::Symbol, TP::Normal),
    Pipe("|", PK::Literal, TC::Symbol, TP::Normal),
    LtLt("<<", PK::Literal, TC::Symbol, TP::Normal),
    GtGt(">>", PK::Literal, TC::Symbol, TP::Normal),
    Plus("+", PK::Literal, TC::Symbol, TP::Normal),
    Minus("-", PK::Literal, TC::Symbol, TP::Normal),
    Star("*", PK::Literal, TC::Symbol, TP::Normal),
    Slash("/", PK::Literal, TC::Symbol, TP::Normal),
    Percent("%", PK::Literal, TC::Symbol, TP::Normal),
    Bang("!", PK::Literal, TC::Symbol, TP::Normal),
    StarStar("**", PK::Literal, TC::Symbol, TP::Normal),

    // assignment operators
    Eq("=", PK::Literal, TC::Symbol, TP::Normal),
    PlusEq("+=", PK::Literal, TC::Symbol, TP::Normal),
    MinusEq("-=", PK::Literal, TC::Symbol, TP::Normal),
    StarEq("*=", PK::Literal, TC::Symbol, TP::Normal),
    SlashEq("/=", PK::Literal, TC::Symbol, TP::Normal),
    PercentEq("%=", PK::Literal, TC::Symbol, TP::Normal),
    AmperEq("&=", PK::Literal, TC::Symbol, TP::Normal),
    CircumflexEq("^=", PK::Literal, TC::Symbol, TP::Normal),
    BarEq("|=", PK::Literal, TC::Symbol, TP::Normal),
}

#[cfg(test)]
mod test {
    use crate::syntax::pos::{FileId, Pos, Span};
    use crate::syntax::token::{tokenize, Token, TokenType};

    #[test]
    fn empty_tokenize() {
        let file = FileId::SINGLE;
        
        assert_eq!(Ok(vec![]), tokenize(file, ""));
        assert_eq!(Ok(vec![Token {
            ty: TokenType::WhiteSpace,
            string: "\n",
            span: Span { start: Pos { file, byte: 0 }, end: Pos { file: file, byte: 1 } },
        }]), tokenize(file, "\n"));
        assert!(tokenize(file, "test foo function \"foo\"").is_ok());
    }
}
