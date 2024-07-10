use std::cmp::min;

use itertools::Itertools;
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

// TODO error check regex overlap in advance at test-time using
//   https://users.rust-lang.org/t/detect-regex-conflict/57184/13
// TODO use lazy_static to compile the regexes only once?
pub fn tokenize(file: FileId, source: &str) -> Result<Vec<Token<&str>>, InvalidToken> {
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

    let regex_set = RegexSet::new(&patterns).unwrap();
    let regex_vec = patterns.iter().map(|p| Regex::new(p).unwrap()).collect_vec();

    let mut left = source;
    let mut tokens = vec![];

    let mut pos = Pos { file, line: 0, col: 0 };

    while !left.is_empty() {
        let left_context = &left[..min(left.len(), ERROR_CONTEXT_LENGTH)];
        let matches = regex_set.matches(left);

        let match_index = match pick_match(matches, &regex_vec, left) {
            None => {
                return Err(InvalidToken {
                    pos,
                    prefix: left_context.to_owned(),
                })
            }
            Some(match_index) => match_index,
        };

        let match_range = regex_vec[match_index].find(left).unwrap().range();
        assert!(match_range.start == 0);

        let match_str = &left[..match_range.end];
        left = &left[match_range.end..];

        let start = pos;
        pos = pos.step_over(match_str);
        let span = Span::new(start, pos);

        tokens.push(Token {
            ty: TOKEN_PATTERNS[match_index].0,
            string: match_str,
            span,
        })
    }

    Ok(tokens)
}

// Picking algorithm:
// * Fail if an any of:
//   * multiple matches with at least one TokenPriority::Unique
//   * regex match and at least one other regex/literal match with same priority
//   * multiple literal matches with same priority and same length
// Tiebreak:
//   * if different priorities, the highest priority wins.
//   * if same priority, the longest one wins.
fn pick_match(matches: SetMatches, regex_vec: &[Regex], left: &str) -> Option<usize> {
    // TODO does this implementation really need to be this ugly?
    let mut match_count: u32 = 0;

    let mut unique = None;

    let mut longest_high: Option<(usize, usize)> = None;
    let mut longest_medium = None;
    let mut longest_low = None;

    let mut regex_high = None;
    let mut regex_medium = None;
    let mut regex_low = None;

    fn try_longest_literal(curr_index: usize, curr_len: usize, slot: &mut Option<(usize, usize)>) {
        let better = match *slot {
            None => true,
            Some((_, other_len)) => {
                assert_ne!(
                    curr_len, other_len,
                    "multiple matching literals with same length and priority should be impossible"
                );
                curr_len < other_len
            }
        };
        if better {
            *slot = Some((curr_index, curr_len));
        }
    }

    fn try_regex(curr_index: usize, slot: &mut Option<usize>) {
        assert!(slot.is_none());
        *slot = Some(curr_index);
    }

    for index in matches.iter() {
        let (_, pattern, kind, prio) = TOKEN_PATTERNS[index];
        match_count += 1;
        match kind {
            PK::Regex => match prio {
                TP::High => try_regex(index, &mut regex_high),
                TP::Medium => try_regex(index, &mut regex_medium),
                TP::Low => try_regex(index, &mut regex_low),
                TP::Unique => unique = Some(index),
            },
            PK::Literal => match prio {
                TP::High => try_longest_literal(index, pattern.len(), &mut longest_high),
                TP::Medium => try_longest_literal(index, pattern.len(), &mut longest_medium),
                TP::Low => try_longest_literal(index, pattern.len(), &mut longest_low),
                TP::Unique => unique = Some(index),
            },
        }
    }

    if let Some(unique) = unique {
        assert!(match_count == 1);
        return Some(unique);
    }

    fn pair(regex: Option<usize>, literal: Option<(usize, usize)>) -> Option<usize> {
        match (regex, literal) {
            (None, None) => None,
            (Some(index), None) | (None, Some((index, _))) => Some(index),
            (Some(_), Some(_)) => panic!("Conflict between regex and literal"),
        }
    }

    // this evaluated all pairs early for extra error checking
    pair(regex_high, longest_high)
        .or(pair(regex_medium, longest_medium))
        .or(pair(regex_low, longest_low))
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

#[derive(Debug, Copy, Clone)]
pub enum TokenPriority {
    High,
    Medium,
    Low,
    Unique,
}

use PatternKind as PK;
use TokenCategory as TC;
use TokenPriority as TP;

declare_tokens! {
    // ignored
    WhiteSpace(r"\s+", PK::Regex, TC::WhiteSpace, TP::Unique),
    LineComment(r"//[^\n\r]*[\n\r]+", PK::Regex, TC::Comment, TP::High),
    BlockComment(r"/\*([^\*]*\*+[^\*/])*([^\*]*\*+|[^\*])*\*/", PK::Regex, TC::Comment, TP::High),

    // patterns
    Identifier(r"(_[a-zA-Z_0-9]+)|([a-zA-Z][a-zA-Z_0-9]*)", PK::Regex, TC::Identifier, TP::Medium),
    IntLiteralDecimal(r"[0-9]+", PK::Regex, TC::IntegerLiteral, TP::Unique),
    IntPatternHexadecimal(r"0x[0-9a-fA-F_?]+", PK::Regex, TC::IntegerLiteral, TP::Unique),
    IntPatternBinary(r"0b[0-9a-fA-F_?]+", PK::Regex, TC::IntegerLiteral, TP::Unique),

    // TODO better string literal pattern with escape codes and string formatting expressions
    StringLiteral(r#""[^"]*""#, PK::Regex, TC::StringLiteral, TP::Unique),

    // keywords
    Use("use", PK::Literal, TC::Keyword, TP::High),
    As("as", PK::Literal, TC::Keyword, TP::High),
    Type("type", PK::Literal, TC::Keyword, TP::High),
    Struct("struct", PK::Literal, TC::Keyword, TP::High),
    Enum("enum", PK::Literal, TC::Keyword, TP::High),
    Ports("ports", PK::Literal, TC::Keyword, TP::High),
    Module("module", PK::Literal, TC::Keyword, TP::High),
    Function("function", PK::Literal, TC::Keyword, TP::High),
    Combinatorial("combinatorial", PK::Literal, TC::Keyword, TP::High),
    Clocked("clocked", PK::Literal, TC::Keyword, TP::High),
    Const("const", PK::Literal, TC::Keyword, TP::High),
    Val("val", PK::Literal, TC::Keyword, TP::High),
    Var("var", PK::Literal, TC::Keyword, TP::High),
    Input("input", PK::Literal, TC::Keyword, TP::High),
    Output("output", PK::Literal, TC::Keyword, TP::High),
    Async("async", PK::Literal, TC::Keyword, TP::High),
    Sync("sync", PK::Literal, TC::Keyword, TP::High),
    Return("return", PK::Literal, TC::Keyword, TP::High),
    Break("break", PK::Literal, TC::Keyword, TP::High),
    Continue("continue", PK::Literal, TC::Keyword, TP::High),
    True("true", PK::Literal, TC::Keyword, TP::High),
    False("false", PK::Literal, TC::Keyword, TP::High),
    If("if", PK::Literal, TC::Keyword, TP::High),
    Else("else", PK::Literal, TC::Keyword, TP::High),
    Loop("loop", PK::Literal, TC::Keyword, TP::High),
    For("for", PK::Literal, TC::Keyword, TP::High),
    In("in", PK::Literal, TC::Keyword, TP::High),
    While("while", PK::Literal, TC::Keyword, TP::High),
    Public("public", PK::Literal, TC::Keyword, TP::High),

    // misc symbols
    Semi(";", PK::Literal, TC::Symbol, TP::Low),
    Colon(":", PK::Literal, TC::Symbol, TP::Low),
    Comma(",", PK::Literal, TC::Symbol, TP::Low),
    Arrow("->", PK::Literal, TC::Symbol, TP::Low),
    Underscore("_", PK::Literal, TC::Symbol, TP::Low),
    ColonColon("::", PK::Literal, TC::Symbol, TP::Low),

    // braces
    OpenC("{", PK::Literal, TC::Symbol, TP::Low),
    CloseC("}", PK::Literal, TC::Symbol, TP::Low),
    OpenR("(", PK::Literal, TC::Symbol, TP::Low),
    CloseR(")", PK::Literal, TC::Symbol, TP::Low),
    OpenS("[", PK::Literal, TC::Symbol, TP::Low),
    CloseS("]", PK::Literal, TC::Symbol, TP::Low),

    // operators
    Dot(".", PK::Literal, TC::Symbol, TP::Low),
    Dots("..", PK::Literal, TC::Symbol, TP::Low),
    DotsEq("..=", PK::Literal, TC::Symbol, TP::Low),
    AmperAmper("&&", PK::Literal, TC::Symbol, TP::Low),
    PipePipe("||", PK::Literal, TC::Symbol, TP::Low),
    EqEq("==", PK::Literal, TC::Symbol, TP::Low),
    Neq("!=", PK::Literal, TC::Symbol, TP::Low),
    Gte(">=", PK::Literal, TC::Symbol, TP::Low),
    Gt(">", PK::Literal, TC::Symbol, TP::Low),
    Lte("<=", PK::Literal, TC::Symbol, TP::Low),
    Lt("<", PK::Literal, TC::Symbol, TP::Low),
    Amper("&", PK::Literal, TC::Symbol, TP::Low),
    Circumflex("^", PK::Literal, TC::Symbol, TP::Low),
    Pipe("|", PK::Literal, TC::Symbol, TP::Low),
    LtLt("<<", PK::Literal, TC::Symbol, TP::Low),
    GtGt(">>", PK::Literal, TC::Symbol, TP::Low),
    Plus("+", PK::Literal, TC::Symbol, TP::Low),
    Minus("-", PK::Literal, TC::Symbol, TP::Low),
    Star("*", PK::Literal, TC::Symbol, TP::Low),
    Slash("/", PK::Literal, TC::Symbol, TP::Low),
    Percent("%", PK::Literal, TC::Symbol, TP::Low),
    Bang("!", PK::Literal, TC::Symbol, TP::Low),
    StarStar("**", PK::Literal, TC::Symbol, TP::Low),

    // assignment operators
    Eq("=", PK::Literal, TC::Symbol, TP::Low),
    PlusEq("+=", PK::Literal, TC::Symbol, TP::Low),
    MinusEq("-=", PK::Literal, TC::Symbol, TP::Low),
    StarEq("*=", PK::Literal, TC::Symbol, TP::Low),
    SlashEq("/=", PK::Literal, TC::Symbol, TP::Low),
    PercentEq("%=", PK::Literal, TC::Symbol, TP::Low),
    AmperEq("&=", PK::Literal, TC::Symbol, TP::Low),
    CircumflexEq("^=", PK::Literal, TC::Symbol, TP::Low),
    BarEq("|=", PK::Literal, TC::Symbol, TP::Low),
}

#[cfg(test)]
mod test {
    use crate::syntax::pos::FileId;
    use crate::syntax::token::tokenize;

    #[test]
    fn empty_tokenize() {
        assert_eq!(Ok(vec![]), tokenize(FileId::SINGLE, ""));
        assert!(tokenize(FileId::SINGLE, "test foo function \"foo\"").is_ok());
    }
}
