use std::cmp::min;

use itertools::Itertools;
use regex::{Regex, RegexSet, SetMatches};

use crate::syntax::pos::{FileId, Pos, Span};


#[derive(Debug, Eq, PartialEq)]
pub struct Token<S> {
    ty: TokenType,
    string: S,
    span: Span,
}

#[derive(Debug, Eq, PartialEq)]
pub struct InvalidToken {
    pos: Pos,
    prefix: String,
}

// TODO error check regex overlap in advance at test-time using
//   https://users.rust-lang.org/t/detect-regex-conflict/57184/13
// TODO use lazy_static to compile the regexes only once?
pub fn tokenize(file: FileId, source: &str) -> Result<Vec<Token<&str>>, InvalidToken> {
    let patterns = TOKEN_PATTERNS
        .iter()
        .map(|(_, pattern, kind)| {
            let bare = match kind {
                PK::Regex => pattern.to_string(),
                PK::Literal => regex::escape(pattern),
            };
            format!("^(:?{bare})")
        })
        .collect_vec();

    println!("patterns:");
    for p in &patterns {
        println!("  {p:?}");
    }

    let regex_set = RegexSet::new(&patterns).unwrap();
    let regex_vec = patterns.iter().map(|p| Regex::new(p).unwrap()).collect_vec();
    println!("{:?}", regex_set);

    let mut left = source;
    let mut tokens = vec![];

    let mut pos = Pos { file, line: 0, col: 0 };

    while !left.is_empty() {
        let log_prefix = &left[..min(left.len(), 16)];
        println!("left: {:?}", log_prefix);

        let matches = regex_set.matches(left);

        println!("matches:");
        for m in matches.iter() {
            let match_str = regex_vec[m].find(left).unwrap().as_str();
            println!("  {m}: {:?} {:?}", &TOKEN_PATTERNS[m], match_str);
        }

        let match_index = match pick_match(matches) {
            None => {
                return Err(InvalidToken {
                    pos,
                    prefix: log_prefix.to_owned(),
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

fn pick_match(matches: SetMatches) -> Option<usize> {
    let mut single_regex = None;
    let mut longest_literal = None;

    for index in matches.iter() {
        let (_, pattern, kind) = TOKEN_PATTERNS[index];
        match kind {
            PK::Regex => {
                assert!(
                    single_regex.is_none(),
                    "overlap between regex {:?} and {}",
                    single_regex,
                    index
                );
                single_regex = Some(index);
            }
            PK::Literal => {
                let curr_len = pattern.len();
                match longest_literal {
                    None => longest_literal = Some((index, curr_len)),
                    Some((_, prev_len)) => {
                        assert!(curr_len != prev_len);
                        if curr_len > prev_len {
                            longest_literal = Some((index, curr_len))
                        }
                    }
                }
            }
        }
    }

    longest_literal.map(|(i, _)| i).or(single_regex)
}

macro_rules! declare_tokens {
    ($($token:ident($string:literal, $pattern_kind:expr, $token_category:expr),)*) => {
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TokenType {
            $($token,)*
        }

        const TOKEN_PATTERNS: &[(TokenType, &'static str, PatternKind)] = &[
            $((TokenType::$token, $string, $pattern_kind),)*
        ];

        impl TokenType {
            pub fn category(self) -> TokenCategory {
                match self {
                    $(TokenType::$token => $token_category,)*
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TokenCategory {
    WhiteSpace,
    Comment,
    Identifier,
    IntegerLiteral,
    StringLiteral,
    Keyword,
    Symbol,
}

use TokenCategory as TC;
use PatternKind as PK;

declare_tokens! {
    // ignored
    WhiteSpace(r"\s+", PK::Regex, TC::WhiteSpace),
    LineComment(r"//[^\n\r]*[\n\r]*", PK::Regex, TC::Comment),
    BlockComment(r"/\*([^\*]*\*+[^\*/])*([^\*]*\*+|[^\*])*\*/", PK::Regex, TC::Comment),

    // patterns
    Identifier(r"(_[a-zA-Z_0-9]+)|([a-zA-Z][a-zA-Z_0-9]*)", PK::Regex, TC::Identifier),
    IntLiteralDecimal(r"[0-9]+", PK::Regex, TC::IntegerLiteral),
    IntPatternHexadecimal(r"0x[0-9a-fA-F_?]+", PK::Regex, TC::IntegerLiteral),
    IntPatternBinary(r"0b[0-9a-fA-F_?]+", PK::Regex, TC::IntegerLiteral),

    // TODO better string literal pattern with escape codes and string formatting expressions
    StringLiteral(r#""[^"]*""#, PK::Regex, TC::StringLiteral),

    // keywords
    Use("use", PK::Literal, TC::Keyword),
    As("as", PK::Literal, TC::Keyword),
    Type("type", PK::Literal, TC::Keyword),
    Struct("struct", PK::Literal, TC::Keyword),
    Enum("enum", PK::Literal, TC::Keyword),
    Ports("ports", PK::Literal, TC::Keyword),
    Module("module", PK::Literal, TC::Keyword),
    Function("function", PK::Literal, TC::Keyword),
    Combinatorial("combinatorial", PK::Literal, TC::Keyword),
    Clocked("clocked", PK::Literal, TC::Keyword),
    Const("const", PK::Literal, TC::Keyword),
    Val("val", PK::Literal, TC::Keyword),
    Var("var", PK::Literal, TC::Keyword),
    Input("input", PK::Literal, TC::Keyword),
    Output("output", PK::Literal, TC::Keyword),
    Async("async", PK::Literal, TC::Keyword),
    Sync("sync", PK::Literal, TC::Keyword),
    Return("return", PK::Literal, TC::Keyword),
    Break("break", PK::Literal, TC::Keyword),
    Continue("continue", PK::Literal, TC::Keyword),
    True("true", PK::Literal, TC::Keyword),
    False("false", PK::Literal, TC::Keyword),
    If("if", PK::Literal, TC::Keyword),
    Else("else", PK::Literal, TC::Keyword),
    Loop("loop", PK::Literal, TC::Keyword),
    For("for", PK::Literal, TC::Keyword),
    In("in", PK::Literal, TC::Keyword),
    While("while", PK::Literal, TC::Keyword),
    Public("public", PK::Literal, TC::Keyword),

    // misc symbols
    Semi(";", PK::Literal, TC::Symbol),
    Colon(":", PK::Literal, TC::Symbol),
    Comma(",", PK::Literal, TC::Symbol),
    Arrow("->", PK::Literal, TC::Symbol),
    Underscore("_", PK::Literal, TC::Symbol),
    ColonColon("::", PK::Literal, TC::Symbol),

    // braces
    OpenC("{", PK::Literal, TC::Symbol),
    CloseC("}", PK::Literal, TC::Symbol),
    OpenR("(", PK::Literal, TC::Symbol),
    CloseR(")", PK::Literal, TC::Symbol),
    OpenS("[", PK::Literal, TC::Symbol),
    CloseS("]", PK::Literal, TC::Symbol),

    // operators
    Dot(".", PK::Literal, TC::Symbol),
    Dots("..", PK::Literal, TC::Symbol),
    DotsEq("..=", PK::Literal, TC::Symbol),
    AmperAmper("&&", PK::Literal, TC::Symbol),
    PipePipe("||", PK::Literal, TC::Symbol),
    EqEq("==", PK::Literal, TC::Symbol),
    Neq("!=", PK::Literal, TC::Symbol),
    Gte(">=", PK::Literal, TC::Symbol),
    Gt(">", PK::Literal, TC::Symbol),
    Lte("<=", PK::Literal, TC::Symbol),
    Lt("<", PK::Literal, TC::Symbol),
    Amper("&", PK::Literal, TC::Symbol),
    Circumflex("^", PK::Literal, TC::Symbol),
    Pipe("|", PK::Literal, TC::Symbol),
    LtLt("<<", PK::Literal, TC::Symbol),
    GtGt(">>", PK::Literal, TC::Symbol),
    Plus("+", PK::Literal, TC::Symbol),
    Minus("-", PK::Literal, TC::Symbol),
    Star("*", PK::Literal, TC::Symbol),
    Slash("/", PK::Literal, TC::Symbol),
    Percent("%", PK::Literal, TC::Symbol),
    Bang("!", PK::Literal, TC::Symbol),
    StarStar("**", PK::Literal, TC::Symbol),

    // assignment operators
    Eq("=", PK::Literal, TC::Symbol),
    PlusEq("+=", PK::Literal, TC::Symbol),
    MinusEq("-=", PK::Literal, TC::Symbol),
    StarEq("*=", PK::Literal, TC::Symbol),
    SlashEq("/=", PK::Literal, TC::Symbol),
    PercentEq("%=", PK::Literal, TC::Symbol),
    AmperEq("&=", PK::Literal, TC::Symbol),
    CircumflexEq("^=", PK::Literal, TC::Symbol),
    BarEq("|=", PK::Literal, TC::Symbol),
}

#[cfg(test)]
mod test {
    use crate::syntax::pos::FileId;
    use crate::syntax::token::tokenize;

    #[test]
    fn empty_tokenize() {
        assert_eq!(Ok(vec![]), tokenize(FileId(0), ""));
        assert!(tokenize(FileId(0), "test foo function \"foo\"").is_ok());
    }
}
