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
                Kind::Regex => pattern.to_string(),
                Kind::Literal => regex::escape(pattern),
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
            Kind::Regex => {
                assert!(
                    single_regex.is_none(),
                    "overlap between regex {:?} and {}",
                    single_regex,
                    index
                );
                single_regex = Some(index);
            }
            Kind::Literal => {
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
    ($($token:ident($string:literal, $kind:expr),)*) => {
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TokenType {
            $($token,)*
        }

        const TOKEN_PATTERNS: &[(TokenType, &'static str, Kind)] = &[
            $((TokenType::$token, $string, $kind),)*
        ];
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Kind {
    Regex,
    Literal,
}

declare_tokens! {
    // ignored
    WhiteSpace(r"\s+", Kind::Regex),
    LineComment(r"//[^\n\r]*[\n\r]*", Kind::Regex),
    BlockComment(r"/\*([^\*]*\*+[^\*/])*([^\*]*\*+|[^\*])*\*/", Kind::Regex),

    // patterns
    Identifier(r"(_[a-zA-Z_0-9]+)|([a-zA-Z][a-zA-Z_0-9]*)", Kind::Regex),
    IntLiteralDecimal(r"[0-9]+", Kind::Regex),
    IntPatternHexadecimal(r"0x[0-9a-fA-F_?]+", Kind::Regex),
    IntPatternBinary(r"0b[0-9a-fA-F_?]+", Kind::Regex),

    // TODO better string literal pattern with escape codes and string formatting expressions
    StringLiteral(r#""[^"]*""#, Kind::Regex),

    // keywords
    Use("use", Kind::Literal),
    As("as", Kind::Literal),
    Type("type", Kind::Literal),
    Struct("struct", Kind::Literal),
    Enum("enum", Kind::Literal),
    Ports("ports", Kind::Literal),
    Module("module", Kind::Literal),
    Function("function", Kind::Literal),
    Combinatorial("combinatorial", Kind::Literal),
    Clocked("clocked", Kind::Literal),
    Const("const", Kind::Literal),
    Val("val", Kind::Literal),
    Var("var", Kind::Literal),
    Input("input", Kind::Literal),
    Output("output", Kind::Literal),
    Async("async", Kind::Literal),
    Sync("sync", Kind::Literal),
    Return("return", Kind::Literal),
    Break("break", Kind::Literal),
    Continue("continue", Kind::Literal),
    True("true", Kind::Literal),
    False("false", Kind::Literal),
    If("if", Kind::Literal),
    Else("else", Kind::Literal),
    Loop("loop", Kind::Literal),
    For("for", Kind::Literal),
    In("in", Kind::Literal),
    While("while", Kind::Literal),
    Public("public", Kind::Literal),

    // misc symbols
    Semi(";", Kind::Literal),
    Colon(":", Kind::Literal),
    Comma(",", Kind::Literal),
    Arrow("->", Kind::Literal),
    Underscore("_", Kind::Literal),
    ColonColon("::", Kind::Literal),

    // braces
    OpenC("{", Kind::Literal),
    CloseC("}", Kind::Literal),
    OpenR("(", Kind::Literal),
    CloseR(")", Kind::Literal),
    OpenS("[", Kind::Literal),
    CloseS("]", Kind::Literal),

    // operators
    Dot(".", Kind::Literal),
    Dots("..", Kind::Literal),
    DotsEq("..=", Kind::Literal),
    AmperAmper("&&", Kind::Literal),
    PipePipe("||", Kind::Literal),
    EqEq("==", Kind::Literal),
    Neq("!=", Kind::Literal),
    Gte(">=", Kind::Literal),
    Gt(">", Kind::Literal),
    Lte("<=", Kind::Literal),
    Lt("<", Kind::Literal),
    Amper("&", Kind::Literal),
    Circumflex("^", Kind::Literal),
    Pipe("|", Kind::Literal),
    LtLt("<<", Kind::Literal),
    GtGt(">>", Kind::Literal),
    Plus("+", Kind::Literal),
    Minus("-", Kind::Literal),
    Star("*", Kind::Literal),
    Slash("/", Kind::Literal),
    Percent("%", Kind::Literal),
    Bang("!", Kind::Literal),
    StarStar("**", Kind::Literal),

    // assignment operators
    Eq("=", Kind::Literal),
    PlusEq("+=", Kind::Literal),
    MinusEq("-=", Kind::Literal),
    StarEq("*=", Kind::Literal),
    SlashEq("/=", Kind::Literal),
    PercentEq("%=", Kind::Literal),
    AmperEq("&=", Kind::Literal),
    CircumflexEq("^=", Kind::Literal),
    BarEq("|=", Kind::Literal),
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
