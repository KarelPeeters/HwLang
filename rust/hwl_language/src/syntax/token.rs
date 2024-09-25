use itertools::Itertools;
use lazy_static::lazy_static;
use regex::{Regex, RegexSet, SetMatches};
use strum::EnumIter;

use crate::syntax::pos::{FileId, Pos, Span};

#[derive(Debug, Eq, PartialEq)]
pub struct Token<S> {
    pub ty: TokenType<S>,
    pub span: Span,
}

// TODO remove string from this? we have better error infrastructure by now
#[derive(Debug, Eq, PartialEq)]
pub enum TokenError {
    InvalidToken { pos: Pos, prefix: String },
    BlockCommentMissingEnd { start: Pos, eof: Pos },
    BlockCommentUnexpectedEnd { pos: Pos, prefix: String },
}

const ERROR_CONTEXT_LENGTH: usize = 16;

pub fn tokenize(file: FileId, source: &str) -> Result<Vec<Token<&str>>, TokenError> {
    Tokenizer::new(file, source).try_collect()
}

// TODO implement recovery by matching without start anchor?
// TODO error check regex overlap in advance at test-time using ... (called from a unit test too)
//   https://users.rust-lang.org/t/detect-regex-conflict/57184/13
// TODO remove string here? only deal with offset, simplifying the lifetime
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

    fn skip(&mut self, n: usize) {
        self.left = &self.left[n..];
        self.curr_byte += n;
    }

    fn curr_pos(&self) -> Pos {
        Pos { file: self.file, byte: self.curr_byte }
    }

    fn prefix(&self) -> String {
        self.left.chars().take(ERROR_CONTEXT_LENGTH).collect()
    }

    /// Block comments are handled separately. They're allowed to nest, which means they're not a regular language and 
    /// they can't be parsed using a Regex engine.
    fn handle_block_comment(&mut self) -> Option<Result<Token<&'s str>, TokenError>> {
        let start = self.curr_pos();

        if self.left.starts_with("*/") {
            self.errored = true;
            return Some(Err(TokenError::BlockCommentUnexpectedEnd {
                pos: start,
                prefix: self.prefix(),
            }));
        } else if self.left.starts_with("/*") {
            let left_start = self.left;
            self.skip(2);

            let mut depth: usize = 1;
            while depth > 0 {
                if self.left.starts_with("/*") {
                    depth += 1;
                    self.skip(2);
                } else if self.left.starts_with("*/") {
                    depth -= 1;
                    self.skip(2);
                } else if self.left.len() > 0 {
                    let c = self.left.chars().next().expect("nonempty string must contain char");
                    self.skip(c.len_utf8())
                } else {
                    // hit end of source
                    self.errored = true;
                    return Some(Err(TokenError::BlockCommentMissingEnd {
                        start,
                        eof: self.curr_pos(),
                    }));
                }
            }

            let span = Span::new(start, self.curr_pos());
            return Some(Ok(Token {
                ty: TokenType::BlockComment(&left_start[..span.len_bytes()]),
                span,
            }));
        }

        None
    }

    fn handle_pattern_token(&mut self) -> Result<Token<&'s str>, TokenError> {
        let start = self.curr_pos();
        let matches = self.compiled.set.matches(self.left);

        let m = match pick_match(matches, &self.compiled.vec, self.left) {
            None => {
                self.errored = true;
                return Err(TokenError::InvalidToken {
                    pos: self.curr_pos(),
                    prefix: self.prefix(),
                });
            }
            Some(match_index) => match_index,
        };

        let match_str = &self.left[..m.len];
        self.skip(m.len);
        let span = Span::new(start, self.curr_pos());

        Ok(Token {
            ty: TOKEN_PATTERNS[m.index].0(match_str),
            span,
        })
    }
}

impl<'s> Iterator for Tokenizer<'s> {
    type Item = Result<Token<&'s str>, TokenError>;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(!self.errored, "Cannot continue calling next on tokenizer that returned an error");
        if self.left.is_empty() {
            return None;
        }

        let start = self.curr_pos();
        if let Some(result) = self.handle_block_comment() {
            return Some(result);
        }
        assert_eq!(start, self.curr_pos());
        Some(self.handle_pattern_token())
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
                    PatternKind::Regex => pattern.to_string(),
                    PatternKind::Literal => regex::escape(pattern),
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

        let range = regex_vec[index]
            .find(left)
            .expect("regex should match, it was one of the matching indices")
            .range();
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
        assert_eq!(match_count, 1);
    }
    result
}

macro_rules! declare_tokens {
    (
        custom {
            $($c_token:ident($c_cat:expr),)*
        }
        regex {
            $($r_token:ident($r_string:literal, $r_cat:expr, $r_prio:expr),)*
        }
        literal {
            $($l_token:ident($l_string:literal, $l_cat:expr, $l_prio:expr),)*
        }
    ) => {
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TokenType<S> {
            $($c_token(S),)*
            $($r_token(S),)*
            $($l_token,)*
        }

        const TOKEN_PATTERNS: &[(fn (&str) -> TokenType<&str>, &'static str, PatternKind, TokenPriority)] = &[
            // intentionally omit custom
            $((|s| TokenType::$r_token(s), $r_string, PatternKind::Regex, $r_prio),)*
            $((|_| TokenType::$l_token, $l_string, PatternKind::Literal, $l_prio),)*
        ];

        // TODO function vs array?
        impl<S> TokenType<S> {
            pub fn category(self) -> TokenCategory {
                match self {
                    $(TokenType::$c_token(_) => $c_cat,)*
                    $(TokenType::$r_token(_) => $r_cat,)*
                    $(TokenType::$l_token => $l_cat,)*
                }
            }
            
            pub fn map<T>(self, f: impl FnOnce(S) -> T) -> TokenType<T> {
                match self {
                    $(TokenType::$c_token(s) => TokenType::$c_token(f(s)),)*
                    $(TokenType::$r_token(s) => TokenType::$r_token(f(s)),)*
                    $(TokenType::$l_token => TokenType::$l_token,)*
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, EnumIter, strum::Display)]
pub enum TokenCategory {
    WhiteSpace,
    Comment,
    Identifier,
    IntegerLiteral,
    StringLiteral,
    Keyword,
    Symbol,
}

impl TokenCategory {
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Priority is only used to tiebreak equal-length matches.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenPriority {
    Unique = 2,
    Normal = 1,
    Low = 0,
}

use crate::data::diagnostic::{Diagnostic, DiagnosticAddable};
use TokenCategory as TC;
use TokenPriority as TP;

declare_tokens! {
    custom {
        BlockComment(TC::Comment),
    }
    regex {
        // ignored
        WhiteSpace(r"[ \t\n\r]+", TC::WhiteSpace, TP::Unique),
        LineComment(r"//[^\n\r]*", TC::Comment, TP::Normal),

        // patterns
        Identifier(r"(_[a-zA-Z_0-9]+)|([a-zA-Z][a-zA-Z_0-9]*)", TC::Identifier, TP::Low),
        IntLiteralDecimal(r"[0-9]+", TC::IntegerLiteral, TP::Unique),
        IntPatternHexadecimal(r"0x[0-9a-fA-F_?]+", TC::IntegerLiteral, TP::Unique),
        IntPatternBinary(r"0b[0-9a-fA-F_?]+", TC::IntegerLiteral, TP::Unique),

        // TODO better string literal pattern with escape codes and string formatting expressions
        StringLiteral(r#""[^"]*""#, TC::StringLiteral, TP::Unique),
    }
    literal {
        // keywords
        Use("use", TC::Keyword, TP::Normal),
        As("as", TC::Keyword, TP::Normal),
        Type("type", TC::Keyword, TP::Normal),
        Any("any", TC::Keyword, TP::Normal),
        Struct("struct", TC::Keyword, TP::Normal),
        Enum("enum", TC::Keyword, TP::Normal),
        Ports("ports", TC::Keyword, TP::Normal),
        Module("module", TC::Keyword, TP::Normal),
        Function("function", TC::Keyword, TP::Normal),
        Combinatorial("combinatorial", TC::Keyword, TP::Normal),
        Clock("clock", TC::Keyword, TP::Normal),
        Clocked("clocked", TC::Keyword, TP::Normal),
        Const("const", TC::Keyword, TP::Normal),
        Val("val", TC::Keyword, TP::Normal),
        Var("var", TC::Keyword, TP::Normal),
        Wire("wire", TC::Keyword, TP::Normal),
        Reg("reg", TC::Keyword, TP::Normal),
        Input("input", TC::Keyword, TP::Normal),
        Output("output", TC::Keyword, TP::Normal),
        Async("async", TC::Keyword, TP::Normal),
        Sync("sync", TC::Keyword, TP::Normal),
        Return("return", TC::Keyword, TP::Normal),
        Break("break", TC::Keyword, TP::Normal),
        Continue("continue", TC::Keyword, TP::Normal),
        True("true", TC::Keyword, TP::Normal),
        False("false", TC::Keyword, TP::Normal),
        If("if", TC::Keyword, TP::Normal),
        Else("else", TC::Keyword, TP::Normal),
        Loop("loop", TC::Keyword, TP::Normal),
        For("for", TC::Keyword, TP::Normal),
        In("in", TC::Keyword, TP::Normal),
        While("while", TC::Keyword, TP::Normal),
        Public("pub", TC::Keyword, TP::Normal),
        Builtin("__builtin", TC::Keyword, TP::Normal),
    
        // misc symbols
        Semi(";", TC::Symbol, TP::Normal),
        Colon(":", TC::Symbol, TP::Normal),
        Comma(",", TC::Symbol, TP::Normal),
        Arrow("->", TC::Symbol, TP::Normal),
        Underscore("_", TC::Symbol, TP::Normal),
        ColonColon("::", TC::Symbol, TP::Normal),
    
        // braces
        OpenC("{", TC::Symbol, TP::Normal),
        CloseC("}", TC::Symbol, TP::Normal),
        OpenR("(", TC::Symbol, TP::Normal),
        CloseR(")", TC::Symbol, TP::Normal),
        OpenS("[", TC::Symbol, TP::Normal),
        CloseS("]", TC::Symbol, TP::Normal),
    
        // operators
        Dot(".", TC::Symbol, TP::Normal),
        Dots("..", TC::Symbol, TP::Normal),
        DotsEq("..=", TC::Symbol, TP::Normal),
        AmperAmper("&&", TC::Symbol, TP::Normal),
        PipePipe("||", TC::Symbol, TP::Normal),
        EqEq("==", TC::Symbol, TP::Normal),
        Neq("!=", TC::Symbol, TP::Normal),
        Gte(">=", TC::Symbol, TP::Normal),
        Gt(">", TC::Symbol, TP::Normal),
        Lte("<=", TC::Symbol, TP::Normal),
        Lt("<", TC::Symbol, TP::Normal),
        Amper("&", TC::Symbol, TP::Normal),
        Circumflex("^", TC::Symbol, TP::Normal),
        Pipe("|", TC::Symbol, TP::Normal),
        LtLt("<<", TC::Symbol, TP::Normal),
        GtGt(">>", TC::Symbol, TP::Normal),
        Plus("+", TC::Symbol, TP::Normal),
        Minus("-", TC::Symbol, TP::Normal),
        Star("*", TC::Symbol, TP::Normal),
        Slash("/", TC::Symbol, TP::Normal),
        Percent("%", TC::Symbol, TP::Normal),
        Bang("!", TC::Symbol, TP::Normal),
        StarStar("**", TC::Symbol, TP::Normal),
    
        // assignment operators
        Eq("=", TC::Symbol, TP::Normal),
        PlusEq("+=", TC::Symbol, TP::Normal),
        MinusEq("-=", TC::Symbol, TP::Normal),
        StarEq("*=", TC::Symbol, TP::Normal),
        SlashEq("/=", TC::Symbol, TP::Normal),
        PercentEq("%=", TC::Symbol, TP::Normal),
        AmperEq("&=", TC::Symbol, TP::Normal),
        CircumflexEq("^=", TC::Symbol, TP::Normal),
        BarEq("|=", TC::Symbol, TP::Normal),
    }
}

impl TokenError {
    pub fn to_diagnostic(self) -> Diagnostic {
        match self {
            TokenError::InvalidToken { pos, prefix: _ } => {
                Diagnostic::new("tokenization error")
                    .add_error(Span::empty_at(pos), "invalid prefix")
                    .finish()
            }
            TokenError::BlockCommentMissingEnd { start, eof } => {
                Diagnostic::new("block comment missing end")
                    .add_info(Span::empty_at(start), "block comment started here")
                    .add_error(Span::empty_at(eof), "end of file reached")
                    .finish()
            }
            TokenError::BlockCommentUnexpectedEnd { pos, prefix: _ } => {
                Diagnostic::new("unexpected end of block comment")
                    .add_error(Span::empty_at(pos), "end of comment here")
                    .finish()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::syntax::pos::{FileId, Pos, Span};
    use crate::syntax::token::{tokenize, PatternKind, Token, TokenType, TOKEN_PATTERNS};

    #[test]
    fn basic_tokenize() {
        let file = FileId::SINGLE;
        
        assert_eq!(Ok(vec![]), tokenize(file, ""));
        assert_eq!(Ok(vec![Token {
            ty: TokenType::WhiteSpace("\n"),
            span: Span { start: Pos { file, byte: 0 }, end: Pos { file, byte: 1 } },
        }]), tokenize(file, "\n"));
        assert!(tokenize(file, "test foo function \"foo\"").is_ok());
    }

    #[test]
    fn comment() {
        let file = FileId::SINGLE;
        assert_eq!(Ok(vec![Token {
            ty: TokenType::BlockComment("/**/"),
            span: Span { start: Pos { file, byte: 0 }, end: Pos { file, byte: 4 } },
        }]), tokenize(file, "/**/"));

        assert_eq!(Ok(vec![Token {
            ty: TokenType::BlockComment("/*/**/*/"),
            span: Span { start: Pos { file, byte: 0 }, end: Pos { file, byte: 8 } },
        }]), tokenize(file, "/*/**/*/"));

        assert!(tokenize(file, "/*/**/").is_err());
    }

    // TODO turn this into a test case that checks whether the grammer is up-to-date
    #[test]
    fn print_grammer_enum() {
        for (token_type, pattern, pattern_kind, _) in TOKEN_PATTERNS {
            match pattern_kind {
                PatternKind::Regex => {
                    println!("Token{:?} => TokenType::{:?},", token_type, token_type)
                }
                PatternKind::Literal => {
                    println!("{:?} => TokenType::{:?},", pattern, token_type)
                }
            }
        }
    }
}
