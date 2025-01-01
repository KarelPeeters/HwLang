use crate::front::diagnostic::{Diagnostic, DiagnosticAddable};
use itertools::Itertools;
use lazy_static::lazy_static;
use logos::Source;
use std::cmp::Reverse;
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
    StringLiteralMissingEnd { start: Pos, eof: Pos },
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
    file: FileId,
    curr_byte: usize,
    left: &'s str,
    errored: bool,
    fixed_tokens: &'static [(&'static str, fn(&str) -> TokenType<&str>, &'static str)],
}

impl<'s> Tokenizer<'s> {
    pub fn new(file: FileId, source: &'s str) -> Self {
        Tokenizer {
            file,
            curr_byte: 0,
            left: source,
            errored: false,
            fixed_tokens: &FIXED_TOKENS,
        }
    }

    fn skip(&mut self, n: usize) {
        self.left = &self.left[n..];
        self.curr_byte += n;
    }

    fn curr_pos(&self) -> Pos {
        Pos {
            file: self.file,
            byte: self.curr_byte,
        }
    }

    fn prefix(&self) -> String {
        self.left.chars().take(ERROR_CONTEXT_LENGTH).collect()
    }

    fn parse_start_continue(
        &mut self,
        is_start: impl Fn(char) -> bool,
        is_continue: impl Fn(char) -> bool,
        build: impl Fn(&'s str) -> TokenType<&'s str>,
    ) -> Option<Token<&'s str>> {
        let start = self.curr_pos();
        let left_start = self.left;

        let mut chars = self.left.chars();
        if let Some(first) = chars.next() {
            if is_start(first) {
                let mut len = 0;
                len += first.len_utf8();

                while let Some(c) = chars.next() {
                    if is_continue(c) {
                        len += c.len_utf8();
                    } else {
                        break;
                    }
                }

                self.skip(len);
                let span = Span::new(start, self.curr_pos());

                return Some(Token {
                    span,
                    ty: build(&left_start[..len]),
                });
            }
        }

        None
    }

    // TODO try reordering to maximize performance
    // TODO try generating a full character-based state machine at compile-time, that might be faster
    fn next_impl(&mut self) -> Result<Token<&'s str>, TokenError> {
        let start = self.curr_pos();
        let left_start = self.left;
        let fixed_tokens = self.fixed_tokens;

        // block comment
        if self.left.starts_with("/*") {
            self.skip(2);

            // block comments are allowed to nest
            let mut depth: usize = 1;
            while depth > 0 {
                if self.left.starts_with("/*") {
                    depth += 1;
                    self.skip(2);
                } else if self.left.starts_with("*/") {
                    depth -= 1;
                    self.skip(2);
                } else if let Some(c) = self.left.chars().next() {
                    self.skip(c.len_utf8())
                } else {
                    // hit end of source
                    return Err(TokenError::BlockCommentMissingEnd {
                        start,
                        eof: self.curr_pos(),
                    });
                }
            }

            let span = Span::new(start, self.curr_pos());
            return Ok(Token {
                ty: TokenType::BlockComment(&left_start[..span.len_bytes()]),
                span,
            });
        }
        if self.left.starts_with("*/") {
            return Err(TokenError::BlockCommentUnexpectedEnd {
                pos: start,
                prefix: self.prefix(),
            });
        }

        // line comment
        // TODO should it include the trailing newline? \n\r handling becomes a bit tricky then
        if self.left.starts_with("//") {
            let len = memchr::memchr2(b'\n', b'\r', self.left.as_bytes()).map_or(self.left.len(), |end| end);
            self.skip(len);

            let span = Span::new(start, self.curr_pos());
            return Ok(Token {
                ty: TokenType::LineComment(&left_start[..len]),
                span,
            });
        }

        let is_whitespace = |c| matches!(c, ' ' | '\t' | '\n' | '\r');
        if let Some(token) = self.parse_start_continue(is_whitespace, is_whitespace, TokenType::WhiteSpace) {
            return Ok(token);
        }

        let is_id_start = |c| matches!(c, '_' | 'a'..='z' | 'A'..='Z');
        let is_id_continue = |c| matches!(c, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9');
        let build_id = move |id| {
            // check for fixed matches, they might overlap with IDs
            // TODO create a separate sublist of fixed tokens that also match identifiers to speed this up a bit more
            for (fixed, build, _) in fixed_tokens {
                if &id == fixed {
                    return build(id);
                }
            }
            return TokenType::Identifier(id);
        };
        if let Some(token) = self.parse_start_continue(is_id_start, is_id_continue, build_id) {
            return Ok(token);
        }

        // string literal
        // TODO escape codes
        // TODO f-strings, also needs parser
        // TODO we're actually already parsing here, it would be better if we could pass the inner string to the parser
        if self.left.starts_with('"') {
            let end_pos = memchr::memchr(b'"', &self.left.as_bytes()[1..]);
            return match end_pos {
                None => {
                    self.skip(self.left.len());
                    Err(TokenError::StringLiteralMissingEnd {
                        start,
                        eof: self.curr_pos(),
                    })
                }
                Some(end_pos) => {
                    let len_bytes = 1 + end_pos + 1;
                    self.skip(len_bytes);
                    let span = Span::new(start, self.curr_pos());
                    Ok(Token {
                        ty: TokenType::StringLiteral(&left_start[..len_bytes]),
                        span,
                    })
                }
            };
        }

        // integer literal/pattern
        // TODO parse hex/bin
        // TODO store the actually parsed int in the token to get more type safety
        let is_decimal_digit = |c| matches!(c, '0'..='9');
        if let Some(token) = self.parse_start_continue(is_decimal_digit, is_decimal_digit, TokenType::IntLiteral) {
            return Ok(token);
        }

        // fixed token (needs to be after identifiers to ensure ids that start with a fixed prefix get matched as ids)
        for (fixed, build, _) in self.fixed_tokens {
            if self.left.starts_with(fixed) {
                self.skip(fixed.len());

                let span = Span::new(start, self.curr_pos());
                return Ok(Token {
                    ty: build(&left_start[..fixed.len()]),
                    span,
                });
            }
        }

        // failed to match anything
        Err(TokenError::InvalidToken {
            pos: self.curr_pos(),
            prefix: self.prefix(),
        })
    }
}

impl<'s> Iterator for Tokenizer<'s> {
    type Item = Result<Token<&'s str>, TokenError>;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(
            !self.errored,
            "Cannot continue calling next on tokenizer that returned an error"
        );
        if self.left.is_empty() {
            None
        } else {
            match self.next_impl() {
                Ok(t) => Some(Ok(t)),
                Err(e) => {
                    self.errored = true;
                    Some(Err(e))
                }
            }
        }
    }
}

macro_rules! declare_tokens {
    (
        custom {
            $($c_token:ident($c_cat:expr),)*
        }
        fixed {
            $($f_token:ident($f_string:literal, $f_cat:expr),)*
        }
    ) => {
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TokenType<S> {
            $($c_token(S),)*
            $($f_token,)*
        }

        #[cfg(test)]
        const CUSTOM_TOKENS: &[(fn (&str) -> TokenType<&str>, &'static str)] = &[
            $((|s| TokenType::$c_token(s), stringify!($c_token)),)*
        ];
        const UNSORTED_FIXED_TOKENS: &[(&'static str, fn (&str) -> TokenType<&str>, &'static str)] = &[
            $(($f_string, |_| TokenType::$f_token, stringify!($f_token)),)*
        ];

        // TODO function vs array?
        impl<S> TokenType<S> {
            pub fn category(self) -> TokenCategory {
                match self {
                    $(TokenType::$c_token(_) => $c_cat,)*
                    $(TokenType::$f_token => $f_cat,)*
                }
            }

            pub fn map<T>(self, f: impl FnOnce(S) -> T) -> TokenType<T> {
                match self {
                    $(TokenType::$c_token(s) => TokenType::$c_token(f(s)),)*
                    $(TokenType::$f_token => TokenType::$f_token,)*
                }
            }
        }
    };
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

use TokenCategory as TC;

declare_tokens! {
    custom {
        // ignored
        WhiteSpace(TC::WhiteSpace),
        BlockComment(TC::Comment),
        LineComment(TC::Comment),

        // patterns
        Identifier(TC::Identifier),
        IntLiteral(TC::IntegerLiteral),
        // TODO better string literal pattern with escape codes and string formatting expressions
        StringLiteral(TC::StringLiteral),
    }
    fixed {
        // keywords
        Import("import", TC::Keyword),
        Type("type", TC::Keyword),
        Struct("struct", TC::Keyword),
        Enum("enum", TC::Keyword),
        Generics("generics", TC::Keyword),
        Ports("ports", TC::Keyword),
        Body("body", TC::Keyword),
        Module("module", TC::Keyword),
        Instance("instance", TC::Keyword),
        Function("function", TC::Keyword),
        Combinatorial("combinatorial", TC::Keyword),
        Clock("clock", TC::Keyword),
        Clocked("clocked", TC::Keyword),
        Const("const", TC::Keyword),
        Val("val", TC::Keyword),
        Var("var", TC::Keyword),
        Wire("wire", TC::Keyword),
        Reg("reg", TC::Keyword),
        In("in", TC::Keyword),
        Out("out", TC::Keyword),
        Async("async", TC::Keyword),
        Sync("sync", TC::Keyword),
        Return("return", TC::Keyword),
        Break("break", TC::Keyword),
        Continue("continue", TC::Keyword),
        True("true", TC::IntegerLiteral),
        False("false", TC::IntegerLiteral),
        Undefined("undefined", TC::IntegerLiteral),
        If("if", TC::Keyword),
        Else("else", TC::Keyword),
        Loop("loop", TC::Keyword),
        For("for", TC::Keyword),
        While("while", TC::Keyword),
        Public("pub", TC::Keyword),
        Builtin("__builtin", TC::Keyword),
        As("as", TC::Keyword),

        // misc symbols
        Semi(";", TC::Symbol),
        Colon(":", TC::Symbol),
        Comma(",", TC::Symbol),
        Arrow("->", TC::Symbol),
        Underscore("_", TC::Symbol),
        ColonColon("::", TC::Symbol),

        // braces
        OpenC("{", TC::Symbol),
        CloseC("}", TC::Symbol),
        OpenR("(", TC::Symbol),
        CloseR(")", TC::Symbol),
        OpenS("[", TC::Symbol),
        CloseS("]", TC::Symbol),

        // operators
        Dot(".", TC::Symbol),
        Dots("..", TC::Symbol),
        DotsEq("..=", TC::Symbol),
        AmperAmper("&&", TC::Symbol),
        PipePipe("||", TC::Symbol),
        CircumflexCircumflex("^^", TC::Symbol),
        EqEq("==", TC::Symbol),
        Neq("!=", TC::Symbol),
        Gte(">=", TC::Symbol),
        Gt(">", TC::Symbol),
        Lte("<=", TC::Symbol),
        Lt("<", TC::Symbol),
        Amper("&", TC::Symbol),
        Pipe("|", TC::Symbol),
        Circumflex("^", TC::Symbol),
        LtLt("<<", TC::Symbol),
        GtGt(">>", TC::Symbol),
        Plus("+", TC::Symbol),
        Minus("-", TC::Symbol),
        Star("*", TC::Symbol),
        Slash("/", TC::Symbol),
        Percent("%", TC::Symbol),
        Bang("!", TC::Symbol),
        StarStar("**", TC::Symbol),

        // assignment operators
        Eq("=", TC::Symbol),
        PlusEq("+=", TC::Symbol),
        MinusEq("-=", TC::Symbol),
        StarEq("*=", TC::Symbol),
        SlashEq("/=", TC::Symbol),
        PercentEq("%=", TC::Symbol),
        AmperEq("&=", TC::Symbol),
        CircumflexEq("^=", TC::Symbol),
        BarEq("|=", TC::Symbol),
    }
}

lazy_static! {
    /// Sorted from long to short, so that longer literals are matched first
    static ref FIXED_TOKENS: Vec<(&'static str, fn (&str) -> TokenType<&str>, &'static str)> = {
        let mut tokens = UNSORTED_FIXED_TOKENS.to_vec();
        tokens.sort_by_key(|(lit, _, _)| Reverse(lit.len()));
        tokens
    };
}

impl TokenError {
    pub fn to_diagnostic(self) -> Diagnostic {
        match self {
            TokenError::InvalidToken { pos, prefix: _ } => Diagnostic::new("tokenization error")
                .add_error(Span::empty_at(pos), "invalid start of token")
                .finish(),
            TokenError::BlockCommentMissingEnd { start, eof } => Diagnostic::new("block comment missing end")
                .add_info(Span::empty_at(start), "block comment started here")
                .add_error(Span::empty_at(eof), "end of file reached")
                .finish(),
            TokenError::BlockCommentUnexpectedEnd { pos, prefix: _ } => {
                Diagnostic::new("unexpected end of block comment")
                    .add_error(Span::empty_at(pos), "end of comment here")
                    .finish()
            }
            TokenError::StringLiteralMissingEnd { start, eof } => Diagnostic::new("string literal missing end")
                .add_info(Span::empty_at(start), "string literal started here")
                .add_error(Span::empty_at(eof), "end of file reached")
                .finish(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::syntax::pos::{FileId, Pos, Span};
    use crate::syntax::token::{tokenize, Token, TokenType, CUSTOM_TOKENS, UNSORTED_FIXED_TOKENS};

    #[test]
    fn basic_tokenize() {
        let file = FileId::SINGLE;

        assert_eq!(Ok(vec![]), tokenize(file, ""));
        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::WhiteSpace("\n"),
                span: Span {
                    start: Pos { file, byte: 0 },
                    end: Pos { file, byte: 1 }
                },
            }]),
            tokenize(file, "\n")
        );
        assert!(tokenize(file, "test foo function \"foo\"").is_ok());
    }

    #[test]
    fn comment() {
        let file = FileId::SINGLE;
        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::BlockComment("/**/"),
                span: Span {
                    start: Pos { file, byte: 0 },
                    end: Pos { file, byte: 4 }
                },
            }]),
            tokenize(file, "/**/")
        );

        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::BlockComment("/*/**/*/"),
                span: Span {
                    start: Pos { file, byte: 0 },
                    end: Pos { file, byte: 8 }
                },
            }]),
            tokenize(file, "/*/**/*/")
        );

        assert!(tokenize(file, "/*/**/").is_err());
    }

    // TODO turn this into a test case that checks whether the grammer is up-to-date
    #[test]
    fn print_grammer_enum() {
        for (_build, name) in CUSTOM_TOKENS {
            println!("Token{name} => TokenType::{name}(<&'s str>),")
        }
        for (literal, _build, name) in UNSORTED_FIXED_TOKENS {
            println!("{literal:?} => TokenType::{name},")
        }
    }
}
