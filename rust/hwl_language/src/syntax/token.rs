use crate::front::diagnostic::{Diagnostic, DiagnosticAddable};
use itertools::Itertools;
use lazy_static::lazy_static;
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
    Tokenizer::new(file, source).into_iter().try_collect()
}

// TODO implement recovery by matching without start anchor?
// TODO error check regex overlap in advance at test-time using ... (called from a unit test too)
//   https://users.rust-lang.org/t/detect-regex-conflict/57184/13
// TODO remove string here? only deal with offset, simplifying the lifetime
pub struct Tokenizer<'s> {
    file: FileId,
    curr_byte: usize,
    left: std::str::Chars<'s>,
    errored: bool,
    fixed_tokens_grouped_by_length: &'static [Vec<FixedTokenInfo>],
}

macro_rules! pattern_whitespace {
    () => {
        ' ' | '\t' | '\n' | '\r'
    };
}

macro_rules! pattern_id_start {
    () => {
        '_' | 'a'..='z' | 'A'..='Z'
    };
}

macro_rules! pattern_id_continue {
    () => {
        '_' | 'a'..='z' | 'A'..='Z' | '0'..='9'
    };
}

macro_rules! pattern_decimal_digit {
    () => {
        '0'..='9'
    };
}

impl<'s> Tokenizer<'s> {
    pub fn new(file: FileId, source: &'s str) -> Self {
        Tokenizer {
            file,
            curr_byte: 0,
            left: source.chars(),
            errored: false,
            fixed_tokens_grouped_by_length: &FIXED_TOKENS_GROUPED_BY_LENGTH,
        }
    }

    fn peek(&self) -> Option<char> {
        self.left.clone().next()
    }

    fn peek_second(&self) -> Option<char> {
        let mut iter = self.left.clone();
        iter.next()?;
        iter.next()
    }

    #[must_use]
    fn pop(&mut self) -> Option<char> {
        self.left.next().inspect(|c| self.curr_byte += c.len_utf8())
    }

    fn skip(&mut self, chars: usize) {
        for _ in 0..chars {
            match self.pop() {
                None => unreachable!(),
                Some(_) => {}
            }
        }
    }

    fn skip_until(&mut self, f: impl Fn(char) -> bool) {
        while let Some(c) = self.peek() {
            if f(c) {
                break;
            }
            self.curr_byte += c.len_utf8();
            self.left.next();
        }
    }

    fn curr_pos(&self) -> Pos {
        Pos {
            file: self.file,
            byte: self.curr_byte,
        }
    }

    fn prefix_for_error_message(&self) -> String {
        self.left.clone().take(ERROR_CONTEXT_LENGTH).collect()
    }

    // TODO generate this match state machine
    // TODO try memchr where it applies, see if it's actually faster
    fn next_inner(&mut self) -> Result<Option<Token<&'s str>>, TokenError> {
        let start = self.curr_pos();
        let start_left_str = self.left.as_str();

        let peek = [
            self.left.clone().nth(0).unwrap_or('\0'),
            self.left.clone().nth(1).unwrap_or('\0'),
        ];

        let mut skip_fixed = |n: usize, ty: TokenType<&'static str>| {
            self.skip(n);
            ty
        };

        let ty = match peek {
            ['\0', _] => return Ok(None),

            // custom
            [pattern_whitespace!(), _] => {
                self.skip(1);
                self.skip_until(|c| !matches!(c, pattern_whitespace!()));
                TokenType::WhiteSpace(&start_left_str[..self.curr_byte - start.byte])
            }
            [pattern_id_start!(), _] => {
                self.skip(1);
                self.skip_until(|c| !matches!(c, pattern_id_continue!()));
                let id = &start_left_str[..self.curr_byte - start.byte];

                // TODO speed up with runtime? literal tree
                match self.fixed_tokens_grouped_by_length.get(id.len()) {
                    None => TokenType::Identifier(id),
                    Some(potential) => potential
                        .iter()
                        .find_map(|info| if info.literal == id { Some(info.ty) } else { None })
                        .unwrap_or(TokenType::Identifier(id)),
                }
            }
            [pattern_decimal_digit!(), _] => {
                self.skip(1);
                #[allow(clippy::manual_is_ascii_check)]
                self.skip_until(|c| !matches!(c, pattern_decimal_digit!()));
                TokenType::IntLiteral(&start_left_str[..self.curr_byte - start.byte])
            }

            ['"', _] => {
                self.skip(1);
                self.skip_until(|c| c == '"');
                match self.peek() {
                    Some('"') => {
                        self.skip(1);
                        TokenType::StringLiteral(&start_left_str[..self.curr_byte - start.byte])
                    }
                    None => {
                        return Err(TokenError::StringLiteralMissingEnd {
                            start,
                            eof: self.curr_pos(),
                        });
                    }
                    _ => unreachable!(),
                }
            }
            ['/', '*'] => {
                // block comments are allowed to nest
                self.skip(2);
                let mut depth: usize = 1;
                while depth > 0 {
                    match (self.peek(), self.peek_second()) {
                        (Some('/'), Some('*')) => {
                            depth += 1;
                            self.skip(2);
                        }
                        (Some('*'), Some('/')) => {
                            depth -= 1;
                            self.skip(2);
                        }
                        (Some(_), _) => {
                            self.skip(1);
                        }
                        (None, _) => {
                            // hit end of source
                            return Err(TokenError::BlockCommentMissingEnd {
                                start,
                                eof: self.curr_pos(),
                            });
                        }
                    }
                }

                // end of comment, depth is 0 again
                TokenType::BlockComment(&start_left_str[..self.curr_byte - start.byte])
            }
            ['/', '/'] => {
                self.skip(2);
                self.skip_until(|c| c == '\n' || c == '\r');
                TokenType::LineComment(&start_left_str[..self.curr_byte - start.byte])
            }

            // trigram
            ['.', '.'] => {
                self.skip(2);
                match self.peek() {
                    Some('=') => {
                        self.skip(1);
                        TokenType::DotsEq
                    }
                    Some('+') => {
                        self.skip(1);
                        TokenType::DotsPlus
                    }
                    _ => TokenType::Dots,
                }
            }

            // simple fixed
            ['=', '='] => skip_fixed(2, TokenType::EqEq),
            ['=', _] => skip_fixed(1, TokenType::Eq),
            ['!', '='] => skip_fixed(2, TokenType::Neq),
            ['%', '='] => skip_fixed(2, TokenType::PercentEq),
            ['%', _] => skip_fixed(1, TokenType::Percent),
            ['!', _] => skip_fixed(1, TokenType::Bang),
            ['>', '='] => skip_fixed(2, TokenType::Gte),
            ['>', '>'] => skip_fixed(2, TokenType::GtGt),
            ['>', _] => skip_fixed(1, TokenType::Gt),
            ['<', '='] => skip_fixed(2, TokenType::Lte),
            ['<', '<'] => skip_fixed(2, TokenType::LtLt),
            ['<', _] => skip_fixed(1, TokenType::Lt),
            ['&', '&'] => skip_fixed(2, TokenType::AmperAmper),
            ['&', '='] => skip_fixed(2, TokenType::AmperEq),
            ['&', _] => skip_fixed(1, TokenType::Amper),
            ['|', '|'] => skip_fixed(2, TokenType::PipePipe),
            ['|', '='] => skip_fixed(2, TokenType::BarEq),
            ['|', _] => skip_fixed(1, TokenType::Pipe),
            ['^', '^'] => skip_fixed(2, TokenType::CircumflexCircumflex),
            ['^', '='] => skip_fixed(2, TokenType::CircumflexEq),
            ['^', _] => skip_fixed(1, TokenType::Circumflex),
            ['+', '='] => skip_fixed(2, TokenType::PlusEq),
            ['+', _] => skip_fixed(1, TokenType::Plus),
            ['-', '='] => skip_fixed(2, TokenType::MinusEq),
            ['-', '>'] => skip_fixed(2, TokenType::Arrow),
            ['-', _] => skip_fixed(1, TokenType::Minus),
            ['*', '='] => skip_fixed(2, TokenType::StarEq),
            ['*', '*'] => skip_fixed(2, TokenType::StarStar),
            ['*', _] => skip_fixed(1, TokenType::Star),
            ['/', '='] => skip_fixed(2, TokenType::SlashEq),
            ['/', _] => skip_fixed(1, TokenType::Slash),
            ['.', _] => skip_fixed(1, TokenType::Dot),
            [';', _] => skip_fixed(1, TokenType::Semi),
            [':', ':'] => skip_fixed(2, TokenType::ColonColon),
            [':', _] => skip_fixed(1, TokenType::Colon),
            [',', _] => skip_fixed(1, TokenType::Comma),
            ['(', _] => skip_fixed(1, TokenType::OpenR),
            [')', _] => skip_fixed(1, TokenType::CloseR),
            ['{', _] => skip_fixed(1, TokenType::OpenC),
            ['}', _] => skip_fixed(1, TokenType::CloseC),
            ['[', _] => skip_fixed(1, TokenType::OpenS),
            [']', _] => skip_fixed(1, TokenType::CloseS),

            [_, _] => {
                return Err(TokenError::InvalidToken {
                    pos: start,
                    prefix: self.prefix_for_error_message(),
                })
            }
        };

        let span = Span::new(start, self.curr_pos());
        Ok(Some(Token { span, ty }))
    }

    fn next(&mut self) -> Result<Option<Token<&'s str>>, TokenError> {
        assert!(
            !self.errored,
            "Cannot continue calling next on tokenizer that returned an error"
        );
        self.next_inner().inspect_err(|_| self.errored = true)
    }
}

impl<'s> IntoIterator for Tokenizer<'s> {
    type Item = Result<Token<&'s str>, TokenError>;
    type IntoIter = TokenizerIterator<'s>;

    fn into_iter(self) -> Self::IntoIter {
        TokenizerIterator { tokenizer: self }
    }
}

pub struct TokenizerIterator<'s> {
    tokenizer: Tokenizer<'s>,
}

impl<'s> Iterator for TokenizerIterator<'s> {
    type Item = Result<Token<&'s str>, TokenError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenizer.next().transpose()
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
        const FIXED_TOKENS: &[FixedTokenInfo] = &[
            $(FixedTokenInfo { name: stringify!($f_token), literal: $f_string, ty: TokenType::$f_token },)*
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
        DotsPlus("..+", TC::Symbol),
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

#[derive(Debug, Copy, Clone)]
struct FixedTokenInfo {
    #[allow(dead_code)]
    name: &'static str,
    literal: &'static str,
    ty: TokenType<&'static str>,
}

lazy_static! {
    static ref FIXED_TOKENS_GROUPED_BY_LENGTH: Vec<Vec<FixedTokenInfo>> = {
        let mut result = vec![];

        for &token in FIXED_TOKENS {
            let i = token.literal.len();
            if i >= result.len() {
                result.resize_with(i + 1, Vec::new);
            }
            result[i].push(token);
        }

        result
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
    use crate::syntax::token::{tokenize, Token, TokenType, CUSTOM_TOKENS, FIXED_TOKENS};
    use std::collections::HashSet;

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

    #[test]
    fn literal_tokens_unique() {
        let mut set = HashSet::new();
        for info in FIXED_TOKENS {
            assert!(!info.literal.is_empty());
            assert!(set.insert(info.literal));
        }
    }

    #[test]
    fn literal_tokens_covered() {
        let mut any_error = false;

        for info in FIXED_TOKENS {
            let file = FileId::SINGLE;

            let result = tokenize(file, info.literal);
            let span = Span::new(
                Pos { file, byte: 0 },
                Pos {
                    file,
                    byte: info.literal.len(),
                },
            );
            let expected = Ok(vec![Token { ty: info.ty, span }]);

            if result != expected {
                any_error = true;
                eprintln!("Failed to parse literal token {:?} {:?}:", info.name, info.literal);
                eprintln!("  Expected: {:?}", expected);
                eprintln!("  Got:      {:?}", result);
            }
        }

        assert!(!any_error);
    }

    // TODO turn this into a test case that checks whether the grammer is up-to-date
    #[test]
    fn print_grammer_enum() {
        for (_build, name) in CUSTOM_TOKENS {
            println!("Token{name} => TokenType::{name}(<&'s str>),")
        }
        for info in FIXED_TOKENS {
            println!("{:?} => TokenType::{},", info.literal, info.name)
        }
    }
}
