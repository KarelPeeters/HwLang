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
    fixed_tokens: &'static [(&'static str, fn(&str) -> TokenType<&str>, &'static str)],
}

impl<'s> Tokenizer<'s> {
    pub fn new(file: FileId, source: &'s str) -> Self {
        Tokenizer {
            file,
            curr_byte: 0,
            left: source.chars(),
            errored: false,
            fixed_tokens: &FIXED_TOKENS,
        }
    }

    pub fn into_iter(self) -> TokenizerIterator<'s> {
        TokenizerIterator { tokenizer: self }
    }

    fn peek(&self) -> Option<char> {
        self.left.clone().next()
    }

    fn skip_until(&mut self, f: impl Fn(char) -> bool) {
        while let Some(c) = self.peek() {
            if f(c) {
                break;
            }
            self.pop();
        }
    }

    fn peek_second(&self) -> Option<char> {
        let mut iter = self.left.clone();
        iter.next()?;
        iter.next()
    }

    fn pop(&mut self) -> Option<char> {
        self.curr_byte += 1;
        self.left.next()
    }

    fn skip_chars(&mut self, n: usize) {
        for _ in 0..n {
            match self.left.next() {
                None => panic!(),
                Some(c) => self.curr_byte += c.len_utf8(),
            }
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

    // fn parse_start_continue(
    //     &mut self,
    //     is_start: impl Fn(char) -> bool,
    //     is_continue: impl Fn(char) -> bool,
    //     build: impl Fn(&'s str) -> TokenType<&'s str>,
    // ) -> Option<Token<&'s str>> {
    //     let start = self.curr_pos();
    //
    //     let mut chars = self.left.chars();
    //     if let Some(first) = chars.next() {
    //         if is_start(first) {
    //             let mut len = 0;
    //             len += first.len_utf8();
    //
    //             while let Some(c) = chars.next() {
    //                 if is_continue(c) {
    //                     len += c.len_utf8();
    //                 } else {
    //                     break;
    //                 }
    //             }
    //
    //             self.skip(len);
    //             let span = Span::new(start, self.curr_pos());
    //
    //             return Some(Token {
    //                 span,
    //                 ty: build(&left_start[..len]),
    //             });
    //         }
    //     }
    //
    //     None
    // }

    // TODO try reordering to maximize performance
    // TODO try generating a full character-based state machine at compile-time, that might be faster
    // TODO try memchr where it applies, see if it's actually faster
    // TODO pop first 2/3 chars immediately, then match on the entire thing as a u32?
    fn next_inner(&mut self) -> Result<Option<Token<&'s str>>, TokenError> {
        let start = self.curr_pos();
        let start_left_str = self.left.as_str();

        let ty = match self.peek() {
            None => return Ok(None),
            Some('=') => {
                self.pop();
                match self.peek() {
                    Some('=') => {
                        self.pop();
                        TokenType::EqEq
                    }
                    _ => TokenType::Eq,
                }
            }
            Some('.') => {
                self.pop();
                match self.peek() {
                    Some('.') => {
                        self.pop();
                        match self.peek() {
                            Some('=') => {
                                self.pop();
                                TokenType::DotsEq
                            }
                            Some('+') => {
                                self.pop();
                                TokenType::DotsPlus
                            }
                            _ => TokenType::Dots,
                        }
                    }
                    _ => TokenType::Dot,
                }
            }

            // TODO all of this would be a lot more compact if we just always popped and then went back
            Some('(') => {
                self.pop();
                TokenType::OpenR
            }
            Some(')') => {
                self.pop();
                TokenType::CloseR
            }
            Some('{') => {
                self.pop();
                TokenType::OpenC
            }
            Some('}') => {
                self.pop();
                TokenType::CloseC
            }
            Some('[') => {
                self.pop();
                TokenType::OpenS
            }
            Some(']') => {
                self.pop();
                TokenType::CloseS
            }
            Some(',') => {
                self.pop();
                TokenType::Comma
            }
            Some(';') => {
                self.pop();
                TokenType::Semi
            }
            Some(':') => {
                self.pop();
                match self.peek() {
                    Some(':') => {
                        self.pop();
                        TokenType::ColonColon
                    }
                    _ => TokenType::Colon,
                }
            }
            Some('!') => {
                self.pop();
                match self.peek() {
                    Some('=') => {
                        self.pop();
                        TokenType::Neq
                    }
                    _ => TokenType::Bang,
                }
            }
            Some('-') => {
                self.pop();
                match self.peek() {
                    Some('=') => {
                        self.pop();
                        TokenType::MinusEq
                    }
                    Some('>') => {
                        self.pop();
                        TokenType::Arrow
                    }
                    _ => TokenType::Minus,
                }
            }
            Some('*') => {
                self.pop();
                match self.peek() {
                    Some('=') => {
                        self.pop();
                        TokenType::StarEq
                    }
                    Some('*') => {
                        self.pop();
                        TokenType::StarStar
                    }
                    _ => TokenType::Star,
                }
            }

            Some('/') => {
                self.pop();
                match self.peek() {
                    Some('*') => todo!("block comment"),
                    Some('/') => {
                        self.pop();
                        self.skip_until(|c| c == '\n' || c == '\r');
                        TokenType::LineComment(&start_left_str[..self.curr_byte - start.byte])
                    }
                    Some('=') => {
                        self.pop();
                        TokenType::SlashEq
                    }
                    _ => TokenType::Slash,
                }
            }
            Some('"') => {
                self.pop();
                self.skip_until(|c| c == '"');
                match self.peek() {
                    Some('"') => {
                        self.pop();
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

            // TODO replace with pattern, might be faster
            Some(c) if is_whitespace(c) => {
                self.pop();
                while self.peek().map_or(false, is_whitespace) {
                    self.pop();
                }
                TokenType::WhiteSpace(&start_left_str[..self.curr_byte - start.byte])
            }
            // TODO replace with pattern, might be faster
            Some(c) if is_id_start(c) => {
                self.pop();
                while self.peek().map_or(false, is_id_continue) {
                    self.pop();
                }
                let id = &start_left_str[..self.curr_byte - start.byte];

                // TODO speed up with runtime? literal tree
                self.fixed_tokens
                    .iter()
                    .find_map(|(fixed, build, _)| if &id == fixed { Some(build(id)) } else { None })
                    .unwrap_or(TokenType::Identifier(id))
            }
            Some(c) if is_decimal_digit(c) => {
                self.pop();
                self.skip_until(|c| !is_decimal_digit(c));
                TokenType::IntLiteral(&start_left_str[..self.curr_byte - start.byte])
            }

            Some(_) => {
                return Err(TokenError::InvalidToken {
                    pos: start,
                    prefix: self.prefix_for_error_message(),
                })
            }
        };

        let span = Span::new(start, self.curr_pos());
        Ok(Some(Token { span, ty }))

        // // block comment
        // if self.left.starts_with("/*") {
        //     self.skip(2);
        //
        //     // block comments are allowed to nest
        //     let mut depth: usize = 1;
        //     while depth > 0 {
        //         if self.left.starts_with("/*") {
        //             depth += 1;
        //             self.skip(2);
        //         } else if self.left.starts_with("*/") {
        //             depth -= 1;
        //             self.skip(2);
        //         } else if let Some(c) = self.left.chars().next() {
        //             self.skip(c.len_utf8())
        //         } else {
        //             // hit end of source
        //             return Err(TokenError::BlockCommentMissingEnd {
        //                 start,
        //                 eof: self.curr_pos(),
        //             });
        //         }
        //     }
        //
        //     let span = Span::new(start, self.curr_pos());
        //     return Ok(Token {
        //         ty: TokenType::BlockComment(&left_start[..span.len_bytes()]),
        //         span,
        //     });
        // }
        // if self.left.starts_with("*/") {
        //     return Err(TokenError::BlockCommentUnexpectedEnd {
        //         pos: start,
        //         prefix: self.prefix_for_error_message(),
        //     });
        // }
        //
        // // line comment
        // // TODO should it include the trailing newline? \n\r handling becomes a bit tricky then
        // if self.left.starts_with("//") {
        //     let len = memchr::memchr2(b'\n', b'\r', self.left.as_bytes()).map_or(self.left.len(), |end| end);
        //     self.skip(len);
        //
        //     let span = Span::new(start, self.curr_pos());
        //     return Ok(Token {
        //         ty: TokenType::LineComment(&left_start[..len]),
        //         span,
        //     });
        // }
        //
        // let is_whitespace = |c| matches!(c, ' ' | '\t' | '\n' | '\r');
        // if let Some(token) = self.parse_start_continue(is_whitespace, is_whitespace, TokenType::WhiteSpace) {
        //     return Ok(token);
        // }
        //
        // let is_id_start = |c| matches!(c, '_' | 'a'..='z' | 'A'..='Z');
        // let is_id_continue = |c| matches!(c, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9');
        // let build_id = move |id| {
        //     check_id_fixed_token(fixed_tokens, id)
        // };
        // if let Some(token) = self.parse_start_continue(is_id_start, is_id_continue, build_id) {
        //     return Ok(token);
        // }
        //
        // // string literal
        // // TODO escape codes
        // // TODO f-strings, also needs parser
        // // TODO we're actually already parsing here, it would be better if we could pass the inner string to the parser
        // if self.left.starts_with('"') {
        //     let end_pos = memchr::memchr(b'"', &self.left.as_bytes()[1..]);
        //     return match end_pos {
        //         None => {
        //             self.skip(self.left.len());
        //             Err(TokenError::StringLiteralMissingEnd {
        //                 start,
        //                 eof: self.curr_pos(),
        //             })
        //         }
        //         Some(end_pos) => {
        //             let len_bytes = 1 + end_pos + 1;
        //             self.skip(len_bytes);
        //             let span = Span::new(start, self.curr_pos());
        //             Ok(Token {
        //                 ty: TokenType::StringLiteral(&left_start[..len_bytes]),
        //                 span,
        //             })
        //         }
        //     };
        // }
        //
        // // integer literal/pattern
        // // TODO parse hex/bin
        // // TODO store the actually parsed int in the token to get more type safety
        // let is_decimal_digit = |c| matches!(c, '0'..='9');
        // if let Some(token) = self.parse_start_continue(is_decimal_digit, is_decimal_digit, TokenType::IntLiteral) {
        //     return Ok(token);
        // }
        //
        // // fixed token (needs to be after identifiers to ensure ids that start with a fixed prefix get matched as ids)
        // if let Some((len, token)) = check_start_fixed_token(fixed_tokens, self.left) {
        //     self.skip(len);
        //     let span = Span::new(start, self.curr_pos());
        //     return Ok(Token {
        //         ty: token,
        //         span,
        //     });
        // }
        //
        // // failed to match anything
        // Err(TokenError::InvalidToken {
        //     pos: self.curr_pos(),
        //     prefix: self.prefix_for_error_message(),
        // })
    }

    fn next(&mut self) -> Result<Option<Token<&'s str>>, TokenError> {
        assert!(
            !self.errored,
            "Cannot continue calling next on tokenizer that returned an error"
        );
        self.next_inner().inspect_err(|_| self.errored = true)
    }
}

fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

fn is_id_start(c: char) -> bool {
    matches!(c, '_' | 'a'..='z' | 'A'..='Z')
}

fn is_id_continue(c: char) -> bool {
    matches!(c, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9')
}

fn is_decimal_digit(c: char) -> bool {
    matches!(c, '0'..='9')
}

#[inline(never)]
fn check_id_fixed_token<'s>(
    fixed_tokens: &[(&str, fn(&str) -> TokenType<&str>, &str)],
    id: &'s str,
) -> TokenType<&'s str> {
    // check for fixed matches, they might overlap with IDs
    // TODO create a separate sublist of fixed tokens that also match identifiers to speed this up a bit more
    for (fixed, build, _) in fixed_tokens {
        if &id == fixed {
            return build(id);
        }
    }
    TokenType::Identifier(id)
}

#[inline(never)]
fn check_start_fixed_token<'s>(
    fixed_tokens: &[(&str, fn(&str) -> TokenType<&str>, &str)],
    start: &'s str,
) -> Option<(usize, TokenType<&'s str>)> {
    // check for fixed matches, they might overlap with IDs
    // TODO create a separate sublist of fixed tokens that also match identifiers to speed this up a bit more
    for (fixed, build, _) in fixed_tokens {
        if start.starts_with(fixed) {
            return Some((fixed.len(), build(&start[..fixed.len()])));
        }
    }
    None
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
