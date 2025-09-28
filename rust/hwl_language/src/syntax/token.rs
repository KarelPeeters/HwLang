use crate::front::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::FileId;
use crate::util::iter::IterExt;
use itertools::Itertools;
use lazy_static::lazy_static;
use std::borrow::Cow;
use strum::EnumIter;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Token {
    pub ty: TokenType,
    pub span: Span,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TokenError {
    InvalidToken { pos: Pos },
    InvalidIntLiteral { span: Span },
    BlockCommentMissingEnd { start: Pos, eof: Pos },
    StringLiteralMissingEnd { start: Pos, eof: Pos },
}

pub fn tokenize(file: FileId, source: &str, emit_incomplete_token: bool) -> Result<Vec<Token>, TokenError> {
    Tokenizer::new(file, source, emit_incomplete_token)
        .into_iter()
        .try_collect()
}

// TODO remove string here? only deal with offset, simplifying the lifetime
pub struct Tokenizer<'s> {
    // happy path state
    curr_byte: usize,
    left: std::str::Chars<'s>,
    mode_stack: Vec<(Pos, Mode)>,
    mode: Mode,

    // error path state
    incomplete_err: Option<TokenError>,
    errored: bool,

    // fixed settings
    file: FileId,
    emit_incomplete_token: bool,
    fixed_tokens_grouped_by_length: &'static [Vec<FixedTokenInfo>],
}

/// https://peps.python.org/pep-0701/
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Mode {
    Normal,
    NormalInSubExpression { str_start: Pos },
    StringMiddle { str_start: Pos, subs: bool },
}

#[rustfmt::skip]
macro_rules! pattern_whitespace { () => { ' ' | '\t' | '\n' | '\r' }; }
#[rustfmt::skip]
macro_rules! pattern_id_start { () => { '_' | 'a'..='z' | 'A'..='Z' }; }
#[rustfmt::skip]
macro_rules! pattern_id_continue { () => { '_' | 'a'..='z' | 'A'..='Z' | '0'..='9' }; }
#[rustfmt::skip]
macro_rules! pattern_decimal_digit { () => { '0'..='9' }; }

#[derive(Debug, Copy, Clone)]
enum NextInnerResult {
    Ty(TokenType),
    Whitespace,
    Eof,
}

impl<'s> Tokenizer<'s> {
    // TODO is there ever really a reason to set emit_incomplete_token to false?
    pub fn new(file: FileId, source: &'s str, emit_incomplete_token: bool) -> Self {
        Tokenizer {
            file,
            curr_byte: 0,
            left: source.chars(),
            mode_stack: vec![],
            mode: Mode::Normal,
            emit_incomplete_token,
            incomplete_err: None,
            errored: false,
            fixed_tokens_grouped_by_length: &FIXED_TOKENS_GROUPED_BY_LENGTH,
        }
    }

    fn peek(&self) -> Option<char> {
        self.peek_n(0)
    }

    fn peek_n(&self, n: usize) -> Option<char> {
        let mut iter = self.left.clone();
        for _ in 0..n {
            iter.next()?;
        }
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

    fn skip_while(&mut self, f: impl Fn(char) -> bool) {
        while let Some(c) = self.peek() {
            if !f(c) {
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

    // TODO automatically generate most of this match logic
    // TODO try memchr where it applies, see if it's actually faster
    fn next_inner_ty(&mut self) -> Result<NextInnerResult, TokenError> {
        let start = self.curr_pos();
        let start_left_str = self.left.as_str();

        // handle string modes
        match self.mode {
            Mode::Normal => {
                // fallthrough
            }
            Mode::NormalInSubExpression { str_start } => {
                if self.peek() == Some('}') {
                    self.skip(1);
                    self.mode = Mode::StringMiddle { str_start, subs: true };
                    return Ok(NextInnerResult::Ty(TokenType::StringSubEnd));
                } else {
                    // fallthrough
                }
            }
            Mode::StringMiddle {
                str_start,
                subs: allow_subs,
            } => {
                let (ty, next_mode) = match self.peek() {
                    Some('"') => {
                        self.skip(1);
                        let (_, next_mode) = self.mode_stack.pop().unwrap();
                        (TokenType::StringEnd, next_mode)
                    }
                    // TODO handle escapes (ie. {{)
                    Some('{') if allow_subs => {
                        self.skip(1);
                        (TokenType::StringSubStart, Mode::NormalInSubExpression { str_start })
                    }
                    Some(_) => {
                        // TODO handle escapes (eg. \n, \u, {{, \", ...)
                        self.skip_while(|c| !(c == '"' || (allow_subs && c == '{')));
                        (
                            TokenType::StringMiddle,
                            Mode::StringMiddle {
                                str_start,
                                subs: allow_subs,
                            },
                        )
                    }
                    None => {
                        return Err(TokenError::StringLiteralMissingEnd {
                            start: str_start,
                            eof: self.curr_pos(),
                        });
                    }
                };

                self.mode = next_mode;
                return Ok(NextInnerResult::Ty(ty));
            }
        };

        // get the first couple of chars
        let peek = {
            let mut iter = self.left.clone();
            let first = match iter.next() {
                Some(first) => first,
                None => {
                    // end of file, valid iff we're not still in a string literal
                    return if let Some((start, _)) = self.mode_stack.last() {
                        Err(TokenError::StringLiteralMissingEnd {
                            start: *start,
                            eof: self.curr_pos(),
                        })
                    } else {
                        Ok(NextInnerResult::Eof)
                    };
                }
            };

            [first, iter.next().unwrap_or('\0'), iter.next().unwrap_or('\0')]
        };
        let mut skip_fixed = |n: usize, ty: TokenType| {
            self.skip(n);
            ty
        };

        let ty = match peek {
            // custom
            [pattern_whitespace!(), _, _] => {
                self.skip(1);
                self.skip_while(|c| matches!(c, pattern_whitespace!()));
                return Ok(NextInnerResult::Whitespace);
            }

            ['"', _, _] => {
                let start = self.curr_pos();
                self.skip(1);
                self.mode_stack.push((start, self.mode));
                self.mode = Mode::StringMiddle {
                    str_start: start,
                    subs: true,
                };
                TokenType::StringStart
            }
            ['r', '"', _] => {
                self.skip(2);
                self.mode_stack.push((start, self.mode));
                self.mode = Mode::StringMiddle {
                    str_start: start,
                    subs: false,
                };
                TokenType::StringStart
            }

            [pattern_id_start!(), _, _] => {
                self.skip(1);
                self.skip_while(|c| matches!(c, pattern_id_continue!()));
                let id = &start_left_str[..self.curr_byte - start.byte];

                // TODO speed up with runtime? literal tree
                match self.fixed_tokens_grouped_by_length.get(id.len()) {
                    None => TokenType::Identifier,
                    Some(potential) => potential
                        .iter()
                        .find_map(|info| if info.literal == id { Some(info.ty) } else { None })
                        .unwrap_or(TokenType::Identifier),
                }
            }

            [pattern_decimal_digit!(), _, _] => {
                self.skip(1);
                #[allow(clippy::manual_is_ascii_check)]
                self.skip_while(|c| matches!(c, pattern_decimal_digit!() | '_' | 'b' | 'x' | 'a'..='f'));

                let token_str = &start_left_str[..self.curr_byte - start.byte];
                let invalid = || {
                    Err(TokenError::InvalidIntLiteral {
                        span: Span::new(self.file, start.byte, self.curr_byte),
                    })
                };

                let f_not_dummy = |c| c != '_';
                match peek {
                    ['0', 'b', _] => {
                        let tail = &token_str[2..];
                        let f = |c| matches!(c, '_' | '0' | '1');
                        if !tail.chars().any(f_not_dummy) || !tail.chars().all(f) {
                            return invalid();
                        }

                        TokenType::IntLiteralBinary
                    }
                    ['0', 'x', _] => {
                        let tail = &token_str[2..];
                        let f = |c| matches!(c, pattern_decimal_digit!() | '_' | 'a'..='f');
                        if !tail.chars().any(f_not_dummy) || !tail.chars().all(f) {
                            return invalid();
                        }

                        TokenType::IntLiteralHexadecimal
                    }
                    _ => {
                        let f = |c| matches!(c, pattern_decimal_digit!() | '_');
                        if !token_str.chars().any(f_not_dummy) || !token_str.chars().all(f) {
                            return invalid();
                        }
                        TokenType::IntLiteralDecimal
                    }
                }
            }

            ['/', '*', _] => {
                // block comments are allowed to nest
                self.skip(2);
                let mut depth: usize = 1;
                while depth > 0 {
                    match (self.peek(), self.peek_n(1)) {
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
                            self.skip_while(|_| true);
                            self.incomplete_err = Some(TokenError::BlockCommentMissingEnd {
                                start,
                                eof: self.curr_pos(),
                            });
                            break;
                        }
                    }
                }

                // end of comment, depth is 0 again
                TokenType::BlockComment
            }
            ['/', '/', _] => {
                self.skip(2);
                self.skip_while(|c| c != '\n' && c != '\r');
                TokenType::LineComment
            }

            // trigram
            ['.', '.', '='] => skip_fixed(3, TokenType::DotDotEq),
            ['+', '.', '.'] => skip_fixed(3, TokenType::PlusDotDot),
            ['.', '.', _] => skip_fixed(2, TokenType::DotDot),

            // simple fixed
            ['=', '=', _] => skip_fixed(2, TokenType::EqEq),
            ['=', '>', _] => skip_fixed(2, TokenType::DoubleArrow),
            ['=', _, _] => skip_fixed(1, TokenType::Eq),
            ['!', '=', _] => skip_fixed(2, TokenType::Neq),
            ['%', '=', _] => skip_fixed(2, TokenType::PercentEq),
            ['%', _, _] => skip_fixed(1, TokenType::Percent),
            ['!', _, _] => skip_fixed(1, TokenType::Bang),
            ['>', '=', _] => skip_fixed(2, TokenType::Gte),
            ['>', '>', _] => skip_fixed(2, TokenType::GtGt),
            ['>', _, _] => skip_fixed(1, TokenType::Gt),
            ['<', '=', _] => skip_fixed(2, TokenType::Lte),
            ['<', '<', _] => skip_fixed(2, TokenType::LtLt),
            ['<', _, _] => skip_fixed(1, TokenType::Lt),
            ['&', '&', _] => skip_fixed(2, TokenType::AmperAmper),
            ['&', '=', _] => skip_fixed(2, TokenType::AmperEq),
            ['&', _, _] => skip_fixed(1, TokenType::Amper),
            ['|', '|', _] => skip_fixed(2, TokenType::PipePipe),
            ['|', '=', _] => skip_fixed(2, TokenType::PipeEq),
            ['|', _, _] => skip_fixed(1, TokenType::Pipe),
            ['^', '^', _] => skip_fixed(2, TokenType::CaretCaret),
            ['^', '=', _] => skip_fixed(2, TokenType::CaretEq),
            ['^', _, _] => skip_fixed(1, TokenType::Caret),
            ['+', '=', _] => skip_fixed(2, TokenType::PlusEq),
            ['+', _, _] => skip_fixed(1, TokenType::Plus),
            ['-', '=', _] => skip_fixed(2, TokenType::MinusEq),
            ['-', '>', _] => skip_fixed(2, TokenType::Arrow),
            ['-', _, _] => skip_fixed(1, TokenType::Minus),
            ['*', '=', _] => skip_fixed(2, TokenType::StarEq),
            ['*', '*', _] => skip_fixed(2, TokenType::StarStar),
            ['*', _, _] => skip_fixed(1, TokenType::Star),
            ['/', '=', _] => skip_fixed(2, TokenType::SlashEq),
            ['/', _, _] => skip_fixed(1, TokenType::Slash),
            ['.', _, _] => skip_fixed(1, TokenType::Dot),
            [';', _, _] => skip_fixed(1, TokenType::Semi),
            [':', ':', _] => skip_fixed(2, TokenType::ColonColon),
            [':', _, _] => skip_fixed(1, TokenType::Colon),
            [',', _, _] => skip_fixed(1, TokenType::Comma),
            ['(', _, _] => skip_fixed(1, TokenType::OpenR),
            [')', _, _] => skip_fixed(1, TokenType::CloseR),
            ['{', _, _] => skip_fixed(1, TokenType::OpenC),
            ['}', _, _] => skip_fixed(1, TokenType::CloseC),
            ['[', _, _] => skip_fixed(1, TokenType::OpenS),
            [']', _, _] => skip_fixed(1, TokenType::CloseS),

            _ => return Err(TokenError::InvalidToken { pos: start }),
        };
        Ok(NextInnerResult::Ty(ty))
    }

    fn next_inner(&mut self) -> Result<Option<Token>, TokenError> {
        // TODO try moving this whitespace loop into the inner function, maybe that's faster
        let (start_byte, ty) = loop {
            let start_byte = self.curr_byte;
            match self.next_inner_ty()? {
                NextInnerResult::Ty(ty) => break (start_byte, ty),
                NextInnerResult::Whitespace => continue,
                NextInnerResult::Eof => return Ok(None),
            };
        };

        let span = Span::new(self.file, start_byte, self.curr_byte);
        Ok(Some(Token { span, ty }))
    }

    fn next(&mut self) -> Result<Option<Token>, TokenError> {
        assert!(
            !self.errored,
            "Cannot continue calling next on tokenizer that returned an error"
        );

        let result = if self.emit_incomplete_token {
            if let Some(err) = self.incomplete_err.take() {
                Err(err)
            } else {
                self.next_inner()
            }
        } else {
            assert!(self.incomplete_err.is_none());
            let result = self.next_inner();
            if let Some(err) = self.incomplete_err.take() {
                Err(err)
            } else {
                result
            }
        };

        result.inspect_err(|_| self.errored = true)
    }
}

pub fn apply_string_literal_escapes(raw: &str) -> Cow<'_, str> {
    // TODO actually do escapes here, keep in sync with tokenizer state machine
    Cow::Borrowed(raw)
}

fn str_is_single_token(s: &str, ty: TokenType) -> bool {
    let tokenizer = Tokenizer::new(FileId::dummy(), s, false);
    match tokenizer.into_iter().single() {
        Some(Ok(token)) => token.ty == ty,
        Some(Err(_)) | None => false,
    }
}

pub fn str_is_valid_identifier(s: &str) -> bool {
    str_is_single_token(s, TokenType::Identifier)
}

pub fn str_is_whitespace_or_empty(s: &str) -> bool {
    s.chars().all(|c| matches!(c, pattern_whitespace!()))
}

pub fn char_is_whitespace(c: char) -> bool {
    matches!(c, pattern_whitespace!())
}

impl<'s> IntoIterator for Tokenizer<'s> {
    type Item = Result<Token, TokenError>;
    type IntoIter = TokenizerIterator<'s>;

    fn into_iter(self) -> Self::IntoIter {
        TokenizerIterator {
            any_err: false,
            tokenizer: self,
        }
    }
}

pub struct TokenizerIterator<'s> {
    any_err: bool,
    tokenizer: Tokenizer<'s>,
}

impl<'s> Iterator for TokenizerIterator<'s> {
    type Item = Result<Token, TokenError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.any_err {
            return None;
        }

        let next = self.tokenizer.next();
        self.any_err |= next.is_err();
        next.transpose()
    }
}

macro_rules! declare_tokens {
    (
        custom {
            $($c_token:ident($c_cat:expr),)*
        }
        fixed {
            $($f_token:ident($f_string:expr, $f_cat:expr),)*
        }
    ) => {
        use TokenCategory as TC;

        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TokenType {
            $($c_token,)*
            $($f_token,)*
        }

        impl TokenType {
            #[cfg(test)]
            const CUSTOM_TOKENS: &[(&str, TokenType)] = &[
                $((stringify!($c_token), TokenType::$c_token),)*
            ];
            const FIXED_TOKENS: &[FixedTokenInfo] = &[
                $(FixedTokenInfo { name: stringify!($f_token), literal: $f_string, ty: TokenType::$f_token },)*
            ];

            pub fn category(self) -> TokenCategory {
                match self {
                    $(TokenType::$c_token => $c_cat,)*
                    $(TokenType::$f_token => $f_cat,)*
                }
            }

            pub fn diagnostic_string(&self) -> &str {
                match self {
                    $(TokenType::$c_token => stringify!($c_token),)*
                    $(TokenType::$f_token => $f_string,)*
                }
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, EnumIter, strum::Display)]
pub enum TokenCategory {
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

// these have dedicated constants so we can refer to them in diagnostics
pub const TOKEN_STR_BUILTIN: &str = "__builtin";
pub const TOKEN_STR_UNSAFE_VALUE_WITH_DOMAIN: &str = "unsafe_value_with_domain";

// TODO rename tokens to match the literal string better
declare_tokens! {
    custom {
        BlockComment(TC::Comment),
        LineComment(TC::Comment),

        Identifier(TC::Identifier),
        IntLiteralBinary(TC::IntegerLiteral),
        IntLiteralDecimal(TC::IntegerLiteral),
        IntLiteralHexadecimal(TC::IntegerLiteral),

        StringStart(TC::StringLiteral),
        StringEnd(TC::StringLiteral),
        StringSubStart(TC::StringLiteral),
        StringSubEnd(TC::StringLiteral),
        StringMiddle(TC::StringLiteral),
    }
    fixed {
        // keywords
        Import("import", TC::Keyword),
        Type("type", TC::Keyword),
        Struct("struct", TC::Keyword),
        Enum("enum", TC::Keyword),
        Ports("ports", TC::Keyword),
        Module("module", TC::Keyword),
        Interface("interface", TC::Keyword),
        Instance("instance", TC::Keyword),
        Fn("fn", TC::Keyword),
        Comb("comb", TC::Keyword),
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
        Undef("undef", TC::IntegerLiteral),
        If("if", TC::Keyword),
        Else("else", TC::Keyword),
        Loop("loop", TC::Keyword),
        Match("match", TC::Keyword),
        For("for", TC::Keyword),
        While("while", TC::Keyword),
        Pub("pub", TC::Keyword),
        As("as", TC::Keyword),
        External("external", TC::Keyword),

        // builtins
        // TODO separate category?
        Builtin(TOKEN_STR_BUILTIN, TC::Keyword),
        UnsafeValueWithDomain(TOKEN_STR_UNSAFE_VALUE_WITH_DOMAIN, TC::Keyword),
        IdFromStr("id_from_str", TC::Keyword),

        // misc symbols
        Semi(";", TC::Symbol),
        Colon(":", TC::Symbol),
        Comma(",", TC::Symbol),
        Arrow("->", TC::Symbol),
        DoubleArrow("=>", TC::Symbol),
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
        DotDot("..", TC::Symbol),
        DotDotEq("..=", TC::Symbol),
        PlusDotDot("+..", TC::Symbol),
        AmperAmper("&&", TC::Symbol),
        PipePipe("||", TC::Symbol),
        CaretCaret("^^", TC::Symbol),
        EqEq("==", TC::Symbol),
        Neq("!=", TC::Symbol),
        Gte(">=", TC::Symbol),
        Gt(">", TC::Symbol),
        Lte("<=", TC::Symbol),
        Lt("<", TC::Symbol),
        Amper("&", TC::Symbol),
        Pipe("|", TC::Symbol),
        Caret("^", TC::Symbol),
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
        PipeEq("|=", TC::Symbol),
        CaretEq("^=", TC::Symbol),
    }
}

#[derive(Debug, Copy, Clone)]
struct FixedTokenInfo {
    #[allow(dead_code)]
    name: &'static str,
    literal: &'static str,
    ty: TokenType,
}

lazy_static! {
    static ref FIXED_TOKENS_GROUPED_BY_LENGTH: Vec<Vec<FixedTokenInfo>> = {
        let mut result = vec![];

        for &token in TokenType::FIXED_TOKENS {
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
            TokenError::InvalidToken { pos } => Diagnostic::new("tokenization error")
                .add_error(Span::empty_at(pos), "invalid start of token")
                .finish(),
            TokenError::InvalidIntLiteral { span } => Diagnostic::new("invalid integer literal")
                .add_error(span, "here")
                .finish(),
            TokenError::BlockCommentMissingEnd { start, eof } => Diagnostic::new("block comment missing end")
                .add_info(Span::empty_at(start), "block comment started here")
                .add_error(Span::empty_at(eof), "end of file reached")
                .finish(),
            TokenError::StringLiteralMissingEnd { start, eof } => Diagnostic::new("string literal missing end")
                .add_info(Span::empty_at(start), "string literal started here")
                .add_error(Span::empty_at(eof), "end of file reached")
                .finish(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::syntax::pos::{Pos, Span};
    use crate::syntax::source::FileId;
    use crate::syntax::token::{Token, TokenError, TokenType, Tokenizer, tokenize};
    use hwl_util::swriteln;
    use std::collections::HashSet;

    fn tokenize_iter(file: FileId, source: &str, emit_incomplete_token: bool) -> Vec<Result<Token, TokenError>> {
        let mut result = vec![];
        for token in Tokenizer::new(file, source, emit_incomplete_token) {
            match token {
                Ok(token) => result.push(Ok(token)),
                Err(err) => {
                    result.push(Err(err));
                    break;
                }
            }
        }
        result
    }

    #[test]
    fn basic_tokenize() {
        let file = FileId::dummy();
        assert_eq!(Ok(vec![]), tokenize(file, "", false));
        assert_eq!(Ok(vec![]), tokenize(file, "\n", false));
        assert!(tokenize(file, "test foo function \"foo\"", false).is_ok());
    }

    #[test]
    fn count_whitespace() {
        let file = FileId::dummy();
        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::Identifier,
                span: Span::new(file, 1, 1 + 5)
            }]),
            tokenize(file, " hello", false)
        );
    }

    #[test]
    fn comment() {
        let file = FileId::dummy();

        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::BlockComment,
                span: Span::new(file, 0, 4)
            }]),
            tokenize(file, "/**/", false)
        );

        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::BlockComment,
                span: Span::new(file, 0, 8)
            }]),
            tokenize(file, "/*/**/*/", false)
        );

        assert!(tokenize(file, "/*/**/", false).is_err());
    }

    #[test]
    fn not_closed() {
        let file = FileId::dummy();

        // comment
        let expected = vec![Err(TokenError::BlockCommentMissingEnd {
            start: Pos { file, byte: 0 },
            eof: Pos { file, byte: 2 },
        })];
        let actual = tokenize_iter(file, "/*", false);
        assert_eq!(expected, actual);

        let expected = vec![
            Ok(Token {
                ty: TokenType::BlockComment,
                span: Span::new(file, 0, 2),
            }),
            Err(TokenError::BlockCommentMissingEnd {
                start: Pos { file, byte: 0 },
                eof: Pos { file, byte: 2 },
            }),
        ];
        let actual = tokenize_iter(file, "/*", true);
        assert_eq!(expected, actual);

        // string
        let expected = vec![
            Ok(Token {
                ty: TokenType::StringStart,
                span: Span::new(file, 0, 1),
            }),
            Err(TokenError::StringLiteralMissingEnd {
                start: Pos { file, byte: 0 },
                eof: Pos { file, byte: 1 },
            }),
        ];
        let actual = tokenize_iter(file, "\"", false);
        assert_eq!(expected, actual);

        let expected = vec![
            Ok(Token {
                ty: TokenType::StringStart,
                span: Span::new(file, 0, 1),
            }),
            Err(TokenError::StringLiteralMissingEnd {
                start: Pos { file, byte: 0 },
                eof: Pos { file, byte: 1 },
            }),
        ];
        let actual = tokenize_iter(file, "\"", true);
        assert_eq!(expected, actual);
    }

    #[test]
    fn fixed_tokens_unique() {
        let mut set = HashSet::new();
        for info in TokenType::FIXED_TOKENS {
            assert!(!info.literal.is_empty());
            assert!(set.insert(info.literal));
        }
    }

    #[test]
    fn fixed_tokens_covered() {
        let mut any_error = false;

        for info in TokenType::FIXED_TOKENS {
            let file = FileId::dummy();

            let result = tokenize(file, info.literal, false);
            let span = Span::new(file, 0, info.literal.len());
            let expected = Ok(vec![Token { ty: info.ty, span }]);

            if result != expected {
                any_error = true;
                eprintln!("Failed to parse literal token {:?} {:?}:", info.name, info.literal);
                eprintln!("  Expected: {expected:?}");
                eprintln!("  Got:      {result:?}");
            }
        }

        assert!(!any_error);
    }

    #[test]
    fn grammar_matches_tokens() {
        let expected = {
            const I: &str = "    ";
            let mut f = String::new();
            swriteln!(f, "{I}enum TokenType {{");
            for (name, _ty) in TokenType::CUSTOM_TOKENS {
                swriteln!(f, "{I}{I}Token{name} => TokenType::{name},");
            }
            for info in TokenType::FIXED_TOKENS {
                swriteln!(f, "{I}{I}{:?} => TokenType::{},", info.literal, info.name);
            }
            swriteln!(f, "{I}}}");
            f
        };

        println!("{expected}");

        let grammar = include_str!("grammar.lalrpop");
        assert!(grammar.contains(&expected));
    }
}
