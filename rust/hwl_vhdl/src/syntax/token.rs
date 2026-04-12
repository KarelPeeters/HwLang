use hwl_common::diagnostic::DiagnosticError;
use hwl_common::pos::{Pos, Span};
use hwl_common::source::FileId;
use itertools::Itertools;
use lazy_static::lazy_static;
use strum::EnumIter;

// TODO fix code duplication

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Token {
    pub ty: TokenType,
    pub span: Span,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TokenError {
    InvalidToken { pos: Pos },
    UnexpectedChar { start: Pos, pos: Pos, actual: char },

    IntLiteralInvalidBase { start: Pos, base: Span },

    UnexpectedEof { kind: &'static str, start: Pos, eof: Pos },
}

pub fn tokenize(file: FileId, source: &str, emit_incomplete_token: bool) -> Result<Vec<Token>, TokenError> {
    Tokenizer::new(file, source, emit_incomplete_token)
        .into_iter()
        .try_collect()
}

pub struct Tokenizer<'s> {
    // fixed settings
    file: FileId,
    emit_incomplete_token: bool,

    // happy path state
    curr_byte: usize,
    left: std::str::Chars<'s>,
    prev_token: Option<TokenType>,

    // error path state
    // TODO rework this
    incomplete_err: Option<TokenError>,
    errored: bool,
}

#[rustfmt::skip]
macro_rules! define_patterns { () => {
    // LRM 15.2 Character set
    macro_rules! pattern_basic_graphic_character {
        () => { pattern_upper_case_letter!() | pattern_digit!() | pattern_special_character!() | ' ' };
    }
    macro_rules! pattern_graphic_character {
        () => { pattern_basic_graphic_character!() | pattern_lower_case_letter!() | pattern_other_special_character!() }
    }
    #[allow(unused_macros)]
    macro_rules! pattern_basic_character {
        () => { pattern_basic_graphic_character!() | format_effectorpattern!() }
    }
    macro_rules! pattern_space { () => { ' ' | '\u{00A0}'} }
    macro_rules! pattern_format_effector { () => { '\t' | pattern_end_of_line!() } }
    macro_rules! pattern_end_of_line { () => { '\u{0B}' | '\r' | '\n' | '\u{0C}' } }

    // TODO complete, eg. accents, more special chars, ...
    macro_rules! pattern_upper_case_letter { () => { 'A'..='Z' } }
    macro_rules! pattern_lower_case_letter { () => { 'a'..='z' } }
    macro_rules! pattern_digit { () => { '0'..='9' } }
    macro_rules! pattern_special_character { () => {
        '"' | '&' | '\'' | '(' | ')' | '+' | ',' | '-' | '.' | '/' | ':' | ';'  | '<' | '=' | '>' | '?' | '@'|
        '[' | ']' | '_' | '`' | '|'
    } }
    macro_rules! pattern_other_special_character { () => {
        '!' | '$' | '%' | '\\' | '^' | '{' | '}' | '~'
    } }

    // LRM 15.4 Identifiers
    macro_rules! pattern_letter { () => { pattern_upper_case_letter!() | pattern_lower_case_letter!() } }
    macro_rules! pattern_letter_or_digit { () => { pattern_letter!() | pattern_digit!() } }

    // convenience
    macro_rules! pattern_whitespace { () => { pattern_space!() | pattern_format_effector!() } }

    macro_rules! pattern_base_specifier_single { () => { 'B' | 'O' | 'X' | 'D' | 'b' | 'o' | 'x'  | 'd' } }
    macro_rules! pattern_base_specifier_pair_0 { () => { 'U' | 'S' | 'u' | 's' } }
    macro_rules! pattern_base_specifier_pair_1 { () => { 'B' | 'O' | 'X' | 'b' | 'o' | 'x' } }
}}
define_patterns!();

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
            emit_incomplete_token,
            curr_byte: 0,
            left: source.chars(),
            prev_token: None,
            incomplete_err: None,
            errored: false,
        }
    }

    fn peek(&self) -> Option<char> {
        self.left.clone().next()
    }

    fn accept(&mut self, c: char) -> bool {
        self.accept_if(|p| p == c)
    }

    fn accept_if(&mut self, f: impl FnOnce(char) -> bool) -> bool {
        let found = self.peek().is_some_and(f);
        if found {
            self.skip(1);
        }
        found
    }

    fn peek_n<const N: usize>(&self) -> [Option<char>; N] {
        let mut iter = self.left.clone();
        let mut res = [None; N];
        for i in 0..N {
            res[i] = iter.next();
        }
        res
    }

    /// Must use, caller needs to check that this is None.
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

    fn skip_expect(&mut self, start: Pos, kind: &'static str, expected: char) -> Result<(), TokenError> {
        let pos = self.curr_pos();
        match self.pop() {
            Some(actual) => {
                if actual == expected {
                    Ok(())
                } else {
                    Err(TokenError::UnexpectedChar { start, pos, actual })
                }
            }
            None => Err(TokenError::UnexpectedEof { kind, start, eof: pos }),
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

    fn next_inner_ty(&mut self) -> Result<NextInnerResult, TokenError> {
        let start = self.curr_pos();
        let start_left_str = self.left.as_str();

        // get the first couple of chars
        if self.peek().is_none() {
            return Ok(NextInnerResult::Eof);
        }
        let peek = self.peek_n::<3>().map(|c| c.unwrap_or('\0'));

        // utility for common cases
        let mut skip_fixed = |n: usize, ty: TokenType| {
            self.skip(n);
            ty
        };

        let ty = match peek {
            // whitespace
            [pattern_whitespace!(), _, _] => {
                self.skip(1);
                self.skip_while(|c| matches!(c, pattern_whitespace!()));
                return Ok(NextInnerResult::Whitespace);
            }

            // LRM 15.5 Abstract literals
            // LRM 15.5.2 Decimal literals
            // LRM 15.5.3 Based literals
            // LRM 15.8 Bit string literals
            //
            // decimal_literal ::= integer [ . integer ] [ exponent ]
            // based_literal ::= integer # based_integer [ . based_integer ] # [ exponent ]
            // bit_string_literal ::= [ integer ] base_specifier " [ bit_value ] "
            //
            // integer ::= digit { [ underline ] digit }
            // based_integer ::= extended_digit { [ underline ] extended_digit }
            // bit_value ::= graphic_character { [ underline ] graphic_character }
            //
            // exponent ::= E [ + ] integer | E – integer
            // base_specifier ::= B | O | X | UB | UO | UX | SB | SO | SX | D
            [pattern_digit!(), _, _] => {
                // initial int
                self.skip_int(start, None)?;

                // check for bit string literal
                match self.peek_n::<3>() {
                    [Some(pattern_base_specifier_single!()), Some('"'), _] => {
                        self.skip(2);
                        self.skip_bit_string(start)?;
                        TokenType::BitStringLiteral
                    }
                    [
                        Some(pattern_base_specifier_pair_0!()),
                        Some(pattern_base_specifier_pair_1!()),
                        Some('"'),
                    ] => {
                        self.skip(3);
                        self.skip_bit_string(start)?;
                        TokenType::BitStringLiteral
                    }
                    _ => {
                        // check for based literal
                        let pos_before_pound = self.curr_pos();
                        let ty = if self.accept('#') {
                            let span_base = Span::new(start.file, start.byte, pos_before_pound.byte);
                            let base = start_left_str[..pos_before_pound.byte - start.byte]
                                .replace('_', "")
                                .parse::<u32>()
                                .map_err(|_| TokenError::IntLiteralInvalidBase { start, base: span_base })?;

                            if base == 0 || base > 16 {
                                return Err(TokenError::IntLiteralInvalidBase { start, base: span_base });
                            }

                            self.skip_int(start, Some(base))?;
                            self.skip_maybe_fractional(start, Some(base))?;
                            self.skip_expect(start, "based literal", '#')?;

                            TokenType::BasedLiteral
                        } else {
                            self.skip_maybe_fractional(start, None)?;

                            TokenType::DecimalLiteral
                        };

                        self.skip_maybe_exp(start)?;
                        ty
                    }
                }
            }
            [pattern_base_specifier_single!(), '"', _] => {
                self.skip(2);
                self.skip_bit_string(start)?;

                TokenType::BitStringLiteral
            }
            [pattern_base_specifier_pair_0!(), pattern_base_specifier_pair_1!(), '"'] => {
                self.skip(3);
                self.skip_bit_string(start)?;
                TokenType::BitStringLiteral
            }

            // LRM 15.4 Identifiers
            // LRM 15.4.2 Basic identifiers
            [pattern_letter!(), _, _] => {
                self.skip(1);
                self.skip_while(|c| matches!(c, '_' | pattern_letter_or_digit!()));

                // check reserved
                let id = &start_left_str[..self.curr_byte - start.byte];
                match TokenType::FIXED_TOKENS
                    .iter()
                    .find(|info| str::eq_ignore_ascii_case(id, info.literal))
                {
                    None => TokenType::Identifier,
                    Some(info) => info.ty,
                }
            }
            // LRM 15.4.3 Extended identifiers
            ['\\', _, _] => {
                self.skip(1);

                loop {
                    // LRM:
                    // > If a backslash is to be used as one of the graphic characters of an extended identifier,
                    // >   it shall be doubled.
                    match self.peek_n::<2>() {
                        [Some('\\'), Some('\\')] => {
                            self.skip(2);
                        }
                        [Some('\\'), _] => {
                            self.skip(1);
                            break;
                        }
                        [Some(c), _] => {
                            if !matches!(c, pattern_graphic_character!()) {
                                return Err(TokenError::UnexpectedChar {
                                    start,
                                    pos: self.curr_pos(),
                                    actual: c,
                                });
                            }

                            self.skip(1);
                            continue;
                        }
                        [None, _] => {
                            return Err(TokenError::UnexpectedEof {
                                kind: "extended identifier",
                                start,
                                eof: self.curr_pos(),
                            });
                        }
                    }
                }

                TokenType::Identifier
            }

            // LRM 15.6 Character literals
            // LRM 8.6 Attribute names
            ['\'', c1, c2] => {
                // break the ambiguity by following https://www.eda-twiki.org/isac/IRs-VHDL-93/IR1045.txt
                if matches!(
                    self.prev_token,
                    Some(TokenType::CloseS | TokenType::CloseR | TokenType::ResAll | TokenType::Identifier)
                ) || c2 != '\''
                {
                    // must be attribute
                    self.skip(1);
                    TokenType::AttributeQuote
                } else {
                    // must be char literal
                    // note: no escaping, ''' is a valid char literal
                    self.skip(1);
                    let pos_c1 = self.curr_pos();
                    self.skip(1);
                    let pos_c2 = self.curr_pos();
                    self.skip(1);

                    if !matches!(c1, pattern_graphic_character!()) {
                        return Err(TokenError::UnexpectedChar {
                            start,
                            pos: pos_c1,
                            actual: c1,
                        });
                    }
                    if c2 != '\'' {
                        return Err(TokenError::UnexpectedChar {
                            start,
                            pos: pos_c2,
                            actual: c2,
                        });
                    }

                    TokenType::CharacterLiteral
                }
            }

            // LRM 15.7 String literals
            ['"', _, _] => {
                // LRM:
                // > If a quotation mark value is to be represented in the sequence of character values,
                // >   then a pair of adjacent quotation marks shall be written at the corresponding
                // >   place within the string literal.
                self.skip(1);
                loop {
                    match self.peek_n::<2>() {
                        [Some('"'), Some('"')] => {
                            self.skip(2);
                            continue;
                        }
                        [Some('"'), _] => {
                            self.skip(1);
                            break;
                        }
                        [Some(_), _] => {
                            self.skip(1);
                            continue;
                        }
                        [None, _] => {
                            // hit end of source
                            self.skip_while(|_| true);
                            self.incomplete_err = Some(TokenError::UnexpectedEof {
                                kind: "string literal",
                                start,
                                eof: self.curr_pos(),
                            });
                            break;
                        }
                    }
                }

                TokenType::StringLiteral
            }

            // LRM 15.9 Comments
            ['-', '-', _] => {
                self.skip_while(|c| !matches!(c, pattern_end_of_line!()));
                return Ok(NextInnerResult::Ty(TokenType::SingleLineComment));
            }
            ['/', '*', _] => {
                // LRM:
                //   > [...] an occurrence of a solidus character immediately followed by an asterisk character
                //   >   within a delimited comment is not interpreted as the start of a nested delimited comment.
                // delimited comments don't nest, so just find the first ending sequence
                self.skip(2);
                loop {
                    match self.peek_n::<2>() {
                        [Some('*'), Some('/')] => {
                            self.skip(2);
                            break;
                        }
                        [Some(_), _] => {
                            self.skip(1);
                            continue;
                        }
                        [None, _] => {
                            // hit end of source
                            self.skip_while(|_| true);
                            self.incomplete_err = Some(TokenError::UnexpectedEof {
                                kind: "delimited comment",
                                start,
                                eof: self.curr_pos(),
                            });
                            break;
                        }
                    }
                }

                TokenType::DelimitedComment
            }

            // trigrams
            ['?', '/', '='] => skip_fixed(3, TokenType::QuestSlashEq),
            ['?', '>', '='] => skip_fixed(3, TokenType::QuestGtEq),
            ['?', '<', '='] => skip_fixed(3, TokenType::QuestLtEq),

            // bigrams
            ['=', '>', _] => skip_fixed(2, TokenType::Arrow),
            [':', '=', _] => skip_fixed(2, TokenType::ColonEq),
            ['/', '=', _] => skip_fixed(2, TokenType::SlashEq),
            ['>', '=', _] => skip_fixed(2, TokenType::GtEq),
            ['<', '=', _] => skip_fixed(2, TokenType::LtEq),
            ['?', '?', _] => skip_fixed(2, TokenType::QuestQuest),
            ['?', '=', _] => skip_fixed(2, TokenType::QuestEq),
            ['?', '<', _] => skip_fixed(2, TokenType::QuestLt),
            ['?', '>', _] => skip_fixed(2, TokenType::QuestGt),
            ['*', '*', _] => skip_fixed(2, TokenType::StarStar),

            // monograms
            ['(', _, _] => skip_fixed(1, TokenType::OpenR),
            [')', _, _] => skip_fixed(1, TokenType::CloseR),
            ['[', _, _] => skip_fixed(1, TokenType::OpenS),
            [']', _, _] => skip_fixed(1, TokenType::CloseS),
            [';', _, _] => skip_fixed(1, TokenType::Semi),
            [':', _, _] => skip_fixed(1, TokenType::Colon),
            ['.', _, _] => skip_fixed(1, TokenType::Dot),
            [',', _, _] => skip_fixed(1, TokenType::Comma),
            ['?', _, _] => skip_fixed(1, TokenType::Quest),
            ['@', _, _] => skip_fixed(1, TokenType::AtSign),
            ['^', _, _] => skip_fixed(1, TokenType::Caret),
            ['|', _, _] => skip_fixed(1, TokenType::Pipe),
            ['=', _, _] => skip_fixed(1, TokenType::Eq),
            ['<', _, _] => skip_fixed(1, TokenType::Lt),
            ['>', _, _] => skip_fixed(1, TokenType::Gt),
            ['+', _, _] => skip_fixed(1, TokenType::Plus),
            ['-', _, _] => skip_fixed(1, TokenType::Minus),
            ['&', _, _] => skip_fixed(1, TokenType::Amper),
            ['*', _, _] => skip_fixed(1, TokenType::Star),
            ['/', _, _] => skip_fixed(1, TokenType::Slash),

            _ => return Err(TokenError::InvalidToken { pos: start }),
        };

        Ok(NextInnerResult::Ty(ty))
    }

    fn skip_int(&mut self, start: Pos, base: Option<u32>) -> Result<(), TokenError> {
        let base_int = base.unwrap_or(10);

        // require at least one real digit
        let first_pos = self.curr_pos();
        match self.pop() {
            Some(actual) => {
                if !matches!(actual, pattern_letter_or_digit!()) || !actual.is_digit(base_int) {
                    return Err(TokenError::UnexpectedChar {
                        start,
                        pos: first_pos,
                        actual,
                    });
                }
            }
            None => {
                return Err(TokenError::UnexpectedEof {
                    kind: "integer",
                    start,
                    eof: first_pos,
                });
            }
        }

        // we can accept more digits or underlines
        while let Some(c) = self.peek() {
            match c {
                '_' => {
                    // accept
                }
                pattern_letter_or_digit!() => {
                    if !c.is_digit(base_int) {
                        if base.is_some() {
                            let pos = self.curr_pos();
                            return Err(TokenError::UnexpectedChar { start, pos, actual: c });
                        } else {
                            break;
                        }
                    }
                    // accept
                }
                _ => break,
            }

            self.skip(1);
        }

        Ok(())
    }

    fn skip_bit_string(&mut self, start: Pos) -> Result<(), TokenError> {
        // require at least one real character
        let pos = self.curr_pos();
        match self.pop() {
            Some(c) if c != '"' && c != '_' && matches!(c, pattern_graphic_character!()) => {
                // accept
            }
            Some(c) => return Err(TokenError::UnexpectedChar { start, pos, actual: c }),
            None => {
                return Err(TokenError::UnexpectedEof {
                    kind: "bit string",
                    start,
                    eof: pos,
                });
            }
        }

        // we can accept more characters or underlines
        #[allow(unreachable_patterns)]
        loop {
            match self.pop() {
                Some('"') => {
                    break;
                }
                Some('_' | pattern_graphic_character!()) => {
                    // accept
                }
                None => {
                    return Err(TokenError::UnexpectedEof {
                        kind: "bit string",
                        start,
                        eof: pos,
                    });
                }
                Some(c) => return Err(TokenError::UnexpectedChar { start, pos, actual: c }),
            }
        }

        Ok(())
    }

    fn skip_maybe_fractional(&mut self, start: Pos, base: Option<u32>) -> Result<(), TokenError> {
        if self.accept('.') {
            self.skip_int(start, base)?;
        }
        Ok(())
    }

    fn skip_maybe_exp(&mut self, start: Pos) -> Result<(), TokenError> {
        if self.accept_if(|c| c.eq_ignore_ascii_case(&'E')) {
            self.accept_if(|c| matches!(c, '+' | '-'));
            self.skip_int(start, None)?;
        }
        Ok(())
    }

    fn next_inner(&mut self) -> Result<Option<Token>, TokenError> {
        // TODO try moving this whitespace loop into the inner function, maybe that's faster
        let (start_byte, ty) = loop {
            let start_byte = self.curr_byte;
            match self.next_inner_ty()? {
                NextInnerResult::Ty(ty) => {
                    self.prev_token = Some(ty);
                    break (start_byte, ty);
                }
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
            pub const CUSTOM_TOKENS: &[(&str, TokenType)] = &[
                $((stringify!($c_token), TokenType::$c_token),)*
            ];
            pub const FIXED_TOKENS: &[FixedTokenInfo] = &[
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
pub const TOKEN_STR_BUILTIN_WITHOUT_UNDERSCORES: &str = "builtin";
pub const TOKEN_STR_UNSAFE_VALUE_WITH_DOMAIN: &str = "unsafe_value_with_domain";

// TODO rename tokens to match the literal string better
declare_tokens! {
    custom {
        SingleLineComment(TC::Comment),
        DelimitedComment(TC::Comment),

        Identifier(TC::Identifier),
        DecimalLiteral(TC::IntegerLiteral),
        BasedLiteral(TC::IntegerLiteral),
        BitStringLiteral(TC::IntegerLiteral),
        CharacterLiteral(TC::StringLiteral),
        StringLiteral(TC::StringLiteral),

        AttributeQuote(TC::Symbol),
    }
    fixed {
        // reserved words
        ResAbs("abs", TC::Keyword),
        ResAccess("access", TC::Keyword),
        ResAfter("after", TC::Keyword),
        ResAlias("alias", TC::Keyword),
        ResAll("all", TC::Keyword),
        ResAnd("and", TC::Keyword),
        ResArchitecture("architecture", TC::Keyword),
        ResArray("array", TC::Keyword),
        ResAssert("assert", TC::Keyword),
        ResAssume("assume", TC::Keyword),
        ResAttribute("attribute", TC::Keyword),
        ResBegin("begin", TC::Keyword),
        ResBlock("block", TC::Keyword),
        ResBody("body", TC::Keyword),
        ResBuffer("buffer", TC::Keyword),
        ResBus("bus", TC::Keyword),
        ResCase("case", TC::Keyword),
        ResComponent("component", TC::Keyword),
        ResConfiguration("configuration", TC::Keyword),
        ResConstant("constant", TC::Keyword),
        ResContext("context", TC::Keyword),
        ResCover("cover", TC::Keyword),
        ResDefault("default", TC::Keyword),
        ResDisconnect("disconnect", TC::Keyword),
        ResDownto("downto", TC::Keyword),
        ResElse("else", TC::Keyword),
        ResElsif("elsif", TC::Keyword),
        ResEnd("end", TC::Keyword),
        ResEntity("entity", TC::Keyword),
        ResExit("exit", TC::Keyword),
        ResFairness("fairness", TC::Keyword),
        ResFile("file", TC::Keyword),
        ResFor("for", TC::Keyword),
        ResForce("force", TC::Keyword),
        ResFunction("function", TC::Keyword),
        ResGenerate("generate", TC::Keyword),
        ResGeneric("generic", TC::Keyword),
        ResGroup("group", TC::Keyword),
        ResGuarded("guarded", TC::Keyword),
        ResIf("if", TC::Keyword),
        ResImpure("impure", TC::Keyword),
        ResIn("in", TC::Keyword),
        ResInertial("inertial", TC::Keyword),
        ResInout("inout", TC::Keyword),
        ResIs("is", TC::Keyword),
        ResLabel("label", TC::Keyword),
        ResLibrary("library", TC::Keyword),
        ResLinkage("linkage", TC::Keyword),
        ResLiteral("literal", TC::Keyword),
        ResLoop("loop", TC::Keyword),
        ResMap("map", TC::Keyword),
        ResMod("mod", TC::Keyword),
        ResNand("nand", TC::Keyword),
        ResNew("new", TC::Keyword),
        ResNext("next", TC::Keyword),
        ResNor("nor", TC::Keyword),
        ResNot("not", TC::Keyword),
        ResNull("null", TC::Keyword),
        ResOf("of", TC::Keyword),
        ResOn("on", TC::Keyword),
        ResOpen("open", TC::Keyword),
        ResOr("or", TC::Keyword),
        ResOthers("others", TC::Keyword),
        ResOut("out", TC::Keyword),
        ResPackage("package", TC::Keyword),
        ResParameter("parameter", TC::Keyword),
        ResPort("port", TC::Keyword),
        ResPostponed("postponed", TC::Keyword),
        ResPrivate("private", TC::Keyword),
        ResProcedure("procedure", TC::Keyword),
        ResProcess("process", TC::Keyword),
        ResProperty("property", TC::Keyword),
        ResProtected("protected", TC::Keyword),
        ResPure("pure", TC::Keyword),
        ResRange("range", TC::Keyword),
        ResRecord("record", TC::Keyword),
        ResRegister("register", TC::Keyword),
        ResReject("reject", TC::Keyword),
        ResRelease("release", TC::Keyword),
        ResRem("rem", TC::Keyword),
        ResReport("report", TC::Keyword),
        ResRestrict("restrict", TC::Keyword),
        ResReturn("return", TC::Keyword),
        ResRol("rol", TC::Keyword),
        ResRor("ror", TC::Keyword),
        ResSelect("select", TC::Keyword),
        ResSequence("sequence", TC::Keyword),
        ResSeverity("severity", TC::Keyword),
        ResShared("shared", TC::Keyword),
        ResSignal("signal", TC::Keyword),
        ResSla("sla", TC::Keyword),
        ResSll("sll", TC::Keyword),
        ResSra("sra", TC::Keyword),
        ResSrl("srl", TC::Keyword),
        ResStrong("strong", TC::Keyword),
        ResSubtype("subtype", TC::Keyword),
        ResThen("then", TC::Keyword),
        ResTo("to", TC::Keyword),
        ResTransport("transport", TC::Keyword),
        ResType("type", TC::Keyword),
        ResUnaffected("unaffected", TC::Keyword),
        ResUnits("units", TC::Keyword),
        ResUntil("until", TC::Keyword),
        ResUse("use", TC::Keyword),
        ResVariable("variable", TC::Keyword),
        ResView("view", TC::Keyword),
        ResVmode("vmode", TC::Keyword),
        ResVpkg("vpkg", TC::Keyword),
        ResVprop("vprop", TC::Keyword),
        ResVunit("vunit", TC::Keyword),
        ResWait("wait", TC::Keyword),
        ResWhen("when", TC::Keyword),
        ResWhile("while", TC::Keyword),
        ResWith("with", TC::Keyword),
        ResXnor("xnor", TC::Keyword),
        ResXor("xor", TC::Keyword),

        // braces
        OpenR("(", TC::Symbol),
        CloseR(")", TC::Symbol),
        OpenS("[", TC::Symbol),
        CloseS("]", TC::Symbol),

        // misc symbols
        Semi(";", TC::Symbol),
        Colon(":", TC::Symbol),
        Dot(".", TC::Symbol),
        Comma(",", TC::Symbol),
        Quest("?", TC::Symbol),
        AtSign("@", TC::Symbol),
        Caret("^", TC::Symbol),
        Pipe("|", TC::Symbol),
        Arrow("=>", TC::Symbol),
        ColonEq(":=", TC::Symbol),

        // operators
        QuestQuest("??", TC::Symbol),
        Eq("=", TC::Symbol),
        SlashEq("/=", TC::Symbol),
        Lt("<", TC::Symbol),
        LtEq("<=", TC::Symbol),
        Gt(">", TC::Symbol),
        GtEq(">=", TC::Symbol),
        QuestEq("?=", TC::Symbol),
        QuestSlashEq("?/=", TC::Symbol),
        QuestLt("?<", TC::Symbol),
        QuestLtEq("?<=", TC::Symbol),
        QuestGt("?>", TC::Symbol),
        QuestGtEq("?>=", TC::Symbol),
        Plus("+", TC::Symbol),
        Minus("-", TC::Symbol),
        Amper("&", TC::Symbol),
        Star("*", TC::Symbol),
        Slash("/", TC::Symbol),
        StarStar("**", TC::Symbol),
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FixedTokenInfo {
    pub name: &'static str,
    pub literal: &'static str,
    pub ty: TokenType,
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
    pub fn to_diagnostic(self) -> DiagnosticError {
        let title = "tokenization error";

        match self {
            TokenError::InvalidToken { pos } => {
                DiagnosticError::new(title, Span::empty_at(pos), "invalid start of token")
            }

            TokenError::UnexpectedChar { start, pos, actual } => DiagnosticError::new(
                format!("{title}: unexpected character"),
                Span::empty_at(pos),
                format!("unexpected character {actual:?}"),
            )
            .add_info(Span::empty_at(start), "while parsing integer literal that started here"),

            TokenError::IntLiteralInvalidBase { start, base } => {
                DiagnosticError::new(format!("{title}: invalid base"), base, "invalid base")
                    .add_info(Span::empty_at(start), "while parsing integer literal that started here")
            }

            TokenError::UnexpectedEof { kind, start, eof } => {
                DiagnosticError::new(format!("{title}: {kind}"), Span::empty_at(eof), "end of file reached")
                    .add_info(Span::empty_at(start), format!("{kind} started here"))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::syntax::token::{tokenize, Token, TokenType};
    use hwl_common::pos::Span;
    use hwl_common::source::FileId;
    use hwl_util::swriteln;
    use std::collections::HashSet;

    #[test]
    fn comment() {
        let file = FileId::dummy();

        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::SingleLineComment,
                span: Span::new(file, 0, 7)
            }]),
            tokenize(file, "-- test", false)
        );

        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::DelimitedComment,
                span: Span::new(file, 0, 4)
            }]),
            tokenize(file, "/**/", false)
        );

        assert_eq!(
            Ok(vec![Token {
                ty: TokenType::DelimitedComment,
                span: Span::new(file, 0, 6)
            }]),
            tokenize(file, "/*/**/", false)
        );
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
