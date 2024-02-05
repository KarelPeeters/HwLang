use std::mem::swap;
use std::num::ParseIntError;

use crate::parse::ast::{Args, Assignment, BinaryOp, Block, Declaration, Expression, ExpressionKind, ForExpression, Identifier, IfExpression, IntPattern, Item, ItemDefConst, ItemDefFunc, ItemDefInterface, ItemDefStruct, ItemDefType, ItemUse, LoopExpression, MaybeIdentifier, PackageContent, Param, ParamKind, Params, Path, Spanned, Statement, StatementKind, StructField, StructLiteral, StructLiteralField, UnaryOp, WhileExpression};
use crate::parse::pos::{FileId, Pos, Span};

#[derive(Debug)]
pub struct Token {
    pub ty: TT,
    pub string: String,
    pub span: Span,
}

impl Token {
    pub fn eof_token(pos: Pos) -> Token {
        Token {
            ty: TT::Eof,
            string: "".to_string(),
            span: Span::empty_at(pos),
        }
    }
}

macro_rules! declare_tokens {
    ($($token:ident$(($string:literal))?,)*) => {
        #[derive(Eq, PartialEq, Copy, Clone, Debug)]
        pub enum TT {
            $($token,)*
        }

        const TRIVIAL_TOKEN_LIST: &[(&'static str, TT)] = &[
            $($(($string, TT::$token),)?)*
        ];
    };
}

declare_tokens![
    Id,

    // [0-9]+
    IntLiteralDec,
    // 0b[01\.]+
    IntPatternBin,
    // 0x[0-9a-f\.]+
    IntPatternHex,

    StringLiteral,

    True("true"),
    False("false"),

    Use("use"),
    Type("type"),
    Struct("struct"),
    Fn("fn"),
    Interface("interface"),
    Return("return"),
    Let("let"),
    Const("const"),
    Static("static"),
    If("if"),
    Else("else"),
    Loop("loop"),
    While("while"),
    For("for"),
    In("in"),
    As("as"),
    Break("break"),
    Continue("continue"),

    Underscore("_"),
    ArrowRight("->"),
    ArrowLeft("<-"),
    DoubleDotEq("..="),
    DoubleDot(".."),

    BangEq("!="),
    DoubleEq("=="),
    GreaterEqual(">="),
    Greater(">"),
    LessEqual("<="),
    Less("<"),

    // TODO find a better and more general solution to token overlaps
    //   (probably by rewriting the tokenizer to split tokens on-demand)
    DoubleAmpersand("&&"),
    DoublePipe("||"),
    DoubleColon("::"),
    DoubleStar("**"),

    PlusEq("+="),
    MinusEq("-="),
    StarEq("*="),
    SlashEq("/="),
    PercentEq("%="),
    AmpersandEq("&="),
    PipeEq("|="),
    HatEq("^="),

    Plus("+"),
    Minus("-"),
    Star("*"),
    Slash("/"),
    Percent("%"),
    Ampersand("&"),
    Pipe("|"),
    Hat("^"),
    Bang("!"),

    Dot("."),
    Semi(";"),
    Colon(":"),
    QuestionMark("?"),
    Comma(","),
    Eq("="),

    OpenB("("),
    CloseB(")"),
    OpenC("{"),
    CloseC("}"),
    OpenS("["),
    CloseS("]"),

    Eof,
];

type Result<T> = std::result::Result<T, ParseError>;

#[derive(Debug)]
pub enum ParseError {
    Char {
        pos: Pos,
        char: char,
    },
    Token {
        pos: Pos,
        ty: TT,
        description: &'static str,
        allowed: Vec<TT>,
    },
    Eof {
        after: Pos,
        expected: &'static str,
    },

    IntLit {
        span: Span,
        value: String,
        error: ParseIntError,
    },
    ExpectedEndOfBlock {
        pos: Pos,
    },
    CannotChainOperator {
        span: Span,
    },
    UnnamedNotAllowedAfterNamed {
        pos: Pos,
    },
}

struct Tokenizer<'s> {
    left: &'s str,
    pos: Pos,

    curr: Token,
    next: Token,
}

impl<'s> Tokenizer<'s> {
    fn new(file: FileId, left: &'s str) -> Result<Self> {
        let pos = Pos { file, line: 1, col: 1 };
        let mut result = Self {
            left,
            pos,
            curr: Token::eof_token(pos),
            next: Token::eof_token(pos),
        };
        result.advance()?;
        result.advance()?;
        Ok(result)
    }

    /// self.left should only be advanced trough this function to ensure self.pos is updated
    fn skip_count(&mut self, count: usize) -> &str {
        //update position
        let skipped = &self.left[0..count];
        if let Some(last_newline) = skipped.rfind('\n') {
            self.pos.col = count - last_newline;
            self.pos.line += skipped.matches('\n').count();
        } else {
            self.pos.col += count;
        }

        self.left = &self.left[count..];
        skipped
    }

    fn skip_past(&mut self, pattern: &'static str, allow_eof: bool) -> Result<()> {
        let start_pos = self.pos;

        match self.left.find(pattern) {
            Some(i) => {
                //skip up to and including the pattern
                self.skip_count(i + pattern.len());
                Ok(())
            }
            None => {
                if !allow_eof { return Err(ParseError::Eof { after: start_pos, expected: pattern }); }

                //skip to the end
                self.skip_count(self.left.len());
                Ok(())
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<()> {
        loop {
            let prev_left = self.left;
            self.skip_count(self.left.len() - self.left.trim_start().len());

            if self.left.starts_with("//") {
                self.skip_past("\n", true)?;
            }
            if self.left.starts_with("/*") {
                self.skip_past("*/", false)?;
            }

            if prev_left == self.left { return Ok(()); }
        }
    }

    fn parse_int_pattern(&mut self) -> Result<Option<Token>> {
        let mut offset = 0;
        let start_pos = self.pos;

        let minus = self.left[offset..].starts_with('-');
        if minus {
            offset += 1;
        }

        let hex = self.left[offset..].starts_with("0x");
        let bin = self.left[offset..].starts_with("0b");

        // TODO The fact that we are combining the minus with the literal is annoying, since that means we require
        //   spaces to subtract a non-"-"-literal. Either delay combining minus with literal until semantics or
        //   allow splitting the minus off again during parsing.
        if minus && (hex | bin) {
            // negative hex and bin literals are not useful, there is no "most negative" edge case anyway
            // we cancel parsing an int here, we want "(-) (0x..)", not "(-0) (x..)" which we are currently trying
            return Ok(None);
        }

        let (ty, prefix_len, is_digit): (_, _, fn(char) -> bool) = if hex {
            (TT::IntPatternHex, 2, |c: char| c.is_ascii_hexdigit() || c == '.')
        } else if bin {
            (TT::IntPatternBin, 2, |c: char| c == '0' || c == '1' || c == '.')
        } else {
            (TT::IntLiteralDec, 0, |c: char| c.is_ascii_digit())
        };

        offset += prefix_len;

        let peek = match self.left[offset..].chars().next() {
            None => {
                self.skip_count(offset);
                return Err(ParseError::Eof { after: self.pos, expected: "integer literal digits" });
            }
            Some(peek) => peek,
        };
        if !is_digit(peek) {
            self.skip_count(offset);
            return Err(ParseError::Char { pos: self.pos, char: peek });
        }

        let digits = self.left[offset..].find(|c: char| !is_digit(c)).unwrap_or(self.left[offset..].len());
        let string = self.skip_count(offset + digits).to_owned();

        Ok(Some(Token {
            ty,
            string,
            span: Span::new(start_pos, self.pos),
        }))
    }

    fn parse_next(&mut self) -> Result<Token> {
        self.skip_whitespace_and_comments()?;
        let start_pos = self.pos;

        let peek = if let Some(peek) = self.left.chars().next() {
            peek
        } else {
            return Ok(Token::eof_token(start_pos));
        };
        let lookahead = self.left.chars().nth(1);

        //integer literal
        if peek.is_ascii_digit() || (peek == '-' && lookahead.as_ref().map_or(false, char::is_ascii_digit)) {
            if let Some(token) = self.parse_int_pattern()? {
                return Ok(token);
            }
        }

        //identifier
        if peek.is_alphabetic() || peek == '_' {
            let end = self.left
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '@'))
                .unwrap_or(self.left.len());
            let string = self.skip_count(end).to_owned();

            //check if it it happens to be a keyword:
            let ty = TRIVIAL_TOKEN_LIST.iter()
                .find(|(pattern, _)| pattern == &string)
                .map(|&(_, ty)| ty)
                .unwrap_or(TT::Id);

            return Ok(Token {
                ty,
                string,
                span: Span::new(start_pos, self.pos),
            });
        }

        //string literal
        if peek == '"' {
            let end = 1 + self.left[1..].find('"')
                .ok_or(ParseError::Eof { after: self.pos, expected: "\"" })?;
            let content = self.skip_count(end + 1)[1..end].to_owned();

            return Ok(Token {
                ty: TT::StringLiteral,
                string: content,
                span: Span::new(start_pos, self.pos),
            });
        }

        //trivial token
        for (pattern, ty) in TRIVIAL_TOKEN_LIST {
            if self.left.starts_with(pattern) {
                self.skip_count(pattern.len());
                let end_pos = self.pos;
                return Ok(Token {
                    ty: *ty,
                    string: pattern.to_string(),
                    span: Span::new(start_pos, end_pos),
                });
            }
        }

        Err(ParseError::Char {
            pos: self.pos,
            char: peek,
        })
    }

    fn advance(&mut self) -> Result<Token> {
        let next = self.parse_next()?;

        let mut result = Token::eof_token(self.pos);

        swap(&mut result, &mut self.curr);
        swap(&mut self.curr, &mut self.next);

        self.next = next;
        Ok(result)
    }
}

#[derive(Debug)]
struct Restrictions {
    no_struct_literal: bool,
    no_ternary: bool,
}

impl Restrictions {
    const NONE: Restrictions = Restrictions { no_struct_literal: false, no_ternary: false };
    const NO_STRUCT_LITERAL: Restrictions = Restrictions { no_struct_literal: true, no_ternary: false };
    const NO_TERNARY: Restrictions = Restrictions { no_struct_literal: false, no_ternary: true };
}

struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    last_popped_end: Pos,
    restrictions: Restrictions,
}

#[derive(Debug)]
struct BinOpInfo {
    level: u8,
    token: TT,
    chain: Chain,
    op: BinaryOp,
}

#[derive(Debug, Copy, Clone)]
enum Chain {
    /// Allow chaining with any operator with the same level.
    SameLevel,
    /// Allow chaining with only the exact same operator.
    SameOp,
    /// Don't allow chaining with any operators of the same level, not even the operator itself.
    Never,
}

// Order of operations table, inspired by
// * Rust: https://doc.rust-lang.org/reference/expressions.html#expression-precedence
// * C++: https://en.cppreference.com/w/cpp/language/operator_precedence
// * Java: https://docs.oracle.com/javase/tutorial/java/nutsandbolts/operators.html
// TODO fuzz this with some randomly generated code to see if the parse tree always looks right
const BINARY_OPERATOR_INFO: &[BinOpInfo] = &[
    // logical
    BinOpInfo { level: 2, token: TT::DoublePipe, chain: Chain::SameOp, op: BinaryOp::BoolOr },
    BinOpInfo { level: 2, token: TT::DoubleAmpersand, chain: Chain::SameOp, op: BinaryOp::BoolAnd },
    // comparison
    BinOpInfo { level: 3, token: TT::DoubleEq, chain: Chain::Never, op: BinaryOp::CmpEq },
    BinOpInfo { level: 3, token: TT::BangEq, chain: Chain::Never, op: BinaryOp::CmpNeq },
    BinOpInfo { level: 3, token: TT::GreaterEqual, chain: Chain::Never, op: BinaryOp::CmpGte },
    BinOpInfo { level: 3, token: TT::Greater, chain: Chain::Never, op: BinaryOp::CmpGt },
    BinOpInfo { level: 3, token: TT::LessEqual, chain: Chain::Never, op: BinaryOp::CmpLte },
    BinOpInfo { level: 3, token: TT::Less, chain: Chain::Never, op: BinaryOp::CmpLt },
    // in
    BinOpInfo { level: 4, token: TT::In, chain: Chain::Never, op: BinaryOp::In },
    // range
    BinOpInfo { level: 5, token: TT::DoubleDot, chain: Chain::Never, op: BinaryOp::Range },
    BinOpInfo { level: 5, token: TT::DoubleDotEq, chain: Chain::Never, op: BinaryOp::RangeInclusive },
    // bitwise
    // TODO why bitwise and logical so far apart? does the order relative to add make sense?
    BinOpInfo { level: 6, token: TT::Pipe, chain: Chain::SameOp, op: BinaryOp::BitOr },
    BinOpInfo { level: 6, token: TT::Hat, chain: Chain::SameOp, op: BinaryOp::BitXor },
    BinOpInfo { level: 6, token: TT::Ampersand, chain: Chain::SameOp, op: BinaryOp::BitAnd },
    // add/sub
    BinOpInfo { level: 7, token: TT::Plus, chain: Chain::SameLevel, op: BinaryOp::Add },
    BinOpInfo { level: 7, token: TT::Minus, chain: Chain::SameLevel, op: BinaryOp::Sub },
    // mul/div/mod
    // TODO is mod chaining disallowing working as expected?
    BinOpInfo { level: 8, token: TT::Star, chain: Chain::SameLevel, op: BinaryOp::Mul },
    BinOpInfo { level: 8, token: TT::Slash, chain: Chain::SameLevel, op: BinaryOp::Div },
    BinOpInfo { level: 8, token: TT::Percent, chain: Chain::Never, op: BinaryOp::Mod },
    // power (we avoid the right-to-left binding by disallowing chaining entirely)
    BinOpInfo { level: 9, token: TT::DoubleStar, chain: Chain::Never, op: BinaryOp::Pow },
];

const BINARY_ASSIGNMENT_OPERATORS: &[(TT, Option<BinaryOp>)] = &[
    (TT::Eq, None),
    (TT::PipeEq, Some(BinaryOp::BitOr)),
    (TT::HatEq, Some(BinaryOp::BitXor)),
    (TT::AmpersandEq, Some(BinaryOp::BitAnd)),
    (TT::PlusEq, Some(BinaryOp::Add)),
    (TT::MinusEq, Some(BinaryOp::Sub)),
    (TT::StarEq, Some(BinaryOp::Mul)),
    (TT::SlashEq, Some(BinaryOp::Div)),
    (TT::PercentEq, Some(BinaryOp::Mod)),
];

struct PrefixOpInfo {
    level: u8,
    token: TT,
    op: UnaryOp,
}

const PREFIX_OPERATOR_INFO: &[PrefixOpInfo] = &[
    PrefixOpInfo { level: 2, token: TT::Minus, op: UnaryOp::Neg },
    PrefixOpInfo { level: 2, token: TT::Bang, op: UnaryOp::Not },
];

// TODO check this and maybe define a level for each suffix op
const POSTFIX_LEVEL_DEFAULT: u8 = 3;
const POSTFIX_LEVEL_STRUCT_LITERAL: u8 = 1;

/// The data required to construct a prefix expression.
struct PrefixState {
    level: u8,
    start: Pos,
    op: UnaryOp,
}

impl PrefixState {
    fn apply(self, inner: Expression) -> Expression {
        Expression {
            span: Span::new(self.start, inner.span.end),
            kind: ExpressionKind::UnaryOp(self.op, Box::new(inner)),
        }
    }
}

/// The data required to construct a postfix expression.
struct PostFixState {
    level: u8,
    end: Pos,
    kind: PostFixStateKind,
}

impl PostFixState {
    fn apply(self, inner: Expression) -> Expression {
        let inner = Box::new(inner);
        let span = Span::new(inner.span.start, self.end);

        let kind = match self.kind {
            PostFixStateKind::Call { args } =>
                ExpressionKind::Call(inner, args),
            PostFixStateKind::ArrayIndex { index } =>
                ExpressionKind::ArrayIndex(inner, index),
            PostFixStateKind::DotIdIndex { index } =>
                ExpressionKind::DotIdIndex(inner, index),
            PostFixStateKind::DotIntIndex { span, index } =>
                ExpressionKind::DotIntIndex(inner, Spanned { span, inner: index }),
            PostFixStateKind::StructInit { fields } =>
                ExpressionKind::StructInit(StructLiteral { struct_ty: inner, fields })
        };

        Expression { span, kind }
    }
}

enum PostFixStateKind {
    Call { args: Args },
    ArrayIndex { index: Box<Expression> },
    DotIdIndex { index: Identifier },
    DotIntIndex { span: Span, index: u32 },
    StructInit { fields: Vec<StructLiteralField> },
}

#[derive(Debug)]
struct ListResult<A> {
    trailing_separator: bool,
    values: Vec<A>,
}

#[allow(dead_code)]
impl<'s> Parser<'s> {
    fn pop(&mut self) -> Result<Token> {
        let token = self.tokenizer.advance()?;
        self.last_popped_end = token.span.end;
        Ok(token)
    }

    fn peek(&self) -> &Token {
        &self.tokenizer.curr
    }

    fn lookahead(&self) -> &Token {
        &self.tokenizer.next
    }

    fn at(&mut self, ty: TT) -> bool {
        self.peek().ty == ty
    }

    /// pop and return the next token if the type matches, otherwise do nothing and return None
    fn accept(&mut self, ty: TT) -> Result<Option<Token>> {
        if self.at(ty) {
            self.pop().map(Some)
        } else {
            Ok(None)
        }
    }

    /// pop and return the next token if the type matches, otherwise return an error
    fn expect(&mut self, ty: TT, description: &'static str) -> Result<Token> {
        if self.at(ty) {
            self.pop()
        } else {
            Err(Self::unexpected_token(
                self.peek(),
                &[ty],
                description,
            ))
        }
    }

    /// call `expect` on each type in sequence, return an error if any `expect` fails
    fn expect_all(&mut self, tys: &[TT], description: &'static str) -> Result<()> {
        for &ty in tys {
            self.expect(ty, description)?;
        }
        Ok(())
    }

    /// pop and return the next token if the type matches any of the given types, otherwise return an error
    fn expect_any(&mut self, tys: &'static [TT], description: &'static str) -> Result<Token> {
        if tys.contains(&self.peek().ty) {
            Ok(self.pop()?)
        } else {
            Err(Self::unexpected_token(self.peek(), tys, description))
        }
    }

    fn unexpected_token(token: &Token, allowed: &[TT], description: &'static str) -> ParseError {
        ParseError::Token {
            ty: token.ty,
            pos: token.span.start,
            allowed: allowed.to_vec(),
            description,
        }
    }

    fn list<A, F: FnMut(&mut Self) -> Result<A>>(
        &mut self,
        end: TT,
        sep: Option<TT>,
        mut item: F,
    ) -> Result<ListResult<A>> {
        let mut values = Vec::new();

        let maybe_trailing_separator = loop {
            if self.accept(end)?.is_some() {
                break true;
            }
            values.push(item(self)?);
            if self.accept(end)?.is_some() {
                break false;
            }
            if let Some(sep) = sep {
                self.expect(sep, "separator")?;
            }
        };

        let result = ListResult {
            trailing_separator: maybe_trailing_separator && !values.is_empty(),
            values,
        };
        Ok(result)
    }
}

impl<'s> Parser<'s> {
    // TODO replace or combine?
    fn restrict<T>(&mut self, res: Restrictions, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = std::mem::replace(&mut self.restrictions, res);
        let result = f(self);
        self.restrictions = old;
        result
    }

    fn unrestrict<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.restrict(Restrictions::NONE, f)
    }

    fn main_package_content(&mut self) -> Result<PackageContent> {
        let start = self.last_popped_end;
        let items = self.list(TT::Eof, None, Self::item)?.values;
        let span = Span::new(start, self.last_popped_end);
        Ok(PackageContent { span, items })
    }

    fn item(&mut self) -> Result<Item> {
        let token = self.peek();

        match token.ty {
            TT::Struct => self.item_def_struct().map(Item::Struct),
            TT::Const => self.item_def_const().map(Item::Const),
            TT::Use => self.item_use().map(Item::Use),
            TT::Type => self.type_alias().map(Item::Type),
            TT::Fn => self.function().map(Item::Func),
            TT::Interface => self.interface().map(Item::Interface),
            _ => Err(Self::unexpected_token(token, &[TT::Struct, TT::Const, TT::Use, TT::Type, TT::Fn], "start of item"))
        }
    }

    fn maybe_params(&mut self) -> Result<Params> {
        if self.at(TT::OpenB) {
            Ok(self.params()?)
        } else {
            Ok(Params { span: Span::empty_at(self.last_popped_end), params: vec![] })
        }
    }

    fn params(&mut self) -> Result<Params> {
        let start = self.expect(TT::OpenB, "start of parameters")?.span.start;
        let params = self.list(TT::CloseB, Some(TT::Comma), Self::param)?.values;
        let span = Span::new(start, self.last_popped_end);
        Ok(Params { span, params })
    }

    fn param(&mut self) -> Result<Param> {
        // TODO a proper grammar DSL would be a lot more convenient here
        // types of parameter:
        // * `ty`
        // * `id: ty`
        // * `id: ty = default`

        let ty_or_id = self.expression()?;

        if let ExpressionKind::Path(Path { span: _, parents, id }) = &ty_or_id.kind {
            if parents.is_empty() {
                if self.accept(TT::Colon)?.is_some() {
                    let ty = self.expression()?;

                    let default = if self.accept(TT::Eq)?.is_some() {
                        Some(self.expression()?)
                    } else {
                        None
                    };

                    return Ok(Param {
                        span: Span::new(ty_or_id.span.start, self.last_popped_end),
                        kind: ParamKind::Named { id: id.clone(), default },
                        ty,
                    });
                }
            }
        }

        Ok(Param {
            span: ty_or_id.span,
            kind: ParamKind::Anonymous,
            ty: ty_or_id,
        })
    }

    fn call_args(&mut self) -> Result<Args> {
        let start = self.expect(TT::OpenB, "args start")?.span.start;

        let mut positional = vec![];
        let mut named = vec![];

        let _ = self.list(TT::CloseB, Some(TT::Comma), |s| {
            if s.at(TT::Id) && s.peek().ty == TT::Eq {
                let id = s.identifier("generic arg name")?;
                s.expect(TT::Eq, "generic arg separator")?;
                let value = s.expression()?;
                named.push((id, value));
            } else {
                if !named.is_empty() {
                    return Err(ParseError::UnnamedNotAllowedAfterNamed { pos: s.peek().span.start });
                }

                let value = s.expression()?;
                positional.push(value);
            }

            Ok(())
        })?;

        let span = Span::new(start, self.last_popped_end);
        Ok(Args { span, positional, named })
    }

    fn item_def_const(&mut self) -> Result<ItemDefConst> {
        let start = self.pop()?;
        let id = self.identifier("const name")?;
        self.expect(TT::Colon, "const type")?;
        let ty = self.expression()?;
        self.expect(TT::Eq, "const initializer")?;

        let value = if self.at(TT::Semi) {
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(TT::Semi, "const end")?;

        let span = Span::new(start.span.start, self.last_popped_end);
        Ok(ItemDefConst { span, id, ty, value })
    }

    fn item_use(&mut self) -> Result<ItemUse> {
        let start_pos = self.expect(TT::Use, "start of use decl")?.span.start;
        let path = self.path()?;
        let as_ = self.accept(TT::As)?.map(|_| self.identifier("as name")).transpose()?;
        self.expect(TT::Semi, "end of item")?;

        let span = Span::new(start_pos, path.span.end);
        Ok(ItemUse { span, path, as_ })
    }

    fn type_alias(&mut self) -> Result<ItemDefType> {
        let start_pos = self.expect(TT::Type, "start of type alias")?.span.start;
        let id = self.identifier("type alias name")?;
        let params = self.maybe_params()?;

        let inner = if self.accept(TT::Eq)?.is_some() {
            Some(self.expression_boxed()?)
        } else {
            None
        };

        self.expect(TT::Semi, "type alias end")?;

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(ItemDefType { span, id, params, inner })
    }

    fn item_def_struct(&mut self) -> Result<ItemDefStruct> {
        let start = self.expect(TT::Struct, "start of struct declaration")?.span.start;
        let id = self.identifier("struct name")?;
        let params = self.maybe_params()?;
        self.expect(TT::OpenC, "start of struct fields")?;
        let fields = self.list(TT::CloseC, Some(TT::Comma), Self::struct_field)?.values;
        let span = Span::new(start, self.last_popped_end);
        Ok(ItemDefStruct { span, id, params, fields })
    }

    fn struct_field(&mut self) -> Result<StructField> {
        let id = self.identifier("field name")?;
        self.expect(TT::Colon, "field type")?;
        let ty = self.expression()?;
        let span = Span::new(id.span.start, ty.span.end);
        Ok(StructField { span, id, ty })
    }

    fn function(&mut self) -> Result<ItemDefFunc> {
        let start_pos = self.peek().span.start;

        self.expect(TT::Fn, "function declaration")?;
        let id = self.identifier("function name")?;
        let params = self.params()?;

        let ret_ty = if self.accept(TT::ArrowRight)?.is_some() {
            Some(self.expression()?)
        } else {
            None
        };

        let body = self.maybe_block()?;

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(ItemDefFunc { span, id, params, ret_ty, body })
    }

    fn interface(&mut self) -> Result<ItemDefInterface> {
        let start_pos = self.peek().span.start;

        self.expect(TT::Interface, "interface declaration")?;

        todo!()
    }

    fn maybe_block(&mut self) -> Result<Option<Block>> {
        if self.at(TT::OpenC) {
            Ok(Some(self.block()?))
        } else {
            Ok(None)
        }
    }

    fn block(&mut self) -> Result<Block> {
        let start_pos = self.expect(TT::OpenC, "start of block")?.span.start;

        // TODO better statement parsing
        let mut must_be_last = false;
        let statements = self.list(TT::CloseC, None, |s| s.statement(&mut must_be_last))?.values;

        Ok(Block { span: Span::new(start_pos, self.last_popped_end), statements })
    }

    fn statement(&mut self, must_be_last: &mut bool) -> Result<Statement> {
        let token = self.peek();
        let start_pos = token.span.start;

        if *must_be_last {
            return Err(ParseError::ExpectedEndOfBlock { pos: start_pos });
        }

        // TODO update, add new statements
        let (kind, need_semi) = match token.ty {
            TT::Let => {
                //declaration
                // TODO mutability?
                let start_pos = self.pop()?.span.start;
                let id = self.maybe_identifier("variable name")?;

                let ty = self.maybe_type_decl()?.map(Box::new);
                let init = self.accept(TT::Eq)?
                    .map(|_| self.expression_boxed())
                    .transpose()?;

                let span = Span::new(start_pos, self.last_popped_end);
                let decl = Declaration { span, id, ty, init };

                (StatementKind::Declaration(decl), true)
            }
            _ => {
                let left = self.expression_boxed()?;

                if let Some(&(_, op)) = BINARY_ASSIGNMENT_OPERATORS.iter().find(|&&(ty, _)| self.at(ty)) {
                    // binary assignment
                    self.pop()?;
                    let right = self.expression_boxed()?;
                    let stmt = StatementKind::Assignment(Assignment {
                        span: Span::new(left.span.start, right.span.end),
                        op,
                        left,
                        right,
                    });
                    (stmt, true)
                } else {
                    //expression
                    let needs_semi = left.kind.needs_semi();
                    (StatementKind::Expression(left), needs_semi)
                }
            }
        };

        if need_semi {
            if !(self.accept(TT::Semi)?.is_some()) {
                *must_be_last = true;
            }
        }

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(Statement { span, kind })
    }

    fn expression_boxed(&mut self) -> Result<Box<Expression>> {
        Ok(Box::new(self.expression()?))
    }

    fn expression(&mut self) -> Result<Expression> {
        let expr = self.precedence_climb_binop(0, None, Chain::SameLevel)?;
        let start = expr.span.start;

        if !self.restrictions.no_ternary && self.accept(TT::QuestionMark)?.is_some() {
            let then_value = self.expression()?;
            self.expect(TT::Colon, "continue ternary expression")?;
            let else_value = self.expression()?;

            let kind = ExpressionKind::TernarySelect(
                Box::new(expr),
                Box::new(then_value),
                Box::new(else_value),
            );

            Ok(Expression {
                span: Span::new(start, self.last_popped_end),
                kind,
            })
        } else {
            Ok(expr)
        }
    }

    fn precedence_climb_binop(&mut self, level: u8, op: Option<BinaryOp>, chain: Chain) -> Result<Expression> {
        let mut curr = self.unary()?;

        loop {
            let token = self.peek();
            let info = BINARY_OPERATOR_INFO.iter()
                .find(|i| i.token == token.ty);

            if let Some(info) = info {
                if info.level == level {
                    let allowed = match chain {
                        Chain::SameLevel => true,
                        Chain::SameOp => info.op == op.expect("Checking chain, need operator"),
                        Chain::Never => false
                    };

                    if !allowed {
                        return Err(ParseError::CannotChainOperator { span: token.span });
                    }
                }
                if info.level <= level {
                    break;
                }

                self.pop()?;

                let right = self.precedence_climb_binop(info.level, Some(info.op), info.chain)?;

                let left = Box::new(curr);
                let right = Box::new(right);
                let span = Span::new(left.span.start, right.span.end);

                let kind = ExpressionKind::BinaryOp(info.op, left, right);
                curr = Expression { span, kind }
            } else {
                break;
            }
        }

        Ok(curr)
    }

    fn unary(&mut self) -> Result<Expression> {
        //collect all operators
        let mut prefix_ops = self.collect_prefix_ops()?;
        let curr = self.atomic()?;
        let mut postfix_ops = self.collect_postfix_ops()?;

        //postfix operations should be applied first-to-last, so reverse
        postfix_ops.reverse();

        //apply operations last-to-first, choosing between pre- and postfix depending on their levels
        let mut curr = curr;
        loop {
            let prefix_level = prefix_ops.last().map(|s| s.level);
            let postfix_level = postfix_ops.last().map(|s| s.level);

            match (prefix_level, postfix_level) {
                (Some(prefix_level), Some(postfix_level)) => {
                    assert_ne!(prefix_level, postfix_level);
                    if prefix_level > postfix_level {
                        curr = prefix_ops.pop().unwrap().apply(curr);
                    } else {
                        curr = postfix_ops.pop().unwrap().apply(curr);
                    }
                }
                (Some(_), None) => {
                    curr = prefix_ops.pop().unwrap().apply(curr);
                }
                (None, Some(_)) => {
                    curr = postfix_ops.pop().unwrap().apply(curr);
                }
                (None, None) => break
            }
        }

        Ok(curr)
    }

    fn collect_prefix_ops(&mut self) -> Result<Vec<PrefixState>> {
        let mut result = Vec::new();

        loop {
            let token = self.peek();
            let info = PREFIX_OPERATOR_INFO.iter()
                .find(|i| i.token == token.ty);

            if let Some(info) = info {
                let token = self.pop()?;
                result.push(PrefixState { level: info.level, start: token.span.start, op: info.op });
            } else {
                break;
            }
        }

        Ok(result)
    }

    fn collect_postfix_ops(&mut self) -> Result<Vec<PostFixState>> {
        let mut result = Vec::new();

        loop {
            let token = self.peek();

            let (level, kind) = match token.ty {
                TT::OpenB => {
                    //call
                    let args = self.call_args()?;
                    (POSTFIX_LEVEL_DEFAULT, PostFixStateKind::Call { args })
                }
                TT::OpenS => {
                    //array indexing
                    self.pop()?;
                    let index = self.expression_boxed()?;
                    self.expect(TT::CloseS, "")?;

                    (POSTFIX_LEVEL_DEFAULT, PostFixStateKind::ArrayIndex { index })
                }
                TT::OpenC if !self.restrictions.no_struct_literal => {
                    // struct literal
                    self.pop()?;
                    let fields = self.list(TT::CloseC, Some(TT::Comma), |s| {
                        let id = s.identifier("tuple literal field")?;
                        s.expect(TT::Colon, "tuple literal field separator")?;

                        let value = s.restrict(Restrictions::NO_TERNARY, |s| {
                            s.expression()
                        })?;

                        let span = Span::new(id.span.start, value.span.end);
                        Ok(StructLiteralField { span, id, value })
                    })?.values;

                    (POSTFIX_LEVEL_STRUCT_LITERAL, PostFixStateKind::StructInit { fields })
                }
                TT::Dot => {
                    //dot indexing
                    self.pop()?;

                    let index = self.expect_any(&[TT::IntLiteralDec, TT::Id], "dot index index")?;
                    let kind = match index.ty {
                        TT::IntLiteralDec => PostFixStateKind::DotIntIndex {
                            span: index.span,
                            index: parse_int_literal(index)?,
                        },
                        TT::Id => PostFixStateKind::DotIdIndex {
                            index: Identifier { span: index.span, string: index.string },
                        },
                        _ => unreachable!(),
                    };

                    (POSTFIX_LEVEL_DEFAULT, kind)
                }
                _ => break
            };

            result.push(PostFixState { level, end: self.last_popped_end, kind });
        }

        Ok(result)
    }

    fn atomic(&mut self) -> Result<Expression> {
        let start_pos = self.peek().span.start;

        let kind = match self.peek().ty {
            TT::IntLiteralDec | TT::IntPatternBin | TT::IntPatternHex => ExpressionKind::IntPattern(self.int_pattern()?.inner),
            TT::True | TT::False => ExpressionKind::BoolLiteral(self.pop()?.string.parse().expect("TTs should parse correctly")),
            TT::StringLiteral => ExpressionKind::StringLiteral(self.pop()?.string),
            TT::Id => ExpressionKind::Path(self.path()?),
            TT::Underscore => {
                self.pop()?;
                ExpressionKind::Wildcard
            }
            TT::Type => {
                self.pop()?;
                ExpressionKind::Type
            }
            TT::OpenB => {
                // func or tuple
                self.pop()?;
                let list = self.list(TT::CloseB, Some(TT::Comma), Self::expression)?;

                if self.accept(TT::ArrowRight)?.is_some() {
                    let ret = self.expression()?;
                    ExpressionKind::TypeFunc(list.values, Box::new(ret))
                } else {
                    if list.values.len() == 1 && !list.trailing_separator {
                        let value = list.values.into_iter().next().unwrap();
                        ExpressionKind::Wrapped(Box::new(value))
                    } else {
                        ExpressionKind::TupleInit(list.values)
                    }
                }
            }
            TT::OpenS => {
                // array initializer
                self.pop()?;
                let items = self.list(TT::CloseS, Some(TT::Comma), Self::expression)?.values;
                ExpressionKind::ArrayInit(items)
            }
            TT::OpenC => {
                // block
                let block = self.unrestrict(|s| s.block())?;
                ExpressionKind::Block(block)
            }
            TT::If => {
                self.pop()?;
                let cond = self.restrict(Restrictions::NO_STRUCT_LITERAL, |s| s.expression_boxed())?;
                let then_block = self.block()?;

                let else_block = self.accept(TT::Else)?
                    .map(|_| self.block())
                    .transpose()?;

                ExpressionKind::If(IfExpression {
                    cond,
                    then_block,
                    else_block,
                })
            }
            TT::Loop => {
                self.pop()?;
                let body = self.block()?;
                ExpressionKind::Loop(LoopExpression { body })
            }
            TT::While => {
                self.pop()?;
                let cond = self.restrict(Restrictions::NO_STRUCT_LITERAL, |s| s.expression_boxed())?;
                let body = self.block()?;
                ExpressionKind::While(WhileExpression { cond, body })
            }
            TT::For => {
                self.pop()?;
                let index = self.maybe_identifier("index variable")?;
                let index_ty = self.maybe_type_decl()?.map(Box::new);
                self.expect(TT::In, "in")?;
                let range = self.restrict(Restrictions::NO_STRUCT_LITERAL, |s| {
                    s.expression_boxed()
                })?;
                let body = self.block()?;

                ExpressionKind::For(ForExpression {
                    index,
                    index_ty,
                    range,
                    body,
                })
            }
            TT::Return => {
                //TODO think about whether this is the right spot to parse a return
                //TODO should return and break be prefix operators?
                self.pop()?;

                // TODO also peak comma?
                let value = if self.peek().ty == TT::Semi {
                    None
                } else {
                    Some(self.expression_boxed()?)
                };

                ExpressionKind::Return(value)
            }
            TT::Continue => {
                self.pop()?;
                ExpressionKind::Continue
            }
            TT::Break => {
                self.pop()?;

                // TODO also peak comma?
                let value = if self.peek().ty == TT::Semi {
                    None
                } else {
                    Some(self.expression_boxed()?)
                };

                ExpressionKind::Break(value)
            }
            // TODO collect expr start tokens
            _ => return Err(Self::unexpected_token(self.peek(), &[], "expression"))
        };

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(Expression { span, kind })
    }

    // TODO parse this more fully here or in the tokenizer
    fn int_pattern(&mut self) -> Result<Spanned<IntPattern>> {
        let token = self.expect_any(&[TT::IntLiteralDec, TT::IntPatternBin, TT::IntPatternHex], "integer pattern")?;

        let pattern = match token.ty {
            TT::IntLiteralDec => IntPattern::Dec(token.string),
            TT::IntPatternBin => IntPattern::Bin(token.string[2..].to_owned()),
            TT::IntPatternHex => IntPattern::Hex(token.string[2..].to_owned()),
            _ => unreachable!(),
        };

        Ok(Spanned { span: token.span, inner: pattern })
    }

    fn path(&mut self) -> Result<Path> {
        let mut parents = Vec::new();
        let mut id = self.identifier("identifier")?;

        while self.accept(TT::DoubleColon)?.is_some() {
            parents.push(id);
            id = self.identifier("path element")?;
        }

        let span = Span::new(id.span.start, self.last_popped_end);
        Ok(Path { span, parents, id })
    }

    fn maybe_identifier(&mut self, description: &'static str) -> Result<MaybeIdentifier> {
        if self.at(TT::Underscore) {
            Ok(MaybeIdentifier::Placeholder(self.pop()?.span))
        } else {
            Ok(MaybeIdentifier::Identifier(self.identifier(description)?))
        }
    }

    fn identifier(&mut self, description: &'static str) -> Result<Identifier> {
        let token = self.expect(TT::Id, description)?;
        Ok(Identifier { span: token.span, string: token.string })
    }

    fn maybe_type_decl(&mut self) -> Result<Option<Expression>> {
        self.accept(TT::Colon)?
            .map(|_| self.expression())
            .transpose()
    }
}

fn parse_int_literal(token: Token) -> Result<u32> {
    token.string.parse().map_err(|e| ParseError::IntLit {
        span: token.span,
        value: token.string,
        error: e,
    })
}

pub fn parse_file(file: FileId, input: &str) -> Result<PackageContent> {
    let mut parser = Parser {
        tokenizer: Tokenizer::new(file, input)?,
        last_popped_end: Pos { file, line: 1, col: 1 },
        restrictions: Restrictions::NONE,
    };
    parser.main_package_content()
}

impl ExpressionKind {
    fn needs_semi(&self) -> bool {
        match self {
            ExpressionKind::Block(_) | ExpressionKind::If(_) | ExpressionKind::Loop(_) | ExpressionKind::While(_) | ExpressionKind::For(_) => false,
            _ => true,
        }
    }
}
