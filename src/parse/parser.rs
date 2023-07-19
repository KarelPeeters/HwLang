use std::mem::swap;
use std::num::ParseIntError;

use crate::parse::ast::{ArrayItem, ArrayItemKind, Assignment, BinaryOp, Block, ConstDecl, ControlFlowExpression, Declaration, Expression, ExpressionKind, FileContent, ForExpression, FuncDecl, FuncParam, GenericArgs, GenericParam, GenericParams, Identifier, IfExpression, IntPattern, Item, LoopExpression, MaybeIdentifier, Path, Signed, SizedIntSize, SizedIntType, Spanned, Statement, StatementKind, StructDecl, StructField, StructLiteral, Type, TypeAlias, TypeKind, UnaryOp, UseDecl, WhileExpression};
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

    UInt("uint"),
    Int("int"),
    Bool("bool"),

    True("true"),
    False("false"),

    Use("use"),
    Type("type"),
    Struct("struct"),
    Fn("fn"),
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
    Arrow("->"),
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
    allow_chain: bool,
    op: BinaryOp,
}

// Order of operations inspired by
// * Rust: https://doc.rust-lang.org/reference/expressions.html#expression-precedence
// * C++: https://en.cppreference.com/w/cpp/language/operator_precedence
// * Java: https://docs.oracle.com/javase/tutorial/java/nutsandbolts/operators.html
const BINARY_OPERATOR_INFO: &[BinOpInfo] = &[
    // logical
    BinOpInfo { level: 1, token: TT::DoublePipe, allow_chain: true, op: BinaryOp::BoolOr },
    BinOpInfo { level: 2, token: TT::DoubleAmpersand, allow_chain: true, op: BinaryOp::BoolAnd },
    // comparison
    BinOpInfo { level: 3, token: TT::DoubleEq, allow_chain: false, op: BinaryOp::CmpEq },
    BinOpInfo { level: 3, token: TT::BangEq, allow_chain: false, op: BinaryOp::CmpNeq },
    BinOpInfo { level: 3, token: TT::GreaterEqual, allow_chain: false, op: BinaryOp::CmpGte },
    BinOpInfo { level: 3, token: TT::Greater, allow_chain: false, op: BinaryOp::CmpGt },
    BinOpInfo { level: 3, token: TT::LessEqual, allow_chain: false, op: BinaryOp::CmpLte },
    BinOpInfo { level: 3, token: TT::Less, allow_chain: false, op: BinaryOp::CmpLt },
    // bitwise
    BinOpInfo { level: 4, token: TT::Pipe, allow_chain: true, op: BinaryOp::BitOr },
    BinOpInfo { level: 5, token: TT::Hat, allow_chain: true, op: BinaryOp::BitXor },
    BinOpInfo { level: 6, token: TT::Ampersand, allow_chain: true, op: BinaryOp::BitAnd },
    // add/sub
    BinOpInfo { level: 7, token: TT::Plus, allow_chain: true, op: BinaryOp::Add },
    BinOpInfo { level: 7, token: TT::Minus, allow_chain: true, op: BinaryOp::Sub },
    // mul/div/mod
    BinOpInfo { level: 8, token: TT::Star, allow_chain: true, op: BinaryOp::Mul },
    BinOpInfo { level: 8, token: TT::Slash, allow_chain: true, op: BinaryOp::Div },
    BinOpInfo { level: 8, token: TT::Percent, allow_chain: true, op: BinaryOp::Mod },
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

const POSTFIX_DEFAULT_LEVEL: u8 = 3;
const POSTFIX_CAST_LEVEL: u8 = 1;

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
            PostFixStateKind::Cast { ty } =>
                ExpressionKind::Cast(inner, ty),
            PostFixStateKind::ArrayIndex { index } =>
                ExpressionKind::ArrayIndex(inner, index),
            PostFixStateKind::DotIdIndex { index } =>
                ExpressionKind::DotIdIndex(inner, index),
            PostFixStateKind::DotIntIndex { span, index } =>
                ExpressionKind::DotIntIndex(inner, Spanned { span, inner: index }),
        };

        Expression { span, kind }
    }
}

enum PostFixStateKind {
    Call { args: Vec<Expression> },
    ArrayIndex { index: Box<Expression> },
    DotIdIndex { index: Identifier },
    DotIntIndex { span: Span, index: u32 },
    Cast { ty: Type },
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
    ) -> Result<(Span, Vec<A>)> {
        let mut result = Vec::new();
        let start_pos = self.peek().span.start;
        let mut end_pos = start_pos;

        while self.accept(end)?.is_none() {
            result.push(item(self)?);
            end_pos = self.last_popped_end;

            if self.accept(end)?.is_some() { break; }

            if let Some(sep) = sep {
                self.expect(sep, "separator")?;
            }
        }

        Ok((Span::new(start_pos, end_pos), result))
    }
}

impl<'s> Parser<'s> {
    fn restrict<T>(&mut self, res: Restrictions, f: impl FnOnce(&mut Self) -> T) -> T {
        let old = std::mem::replace(&mut self.restrictions, res);
        let result = f(self);
        self.restrictions = old;
        result
    }

    fn unrestrict<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.restrict(Restrictions::NONE, f)
    }

    fn file(&mut self) -> Result<FileContent> {
        let start = self.last_popped_end;
        let (_, items) = self.list(TT::Eof, None, Self::item)?;
        let span = Span::new(start, self.last_popped_end);
        Ok(FileContent { span, items })
    }

    fn item(&mut self) -> Result<Item> {
        let token = self.peek();

        match token.ty {
            TT::Struct => self.struct_().map(Item::Struct),
            TT::Const => self.const_().map(Item::Const),
            TT::Use => self.use_decl().map(Item::Use),
            TT::Type => self.type_alias().map(Item::Type),
            TT::Fn => self.function().map(Item::Func),
            _ => Err(Self::unexpected_token(token, &[TT::Struct, TT::Const, TT::Use, TT::Type, TT::Fn], "start of item"))
        }
    }

    fn maybe_generic_params(&mut self) -> Result<GenericParams> {
        let start = self.peek().span.start;

        let params = if self.accept(TT::OpenS)?.is_some() {
            self.list(TT::CloseS, Some(TT::Comma), Self::generic_param)?.1
        } else {
            vec![]
        };

        let span = Span::new(start, self.last_popped_end);
        Ok(GenericParams { span, params })
    }

    fn generic_param(&mut self) -> Result<GenericParam> {
        let start = self.peek().span.start;
        let id = self.identifier("generic parameter name")?;

        let bound = if self.accept(TT::Colon)?.is_some() {
            Some(self.type_()?)
        } else {
            None
        };

        let span = Span::new(start, self.last_popped_end);
        Ok(GenericParam { span, id, bound })
    }

    fn maybe_generic_args(&mut self) -> Result<GenericArgs> {
        let start = self.peek().span.start;

        if self.accept(TT::OpenS)?.is_some() {
            let mut un_named = vec![];
            let mut named = vec![];

            let _ = self.list(TT::CloseS, Some(TT::Comma), |s| {
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
                    un_named.push(value);
                }

                Ok(())
            })?;

            let span = Span::new(start, self.last_popped_end);
            Ok(GenericArgs { span, un_named, named })
        } else {
            let span = Span::empty_at(self.last_popped_end);
            Ok(GenericArgs { span, un_named: vec![], named: vec![] })
        }
    }

    fn const_(&mut self) -> Result<ConstDecl> {
        let start = self.pop()?;
        let id = self.identifier("const name")?;
        self.expect(TT::Colon, "const type")?;
        let ty = self.type_()?;
        self.expect(TT::Eq, "const initializer")?;
        let init = self.expression()?;
        self.expect(TT::Semi, "const end")?;

        let span = Span::new(start.span.start, self.last_popped_end);
        Ok(ConstDecl { span, id, ty, init })
    }

    fn use_decl(&mut self) -> Result<UseDecl> {
        let start_pos = self.expect(TT::Use, "start of use decl")?.span.start;
        let path = self.path()?;
        let as_ = self.accept(TT::As)?.map(|_| self.identifier("as name")).transpose()?;
        self.expect(TT::Semi, "end of item")?;

        let span = Span::new(start_pos, path.span.end);
        Ok(UseDecl { span, path, as_ })
    }

    fn type_alias(&mut self) -> Result<TypeAlias> {
        let start_pos = self.expect(TT::Type, "start of type alias")?.span.start;
        let id = self.identifier("type alias name")?;
        let gen_params = self.maybe_generic_params()?;
        self.expect(TT::Eq, "type alias equal sign")?;
        let ty = self.type_()?;
        self.expect(TT::Semi, "type alias end")?;

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(TypeAlias { span, id, gen_params, ty })
    }

    fn struct_(&mut self) -> Result<StructDecl> {
        let start = self.expect(TT::Struct, "start of struct declaration")?.span.start;
        let id = self.identifier("struct name")?;
        let gen_params = self.maybe_generic_params()?;
        self.expect(TT::OpenC, "start of struct fields")?;
        let (_, fields) = self.list(TT::CloseC, Some(TT::Comma), Self::struct_field)?;
        let span = Span::new(start, self.last_popped_end);
        Ok(StructDecl { span, id, gen_params, fields })
    }

    fn struct_field(&mut self) -> Result<StructField> {
        let id = self.identifier("field name")?;
        self.expect(TT::Colon, "field type")?;
        let ty = self.type_()?;

        let span = Span::new(id.span.start, ty.span.end);
        Ok(StructField { span, id, ty })
    }

    fn function(&mut self) -> Result<FuncDecl> {
        let start_pos = self.peek().span.start;

        self.expect(TT::Fn, "function declaration")?;
        let id = self.identifier("function name")?;
        let gen_params = self.maybe_generic_params()?;
        self.expect(TT::OpenB, "start of parameters")?;
        let (_, params) = self.list(TT::CloseB, Some(TT::Comma), Self::func_param)?;

        let ret_ty = if self.accept(TT::Arrow)?.is_some() {
            Some(self.type_()?)
        } else {
            None
        };

        let body = self.block()?;

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(FuncDecl { span, id, gen_params, params, ret_ty, body })
    }

    fn func_param(&mut self) -> Result<FuncParam> {
        let start = self.peek().span.start;
        let id = self.maybe_identifier("parameter name")?;
        self.expect(TT::Colon, "parameter type")?;
        let ty = self.type_()?;

        let span = Span::new(start, ty.span.end);
        Ok(FuncParam { span, id, ty })
    }

    fn block(&mut self) -> Result<Block> {
        let start_pos = self.expect(TT::OpenC, "start of block")?.span.start;

        // TODO better statement parsing
        let mut must_be_last = false;

        let (span, statements) = self.list(TT::CloseC, None, |s| s.statement(&mut must_be_last))?;

        Ok(Block { span: Span::new(start_pos, span.end), statements })
    }

    fn statement(&mut self, must_be_last: &mut bool) -> Result<Statement> {
        let token = self.peek();
        let start_pos = token.span.start;

        if *must_be_last {
            return Err(ParseError::ExpectedEndOfBlock { pos: start_pos });
        }

        let (kind, need_semi) = match token.ty {
            TT::Let => {
                //declaration
                let decl = self.variable_declaration(TT::Let)?;
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

    fn variable_declaration(&mut self, ty: TT) -> Result<Declaration> {
        let start_pos = self.expect(ty, "variable declaration")?.span.start;
        let id = self.maybe_identifier("variable name")?;

        let ty = self.maybe_type_decl()?;
        let init = self.accept(TT::Eq)?
            .map(|_| self.expression_boxed())
            .transpose()?;

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(Declaration { span, id, ty, init })
    }

    fn expression_boxed(&mut self) -> Result<Box<Expression>> {
        Ok(Box::new(self.expression()?))
    }

    fn expression(&mut self) -> Result<Expression> {
        let expr = self.precedence_climb_binop(0, true)?;
        let start = expr.span.start;

        if !self.restrictions.no_ternary && self.accept(TT::QuestionMark)?.is_some() {
            let then_value = self.expression()?;
            self.expect(TT::Colon, "continue ternary expression")?;
            let else_value = self.expression()?;

            let kind = ExpressionKind::TernaryOp(
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

    fn precedence_climb_binop(&mut self, level: u8, allow_chain: bool) -> Result<Expression> {
        let mut curr = self.unary()?;

        loop {
            let token = self.peek();
            let info = BINARY_OPERATOR_INFO.iter()
                .find(|i| i.token == token.ty);

            if let Some(info) = info {
                if info.level == level && !allow_chain {
                    return Err(ParseError::CannotChainOperator { span: token.span });
                }
                if info.level <= level {
                    break;
                }

                self.pop()?;

                let right = self.precedence_climb_binop(info.level, info.allow_chain)?;

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
                    self.pop()?;
                    let (_, args) = self.list(TT::CloseB, Some(TT::Comma), Self::expression)?;

                    (POSTFIX_DEFAULT_LEVEL, PostFixStateKind::Call { args })
                }
                TT::OpenS => {
                    //array indexing
                    self.pop()?;
                    let index = self.expression_boxed()?;
                    self.expect(TT::CloseS, "")?;

                    (POSTFIX_DEFAULT_LEVEL, PostFixStateKind::ArrayIndex { index })
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

                    (POSTFIX_DEFAULT_LEVEL, kind)
                }
                TT::As => {
                    //casting
                    self.pop()?;
                    let ty = self.type_()?;

                    (POSTFIX_CAST_LEVEL, PostFixStateKind::Cast { ty })
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
            TT::IntLiteralDec | TT::IntPatternBin | TT::IntPatternHex => {
                let value = self.int_pattern()?;
                ExpressionKind::IntPattern(value.inner)
            }
            TT::True | TT::False => {
                let token = self.pop()?;
                ExpressionKind::BoolLiteral(token.string.parse().expect("TTs should parse correctly"))
            }
            TT::StringLiteral => {
                let token = self.pop()?;
                ExpressionKind::StringLiteral(token.string)
            }
            TT::Id => {
                let path = self.path()?;

                if !self.restrictions.no_struct_literal && self.at(TT::OpenC) {
                    return self.struct_literal(path);
                }

                ExpressionKind::Path(path)
            }
            TT::If => {
                self.pop()?;
                let cond = self.restrict(Restrictions::NO_STRUCT_LITERAL, |s| s.expression_boxed())?;
                let then_block = self.block()?;

                let else_block = self.accept(TT::Else)?
                    .map(|_| self.block())
                    .transpose()?;

                ExpressionKind::ControlFlow(ControlFlowExpression::If(IfExpression {
                    cond,
                    then_block,
                    else_block,
                }))
            }
            TT::Loop => {
                self.pop()?;
                let body = self.block()?;
                ExpressionKind::ControlFlow(ControlFlowExpression::Loop(LoopExpression { body }))
            }
            TT::While => {
                self.pop()?;
                let cond = self.restrict(Restrictions::NO_STRUCT_LITERAL, |s| s.expression_boxed())?;
                let body = self.block()?;
                ExpressionKind::ControlFlow(ControlFlowExpression::While(WhileExpression { cond, body }))
            }
            TT::For => {
                self.pop()?;

                let index = self.maybe_identifier("index variable")?;
                let index_ty = self.maybe_type_decl()?;

                self.expect(TT::In, "in")?;
                let start = self.expression_boxed()?;
                self.expect(TT::DoubleDot, "range separator")?;

                let end = self.restrict(Restrictions::NO_STRUCT_LITERAL, |s| {
                    s.expression_boxed()
                })?;

                let body = self.block()?;

                ExpressionKind::ControlFlow(ControlFlowExpression::For(ForExpression {
                    index,
                    index_ty,
                    start,
                    end,
                    body,
                }))
            }
            TT::OpenB => {
                self.pop()?;
                // TODO shouldn't many more things call unrestrict?
                let inner = self.unrestrict(|s| s.expression_boxed())?;
                self.expect(TT::CloseB, "closing parenthesis")?;
                ExpressionKind::Wrapped(inner)
            }
            TT::OpenC => {
                let block = self.unrestrict(|s| s.block())?;
                ExpressionKind::Block(block)
            }
            TT::OpenS => {
                self.pop()?;
                let (_, values) = self.list(TT::CloseS, Some(TT::Comma), |s| s.array_item())?;
                ExpressionKind::ArrayLiteral(values)
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

    // TODO allow repetition in array literal?
    fn array_item(&mut self) -> Result<ArrayItem> {
        let start_pos = self.peek().span.start;

        let spread = self.accept(TT::Star)?.is_some();
        let value = self.expression_boxed()?;

        let kind = if spread {
            ArrayItemKind::Spread(value)
        } else {
            ArrayItemKind::Value(value)
        };

        let span = Span::new(start_pos, self.last_popped_end);
        Ok(ArrayItem { span, kind })
    }

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

    fn struct_literal(&mut self, path: Path) -> Result<Expression> {
        self.expect(TT::OpenC, "start of struct literal")?;

        let (_, fields) = self.list(TT::CloseC, Some(TT::Comma), |s| {
            let id = s.identifier("tuple literal field")?;
            s.expect(TT::Colon, "tuple literal field separator")?;

            let value = s.restrict(Restrictions::NO_TERNARY, |s| {
                s.expression()
            })?;

            Ok((id, value))
        })?;

        let span = Span::new(path.span.start, self.last_popped_end);
        let kind = ExpressionKind::StructLiteral(StructLiteral {
            struct_path: path,
            fields,
        });
        Ok(Expression { span, kind })
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

    fn maybe_type_decl(&mut self) -> Result<Option<Type>> {
        self.accept(TT::Colon)?
            .map(|_| self.type_())
            .transpose()
    }

    fn maybe_int_ty(&mut self) -> Result<Option<Type>> {
        // already parsed as path types: U[N], I[N], B[N]
        // already parsed as single-tokens: uint, int

        let curr = self.peek();
        let next = self.lookahead();
        let start_pos = curr.span.start;

        if curr.ty == TT::Id {
            let signed = match curr.string.as_str() {
                "b" => Signed::Bit,
                "u" => Signed::Unsigned,
                "i" => Signed::Signed,
                _ => return Ok(None),
            };

            let size = if next.ty == TT::IntLiteralDec {
                // u32
                let size = match next.string.parse::<u32>() {
                    Ok(size) => size,
                    Err(_) => return Ok(None),
                };

                self.pop()?;
                self.pop()?;

                SizedIntSize::Literal(size)
            } else if next.ty == TT::OpenS {
                // u[N]
                self.pop()?;
                self.pop()?;
                let size = self.expression_boxed()?;
                self.expect(TT::CloseS, "int type size close bracket")?;

                SizedIntSize::Expression(size)
            } else {
                return Ok(None);
            };

            let kind = TypeKind::SizedInt(SizedIntType { signed, size });
            let span = Span::new(start_pos, self.last_popped_end);
            return Ok(Some(Type { span, kind }));
        }

        return Ok(None);
    }

    fn type_(&mut self) -> Result<Type> {
        let start_pos = self.peek().span.start;

        if let Some(ty) = self.maybe_int_ty()? {
            return Ok(ty);
        }

        match self.peek().ty {
            TT::Underscore => Ok(Type { span: self.pop()?.span, kind: TypeKind::Wildcard }),

            TT::UInt => Ok(Type { span: self.pop()?.span, kind: TypeKind::Int(Signed::Unsigned) }),
            TT::Int => Ok(Type { span: self.pop()?.span, kind: TypeKind::Int(Signed::Signed) }),
            TT::Bool => Ok(Type { span: self.pop()?.span, kind: TypeKind::Bool }),

            TT::Id => {
                let path = self.path()?;
                let gen_args = self.maybe_generic_args()?;
                Ok(Type {
                    span: path.span,
                    kind: TypeKind::Path(path, gen_args),
                })
            }
            TT::OpenB => {
                //func or tuple
                self.pop()?;
                let (_, list) = self.list(TT::CloseB, Some(TT::Comma), Self::type_)?;

                let kind = if self.accept(TT::Arrow)?.is_some() {
                    let ret = self.type_()?;
                    TypeKind::Func(list, Box::new(ret))
                } else {
                    TypeKind::Tuple(list)
                };

                Ok(Type {
                    span: Span::new(start_pos, self.last_popped_end),
                    kind,
                })
            }
            TT::OpenS => {
                //array
                self.pop()?;
                let inner = self.type_()?;
                self.expect(TT::Semi, "array type delimiter")?;
                let len = self.expression_boxed()?;
                self.expect(TT::CloseS, "end of array type")?;

                Ok(Type {
                    span: Span::new(start_pos, self.last_popped_end),
                    kind: TypeKind::Array(Box::new(inner), len),
                })
            }
            // TODO collect type start tokens
            _ => Err(Self::unexpected_token(self.peek(), &[], "type declaration")),
        }
    }
}

fn parse_int_literal(token: Token) -> Result<u32> {
    token.string.parse().map_err(|e| ParseError::IntLit {
        span: token.span,
        value: token.string,
        error: e,
    })
}

pub fn parse_file(file: FileId, input: &str) -> Result<FileContent> {
    let mut parser = Parser {
        tokenizer: Tokenizer::new(file, input)?,
        last_popped_end: Pos { file, line: 1, col: 1 },
        restrictions: Restrictions::NONE,
    };
    parser.file()
}

impl ExpressionKind {
    fn needs_semi(&self) -> bool {
        match self {
            ExpressionKind::Block(_) | ExpressionKind::ControlFlow(_) => false,
            _ => true,
        }
    }
}
