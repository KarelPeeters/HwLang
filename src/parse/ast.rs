use crate::parse::pos::Span;

#[derive(Debug)]
pub struct PackageContent {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug)]
pub enum Item {
    Use(ItemUse),
    Package(ItemDefPackage),
    Const(ItemDefConst),
    Type(ItemDefType),
    Struct(ItemDefStruct),
    Enum(ItemDefEnum),
    Func(ItemDefFunc),
    Interface(ItemDefInterface),
}

#[derive(Debug)]
pub struct ItemUse {
    pub span: Span,
    pub path: Path,
    pub as_: Option<Identifier>,
}

#[derive(Debug)]
pub struct ItemDefPackage {
    pub span: Span,
    pub name: MaybeIdentifier,
    pub content: PackageContent,
}

#[derive(Debug)]
pub struct ItemDefConst {
    pub span: Span,
    pub id: Identifier,
    pub ty: Expression,
    pub value: Option<Expression>,
}

#[derive(Debug)]
pub struct ItemDefType {
    pub span: Span,
    pub id: Identifier,
    pub params: Params,
    pub inner: Option<Box<Expression>>,
}

// TODO allow "if" in a bunch of places? eg. struct fields
#[derive(Debug)]
pub struct ItemDefStruct {
    pub span: Span,
    pub id: Identifier,
    pub params: Params,
    pub fields: Vec<StructField>,
}

#[derive(Debug)]
pub struct StructField {
    pub span: Span,
    pub id: Identifier,
    pub ty: Expression,
}

#[derive(Debug)]
pub struct ItemDefEnum {
    pub span: Span,
    pub options: Vec<Identifier>,
}

#[derive(Debug)]
pub struct ItemDefFunc {
    pub span: Span,
    pub id: Identifier,
    pub params: Params,
    pub ret_ty: Option<Expression>,
    pub body: Option<Block>,
}

#[derive(Debug)]
pub struct ItemDefInterface {
    pub span: Span,
    pub id: Identifier,
    // either None or non-empty
    pub modes: Option<Vec<Identifier>>,
    pub params: Params,
    pub fields: Vec<InterfaceField>,
}

#[derive(Debug)]
pub struct InterfaceField {
    pub span: Span,
    pub id: Identifier,
    pub dir: Direction,
    pub ty: Expression,
}

#[derive(Debug)]
pub struct Params {
    pub span: Span,
    pub params: Vec<Param>,
}

#[derive(Debug)]
pub struct Param {
    pub span: Span,
    pub dir: Option<Direction>,
    pub kind: ParamKind,
    pub ty: Expression,
}

#[derive(Debug)]
pub enum ParamKind {
    Anonymous,
    Named { id: Identifier, default: Option<Expression> }
}

#[derive(Debug)]
pub struct Block {
    pub span: Span,
    pub statements: Vec<Statement>,
}

#[derive(Debug)]
pub struct Statement {
    pub span: Span,
    pub kind: StatementKind,
}

#[derive(Debug)]
pub enum StatementKind {
    Declaration(Declaration),
    Assignment(Assignment),
    Expression(Box<Expression>),
    ReturnExpression(Box<Expression>),
    BreakExpression(Box<Expression>),
}

#[derive(Debug)]
pub struct Declaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub ty: Option<Box<Expression>>,
    pub init: Option<Box<Expression>>,
}

#[derive(Debug)]
pub struct Assignment {
    pub span: Span,
    pub op: Option<BinaryOp>,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

#[derive(Debug)]
pub struct Expression {
    pub span: Span,
    pub kind: ExpressionKind,
}

#[derive(Debug)]
pub enum ExpressionKind {
    // Miscellaneous
    Wildcard,
    Path(Path),
    Wrapped(Box<Expression>),
    // the special "type" type
    Type,

    // Control flow
    Block(Block),
    If(IfExpression),
    Loop(LoopExpression),
    While(WhileExpression),
    For(ForExpression),
    Sync(SyncExpression),

    Return(Option<Box<Expression>>),
    Break(Option<Box<Expression>>),
    Continue,

    // Literals
    IntPattern(IntPattern),
    BoolLiteral(bool),
    StringLiteral(String),

    // Structures
    ArrayInit(Vec<Expression>),
    TupleInit(Vec<Expression>),
    StructInit(StructLiteral),
    TypeFunc(Vec<Expression>, Box<Expression>),
    Range { inclusive: bool, start: Box<Expression>, end: Box<Expression> },

    // Operations
    UnaryOp(UnaryOp, Box<Expression>),
    BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
    TernarySelect(Box<Expression>, Box<Expression>, Box<Expression>),

    // Indexing
    ArrayIndex(Box<Expression>, Box<Expression>),
    FieldAccess(Box<Expression>, Identifier),
    DotIdIndex(Box<Expression>, Identifier),
    DotIntIndex(Box<Expression>, Spanned<u32>),

    // Calls
    Call(Box<Expression>, Args),
}

#[derive(Debug)]
pub struct Args {
    pub span: Span,
    pub positional: Vec<Expression>,
    pub named: Vec<(Identifier, Expression)>,
}

#[derive(Debug)]
pub struct StructLiteral {
    pub struct_ty: Box<Expression>,
    pub fields: Vec<StructLiteralField>,
}

#[derive(Debug)]
pub struct StructLiteralField {
    pub span: Span,
    pub id: Identifier,
    pub value: Expression,
}

#[derive(Debug)]
pub struct IfExpression {
    pub cond: Box<Expression>,
    pub then_block: Block,
    pub else_block: Option<Block>,
}

#[derive(Debug)]
pub struct LoopExpression {
    pub body: Block,
}

#[derive(Debug)]
pub struct WhileExpression {
    pub cond: Box<Expression>,
    pub body: Block,
}

#[derive(Debug)]
pub struct ForExpression {
    pub index: MaybeIdentifier,
    pub index_ty: Option<Box<Expression>>,
    pub range: Box<Expression>,
    pub body: Block,
}

#[derive(Debug)]
pub struct SyncExpression {
    pub clk: Box<Expression>,
    pub body: Block,
}

// TODO allow: undefined, wildcard, 0, 1, hex, decimal, ...
// TODO wildcard symbol: `_`, `?`, `*`, `#`?
//     `*` is a bad idea
// (don't allow any of the fancy stuff stuff for decimal ofc)
#[derive(Debug)]
pub enum IntPattern {
    // [0-9a-fA-F_]*
    Hex(String),
    // [01_]*
    Bin(String),
    // [0-9]*
    Dec(String),
}

#[derive(Debug)]
pub enum MaybeIdentifier {
    Placeholder(Span),
    Identifier(Identifier),
}

#[derive(Debug, Clone)]
pub struct Identifier {
    pub span: Span,
    pub string: String,
}

#[derive(Debug)]
pub struct Path {
    pub span: Span,
    pub parents: Vec<Identifier>,
    pub id: Identifier,
}

// TODO move to parser utilities module
pub fn build_binary_op(op: BinaryOp, left: Expression, right: Expression) -> Expression {
    Expression {
        span: Span::new(left.span.start, right.span.end),
        kind: ExpressionKind::BinaryOp(op, Box::new(left), Box::new(right)),
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,

    BitAnd,
    BitOr,
    BitXor,
    BoolAnd,
    BoolOr,

    Shl,
    Shr,

    CmpEq,
    CmpNeq,
    CmpLt,
    CmpLte,
    CmpGt,
    CmpGte,

    In,
    Range,
    RangeInclusive,
}

#[derive(Debug, Copy, Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    In,
    Out,
}

// TODO remove if unnecessary
// TODO replace Identifier with MaybeIdentifier whenever possible
#[derive(Debug)]
pub struct Spanned<T> {
    pub span: Span,
    pub inner: T,
}
