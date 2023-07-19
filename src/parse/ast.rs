use crate::parse::pos::Span;

// TODO remove?
#[derive(Debug)]
pub struct Spanned<T> {
    pub span: Span,
    pub inner: T,
}

#[derive(Debug)]
pub struct FileContent {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug)]
pub enum Item {
    Use(UseDecl),
    Type(TypeAlias),
    Struct(StructDecl),
    Func(FuncDecl),
    Mod(ModuleDecl),
    Const(ConstDecl),
}

#[derive(Debug)]
pub struct UseDecl {
    pub span: Span,
    pub path: Path,
    pub as_: Option<Identifier>,
}

#[derive(Debug)]
pub struct FuncDecl {
    pub span: Span,
    pub id: Identifier,
    pub gen_params: GenericParams,
    pub params: Vec<FuncParam>,
    pub ret_ty: Option<Type>,
    pub body: Block,
}

#[derive(Debug)]
pub struct FuncParam {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub ty: Type,
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
}

#[derive(Debug)]
pub struct Declaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub ty: Option<Type>,
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
pub struct ModuleDecl {
    // TODO
}

// TODO allow "if" in a bunch of places? eg. struct fields
#[derive(Debug)]
pub struct StructDecl {
    pub span: Span,
    pub id: Identifier,
    pub gen_params: GenericParams,
    pub fields: Vec<StructField>,
}

#[derive(Debug)]
pub struct StructField {
    pub span: Span,
    pub id: Identifier,
    pub ty: Type,
}

#[derive(Debug)]
pub struct TypeAlias {
    pub span: Span,
    pub id: Identifier,
    pub gen_params: GenericParams,
    pub ty: Type,
}

#[derive(Debug)]
pub struct ConstDecl {
    pub span: Span,
    pub id: Identifier,
    pub ty: Type,
    pub init: Expression,
}

#[derive(Debug)]
pub struct Bound {
    pub span: Span,
    pub kind: BoundKind,
}

#[derive(Debug)]
pub enum BoundKind {
    TypeType(Span),
    Type(Type),
}

#[derive(Debug)]
pub struct Type {
    pub span: Span,
    pub kind: TypeKind,
}

#[derive(Debug)]
pub enum TypeKind {
    Wildcard,
    Path(Path, GenericArgs),

    Bool,
    Int(Signed),
    SizedInt(SizedIntType),

    Tuple(Vec<Type>),
    Array(Box<Type>, Box<Expression>),

    Func(Vec<Type>, Box<Type>),
}

#[derive(Debug)]
pub struct SizedIntType {
    pub signed: Signed,
    pub size: SizedIntSize,

    // TODO min/max args?
    // gen_args: GenericArgs,
}

#[derive(Debug)]
pub enum SizedIntSize {
    Literal(u32),
    Expression(Box<Expression>),
}

#[derive(Debug)]
pub enum Signed {
    Bit,
    Unsigned,
    Signed,
}

#[derive(Debug)]
pub enum FixedIntSize {
    Literal(u32),
    Expression(Box<Expression>),
}

#[derive(Debug)]
pub struct Expression {
    pub span: Span,
    pub kind: ExpressionKind,
}

#[derive(Debug)]
pub enum ExpressionKind {
    Block(Block),
    ControlFlow(ControlFlowExpression),

    Path(Path),
    Wrapped(Box<Expression>),

    // TODO allow specifying type?
    // TODO only allow int literals for int types, and only allow binary literals for bit types?
    IntPattern(IntPattern),
    BoolLiteral(bool),
    StringLiteral(String),

    // TODO array vs concat? are unsigned ints really just arrays?
    ArrayLiteral(Vec<ArrayItem>),
    StructLiteral(StructLiteral),

    UnaryOp(UnaryOp, Box<Expression>),
    BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
    TernaryOp(Box<Expression>, Box<Expression>, Box<Expression>),

    Call(Box<Expression>, Vec<Expression>),
    Cast(Box<Expression>, Type),

    ArrayIndex(Box<Expression>, Box<Expression>),
    FieldAccess(Box<Expression>, Identifier),
    DotIdIndex(Box<Expression>, Identifier),
    DotIntIndex(Box<Expression>, Spanned<u32>),

    Return(Option<Box<Expression>>),
    Break(Option<Box<Expression>>),
    Continue,
}

#[derive(Debug)]
pub enum ControlFlowExpression {
    If(IfExpression),
    Match(MatchExpression),
    Loop(LoopExpression),
    While(WhileExpression),
    For(ForExpression),
}

#[derive(Debug)]
pub struct ArrayItem {
    pub span: Span,
    pub kind: ArrayItemKind,
}

#[derive(Debug)]
pub enum ArrayItemKind {
    Value(Box<Expression>),
    Spread(Box<Expression>),
}

#[derive(Debug)]
pub struct StructLiteral {
    pub struct_path: Path,
    pub fields: Vec<(Identifier, Expression)>,
}

#[derive(Debug)]
pub struct IfExpression {
    pub cond: Box<Expression>,
    pub then_block: Block,
    pub else_block: Option<Block>,
}

#[derive(Debug)]
pub struct MatchExpression {
    pub expr: Box<Expression>,
    pub arms: Vec<MatchArm>,
}

#[derive(Debug)]
pub struct MatchArm {
    // TODO
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
    pub index_ty: Option<Type>,
    pub start: Box<Expression>,
    pub end: Box<Expression>,
    pub body: Block,
}

// TODO allow: undefined, wildcard, 0, 1, hex, decimal, ...
// (don't allow any of the fancy stuff stuff for decimal ofc)
#[derive(Debug)]
pub enum IntPattern {
    // [0-9], [a-f], _
    Hex(String),
    // [0-1], _
    Bin(String),
    // [0-9]
    Dec(String),
}

#[derive(Debug)]
pub struct GenericParams {
    pub span: Span,
    pub params: Vec<GenericParam>,
}

#[derive(Debug)]
pub struct GenericParam {
    pub span: Span,
    pub id: Identifier,
    pub bound: Option<Type>,
}

#[derive(Debug)]
pub struct GenericArgs {
    pub span: Span,
    pub un_named: Vec<Expression>,
    pub named: Vec<(Identifier, Expression)>
}

#[derive(Debug)]
pub enum MaybeIdentifier {
    Placeholder(Span),
    Identifier(Identifier),
}

#[derive(Debug)]
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

#[derive(Debug, Copy, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,

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
}

#[derive(Debug, Copy, Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}
