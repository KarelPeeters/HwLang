use crate::syntax::pos::Span;

// TODO remove "clone" from everything, and use ast lifetimes everywhere

#[derive(Debug, Clone)]
pub struct FileContent {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Visibility {
    Public(Span),
    Private,
}

// TODO add "doc comment" field to items?
#[derive(Debug, Clone)]
pub enum Item {
    Use(ItemUse),
    // Package(ItemDefPackage),
    Const(ItemDefConst),
    Type(ItemDefType),
    Struct(ItemDefStruct),
    Enum(ItemDefEnum),
    Function(ItemDefFunction),
    // TODO rename to "block"?
    Module(ItemDefModule),
    // TODO rename to "bus" and reserve interface for Rust trait/C++ concept/Java interface?
    Interface(ItemDefInterface),
}

// TODO split this out from the items that actually define _new_ symbols?
#[derive(Debug, Clone)]
pub struct ItemUse {
    pub span: Span,
    pub path: Path,
    pub as_: Option<MaybeIdentifier>,
}

pub enum ItemUseKind {
    Root,
}

// TODO remove
#[derive(Debug, Clone)]
pub struct ItemDefPackage {
    pub span: Span,
    pub name: MaybeIdentifier,
    pub content: FileContent,
}

#[derive(Debug, Clone)]
pub struct ItemDefConst {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub ty: Expression,
    pub value: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct ItemDefType {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParam>>>,
    pub inner: Box<Expression>,
}

// TODO allow "if" in a bunch of places? eg. struct fields
#[derive(Debug, Clone)]
pub struct ItemDefStruct {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParam>>>,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub span: Span,
    pub id: Identifier,
    pub ty: Expression,
}

// TODO proper sum type
#[derive(Debug, Clone)]
pub struct ItemDefEnum {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParam>>>,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub span: Span,
    pub id: Identifier,
    pub content: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct ItemDefFunction {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Spanned<Vec<FunctionParam>>,
    pub ret_ty: Option<Expression>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct ItemDefModule {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParam>>>,
    pub ports: Spanned<Vec<ModulePort>>,
    pub body: Block,
}

// TODO think about the syntax and meaning of this
#[derive(Debug, Clone)]
pub struct ItemDefInterface {
    pub span: Span,
    pub id: Identifier,
    pub vis: Visibility,
    // either None or non-empty
    pub modes: Option<Vec<Identifier>>,
    // TODO params?
    // pub params: Params,
    pub fields: Vec<InterfaceField>,
}

#[derive(Debug, Clone)]
pub struct InterfaceField {
    pub span: Span,
    pub id: Identifier,
    pub dir: Direction,
    pub ty: Expression,
}

#[derive(Debug, Clone)]
pub struct GenericParam {
    pub span: Span,
    pub id: Identifier,
    pub kind: GenericParamKind,
}

#[derive(Debug, Clone)]
pub enum GenericParamKind {
    Type,
    ValueOfType(Expression)
}

#[derive(Debug, Clone)]
pub struct FunctionParam {
    pub span: Span,
    // pub is_const: bool,
    pub id: Identifier,
    pub ty: Expression,
}

#[derive(Debug, Clone)]
pub struct ModulePort {
    pub span: Span,
    pub id: Identifier,
    pub direction: Spanned<PortDirection>,
    // TODO expand sync to more complicated expressions again
    pub kind: Spanned<PortKind<Spanned<SyncKind<Identifier>>, Box<Expression>>>,
}

#[derive(Debug, Clone)]
pub enum PortKind<S, T> {
    Clock,
    Normal { sync: S, ty: T },
}

#[derive(Debug, Clone)]
pub enum SyncKind<S> {
    Async,
    Sync(S),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub span: Span,
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub span: Span,
    pub kind: StatementKind,
}

#[derive(Debug, Clone)]
pub enum StatementKind {
    Declaration(Declaration),
    Assignment(Assignment),
    Expression(Box<Expression>),
    CombinatorialBlock(CombinatorialBlock),
    ClockedBlock(ClockedBlock),
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub span: Span,
    pub kind: DeclarationKind,
    pub id: MaybeIdentifier,
    pub ty: Option<Box<Expression>>,
    pub init: Option<Box<Expression>>,
}

#[derive(Debug, Copy, Clone)]
pub enum DeclarationKind {
    Const,
    Var,
    Val
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub span: Span,
    pub op: Spanned<Option<BinaryOp>>,
    pub target: Box<Expression>,
    pub value: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct CombinatorialBlock {
    pub span: Span,
    pub block: Box<Block>,
}

#[derive(Debug, Clone)]
pub struct ClockedBlock {
    pub span: Span,
    pub clock: Box<Expression>,
    pub reset: Box<Expression>,
    pub block: Box<Block>,
}

pub type Expression = Spanned<ExpressionKind>;

// TODO create separate parallel expression hierarchy, dedicated to types?
//   can they ever even mix? separating them would make future mixed system harder again though
#[derive(Debug, Clone)]
pub enum ExpressionKind {
    // Miscellaneous
    Dummy,
    Id(Identifier),
    // Wrapped just means an expression that's surrounded by parenthesis.
    // It has to be a dedicated expression to ensure it gets a separate span.
    Wrapped(Box<Expression>),

    // function type signature
    TypeFunc(Vec<Expression>, Box<Expression>),

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
    ArrayLiteral(Vec<Expression>),
    TupleLiteral(Vec<Expression>),
    StructLiteral(StructLiteral),
    RangeLiteral(RangeLiteral),

    // Operations
    UnaryOp(UnaryOp, Box<Expression>),
    BinaryOp(BinaryOp, Box<Expression>, Box<Expression>),
    TernarySelect(Box<Expression>, Box<Expression>, Box<Expression>),

    // Indexing
    ArrayIndex(Box<Expression>, Args),
    DotIdIndex(Box<Expression>, Identifier),
    DotIntIndex(Box<Expression>, Spanned<String>),

    // Calls
    Call(Box<Expression>, Args),
}

#[derive(Debug, Clone)]
pub struct Args {
    pub span: Span,
    pub inner: Vec<Expression>,
    // TODO implement named args
    // pub named: Vec<(Identifier, Expression)>,
}

#[derive(Debug, Clone)]
pub struct StructLiteral {
    pub struct_ty: Box<Expression>,
    pub fields: Vec<StructLiteralField>,
}

#[derive(Debug, Clone)]
pub struct StructLiteralField {
    pub span: Span,
    pub id: Identifier,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct RangeLiteral {
    pub end_inclusive: bool,
    pub start: Option<Box<Expression>>,
    pub end: Option<Box<Expression>>
}

#[derive(Debug, Clone)]
pub struct IfExpression {
    pub cond: Box<Expression>,
    pub then_block: Block,
    pub else_if_pairs: Vec<ElseIfPair>,
    pub else_block: Option<Block>,
}

#[derive(Debug, Clone)]
pub struct ElseIfPair {
    pub span: Span,
    pub cond: Box<Expression>,
    pub block: Block,
}

#[derive(Debug, Clone)]
pub struct LoopExpression {
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct WhileExpression {
    pub cond: Box<Expression>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct ForExpression {
    pub index: MaybeIdentifier,
    pub index_ty: Option<Box<Expression>>,
    pub iter: Box<Expression>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct SyncExpression {
    pub clk: Box<Expression>,
    pub body: Block,
}

// TODO allow: undefined, wildcard, 0, 1, hex, decimal, ...
// TODO wildcard symbol: `_`, `?`, `*`, `#`?
//     `*` is a bad idea
// (don't allow any of the fancy stuff stuff for decimal ofc)
#[derive(Debug, Clone)]
pub enum IntPattern {
    // [0-9a-fA-F_]+
    Hex(String),
    // [01_]+
    Bin(String),
    // [0-9_]+
    Dec(String),
}

#[derive(Debug, Clone)]
pub enum MaybeIdentifier {
    Dummy(Span),
    Identifier(Identifier),
}

// TODO this is also just a spanned string
// TODO do we want to commit to identifier definition uniqueness
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Identifier {
    pub span: Span,
    pub string: String,
}

#[derive(Debug, Clone)]
pub struct Path {
    pub span: Span,
    pub steps: Vec<Identifier>,
    pub id: Identifier,
}

// TODO move to parser utilities module
pub fn build_binary_op(op: BinaryOp, left: Expression, right: Expression) -> ExpressionKind {
    ExpressionKind::BinaryOp(op, Box::new(left), Box::new(right))
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
}

#[derive(Debug, Copy, Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Copy, Clone)]
pub enum Direction {
    None,
    In,
    Out,
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub span: Span,
    pub inner: T,
}

impl<T> Spanned<T> {
    pub fn map_inner<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            span: self.span,
            inner: f(self.inner),
        }
    }
}

impl Item {
    pub fn id_vis(&self) -> (MaybeIdentifier, &Visibility) {
        match self {
            Item::Use(item) => {
                let id = match &item.as_ {
                    None => MaybeIdentifier::Identifier(item.path.id.clone()),
                    Some(as_) => as_.clone(),
                };
                (id, &Visibility::Private)
            },
            Item::Const(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
            Item::Type(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
            Item::Struct(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
            Item::Enum(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
            Item::Function(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
            Item::Module(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
            Item::Interface(item) => (MaybeIdentifier::Identifier(item.id.clone()), &item.vis),
        }
    }
    
    pub fn span(&self) -> Span {
        match self {
            Item::Use(item) => item.span,
            Item::Const(item) => item.span,
            Item::Type(item) => item.span,
            Item::Struct(item) => item.span,
            Item::Enum(item) => item.span,
            Item::Function(item) => item.span,
            Item::Module(item) => item.span,
            Item::Interface(item) => item.span,
        }
    }
}
