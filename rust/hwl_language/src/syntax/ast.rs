use crate::syntax::pos::Span;
use itertools::Itertools;

// TODO remove "clone" from everything, and use ast lifetimes everywhere

#[derive(Debug, Clone)]
pub struct FileContent {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug, Copy, Clone)]
pub enum Visibility {
    Public(Span),
    Private,
}

// TODO add "doc comment" field to items?
// TODO rename to ItemDef
#[derive(Debug, Clone)]
pub enum Item {
    Import(ItemImport),
    // Package(ItemDefPackage),
    Const(ConstDeclaration<Visibility>),
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
pub struct ItemImport {
    pub span: Span,
    pub parents: Spanned<Vec<Identifier>>,
    pub entry: Spanned<ImportFinalKind>,
}

#[derive(Debug, Clone)]
pub enum ImportFinalKind {
    Single(ImportEntry),
    Multi(Vec<ImportEntry>),
}

#[derive(Debug, Clone)]
pub struct ImportEntry {
    pub span: Span,
    pub id: Identifier,
    pub as_: Option<MaybeIdentifier>,
}

// TODO remove
#[derive(Debug, Clone)]
pub struct ItemDefPackage {
    pub span: Span,
    pub name: MaybeIdentifier,
    pub content: FileContent,
}

#[derive(Debug, Clone)]
pub struct ItemDefType {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParameter>>>,
    pub inner: Box<Expression>,
}

// TODO allow "if" in a bunch of places? eg. struct fields
#[derive(Debug, Clone)]
pub struct ItemDefStruct {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParameter>>>,
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
    pub params: Option<Spanned<Vec<GenericParameter>>>,
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
    /// All function parameters are "generic", which means they can be types.
    /// It doesn't make sense to force a distinction similar to modules.
    pub params: Spanned<Vec<GenericParameter>>,
    // TODO should the return type by "generic" too, ie. should functions be allowed to return types?
    pub ret_ty: Option<Expression>,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ItemDefModule {
    pub span: Span,
    pub vis: Visibility,
    pub id: Identifier,
    pub params: Option<Spanned<Vec<GenericParameter>>>,
    pub ports: Spanned<Vec<ModulePortItem>>,
    pub body: Block<ModuleStatement>,
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
    pub dir: InterfaceDirection,
    pub ty: Expression,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GenericParameter {
    pub span: Span,
    pub id: Identifier,
    pub ty: Expression,
}

#[derive(Debug, Clone)]
pub enum ModulePortItem {
    Single(ModulePortSingle),
    Block(ModulePortBlock),
}

#[derive(Debug, Clone)]
pub struct ModulePortSingle {
    pub span: Span,
    pub id: Identifier,
    pub direction: Spanned<PortDirection>,
    pub kind: Spanned<PortKind<Spanned<DomainKind<Box<Expression>>>, Box<Expression>>>,
}

#[derive(Debug, Clone)]
pub struct ModulePortBlock {
    pub span: Span,
    pub domain: Spanned<DomainKind<Box<Expression>>>,
    pub ports: Vec<ModulePortInBlock>,
}

#[derive(Debug, Clone)]
pub struct ModulePortInBlock {
    pub span: Span,
    pub id: Identifier,
    pub direction: Spanned<PortDirection>,
    pub ty: Box<Expression>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PortKind<S, T> {
    Clock,
    Normal { domain: S, ty: T },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DomainKind<S> {
    Async,
    Sync(SyncDomain<S>),
}

// TODO how to represent the difference between sync and async reset?
//   this is not the same as the sync/async-ness of the reset itself! (or is it?)
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SyncDomain<S> {
    pub clock: S,
    // TODO make reset optional
    pub reset: S,
}

impl<S> SyncDomain<S> {
    pub fn as_ref(&self) -> SyncDomain<&S> {
        SyncDomain {
            clock: &self.clock,
            reset: &self.reset,
        }
    }

    pub fn map_inner<U>(self, mut f: impl FnMut(S) -> U) -> SyncDomain<U> {
        SyncDomain {
            clock: f(self.clock),
            reset: f(self.reset),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PortDirection {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub struct Block<S> {
    pub span: Span,
    pub statements: Vec<S>,
}

pub type ModuleStatement = Spanned<ModuleStatementKind>;
pub type BlockStatement = Spanned<BlockStatementKind>;

#[derive(Debug, Clone)]
pub enum ModuleStatementKind {
    // declarations
    ConstDeclaration(ConstDeclaration<()>),
    RegDeclaration(RegDeclaration),
    WireDeclaration(WireDeclaration),

    // marker
    RegOutPortMarker(RegOutPortMarker),

    // blocks
    CombinatorialBlock(CombinatorialBlock),
    ClockedBlock(ClockedBlock),
    Instance(ModuleInstance),

    // TODO control flow (if, for), probably not while/break/continue
    // TODO allow function/type/module(?) definitions in blocks
}

#[derive(Debug, Clone)]
pub enum BlockStatementKind {
    ConstDeclaration(ConstDeclaration<()>),
    VariableDeclaration(VariableDeclaration),
    Assignment(Assignment),
    Expression(Box<Expression>),

    // control flow
    Block(Block<BlockStatement>),
    If(IfStatement),
    While(WhileStatement),
    For(ForStatement),
    // control flow terminators
    Return(Option<Box<Expression>>),
    Break(Option<Box<Expression>>),
    Continue,

    // TODO allow function/type definitions in blocks
}

#[derive(Debug, Clone)]
pub struct IfStatement {
    pub cond: Box<Expression>,
    pub then_block: Block<BlockStatement>,
    pub else_if_pairs: Vec<ElseIfPair>,
    pub else_block: Option<Block<BlockStatement>>,
}

#[derive(Debug, Clone)]
pub struct ElseIfPair {
    pub span: Span,
    pub cond: Box<Expression>,
    pub block: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct WhileStatement {
    pub cond: Box<Expression>,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ForStatement {
    pub index: MaybeIdentifier,
    pub index_ty: Option<Box<Expression>>,
    pub iter: Box<Expression>,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct RegOutPortMarker {
    pub span: Span,
    pub id: Identifier,
    pub init: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct RegDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    // TODO make optional and infer
    pub sync: Spanned<SyncDomain<Box<Expression>>>,
    pub ty: Box<Expression>,
    pub init: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct WireDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    // TODO make optional and infer
    pub sync: Spanned<DomainKind<Box<Expression>>>,
    pub ty: Box<Expression>,
    pub value: Option<Box<Expression>>,
}

#[derive(Debug, Clone)]
pub struct ConstDeclaration<V> {
    pub span: Span,
    pub vis: V,
    pub id: MaybeIdentifier,
    pub ty: Option<Box<Expression>>,
    pub value: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub span: Span,
    pub mutable: bool,
    pub id: MaybeIdentifier,
    pub ty: Option<Box<Expression>>,
    pub init: Box<Expression>,
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
    pub span_keyword: Span,
    pub block: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ClockedBlock {
    pub span: Span,
    pub span_keyword: Span,
    pub domain: Spanned<SyncDomain<Box<Expression>>>,
    pub block: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ModuleInstance {
    pub span: Span,
    pub span_keyword: Span,
    pub name: Option<Identifier>,
    pub module: Box<Expression>,
    pub generic_args: Option<Args>,
    pub port_connections: Spanned<Vec<(Identifier, Spanned<Option<Expression>>)>>,
}

pub type Expression = Spanned<ExpressionKind>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ExpressionKind {
    // Miscellaneous
    Dummy,
    Undefined,

    /// Wrapped just means an expression that's surrounded by parenthesis.
    /// It has to be a dedicated expression to ensure it gets a separate span.
    Wrapped(Box<Expression>),
    Id(Identifier),

    // Function type signature
    TypeFunc(Vec<Expression>, Box<Expression>),

    // Literals
    IntPattern(IntPattern),
    BoolLiteral(bool),
    StringLiteral(String),

    // Structures
    ArrayLiteral(Vec<ArrayLiteralElement<Expression>>),
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
    Builtin(Spanned<Vec<Expression>>),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Args<T = Expression> {
    pub span: Span,
    pub inner: Vec<Arg<T>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Arg<T = Expression> {
    pub span: Span,
    pub name: Option<Identifier>,
    pub value: T,
}

impl<T> Args<T> {
    pub fn map_inner<U>(&self, mut f: impl FnMut(&T) -> U) -> Args<U> {
        Args {
            span: self.span,
            inner: self.inner.iter().map(|arg| Arg {
                span: arg.span,
                name: arg.name.clone(),
                value: f(&arg.value),
            }).collect(),
        }
    }

    pub fn try_map_inner<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<Args<U>, E> {
        Ok(Args {
            span: self.span,
            inner: self.inner.into_iter().map(|arg| Ok(Arg {
                span: arg.span,
                name: arg.name,
                value: f(arg.value)?,
            })).try_collect()?,
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ArrayLiteralElement<V> {
    pub spread: Option<Span>,
    pub value: V,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StructLiteral {
    pub struct_ty: Box<Expression>,
    pub fields: Vec<StructLiteralField>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StructLiteralField {
    pub span: Span,
    pub id: Identifier,
    pub value: Expression,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RangeLiteral {
    pub end_inclusive: bool,
    pub start: Option<Box<Expression>>,
    pub end: Option<Box<Expression>>
}

#[derive(Debug, Clone)]
pub struct SyncExpression {
    pub clock: Box<Expression>,
    pub body: Block<BlockStatement>,
}

// TODO allow: undefined, wildcard, 0, 1, hex, decimal, ...
// TODO wildcard symbol: `_`, `?`, `*`, `#`?
//     `*` is a bad idea
// (don't allow any of the fancy stuff stuff for decimal ofc)
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum IntPattern {
    // [0-9a-fA-F_]+
    Hex(String),
    // [01_]+
    Bin(String),
    // [0-9_]+
    Dec(String),
}

#[derive(Debug, Clone)]
pub enum MaybeIdentifier<I = Identifier> {
    Dummy(Span),
    Identifier(I),
}

// TODO this is also just a spanned string
// TODO intern identifiers 
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Identifier {
    pub span: Span,
    pub string: String,
}

// TODO move to parser utilities module
pub fn build_binary_op(op: BinaryOp, left: Expression, right: Expression) -> ExpressionKind {
    ExpressionKind::BinaryOp(op, Box::new(left), Box::new(right))
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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
    BoolXor,

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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Copy, Clone)]
pub enum InterfaceDirection {
    None,
    In,
    Out,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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

    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            span: self.span,
            inner: &self.inner,
        }
    }
}

impl<T, E> Spanned<Result<T, E>> {
    pub fn transpose(self) -> Result<Spanned<T>, E> {
        self.inner.map(|inner| Spanned {
            span: self.span,
            inner,
        })
    }
}

impl<I> MaybeIdentifier<I> {
    pub fn as_ref(&self) -> MaybeIdentifier<&I> {
        match self {
            &MaybeIdentifier::Dummy(span) => MaybeIdentifier::Dummy(span),
            MaybeIdentifier::Identifier(id) => MaybeIdentifier::Identifier(id),
        }
    }

    pub fn map_inner<U>(self, f: impl FnOnce(I) -> U) -> MaybeIdentifier<U> {
        match self {
            MaybeIdentifier::Dummy(span) => MaybeIdentifier::Dummy(span),
            MaybeIdentifier::Identifier(id) => MaybeIdentifier::Identifier(f(id)),
        }
    }
}

impl MaybeIdentifier {
    pub fn span(&self) -> Span {
        match self {
            &MaybeIdentifier::Dummy(span) => span,
            MaybeIdentifier::Identifier(id) => id.span,
        }
    }

    pub fn string(&self) -> &str {
        match self {
            MaybeIdentifier::Dummy(_span) => "_",
            MaybeIdentifier::Identifier(id) => &id.string,
        }
    }
}

impl<'a> MaybeIdentifier<&'a Identifier> {
    pub fn span(&self) -> Span {
        match self {
            &MaybeIdentifier::Dummy(span) => span,
            MaybeIdentifier::Identifier(id) => id.span,
        }
    }

    pub fn string(&self) -> Option<&'a str> {
        match self {
            MaybeIdentifier::Dummy(_span) => None,
            MaybeIdentifier::Identifier(id) => Some(&id.string),
        }
    }
}

#[derive(Debug)]
pub struct ItemCommonInfo {
    pub span_full: Span,
    pub span_short: Span,
}

#[derive(Debug)]
pub struct ItemDeclarationInfo<'s> {
    pub vis: Visibility,
    pub id: MaybeIdentifier<&'s Identifier>,
}

impl Item {
    pub fn common_info(&self) -> ItemCommonInfo {
        self.info().0
    }

    pub fn declaration_info(&self) -> Option<ItemDeclarationInfo> {
        self.info().1
    }

    fn info(&self) -> (ItemCommonInfo, Option<ItemDeclarationInfo>) {
        match self {
            Item::Import(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.span }, None),
            Item::Const(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span() }, Some(ItemDeclarationInfo { vis: item.vis, id: item.id.as_ref() })),
            Item::Type(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span }, Some(ItemDeclarationInfo { vis: item.vis, id: MaybeIdentifier::Identifier(&item.id) })),
            Item::Struct(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span }, Some(ItemDeclarationInfo { vis: item.vis, id: MaybeIdentifier::Identifier(&item.id) })),
            Item::Enum(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span }, Some(ItemDeclarationInfo { vis: item.vis, id: MaybeIdentifier::Identifier(&item.id) })),
            Item::Function(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span }, Some(ItemDeclarationInfo { vis: item.vis, id: MaybeIdentifier::Identifier(&item.id) })),
            Item::Module(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span }, Some(ItemDeclarationInfo { vis: item.vis, id: MaybeIdentifier::Identifier(&item.id) })),
            Item::Interface(item) =>
                (ItemCommonInfo { span_full: item.span, span_short: item.id.span }, Some(ItemDeclarationInfo { vis: item.vis, id: MaybeIdentifier::Identifier(&item.id) })),
        }
    }
}

impl BinaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Pow => "**",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::BoolAnd => "&&",
            BinaryOp::BoolOr => "||",
            BinaryOp::BoolXor => "^^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::CmpEq => "==",
            BinaryOp::CmpNeq => "!=",
            BinaryOp::CmpLt => "<",
            BinaryOp::CmpLte => "<=",
            BinaryOp::CmpGt => ">",
            BinaryOp::CmpGte => ">=",
            BinaryOp::In => "in",
        }
    }
}