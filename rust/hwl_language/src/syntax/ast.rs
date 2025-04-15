use crate::front::value::Value;
use crate::syntax::pos::Span;
use crate::util::iter::IterExt;
// TODO remove "clone" from everything, and use ast lifetimes everywhere

#[derive(Debug, Clone)]
pub struct FileContent {
    pub span: Span,
    pub items: Vec<Item>,
}

#[derive(Debug, Copy, Clone)]
pub enum Visibility<S> {
    Public(S),
    Private,
}

// TODO add "doc comment" field to items?
// TODO rename to ItemDef
#[derive(Debug, Clone)]
pub enum Item {
    // non-declaring items
    Import(ItemImport),
    Instance(ModuleInstanceHeader),
    // common declarations that are allowed anywhere
    CommonDeclaration(ItemDeclaration),
    // declarations that are only allowed top-level
    // TODO maybe we should also just allow module declarations anywhere?
    Module(ItemDefModule),
}

#[derive(Debug, Clone)]
pub struct ItemDeclaration {
    pub vis: Visibility<Span>,
    pub decl: CommonDeclaration,
}

#[derive(Debug, Clone)]
pub enum CommonDeclaration {
    Type(TypeDeclaration),
    Const(ConstDeclaration),
    Struct(StructDeclaration),
    Enum(EnumDeclaration),
    Function(FunctionDeclaration),
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

// TODO allow "if" in a bunch of places? eg. struct fields
#[derive(Debug, Clone)]
pub struct StructDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub params: Option<Spanned<Vec<Parameter>>>,
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
pub struct EnumDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub params: Option<Spanned<Vec<Parameter>>>,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub span: Span,
    pub id: Identifier,
    pub content: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct FunctionDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    /// All function parameters are "generic", which means they can be types.
    /// It doesn't make sense to force a distinction similar to modules.
    pub params: Spanned<Vec<Parameter>>,
    pub ret_ty: Option<Expression>,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ItemDefModule {
    pub span: Span,
    pub vis: Visibility<Span>,
    pub id: MaybeIdentifier,
    pub params: Option<Spanned<Vec<Parameter>>>,
    pub ports: Spanned<Vec<ModulePortItem>>,
    pub body: Block<ModuleStatement>,
}

#[derive(Debug, Clone)]
pub struct InterfaceField {
    pub span: Span,
    pub id: Identifier,
    pub dir: InterfaceDirection,
    pub ty: Expression,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub span: Span,
    pub id: Identifier,
    pub ty: Expression,
    // TODO add default value
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
    pub kind: Spanned<WireKind<Spanned<DomainKind<Box<Expression>>>, Box<Expression>>>,
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
pub enum WireKind<S, T> {
    Clock,
    Normal { domain: S, ty: T },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DomainKind<S> {
    Const,
    Async,
    Sync(SyncDomain<S>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SyncDomain<S> {
    // TODO do we even need names here, is this not just a set of posedge sensitivities?
    pub clock: S,
    /// No reset means there is no separate sensitivity for a reset signal.
    /// This can be cause there is no reset, or because the reset is synchronous.
    pub reset: Option<S>,
}

impl<S> DomainKind<S> {
    pub fn map_signal<U>(self, f: impl FnMut(S) -> U) -> DomainKind<U> {
        match self {
            DomainKind::Const => DomainKind::Const,
            DomainKind::Async => DomainKind::Async,
            DomainKind::Sync(sync) => DomainKind::Sync(sync.map_signal(f)),
        }
    }
}

impl<S> SyncDomain<S> {
    pub fn as_ref(&self) -> SyncDomain<&S> {
        SyncDomain {
            clock: &self.clock,
            reset: self.reset.as_ref(),
        }
    }

    pub fn map_signal<U>(self, mut f: impl FnMut(S) -> U) -> SyncDomain<U> {
        SyncDomain {
            clock: f(self.clock),
            reset: self.reset.map(f),
        }
    }

    pub fn try_map_signal<U, E>(self, mut f: impl FnMut(S) -> Result<U, E>) -> Result<SyncDomain<U>, E> {
        Ok(SyncDomain {
            clock: f(self.clock)?,
            reset: self.reset.map(f).transpose()?,
        })
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

#[derive(Debug, Clone)]
pub struct BlockExpression {
    pub statements: Vec<BlockStatement>,
    pub expression: Box<Expression>,
}

pub type ModuleStatement = Spanned<ModuleStatementKind>;
pub type BlockStatement = Spanned<BlockStatementKind>;

#[derive(Debug, Clone)]
pub enum ModuleStatementKind {
    // control flow
    Block(Block<ModuleStatement>),
    If(IfStatement<ModuleStatement>),
    For(ForStatement<ModuleStatement>),
    // declarations
    CommonDeclaration(CommonDeclaration),
    RegDeclaration(RegDeclaration),
    WireDeclaration(WireDeclaration),
    // marker
    RegOutPortMarker(RegOutPortMarker),
    // blocks
    CombinatorialBlock(CombinatorialBlock),
    ClockedBlock(ClockedBlock),
    Instance(ModuleInstance),
}

#[derive(Debug, Clone)]
pub enum BlockStatementKind {
    // declarations
    CommonDeclaration(CommonDeclaration),
    VariableDeclaration(VariableDeclaration),

    // basic statements
    Assignment(Assignment),
    // TODO remove expressions, maybe with exception for function calls?
    Expression(Box<Expression>),

    // control flow
    Block(Block<BlockStatement>),
    If(IfStatement<BlockStatement>),
    For(ForStatement<BlockStatement>),
    While(WhileStatement),

    // control flow terminators
    Return(ReturnStatement),
    Break(Span),
    Continue(Span),
}

#[derive(Debug, Clone)]
pub struct IfStatement<S> {
    pub initial_if: IfCondBlockPair<S>,
    pub else_ifs: Vec<IfCondBlockPair<S>>,
    pub final_else: Option<Block<S>>,
}

#[derive(Debug, Clone)]
pub struct IfCondBlockPair<S> {
    pub span: Span,
    pub span_if: Span,
    pub cond: Box<Expression>,
    pub block: Block<S>,
}

#[derive(Debug, Clone)]
pub struct WhileStatement {
    pub span_keyword: Span,
    pub cond: Box<Expression>,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ForStatement<S> {
    pub span_keyword: Span,
    pub index: MaybeIdentifier,
    pub index_ty: Option<Box<Expression>>,
    pub iter: Box<Expression>,
    pub body: Block<S>,
}

#[derive(Debug, Clone)]
pub struct ReturnStatement {
    pub span_return: Span,
    pub value: Option<Box<Expression>>,
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
    pub kind: Spanned<WireKind<Spanned<DomainKind<Box<Expression>>>, Box<Expression>>>,
    pub value: Option<Box<Expression>>,
}

#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub params: Option<Spanned<Vec<Parameter>>>,
    pub body: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct ConstDeclaration {
    pub span: Span,
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
    pub init: Option<Box<Expression>>,
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
    pub span_domain: Span,
    pub clock: Box<Expression>,
    /// No reset means this block does not have a reset.
    pub reset: Option<Spanned<ClockedBlockReset<Box<Expression>>>>,
    pub block: Block<BlockStatement>,
}

#[derive(Debug, Copy, Clone)]
pub struct ClockedBlockReset<S> {
    pub kind: Spanned<ResetKind>,
    pub signal: S,
}

impl<S> ClockedBlockReset<S> {
    pub fn map_signal<U>(self, f: impl FnOnce(S) -> U) -> ClockedBlockReset<U> {
        ClockedBlockReset {
            kind: self.kind,
            signal: f(self.signal),
        }
    }

    pub fn as_ref(&self) -> ClockedBlockReset<&S> {
        ClockedBlockReset {
            kind: self.kind,
            signal: &self.signal,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ResetKind {
    Async,
    Sync,
}

#[derive(Debug, Clone)]
pub struct ModuleInstance {
    pub name: Option<Identifier>,
    pub header: ModuleInstanceHeader,
    pub port_connections: Spanned<Vec<Spanned<PortConnection>>>,
}

#[derive(Debug, Clone)]
pub struct ModuleInstanceHeader {
    pub span: Span,
    pub span_keyword: Span,
    pub module: Box<Expression>,
    pub generic_args: Option<Args>,
}

#[derive(Debug, Clone)]
pub struct PortConnection {
    pub id: Identifier,
    pub expr: Expression,
}

pub type Expression = Spanned<ExpressionKind>;

#[derive(Debug, Clone)]
pub enum ExpressionKind {
    // Miscellaneous
    Dummy,
    Undefined,
    Type,
    TypeFunction,
    /// Wrapped just means an expression that's surrounded by parenthesis.
    /// It has to be a dedicated expression to ensure it gets a separate span.
    Wrapped(Box<Expression>),
    Block(BlockExpression),
    Id(Identifier),

    // Literals
    IntLiteral(IntLiteral),
    BoolLiteral(bool),
    StringLiteral(String),

    // Structures
    ArrayLiteral(Vec<ArrayLiteralElement<Expression>>),
    TupleLiteral(Vec<Expression>),
    StructLiteral(StructLiteral),
    RangeLiteral(RangeLiteral),
    ArrayComprehension(ArrayComprehension),

    // Operations
    UnaryOp(Spanned<UnaryOp>, Box<Expression>),
    BinaryOp(Spanned<BinaryOp>, Box<Expression>, Box<Expression>),
    TernarySelect(Box<Expression>, Box<Expression>, Box<Expression>),

    // Indexing
    ArrayType(Spanned<Vec<ArrayLiteralElement<Expression>>>, Box<Expression>),
    ArrayIndex(Box<Expression>, Spanned<Vec<Expression>>),
    DotIdIndex(Box<Expression>, Identifier),
    DotIntIndex(Box<Expression>, Spanned<String>),

    // Calls
    Call(Box<Expression>, Args),
    Builtin(Spanned<Vec<Expression>>),
    RegisterDelay(RegisterDelay),
}

#[derive(Debug, Clone)]
pub struct RegisterDelay {
    pub span_keyword: Span,
    pub value: Box<Expression>,
    pub init: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct Args<N = Option<Identifier>, T = Expression> {
    pub span: Span,
    pub inner: Vec<Arg<N, T>>,
}

#[derive(Debug, Clone)]
pub struct Arg<N = Option<Identifier>, T = Expression> {
    pub span: Span,
    pub name: N,
    pub value: T,
}

impl<N, T> Args<N, T> {
    pub fn map_inner<U>(&self, mut f: impl FnMut(&T) -> U) -> Args<N, U>
    where
        N: Clone,
    {
        Args {
            span: self.span,
            inner: self
                .inner
                .iter()
                .map(|arg| Arg {
                    span: arg.span,
                    name: arg.name.clone(),
                    value: f(&arg.value),
                })
                .collect(),
        }
    }

    pub fn try_map_inner_all<U, E>(&self, mut f: impl FnMut(&T) -> Result<U, E>) -> Result<Args<N, U>, E>
    where
        N: Clone,
    {
        Ok(Args {
            span: self.span,
            inner: self
                .inner
                .iter()
                .map(|arg| {
                    Ok(Arg {
                        span: arg.span,
                        name: arg.name.clone(),
                        value: f(&arg.value)?,
                    })
                })
                .try_collect_all()?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum ArrayLiteralElement<V> {
    Single(V),
    Spread(Span, V),
}

impl<V> ArrayLiteralElement<Spanned<V>> {
    pub fn span(&self) -> Span {
        match self {
            ArrayLiteralElement::Spread(span, value) => span.join(value.span),
            ArrayLiteralElement::Single(value) => value.span,
        }
    }
}

impl<V> ArrayLiteralElement<V> {
    pub fn map_inner<W>(&self, f: impl FnOnce(&V) -> W) -> ArrayLiteralElement<W> {
        match self {
            ArrayLiteralElement::Single(value) => ArrayLiteralElement::Single(f(value)),
            ArrayLiteralElement::Spread(span, value) => ArrayLiteralElement::Spread(*span, f(value)),
        }
    }

    pub fn value(&self) -> &V {
        match self {
            ArrayLiteralElement::Single(value) => value,
            ArrayLiteralElement::Spread(_, value) => value,
        }
    }
}

impl<T, E> ArrayLiteralElement<Result<T, E>> {
    pub fn transpose(self) -> Result<ArrayLiteralElement<T>, E> {
        match self {
            ArrayLiteralElement::Single(value) => Ok(ArrayLiteralElement::Single(value?)),
            ArrayLiteralElement::Spread(span, value) => Ok(ArrayLiteralElement::Spread(span, value?)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArrayComprehension {
    pub body: Box<ArrayLiteralElement<Expression>>,
    pub index: MaybeIdentifier,
    pub span_keyword: Span,
    pub iter: Box<Expression>,
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
pub enum RangeLiteral {
    ExclusiveEnd {
        op_span: Span,
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
    },
    InclusiveEnd {
        op_span: Span,
        start: Option<Box<Expression>>,
        end: Box<Expression>,
    },
    Length {
        op_span: Span,
        start: Box<Expression>,
        len: Box<Expression>,
    },
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
#[derive(Debug, Clone)]
pub enum IntLiteral {
    // 0b[01_]+
    Binary(String),
    // [0-9_]+
    Decimal(String),
    // 0x[0-9a-fA-F_]+
    Hexadecimal(String),
}

#[derive(Debug, Clone)]
pub enum MaybeIdentifier<I = Identifier> {
    Dummy(Span),
    Identifier(I),
}

// TODO this is also just a spanned string
// TODO intern identifiers
#[derive(Debug, Clone)]
pub struct Identifier {
    pub span: Span,
    pub string: String,
}

// TODO move to parser utilities module
pub fn build_binary_op(op_span: Span, op: BinaryOp, left: Expression, right: Expression) -> ExpressionKind {
    ExpressionKind::BinaryOp(
        Spanned {
            span: op_span,
            inner: op,
        },
        Box::new(left),
        Box::new(right),
    )
}

#[derive(Debug, Copy, Clone)]
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

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Plus,
    Neg,
    Not,
}

#[derive(Debug, Copy, Clone)]
pub enum InterfaceDirection {
    None,
    In,
    Out,
}

// TODO move to pos?
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Spanned<T> {
    pub span: Span,
    pub inner: T,
}

impl<T> Spanned<T> {
    pub fn new(span: Span, inner: T) -> Spanned<T> {
        Spanned { span, inner }
    }

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

impl<T> Spanned<&T> {
    pub fn cloned(&self) -> Spanned<T>
    where
        T: Clone,
    {
        Spanned {
            span: self.span,
            inner: self.inner.clone(),
        }
    }
}

impl<T, E> Spanned<Result<T, E>> {
    pub fn transpose(self) -> Result<Spanned<T>, E> {
        self.inner.map(|inner| Spanned { span: self.span, inner })
    }
}

impl<T> Spanned<Option<T>> {
    pub fn transpose(self) -> Option<Spanned<T>> {
        self.inner.map(|inner| Spanned { span: self.span, inner })
    }
}

impl<T, C> Spanned<Value<T, C>> {
    pub fn transpose(self) -> Value<Spanned<T>, Spanned<C>> {
        match self.inner {
            Value::Compile(value) => Value::Compile(Spanned {
                span: self.span,
                inner: value,
            }),
            Value::Hardware(value) => Value::Hardware(Spanned {
                span: self.span,
                inner: value,
            }),
        }
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
    pub vis: Visibility<Span>,
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
            Item::Import(item) => (
                ItemCommonInfo {
                    span_full: item.span,
                    span_short: item.span,
                },
                None,
            ),
            Item::Instance(item) => (
                ItemCommonInfo {
                    span_full: item.span,
                    span_short: item.span_keyword,
                },
                None,
            ),
            Item::CommonDeclaration(item) => {
                let (span, id) = item.decl.info();
                (
                    ItemCommonInfo {
                        span_full: span,
                        span_short: id.span(),
                    },
                    Some(ItemDeclarationInfo {
                        vis: item.vis,
                        id: id.as_ref(),
                    }),
                )
            }
            Item::Module(item) => (
                ItemCommonInfo {
                    span_full: item.span,
                    span_short: item.id.span(),
                },
                Some(ItemDeclarationInfo {
                    vis: item.vis,
                    id: item.id.as_ref(),
                }),
            ),
        }
    }
}

impl CommonDeclaration {
    pub fn info(&self) -> (Span, &MaybeIdentifier) {
        match self {
            CommonDeclaration::Type(decl) => (decl.span, &decl.id),
            CommonDeclaration::Const(decl) => (decl.span, &decl.id),
            CommonDeclaration::Struct(decl) => (decl.span, &decl.id),
            CommonDeclaration::Enum(decl) => (decl.span, &decl.id),
            CommonDeclaration::Function(decl) => (decl.span, &decl.id),
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
