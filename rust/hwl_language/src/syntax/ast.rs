use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::syntax::source::SourceDatabase;
use crate::syntax::token::TokenType;
use crate::util::arena::Arena;
use crate::{impl_has_span, new_index_type};

new_index_type!(pub ExpressionKindIndex);

#[derive(Debug)]
pub struct FileContent {
    pub span: Span,
    pub items: Vec<Item>,

    pub arena_expressions: ArenaExpressions,
}

pub type ArenaExpressions = Arena<ExpressionKindIndex, ExpressionKind>;

#[derive(Debug, Copy, Clone)]
pub enum Visibility<S = Span> {
    Public(S),
    Private,
}

// TODO add "doc comment" field to items?
// TODO rename to ItemDef or maybe FileItem
#[derive(Debug, Clone)]
pub enum Item {
    // non-declaring items
    Import(ItemImport),
    // common declarations that are allowed anywhere
    CommonDeclaration(Spanned<CommonDeclaration<Visibility>>),
    // declarations that are only allowed top-level
    // TODO turn all of these into common declarations, even the imports
    ModuleInternal(ItemDefModuleInternal),
    ModuleExternal(ItemDefModuleExternal),
    Interface(ItemDefInterface),
}

// TODO rename, not all of these are actually declarations
#[derive(Debug, Clone)]
pub enum CommonDeclaration<V> {
    Named(CommonDeclarationNamed<V>),
    ConstBlock(ConstBlock),
}

#[derive(Debug, Clone)]
pub struct CommonDeclarationNamed<V> {
    pub vis: V,
    pub kind: CommonDeclarationNamedKind,
}

#[derive(Debug, Clone)]
pub enum CommonDeclarationNamedKind {
    Type(TypeDeclaration),
    Const(ConstDeclaration),
    Struct(StructDeclaration),
    Enum(EnumDeclaration),
    Function(FunctionDeclaration),
}

#[derive(Debug, Clone)]
pub struct ConstBlock {
    pub span_keyword: Span,
    pub block: Block<BlockStatement>,
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
    pub span_body: Span,
    pub id: MaybeIdentifier,
    pub params: Option<Parameters>,
    pub fields: ExtraList<StructField>,
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
    pub params: Option<Parameters>,
    pub variants: ExtraList<EnumVariant>,
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
    pub params: Parameters,
    pub ret_ty: Option<Expression>,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ItemDefModuleInternal {
    pub span: Span,
    pub vis: Visibility,
    pub id: MaybeIdentifier,
    pub params: Option<Parameters>,
    pub ports: Spanned<ExtraList<ModulePortItem>>,
    pub body: Block<ModuleStatement>,
}

#[derive(Debug, Clone)]
pub struct ItemDefModuleExternal {
    pub span: Span,
    pub vis: Visibility,
    pub span_ext: Span,
    pub id: Identifier,
    pub params: Option<Parameters>,
    pub ports: Spanned<ExtraList<ModulePortItem>>,
}

#[derive(Debug, Clone)]
pub struct ItemDefInterface {
    pub span: Span,
    pub vis: Visibility,
    pub id: MaybeIdentifier,
    pub params: Option<Parameters>,
    pub span_body: Span,
    pub port_types: ExtraList<(Identifier, Expression)>,
    pub views: Vec<InterfaceView>,
}

#[derive(Debug, Clone)]
pub struct InterfaceView {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub port_dirs: ExtraList<(Identifier, Spanned<PortDirection>)>,
}

#[derive(Debug, Clone)]
pub struct Parameters {
    pub span: Span,
    pub items: ExtraList<Parameter>,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub span: Span,
    pub id: Identifier,
    pub ty: Expression,
    pub default: Option<Expression>,
}

// TODO maybe this is the same as Block?
#[derive(Debug, Clone)]
pub struct ExtraList<I> {
    pub span: Span,
    pub items: Vec<ExtraItem<I>>,
}

#[derive(Debug, Clone)]
pub enum ExtraItem<I> {
    Inner(I),
    Declaration(CommonDeclaration<()>),
    // TODO add `match`
    If(IfStatement<ExtraList<I>>),
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
    pub kind: ModulePortSingleKind,
}

#[derive(Debug, Clone)]
pub struct ModulePortBlock {
    pub span: Span,
    pub domain: Spanned<DomainKind<Expression>>,
    pub ports: ExtraList<ModulePortInBlock>,
}

#[derive(Debug, Clone)]
pub struct ModulePortInBlock {
    pub span: Span,
    pub id: Identifier,
    pub kind: ModulePortInBlockKind,
}

#[derive(Debug, Clone)]
pub enum ModulePortSingleKind {
    Port {
        direction: Spanned<PortDirection>,
        kind: PortSingleKindInner,
    },
    Interface {
        span_keyword: Span,
        domain: Spanned<DomainKind<Expression>>,
        interface: Expression,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum PortSingleKindInner {
    Clock {
        span_clock: Span,
    },
    Normal {
        domain: Spanned<DomainKind<Expression>>,
        ty: Expression,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum ModulePortInBlockKind {
    Port {
        direction: Spanned<PortDirection>,
        ty: Expression,
    },
    Interface {
        span_keyword: Span,
        interface: Expression,
    },
}

// TODO rename to HardwareDomain and include clock here?
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

impl PortDirection {
    pub fn diagnostic_string(self) -> &'static str {
        match self {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
        }
    }

    pub fn token(self) -> TokenType {
        match self {
            PortDirection::Input => TokenType::In,
            PortDirection::Output => TokenType::Out,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block<S> {
    pub span: Span,
    pub statements: Vec<S>,
}

#[derive(Debug, Clone)]
pub struct BlockExpression {
    pub statements: Vec<BlockStatement>,
    pub expression: Expression,
}

pub type ModuleStatement = Spanned<ModuleStatementKind>;
pub type BlockStatement = Spanned<BlockStatementKind>;

#[derive(Debug, Clone)]
pub enum ModuleStatementKind {
    // control flow
    Block(Block<ModuleStatement>),
    If(IfStatement<Block<ModuleStatement>>),
    For(ForStatement<ModuleStatement>),
    // declarations
    CommonDeclaration(CommonDeclaration<()>),
    RegDeclaration(RegDeclaration),
    WireDeclaration(WireDeclaration),
    // marker
    RegOutPortMarker(RegOutPortMarker),
    // children
    CombinatorialBlock(CombinatorialBlock),
    ClockedBlock(ClockedBlock),
    Instance(ModuleInstance),
}

#[derive(Debug, Clone)]
pub enum BlockStatementKind {
    // declarations
    CommonDeclaration(CommonDeclaration<()>),
    VariableDeclaration(VariableDeclaration),

    // basic statements
    Assignment(Assignment),
    // TODO remove expressions, maybe with exception for function calls?
    Expression(Expression),

    // control flow
    Block(Block<BlockStatement>),
    If(IfStatement<Block<BlockStatement>>),
    Match(MatchStatement<Block<BlockStatement>>),

    For(ForStatement<BlockStatement>),
    While(WhileStatement),

    // control flow terminators
    Return(ReturnStatement),
    Break(Span),
    Continue(Span),
}

#[derive(Debug, Clone)]
pub struct IfStatement<B> {
    pub span: Span,
    pub initial_if: IfCondBlockPair<B>,
    pub else_ifs: Vec<IfCondBlockPair<B>>,
    pub final_else: Option<B>,
}

#[derive(Debug, Clone)]
pub struct IfCondBlockPair<B> {
    pub span: Span,
    pub span_if: Span,
    pub cond: Expression,
    pub block: B,
}

#[derive(Debug, Clone)]
pub struct MatchStatement<B> {
    pub target: Expression,
    pub span_branches: Span,
    pub branches: Vec<MatchBranch<B>>,
}

#[derive(Debug, Clone)]
pub struct MatchBranch<B> {
    pub pattern: Spanned<MatchPattern>,
    pub block: B,
}

#[derive(Debug, Clone)]
pub enum MatchPattern<E = Expression, R = Expression, V = Identifier, I = Identifier> {
    Wildcard,
    Val(I),
    Equal(E),
    In(R),
    EnumVariant(V, Option<MaybeIdentifier>),
}

#[derive(Debug, Clone)]
pub struct WhileStatement {
    pub span_keyword: Span,
    pub cond: Expression,
    pub body: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ForStatement<S> {
    pub span_keyword: Span,
    pub index: MaybeIdentifier,
    pub index_ty: Option<Expression>,
    pub iter: Expression,
    pub body: Block<S>,
}

#[derive(Debug, Clone)]
pub struct ReturnStatement {
    pub span_return: Span,
    pub value: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct RegOutPortMarker {
    pub id: Identifier,
    pub init: Expression,
}

#[derive(Debug, Clone)]
pub struct RegDeclaration {
    pub vis: Visibility,
    pub id: MaybeGeneralIdentifier,
    pub sync: Option<Spanned<SyncDomain<Expression>>>,
    pub ty: Expression,
    pub init: Expression,
}

#[derive(Debug, Clone)]
pub struct WireDeclaration {
    pub vis: Visibility,
    pub span_keyword: Span,
    pub id: MaybeGeneralIdentifier,
    pub kind: WireDeclarationKind,
}

#[derive(Debug, Copy, Clone)]
pub enum WireDeclarationKind {
    Normal {
        domain_ty: WireDeclarationDomainTyKind,
        assign_span_and_value: Option<(Span, Expression)>,
    },
    Interface {
        domain: Option<Spanned<DomainKind<Expression>>>,
        span_keyword: Span,
        interface: Expression,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum WireDeclarationDomainTyKind {
    Clock {
        span_clock: Span,
    },
    Normal {
        domain: Option<Spanned<DomainKind<Expression>>>,
        ty: Option<Expression>,
    },
}

#[derive(Debug, Clone)]
pub struct TypeDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub params: Option<Parameters>,
    pub body: Expression,
}

#[derive(Debug, Clone)]
pub struct ConstDeclaration {
    pub span: Span,
    pub id: MaybeIdentifier,
    pub ty: Option<Expression>,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub span: Span,
    pub mutable: bool,
    pub id: MaybeIdentifier,
    pub ty: Option<Expression>,
    pub init: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub span: Span,
    pub op: Spanned<Option<AssignBinaryOp>>,
    pub target: Expression,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct CombinatorialBlock {
    pub span_keyword: Span,
    pub block: Block<BlockStatement>,
}

#[derive(Debug, Clone)]
pub struct ClockedBlock {
    pub span_keyword: Span,
    pub span_domain: Span,
    pub clock: Expression,
    /// No reset means this block does not have a reset.
    pub reset: Option<Spanned<ClockedBlockReset<Expression>>>,
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

#[derive(Debug, Copy, Clone)]
pub enum ResetKind {
    Async,
    Sync,
}

impl ResetKind {
    pub fn token(self) -> TokenType {
        match self {
            ResetKind::Async => TokenType::Async,
            ResetKind::Sync => TokenType::Sync,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleInstanceItem {
    pub span: Span,
    pub span_keyword: Span,
    pub module: Expression,
}

#[derive(Debug, Clone)]
pub struct ModuleInstance {
    pub name: Option<Identifier>,
    pub span_keyword: Span,
    pub module: Expression,
    // TODO this should be an extra list
    pub port_connections: Spanned<Vec<Spanned<PortConnection>>>,
}

// TODO find a way to avoid this expression representation weirdness
#[derive(Debug, Clone)]
pub struct PortConnection {
    pub id: Identifier,
    pub expr: PortConnectionExpression,
}

#[derive(Debug, Copy, Clone)]
pub enum PortConnectionExpression {
    FakeId(Expression),
    Real(Expression),
}

impl PortConnectionExpression {
    pub fn expr(&self) -> Expression {
        match *self {
            PortConnectionExpression::FakeId(expr) => expr,
            PortConnectionExpression::Real(expr) => expr,
        }
    }
}

// TODO we're using Box<Spanned<ExpressionKind>> a lot, but maybe
//   Spanned<Box<ExpressionKind>> is better?
pub type Expression = Spanned<ExpressionKindIndex>;

#[derive(Debug, Clone)]
pub enum ExpressionKind {
    // Miscellaneous
    Dummy,
    // TODO maybe this should not be part of expression, it's only allowed in very few places
    //   just the existence of this makes type checking harder to trust
    Undefined,
    Type,
    TypeFunction,
    Builtin,
    /// Wrapped just means an expression that's surrounded by parenthesis.
    /// It has to be a dedicated expression to ensure it gets a separate span.
    Wrapped(Expression),
    Block(BlockExpression),
    Id(GeneralIdentifier),

    // Literals
    IntLiteral(IntLiteral),
    BoolLiteral(bool),
    StringLiteral(Vec<StringPiece>),

    // Structures
    ArrayLiteral(Vec<ArrayLiteralElement<Expression>>),
    TupleLiteral(Vec<Expression>),
    RangeLiteral(RangeLiteral),
    ArrayComprehension(ArrayComprehension),

    // Operations
    UnaryOp(Spanned<UnaryOp>, Expression),
    BinaryOp(Spanned<BinaryOp>, Expression, Expression),

    // Indexing
    ArrayType(Spanned<Vec<ArrayLiteralElement<Expression>>>, Expression),
    ArrayIndex(Expression, Spanned<Vec<Expression>>),
    DotIndex(Expression, DotIndexKind),

    // Calls
    Call(Expression, Args),
    UnsafeValueWithDomain(Expression, Spanned<DomainKind<Expression>>),
    RegisterDelay(RegisterDelay),
}

#[derive(Debug, Copy, Clone)]
pub enum DotIndexKind {
    Id(Identifier),
    Int(Span),
}

#[derive(Debug, Copy, Clone)]
pub enum StringPiece {
    Literal(Span),
    Substitute(Expression),
}

#[derive(Debug, Clone)]
pub struct RegisterDelay {
    pub span_keyword: Span,
    pub value: Expression,
    pub init: Expression,
}

#[derive(Debug, Clone)]
pub struct Args<N = Option<Identifier>, T = Expression> {
    pub span: Span,
    // TODO this should be an ExtraList
    pub inner: Vec<Arg<N, T>>,
}

#[derive(Debug, Copy, Clone)]
pub struct Arg<N = Option<Identifier>, T = Expression> {
    pub span: Span,
    pub name: N,
    pub value: T,
}

#[derive(Debug, Copy, Clone)]
pub enum ArrayLiteralElement<V> {
    Single(V),
    Spread(Span, V),
}

impl<V> ArrayLiteralElement<V> {
    pub fn map_inner<W>(self, f: impl FnOnce(V) -> W) -> ArrayLiteralElement<W> {
        match self {
            ArrayLiteralElement::Single(value) => ArrayLiteralElement::Single(f(value)),
            ArrayLiteralElement::Spread(span, value) => ArrayLiteralElement::Spread(span, f(value)),
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
    pub body: ArrayLiteralElement<Expression>,
    pub index: MaybeIdentifier,
    pub span_keyword: Span,
    pub iter: Expression,
}

#[derive(Debug, Clone)]
pub struct StructLiteral {
    pub struct_ty: Expression,
    pub fields: Vec<StructLiteralField>,
}

#[derive(Debug, Clone)]
pub struct StructLiteralField {
    pub span: Span,
    pub id: Identifier,
    pub value: Expression,
}

#[derive(Debug, Copy, Clone)]
pub enum RangeLiteral {
    ExclusiveEnd {
        op_span: Span,
        start: Option<Expression>,
        end: Option<Expression>,
    },
    InclusiveEnd {
        op_span: Span,
        start: Option<Expression>,
        end: Expression,
    },
    Length {
        op_span: Span,
        start: Expression,
        len: Expression,
    },
}

#[derive(Debug, Clone)]
pub struct SyncExpression {
    pub clock: Expression,
    pub body: Block<BlockStatement>,
}

// TODO allow: undefined, wildcard, 0, 1, hex, decimal, ...
// TODO wildcard symbol: `_`, `?`, `*`, `#`?
//     `*` is a bad idea
// (don't allow any of the fancy stuff stuff for decimal ofc)
#[derive(Debug, Clone)]
pub enum IntLiteral {
    // 0b[01_]+
    Binary(Span),
    // [0-9_]+
    Decimal(Span),
    // 0x[0-9a-fA-F_]+
    Hexadecimal(Span),
}

// TODO rename back to Identifier?
#[derive(Debug, Copy, Clone)]
pub struct Identifier {
    pub span: Span,
}

// TODO intern identifiers?
#[derive(Debug, Copy, Clone)]
pub enum GeneralIdentifier {
    Simple(Identifier),
    FromString(Span, Expression),
}

#[derive(Debug, Copy, Clone)]
pub enum MaybeIdentifier<I = Identifier> {
    Dummy(Span),
    Identifier(I),
}

pub type MaybeGeneralIdentifier = MaybeIdentifier<GeneralIdentifier>;

// TODO move to parser utilities module
pub fn build_binary_op(
    arena_expressions: &mut Arena<ExpressionKindIndex, ExpressionKind>,
    op_span: Span,
    op: BinaryOp,
    left: Spanned<ExpressionKind>,
    right: Spanned<ExpressionKind>,
) -> ExpressionKind {
    ExpressionKind::BinaryOp(
        Spanned {
            span: op_span,
            inner: op,
        },
        left.map_inner(|e| arena_expressions.push(e)),
        right.map_inner(|e| arena_expressions.push(e)),
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

#[derive(Debug, Copy, Clone)]
pub enum AssignBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // TODO add boolean and shift operators
    BitAnd,
    BitOr,
    BitXor,
}

#[derive(Debug, Copy, Clone)]
pub enum UnaryOp {
    Plus,
    Neg,
    Not,
}

impl Identifier {
    pub fn str(self, source: &SourceDatabase) -> &str {
        source.span_str(self.span)
    }

    pub fn spanned_str(self, source: &SourceDatabase) -> Spanned<&str> {
        Spanned::new(self.span, self.str(source))
    }

    pub fn spanned_string(self, source: &SourceDatabase) -> Spanned<String> {
        self.spanned_str(source).map_inner(str::to_owned)
    }
}

impl<I> MaybeIdentifier<I> {
    pub fn map_id<J>(self, f: impl FnOnce(I) -> J) -> MaybeIdentifier<J> {
        match self {
            MaybeIdentifier::Dummy(span) => MaybeIdentifier::Dummy(span),
            MaybeIdentifier::Identifier(id) => MaybeIdentifier::Identifier(f(id)),
        }
    }

    pub fn as_ref(&self) -> MaybeIdentifier<&I> {
        match self {
            &MaybeIdentifier::Dummy(span) => MaybeIdentifier::Dummy(span),
            MaybeIdentifier::Identifier(id) => MaybeIdentifier::Identifier(id),
        }
    }
}

impl MaybeIdentifier<Identifier> {
    pub fn spanned_str(self, source: &SourceDatabase) -> MaybeIdentifier<Spanned<&str>> {
        match self {
            MaybeIdentifier::Dummy(span) => MaybeIdentifier::Dummy(span),
            MaybeIdentifier::Identifier(id) => {
                MaybeIdentifier::Identifier(Spanned::new(id.span, source.span_str(id.span)))
            }
        }
    }

    pub fn spanned_string(self, source: &SourceDatabase) -> Spanned<Option<String>> {
        match self {
            MaybeIdentifier::Dummy(span) => Spanned::new(span, None),
            MaybeIdentifier::Identifier(id) => Spanned::new(id.span, Some(source.span_str(id.span).to_owned())),
        }
    }
}

impl<S: AsRef<str>> MaybeIdentifier<Spanned<S>> {
    pub fn diagnostic_str(&self) -> &str {
        match self {
            MaybeIdentifier::Dummy(_) => "_",
            MaybeIdentifier::Identifier(id) => id.inner.as_ref(),
        }
    }

    pub fn spanned_string(&self) -> Spanned<Option<String>> {
        match self {
            MaybeIdentifier::Dummy(span) => Spanned::new(*span, None),
            MaybeIdentifier::Identifier(id) => Spanned::new(id.span, Some(id.inner.as_ref().to_owned())),
        }
    }
}

#[derive(Debug)]
pub struct ItemInfo {
    pub span_full: Span,
    pub span_short: Span,
    pub declaration: Option<ItemDeclarationInfo>,
}

#[derive(Debug)]
pub struct ItemDeclarationInfo {
    pub vis: Visibility,
    pub id: MaybeIdentifier,
}

impl Item {
    pub fn info(&self) -> ItemInfo {
        match self {
            Item::Import(item) => ItemInfo {
                span_full: item.span,
                span_short: item.span,
                declaration: None,
            },
            Item::CommonDeclaration(item) => ItemInfo {
                span_full: item.span,
                span_short: item.inner.span_short(),
                declaration: match &item.inner {
                    CommonDeclaration::Named(decl) => Some(ItemDeclarationInfo {
                        vis: decl.vis,
                        id: decl.kind.id(),
                    }),
                    CommonDeclaration::ConstBlock(_) => None,
                },
            },
            Item::ModuleInternal(item) => ItemInfo {
                span_full: item.span,
                span_short: item.id.span(),
                declaration: Some(ItemDeclarationInfo {
                    vis: item.vis,
                    id: item.id,
                }),
            },
            Item::ModuleExternal(item) => ItemInfo {
                span_full: item.span,
                span_short: item.id.span,
                declaration: Some(ItemDeclarationInfo {
                    vis: item.vis,
                    id: MaybeIdentifier::Identifier(item.id),
                }),
            },
            Item::Interface(item) => ItemInfo {
                span_full: item.span,
                span_short: item.id.span(),
                declaration: Some(ItemDeclarationInfo {
                    vis: item.vis,
                    id: item.id,
                }),
            },
        }
    }
}

impl<V> CommonDeclaration<V> {
    pub fn span_short(&self) -> Span {
        match self {
            CommonDeclaration::Named(decl) => decl.kind.id().span(),
            CommonDeclaration::ConstBlock(block) => block.span_keyword,
        }
    }
}

impl CommonDeclarationNamedKind {
    pub fn id(&self) -> MaybeIdentifier {
        match self {
            CommonDeclarationNamedKind::Type(decl) => decl.id,
            CommonDeclarationNamedKind::Const(decl) => decl.id,
            CommonDeclarationNamedKind::Struct(decl) => decl.id,
            CommonDeclarationNamedKind::Enum(decl) => decl.id,
            CommonDeclarationNamedKind::Function(decl) => decl.id,
        }
    }
}

impl UnaryOp {
    pub fn token(self) -> TokenType {
        match self {
            UnaryOp::Plus => TokenType::Plus,
            UnaryOp::Neg => TokenType::Minus,
            UnaryOp::Not => TokenType::Bang,
        }
    }
}

impl BinaryOp {
    pub fn token(self) -> TokenType {
        match self {
            BinaryOp::Add => TokenType::Plus,
            BinaryOp::Sub => TokenType::Minus,
            BinaryOp::Mul => TokenType::Star,
            BinaryOp::Div => TokenType::Slash,
            BinaryOp::Mod => TokenType::Percent,
            BinaryOp::Pow => TokenType::StarStar,
            BinaryOp::BitAnd => TokenType::Amper,
            BinaryOp::BitOr => TokenType::Pipe,
            BinaryOp::BitXor => TokenType::Caret,
            BinaryOp::BoolAnd => TokenType::AmperAmper,
            BinaryOp::BoolOr => TokenType::PipePipe,
            BinaryOp::BoolXor => TokenType::CaretCaret,
            BinaryOp::Shl => TokenType::LtLt,
            BinaryOp::Shr => TokenType::GtGt,
            BinaryOp::CmpEq => TokenType::EqEq,
            BinaryOp::CmpNeq => TokenType::Neq,
            BinaryOp::CmpLt => TokenType::Lt,
            BinaryOp::CmpLte => TokenType::Lte,
            BinaryOp::CmpGt => TokenType::Gt,
            BinaryOp::CmpGte => TokenType::Gte,
            BinaryOp::In => TokenType::In,
        }
    }
}

impl AssignBinaryOp {
    pub fn token(self) -> TokenType {
        match self {
            AssignBinaryOp::Add => TokenType::PlusEq,
            AssignBinaryOp::Sub => TokenType::MinusEq,
            AssignBinaryOp::Mul => TokenType::StarEq,
            AssignBinaryOp::Div => TokenType::SlashEq,
            AssignBinaryOp::Mod => TokenType::PercentEq,
            AssignBinaryOp::BitAnd => TokenType::AmperEq,
            AssignBinaryOp::BitOr => TokenType::PipeEq,
            AssignBinaryOp::BitXor => TokenType::CaretEq,
        }
    }

    pub fn to_binary_op(self) -> BinaryOp {
        match self {
            AssignBinaryOp::Add => BinaryOp::Add,
            AssignBinaryOp::Sub => BinaryOp::Sub,
            AssignBinaryOp::Mul => BinaryOp::Mul,
            AssignBinaryOp::Div => BinaryOp::Div,
            AssignBinaryOp::Mod => BinaryOp::Mod,
            AssignBinaryOp::BitAnd => BinaryOp::BitAnd,
            AssignBinaryOp::BitOr => BinaryOp::BitOr,
            AssignBinaryOp::BitXor => BinaryOp::BitXor,
        }
    }
}

// TODO this could be reduced a bit with a derive macro, eg. for enums it could automatically combine the branches
impl_has_span!(Identifier);
impl_has_span!(Parameter);
impl_has_span!(ModulePortInBlock);
impl_has_span!(StructField);
impl_has_span!(EnumDeclaration);
impl_has_span!(EnumVariant);

impl HasSpan for ModulePortItem {
    fn span(&self) -> Span {
        match self {
            ModulePortItem::Single(port) => port.span,
            ModulePortItem::Block(block) => block.span,
        }
    }
}

impl<I: HasSpan> HasSpan for ExtraItem<I> {
    fn span(&self) -> Span {
        match self {
            ExtraItem::Inner(item) => item.span(),
            ExtraItem::Declaration(decl) => decl.span(),
            ExtraItem::If(if_stmt) => if_stmt.span,
        }
    }
}

impl<V> HasSpan for ArrayLiteralElement<Spanned<V>> {
    fn span(&self) -> Span {
        match self {
            ArrayLiteralElement::Spread(span, value) => span.join(value.span),
            ArrayLiteralElement::Single(value) => value.span,
        }
    }
}

impl<V> HasSpan for ArrayLiteralElement<Box<Spanned<V>>> {
    fn span(&self) -> Span {
        match self {
            ArrayLiteralElement::Spread(span, value) => span.join(value.span),
            ArrayLiteralElement::Single(value) => value.span,
        }
    }
}

impl<T> HasSpan for MaybeIdentifier<Spanned<T>> {
    fn span(&self) -> Span {
        match self {
            &MaybeIdentifier::Dummy(span) => span,
            MaybeIdentifier::Identifier(id) => id.span,
        }
    }
}

impl HasSpan for MaybeIdentifier<Identifier> {
    fn span(&self) -> Span {
        match self {
            MaybeIdentifier::Dummy(span) => *span,
            MaybeIdentifier::Identifier(id) => id.span,
        }
    }
}

impl HasSpan for GeneralIdentifier {
    fn span(&self) -> Span {
        match self {
            GeneralIdentifier::Simple(id) => id.span,
            GeneralIdentifier::FromString(span, _) => *span,
        }
    }
}

impl HasSpan for MaybeIdentifier<GeneralIdentifier> {
    fn span(&self) -> Span {
        match self {
            MaybeIdentifier::Dummy(span) => *span,
            MaybeIdentifier::Identifier(id) => id.span(),
        }
    }
}

impl<V> HasSpan for CommonDeclaration<V> {
    fn span(&self) -> Span {
        match self {
            CommonDeclaration::Named(decl) => decl.kind.span(),
            CommonDeclaration::ConstBlock(decl) => decl.span(),
        }
    }
}

impl HasSpan for ConstBlock {
    fn span(&self) -> Span {
        let ConstBlock { span_keyword, block } = self;
        span_keyword.join(block.span)
    }
}

impl HasSpan for CommonDeclarationNamedKind {
    fn span(&self) -> Span {
        match self {
            CommonDeclarationNamedKind::Type(decl) => decl.span,
            CommonDeclarationNamedKind::Const(decl) => decl.span,
            CommonDeclarationNamedKind::Struct(decl) => decl.span,
            CommonDeclarationNamedKind::Enum(decl) => decl.span,
            CommonDeclarationNamedKind::Function(decl) => decl.span,
        }
    }
}
