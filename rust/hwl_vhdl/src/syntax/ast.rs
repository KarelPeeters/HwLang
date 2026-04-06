use hwl_common::pos::Span;
use hwl_common::util::data::NonEmptyVec;

// LRM 3.2 Entity declarations
#[derive(Debug)]
pub struct EntityDeclaration {
    pub name: Identifier,
    pub generic: Option<GenericClause>,
    pub port: Option<PortClause>,
    pub decl: Vec<EntityDeclarativeItem>,
    pub stmt: Option<Vec<EntityStatement>>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum EntityDeclarativeItem {
    Type(TypeDeclaration),
}

#[derive(Debug)]
pub enum EntityStatement {
    // TODO
}

// LRM 3.3 Architecture bodies
#[derive(Debug)]
pub struct ArchitectureBody {
    pub name: Identifier,
    pub entity_name: Identifier,
    pub decl: Vec<ArchitectureDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum ArchitectureDeclarativeItem {
    // TODO
}

// LRM 4 Subprograms and packages
// LRM 4.7 Package declarations
#[derive(Debug)]
pub struct PackageDeclaration {
    pub name: Identifier,
    pub generic: Option<GenericClause>,
    pub decl: Vec<PackageDeclarativeItem>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum PackageDeclarativeItem {
    Type(TypeDeclaration),
    Constant(ConstantDeclaration),
}

// LRM 4.8 Package bodies
#[derive(Debug)]
pub struct PackageBody {
    pub name: Identifier,
    pub decl: Vec<PackageBodyDeclarativeItem>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum PackageBodyDeclarativeItem {
    Type(TypeDeclaration),
    Constant(ConstantDeclaration),
}

// LRM 5 Types
// LRM 5.2 Scalar types
#[derive(Debug)]
pub enum ScalarTypeDefinition {
    Enum(EnumTypeDefinition),
    Integer(IntegerTypeDefinition),
    Physical(PhysicalTypeDefinition),
}

#[derive(Debug)]
pub struct RangeConstraint {
    pub range: Range,
}

#[derive(Debug)]
pub enum Range {
    Attribute(/*TODO*/),
    Simple(SimpleRange),
    Expression(Expression),
}

#[derive(Debug)]
pub struct SimpleRange {
    pub left: SimpleExpression,
    pub direction: RangeDirection,
    pub right: SimpleExpression,
}

#[derive(Debug, Copy, Clone)]
pub enum RangeDirection {
    To,
    DownTo,
}

// LRM 5.2.2 Enumeration types
#[derive(Debug)]
pub struct EnumTypeDefinition {
    pub literals: NonEmptyVec<EnumLiteral>,
}

#[derive(Debug)]
pub enum EnumLiteral {
    Identifier(Identifier),
    CharLiteral(/*TODO*/),
}

// LRM 5.2.3 Integer types
#[derive(Debug)]
pub struct IntegerTypeDefinition {
    pub range: RangeConstraint,
}

// LRM 5.2.4 Physical types
#[derive(Debug)]
pub struct PhysicalTypeDefinition {
    pub range: RangeConstraint,
    pub primary_unit: Identifier,
    pub secondary_units: Vec<SecondaryUnitDeclaration>,
    pub end_name: Option<Identifier>,
}
#[derive(Debug)]
pub struct SecondaryUnitDeclaration {
    // TODO name?
    pub name: Identifier,
    pub value: PhysicalLiteral,
}
#[derive(Debug)]
pub struct PhysicalLiteral {
    pub value: AbstractLiteral,
    pub unit: Identifier,
}

// LRM 6 Declarations

// LRM 6.2 Type declarations

#[derive(Debug)]
pub enum TypeDeclaration {
    Full(FullTypeDeclaration),
}

#[derive(Debug)]
pub struct FullTypeDeclaration {
    pub name: Identifier,
    pub def: TypeDefinition,
}

#[derive(Debug)]
pub enum TypeDefinition {
    Scalar(ScalarTypeDefinition),
}

// LRM 6.3 Subtype declarations
#[derive(Debug)]
pub struct SubTypeDeclaration {
    pub name: Identifier,
    pub indication: SubTypeIndication,
}

#[derive(Debug)]
pub struct SubTypeIndication {
    // TODO resolution
    // TODO name instead of identifier
    pub type_mark: Identifier,
    pub constraint: Option<Constraint>,
}

#[derive(Debug)]
pub enum Constraint {
    Range(RangeConstraint),
}

// LRM 6.4 Objects
// LRM 6.4.2 Object declarations
// LRM 6.4.2.2 Constant declarations
#[derive(Debug)]
pub struct ConstantDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
    pub init: Option<ConditionalExpression>,
}

// LRM 6.5 Interface declarations
// LRM 6.5.2 Interface object declarations
#[derive(Debug)]
pub struct InterfaceConstantDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: InterfaceTypeIndication,
    pub init: Option<ConditionalExpression>,
}

#[derive(Debug)]
pub struct InterfaceSignalDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub mode_indication: ModeIndication,
}

#[derive(Debug)]
pub enum InterfaceTypeIndication {
    Subtype(SubTypeIndication),
    Unspecified,
}

#[derive(Debug)]
pub enum ModeIndication {
    Simple {
        mode: Option<Mode>,
        ty: InterfaceTypeIndication,
        bus: bool,
        init: Option<ConditionalExpression>,
    },
    RecordView(/*TODO*/),
}

#[derive(Debug, Copy, Clone)]
pub enum Mode {
    In,
    Out,
    Inout,
    Buffer,
    Linkage,
}

// LRM 6.5.3 Interface type declarations

// LRM 6.5.4 Interface subprogram declarations

// LRM 6.5.5 Interface package declarations

// LRM 6.5.6 Interface lists
// The LRM mixes all interface declarations together, but that makes the grammar ambiguous.
// We keep generics and port separated instead.
#[derive(Debug)]
pub struct GenericClause {
    pub list: NonEmptyVec<GenericInterfaceDeclaration>,
}

#[derive(Debug)]
pub enum GenericInterfaceDeclaration {
    Constant(InterfaceConstantDeclaration),
    Type(/*TODO*/),
    Subprogram(/*TODO*/),
    Package(/*TODO*/),
}

#[derive(Debug)]
pub struct PortClause {
    pub list: NonEmptyVec<PortInterfaceDeclaration>,
}

#[derive(Debug)]
pub enum PortInterfaceDeclaration {
    Signal(InterfaceSignalDeclaration),
    Variable(/*TODO*/),
}

// LRM 9 Expressions
#[derive(Debug)]
pub struct ConditionalOrUnAffectedExpression {
    pub conditional: ConditionalExpression<ExpressionOrUnAffected>,
    pub final_condition: Option<Expression>,
}
#[derive(Debug)]
pub enum ExpressionOrUnAffected {
    Expression(Expression),
    Unaffected,
}

#[derive(Debug)]
pub struct ConditionalExpression<T = Expression> {
    pub value_first: T,
    pub branches: Vec<ConditionalExpressionBranch<T>>,
}
#[derive(Debug)]
pub struct ConditionalExpressionBranch<T> {
    pub condition: Expression,
    pub value_else: T,
}

// TODO flatten this into a single enum, we only need these levels in the parser itself
#[derive(Debug)]
pub enum Expression {
    ConditionOperator(PrimaryExpression),
    Logical(LogicalExpression),
}

#[derive(Debug)]
pub enum LogicalExpression {
    Relation(RelationExpression),

    And(RelationExpression, NonEmptyVec<RelationExpression>),
    Or(RelationExpression, NonEmptyVec<RelationExpression>),
    Nand(RelationExpression, RelationExpression),
    Nor(RelationExpression, RelationExpression),
    Xor(RelationExpression, NonEmptyVec<RelationExpression>),
    Xnor(RelationExpression, NonEmptyVec<RelationExpression>),
}

#[derive(Debug)]
pub struct RelationExpression {
    pub left: ShiftExpression,
    pub op_right: Option<(RelationalOperator, ShiftExpression)>,
}

#[derive(Debug)]
pub struct ShiftExpression {
    pub left: SimpleExpression,
    pub op_right: Option<(ShiftOperator, SimpleExpression)>,
}

#[derive(Debug)]
pub struct SimpleExpression {
    pub sign: Option<Sign>,
    pub left: TermExpression,
    pub op_right: Vec<(AddingOperator, TermExpression)>,
}

#[derive(Debug)]
pub struct TermExpression {
    pub left: Factor,
    pub op_right: Vec<(MultiplyingOperator, Factor)>,
}

#[derive(Debug)]
pub struct Factor {
    pub left: UnaryExpression,
    pub power_right: Option<UnaryExpression>,
}

#[derive(Debug)]
pub enum UnaryExpression {
    Primary(PrimaryExpression),
    Abs(PrimaryExpression),
    Not(PrimaryExpression),
    Logical(LogicalOperator, PrimaryExpression),
}

#[derive(Debug)]
pub enum PrimaryExpression {
    // TODO expand, maybe name is not even correct
    Name(Identifier),
    DecimalLiteral,
}

#[derive(Debug, Copy, Clone)]
pub enum LogicalOperator {
    And,
    Or,
    Nand,
    Nor,
    Xor,
    Xnor,
}

#[derive(Debug, Copy, Clone)]
pub enum RelationalOperator {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    LogicalEq,
    LogicalNeq,
    LogicalLt,
    LogicalLte,
    LogicalGt,
    LogicalGte,
}

#[derive(Debug, Copy, Clone)]
pub enum ShiftOperator {
    Sll,
    Srl,
    Sla,
    Sra,
    Rol,
    Ror,
}

#[derive(Debug, Copy, Clone)]
pub enum Sign {
    Plus,
    Minus,
}

#[derive(Debug, Copy, Clone)]
pub enum AddingOperator {
    Add,
    Sub,
    And,
}

#[derive(Debug, Copy, Clone)]
pub enum MultiplyingOperator {
    Mul,
    Div,
    Mod,
    Rem,
}

// LRM 9.3 Operands
// LRM 9.3.2 Literals
pub enum Literal {
    // TODO expand
    DecimalLiteral,
}

// LRM 10 Sequential statements
// LRM 10.5 Simple assignment statement
// LRM 10.5.2 Simple signal assignments
#[derive(Debug)]
pub enum DelayMechanism {
    Transport,
    Inertial { reject_time: Option<Expression> },
}

#[derive(Debug)]
pub enum Target {
    // TODO LRM syntax implies this can be many other things too, is that true?
    Name(Identifier),
    Aggregate(/*TODO*/),
}

#[derive(Debug)]
pub enum Waveform {
    Elements(NonEmptyVec<WaveformElement>),
    Unaffected,
}

// LRM 10.5.2.2 Executing a simple assignment statement
#[derive(Debug)]
pub struct WaveformElement {
    pub value: WaveformElementValue,
    pub after: Option<Expression>,
}

#[derive(Debug)]
pub enum WaveformElementValue {
    Expression(Expression),
    Null,
}

// LRM 11 Concurrent statements
#[derive(Debug)]
pub enum ConcurrentStatement {
    Block(/*TODO*/),
    Process(/*TODO*/),
    ProcedureCall(/*TODO*/),
    Assertion(/*TODO*/),
    SignalAssignment(ConcurrentSignalAssignmentStatement),
    ComponentInstantiation(/*TODO*/),
    Generate(/*TODO*/),
}

// LRM 11.6 Concurrent signal assignment statements
#[derive(Debug)]
pub struct ConcurrentSignalAssignmentStatement {
    pub label: Option<Identifier>,
    pub postponed: bool,
    pub kind: ConcurrentSignalAssignmentKind,
}

#[derive(Debug)]
pub enum ConcurrentSignalAssignmentKind {
    Simple {
        target: Target,
        guarded: bool,
        delay: Option<DelayMechanism>,
        waveform: Waveform,
    },
    Conditional(/*TODO*/),
    Selected(/*TODO*/),
}

// LRM 13.1 Design units
#[derive(Debug)]
pub struct DesignFile {
    pub units: Vec<DesignUnit>,
}

#[derive(Debug)]
pub struct DesignUnit {
    pub context_clause: ContextClause,
    pub library_unit: LibraryUnit,
}

#[derive(Debug)]
pub enum LibraryUnit {
    // primary unit
    EntityDeclaration(EntityDeclaration),
    ConfigurationDeclaration(/*TODO*/),
    PackageDeclaration(PackageDeclaration),
    PackageInstantiationDeclaration(/*TODO*/),
    ContextDeclaration(/*TODO*/),

    // secondary unit
    ArchitectureBody(ArchitectureBody),
    PackageBody(PackageBody),
}

// LRM 13.2 Design libraries
#[derive(Debug)]
pub struct LibraryClause {
    pub logical_name_list: NonEmptyVec<Identifier>,
}

// LRM 13.4 Context clauses
#[derive(Debug)]
pub struct ContextClause {
    // TODO
}

// LRM 15 Lexical elements

// LRM 15.4 Identifiers
#[derive(Debug)]
pub struct Identifier {
    pub span: Span,
}

// LRM 15.5 Abstract literals
#[derive(Debug)]
pub enum AbstractLiteral {
    Decimal(DecimalLiteral),
    Based(BasedLiteral),
}
#[derive(Debug)]
pub struct DecimalLiteral {
    pub span: Span,
}
#[derive(Debug)]
pub struct BasedLiteral {
    pub span: Span,
}
