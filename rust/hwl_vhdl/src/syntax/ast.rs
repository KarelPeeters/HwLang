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
    Package(PackageDeclaration),
    PackageBody(PackageBody),
    Type(TypeDeclaration),
    Subtype(SubTypeDeclaration),
    Constant(ConstantDeclaration),
    Signal(SignalDeclaration),
    Variable(VariableDeclaration),
    File(FileDeclaration),
    Alias(AliasDeclaration),
    Attribute(AttributeDeclaration),
    Component(ComponentDeclaration),
    Procedure(ProcedureDeclaration),
    Function(FunctionDeclaration),
    ProcedureBody(ProcedureBody),
    FunctionBody(FunctionBody),
    Use(UseClause),
}

#[derive(Debug)]
pub enum EntityStatement {
    Concurrent(ConcurrentStatement),
}

// LRM 3.3 Architecture bodies
#[derive(Debug)]
pub struct ArchitectureBody {
    pub name: Identifier,
    pub entity_name: Identifier,
    pub decl: Vec<BlockDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum BlockDeclarativeItem {
    Package(PackageDeclaration),
    PackageBody(PackageBody),
    Type(TypeDeclaration),
    Subtype(SubTypeDeclaration),
    Constant(ConstantDeclaration),
    Signal(SignalDeclaration),
    Variable(VariableDeclaration),
    File(FileDeclaration),
    Alias(AliasDeclaration),
    Attribute(AttributeDeclaration),
    Component(ComponentDeclaration),
    Procedure(ProcedureDeclaration),
    Function(FunctionDeclaration),
    ProcedureBody(ProcedureBody),
    FunctionBody(FunctionBody),
    Use(UseClause),
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
    Package(PackageDeclaration),
    Type(TypeDeclaration),
    Subtype(SubTypeDeclaration),
    Constant(ConstantDeclaration),
    Signal(SignalDeclaration),
    Variable(VariableDeclaration),
    File(FileDeclaration),
    Alias(AliasDeclaration),
    Component(ComponentDeclaration),
    Attribute(AttributeDeclaration),
    Procedure(ProcedureDeclaration),
    Function(FunctionDeclaration),
    ProcedureBody(ProcedureBody),
    FunctionBody(FunctionBody),
    Use(UseClause),
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
    Package(PackageDeclaration),
    PackageBody(PackageBody),
    Type(TypeDeclaration),
    Subtype(SubTypeDeclaration),
    Constant(ConstantDeclaration),
    Variable(VariableDeclaration),
    File(FileDeclaration),
    Alias(AliasDeclaration),
    Attribute(AttributeDeclaration),
    Procedure(ProcedureDeclaration),
    Function(FunctionDeclaration),
    ProcedureBody(ProcedureBody),
    FunctionBody(FunctionBody),
    Use(UseClause),
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
    pub range: Expression,
}

#[derive(Debug)]
pub enum Range {
    Attribute(/*TODO*/),
    Simple(SimpleRange),
    Expression(Expression),
}

#[derive(Debug)]
pub struct SimpleRange {
    pub left: Box<Expression>,
    pub direction: RangeDirection,
    pub right: Box<Expression>,
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

// LRM 5.3 Composite types
#[derive(Debug)]
pub enum CompositeTypeDefinition {
    Array(ArrayTypeDefinition),
    Record(/*TODO*/),
}

// LRM 5.3.2 Array types
#[derive(Debug)]
pub enum ArrayTypeDefinition {
    Unbounded(UnboundedArrayTypeDefinition),
    Constrained(ConstrainedArrayTypeDefinition),
}
#[derive(Debug)]
pub struct UnboundedArrayTypeDefinition {
    pub index_types: NonEmptyVec<IndexSubTypeDefinition>,
    pub element_type: SubTypeIndication,
}
#[derive(Debug)]
pub struct ConstrainedArrayTypeDefinition {
    pub index_constraint: IndexConstraint,
    pub element_type: SubTypeIndication,
}

#[derive(Debug)]
pub struct IndexSubTypeDefinition {
    pub type_mark: TypeMark,
}
#[derive(Debug)]
pub struct ArrayConstraint {
    pub index_constraint: IndexConstraintOrOpen,
    pub array_element_constraint: Option<Box<ElementConstraint>>,
}
#[derive(Debug)]
pub enum IndexConstraintOrOpen {
    Open,
    Constraint(IndexConstraint),
}
#[derive(Debug)]
pub struct IndexConstraint {
    pub ranges: NonEmptyVec<Expression>,
}

// LRM 5.3.3 Record types

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
    Composite(CompositeTypeDefinition),
}

// LRM 6.3 Subtype declarations
#[derive(Debug)]
pub struct SubTypeDeclaration {
    pub name: Identifier,
    pub indication: SubTypeIndication,
}

pub type SubTypeIndication = Expression;

pub type TypeMark = Name;

#[derive(Debug)]
pub enum Constraint {
    Range(RangeConstraint),
    Array(ArrayConstraint),
    Record(/*TODO*/),
}
#[derive(Debug)]
pub enum ElementConstraint {
    Array(ArrayConstraint),
    Record(/*TODO*/),
}

// LRM 6.4 Objects
// LRM 6.4.2 Object declarations
// LRM 6.4.2.2 Constant declarations
#[derive(Debug)]
pub struct ConstantDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
    pub init: Option<Expression>,
}

// LRM 6.4.2.3 Signal declarations
#[derive(Debug)]
pub struct SignalDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
    pub kind: Option<SignalKind>,
    pub init: Option<Expression>,
}

#[derive(Debug, Copy, Clone)]
pub enum SignalKind {
    Register,
    Bus,
}

// LRM 6.4.2.4 Variable declarations
#[derive(Debug)]
pub struct VariableDeclaration {
    pub shared: bool,
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
    pub init: Option<Expression>,
}

// LRM 6.4.2.5 File declarations
#[derive(Debug)]
pub struct FileDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
}

// LRM 6.6 Alias declarations
#[derive(Debug)]
pub struct AliasDeclaration {
    pub name: Identifier,
    pub ty: Option<SubTypeIndication>,
    pub target: Name,
}

// LRM 7.2 Attribute declarations
#[derive(Debug)]
pub struct AttributeDeclaration {
    pub name: Identifier,
    pub ty: Name,
}

// LRM 6.8 Component declarations
#[derive(Debug)]
pub struct ComponentDeclaration {
    pub name: Identifier,
    pub generic: Option<GenericClause>,
    pub port: Option<PortClause>,
    pub end_name: Option<Identifier>,
}

// LRM 6.5 Interface declarations
// LRM 6.5.2 Interface object declarations
#[derive(Debug)]
pub struct InterfaceConstantDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: Expression,
    pub init: Option<Expression>,
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
        init: Option<Expression>,
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

// LRM 4.2 Subprogram declarations
#[derive(Debug)]
pub struct ProcedureDeclaration {
    pub name: Identifier,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
}

#[derive(Debug)]
pub struct FunctionDeclaration {
    pub name: Identifier,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
    pub return_type: SubTypeIndication,
}

// LRM 4.3 Subprogram bodies
#[derive(Debug)]
pub struct ProcedureBody {
    pub name: Identifier,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
    pub decl: Vec<SubprogramDeclarativeItem>,
    pub stmt: Vec<SequentialStatement>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub struct FunctionBody {
    pub name: Identifier,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
    pub return_type: SubTypeIndication,
    pub decl: Vec<SubprogramDeclarativeItem>,
    pub stmt: Vec<SequentialStatement>,
    pub end_name: Option<Identifier>,
}

// LRM 4.2.1 Formal parameters
#[derive(Debug)]
pub struct SubprogramParameterDeclaration {
    pub class: Option<SubprogramParameterClass>,
    pub names: NonEmptyVec<Identifier>,
    pub mode: Option<Mode>,
    pub ty: SubTypeIndication,
    pub init: Option<Expression>,
}

#[derive(Debug, Copy, Clone)]
pub enum SubprogramParameterClass {
    Constant,
    Variable,
    Signal,
}

#[derive(Debug)]
pub enum SubprogramDeclarativeItem {
    Package(PackageDeclaration),
    PackageBody(PackageBody),
    Type(TypeDeclaration),
    Subtype(SubTypeDeclaration),
    Constant(ConstantDeclaration),
    Variable(VariableDeclaration),
    File(FileDeclaration),
    Alias(AliasDeclaration),
    Attribute(AttributeDeclaration),
    Procedure(ProcedureDeclaration),
    Function(FunctionDeclaration),
    ProcedureBody(ProcedureBody),
    FunctionBody(FunctionBody),
    Use(UseClause),
}

// LRM 8 Names
// LRM 8.1 General
#[derive(Debug)]
pub enum Name {
    // TODO
    Simple(Identifier),
    Selected(Box<SelectedName>),
}

// LRM 8.3 Selected names
#[derive(Debug)]
pub struct SelectedName {
    pub prefix: Name,
    pub suffix: Suffix,
}
#[derive(Debug)]
pub enum Suffix {
    Identifier(Identifier),
    All,
}

// LRM 9 Expressions
// For expressions and related constructs we take a complete different approach from the LRM. The LRM grammars are
// trying to capture too many semantics already. This makes it hard to parse VHDL without already resolving symbols and
// types, which we want to avoid.
#[derive(Debug)]
pub enum Expression {
    // LRM 6.3 Subtype declarations
    SubtypeIndication {
        type_mark: TypeMark,
        constraint: Box<Constraint>,
    },

    // LRM  5.2 Scalar types
    Range {
        left: Box<Expression>,
        direction: RangeDirection,
        right: Box<Expression>,
    },

    // LRM 9 Expressions
    Conditional(ConditionalExpression),
    Unaffected,
    Binary {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Unary {
        op: UnaryOperator,
        inner: Box<Expression>,
    },
    Signed {
        sign: Sign,
        inner: Box<Expression>,
    },
    // primary
    Name(Name),
    Call {
        name: Name,
        args: Vec<Expression>,
    },
    Attribute {
        name: Name,
        attr: Identifier,
    },
    PhysicalLiteral {
        value: AbstractLiteral,
        unit: Identifier,
    },
    DecimalLiteral,
    StringLiteral,
    CharLiteral,
}

#[derive(Debug)]
pub struct ConditionalExpression {
    pub value_first: Box<Expression>,
    pub branches: Vec<ConditionalExpressionBranch<Box<Expression>>>,
    pub condition_final: Option<Box<Expression>>,
}
#[derive(Debug)]
pub struct ConditionalExpressionBranch<T> {
    pub condition: Expression,
    pub value_else: T,
}

#[derive(Debug, Copy, Clone)]
pub enum BinaryOperator {
    Relational(RelationalOperator),
    Logical(LogicalOperator),
    Shift(ShiftOperator),
    Adding(AddingOperator),
    Multiplying(MultiplyingOperator),
    Power,
}

pub fn build_binary_op(op: BinaryOperator, left: Expression, right: Expression) -> Expression {
    Expression::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
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

#[derive(Debug, Copy, Clone)]
pub enum UnaryOperator {
    Condition,
    Abs,
    Not,
    Logical(LogicalOperator),
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
    Block(BlockStatement),
    Process(ProcessStatement),
    ProcedureCall(ConcurrentProcedureCallStatement),
    Assertion(ConcurrentAssertionStatement),
    SignalAssignment(ConcurrentSignalAssignmentStatement),
    ComponentInstantiation(ComponentInstantiationStatement),
    Generate(GenerateStatement),
}

// LRM 11.2 Block statements
#[derive(Debug)]
pub struct BlockStatement {
    pub label: Option<Identifier>,
    pub guard_expression: Option<Expression>,
    pub decl: Vec<BlockDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
    pub end_label: Option<Identifier>,
}

// LRM 11.3 Process statements
#[derive(Debug)]
pub struct ProcessStatement {
    pub label: Option<Identifier>,
    pub postponed: bool,
    pub sensitivity: Option<ProcessSensitivityList>,
    pub decl: Vec<ProcessDeclarativeItem>,
    pub stmt: Vec<SequentialStatement>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub enum ProcessSensitivityList {
    Names(NonEmptyVec<Name>),
    All,
}

#[derive(Debug)]
pub enum ProcessDeclarativeItem {
    Package(PackageDeclaration),
    PackageBody(PackageBody),
    Type(TypeDeclaration),
    Subtype(SubTypeDeclaration),
    Constant(ConstantDeclaration),
    Variable(VariableDeclaration),
    File(FileDeclaration),
    Alias(AliasDeclaration),
    Attribute(AttributeDeclaration),
    Procedure(ProcedureDeclaration),
    Function(FunctionDeclaration),
    ProcedureBody(ProcedureBody),
    FunctionBody(FunctionBody),
    Use(UseClause),
}

#[derive(Debug)]
pub enum SequentialStatement {
    Wait(WaitStatement),
    Assertion(AssertionStatement),
    Report(ReportStatement),
    SignalAssignment(SequentialSignalAssignmentStatement),
    VariableAssignment(VariableAssignmentStatement),
    ProcedureCall(ProcedureCallStatement),
    If(IfStatement),
    Case(CaseStatement),
    Loop(LoopStatement),
    Next(NextStatement),
    Exit(ExitStatement),
    Return(ReturnStatement),
    Null,
    Block(SequentialBlockStatement),
}

// LRM 11.4 Concurrent procedure call statements
#[derive(Debug)]
pub struct ConcurrentProcedureCallStatement {
    pub label: Option<Identifier>,
    pub postponed: bool,
    pub procedure: Name,
    pub args: Vec<Expression>,
}

// LRM 11.5 Concurrent assertion statements
#[derive(Debug)]
pub struct ConcurrentAssertionStatement {
    pub label: Option<Identifier>,
    pub postponed: bool,
    pub condition: Expression,
    pub report: Option<Expression>,
    pub severity: Option<Expression>,
}

// LRM 10 Sequential statements
#[derive(Debug)]
pub struct WaitStatement {
    pub label: Option<Identifier>,
    pub sensitivity: Option<NonEmptyVec<Name>>,
    pub condition: Option<Expression>,
    pub timeout: Option<Expression>,
}

#[derive(Debug)]
pub struct AssertionStatement {
    pub label: Option<Identifier>,
    pub condition: Expression,
    pub report: Option<Expression>,
    pub severity: Option<Expression>,
}

#[derive(Debug)]
pub struct ReportStatement {
    pub label: Option<Identifier>,
    pub report: Expression,
    pub severity: Option<Expression>,
}

#[derive(Debug)]
pub struct SequentialSignalAssignmentStatement {
    pub label: Option<Identifier>,
    pub target: Target,
    pub delay: Option<DelayMechanism>,
    pub waveform: Waveform,
}

#[derive(Debug)]
pub struct VariableAssignmentStatement {
    pub label: Option<Identifier>,
    pub target: Target,
    pub value: Expression,
}

#[derive(Debug)]
pub struct ProcedureCallStatement {
    pub label: Option<Identifier>,
    pub procedure: Name,
    pub args: Vec<Expression>,
}

#[derive(Debug)]
pub struct IfStatement {
    pub label: Option<Identifier>,
    pub first_condition: Expression,
    pub first_body: Vec<SequentialStatement>,
    pub elsif: Vec<(Expression, Vec<SequentialStatement>)>,
    pub else_body: Option<Vec<SequentialStatement>>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub struct CaseStatement {
    pub label: Option<Identifier>,
    pub expression: Expression,
    pub alternatives: NonEmptyVec<CaseAlternative>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub struct CaseAlternative {
    pub choices: NonEmptyVec<Choice>,
    pub body: Vec<SequentialStatement>,
}

#[derive(Debug)]
pub enum Choice {
    Expr(Expression),
    Others,
}

#[derive(Debug)]
pub struct LoopStatement {
    pub label: Option<Identifier>,
    pub scheme: LoopScheme,
    pub body: Vec<SequentialStatement>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub enum LoopScheme {
    Infinite,
    While(Expression),
    For {
        param: Identifier,
        range: Expression,
    },
}

#[derive(Debug)]
pub struct NextStatement {
    pub label: Option<Identifier>,
    pub loop_label: Option<Identifier>,
    pub condition: Option<Expression>,
}

#[derive(Debug)]
pub struct ExitStatement {
    pub label: Option<Identifier>,
    pub loop_label: Option<Identifier>,
    pub condition: Option<Expression>,
}

#[derive(Debug)]
pub enum ReturnStatement {
    Plain {
        label: Option<Identifier>,
    },
    Value {
        label: Option<Identifier>,
        value: Expression,
    },
}

#[derive(Debug)]
pub struct SequentialBlockStatement {
    pub label: Option<Identifier>,
    pub decl: Vec<ProcessDeclarativeItem>,
    pub stmt: Vec<SequentialStatement>,
    pub end_label: Option<Identifier>,
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

// LRM 11.7 Component instantiation statements
#[derive(Debug)]
pub struct ComponentInstantiationStatement {
    pub label: Identifier,
    pub unit: Name,
    pub generic_map: Option<Vec<Expression>>,
    pub port_map: Option<Vec<Expression>>,
}

// LRM 11.8 Generate statements
#[derive(Debug)]
pub enum GenerateStatement {
    For(ForGenerateStatement),
    If(IfGenerateStatement),
}

#[derive(Debug)]
pub struct ForGenerateStatement {
    pub label: Identifier,
    pub param: Identifier,
    pub range: Expression,
    pub decl: Vec<BlockDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub struct IfGenerateStatement {
    pub label: Identifier,
    pub first_condition: Expression,
    pub first_body: GenerateBody,
    pub elsif: Vec<(Expression, GenerateBody)>,
    pub else_body: Option<GenerateBody>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub struct GenerateBody {
    pub decl: Vec<BlockDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
}

// LRM 12 Scope and visibility
// LRM 12.4 Use clauses
#[derive(Debug)]
pub struct UseClause {
    pub names: NonEmptyVec<SelectedName>,
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
    pub names: NonEmptyVec<Identifier>,
}

// LRM 13.4 Context clauses
#[derive(Debug)]
pub struct ContextClause {
    pub items: Vec<ContextItem>,
}

#[derive(Debug)]
pub enum ContextItem {
    Library(LibraryClause),
    Use(UseClause),
    Context(/*TODO*/),
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
