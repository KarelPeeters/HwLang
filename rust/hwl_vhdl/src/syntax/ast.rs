use hwl_common::pos::Span;
use hwl_common::util::data::NonEmptyVec;

// LRM 3 Design entities and configurations
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
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramBody(SubprogramBody),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageBody(PackageBody),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ModeViewDeclaration(ModeViewDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    SignalDeclaration(SignalDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    DisconnectionSpecification(DisconnectionSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
}

// LRM 3.2.4 entity_statement
// Only assertion, passive procedure call, and passive process are allowed in entities.
#[derive(Debug)]
pub enum EntityStatement {
    Process(ProcessStatement),
    ProcedureCall(ConcurrentProcedureCallStatement),
    Assertion(ConcurrentAssertionStatement),
}

// LRM 3.3 Architecture bodies
#[derive(Debug)]
pub struct ArchitectureBody {
    pub name: Identifier,
    pub entity_name: Expression,
    pub decl: Vec<BlockDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
    pub end_name: Option<Identifier>,
}

// LRM 3.4 Configuration declarations
#[derive(Debug)]
pub struct ConfigurationDeclaration {
    pub name: Identifier,
    pub entity_name: Expression,
    pub decl: Vec<ConfigurationDeclarativeItem>,
    pub block_config: BlockConfiguration,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum ConfigurationDeclarativeItem {
    UseClause(UseClause),
    AttributeSpecification(AttributeSpecification),
    GroupDeclaration(GroupDeclaration),
}

// LRM 3.4.2 Block configuration
#[derive(Debug)]
pub struct BlockConfiguration {
    pub block_spec: Expression,
    pub use_clauses: Vec<UseClause>,
    pub items: Vec<ConfigurationItem>,
}

#[derive(Debug)]
pub enum ConfigurationItem {
    Block(BlockConfiguration),
    Component(ComponentConfiguration),
}

// LRM 3.4.3 Component configuration
#[derive(Debug)]
pub struct ComponentConfiguration {
    pub spec: ComponentSpecification,
    pub binding: Option<BindingIndication>,
    pub block_config: Option<BlockConfiguration>,
}

// LRM 3.3.2 Architecture declarative part
#[derive(Debug)]
pub enum BlockDeclarativeItem {
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramBody(SubprogramBody),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageBody(PackageBody),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ModeViewDeclaration(ModeViewDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    SignalDeclaration(SignalDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    ComponentDeclaration(ComponentDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    ConfigurationSpecification(ConfigurationSpecification),
    DisconnectionSpecification(DisconnectionSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
}

// LRM 4 Subprograms and packages

// LRM 4.2 Subprogram declarations
#[derive(Debug)]
pub struct SubprogramDeclaration {
    pub spec: SubprogramSpecification,
}

#[derive(Debug)]
pub enum SubprogramSpecification {
    Procedure(ProcedureSpecification),
    Function(FunctionSpecification),
}

#[derive(Debug)]
pub struct ProcedureSpecification {
    pub name: Designator,
    pub header: SubprogramHeader,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
}

#[derive(Debug)]
pub struct FunctionSpecification {
    pub purity: Option<FunctionPurity>,
    pub name: Designator,
    pub header: SubprogramHeader,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
    // LRM 4.2.1 return [return_identifier of] type_mark
    pub return_identifier: Option<Identifier>,
    pub return_type: SubTypeIndication,
}

#[derive(Debug)]
pub struct SubprogramHeader {
    pub generic: GenericList,
    pub generic_map: Vec<Expression>,
}

#[derive(Debug)]
pub enum SubprogramBody {
    Procedure(ProcedureBody),
    Function(FunctionBody),
}

#[derive(Debug)]
pub enum SubprogramKind {
    Procedure,
    Function,
}

#[derive(Debug)]
pub struct SubprogramInstantiationDeclaration {
    pub kind: SubprogramKind,
    pub designator: Designator,
    pub uninstantiated: Expression,
    pub signature: Option<Signature>,
    pub generic_map: Option<Vec<Expression>>,
}

// LRM 4.3 Subprogram bodies
#[derive(Debug)]
pub struct ProcedureBody {
    pub name: Designator,
    pub header: SubprogramHeader,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
    pub decl: Vec<SubprogramDeclarativeItem>,
    pub stmt: Vec<SequentialStatement>,
    pub end_name: Option<Designator>,
}

#[derive(Debug)]
pub struct FunctionBody {
    pub purity: Option<FunctionPurity>,
    pub name: Designator,
    pub header: SubprogramHeader,
    pub params: Option<NonEmptyVec<SubprogramParameterDeclaration>>,
    pub return_identifier: Option<Identifier>,
    pub return_type: SubTypeIndication,
    pub decl: Vec<SubprogramDeclarativeItem>,
    pub stmt: Vec<SequentialStatement>,
    pub end_name: Option<Designator>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FunctionPurity {
    Pure,
    Impure,
}

// LRM 4.7 Package declarations
#[derive(Debug)]
pub struct PackageDeclaration {
    pub name: Identifier,
    pub generic: Option<GenericClause>,
    pub generic_map: Option<Vec<Expression>>,
    pub decl: Vec<PackageDeclarativeItem>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum PackageDeclarativeItem {
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ModeViewDeclaration(ModeViewDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    SignalDeclaration(SignalDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    ComponentDeclaration(ComponentDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    DisconnectionSpecification(DisconnectionSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
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
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramBody(SubprogramBody),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageBody(PackageBody),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
}


// LRM 4.9 Package instantiation declarations
#[derive(Debug)]
pub struct PackageInstantiationDeclaration {
    pub name: Identifier,
    pub uninstantiated: Expression,
    pub generic_map: Option<Vec<Expression>>,
}

// LRM 5.6.2 Protected type declarations
#[derive(Debug)]
pub struct PrivateVariableDeclaration {
    pub decl: VariableDeclaration,
}

// LRM 6.5.2 Mode view declarations
#[derive(Debug)]
pub struct ModeViewDeclaration {
    pub name: Identifier,
    pub subtype: SubTypeIndication,
    pub elements: Vec<ModeViewElementDefinition>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub struct ModeViewElementDefinition {
    pub names: NonEmptyVec<Identifier>,
    pub indication: ElementModeIndication,
}

#[derive(Debug)]
pub enum ElementModeIndication {
    Mode(Mode),
    RecordView(Expression),
    ArrayView(Expression),
}

// LRM 7.4 Disconnection specification
#[derive(Debug)]
pub struct DisconnectionSpecification {
    pub guarded_signal_specification: GuardedSignalSpecification,
    pub after: Expression,
}

#[derive(Debug)]
pub struct GuardedSignalSpecification {
    pub signals: SignalList,
    pub ty: Name,
}

#[derive(Debug)]
pub enum SignalList {
    Names(NonEmptyVec<Name>),
    Others,
    All,
}

// LRM 7.3 Configuration specification
#[derive(Debug)]
pub struct ConfigurationSpecification {
    pub component_specification: ComponentSpecification,
    pub binding_indication: BindingIndication,
}

#[derive(Debug)]
pub struct ComponentSpecification {
    pub instantiation_list: InstantiationList,
    pub component_name: Name,
}

#[derive(Debug)]
pub enum InstantiationList {
    Labels(NonEmptyVec<Identifier>),
    Others,
    All,
}

// LRM 7.3.2 Binding indication
#[derive(Debug)]
pub struct BindingIndication {
    pub entity_aspect: Option<EntityAspect>,
    pub generic_map: Option<Vec<Expression>>,
    pub port_map: Option<Vec<Expression>>,
}

// LRM 7.3.2.2 Entity aspect
#[derive(Debug)]
pub enum EntityAspect {
    Entity(Name, Option<Identifier>),
    Configuration(Name),
    Open,
}

// LRM 6.9 Group template declarations
#[derive(Debug)]
pub struct GroupTemplateDeclaration {
    pub name: Identifier,
    pub entries: NonEmptyVec<EntityClassEntry>,
}

#[derive(Debug)]
pub struct EntityClassEntry {
    pub class: EntityClass,
    pub boxed: bool,
}

// LRM 6.10 Group declarations
#[derive(Debug)]
pub struct GroupDeclaration {
    pub name: Identifier,
    pub template: Name,
    pub constituents: NonEmptyVec<GroupConstituent>,
}

#[derive(Debug)]
pub enum GroupConstituent {
    Name(Name),
    CharLiteral,
}

// LRM 5 Types
// LRM 5.2 Scalar types
#[derive(Debug)]
pub enum ScalarTypeDefinition {
    Enum(EnumTypeDefinition),
    // LRM 5.2.3 / 5.2.5: integer and floating types are syntactically identical (range_constraint),
    // disambiguation requires semantic analysis
    Numeric(NumericTypeDefinition),
    Physical(PhysicalTypeDefinition),
}

#[derive(Debug)]
pub struct RangeConstraint {
    pub range: Expression,
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
    CharLiteral(Span),
}

// LRM 5.2.3 Integer types / LRM 5.2.5 Floating-point types
#[derive(Debug)]
pub struct NumericTypeDefinition {
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
    pub value: Option<AbstractLiteral>,
    pub unit: Identifier,
}

// LRM 5.3 Composite types
#[derive(Debug)]
pub enum CompositeTypeDefinition {
    Array(ArrayTypeDefinition),
    Record(RecordTypeDefinition),
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
#[derive(Debug)]
pub struct RecordTypeDefinition {
    pub elements: Vec<ElementDeclaration>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub struct ElementDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
}

// LRM 5.6.2 Protected type declarations

// LRM 5.6.3 Protected type bodies

// LRM 6 Declarations
// LRM 6.2 Type declarations
#[derive(Debug)]
pub enum TypeDeclaration {
    Full(FullTypeDeclaration),
    Incomplete(IncompleteTypeDeclaration),
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
    Access(AccessTypeDefinition),
    File(FileTypeDefinition),
    Protected(ProtectedTypeDefinition),
    ProtectedBody(ProtectedTypeBody),
}

#[derive(Debug)]
pub struct AccessTypeDefinition {
    pub designated_subtype: SubTypeIndication,
}

// LRM 5.5 File types
#[derive(Debug)]
pub struct FileTypeDefinition {
    pub type_mark: TypeMark,
}

// LRM 6.2 Incomplete type declarations
#[derive(Debug)]
pub struct IncompleteTypeDeclaration {
    pub name: Identifier,
}

#[derive(Debug)]
pub struct ProtectedTypeDefinition {
    pub decl: Vec<ProtectedTypeDeclarativeItem>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum ProtectedTypeDeclarativeItem {
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    AttributeSpecification(AttributeSpecification),
    UseClause(UseClause),
    PrivateVariableDeclaration(PrivateVariableDeclaration),
    AliasDeclaration(AliasDeclaration),
}

#[derive(Debug)]
pub struct ProtectedTypeBody {
    pub decl: Vec<ProtectedTypeBodyDeclarativeItem>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum ProtectedTypeBodyDeclarativeItem {
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramBody(SubprogramBody),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageBody(PackageBody),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
}

// LRM 6.3 Subtype declarations
#[derive(Debug)]
pub struct SubTypeDeclaration {
    pub name: Identifier,
    pub indication: SubTypeIndication,
}

pub type SubTypeIndication = Expression;

pub type TypeMark = Expression;

// LRM 5.3.3 record_constraint
#[derive(Debug)]
pub enum Constraint {
    Range(RangeConstraint),
    Array(ArrayConstraint),
}

// LRM 5.3.3 element_constraint
#[derive(Debug)]
pub enum ElementConstraint {
    Array(ArrayConstraint),
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
    pub generic_map: Option<Vec<Expression>>,
    pub init: Option<Expression>,
}

// LRM 6.4.2.5 File declarations
#[derive(Debug)]
pub struct FileDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: SubTypeIndication,
    pub open_info: Option<FileOpenInformation>,
}

// LRM 6.4.2.5 File open information
#[derive(Debug)]
pub struct FileOpenInformation {
    pub open_kind: Option<Expression>,
    pub logical_name: Expression,
}

// LRM 6.6 Alias declarations
// LRM 4.5.3 Signatures
#[derive(Debug)]
pub struct Signature {
    pub parameter_types: Vec<Name>,
    pub return_type: Option<Name>,
}

#[derive(Debug)]
pub struct AliasDeclaration {
    pub name: Designator,
    pub ty: Option<SubTypeIndication>,
    pub target: Expression,
    pub signature: Option<Signature>,
}

// LRM 6.7 Attribute declarations
#[derive(Debug)]
pub struct AttributeDeclaration {
    pub name: Identifier,
    pub ty: Name,
}

// LRM 7.2 Attribute specification
#[derive(Debug)]
pub struct AttributeSpecification {
    pub name: Identifier,
    pub entities: NonEmptyVec<AttributeEntityDesignator>,
    pub entity_class: EntityClass,
    pub expr: Expression,
}

// LRM 7.2 entity_designator
#[derive(Debug)]
pub struct AttributeEntityDesignator {
    pub tag: EntityTag,
    pub signature: Option<Signature>,
}

#[derive(Debug)]
pub enum EntityTag {
    Name(Name),
    OperatorSymbol(Span),
    CharacterLiteral(Span),
    Others,
    All,
}

#[derive(Debug, Copy, Clone)]
pub enum EntityClass {
    Entity,
    Architecture,
    Configuration,
    Procedure,
    Function,
    Package,
    Type,
    Subtype,
    Constant,
    Signal,
    Variable,
    Component,
    Label,
    Literal,
    Units,
    Group,
    File,
    Property,
    Sequence,
    View,
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

// LRM 6.5.2 Interface variable declarations
#[derive(Debug)]
pub struct InterfaceVariableDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub mode_indication: ModeIndication,
}

// LRM 6.5.2 interface_type_indication
#[derive(Debug)]
pub enum InterfaceTypeIndication {
    Subtype(SubTypeIndication),
    // LRM 5.8 unspecified_type_indication ::= type is incomplete_type_definition
    Unspecified(IncompleteTypeDefinition),
}

#[derive(Debug)]
pub enum ModeIndication {
    Simple {
        mode: Option<Mode>,
        ty: InterfaceTypeIndication,
        bus: bool,
        init: Option<Expression>,
    },
    // LRM 6.5.2.2 record_mode_view_indication
    RecordView {
        view_name: Expression,
        subtype: Option<Expression>,
    },
    // LRM 6.5.2.2 array_mode_view_indication
    ArrayView {
        view_name: Expression,
        subtype: Option<Expression>,
    },
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
#[derive(Debug)]
pub struct InterfaceTypeDeclaration {
    pub name: Identifier,
    pub constraint: Option<IncompleteTypeDefinition>,
}

// LRM 5.4.2 Incomplete type definitions
#[derive(Debug)]
pub enum IncompleteTypeDefinition {
    Private,
    Scalar,
    Discrete,
    Integer,
    Floating,
    Access(Box<ElementIncompleteType>),
    File(Box<ElementIncompleteType>),
    Array { indexes: NonEmptyVec<ArrayIncompleteIndex>, element: Box<ElementIncompleteType> },
    Physical,
}

// LRM 5.4.2 element_incomplete_type_definition
#[derive(Debug)]
pub enum ElementIncompleteType {
    Subtype(Expression),
    Incomplete(IncompleteTypeDefinition),
}

#[derive(Debug)]
pub enum ArrayIncompleteIndex {
    Incomplete(IncompleteTypeDefinition),
    Unspecified,
    Range(Expression),
}

// LRM 6.5.4 Interface subprogram declarations

// LRM 6.5.5 Interface package declarations

// LRM 6.5.6 Interface lists
// The LRM mixes all interface declarations together, but that makes the grammar ambiguous.
// We keep generics and port separated instead.
#[derive(Debug)]
pub struct GenericClause {
    pub list: GenericList,
}
pub type GenericList = NonEmptyVec<GenericInterfaceDeclaration>;

#[derive(Debug)]
pub enum GenericInterfaceDeclaration {
    Constant(InterfaceConstantDeclaration),
    Type(InterfaceTypeDeclaration),
    Subprogram(InterfaceSubprogramDeclaration),
    Package(InterfacePackageDeclaration),
    // Not valid per LRM, but needed to parse syntactically for error recovery
    Signal(InterfaceSignalDeclaration),
    Variable(InterfaceVariableDeclaration),
    File(InterfaceFileDeclaration),
}

// LRM 6.5.4 Interface subprogram declarations
#[derive(Debug)]
pub struct InterfaceSubprogramDeclaration {
    pub spec: SubprogramSpecification,
    pub default: Option<InterfaceSubprogramDefault>,
}

#[derive(Debug)]
pub enum InterfaceSubprogramDefault {
    Name(Name),
    Box,
}

// LRM 6.5.5 Interface package declarations
#[derive(Debug)]
pub struct InterfacePackageDeclaration {
    pub name: Identifier,
    pub uninstantiated: Expression,
    pub generic_map: InterfacePackageGenericMap,
}

#[derive(Debug)]
pub enum InterfacePackageGenericMap {
    Map(Vec<Expression>),
    Box,
    Default,
}

#[derive(Debug)]
pub struct PortClause {
    pub list: NonEmptyVec<PortInterfaceDeclaration>,
}

#[derive(Debug)]
pub enum PortInterfaceDeclaration {
    Signal(InterfaceSignalDeclaration),
    Variable(InterfaceVariableDeclaration),
    Constant(InterfaceConstantDeclaration),
    File(InterfaceFileDeclaration),
}

// LRM 6.5.2 Interface file declarations
#[derive(Debug)]
pub struct InterfaceFileDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: Expression,
}

// LRM 4.2.1 Formal parameters
#[derive(Debug)]
pub enum SubprogramParameterDeclaration {
    Object {
        class: Option<SubprogramParameterClass>,
        names: NonEmptyVec<Identifier>,
        mode_indication: ModeIndication,
    },
    Type(InterfaceTypeDeclaration),
}

#[derive(Debug, Copy, Clone)]
pub enum SubprogramParameterClass {
    Constant,
    Variable,
    Signal,
    File,
}

#[derive(Debug)]
pub enum SubprogramDeclarativeItem {
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramBody(SubprogramBody),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageBody(PackageBody),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
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
    OperatorSymbol(Span),
    CharacterLiteral(Span),
    All,
    Attribute(Identifier),
}

// LRM 9 Expressions
// For expressions and related constructs we take a complete different approach from the LRM. The LRM grammars are
// trying to capture too many semantics already. This makes it hard to parse VHDL without already resolving symbols and
// types, which we want to avoid.
#[derive(Debug)]
pub enum Expression {
    // LRM 6.3 Subtype declarations
    SubtypeIndication {
        resolution: Option<Box<Expression>>,
        type_mark: Box<TypeMark>,
        constraint: Option<Box<Constraint>>,
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
    New(Box<Expression>),
    Null,
    Open,
    QualifiedExpression {
        type_mark: Box<Expression>,
        args: Vec<Expression>,
    },
    Call {
        callee: Box<Expression>,
        args: Vec<Expression>,
    },
    Select {
        value: Box<Expression>,
        suffix: Suffix,
    },
    Association {
        formal: Box<Expression>,
        actual: Box<Expression>,
    },
    Attribute {
        value: Box<Expression>,
        attr: Identifier,
        args: Vec<Expression>,
    },
    PhysicalLiteral {
        value: AbstractLiteral,
        unit: Identifier,
    },
    // LRM 8.7 External names
    ExternalName {
        class: ExternalNameClass,
        path: NonEmptyVec<ExternalPathElement>,
        ty: Box<Expression>,
    },
    BitStringLiteral(Span),
    Aggregate(NonEmptyVec<Expression>),
    OthersChoice,
    // LRM 9.3.3.3 Choices in aggregates (choice | choice | ...)
    Choices(NonEmptyVec<Expression>),
    DecimalLiteral(Span),
    BasedLiteral(Span),
    StringLiteral(Span),
    CharLiteral(Span),
    // LRM 6.5.7.1 Port map aspects
    Inertial(Box<Expression>),
    // LRM 4.5.3 Signature (name[signature])
    WithSignature {
        value: Box<Expression>,
        signature: Signature,
    },
    // LRM 9.3.4 function/procedure call with generic_map_aspect
    GenericMapCall {
        callee: Box<Expression>,
        generic_args: Vec<Expression>,
    },
}

// Helper for parsing chained postfix: name(args)'attr[sig] etc.
#[derive(Debug)]
pub enum PostfixSuffix {
    Call(Vec<Expression>),
    Select(Suffix),
    Attribute(Identifier),
    QualifiedExpression(Vec<Expression>),
    Signature(Signature),
    // LRM 9.3.4 generic_map_aspect in function/procedure calls
    GenericMap(Vec<Expression>),
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

// LRM 8.7 External names
#[derive(Debug, Copy, Clone)]
pub enum ExternalNameClass {
    Constant,
    Signal,
    Variable,
}

#[derive(Debug)]
pub enum ExternalPathElement {
    // LRM 8.7 pathname_element: identifier [ ( static_expression ) ]
    Identifier(Identifier, Option<Expression>),
    // LRM 8.7 absolute pathname: leading .
    Absolute,
    // LRM 8.7 relative pathname: ^
    Relative,
    // LRM 8.7 package pathname: @
    Package,
    // instantiation label (selected name element)
    All,
}

// LRM 10.5.2 Simple signal assignments
#[derive(Debug, Copy, Clone)]
pub enum ForceMode {
    In,
    Out,
}

// LRM 10.5.2.1 Simple force assignment
#[derive(Debug)]
pub struct SignalForceAssignmentStatement {
    pub label: Option<Identifier>,
    pub target: Target,
    pub force_mode: Option<ForceMode>,
    pub value: Expression,
}

// LRM 10.5.2.1 Simple release assignment
#[derive(Debug)]
pub struct SignalReleaseAssignmentStatement {
    pub label: Option<Identifier>,
    pub target: Target,
    pub force_mode: Option<ForceMode>,
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

pub fn build_call(callee: Expression, args: Vec<Expression>) -> Expression {
    Expression::Call {
        callee: Box::new(callee),
        args,
    }
}

pub fn build_name_call(name: Name, args: Option<Vec<Expression>>) -> Expression {
    let expr = Expression::Name(name);
    match args {
        Some(args) => build_call(expr, args),
        None => expr,
    }
}

pub fn apply_postfix_suffix(expr: Expression, suffix: PostfixSuffix) -> Expression {
    match suffix {
        PostfixSuffix::Call(args) => build_call(expr, args),
        PostfixSuffix::Select(suffix) => Expression::Select { value: Box::new(expr), suffix },
        PostfixSuffix::Attribute(attr) => Expression::Attribute { value: Box::new(expr), attr, args: vec![] },
        PostfixSuffix::QualifiedExpression(args) => Expression::QualifiedExpression { type_mark: Box::new(expr), args },
        PostfixSuffix::Signature(sig) => Expression::WithSignature { value: Box::new(expr), signature: sig },
        PostfixSuffix::GenericMap(args) => Expression::GenericMapCall { callee: Box::new(expr), generic_args: args },
    }
}

pub fn apply_nondot_postfix_suffix(expr: Expression, suffix: PostfixSuffix) -> Expression {
    match suffix {
        PostfixSuffix::Select(_) => unreachable!(),
        _ => apply_postfix_suffix(expr, suffix),
    }
}

pub fn build_subtype_indication(
    resolution: Option<Expression>,
    type_mark: TypeMark,
    constraint: Option<Constraint>,
) -> Expression {
    Expression::SubtypeIndication {
        resolution: resolution.map(Box::new),
        type_mark: Box::new(type_mark),
        constraint: constraint.map(Box::new),
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
    Concat,
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

// LRM 10.5.2 target ::= name | aggregate
#[derive(Debug)]
pub enum Target {
    Expr(Expression),
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
    SelectedSignalAssignment(SelectedWaveformAssignment),
    ComponentInstantiation(ComponentInstantiationStatement),
    Generate(GenerateStatement),
}

// LRM 11.2 Block statements
#[derive(Debug, Default)]
pub struct BlockHeader {
    pub generic: Option<GenericClause>,
    pub generic_map: Option<Vec<Expression>>,
    pub port: Option<PortClause>,
    pub port_map: Option<Vec<Expression>>,
}

#[derive(Debug)]
pub struct BlockStatement {
    pub label: Option<Identifier>,
    pub guard_expression: Option<Expression>,
    pub header: BlockHeader,
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
    Names(NonEmptyVec<Expression>),
    All,
}

#[derive(Debug)]
pub enum ProcessDeclarativeItem {
    SubprogramDeclaration(SubprogramDeclaration),
    SubprogramBody(SubprogramBody),
    SubprogramInstantiationDeclaration(SubprogramInstantiationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageBody(PackageBody),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    TypeDeclaration(TypeDeclaration),
    SubtypeDeclaration(SubTypeDeclaration),
    ConstantDeclaration(ConstantDeclaration),
    VariableDeclaration(VariableDeclaration),
    FileDeclaration(FileDeclaration),
    AliasDeclaration(AliasDeclaration),
    AttributeDeclaration(AttributeDeclaration),
    AttributeSpecification(AttributeSpecification),
    UseClause(UseClause),
    GroupTemplateDeclaration(GroupTemplateDeclaration),
    GroupDeclaration(GroupDeclaration),
}

#[derive(Debug)]
pub enum SequentialStatement {
    Wait(WaitStatement),
    Assertion(AssertionStatement),
    Report(ReportStatement),
    SignalAssignment(SequentialSignalAssignmentStatement),
    SignalForceAssignment(SignalForceAssignmentStatement),
    SignalReleaseAssignment(SignalReleaseAssignmentStatement),
    SelectedSignalAssignment(SelectedWaveformAssignment),
    SelectedVariableAssignment(SelectedVariableAssignment),
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
    pub call: Expression,
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
    pub sensitivity: Option<NonEmptyVec<Expression>>,
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
    pub kind: SequentialSignalAssignmentKind,
}

#[derive(Debug)]
pub enum SequentialSignalAssignmentKind {
    Simple {
        waveform: Waveform,
    },
    // LRM 10.5.3 Conditional signal assignments
    Conditional {
        conditionals: Vec<ConditionalWaveform>,
        else_waveform: Option<Waveform>,
    },
}

// LRM 10.5.4 Selected signal assignments
#[derive(Debug)]
pub struct SelectedWaveformAssignment {
    pub label: Option<Identifier>,
    pub selector: Expression,
    pub matching: bool,
    pub target: Target,
    pub guarded: bool,
    pub delay: Option<DelayMechanism>,
    pub alternatives: NonEmptyVec<SelectedWaveform>,
}
#[derive(Debug)]
pub struct SelectedWaveform {
    pub waveform: Waveform,
    pub choices: NonEmptyVec<Choice>,
}

// LRM 10.6.3 Selected variable assignments
#[derive(Debug)]
pub struct SelectedVariableAssignment {
    pub label: Option<Identifier>,
    pub selector: Expression,
    pub matching: bool,
    pub target: Target,
    pub alternatives: NonEmptyVec<SelectedExpression>,
}
#[derive(Debug)]
pub struct SelectedExpression {
    pub value: Expression,
    pub choices: NonEmptyVec<Choice>,
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
    pub call: Expression,
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
    pub matching: bool,
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
    // LRM 10.13 plain_return_statement: return [when condition];
    Plain {
        label: Option<Identifier>,
        condition: Option<Expression>,
    },
    // LRM 10.13 value_return_statement
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
    // LRM 11.6 Concurrent conditional signal assignment
    Conditional {
        target: Target,
        guarded: bool,
        delay: Option<DelayMechanism>,
        conditionals: Vec<ConditionalWaveform>,
        else_waveform: Option<Waveform>,
    },
}

// LRM 11.6 conditional_waveforms
#[derive(Debug)]
pub struct ConditionalWaveform {
    pub waveform: Waveform,
    pub condition: Expression,
}

// LRM 11.7 Component instantiation statements
#[derive(Debug)]
pub struct ComponentInstantiationStatement {
    pub label: Option<Identifier>,
    pub unit: Name,
    pub architecture: Option<Identifier>,
    pub generic_map: Option<Vec<Expression>>,
    pub port_map: Option<Vec<Expression>>,
}

// LRM 11.8 Generate statements
#[derive(Debug)]
pub enum GenerateStatement {
    For(ForGenerateStatement),
    If(IfGenerateStatement),
    // LRM 11.8 Case generate statements
    Case(CaseGenerateStatement),
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

// LRM 11.8 Case generate statements
#[derive(Debug)]
pub struct CaseGenerateStatement {
    pub label: Identifier,
    pub expression: Expression,
    pub alternatives: Vec<CaseGenerateAlternative>,
    pub end_label: Option<Identifier>,
}

#[derive(Debug)]
pub struct CaseGenerateAlternative {
    pub choices: NonEmptyVec<Choice>,
    pub body: GenerateBody,
}

#[derive(Debug)]
pub struct GenerateBody {
    pub alternative_label: Option<Identifier>,
    pub decl: Vec<BlockDeclarativeItem>,
    pub stmt: Vec<ConcurrentStatement>,
    pub end_label: Option<Identifier>,
}

// Intermediate parse types for generate statement inner end disambiguation.
// After ConcurrentStatement* "end", the next token disambiguates:
// "generate" = outer end, Identifier/; = inner end (LRM 11.8).

/// Recursive tail of an if-generate statement (elsif/else/end chains).
#[derive(Debug)]
pub(crate) enum IfGenTail {
    End {
        prev_inner_end: Option<Identifier>,
        end_label: Option<Identifier>,
    },
    Elsif {
        prev_inner_end: Option<Identifier>,
        alt_label: Option<Identifier>,
        condition: Expression,
        body: (Vec<BlockDeclarativeItem>, Vec<ConcurrentStatement>),
        rest: Box<IfGenTail>,
    },
    Else {
        prev_inner_end: Option<Identifier>,
        alt_label: Option<Identifier>,
        body: (Vec<BlockDeclarativeItem>, Vec<ConcurrentStatement>),
        else_inner_end: Option<Identifier>,
        end_label: Option<Identifier>,
    },
}

/// Recursive tail of a case-generate statement (when/end chains).
#[derive(Debug)]
pub(crate) enum CaseGenTail {
    End {
        prev_inner_end: Option<Identifier>,
        end_label: Option<Identifier>,
    },
    When {
        prev_inner_end: Option<Identifier>,
        alt_label: Option<Identifier>,
        choices: NonEmptyVec<Choice>,
        body: (Vec<BlockDeclarativeItem>, Vec<ConcurrentStatement>),
        rest: Box<CaseGenTail>,
    },
}

impl IfGenTail {
    /// Flatten the recursive tail into (elsif_branches, else_body, end_label),
    /// assigning each inner end label to the preceding body.
    pub(crate) fn flatten(self, prev_body: &mut GenerateBody) -> (
        Vec<(Expression, GenerateBody)>,
        Option<GenerateBody>,
        Option<Identifier>,
    ) {
        match self {
            IfGenTail::End { prev_inner_end, end_label } => {
                prev_body.end_label = prev_inner_end;
                (vec![], None, end_label)
            }
            IfGenTail::Elsif { prev_inner_end, alt_label, condition, body, rest } => {
                prev_body.end_label = prev_inner_end;
                let mut branch_body = GenerateBody {
                    alternative_label: alt_label,
                    decl: body.0,
                    stmt: body.1,
                    end_label: None,
                };
                let (mut elsifs, else_body, end_label) = rest.flatten(&mut branch_body);
                elsifs.insert(0, (condition, branch_body));
                (elsifs, else_body, end_label)
            }
            IfGenTail::Else { prev_inner_end, alt_label, body, else_inner_end, end_label } => {
                prev_body.end_label = prev_inner_end;
                let else_body = GenerateBody {
                    alternative_label: alt_label,
                    decl: body.0,
                    stmt: body.1,
                    end_label: else_inner_end,
                };
                (vec![], Some(else_body), end_label)
            }
        }
    }
}

impl CaseGenTail {
    /// Flatten the recursive tail into (alternatives, end_label).
    pub(crate) fn flatten(self, prev_body: &mut GenerateBody) -> (
        Vec<CaseGenerateAlternative>,
        Option<Identifier>,
    ) {
        match self {
            CaseGenTail::End { prev_inner_end, end_label } => {
                prev_body.end_label = prev_inner_end;
                (vec![], end_label)
            }
            CaseGenTail::When { prev_inner_end, alt_label, choices, body, rest } => {
                prev_body.end_label = prev_inner_end;
                let mut alt_body = GenerateBody {
                    alternative_label: alt_label,
                    decl: body.0,
                    stmt: body.1,
                    end_label: None,
                };
                let (mut alts, end_label) = rest.flatten(&mut alt_body);
                alts.insert(0, CaseGenerateAlternative { choices, body: alt_body });
                (alts, end_label)
            }
        }
    }
}

// LRM 12 Scope and visibility
// LRM 12.4 Use clauses
#[derive(Debug)]
pub struct UseClause {
    pub names: NonEmptyVec<Name>,
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
    ConfigurationDeclaration(ConfigurationDeclaration),
    PackageDeclaration(PackageDeclaration),
    PackageInstantiationDeclaration(PackageInstantiationDeclaration),
    ContextDeclaration(ContextDeclaration),

    // secondary unit
    ArchitectureBody(ArchitectureBody),
    PackageBody(PackageBody),
}

// LRM 13.2 Design libraries
#[derive(Debug)]
pub struct LibraryClause {
    pub names: NonEmptyVec<Identifier>,
}

// LRM 13.3 Context declarations
#[derive(Debug)]
pub struct ContextDeclaration {
    pub name: Identifier,
    pub items: ContextClause,
    pub end_name: Option<Identifier>,
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
    Context(ContextReference),
}

// LRM 13.4.2 Context references
#[derive(Debug)]
pub struct ContextReference {
    pub names: NonEmptyVec<Name>,
}

// LRM 15 Lexical elements

// LRM 15.4 Identifiers
#[derive(Debug)]
pub struct Identifier {
    pub span: Span,
}

// LRM 4.2 Subprogram declarations: designator
#[derive(Debug)]
pub enum Designator {
    Identifier(Identifier),
    OperatorSymbol(Span),
    CharLiteral(Span),
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
