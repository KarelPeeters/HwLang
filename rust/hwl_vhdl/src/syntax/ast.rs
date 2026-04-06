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
pub enum EntityDeclarativeItem {}

#[derive(Debug)]
pub enum EntityStatement {}

// LRM 3.3 Architecture bodies
#[derive(Debug)]
pub struct ArchitectureBody {
    pub name: Identifier,
    pub entity_name: Identifier,
    pub decl: Vec<ArchitectureDeclarativeItem>,
    pub stmt: Vec<ArchitectureStatement>,
    pub end_name: Option<Identifier>,
}

#[derive(Debug)]
pub enum ArchitectureDeclarativeItem {}

#[derive(Debug)]
pub enum ArchitectureStatement {}

// LRM 6.5 Interface declarations
// LRM 6.5.2 Interface object declarations
#[derive(Debug)]
pub struct InterfaceConstantDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub ty: InterfaceTypeIndication,
    pub init: Option<Expression>,
}

#[derive(Debug)]
pub struct InterfaceSignalDeclaration {
    pub names: NonEmptyVec<Identifier>,
    pub mode_indication: ModeIndication,
}

#[derive(Debug)]
pub enum InterfaceTypeIndication {
    Subtype,
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
    RecordView,
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
    Type,
    Subprogram,
    Package,
}

#[derive(Debug)]
pub struct PortClause {
    pub list: NonEmptyVec<PortInterfaceDeclaration>,
}

#[derive(Debug)]
pub enum PortInterfaceDeclaration {
    Signal(InterfaceSignalDeclaration),
    Variable,
}

// LRM 9 Expressions
#[derive(Debug)]
pub enum Expression {}

// LRM 10 Sequential statements
// LRM 10.5 Simple assignment statement
// LRM 10.5.2 Simple signal assignments
#[derive(Debug)]
pub struct DelayMechanism {
    pub reject: Option<Expression>,
}

#[derive(Debug)]
pub enum Waveform {
    Elements(NonEmptyVec<WaveformElement>),
    Unaffected,
}

// LRM 10.5.2.2 Executing a simple assignment statement
#[derive(Debug)]
pub struct WaveformElement {
    pub value: Option<Expression>,
    pub after: Option<Expression>,
}

// LRM 11 Concurrent statements
#[derive(Debug)]
pub enum ConcurrentStatement {
    Block,
    Process,
    ProcedureCall,
    Assertion,
    SignalAssignment(ConcurrentSignalAssignmentStatement),
    ComponentInstantiation,
    Generate,
}

// LRM 11.6 Concurrent signal assignment statements
#[derive(Debug)]
pub struct ConcurrentSignalAssignmentStatement {
    pub label: Option<Identifier>,
    pub postponed: bool,
}

pub enum ConcurrentSignalAssignment {
    Simple {
        guarded: bool,
        delay: Option<DelayMechanism>,
        waveform: Waveform,
    },
    Conditional,
    Selected,
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
    ConfigurationDeclaration,
    PackageDeclaration,
    PackageInstantiationDeclaration,
    ContextDeclaration,

    // secondary unit
    ArchitectureBody(ArchitectureBody),
    PackageBody,
}

// LRM 13.2 Design libraries
#[derive(Debug)]
pub struct LibraryClause {
    pub logical_name_list: NonEmptyVec<Identifier>,
}

// LRM 13.4 Context clauses
#[derive(Debug)]
pub struct ContextClause {}

// TODO sort
#[derive(Debug)]
pub struct Identifier {
    pub span: Span,
}
