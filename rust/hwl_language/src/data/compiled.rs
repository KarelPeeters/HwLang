use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::module_body::{LowerBlock, LowerModule};
use crate::data::parsed::{ItemAstReference, ModulePortAstReference, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, TypeOrValue};
use crate::front::scope::{Scope, ScopeInfo, Scopes};
use crate::front::solver::SolverInt;
use crate::front::types::{MaybeConstructor, Type};
use crate::front::value::{DomainSignal, RangeInfo, Value, ValueDomain};
use crate::syntax::ast::{DomainKind, Identifier, MaybeIdentifier, PortDirection, PortKind, SyncDomain};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::Arena;
use crate::{new_index_type, swrite};
use indexmap::IndexMap;

macro_rules! impl_index {
    ($arena:ident, $index:ty, $info:ty) => {
        impl<'a, S: CompiledStage> std::ops::Index<$index> for CompiledDatabase<'a, S> {
            type Output = $info;
            fn index(&self, index: $index) -> &Self::Output {
                &self.$arena[index]
            }
        }
    };
}

macro_rules! impl_index_mut {
    ($arena:ident, $index:ty, $info:ty) => {
        impl<'a> std::ops::IndexMut<$index> for CompiledDatabase<'a, CompiledStagePartial> {
            fn index_mut(&mut self, index: $index) -> &mut $info {
                &mut self.$arena[index]
            }
        }
    };
}

pub type CompiledDatabasePartial<'a> = CompiledDatabase<'a, CompiledStagePartial>;

pub struct CompiledDatabase<'a, S: CompiledStage = CompiledStateFull> {
    pub scopes: Scopes<ScopedEntry>,

    pub file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    pub items: Arena<Item, ItemInfo<S::ItemInfoSignature, S::ItemInfoBody<'a>>>,

    pub generic_type_params: Arena<GenericTypeParameter, GenericTypeParameterInfo>,
    pub generic_value_params: Arena<GenericValueParameter, GenericValueParameterInfo>,
    pub module_info: IndexMap<Item, ModuleSignatureInfo>,
    pub module_ports: Arena<ModulePort, ModulePortInfo>,
    pub function_info: IndexMap<Item, FunctionSignatureInfo>,
    pub registers: Arena<Register, RegisterInfo>,
    pub wires: Arena<Wire, WireInfo>,
    pub variables: Arena<Variable, VariableInfo>,
    pub constants: Arena<Constant, ConstantInfo>,
}

impl_index!(items, Item, ItemInfo<S::ItemInfoSignature, S::ItemInfoBody<'a>>);
impl_index_mut!(items, Item, ItemInfoPartial<'a>);
impl_index!(scopes, Scope, ScopeInfo<ScopedEntry>);
impl_index_mut!(scopes, Scope, ScopeInfo<ScopedEntry>);

impl_index!(generic_type_params, GenericTypeParameter, GenericTypeParameterInfo);
impl_index!(generic_value_params, GenericValueParameter, GenericValueParameterInfo);
impl_index!(module_ports, ModulePort, ModulePortInfo);
impl_index!(registers, Register, RegisterInfo);
impl_index!(wires, Wire, WireInfo);
impl_index!(variables, Variable, VariableInfo);
impl_index!(constants, Constant, ConstantInfo);

impl<S: CompiledStage> std::ops::Index<FileId> for CompiledDatabase<'_, S> {
    type Output = Result<FileScopes, ErrorGuaranteed>;
    fn index(&self, index: FileId) -> &Self::Output {
        self.file_scopes.get(&index).unwrap()
    }
}

pub trait CompiledStage {
    type ItemInfoSignature;
    type ItemInfoBody<'a>;
}

pub struct CompiledStagePartial;

impl CompiledStage for CompiledStagePartial {
    type ItemInfoSignature = Option<MaybeConstructor<TypeOrValue>>;
    type ItemInfoBody<'a> = Option<ItemChecked<'a>>;
}

pub struct CompiledStateFull;

impl CompiledStage for CompiledStateFull {
    type ItemInfoSignature = MaybeConstructor<TypeOrValue>;
    type ItemInfoBody<'a> = ItemChecked<'a>;
}

new_index_type!(pub Item);
new_index_type!(pub GenericTypeParameter);
new_index_type!(pub GenericValueParameter);
new_index_type!(pub ModulePort);
new_index_type!(pub Register);
new_index_type!(pub Wire);
new_index_type!(pub Variable);
new_index_type!(pub Constant);

#[derive(Debug)]
pub struct FileScopes {
    /// The scope that only includes top-level items defined in this file. 
    pub scope_outer_declare: Scope,
    /// Child scope of [scope_declare] that includes all imported items.
    pub scope_inner_import: Scope,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum GenericParameter {
    Type(GenericTypeParameter),
    Value(GenericValueParameter),
}

pub type ItemInfoPartial<'a> = ItemInfo<Option<MaybeConstructor<TypeOrValue>>, Option<ItemChecked<'a>>>;
pub type ItemInfoFull<'a> = ItemInfo<MaybeConstructor<TypeOrValue>, ItemChecked<'a>>;

#[derive(Debug)]
pub struct ItemInfo<T, B> {
    pub defining_id: MaybeIdentifier,
    pub ast_ref: ItemAstReference,
    pub signature: T,
    pub body: B,
}

#[derive(Debug)]
pub struct GenericTypeParameterInfo {
    pub defining_item: Item,
    pub defining_id: Identifier,

    // TODO type constraints once we add those
}

#[derive(Debug)]
pub struct GenericValueParameterInfo {
    pub defining_item: Item,
    pub defining_id: Identifier,
    pub defining_item_kind: GenericItemKind,

    pub ty: Type,
    // TODO it's a bit weird that we're tracking the span here, this should just be part of Type
    pub ty_span: Span,
}

#[derive(Debug, Copy, Clone)]
pub enum GenericItemKind {
    Type,
    Struct,
    Enum,
    Module,
    Function,
}

#[derive(Debug)]
pub struct FunctionTypeParameterInfo {
    pub defining_item: Item,
    pub defining_id: Identifier,

    // TODO type constraints
}

#[derive(Debug)]
pub struct FunctionValueParameterInfo {
    pub defining_item: Item,
    pub defining_id: Identifier,

    pub ty: Type,
    // TODO it's a bit weird that we're tracking the span here, this should just be part of Type
    pub ty_span: Span,
}

// TODO move into ItemInfo?
#[derive(Debug)]
pub struct FunctionSignatureInfo {
    pub scope_inner: Scope,
    pub ret_ty: Type,
}

// TODO move into ItemInfo?
#[derive(Debug)]
pub struct ModuleSignatureInfo {
    pub scope_ports: Scope,
    pub ports: Vec<ModulePort>,
}

#[derive(Debug)]
pub struct ModulePortInfo {
    pub ast: ModulePortAstReference,
    pub direction: PortDirection,
    pub kind: PortKind<DomainKind<DomainSignal>, Type>,
}

impl PortKind<DomainKind<DomainSignal>, Type> {
    pub fn ty(&self) -> &Type {
        match self {
            PortKind::Clock => &Type::Clock,
            PortKind::Normal { domain: _, ty } => ty,
        }
    }

    pub fn domain(&self) -> ValueDomain {
        match self {
            PortKind::Clock => ValueDomain::Clock,
            PortKind::Normal { domain, ty: _ } => ValueDomain::from_domain_kind(domain.clone()),
        }
    }
}

// TODO should the init value be here or in the module?
#[derive(Debug)]
pub struct RegisterInfo {
    pub defining_item: Item,
    pub defining_id: MaybeIdentifier,

    pub domain: SyncDomain<Value>,
    pub ty: Type,
}

#[derive(Debug)]
pub struct WireInfo {
    pub defining_item: Item,
    pub defining_id: MaybeIdentifier,

    pub domain: DomainKind<Value>,
    pub ty: Type,
    pub has_declaration_value: bool,
}

// TODO most of this is redundant with the fields in Value 
#[derive(Debug)]
pub struct VariableInfo {
    pub defining_id: MaybeIdentifier,

    // TODO is it okay that this type does not get its generics replaced?
    pub ty: Type,
    pub mutable: bool,

    pub domain: ValueDomain,
}

#[derive(Debug)]
pub struct ConstantInfo {
    pub defining_id: MaybeIdentifier,
    // TODO is it okay that this type does not get its generics replaced?
    pub ty: Type,
    pub value: Value,
}

/// The result of item body checking.
///
/// Typechecking can store any additional information beyond the AST here,
/// which can be used during lowering.
#[derive(Debug)]
pub enum ItemChecked<'a> {
    /// For items that don't have a body.
    None,
    Module(LowerModule<'a>),
    Function(FunctionChecked<'a>),
    Error(ErrorGuaranteed),
}

#[derive(Debug)]
pub struct FunctionChecked<'a> {
    #[allow(dead_code)]
    pub block: LowerBlock<'a>,
}

impl<S: CompiledStage> CompiledDatabase<'_, S> {
    pub fn solver_int_to_readable_str(&self, _: SolverInt) -> String {
        // TODO do something useful here, this will require adding debug info to the solver
        "solver_int".to_string()
    }

    // TODO make sure to always print in disambiguated form
    // TODO return something that automatically includes backticks during formatting
    pub fn type_to_readable_str(&self, source: &SourceDatabase, parsed: &ParsedDatabase, ty: &Type) -> String {
        match ty {
            Type::Error(_) => "error".to_string(),
            &Type::GenericParameter(p) => self[p].defining_id.string.clone(),
            Type::Any => "any".to_string(),
            Type::Unchecked => "unchecked".to_string(),
            Type::Never => "never".to_string(),
            Type::Unit => "()".to_string(),
            Type::Boolean => "bool".to_string(),
            Type::Clock => "clock".to_string(),
            Type::String => "string".to_string(),
            Type::Range => "range".to_string(),
            Type::Module => "module".to_string(),
            &Type::Array(ref inner, n) => {
                let mut indices = String::new();
                let f = &mut indices;

                swrite!(f, "{}", self.solver_int_to_readable_str(n));
                let mut inner = inner;
                while let &Type::Array(ref next, n_next) = &**inner {
                    swrite!(f, ", {}", self.solver_int_to_readable_str(n_next));
                    inner = next;
                }

                let ty_str = self.type_to_readable_str(source, parsed, inner);
                format!("({ty_str}[{indices}])")
            }
            Type::Integer(RangeInfo { start_inc, end_inc }) => {
                // TODO match specific patterns for even more readable strings, eg. uint?
                let start_str = start_inc.map_or("".to_string(), |v| self.solver_int_to_readable_str(v));
                let end_str = end_inc.map_or("".to_string(), |v| self.solver_int_to_readable_str(v));
                format!("int_range({start_str}..{end_str})")
            }
            // TODO
            Type::Function(_) => "function".to_string(),
            Type::Tuple(_) => "tuple".to_string(),
            Type::Struct(_) => "struct".to_string(),
            Type::Enum(_) => "enum".to_string(),
        }
    }

    pub fn domain_signal_to_readable_string(&self, source: &SourceDatabase, parsed: &ParsedDatabase, signal: &DomainSignal) -> String {
        match signal {
            DomainSignal::Error(_) => "error".to_string(),
            &DomainSignal::Port(port) => parsed.module_port_ast(self[port].ast).id().string.clone(),
            &DomainSignal::Wire(wire) => defining_id_to_readable_string(self[wire].defining_id.as_ref()).to_string(),
            &DomainSignal::Register(reg) => defining_id_to_readable_string(self[reg].defining_id.as_ref()).to_string(),
            DomainSignal::Invert(inner) => {
                let inner_str = self.domain_signal_to_readable_string(source, parsed, inner);
                format!("(!{})", inner_str)
            }
        }
    }

    // TODO these interfaces are getting really ugly, create combined database types
    pub fn sync_kind_to_readable_string(&self, source: &SourceDatabase, parsed: &ParsedDatabase, sync: &ValueDomain) -> String {
        match sync {
            ValueDomain::Error(_) => "error".to_string(),
            ValueDomain::CompileTime => "const".to_string(),
            ValueDomain::Clock => "clock".to_string(),
            ValueDomain::Async => "async".to_string(),
            ValueDomain::Sync(SyncDomain { clock, reset }) => {
                let clock_str = self.domain_signal_to_readable_string(source, parsed, clock);
                let reset_str = self.domain_signal_to_readable_string(source, parsed, reset);
                format!("sync({clock_str}, {reset_str})")
            }
            &ValueDomain::FunctionBody(function) => {
                format!("function_body({})", defining_id_to_readable_string(self[function].defining_id.as_ref()))
            }
        }
    }
}

pub fn defining_id_to_readable_string(id: MaybeIdentifier<&Identifier>) -> &str {
    id.string().unwrap_or("_")
}