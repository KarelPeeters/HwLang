use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::module_body::{LowerBlock, ModuleChecked};
use crate::data::parsed::{ItemAstReference, ModulePortAstReference, ParsedDatabase};
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, TypeOrValue, ValueDomain};
use crate::front::scope::{Scope, ScopeInfo, Scopes};
use crate::front::types::{IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::new_index_type;
use crate::syntax::ast::{DomainKind, Identifier, MaybeIdentifier, PortDirection, PortKind, SyncDomain};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::Arena;
use indexmap::IndexMap;
use num_traits::Signed;

macro_rules! impl_index {
    ($arena:ident, $index:ty, $info:ty) => {
        impl<S: CompiledStage> std::ops::Index<$index> for CompiledDatabase<S> {
            type Output = $info;
            fn index(&self, index: $index) -> &Self::Output {
                &self.$arena[index]
            }
        }
    };
}

macro_rules! impl_index_mut {
    ($arena:ident, $index:ty, $info:ty) => {
        impl std::ops::IndexMut<$index> for CompiledDatabase<CompiledStagePartial> {
            fn index_mut(&mut self, index: $index) -> &mut $info {
                &mut self.$arena[index]
            }
        }
    };
}

pub type CompiledDatabasePartial = CompiledDatabase<CompiledStagePartial>;

pub struct CompiledDatabase<S: CompiledStage = CompiledStateFull> {
    pub scopes: Scopes<ScopedEntry>,

    pub file_scopes: IndexMap<FileId, Result<FileScopes, ErrorGuaranteed>>,
    pub items: Arena<Item, ItemInfo<S::ItemInfoSignature, S::ItemInfoBody>>,

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

impl_index!(items, Item, ItemInfo<S::ItemInfoSignature, S::ItemInfoBody>);
impl_index_mut!(items, Item, ItemInfoPartial);
impl_index!(scopes, Scope, ScopeInfo<ScopedEntry>);
impl_index_mut!(scopes, Scope, ScopeInfo<ScopedEntry>);

impl_index!(generic_type_params, GenericTypeParameter, GenericTypeParameterInfo);
impl_index!(generic_value_params, GenericValueParameter, GenericValueParameterInfo);
impl_index!(module_ports, ModulePort, ModulePortInfo);
impl_index!(registers, Register, RegisterInfo);
impl_index!(wires, Wire, WireInfo);
impl_index!(variables, Variable, VariableInfo);
impl_index!(constants, Constant, ConstantInfo);

impl<S: CompiledStage> std::ops::Index<FileId> for CompiledDatabase<S> {
    type Output = Result<FileScopes, ErrorGuaranteed>;
    fn index(&self, index: FileId) -> &Self::Output {
        self.file_scopes.get(&index).unwrap()
    }
}

pub trait CompiledStage {
    type ItemInfoSignature;
    type ItemInfoBody;
}

pub struct CompiledStagePartial;

impl CompiledStage for CompiledStagePartial {
    type ItemInfoSignature = Option<MaybeConstructor<TypeOrValue>>;
    type ItemInfoBody = Option<ItemChecked>;
}

pub struct CompiledStateFull;

impl CompiledStage for CompiledStateFull {
    type ItemInfoSignature = MaybeConstructor<TypeOrValue>;
    type ItemInfoBody = ItemChecked;
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

pub type ItemInfoPartial = ItemInfo<Option<MaybeConstructor<TypeOrValue>>, Option<ItemChecked>>;

#[derive(Debug)]
pub struct ItemInfo<T = MaybeConstructor<TypeOrValue>, B = ItemChecked> {
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
    pub kind: PortKind<DomainKind<Value>, Type>,
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

// TODO variables that are assigned to multiple different domains need to get the merged domain,
//   but that is not implemented yet. Currently only obvious domains (ie. clocked, function) are set.
//   This mostly works fine, except for combinatorial blocks.
#[derive(Debug, Clone)]
pub enum VariableDomain {
    Unknown,
    Known(ValueDomain),
}

#[derive(Debug)]
pub struct VariableInfo {
    pub defining_id: MaybeIdentifier,

    // TODO is it okay that this type does not get its generics replaced?
    pub ty: Type,
    pub mutable: bool,

    pub domain: VariableDomain,
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
pub enum ItemChecked {
    /// For items that don't have a body.
    None,
    Module(ModuleChecked),
    Function(FunctionChecked),
    Error(ErrorGuaranteed),
}

#[derive(Debug)]
pub struct FunctionChecked {
    #[allow(dead_code)]
    pub block: LowerBlock,
}

impl<S: CompiledStage> CompiledDatabase<S> {
    // TODO make sure generic variables are properly disambiguated
    // TODO insert this into value_to_readable_str, no point keeping this one separate
    pub fn range_to_readable_str(&self, source: &SourceDatabase, parsed: &ParsedDatabase, range: &RangeInfo<Box<Value>>) -> String {
        let RangeInfo { ref start, ref end } = *range;

        let start = start.as_ref().map_or(String::new(), |v| self.value_to_readable_str(source, parsed, v));
        let end = end.as_ref().map_or(String::new(), |v| self.value_to_readable_str(source, parsed, v));

        format!("({}..{})", start, end)
    }

    // TODO integrate generic parameters properly in the diagnostic, by pointing to them
    // TODO include span if there is any ambiguity
    pub fn value_to_readable_str(&self, source: &SourceDatabase, parsed: &ParsedDatabase, value: &Value) -> String {
        match value {
            Value::Error(_) => "error".to_string(),
            &Value::GenericParameter(p) => self[p].defining_id.string.clone(),
            &Value::ModulePort(p) => parsed.module_port_ast(self[p].ast).id.string.clone(),
            Value::Never => "never".to_string(),
            Value::Undefined => "undefined".to_string(),
            Value::Unit => "()".to_string(),
            Value::BoolConstant(b) => format!("{}", b),
            Value::IntConstant(v) => {
                if v.is_negative() {
                    format!("({})", v)
                } else {
                    format!("{}", v)
                }
            }
            Value::StringConstant(s) => format!("{:?}", s),
            Value::Range(range) => self.range_to_readable_str(source, parsed, range),
            &Value::Binary(op, ref left, ref right) => {
                let left = self.value_to_readable_str(source, parsed, left);
                let right = self.value_to_readable_str(source, parsed, right);
                let symbol = op.symbol();
                format!("({} {} {})", left, symbol, right)
            }
            Value::UnaryNot(inner) => {
                let inner = self.value_to_readable_str(source, parsed, inner);
                format!("(!{})", inner)
            }
            // TODO how to display function return values? we don't know the function call args here any more!
            Value::FunctionReturn(ret) =>
                format!("function_return({})", self.defining_id_to_readable_string(&self[ret.item].defining_id)),
            Value::Module(module) =>
                format!("module({})", self.defining_id_to_readable_string(&self[module.nominal_type_unique.item].defining_id)),
            &Value::Wire(wire) =>
                self.defining_id_to_readable_string(&self[wire].defining_id).to_string(),
            &Value::Register(reg) =>
                self.defining_id_to_readable_string(&self[reg].defining_id).to_string(),
            &Value::Variable(var) =>
                self.defining_id_to_readable_string(&self[var].defining_id).to_string(),
            &Value::Constant(c) =>
                self.defining_id_to_readable_string(&self[c].defining_id).to_string(),
        }
    }

    // TODO make sure to always print in disambiguated form
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
            Type::Bits(n) => match n {
                None => "bits".to_string(),
                Some(n) => format!("bits({})", self.value_to_readable_str(source, parsed, n)),
            },
            Type::Range => "range".to_string(),
            Type::Array(inner, n) =>
                format!("Array({}, {})", self.type_to_readable_str(source, parsed, inner), self.value_to_readable_str(source, parsed, n)),
            Type::Integer(IntegerTypeInfo { range }) => {
                // TODO match specific patterns again, eg. uint?
                format!("int_range({})", self.value_to_readable_str(source, parsed, range))
            }
            // TODO
            Type::Function(_) => "function".to_string(),
            Type::Tuple(_) => "tuple".to_string(),
            Type::Struct(_) => "struct".to_string(),
            Type::Enum(_) => "enum".to_string(),
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
                let clock_str = self.value_to_readable_str(source, parsed, clock);
                let reset_str = self.value_to_readable_str(source, parsed, reset);
                format!("sync({clock_str}, {reset_str})")
            }
            &ValueDomain::FunctionBody(function) => {
                format!("function_body({})", self.defining_id_to_readable_string(&self[function].defining_id))
            }
        }
    }

    pub fn defining_id_to_readable_string<'a>(&self, id: &'a MaybeIdentifier) -> &'a str {
        id.string().unwrap_or("_")
    }
}
