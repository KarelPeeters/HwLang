use crate::data::diagnostic::ErrorGuaranteed;
use crate::data::module_body::ModuleChecked;
use crate::data::parsed::ItemAstReference;
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, TypeOrValue};
use crate::front::scope::{Scope, ScopeInfo, Scopes};
use crate::front::types::{IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::new_index_type;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, PortKind, SyncDomain, SyncKind};
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
    pub file_scope: IndexMap<FileId, Result<Scope, ErrorGuaranteed>>,
    pub scopes: Scopes<ScopedEntry>,

    pub items: Arena<Item, ItemInfo<S::ItemInfoSignature, S::ItemInfoBody>>,

    pub generic_type_params: Arena<GenericTypeParameter, GenericTypeParameterInfo>,
    pub generic_value_params: Arena<GenericValueParameter, GenericValueParameterInfo>,
    pub module_info: IndexMap<Item, ModuleSignatureInfo>,
    pub module_ports: Arena<ModulePort, ModulePortInfo>,
    pub function_info: IndexMap<Item, FunctionSignatureInfo>,
    pub registers: Arena<Register, RegisterInfo>,
    pub variables: Arena<Variable, VariableInfo>,
}

impl_index!(items, Item, ItemInfo<S::ItemInfoSignature, S::ItemInfoBody>);
impl_index_mut!(items, Item, ItemInfoPartial);
impl_index!(scopes, Scope, ScopeInfo<ScopedEntry>);
impl_index_mut!(scopes, Scope, ScopeInfo<ScopedEntry>);

impl_index!(generic_type_params, GenericTypeParameter, GenericTypeParameterInfo);
impl_index!(generic_value_params, GenericValueParameter, GenericValueParameterInfo);
impl_index!(module_ports, ModulePort, ModulePortInfo);
impl_index!(registers, Register, RegisterInfo);
impl_index!(variables, Variable, VariableInfo);

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
new_index_type!(pub Variable);

pub type GenericParameter = TypeOrValue<GenericTypeParameter, GenericValueParameter>;

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

    pub ty: Type,
    // TODO it's a bit weird that we're tracking the span here, this should just be part of Type
    pub ty_span: Span,
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
}

#[derive(Debug)]
pub struct ModulePortInfo {
    // These ids might not be unique, compilation continues even if a duplicate port name was used.
    pub defining_item: Item,
    pub defining_id: Identifier,
    pub direction: PortDirection,
    // TODO this is probably wrong, the type and value here don't get replaced properly
    pub kind: PortKind<SyncKind<Value>, Type>,
}

#[derive(Debug)]
pub struct RegisterInfo {
    pub defining_item: Item,
    pub defining_id: MaybeIdentifier,

    // TODO is it okay that this type and value does not get its generics replaced?
    pub sync: SyncDomain<Value>,
    pub ty: Type,
}

#[derive(Debug)]
pub struct VariableInfo {
    pub defining_id: MaybeIdentifier,

    // TODO is it okay that this type does not get its generics replaced?
    pub ty: Type,
    pub mutable: bool,
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
    // TODO at least the types of all local variables
}

impl<S: CompiledStage> CompiledDatabase<S> {
    // TODO make sure generic variables are properly disambiguated
    // TODO insert this into value_to_readable_str, no point keeping this one separate
    pub fn range_to_readable_str(&self, source: &SourceDatabase, range: &RangeInfo<Box<Value>>) -> String {
        let RangeInfo { ref start, ref end, end_inclusive } = *range;

        let start = start.as_ref().map_or(String::new(), |v| self.value_to_readable_str(source, v));
        let end = end.as_ref().map_or(String::new(), |v| self.value_to_readable_str(source, v));
        let symbol = if end_inclusive { "..=" } else { ".." };

        format!("({}{}{})", start, symbol, end)
    }

    // TODO integrate generic parameters properly in the diagnostic, by pointing to them
    // TODO make this less ugly for end users, eg. omit the span if there's no ambiguity
    pub fn value_to_readable_str(&self, source: &SourceDatabase, value: &Value) -> String {
        match value {
            // this should never actually come up, since there will never more future errors on error values
            Value::Error(_) => "value that corresponds to previously reported error".to_string(),
            &Value::GenericParameter(p) => {
                let id = &self[p].defining_id;
                format!("generic_param({:?}, {:?})", id.string, source.expand_pos(id.span.start))
            }
            &Value::ModulePort(p) => {
                let id = &self[p].defining_id;
                format!("module_port({:?}, {:?})", id.string, source.expand_pos(id.span.start))
            }
            Value::Never => "never".to_string(),
            Value::Unit => "()".to_string(),
            Value::Int(v) => {
                if v.is_negative() {
                    format!("({})", v)
                } else {
                    format!("{}", v)
                }
            }
            Value::Range(range) => self.range_to_readable_str(source, range),
            &Value::Binary(op, ref left, ref right) => {
                let left = self.value_to_readable_str(source, left);
                let right = self.value_to_readable_str(source, right);
                let symbol = op.symbol();
                format!("({} {} {})", left, symbol, right)
            }
            Value::UnaryNot(inner) => {
                let inner = self.value_to_readable_str(source, inner);
                format!("(!{})", inner)
            }
            Value::FunctionReturn(ret) =>
                format!("function_return({})", defining_id_to_string_pair(source, &self[ret.item].defining_id)),
            Value::Module(module) =>
                format!("module({})", defining_id_to_string_pair(source, &self[module.ty.nominal_type_unique.item].defining_id)),
            Value::Wire => "wire".to_string(),
            &Value::Register(reg) =>
                format!("register({})", defining_id_to_string_pair(source, &self[reg].defining_id)),
            &Value::Variable(var) =>
                format!("variable({})", defining_id_to_string_pair(source, &self[var].defining_id)),
        }
    }

    // TODO make sure to always print in disambiguated form
    pub fn type_to_readable_str(&self, source: &SourceDatabase, ty: &Type) -> String {
        match ty {
            // this should never actually come up, since there will never more future errors on error types
            Type::Error(_) => "type that corresponds to previously reported error".to_string(),
            &Type::GenericParameter(p) => {
                let id = &self[p].defining_id;
                format!("generic_param({:?}, {:?})", id.string, source.expand_pos(id.span.start))
            }
            Type::Any => "any".to_string(),
            Type::Unchecked => "unchecked".to_string(),
            Type::Never => "never".to_string(),
            Type::Unit => "()".to_string(),
            Type::Boolean => "bool".to_string(),
            Type::Bits(n) => match n {
                None => "bits".to_string(),
                Some(n) => format!("bits({})", self.value_to_readable_str(source, n)),
            },
            Type::Range => "range".to_string(),
            Type::Array(inner, n) =>
                format!("Array({}, {})", self.type_to_readable_str(source, inner), self.value_to_readable_str(source, n)),
            Type::Integer(IntegerTypeInfo { range }) => {
                // TODO match specific patterns again, eg. uint?
                format!("int_range({})", self.value_to_readable_str(source, range))
            }
            // TODO
            Type::Function(_) => "function".to_string(),
            Type::Tuple(_) => "tuple".to_string(),
            Type::Struct(_) => "struct".to_string(),
            Type::Enum(_) => "enum".to_string(),
            Type::Module(_) => "module".to_string(),
        }
    }
}

fn defining_id_to_string_pair(source: &SourceDatabase, id: &MaybeIdentifier) -> String {
    let id_str = match id {
        MaybeIdentifier::Dummy(_) => "_",
        MaybeIdentifier::Identifier(id) => &id.string,
    };
    format!("{:?}, {:?}", id_str, source.expand_pos(id.span().start))
}
