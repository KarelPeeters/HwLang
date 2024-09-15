use crate::data::diagnostic::{ErrorGuaranteed, ResultOrGuaranteed};
use crate::data::module_body::ModuleBody;
use crate::data::parsed::ItemAstReference;
use crate::data::source::SourceDatabase;
use crate::front::common::{ScopedEntry, TypeOrValue};
use crate::front::scope::{Scope, ScopeInfo, Scopes};
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::new_index_type;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, PortKind, SyncKind};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::Arena;
use indexmap::IndexMap;
use num_traits::Signed;

pub type CompiledDatabasePartial = CompiledDatabase<CompiledStagePartial>;

pub struct CompiledDatabase<S: CompiledStage = CompiledStateFull> {
    pub file_scope: IndexMap<FileId, ResultOrGuaranteed<Scope>>,
    pub scopes: Scopes<ScopedEntry>,

    pub items: Arena<Item, ItemInfo<S::ItemInfoT, S::ItemInfoB>>,
    
    pub generic_type_params: Arena<GenericTypeParameter, GenericTypeParameterInfo>,
    pub generic_value_params: Arena<GenericValueParameter, GenericValueParameterInfo>,
    pub function_params: Arena<FunctionParameter, FunctionParameterInfo>,
    pub module_info: IndexMap<Item, ModuleInfo>,
    pub module_ports: Arena<ModulePort, ModulePortInfo>,
}

pub trait CompiledStage {
    type ItemInfoT;
    type ItemInfoB;
}

pub struct CompiledStagePartial;

impl CompiledStage for CompiledStagePartial {
    type ItemInfoT = Option<MaybeConstructor<Type>>;
    type ItemInfoB = Option<ItemBody>;
}

pub struct CompiledStateFull;

impl CompiledStage for CompiledStateFull {
    type ItemInfoT = MaybeConstructor<Type>;
    type ItemInfoB = ItemBody;
}

new_index_type!(pub Item);
new_index_type!(pub GenericTypeParameter);
new_index_type!(pub GenericValueParameter);
new_index_type!(pub FunctionParameter);
new_index_type!(pub ModulePort);

pub type GenericParameter = TypeOrValue<GenericTypeParameter, GenericValueParameter>;

pub type ItemInfoPartial = ItemInfo<Option<MaybeConstructor<Type>>, Option<ItemBody>>;

#[derive(Debug)]
pub struct ItemInfo<T = MaybeConstructor<Type>, B = ItemBody> {
    pub ast_ref: ItemAstReference,
    pub ty: T,
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
    // TODO it's a bit weird that we're tracking the span here
    pub ty_span: Span,
}

#[derive(Debug)]
pub struct FunctionParameterInfo {
    pub defining_item: Item,
    pub defining_id: MaybeIdentifier,

    pub ty: Type,
}

#[derive(Debug)]
pub struct ModuleInfo {
    pub scope_ports: Scope,
    // TODO scope_inner?
}

#[derive(Debug)]
pub struct ModulePortInfo {
    pub defining_item: Item,
    pub defining_id: Identifier,

    pub direction: PortDirection,
    pub kind: PortKind<SyncKind<Value>, Type>,
}

#[derive(Debug)]
pub enum ItemBody {
    /// For items that don't have a body.
    None,
    Module(ModuleBody),
    // TODO expand to remaining items
    Error(ErrorGuaranteed),
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
    pub fn value_to_readable_str(&self, source: &SourceDatabase, value: &Value) -> String {
        match *value {
            // this should never actually come up, since there will never more future errors on error values
            Value::Error(_) => "value that corresponds to previously reported error".to_string(),
            // TODO include span to disambiguate
            Value::GenericParameter(p) => {
                let id = &self[p].defining_id;
                format!("generic_param({:?}, {:?})", id.string, source.expand_pos(id.span.start))
            }
            Value::FunctionParameter(p) => {
                let id = &self[p].defining_id;
                format!("function_param({:?}, {:?})", id.string(), source.expand_pos(id.span().start))
            }
            Value::ModulePort(p) => {
                let id = &self[p].defining_id;
                format!("module_port({:?}, {:?})", id.string, source.expand_pos(id.span.start))
            }
            Value::Int(ref v) => {
                if v.is_negative() {
                    format!("({})", v)
                } else {
                    format!("{}", v)
                }
            },
            Value::Range(ref range) => self.range_to_readable_str(source, range),
            Value::Binary(op, ref left, ref right) => {
                let left = self.value_to_readable_str(source, left);
                let right = self.value_to_readable_str(source, right);
                let symbol = op.symbol();
                format!("({} {} {})", left, symbol, right)
            }
            Value::UnaryNot(ref inner) => {
                let inner = self.value_to_readable_str(source, inner);
                format!("(!{})", inner)
            }
            Value::Function(_) => todo!(),
            Value::Module(_) => todo!(),
        }
    }
}

impl<S: CompiledStage> std::ops::Index<Item> for CompiledDatabase<S> {
    type Output = ItemInfo<S::ItemInfoT, S::ItemInfoB>;
    fn index(&self, index: Item) -> &Self::Output {
        &self.items[index]
    }
}

impl std::ops::IndexMut<Item> for CompiledDatabase<CompiledStagePartial> {
    fn index_mut(&mut self, index: Item) -> &mut ItemInfoPartial {
        &mut self.items[index]
    }
}

impl<S: CompiledStage> std::ops::Index<GenericTypeParameter> for CompiledDatabase<S> {
    type Output = GenericTypeParameterInfo;
    fn index(&self, index: GenericTypeParameter) -> &Self::Output {
        &self.generic_type_params[index]
    }
}

impl<S: CompiledStage> std::ops::Index<GenericValueParameter> for CompiledDatabase<S> {
    type Output = GenericValueParameterInfo;
    fn index(&self, index: GenericValueParameter) -> &Self::Output {
        &self.generic_value_params[index]
    }
}

impl<S: CompiledStage> std::ops::Index<FunctionParameter> for CompiledDatabase<S> {
    type Output = FunctionParameterInfo;
    fn index(&self, index: FunctionParameter) -> &Self::Output {
        &self.function_params[index]
    }
}

impl<S: CompiledStage> std::ops::Index<ModulePort> for CompiledDatabase<S> {
    type Output = ModulePortInfo;
    fn index(&self, index: ModulePort) -> &Self::Output {
        &self.module_ports[index]
    }
}

impl<S: CompiledStage> std::ops::Index<Scope> for CompiledDatabase<S> {
    type Output = ScopeInfo<ScopedEntry>;
    fn index(&self, index: Scope) -> &Self::Output {
        &self.scopes[index]
    }
}
