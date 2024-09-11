use crate::data::diagnostic::DiagnosticResult;
use crate::data::module_body::ModuleBody;
use crate::front::common::{ScopedEntry, TypeOrValue};
use crate::front::scope::{Scope, ScopeInfo, Scopes};
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::new_index_type;
use crate::syntax::ast;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, PortKind, SyncKind};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::Arena;
use crate::util::ResultExt;
use indexmap::IndexMap;

// TODO move this somewhere else, this is more of a public interface
// TODO separate read-only and clearly done iteminfo struct 
pub struct CompiledDatabase {
    pub file_auxiliary: IndexMap<FileId, DiagnosticResult<FileAuxiliary>>,
    pub items: Arena<Item, ItemInfo>,
    pub generic_type_params: Arena<GenericTypeParameter, GenericTypeParameterInfo>,
    pub generic_value_params: Arena<GenericValueParameter, GenericValueParameterInfo>,
    pub function_params: Arena<FunctionParameter, FunctionParameterInfo>,
    pub module_ports: Arena<ModulePort, ModulePortInfo>,
    pub scopes: Scopes<ScopedEntry>,
}

new_index_type!(pub Item);
new_index_type!(pub GenericTypeParameter);
new_index_type!(pub GenericValueParameter);
new_index_type!(pub FunctionParameter);
new_index_type!(pub ModulePort);

pub type GenericParameter = TypeOrValue<GenericTypeParameter, GenericValueParameter>;

pub struct FileAuxiliary {
    pub ast: ast::FileContent,
    // TODO distinguish scopes properly, there are up to 3:
    //   * containing items defined in this file
    //   * containing sibling files
    //   * including imports
    pub local_scope: Scope,
}

pub struct ItemInfo {
    pub file: FileId,
    pub file_item_index: usize,
    pub ty: DiagnosticResult<MaybeConstructor<Type>>,
    pub body: DiagnosticResult<ItemBody>,
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
}

impl CompiledDatabase {
    pub fn get_item_ast(&self, item: Item) -> DiagnosticResult<&ast::Item> {
        let info = &self[item];
        self.file_auxiliary.get(&info.file).unwrap()
            .as_ref_ok()
            .map(|aux| &aux.ast.items[info.file_item_index])
    }
}

impl std::ops::Index<FileId> for CompiledDatabase {
    type Output = DiagnosticResult<FileAuxiliary>;
    fn index(&self, index: FileId) -> &Self::Output {
        self.file_auxiliary.get(&index).unwrap()
    }
}

impl std::ops::Index<Item> for CompiledDatabase {
    type Output = ItemInfo;
    fn index(&self, index: Item) -> &Self::Output {
        &self.items[index]
    }
}

impl std::ops::Index<GenericTypeParameter> for CompiledDatabase {
    type Output = GenericTypeParameterInfo;
    fn index(&self, index: GenericTypeParameter) -> &Self::Output {
        &self.generic_type_params[index]
    }
}

impl std::ops::Index<GenericValueParameter> for CompiledDatabase {
    type Output = GenericValueParameterInfo;
    fn index(&self, index: GenericValueParameter) -> &Self::Output {
        &self.generic_value_params[index]
    }
}

impl std::ops::Index<FunctionParameter> for CompiledDatabase {
    type Output = FunctionParameterInfo;
    fn index(&self, index: FunctionParameter) -> &Self::Output {
        &self.function_params[index]
    }
}

impl std::ops::Index<ModulePort> for CompiledDatabase {
    type Output = ModulePortInfo;
    fn index(&self, index: ModulePort) -> &Self::Output {
        &self.module_ports[index]
    }
}

impl std::ops::Index<Scope> for CompiledDatabase {
    type Output = ScopeInfo<ScopedEntry>;
    fn index(&self, index: Scope) -> &Self::Output {
        &self.scopes[index]
    }
}
