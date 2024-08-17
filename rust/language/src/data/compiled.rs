use crate::front::common::ItemReference;
use crate::front::driver::Item;
use crate::front::scope::Scope;
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::new_index_type;
use crate::syntax::ast;
use crate::syntax::ast::{GenericParameterKind, Identifier, MaybeIdentifier, PortDirection, PortKind, SyncKind};
use crate::syntax::pos::FileId;
use crate::util::arena::Arena;
use indexmap::IndexMap;

// TODO move this somewhere else, this is more of a public interface
// TODO separate read-only and clearly done iteminfo struct 
pub struct CompiledDataBase {
    pub file_auxiliary: IndexMap<FileId, FileAuxiliary>,
    pub items: Arena<Item, ItemInfo>,
    pub generic_params: Arena<GenericParameter, GenericParameterInfo>,
    pub function_params: Arena<FunctionParameter, FunctionParameterInfo>,
    pub module_ports: Arena<ModulePort, ModulePortInfo>,
}

// TODO add typed wrappers that distinguish between types and values?
new_index_type!(pub GenericParameter);
new_index_type!(pub FunctionParameter);
new_index_type!(pub ModulePort);

pub struct FileAuxiliary {
    pub ast: ast::FileContent,
    // TODO distinguish scopes properly, there are up to 3:
    //   * containing items defined in this file
    //   * containing sibling files
    //   * including imports
    pub local_scope: Scope<'static>,
}

pub struct ItemInfo {
    pub item_reference: ItemReference,
    pub ty: MaybeConstructor<Type>,
}

#[derive(Debug)]
pub struct GenericParameterInfo {
    pub defining_item: ItemReference,
    pub defining_id: Identifier,

    pub kind: GenericParameterKind<Type>,
}

#[derive(Debug)]
pub struct FunctionParameterInfo {
    pub defining_item: ItemReference,
    pub defining_id: MaybeIdentifier,

    pub ty: Type,
}

#[derive(Debug)]
pub struct ModulePortInfo {
    pub defining_item: ItemReference,
    pub defining_id: Identifier,

    pub direction: PortDirection,
    pub kind: PortKind<SyncKind<Value>, Type>,
}

impl CompiledDataBase {
    pub fn get_item_ast(&self, item_reference: ItemReference) -> &ast::Item {
        let ItemReference { file, item_index } = item_reference;
        let ast = &self.file_auxiliary.get(&file).unwrap().ast;
        &ast.items[item_index]
    }
}

impl std::ops::Index<FileId> for CompiledDataBase {
    type Output = FileAuxiliary;
    fn index(&self, index: FileId) -> &Self::Output {
        &self.file_auxiliary.get(&index).unwrap()
    }
}

impl std::ops::Index<Item> for CompiledDataBase {
    type Output = ItemInfo;
    fn index(&self, index: Item) -> &Self::Output {
        &self.items[index]
    }
}

impl std::ops::Index<GenericParameter> for CompiledDataBase {
    type Output = GenericParameterInfo;
    fn index(&self, index: GenericParameter) -> &Self::Output {
        &self.generic_params[index]
    }
}

impl std::ops::Index<FunctionParameter> for CompiledDataBase {
    type Output = FunctionParameterInfo;
    fn index(&self, index: FunctionParameter) -> &Self::Output {
        &self.function_params[index]
    }
}

impl std::ops::Index<ModulePort> for CompiledDataBase {
    type Output = ModulePortInfo;
    fn index(&self, index: ModulePort) -> &Self::Output {
        &self.module_ports[index]
    }
}
