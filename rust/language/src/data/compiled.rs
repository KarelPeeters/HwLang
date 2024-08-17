use crate::front::common::{ItemReference, TypeOrValue};
use crate::front::driver::Item;
use crate::front::scope::Scope;
use crate::front::types::{MaybeConstructor, Type};
use crate::front::values::Value;
use crate::new_index_type;
use crate::syntax::ast;
use crate::syntax::ast::{Identifier, MaybeIdentifier, PortDirection, PortKind, SyncKind};
use crate::syntax::pos::{FileId, Span};
use crate::util::arena::Arena;
use indexmap::IndexMap;

// TODO move this somewhere else, this is more of a public interface
// TODO separate read-only and clearly done iteminfo struct 
pub struct CompiledDatabase {
    pub file_auxiliary: IndexMap<FileId, FileAuxiliary>,
    pub items: Arena<Item, ItemInfo>,
    pub generic_type_params: Arena<GenericTypeParameter, GenericTypeParameterInfo>,
    pub generic_value_params: Arena<GenericValueParameter, GenericValueParameterInfo>,
    pub function_params: Arena<FunctionParameter, FunctionParameterInfo>,
    pub module_ports: Arena<ModulePort, ModulePortInfo>,
}

pub type GenericParameter = TypeOrValue<GenericTypeParameter, GenericValueParameter>;

new_index_type!(pub GenericTypeParameter);
new_index_type!(pub GenericValueParameter);
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
pub struct GenericTypeParameterInfo {
    pub defining_item: ItemReference,
    pub defining_id: Identifier,

    // TODO type constraints once we add those
}

#[derive(Debug)]
pub struct GenericValueParameterInfo {
    pub defining_item: ItemReference,
    pub defining_id: Identifier,

    pub ty: Type,
    // TODO it's a bit weird that we're tracking the span here
    pub ty_span: Span,
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

impl CompiledDatabase {
    pub fn get_item_ast(&self, item_reference: ItemReference) -> &ast::Item {
        let ItemReference { file, item_index } = item_reference;
        let ast = &self.file_auxiliary.get(&file).unwrap().ast;
        &ast.items[item_index]
    }
}

impl std::ops::Index<FileId> for CompiledDatabase {
    type Output = FileAuxiliary;
    fn index(&self, index: FileId) -> &Self::Output {
        &self.file_auxiliary.get(&index).unwrap()
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
