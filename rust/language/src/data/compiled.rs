use crate::front::common::ItemReference;
use crate::front::driver::Item;
use crate::front::scope::Scope;
use crate::front::types::{MaybeConstructor, Type};
use crate::syntax::ast;
use crate::syntax::pos::FileId;
use crate::util::arena::Arena;
use indexmap::IndexMap;

// TODO move this somewhere else, this is more of a public interface
// TODO separate read-only and clearly done iteminfo struct 
pub struct CompiledDataBase {
    pub file_auxiliary: IndexMap<FileId, FileAuxiliary>,
    pub items: Arena<Item, ItemInfo>,
}

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