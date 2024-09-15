use crate::data::diagnostic::ResultOrGuaranteed;
use crate::syntax::ast;
use crate::syntax::pos::FileId;
use indexmap::IndexMap;

// TODO represent the set of existing items here already, so direct lookups become possible
pub struct ParsedDatabase {
    pub file_ast: IndexMap<FileId, ResultOrGuaranteed<ast::FileContent>>,
}

#[derive(Debug, Copy, Clone)]
pub struct ItemAstReference {
    pub file: FileId,
    pub file_item_index: usize,
}

impl ParsedDatabase {
    pub fn item_ast(&self, item: ItemAstReference) -> &ast::Item {
        let ItemAstReference { file, file_item_index } = item;
        let aux = self.file_ast.get(&file).unwrap()
            .as_ref()
            .expect("the item existing implies that the auxiliary info exists too");
        &aux.items[file_item_index]
    }
}
