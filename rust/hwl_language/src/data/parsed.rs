use crate::data::diagnostic::ErrorGuaranteed;
use crate::syntax::ast;
use crate::syntax::ast::FileContent;
use crate::syntax::pos::FileId;
use indexmap::IndexMap;
use unwrap_match::unwrap_match;

// TODO represent the set of existing items here already, so direct lookups become possible
pub struct ParsedDatabase {
    pub file_ast: IndexMap<FileId, Result<FileContent, ErrorGuaranteed>>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ItemAstReference {
    pub file: FileId,
    pub file_item_index: usize,
}

// TODO general way to point back into the ast?
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ModulePortAstReference {
    pub item: ItemAstReference,
    pub index: usize,
}

impl ParsedDatabase {
    pub fn item_ast(&self, item: ItemAstReference) -> &ast::Item {
        let ItemAstReference { file, file_item_index } = item;
        let aux = self.file_ast.get(&file).unwrap()
            .as_ref()
            .expect("the item existing implies that the auxiliary info exists too");
        &aux.items[file_item_index]
    }

    pub fn module_port_ast(&self, port: ModulePortAstReference) -> &ast::ModulePort {
        let ModulePortAstReference { item, index } = port;
        let item_ast = self.item_ast(item);
        let module_ast = unwrap_match!(item_ast, ast::Item::Module(item) => item);
        &module_ast.ports.inner[index]
    }
}
