use crate::data::diagnostic::ErrorGuaranteed;
use crate::syntax::ast;
use crate::syntax::ast::{FileContent, Identifier, ModulePortInBlock, ModulePortSingle, PortKind, Spanned};
use crate::syntax::pos::{FileId, Span};
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
    pub port_item_index: usize,
    pub port_in_block_index: Option<usize>,
}

#[derive(Debug, Copy, Clone)]
pub enum ModulePort<'a> {
    Single(&'a ModulePortSingle),
    InBlock(&'a ModulePortInBlock),
}

impl ParsedDatabase {
    pub fn item_ast(&self, item: ItemAstReference) -> &ast::Item {
        let ItemAstReference { file, file_item_index } = item;
        let aux = self.file_ast.get(&file).unwrap()
            .as_ref()
            .expect("the item existing implies that the auxiliary info exists too");
        &aux.items[file_item_index]
    }

    pub fn module_ast(&self, item: ItemAstReference) -> &ast::ItemDefModule {
        let item_ast = self.item_ast(item);
        unwrap_match!(item_ast, ast::Item::Module(item) => item)
    }

    pub fn module_port_ast(&self, port: ModulePortAstReference) -> ModulePort {
        let ModulePortAstReference { item, port_item_index, port_in_block_index } = port;
        let module_ast = self.module_ast(item);
        let item = &module_ast.ports.inner[port_item_index];

        match port_in_block_index {
            None => ModulePort::Single(unwrap_match!(item, ast::ModulePortItem::Single(port) => port)),
            Some(port_in_block_index) => {
                let block = unwrap_match!(item, ast::ModulePortItem::Block(block) => block);
                ModulePort::InBlock(&block.ports[port_in_block_index])
            }
        }
    }
}

impl<'a> ModulePort<'a> {
    pub fn id(self) -> &'a Identifier {
        match self {
            ModulePort::Single(port) => &port.id,
            ModulePort::InBlock(port) => &port.id,
        }
    }

    pub fn ty_span(self) -> Span {
        match self {
            ModulePort::Single(port) => match &port.kind.inner {
                PortKind::Clock => port.kind.span,
                PortKind::Normal { domain: _, ty } => ty.span,
            },
            ModulePort::InBlock(port) => port.ty.span,
        }
    }

    pub fn direction(self) -> &'a Spanned<ast::PortDirection> {
        match self {
            ModulePort::Single(port) => &port.direction,
            ModulePort::InBlock(port) => &port.direction,
        }
    }
}