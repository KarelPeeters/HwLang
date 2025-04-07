use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::syntax::ast::{FileContent, Identifier, ModulePortInBlock, ModulePortSingle, Spanned, Visibility, WireKind};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::{ast, parse_error_to_diagnostic, parse_file_content};
use crate::util::arena::IndexType;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use std::fmt::{Debug, Formatter};
use unwrap_match::unwrap_match;

// TODO represent the set of existing items here already, so direct lookups become possible
// TODO merge with SourceDatabase?
pub struct ParsedDatabase {
    file_ast: IndexMap<FileId, Result<FileContent, ErrorGuaranteed>>,
}

impl ParsedDatabase {
    pub fn new(diags: &Diagnostics, source: &SourceDatabase) -> Self {
        let mut file_ast = IndexMap::new();

        for file_id in source.files() {
            let file_info = &source[file_id];
            let ast =
                parse_file_content(file_id, &file_info.source).map_err(|e| diags.report(parse_error_to_diagnostic(e)));
            file_ast.insert_first(file_id, ast);
        }

        Self { file_ast }
    }

    pub fn module_port_ast(&self, port: AstRefModulePort) -> ModulePort {
        let AstRefModulePort {
            module,
            port_item_index,
            port_in_block_index,
        } = port;
        let module_ast = &self[module];
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

// TODO general way to point back into the ast? should we just switch to actual references with lifetimes?
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct AstRefItem {
    file: FileId,
    file_item_index: usize,
}

impl Debug for AstRefItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AstRefItem([{}]#{})",
            self.file.inner().index(),
            self.file_item_index
        )
    }
}

impl AstRefItem {
    pub fn file(self) -> FileId {
        self.file
    }
}

macro_rules! impl_ast_ref_alias {
    ($ref_name:ident, $item_path:path, $ast_path:path) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
        pub struct $ref_name {
            item: AstRefItem,
        }

        impl $ref_name {
            pub fn new_unchecked(item: AstRefItem) -> Self {
                Self { item }
            }

            pub fn file(self) -> FileId {
                self.item.file()
            }
        }

        impl std::ops::Index<$ref_name> for ParsedDatabase {
            type Output = $ast_path;
            fn index(&self, item: $ref_name) -> &Self::Output {
                let item_ast = &self[item.item];
                unwrap_match!(item_ast, $item_path(inner) => inner)
            }
        }
    };
}

impl_ast_ref_alias!(AstRefModule, ast::Item::Module, ast::ItemDefModule);
impl_ast_ref_alias!(AstRefDefType, ast::Item::Type, ast::TypeDeclaration<Visibility<Span>>);
impl_ast_ref_alias!(AstRefDefStruct, ast::Item::Struct, ast::ItemDefStruct);
impl_ast_ref_alias!(AstRefDefEnum, ast::Item::Enum, ast::ItemDefEnum);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct AstRefModulePort {
    module: AstRefModule,
    port_item_index: usize,
    port_in_block_index: Option<usize>,
}

#[derive(Debug, Copy, Clone)]
pub enum ModulePort<'a> {
    Single(&'a ModulePortSingle),
    InBlock(&'a ModulePortInBlock),
}

impl FileContent {
    pub fn items_with_ref(&self) -> impl Iterator<Item = (AstRefItem, &ast::Item)> {
        self.items.iter().enumerate().map(move |(i, item)| {
            (
                AstRefItem {
                    file: self.span.start.file,
                    file_item_index: i,
                },
                item,
            )
        })
    }
}

impl std::ops::Index<FileId> for ParsedDatabase {
    type Output = Result<FileContent, ErrorGuaranteed>;
    fn index(&self, file: FileId) -> &Self::Output {
        self.file_ast.get(&file).unwrap()
    }
}

impl std::ops::Index<AstRefItem> for ParsedDatabase {
    type Output = ast::Item;
    fn index(&self, item: AstRefItem) -> &Self::Output {
        let AstRefItem { file, file_item_index } = item;
        let aux = self
            .file_ast
            .get(&file)
            .unwrap()
            .as_ref()
            .expect("the item existing implies that the auxiliary info exists too");
        &aux.items[file_item_index]
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
                WireKind::Clock => port.kind.span,
                WireKind::Normal { domain: _, ty } => ty.span,
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
