use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::syntax::ast::{Expression, ExpressionKind, FileContent};
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

    pub fn get_expr(&self, expr: Expression) -> &ExpressionKind {
        let file_content = self[expr.span.start.file].as_ref().unwrap();
        &file_content.arena_expressions[expr.inner]
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
            pub fn new_unchecked(item: AstRefItem, item_ref: &$ast_path) -> Self {
                // just for some soft extra checking
                let _ = item_ref;
                Self { item }
            }

            pub fn item(self) -> AstRefItem {
                self.item
            }

            pub fn file(self) -> FileId {
                self.item.file()
            }
        }

        impl From<$ref_name> for AstRefItem {
            fn from(item: $ref_name) -> Self {
                item.item
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

impl_ast_ref_alias!(
    AstRefModuleInternal,
    ast::Item::ModuleInternal,
    ast::ItemDefModuleInternal
);
impl_ast_ref_alias!(
    AstRefModuleExternal,
    ast::Item::ModuleExternal,
    ast::ItemDefModuleExternal
);
impl_ast_ref_alias!(AstRefInterface, ast::Item::Interface, ast::ItemDefInterface);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct AstRefModulePort {
    module: AstRefModuleInternal,
    port_item_index: usize,
    port_in_block_index: Option<usize>,
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
