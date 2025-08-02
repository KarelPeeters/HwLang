//! Core ideas are based on:
//! * https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/
//! * https://yorickpeterse.com/articles/how-to-write-a-code-formatter/
//!
//! TODO think about comments
//! Comments are handled in a separate final pass.

// TODO should input be a string, an ast, tokens, ...?
//  we need access to the tokens anyway, so maybe just take a string? but then what if it's invalid?

use crate::syntax::ast::{
    ArenaExpressions, Block, CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, Expression,
    ExpressionKind, ExtraList, FileContent, Identifier, ImportAs, ImportEntry, ImportFinalKind, IntLiteral, Item,
    ItemImport, MaybeIdentifier, ModulePortItem, Parameters, Visibility,
};
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::syntax::token::tokenize;
use crate::syntax::{parse_file_content, ParseError};
use hwl_util::{swrite, swriteln};
use itertools::enumerate;

// TODO split out the core abstract formatting engine and the ast->IR conversion into separate modules?
// TODO add to python and wasm as a standalone function
// TODO add as LSP action
#[derive(Debug)]
pub struct FormatSettings {
    pub indent_str: String,
    pub tab_size: usize,
    pub max_line_length: usize,
    // TODO think about newline settings, maybe orthogonal axes:
    //   * recombine short lines: bool
    //   * wrap overlong lines: bool
    //   * fixup empty lines between defs/statements/comments
    // pub line_wrap: bool,
}

impl Default for FormatSettings {
    fn default() -> Self {
        Self {
            indent_str: "    ".to_string(),
            tab_size: 4,
            max_line_length: 120,
        }
    }
}

// TODO preserve comments?
// TODO what to do about multi-line comments?
// TODO fuzz test:
//   * insert block or line comment at every possible location, see if they all survive
//   * also do a variant where every tokens is on a separate line
pub fn format(source: &str, settings: &FormatSettings) -> Result<String, ParseError> {
    // tokenize and parse
    let file_id = FileId::dummy();
    let tokens = tokenize(file_id, source, false).map_err(|error| ParseError::User { error })?;
    let ast = parse_file_content(file_id, source)?;

    // convert to IR
    // TODO walk ast
    // TODO simple obvious data structure, don't bother with fancy arena allocation

    // flatten IR to string
    // TODO first basic
    // TODO restore comments

    // check
    // TODO re-parse to double-check that we didn't lose/gain any tokens?
    //    also check token contents for eg. string escaping
    //    maybe allow for changes to trailing commas

    // different idea, TODO decide whether to use it or the tree with nodes
    let mut f = String::new();

    let c = Context {
        source,
        arena_expressions: &ast.arena_expressions,
    };

    let node = c.format_file(&ast);

    println!("{}", node.tree_to_string(source));

    Ok(f)
}


// TODO how to handle spaces between keywords, should we include them in the literals? where should comments go?
#[derive(Debug)]
enum Node {


    Sequence(Vec<Node>),
    Literal(Span),
    // TODO refer to group id?
    SpaceOrLine,
    Line,
}

impl Node {
    fn empty() -> Node {
        Node::Sequence(vec![])
    }
}

struct Context<'a> {
    source: &'a str,
    arena_expressions: &'a ArenaExpressions,
}

impl Context<'_> {
    fn format_file(&self, file: &FileContent) -> Node {
        let FileContent {
            span: _,
            items,
            arena_expressions: _,
        } = file;

        let mut nodes = vec![];

        for (i, item) in enumerate(items) {
            nodes.push(self.format_item(item));
            nodes.push(Node::Line);

            // add separating newlines between non-import items
            if let Some(next) = items.get(i + 1) {
                if !(matches!(item, Item::Import(_)) && matches!(next, Item::Import(_))) {
                    nodes.push(Node::Line);
                }
            }
        }

        Node::Sequence(nodes)
    }

    fn format_item(&self, item: &Item) -> Node {
        match item {
            Item::Import(ItemImport {
                span_import,
                parents,
                entry,
                span_semi,
            }) => {
                // TODO sort imports, both within and across lines?
                let mut nodes = vec![];
                nodes.push(Node::Literal(*span_import));
                for parent in &parents.inner {
                    nodes.push(self.format_id(parent));
                }
                match &entry.inner {
                    ImportFinalKind::Single(entry) => {
                        nodes.push(self.format_import_entry(entry));
                    }
                    ImportFinalKind::Multi(entries) => {
                        // TODO
                        // todo!()
                        // swrite!(self.f, "[");
                        // for (entry, last) in entries.iter().with_last() {
                        //     self.format_import_entry(entry);
                        //     if !last {
                        //         swrite!(self.f, ", ");
                        //     }
                        // }
                        // swrite!(self.f, "]");
                    }
                }
                nodes.push(Node::Literal(*span_semi));
                Node::Sequence(nodes)
            }
            Item::CommonDeclaration(decl) => self.format_common_declaration(&decl.inner),
            Item::ModuleInternal(decl) => {
                todo!()
                // let ItemDefModuleInternal {
                //     span: _,
                //     vis,
                //     id,
                //     params,
                //     ports,
                //     body,
                // } = decl;
                // self.format_visibility(vis);
                // swrite!(self.f, "module ");
                // self.format_maybe_id(id);
                // if let Some(params) = params {
                //     self.format_params(params);
                // }
                // self.format_ports(&ports.inner);
                // self.format_block(body, |c, s| todo!());
            }
            Item::ModuleExternal(_) => todo!(),
            Item::Interface(_) => todo!(),
        }
    }

    fn format_import_entry(&self, entry: &ImportEntry) -> Node {
        let ImportEntry { span: _, id, as_ } = entry;

        let mut nodes = vec![self.format_id(id)];

        // TODO does this need a wrapping group?
        // TODO does every node sequence have the ability to wrap?
        if let Some(ImportAs { span_as, id: as_id }) = as_ {
            nodes.push(Node::SpaceOrLine);
            nodes.push(Node::Literal(*span_as));
            nodes.push(self.format_maybe_id(as_id));
        }

        Node::Sequence(nodes)
    }

    fn format_common_declaration<V: FormatVisibility>(&self, decl: &CommonDeclaration<V>) -> Node {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                self.format_visibility(vis);

                match kind {
                    CommonDeclarationNamedKind::Type(_) => todo!(),
                    CommonDeclarationNamedKind::Const(decl) => {
                        // let ConstDeclaration { span: _, id, ty, value } = decl;
                        //
                        // swrite!(self.f, "const ");
                        // self.format_maybe_id(id);
                        // if let Some(ty) = ty {
                        //     swrite!(self.f, ": ");
                        //     self.format_expr(ty);
                        // }
                        // swrite!(self.f, " = ");
                        // self.format_expr(value);
                        // swrite!(self.f, ";");

                        todo!()
                    }
                    CommonDeclarationNamedKind::Struct(_) => todo!(),
                    CommonDeclarationNamedKind::Enum(_) => todo!(),
                    CommonDeclarationNamedKind::Function(_) => todo!(),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }
    }

    fn format_params(&self, params: &Parameters) -> Node {
        todo!()
    }

    fn format_ports(&self, ports: &ExtraList<ModulePortItem>) -> Node {
        todo!()
        // let ExtraList { span: _, items } = ports;<
        //
        // if items.is_empty() {
        //     swrite!(self.f, "ports()");
        //     return;
        // }
        //
        // swriteln!(self.f, "ports(");
        // for port in items {
        //     match port {
        //         ExtraItem::Inner(item) => {
        //             todo!()
        //         }
        //         ExtraItem::Declaration(_) => todo!(),
        //         ExtraItem::If(_) => todo!(),
        //     }
        // }
        // swrite!(self.f, ")");
    }

    fn format_block<S>(&self, block: &Block<S>, f: impl Fn(&mut Self, &S)) -> Node {
        todo!()
    }

    fn format_expr(&self, expr: &Expression) -> Node {
        match &self.arena_expressions[expr.inner] {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Undefined => todo!(),
            ExpressionKind::Type => todo!(),
            ExpressionKind::TypeFunction => todo!(),
            ExpressionKind::Wrapped(_) => todo!(),
            ExpressionKind::Block(_) => todo!(),
            ExpressionKind::Id(_) => todo!(),
            ExpressionKind::IntLiteral(lit) => {
                let span = match *lit {
                    IntLiteral::Binary(span) => span,
                    IntLiteral::Decimal(span) => span,
                    IntLiteral::Hexadecimal(span) => span,
                };
                Node::Literal(span)
            }
            ExpressionKind::BoolLiteral(_) => todo!(),
            ExpressionKind::StringLiteral(_) => todo!(),
            ExpressionKind::ArrayLiteral(_) => todo!(),
            ExpressionKind::TupleLiteral(_) => todo!(),
            ExpressionKind::RangeLiteral(_) => todo!(),
            ExpressionKind::ArrayComprehension(_) => todo!(),
            ExpressionKind::UnaryOp(_, _) => todo!(),
            ExpressionKind::BinaryOp(_, _, _) => todo!(),
            ExpressionKind::ArrayType(_, _) => todo!(),
            ExpressionKind::ArrayIndex(_, _) => todo!(),
            ExpressionKind::DotIdIndex(_, _) => todo!(),
            ExpressionKind::DotIntIndex(_, _) => todo!(),
            ExpressionKind::Call(_, _) => todo!(),
            ExpressionKind::Builtin(_) => todo!(),
            ExpressionKind::UnsafeValueWithDomain(_, _) => todo!(),
            ExpressionKind::RegisterDelay(_) => todo!(),
        }
    }

    fn format_visibility<V: FormatVisibility>(&self, vis: &V) -> Node {
        V::format_visibility(self, vis)
    }

    fn format_maybe_id(&self, id: &MaybeIdentifier) -> Node {
        match id {
            &MaybeIdentifier::Dummy(span) => Node::Literal(span),
            MaybeIdentifier::Identifier(id) => self.format_id(id),
        }
    }

    fn format_id(&self, id: &Identifier) -> Node {
        let &Identifier { span } = id;
        Node::Literal(span)
    }
}

trait FormatVisibility {
    fn format_visibility(c: &Context, vis: &Self) -> Node;
}

impl FormatVisibility for Visibility {
    fn format_visibility(c: &Context, vis: &Visibility) -> Node {
        match *vis {
            Visibility::Public(span) => Node::Sequence(vec![Node::Literal(span), Node::SpaceOrLine]),
            Visibility::Private => Node::empty(),
        }
    }
}

impl FormatVisibility for () {
    fn format_visibility(_: &Context, _: &()) -> Node {
        Node::empty()
    }
}

impl Node {
    pub fn tree_to_string(&self, source: &str) -> String {
        fn node_to_string(f: &mut String, source: &str, indent: usize, node: &Node) {
            for _ in 0..indent {
                swrite!(f, " |  ");
            }
            match node {
                Node::Sequence(children) => {
                    swriteln!(f, "Sequence");
                    for child in children {
                        node_to_string(f, source, indent + 1, child);
                    }
                }
                &Node::Literal(span) => {
                    let s = &source[span.range_bytes()];
                    swriteln!(f, "Literal({span:?}, {s:?})",)
                }
                Node::SpaceOrLine => swriteln!(f, "SpaceOrLine"),
                Node::Line => swriteln!(f, "Line"),
            }
        }

        let mut f = String::new();
        node_to_string(&mut f, source, 0, self);
        f
    }
}
