//! Core ideas are based on:
//! * https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/
//! * https://yorickpeterse.com/articles/how-to-write-a-code-formatter/
//!
//! TODO think about comments
//! Comments are handled in a separate final pass.

// TODO should input be a string, an ast, tokens, ...?
//  we need access to the tokens anyway, so maybe just take a string? but then what if it's invalid?

use crate::syntax::ast::{
    ArenaExpressions, Block, CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration,
    Expression, ExpressionKind, ExtraList, FileContent, Identifier, ImportAs, ImportEntry, ImportFinalKind, ImportStep,
    IntLiteral, Item, ItemImport, MaybeIdentifier, ModulePortItem, Parameters, Visibility,
};
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::syntax::token::tokenize;
use crate::syntax::{parse_file_content, ParseError};
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
    let mut result = String::new();

    let mut c = Context {
        source_file: file_id,
        source_str: source,
        arena_expressions: &ast.arena_expressions,
        result: &mut result,
    };
    c.format_file(&ast);

    // TODO re-parse, check token equivalence (and maybe eventually AST equivalence?)

    Ok(result)
}

struct Context<'a> {
    source_file: FileId,
    source_str: &'a str,
    arena_expressions: &'a ArenaExpressions,

    result: &'a mut String,
}

// TODO split the non-ast specific stuff into separate file/module
impl Context<'_> {
    fn push_copy(&mut self, span: Span) {
        // TODO copy over any comments that happened before span
        assert_eq!(span.file, self.source_file);
        self.result.push_str(&self.source_str[span.range_bytes()]);
    }

    fn push_space(&mut self) {
        // no need to check for comments
        // TODO is that right?
        self.result.push(' ');
    }

    fn push_newline(&mut self) {
        // no need to check for comments
        // TODO is that right?
        // TODO indentation
        // TODO should we use \r\n on windows?
        self.result.push('\n');
    }
}

impl Context<'_> {
    fn format_file(&mut self, file: &FileContent) {
        let FileContent {
            span: _,
            items,
            arena_expressions: _,
        } = file;

        for (i, item) in enumerate(items) {
            self.format_item(item);

            // add extra newlines between non-import items
            if let Some(next) = items.get(i + 1) {
                if !(matches!(item, Item::Import(_)) && matches!(next, Item::Import(_))) {
                    self.push_newline();
                }
            }
        }
    }

    fn format_item(&mut self, item: &Item) {
        match item {
            Item::Import(item) => {
                let &ItemImport {
                    span_import,
                    ref parents,
                    ref entry,
                    span_semi,
                } = item;
                // TODO sort imports, both within and across lines, combining them in some single-correct way
                // TODO move imports to top of file?

                // TODO implement line wrapping
                self.push_copy(span_import);
                self.push_space();
                for parent in &parents.inner {
                    let &ImportStep { id: id, span_dot } = parent;
                    self.format_id(id);
                    self.push_copy(span_dot);
                }
                match &entry.inner {
                    ImportFinalKind::Single(entry) => {
                        self.format_import_entry(entry);
                    }
                    ImportFinalKind::Multi(entries) => {
                        // TODO definitely implement line wrapping here

                        // TODO
                        todo!()
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
                self.push_copy(span_semi);
                self.push_newline();
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

    fn format_import_entry(&mut self, entry: &ImportEntry) {
        // TODO support wrapping?
        let &ImportEntry { span: _, id, as_ } = entry;
        self.format_id(id);
        if let Some(ImportAs { span_as, as_id }) = as_ {
            self.push_space();
            self.push_copy(span_as);
            self.push_space();
            self.format_maybe_id(as_id);
        }
    }

    fn format_common_declaration<V: FormatVisibility>(&mut self, decl: &CommonDeclaration<V>) {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                self.format_visibility(vis);

                match kind {
                    CommonDeclarationNamedKind::Type(_) => todo!(),
                    CommonDeclarationNamedKind::Const(decl) => {
                        let &ConstDeclaration {
                            span_const,
                            id,
                            ty,
                            span_eq,
                            value,
                            span_semi,
                        } = decl;

                        self.push_copy(span_const);
                        self.push_space();
                        self.format_maybe_id(id);
                        if let Some(ty) = ty {
                            todo!()
                        }
                        self.push_space();
                        self.push_copy(span_eq);
                        self.push_space();
                        self.format_expr(value);
                        self.push_copy(span_semi);
                    }
                    CommonDeclarationNamedKind::Struct(_) => todo!(),
                    CommonDeclarationNamedKind::Enum(_) => todo!(),
                    CommonDeclarationNamedKind::Function(_) => todo!(),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }
    }

    fn format_params(&mut self, params: &Parameters) {
        todo!()
    }

    fn format_ports(&mut self, ports: &ExtraList<ModulePortItem>) {
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

    fn format_block<S>(&mut self, block: &Block<S>, f: impl Fn(&mut Self, &S)) {
        todo!()
    }

    fn format_expr(&mut self, expr: Expression) {
        match &self.arena_expressions[expr.inner] {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Undefined => todo!(),
            ExpressionKind::Type => todo!(),
            ExpressionKind::TypeFunction => todo!(),
            ExpressionKind::Wrapped(_) => todo!(),
            ExpressionKind::Block(_) => todo!(),
            ExpressionKind::Id(_) => todo!(),
            ExpressionKind::IntLiteral(lit) => match *lit {
                IntLiteral::Binary(span) => self.push_copy(span),
                IntLiteral::Decimal(span) => self.push_copy(span),
                IntLiteral::Hexadecimal(span) => self.push_copy(span),
            },
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

    fn format_visibility<V: FormatVisibility>(&mut self, vis: &V) {
        V::format_visibility(self, vis)
    }

    fn format_maybe_id(&mut self, id: MaybeIdentifier) {
        match id {
            MaybeIdentifier::Dummy(span) => todo!(),
            MaybeIdentifier::Identifier(id) => self.format_id(id),
        }
    }

    fn format_id(&mut self, id: Identifier) {
        self.push_copy(id.span);
    }
}

trait FormatVisibility {
    fn format_visibility(c: &Context, vis: &Self);
}

impl FormatVisibility for Visibility {
    fn format_visibility(c: &Context, vis: &Visibility) {
        match *vis {
            Visibility::Public(span) => todo!(),
            Visibility::Private => {
                // do nothing, private visibility is the default
            }
        }
    }
}

impl FormatVisibility for () {
    fn format_visibility(_: &Context, _: &()) {
        todo!()
    }
}
