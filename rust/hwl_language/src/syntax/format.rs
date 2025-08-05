//! Core ideas are based on:
//! * https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/
//! * https://yorickpeterse.com/articles/how-to-write-a-code-formatter/
//!
//! TODO think about comments
//! Comments are handled in a separate final pass.

// TODO should input be a string, an ast, tokens, ...?
//  we need access to the tokens anyway, so maybe just take a string? but then what if it's invalid?
// TODO create a test file that contains every syntactical construct, which should fully cover the parser and this

use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::syntax::ast::{
    ArenaExpressions, Block, CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration,
    Expression, ExpressionKind, ExtraList, FileContent, Identifier, ImportAs, ImportEntry, ImportFinalKind, ImportStep,
    IntLiteral, Item, ItemImport, MaybeIdentifier, ModulePortItem, Parameters, Visibility,
};
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{tokenize, Token, TokenCategory, TokenType};
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};
use crate::util::iter::IterExt;
use itertools::enumerate;

// TODO split out the core abstract formatting engine and the ast->IR conversion into separate modules?
// TODO add to python and wasm as a standalone function
// TODO add as LSP action
#[derive(Debug)]
pub struct FormatSettings {
    pub indent_str: String,
    // TODO tab size is such a weird setting, it only matters if `indent_str` or literals contain tabs,
    //   do we really want to support this?
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
// TODO should this work with diags too?
pub fn format(
    diags: &Diagnostics,
    source: &SourceDatabase,
    file: FileId,
    settings: &FormatSettings,
) -> DiagResult<String> {
    // tokenize and parse
    let source_str = &source[file].content;
    let tokens = tokenize(file, source_str, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    let ast = parse_file_content(file, source_str).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;

    // format to string
    let mut result = String::new();

    let mut c = Context {
        source_file: file,
        source_str,
        source_tokens: &tokens,
        source_expressions: &ast.arena_expressions,

        diags,
        next_source_token: 0,
        result: &mut result,
    };
    c.format_file(&ast)?;

    // TODO re-tokenize, check token equivalence (and maybe eventually AST equivalence?)

    Ok(result)
}

struct Context<'a> {
    source_file: FileId,
    source_str: &'a str,
    source_tokens: &'a [Token],
    source_expressions: &'a ArenaExpressions,

    diags: &'a Diagnostics,
    result: &'a mut String,
    next_source_token: usize,
}

impl Context<'_> {
    fn push(&mut self, ty: TokenType) -> DiagResult<()> {
        // TODO improve and double-check comma explanation: (dis)appear, unambiguous
        let diags = self.diags;

        let token_str = loop {
            // pop the next source token
            let source_token = self.source_tokens.get(self.next_source_token).ok_or_else(|| {
                let end_pos = Pos {
                    file: self.source_file,
                    byte: self.source_str.len(),
                };
                let end_span = Span::empty_at(end_pos);
                let msg = format!("pushing token {ty:?} but no tokens left");
                diags.report_internal_error(end_span, msg)
            })?;
            self.next_source_token += 1;

            match source_token.ty.category() {
                // drop whitespace, the formatter creates its own
                TokenCategory::WhiteSpace => continue,
                // emit comments
                TokenCategory::Comment => {
                    // TODO emit comments instead of dropping them
                    continue;
                }
                // these are real tokens, fallthrough into matching the type
                TokenCategory::Identifier
                | TokenCategory::IntegerLiteral
                | TokenCategory::StringLiteral
                | TokenCategory::Keyword
                | TokenCategory::Symbol => {}
            }

            if source_token.ty == ty {
                // found matching token, emit it
                break &self.source_str[source_token.span.range_bytes()];
            }

            if source_token.ty == TokenType::Comma {
                // comma in original source that disappeared, continue searching for the actually matching token
                continue;
            }
            if ty == TokenType::Comma {
                // comma that was not in original source but appeared due to formatting,
                //   un-pop the source token and act as if we found a match
                self.next_source_token -= 1;
                break ",";
            }

            let msg = format!("push_token expected {:?} but got {:?}", ty, source_token.ty);
            return Err(diags.report_internal_error(source_token.span, msg));
        };

        self.result.push_str(token_str);
        Ok(())
    }

    fn push_with_span(&mut self, ty: TokenType, span: Span) -> DiagResult<()> {
        // TODO use span
        let _ = span;
        self.push(ty)
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
    fn format_file(&mut self, file: &FileContent) -> DiagResult<()> {
        let FileContent {
            span: _,
            items,
            arena_expressions: _,
        } = file;

        for (i, item) in enumerate(items) {
            self.format_item(item)?;

            // add extra newlines between non-import items
            if let Some(next) = items.get(i + 1) {
                if !(matches!(item, Item::Import(_)) && matches!(next, Item::Import(_))) {
                    self.push_newline();
                }
            }
        }

        Ok(())
    }

    fn format_item(&mut self, item: &Item) -> DiagResult<()> {
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
                self.push(TokenType::Import)?;
                self.push_space();
                for parent in &parents.inner {
                    let &ImportStep { id, span_dot } = parent;
                    self.format_id(id)?;
                    self.push(TokenType::Dot)?;
                }
                match &entry.inner {
                    ImportFinalKind::Single(entry) => {
                        self.format_import_entry(entry)?;
                    }
                    ImportFinalKind::Multi(entries) => {
                        // TODO definitely implement line wrapping here, but not the typical "all or nothing" way,
                        //    more soft-wrap like where we keep things as compact as possible

                        self.push(TokenType::OpenS)?;
                        for (entry, last) in entries.iter().with_last() {
                            self.format_import_entry(entry)?;
                            if !last {
                                self.push(TokenType::Comma)?;
                                self.push_space();
                            }
                        }
                        self.push(TokenType::CloseS)?;
                    }
                }
                self.push(TokenType::Semi)?;
                self.push_newline();
                Ok(())
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

    fn format_import_entry(&mut self, entry: &ImportEntry) -> DiagResult<()> {
        // TODO support wrapping?
        let &ImportEntry { span: _, id, as_ } = entry;
        self.format_id(id)?;
        if let Some(ImportAs { span_as, as_id }) = as_ {
            self.push_space();
            self.push(TokenType::As)?;
            self.push_space();
            self.format_maybe_id(as_id)?;
        }
        Ok(())
    }

    fn format_common_declaration<V: FormatVisibility>(&mut self, decl: &CommonDeclaration<V>) -> DiagResult<()> {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                self.format_visibility(vis)?;

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

                        self.push(TokenType::Const)?;
                        self.push_space();
                        self.format_maybe_id(id)?;
                        if let Some(ty) = ty {
                            self.push(TokenType::Colon)?;
                            self.push_space();
                            self.format_expr(ty)?;
                        }
                        self.push_space();
                        self.push(TokenType::Eq)?;
                        self.push_space();
                        self.format_expr(value)?;
                        self.push(TokenType::Semi)?;
                    }
                    CommonDeclarationNamedKind::Struct(_) => todo!(),
                    CommonDeclarationNamedKind::Enum(_) => todo!(),
                    CommonDeclarationNamedKind::Function(_) => todo!(),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }

        Ok(())
    }

    fn format_params(&mut self, params: &Parameters) -> DiagResult<()> {
        todo!()
    }

    fn format_ports(&mut self, ports: &ExtraList<ModulePortItem>) -> DiagResult<()> {
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

    fn format_block<S>(&mut self, block: &Block<S>, f: impl Fn(&mut Self, &S)) -> DiagResult<()> {
        todo!()
    }

    fn format_expr(&mut self, expr: Expression) -> DiagResult<()> {
        match &self.source_expressions[expr.inner] {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Undefined => todo!(),
            ExpressionKind::Type => todo!(),
            ExpressionKind::TypeFunction => todo!(),
            ExpressionKind::Wrapped(_) => todo!(),
            ExpressionKind::Block(_) => todo!(),
            ExpressionKind::Id(_) => todo!(),
            ExpressionKind::IntLiteral(lit) => match *lit {
                IntLiteral::Binary(span) => self.push(TokenType::IntLiteralBinary),
                IntLiteral::Decimal(span) => self.push(TokenType::IntLiteralDecimal),
                IntLiteral::Hexadecimal(span) => self.push(TokenType::IntLiteralHexadecimal),
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

    fn format_visibility<V: FormatVisibility>(&mut self, vis: &V) -> DiagResult<()> {
        V::format_visibility(self, vis)
    }

    fn format_maybe_id(&mut self, id: MaybeIdentifier) -> DiagResult<()> {
        match id {
            MaybeIdentifier::Dummy(span) => self.push_with_span(TokenType::Underscore, span),
            MaybeIdentifier::Identifier(id) => self.format_id(id),
        }
    }

    fn format_id(&mut self, id: Identifier) -> DiagResult<()> {
        let Identifier { span } = id;
        self.push_with_span(TokenType::Identifier, span)
    }
}

trait FormatVisibility {
    fn format_visibility(c: &Context, vis: &Self) -> DiagResult<()>;
}

impl FormatVisibility for Visibility {
    fn format_visibility(c: &Context, vis: &Visibility) -> DiagResult<()> {
        match *vis {
            Visibility::Public(span) => todo!(),
            Visibility::Private => {
                // do nothing, private visibility is the default
                Ok(())
            }
        }
    }
}

impl FormatVisibility for () {
    fn format_visibility(_: &Context, _: &()) -> DiagResult<()> {
        todo!()
    }
}
