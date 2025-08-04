//! Core ideas are based on:
//! * https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/
//! * https://yorickpeterse.com/articles/how-to-write-a-code-formatter/
//!
//! TODO think about comments
//! Comments are handled in a separate final pass.

// TODO should input be a string, an ast, tokens, ...?
//  we need access to the tokens anyway, so maybe just take a string? but then what if it's invalid?

use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::syntax::ast::{
    ArenaExpressions, Block, CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration,
    Expression, ExpressionKind, ExtraList, FileContent, Identifier, ImportAs, ImportEntry, ImportFinalKind, ImportStep,
    IntLiteral, Item, ItemImport, MaybeIdentifier, ModulePortItem, Parameters, Visibility,
};
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{tokenize, FixedTokenType, Token, TokenCategory, TokenType};
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

// TODO split the non-ast specific stuff into separate file/module
impl Context<'_> {
    // TODO merge both push functions into one, they're actually redundant
    fn push_copy(&mut self, span: Span) -> DiagResult<()> {
        assert_eq!(span.file, self.source_file);
        let diags = self.diags;

        loop {
            let source_token = self
                .source_tokens
                .get(self.next_source_token)
                .ok_or_else(|| diags.report_internal_error(span, "pushing copy but no tokens left"))?;
            self.next_source_token += 1;

            match source_token.ty.category() {
                TokenCategory::WhiteSpace => continue,
                TokenCategory::Comment => {
                    // TODO emit comments instead of dropping them
                    continue;
                }
                TokenCategory::Identifier
                | TokenCategory::IntegerLiteral
                | TokenCategory::StringLiteral
                | TokenCategory::Keyword
                | TokenCategory::Symbol => {}
            }

            if source_token.span == span {
                if source_token.ty.as_fixed().is_some() {
                    let msg = format!(
                        "push_copy should only be used for non-fixed tokens, got {:?}",
                        source_token.ty
                    );
                    return Err(diags.report_internal_error(span, msg));
                }
                // matched token, push it
                break;
            }

            // span mismatch
            let diag = Diagnostic::new_internal_error(span, "push_copy failed to find matching span")
                .add_info(source_token.span, format!("curr token has ty {:?}", source_token.ty))
                .finish();
            return Err(diags.report(diag));
        }

        self.result.push_str(&self.source_str[span.range_bytes()]);
        Ok(())
    }

    // TODO fix code duplication
    fn push_fixed(&mut self, ty: FixedTokenType) -> DiagResult<()> {
        // TODO improve and double-check explanation
        // Emit any tokens that were present before the
        // The only token that can (dis)appear is the comma token, at the end of sequences
        //   fortunately consecutive commas are not allowed in the grammar, so there's never any ambiguity.
        // A comma can also never be the last token in the file, so we always expect there to be a next token.
        loop {
            let source_token = self.source_tokens.get(self.next_source_token).ok_or_else(|| {
                let end_pos = Pos {
                    file: self.source_file,
                    byte: self.source_str.len(),
                };
                let end_span = Span::empty_at(end_pos);
                self.diags
                    .report_internal_error(end_span, "pushing token but no tokens left")
            })?;
            self.next_source_token += 1;

            match source_token.ty.category() {
                TokenCategory::WhiteSpace => continue,
                TokenCategory::Comment => {
                    // TODO emit comments instead of dropping them
                    continue;
                }
                TokenCategory::Identifier
                | TokenCategory::IntegerLiteral
                | TokenCategory::StringLiteral
                | TokenCategory::Keyword
                | TokenCategory::Symbol => {}
            }

            if source_token.ty == ty.as_token() {
                break;
            }

            if source_token.ty == TokenType::Comma {
                // comma that disappeared, that's fine
                return Ok(());
            }
            if ty == FixedTokenType::Comma {
                // comma that appeared, that's fine (if we un-pop the source token)
                self.next_source_token -= 1;
                return Ok(());
            }

            let msg = format!("push_token expected {:?} but got {:?}", ty, source_token.ty);
            return Err(self.diags.report_internal_error(source_token.span, msg));
        }

        // skipped past any early tokens and checked for type match, we can safely emit it now
        self.result.push_str(ty.as_str());
        Ok(())
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
                self.push_fixed(FixedTokenType::Import)?;
                self.push_space();
                for parent in &parents.inner {
                    let &ImportStep { id, span_dot } = parent;
                    self.format_id(id)?;
                    self.push_fixed(FixedTokenType::Dot)?;
                }
                match &entry.inner {
                    ImportFinalKind::Single(entry) => {
                        self.format_import_entry(entry)?;
                    }
                    ImportFinalKind::Multi(entries) => {
                        // TODO definitely implement line wrapping here, but not the typical "all or nothing" way,
                        //    more soft-wrap like where we keep things as compact as possible

                        self.push_fixed(FixedTokenType::OpenS)?;
                        for (entry, last) in entries.iter().with_last() {
                            self.format_import_entry(entry)?;
                            if !last {
                                self.push_fixed(FixedTokenType::Comma)?;
                                self.push_space();
                            }
                        }
                        self.push_fixed(FixedTokenType::CloseS)?;
                    }
                }
                self.push_fixed(FixedTokenType::Semi)?;
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
            self.push_fixed(FixedTokenType::As)?;
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

                        self.push_fixed(FixedTokenType::Const)?;
                        self.push_space();
                        self.format_maybe_id(id)?;
                        if let Some(ty) = ty {
                            todo!()
                        }
                        self.push_space();
                        self.push_fixed(FixedTokenType::Eq)?;
                        self.push_space();
                        self.format_expr(value)?;
                        self.push_fixed(FixedTokenType::Semi)?;
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

    fn format_visibility<V: FormatVisibility>(&mut self, vis: &V) -> DiagResult<()> {
        V::format_visibility(self, vis)
    }

    fn format_maybe_id(&mut self, id: MaybeIdentifier) -> DiagResult<()> {
        match id {
            MaybeIdentifier::Dummy(span) => todo!(),
            MaybeIdentifier::Identifier(id) => self.format_id(id),
        }
    }

    fn format_id(&mut self, id: Identifier) -> DiagResult<()> {
        self.push_copy(id.span)
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
