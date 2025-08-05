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
    ArenaExpressions, Arg, Args, Block, CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind,
    ConstDeclaration, Expression, ExpressionKind, ExtraList, FileContent, GeneralIdentifier, Identifier, ImportEntry,
    ImportFinalKind, IntLiteral, Item, ItemImport, MaybeIdentifier, ModulePortItem, Parameters, Visibility,
};
use crate::syntax::pos::{LineOffsets, Span};
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

    let mut c = FormatContext {
        source_file: file,
        source_str,
        source_offsets: &source[file].offsets,
        source_tokens: &tokens,
        source_expressions: &ast.arena_expressions,
        settings,
        diags,
        next_source_token: 0,
        result: &mut result,
        curr_line_length: 0,
        event_checkpoint: 0,
        event_restore: 0,
        chars_restore: 0,
    };
    c.format_file(&ast)?;

    // TODO remove
    println!(
        "formatting events: checkpoint={}, restore={}, chars_restore={}",
        c.event_checkpoint, c.event_restore, c.chars_restore
    );

    // TODO re-tokenize, check token equivalence (and maybe eventually AST equivalence?)
    //   actually, the push method already kind of does that, but maybe it's safer to repeat it again

    Ok(result)
}

struct FormatContext<'a> {
    source_file: FileId,
    source_str: &'a str,
    source_offsets: &'a LineOffsets,
    source_tokens: &'a [Token],
    source_expressions: &'a ArenaExpressions,

    settings: &'a FormatSettings,

    diags: &'a Diagnostics,
    next_source_token: usize,
    result: &'a mut String,
    curr_line_length: usize,

    event_checkpoint: u64,
    event_restore: u64,
    chars_restore: u64,
}

// TODO this can cause exponential complexity
#[derive(Debug, Copy, Clone)]
struct CheckPoint {
    next_source_token: usize,
    curr_line_length: usize,
    result_len: usize,
}

impl FormatContext<'_> {
    fn checkpoint(&mut self) -> CheckPoint {
        self.event_checkpoint += 1;
        CheckPoint {
            next_source_token: self.next_source_token,
            curr_line_length: self.curr_line_length,
            result_len: self.result.len(),
        }
    }

    fn restore(&mut self, checkpoint: CheckPoint) {
        assert!(self.result.len() >= checkpoint.result_len);

        self.event_restore += 1;
        self.chars_restore += (self.result.len() - checkpoint.result_len) as u64;

        self.next_source_token = checkpoint.next_source_token;
        self.curr_line_length = checkpoint.curr_line_length;
        self.result.truncate(checkpoint.result_len);
    }

    fn line_overflow(&self) -> bool {
        self.curr_line_length > self.settings.max_line_length
    }

    fn push(&mut self, ty: TokenType) -> DiagResult<()> {
        // TODO improve and double-check comma explanation: (dis)appear, unambiguous
        let diags = self.diags;

        if ty == TokenType::WhiteSpace {
            let curr_span = match self.source_tokens.get(self.next_source_token) {
                Some(token) => Span::empty_at(token.span.start()),
                None => self.source_offsets.end_span(self.source_file),
            };
            return Err(self
                .diags
                .report_internal_error(curr_span, "pushing whitespace tokens is not allowed"));
        }

        let token_str = loop {
            // pop the next source token
            let source_token = self.source_tokens.get(self.next_source_token).ok_or_else(|| {
                let end_span = self.source_offsets.end_span(self.source_file);
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

            let msg = format!("push_token pushing {:?} but source has {:?}", ty, source_token.ty);
            return Err(diags.report_internal_error(source_token.span, msg));
        };

        self.result.push_str(token_str);
        // TODO handle newlines in string literals and comments?
        self.curr_line_length += token_str.len();

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
        self.curr_line_length += 1;
    }

    fn push_newline(&mut self) {
        // no need to check for comments
        // TODO is that right?
        // TODO indentation
        // TODO should we use \r\n on windows?
        self.result.push('\n');
        self.curr_line_length = 0;
    }

    fn push_indent(&mut self) {
        self.result.push_str(&self.settings.indent_str);
        self.curr_line_length += self.settings.indent_str.len();
    }
}

impl FormatContext<'_> {
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
            Item::Import(item) => self.format_import(item),
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

    fn format_import(&mut self, item: &ItemImport) -> DiagResult<()> {
        let ItemImport {
            span: _,
            parents,
            entry,
        } = item;

        // TODO sort imports, both within and across lines, combining them in some single-correct way
        // TODO move imports to top of file?

        self.push(TokenType::Import)?;
        self.push_space();
        for &parent in &parents.inner {
            self.format_id(parent)?;
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

                // try single line
                let check = self.checkpoint();
                for (entry, last) in entries.iter().with_last() {
                    self.format_import_entry(entry)?;
                    if !last {
                        self.push(TokenType::Comma)?;
                        self.push_space();
                    }
                }
                self.push(TokenType::CloseS)?;

                // maybe fallback to multi-line, as many entries as possible per line
                if self.line_overflow() && !entries.is_empty() {
                    self.restore(check);
                    self.push_newline();
                    self.push_indent();

                    let mut first_on_line = true;

                    for entry in entries.iter() {
                        // try to fit the entry on the current line
                        let check_entry = self.checkpoint();
                        if !first_on_line {
                            self.push_space();
                        }
                        self.format_import_entry(entry)?;
                        self.push(TokenType::Comma)?;

                        // maybe overflow to next line
                        if self.line_overflow() {
                            self.restore(check_entry);
                            self.push_newline();
                            self.push_indent();
                            self.format_import_entry(entry)?;
                            self.push(TokenType::Comma)?;
                        }

                        first_on_line = false;
                    }
                    self.push_newline();
                    self.push(TokenType::CloseS)?;
                }
            }
        }
        self.push(TokenType::Semi)?;
        self.push_newline();
        Ok(())
    }

    fn format_import_entry(&mut self, entry: &ImportEntry) -> DiagResult<()> {
        // TODO support wrapping?
        let &ImportEntry { span: _, id, as_ } = entry;
        self.format_id(id)?;
        if let Some(as_) = as_ {
            self.push_space();
            self.push(TokenType::As)?;
            self.push_space();
            self.format_maybe_id(as_)?;
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
                        let &ConstDeclaration { span, id, ty, value } = decl;

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
            &ExpressionKind::Id(id) => self.format_general_id(id),
            ExpressionKind::IntLiteral(lit) => match *lit {
                IntLiteral::Binary(span) => self.push_with_span(TokenType::IntLiteralBinary, span),
                IntLiteral::Decimal(span) => self.push_with_span(TokenType::IntLiteralDecimal, span),
                IntLiteral::Hexadecimal(span) => self.push_with_span(TokenType::IntLiteralHexadecimal, span),
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
            &ExpressionKind::Call(target, ref args) => {
                self.format_expr(target)?;
                self.format_args(args)?;
                Ok(())
            }
            ExpressionKind::Builtin(_) => todo!(),
            ExpressionKind::UnsafeValueWithDomain(_, _) => todo!(),
            ExpressionKind::RegisterDelay(_) => todo!(),
        }
    }

    fn format_args(&mut self, args: &Args) -> DiagResult<()> {
        let Args { span: _, inner } = args;

        self.push(TokenType::OpenR)?;

        // try single line
        // TODO common "comma separated list that wraps to full multiline" formatting function,
        //    maybe even with a compact variant that can be used for imports?
        let check = self.checkpoint();
        for (&arg, last) in inner.iter().with_last() {
            self.format_arg(arg)?;
            if !last {
                self.push(TokenType::Comma)?;
                self.push_space();
            }
        }

        // TODO this should be "line_overflow or any child wrapped",
        //   maybe we can force inner expressions to not wrap, which should be equivalent?
        // maybe fallback to multi-line, one arg per line
        if self.line_overflow() {
            self.restore(check);
            self.push_newline();
            for &arg in inner {
                self.push_indent();
                self.format_arg(arg)?;
                self.push(TokenType::Comma)?;
                self.push_newline();
            }
        }

        self.push(TokenType::CloseR)?;

        Ok(())
    }

    fn format_arg(&mut self, arg: Arg) -> DiagResult<()> {
        let Arg { span: _, name, value } = arg;
        if let Some(name) = name {
            self.format_id(name)?;
            self.push(TokenType::Eq)?;
        }
        self.format_expr(value)?;
        Ok(())
    }

    fn format_visibility<V: FormatVisibility>(&mut self, vis: &V) -> DiagResult<()> {
        V::format_visibility(self, vis)
    }

    fn format_general_id(&mut self, id: GeneralIdentifier) -> DiagResult<()> {
        match id {
            GeneralIdentifier::Simple(id) => self.format_id(id),
            GeneralIdentifier::FromString(_, _) => todo!(),
        }
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
    fn format_visibility(c: &mut FormatContext, vis: &Self) -> DiagResult<()>;
}

impl FormatVisibility for Visibility {
    fn format_visibility(c: &mut FormatContext, vis: &Visibility) -> DiagResult<()> {
        match *vis {
            Visibility::Public(span) => {
                c.push_with_span(TokenType::Public, span)?;
                c.push_space();
                Ok(())
            }
            Visibility::Private => {
                // do nothing, private visibility is the default
                Ok(())
            }
        }
    }
}

impl FormatVisibility for () {
    fn format_visibility(_: &mut FormatContext, _: &()) -> DiagResult<()> {
        // do nothing, no visibility
        Ok(())
    }
}
