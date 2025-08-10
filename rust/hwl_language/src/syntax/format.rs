use crate::front::diagnostic::{DiagResult, Diagnostics};
// TODO remove star
use crate::syntax::ast::*;
use crate::syntax::pos::{HasSpan, LineOffsets, Span};
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{Token, TokenType as TT, is_whitespace_or_empty, tokenize};
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};
use crate::util::iter::IterExt;
use itertools::{Either, enumerate};

// TODO core:
//   * preserve comments?
//   * what to do about multi-line comments?
//   * record sequence of pushed tokens to check for re-parsing at the end,
//       to double check that no tokens got incorrectly joined
//   * sort and combine imports
//   * maybe switch to some IR anyway, it's becoming a bit hacky

// TODO test
//   * create a test file that contains every syntactical construct, which should fully cover the parser and this
//     * check coverage
//   * full fuzz test, try a bunch of random inputs, if they parse then reformat
//   * fuzz test:
//     * insert block or line comment at every possible location, see if they all survive
//     * also do a variant where every tokens is on a separate line
//   * test idempotency

// TODO interface:
//   * add to python and wasm as a standalone function
//   * add as LSP action
//   * implement proper fmt binary
//     * modes:
//       * format entire project in-place
//       * format single file in-place
//       * check entire project/single file for CI jobs, print diff
//     * if no manifest specified, print warning to stderr and use defaults

// TODO docs/cleanup:
//   * split out the core abstract formatting engine and the ast->IR conversion into separate modules?
//   * cleanup old docs and ideas
//   * write proper docs about what the implementation actually ended up being
//   * core ideas are based on:
//      * https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/
//      * https://yorickpeterse.com/articles/how-to-write-a-code-formatter/
//   * checkpointing can be exponential
//   * improve and double-check comma explanation: (dis)appear, unambiguous

#[derive(Debug)]
pub struct FormatSettings {
    pub indent_str: String,
    // TODO tab size is such a weird setting, it only matters if `indent_str` or literals contain tabs,
    //   do we really want to support this?
    // TODO assert settings validness somewhere, eg. indent_str should parse as a single whitespace token
    pub tab_size: usize,
    pub max_line_length: usize,
    // TODO add option to sort imports, tricky because tokens won't match and we might lose even more comments
    // pub sort_imports: bool,
    // pub newline_str: String,
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

#[derive(Debug)]
#[allow(dead_code)]
pub struct FormatStats {
    pub event_checkpoint: u64,
    pub event_restore: u64,
    pub chars_restore: u64,
    pub result_len: usize,
    pub restore_cost: f32,
}

pub fn format(
    diags: &Diagnostics,
    source: &SourceDatabase,
    file: FileId,
    settings: &FormatSettings,
) -> DiagResult<String> {
    // tokenize and parse
    let source_str = &source[file].content;
    let mut tokens = tokenize(file, source_str, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    // TODO remove whitespace tokens, everyone just filters them out anyway, we can always recover whitespace as "stuff between other tokens" if we really want to
    tokens.retain(|t| t.ty != TT::WhiteSpace);
    let ast = parse_file_content(file, source_str).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;

    // format to string
    let mut result = String::new();
    let mut c = FormatContext {
        settings,
        source_file: file,
        source_str,
        source_offsets: &source[file].offsets,
        source_tokens: &tokens,
        source_expressions: &ast.arena_expressions,
        diags,
        event_checkpoint: 0,
        event_restore: 0,
        chars_restore: 0,
        result: &mut result,
        state: CheckpointState {
            next_source_token: 0,
            curr_line_length: 0,
            overflowing_lines: 0,
            line_comment_count: 0,
            indent: 0,
        },
    };
    c.format_file(&ast)?;

    // collect some debug stats
    let result_len = c.result.len();
    let stats = FormatStats {
        event_checkpoint: c.event_checkpoint,
        event_restore: c.event_restore,
        chars_restore: c.chars_restore,
        result_len,
        restore_cost: c.chars_restore as f32 / result_len as f32,
    };
    let _ = stats;

    Ok(result)
}

struct FormatContext<'a> {
    // settings
    settings: &'a FormatSettings,

    // input source
    source_file: FileId,
    source_str: &'a str,
    source_offsets: &'a LineOffsets,
    source_tokens: &'a [Token],
    source_expressions: &'a ArenaExpressions,

    // diagnostic recording
    diags: &'a Diagnostics,
    event_checkpoint: u64,
    event_restore: u64,
    chars_restore: u64,

    // core state
    result: &'a mut String,
    state: CheckpointState,
}

#[derive(Debug, Copy, Clone)]
struct CheckpointState {
    next_source_token: usize,
    curr_line_length: usize,
    /// The number of lines that have overflowed, not counting the current line.
    overflowing_lines: usize,
    line_comment_count: usize,
    indent: usize,
}

#[derive(Debug, Copy, Clone)]
struct CheckPoint {
    state: CheckpointState,
    result_len: usize,
}

impl FormatContext<'_> {
    fn checkpoint(&mut self) -> CheckPoint {
        self.event_checkpoint += 1;
        CheckPoint {
            result_len: self.result.len(),
            state: self.state,
        }
    }

    fn restore(&mut self, checkpoint: CheckPoint) {
        assert!(self.result.len() >= checkpoint.result_len);

        self.event_restore += 1;
        self.chars_restore += (self.result.len() - checkpoint.result_len) as u64;

        self.result.truncate(checkpoint.result_len);
        self.state = checkpoint.state;
    }

    fn overflow_since(&self, check: CheckPoint) -> bool {
        let curr = self.state;
        curr.overflowing_lines > check.state.overflowing_lines
            || curr.curr_line_length > self.settings.max_line_length
            || curr.line_comment_count > check.state.line_comment_count
    }

    fn anything_since(&self, check: CheckPoint) -> bool {
        self.result.len() > check.result_len
    }

    fn push_raw_str(&mut self, s: &str, indent: bool) {
        // indent if asked and this is the first thing on the line
        if indent && self.state.curr_line_length == 0 {
            for _ in 0..self.state.indent {
                self.result.push_str(&self.settings.indent_str);
                self.state.curr_line_length += self.settings.indent_str.len();
            }
        }

        // push the string itself
        self.result.push_str(s);

        // update newlines and line length
        for c in s.chars() {
            if LineOffsets::LINE_ENDING_CHARS.contains(&c) {
                if self.state.curr_line_length > self.settings.max_line_length {
                    self.state.overflowing_lines += 1;
                }
                self.state.curr_line_length = 0;
            } else {
                // TODO tab size?
                self.state.curr_line_length += 1;
            }
        }
    }

    fn push_comment(&mut self, span: Span, line_comment: bool, next: Option<TT>) {
        // TODO how exactly to format comments probably depends on:
        //   * the current indent level
        //   * the line of the previous token and the comment

        let curr_start = self.source_offsets.expand_pos(span.start());

        // push newline if the comment started on a new line
        if self.state.next_source_token > 0 {
            let prev_token = self.source_tokens[self.state.next_source_token - 1];
            let prev_end = self.source_offsets.expand_pos(prev_token.span.end());
            if curr_start.line_0 > prev_end.line_0 {
                self.push_newline();
            }
        }

        let comment_str = &self.source_str[span.range_bytes()];
        self.push_raw_str(comment_str, true);

        if line_comment {
            self.state.line_comment_count += 1;
            self.push_newline();
        } else {
            // TODO only push newline if next token is on a new line
        }
    }

    fn push(&mut self, ty: TT) -> DiagResult {
        let diags = self.diags;

        if ty == TT::WhiteSpace {
            let curr_span = match self.source_tokens.get(self.state.next_source_token) {
                Some(token) => Span::empty_at(token.span.start()),
                None => self.source_offsets.end_span(self.source_file),
            };
            return Err(self
                .diags
                .report_internal_error(curr_span, "pushing whitespace tokens is not allowed"));
        }

        let token_str = loop {
            // pop the next source token
            let source_token = self.source_tokens.get(self.state.next_source_token).ok_or_else(|| {
                let end_span = self.source_offsets.end_span(self.source_file);
                let msg = format!("pushing token {ty:?} but no tokens left");
                diags.report_internal_error(end_span, msg)
            })?;
            self.state.next_source_token += 1;

            match source_token.ty {
                // drop whitespace, the formatter creates its own
                TT::WhiteSpace => continue,
                // emit comments
                TT::LineComment => {
                    self.push_comment(source_token.span, true, Some(ty));
                    continue;
                }
                TT::BlockComment => {
                    self.push_comment(source_token.span, false, Some(ty));
                    continue;
                }
                // everything else is a real token, fallthrough into matching the type
                _ => {}
            }

            if source_token.ty == ty {
                // found matching token, emit it
                break &self.source_str[source_token.span.range_bytes()];
            }

            if source_token.ty == TT::Comma {
                // comma in original source that disappeared, continue searching for the actually matching token
                continue;
            }
            if ty == TT::Comma {
                // comma that was not in original source but appeared due to formatting,
                //   un-pop the source token and act as if we found a match
                self.state.next_source_token -= 1;
                break ",";
            }

            let msg = format!("push_token pushing {:?} but source has {:?}", ty, source_token.ty);
            return Err(diags.report_internal_error(source_token.span, msg));
        };

        // finally we can push the actual string
        self.push_raw_str(token_str, true);

        Ok(())
    }

    fn push_space(&mut self) {
        // TODO error if this is the first thing on the line
        self.push_raw_str(" ", true);
    }

    fn push_newline(&mut self) {
        self.push_raw_str("\n", false);
    }

    fn preserve_blank_line(&mut self, curr_span: Span, next_span: Span) {
        let curr_end = self.source_offsets.expand_pos(curr_span.end());
        let next_start = self.source_offsets.expand_pos(next_span.start());

        let any_blank_line = (curr_end.line_0..next_start.line_0).any(|line_0| {
            let line_range = self.source_offsets.line_range(line_0, false);
            let line_str = &self.source_str[line_range];
            is_whitespace_or_empty(line_str)
        });
        if any_blank_line {
            self.push_newline();
        }
    }

    fn indent_newline<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.indent_impl(true, f)
    }

    fn indent_no_newline<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.indent_impl(false, f)
    }

    fn indent_impl<R>(&mut self, newline: bool, f: impl FnOnce(&mut Self) -> R) -> R {
        let before = self.state.indent;
        self.state.indent += 1;

        let check_before_line = self.checkpoint();
        if newline {
            self.push_newline();
        }

        let check_before_f = self.checkpoint();
        let result = f(self);
        if !self.anything_since(check_before_f) {
            self.restore(check_before_line);
        }

        assert_eq!(self.state.indent, before + 1,);
        self.state.indent -= 1;

        result
    }
}

impl FormatContext<'_> {
    fn format_file(&mut self, file: &FileContent) -> DiagResult {
        // emit all tokens in the ast
        let FileContent {
            span: _,
            items,
            arena_expressions: _,
        } = file;
        for (i, item) in enumerate(items) {
            self.format_item(item)?;

            if let Some(next) = items.get(i + 1)
                && !(matches!(item, Item::Import(_)) && matches!(next, Item::Import(_)))
            {
                self.preserve_blank_line(item.info().span_full, next.info().span_full);
            }
        }

        // check for leftover tokens and emit comments
        // TODO respect newlines between comments?
        for token in &self.source_tokens[self.state.next_source_token..] {
            match token.ty {
                // ignore whitespace
                TT::WhiteSpace => {}
                // emit comments
                TT::LineComment => self.push_space(),
                TT::BlockComment => todo!(),
                // other tokens are unexpected
                _ => {
                    let err = self
                        .diags
                        .report_internal_error(token.span, "leftover token not printed by ast");
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    fn format_item(&mut self, item: &Item) -> DiagResult {
        match item {
            Item::Import(item) => self.format_import(item)?,
            Item::CommonDeclaration(decl) => self.format_common_declaration(&decl.inner)?,
            Item::ModuleInternal(decl) => {
                let &ItemDefModuleInternal {
                    span: _,
                    vis,
                    id,
                    ref params,
                    ref ports,
                    ref body,
                } = decl;
                self.format_module(vis, false, id, params.as_ref(), &ports.inner, Some(body))?;
            }
            Item::ModuleExternal(decl) => {
                let &ItemDefModuleExternal {
                    span: _,
                    vis,
                    span_ext: _,
                    id,
                    ref params,
                    ref ports,
                } = decl;
                self.format_module(
                    vis,
                    true,
                    MaybeIdentifier::Identifier(id),
                    params.as_ref(),
                    &ports.inner,
                    None,
                )?;
            }
            Item::Interface(decl) => self.format_interface(decl)?,
        }
        Ok(())
    }

    fn format_module(
        &mut self,
        vis: Visibility,
        external: bool,
        id: MaybeIdentifier,
        params: Option<&Parameters>,
        ports: &ExtraList<ModulePortItem>,
        body: Option<&Block<ModuleStatement>>,
    ) -> DiagResult {
        self.format_visibility(&vis)?;
        if external {
            self.push(TT::External)?;
            self.push_space();
        }
        self.push(TT::Module)?;
        self.push_space();
        self.format_maybe_id(id)?;
        // TODO test line wrapping here
        if let Some(params) = params {
            self.format_params(params)?;
        }
        self.push_space();
        self.push(TT::Ports)?;
        self.push(TT::OpenR)?;
        // TODO maybe allow single-line ports?
        self.format_extra_list_always_wrap(ports, &Self::format_port_item)?;
        self.push(TT::CloseR)?;
        if let Some(body) = body {
            self.push_space();
            self.format_block_general(body, Self::format_module_statement)?;
        }
        self.push_newline();
        Ok(())
    }

    fn format_interface(&mut self, interface: &ItemDefInterface) -> DiagResult {
        let &ItemDefInterface {
            span: _,
            vis,
            id,
            ref params,
            span_body: _,
            ref port_types,
            ref views,
        } = interface;

        self.format_visibility(&vis)?;
        self.push(TT::Interface)?;
        self.push_space();
        self.format_maybe_id(id)?;
        if let Some(params) = params {
            self.format_params(params)?;
        }
        self.push_space();
        self.push(TT::OpenC)?;

        if !port_types.items.is_empty() {
            self.format_extra_list_always_wrap(port_types, &|slf, port_type: &(Identifier, Expression)| {
                let (id, expr) = port_type;
                slf.format_id(*id)?;
                slf.push(TT::Colon)?;
                slf.push_space();
                slf.format_expr(*expr, true)?;
                Ok(())
            })?;
        }

        self.indent_newline(|slf| {
            for (i, view) in views.iter().enumerate() {
                let InterfaceView { span: _, id, port_dirs } = view;

                slf.push(TT::Interface)?;
                slf.push_space();
                slf.format_maybe_id(*id)?;
                slf.push_space();
                slf.push(TT::OpenC)?;
                slf.format_extra_list_always_wrap(port_dirs, &|inner_slf, port_dir| {
                    let (id, direction) = port_dir;
                    inner_slf.format_id(*id)?;
                    inner_slf.push(TT::Colon)?;
                    inner_slf.push_space();
                    inner_slf.push(direction.inner.token())?;
                    Ok(())
                })?;
                slf.push(TT::CloseC)?;
                slf.push_newline();

                if let Some(next_view) = views.get(i + 1) {
                    slf.preserve_blank_line(view.span, next_view.span);
                }
            }
            Ok(())
        })?;

        self.push(TT::CloseC)?;
        self.push_newline();
        Ok(())
    }

    fn format_module_statement(&mut self, stmt: &ModuleStatement) -> DiagResult {
        match &stmt.inner {
            ModuleStatementKind::Block(block) => {
                self.format_block_general(block, Self::format_module_statement)?;
                self.push_newline();
            }
            ModuleStatementKind::If(if_) => {
                self.format_if(if_, |slf, block| {
                    slf.format_block_general(block, Self::format_module_statement)
                })?;
            }
            ModuleStatementKind::For(for_) => {
                self.format_for_statement(for_, |slf, block| {
                    slf.format_block_general(block, Self::format_module_statement)
                })?;
            }
            ModuleStatementKind::CommonDeclaration(decl) => {
                self.format_common_declaration(decl)?;
            }
            ModuleStatementKind::RegDeclaration(decl) => {
                let &RegDeclaration {
                    vis,
                    id,
                    ref sync,
                    ty,
                    init,
                } = decl;
                self.format_visibility(&vis)?;
                self.push(TT::Reg)?;
                self.push_space();
                self.format_maybe_general_id(id)?;
                self.push(TT::Colon)?;
                self.push_space();
                if let Some(sync) = sync {
                    self.format_domain(DomainKind::Sync(sync.inner), true)?;
                    self.push_space();
                }
                self.format_expr(ty, true)?;
                self.push_space();
                self.push(TT::Eq)?;
                self.push_space();
                self.format_expr(init, true)?;
                self.push(TT::Semi)?;
                self.push_newline();
            }
            ModuleStatementKind::WireDeclaration(decl) => {
                let &WireDeclaration {
                    vis,
                    span_keyword: _,
                    id,
                    ref kind,
                } = decl;
                self.format_visibility(&vis)?;
                self.push(TT::Wire)?;
                self.push_space();
                self.format_maybe_general_id(id)?;
                match kind {
                    WireDeclarationKind::Normal {
                        domain_ty,
                        assign_span_and_value,
                    } => {
                        match domain_ty {
                            WireDeclarationDomainTyKind::Clock { span_clock: _ } => {
                                self.push(TT::Colon)?;
                                self.push_space();
                                self.push(TT::Clock)?;
                            }
                            WireDeclarationDomainTyKind::Normal { domain, ty } => {
                                if let Some(domain) = domain {
                                    self.push(TT::Colon)?;
                                    self.push_space();
                                    self.format_domain(domain.inner, true)?;
                                }
                                if let Some(ty) = ty {
                                    self.push_space();
                                    self.format_expr(*ty, true)?;
                                }
                            }
                        }
                        if let Some((_, value)) = assign_span_and_value {
                            self.push_space();
                            self.push(TT::Eq)?;
                            self.push_space();
                            self.format_expr(*value, true)?;
                        }
                    }
                    WireDeclarationKind::Interface {
                        domain,
                        span_keyword: _,
                        interface,
                    } => {
                        self.push(TT::Colon)?;
                        if let Some(domain) = domain {
                            self.push_space();
                            self.format_domain(domain.inner, true)?;
                        }
                        self.push_space();
                        self.push(TT::Interface)?;
                        self.push_space();
                        self.format_expr(*interface, true)?;
                    }
                }
                self.push(TT::Semi)?;
                self.push_newline();
            }
            ModuleStatementKind::RegOutPortMarker(marker) => {
                let &RegOutPortMarker { id, init } = marker;
                self.push(TT::Reg)?;
                self.push_space();
                self.push(TT::Out)?;
                self.push_space();
                self.format_id(id)?;
                self.push_space();
                self.push(TT::Eq)?;
                self.push_space();
                self.format_expr(init, true)?;
                self.push(TT::Semi)?;
                self.push_newline();
            }
            ModuleStatementKind::CombinatorialBlock(block) => {
                let CombinatorialBlock { span_keyword: _, block } = block;
                self.push(TT::Combinatorial)?;
                self.push_space();
                self.format_block(block)?;
            }
            ModuleStatementKind::ClockedBlock(block) => {
                let &ClockedBlock {
                    span_keyword: _,
                    span_domain: _,
                    clock,
                    ref reset,
                    ref block,
                } = block;
                self.push(TT::Clocked)?;
                self.push(TT::OpenR)?;
                self.format_expr(clock, true)?;
                if let Some(reset) = reset {
                    self.push(TT::Comma)?;
                    self.push_space();
                    self.push(reset.inner.kind.inner.token())?;
                    self.push_space();
                    self.format_expr(reset.inner.signal, true)?;
                }
                self.push(TT::CloseR)?;
                self.push_space();
                self.format_block(block)?;
            }
            ModuleStatementKind::Instance(instance) => {
                let &ModuleInstance {
                    name,
                    span_keyword: _,
                    module,
                    ref port_connections,
                } = instance;
                if let Some(name) = name {
                    self.format_id(name)?;
                    self.push(TT::Colon)?;
                    self.push_space();
                }
                self.push(TT::Instance)?;
                self.push_space();
                self.format_expr(module, true)?;
                self.push_space();
                self.push(TT::Ports)?;
                self.push(TT::OpenR)?;

                if !port_connections.inner.is_empty() {
                    let check = self.checkpoint();
                    for (conn, last) in port_connections.inner.iter().with_last() {
                        let &PortConnection { id, expr } = &conn.inner;
                        self.format_id(id)?;
                        self.push(TT::Eq)?;
                        self.format_expr(expr, false)?;
                        if !last {
                            self.push(TT::Comma)?;
                            self.push_space();
                        }
                    }

                    if self.overflow_since(check) {
                        self.restore(check);
                        self.indent_newline(|slf| {
                            for conn in &port_connections.inner {
                                let &PortConnection { id, expr } = &conn.inner;
                                slf.format_id(id)?;
                                slf.push(TT::Eq)?;
                                slf.format_expr(expr, true)?;
                                slf.push(TT::Comma)?;
                                slf.push_newline();
                            }
                            Ok(())
                        })?;
                    }
                }

                self.push(TT::CloseR)?;
                self.push(TT::Semi)?;
                self.push_newline();
            }
        }
        Ok(())
    }

    fn format_import(&mut self, item: &ItemImport) -> DiagResult {
        let ItemImport {
            span: _,
            parents,
            entry,
        } = item;

        self.push(TT::Import)?;
        self.push_space();
        for &parent in &parents.inner {
            self.format_id(parent)?;
            self.push(TT::Dot)?;
        }
        match &entry.inner {
            ImportFinalKind::Single(entry) => {
                self.format_import_entry(entry)?;
            }
            ImportFinalKind::Multi(entries) => {
                self.push(TT::OpenS)?;

                // try single line
                let check = self.checkpoint();
                for (entry, last) in entries.iter().with_last() {
                    self.format_import_entry(entry)?;
                    if !last {
                        self.push(TT::Comma)?;
                        self.push_space();
                    }
                }
                self.push(TT::CloseS)?;

                // maybe fallback to multi-line, as many entries as possible per line
                if self.overflow_since(check) && !entries.is_empty() {
                    self.restore(check);

                    self.indent_newline(|slf| {
                        let mut first_on_line = true;
                        for entry in entries.iter() {
                            // try to fit the entry on the current line
                            let check_entry = slf.checkpoint();
                            if !first_on_line {
                                slf.push_space();
                            }
                            slf.format_import_entry(entry)?;
                            slf.push(TT::Comma)?;

                            // maybe overflow to next line
                            if slf.overflow_since(check_entry) {
                                slf.restore(check_entry);
                                slf.push_newline();
                                slf.format_import_entry(entry)?;
                                slf.push(TT::Comma)?;
                            }

                            first_on_line = false;
                        }

                        Ok(())
                    })?;

                    self.push_newline();
                    self.push(TT::CloseS)?;
                }
            }
        }
        self.push(TT::Semi)?;
        self.push_newline();
        Ok(())
    }

    fn format_import_entry(&mut self, entry: &ImportEntry) -> DiagResult {
        let &ImportEntry { span: _, id, as_ } = entry;
        self.format_id(id)?;
        if let Some(as_) = as_ {
            self.push_space();
            self.push(TT::As)?;
            self.push_space();
            self.format_maybe_id(as_)?;
        }
        Ok(())
    }

    fn format_common_declaration<V: FormatVisibility>(&mut self, decl: &CommonDeclaration<V>) -> DiagResult {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                self.format_visibility(vis)?;

                match kind {
                    CommonDeclarationNamedKind::Type(decl) => {
                        let &TypeDeclaration {
                            span: _,
                            id,
                            ref params,
                            body,
                        } = decl;
                        self.format_var_declaration(TT::Type, id, params.as_ref(), None, Some(body))?;
                    }
                    CommonDeclarationNamedKind::Const(decl) => {
                        let &ConstDeclaration { span: _, id, ty, value } = decl;
                        self.format_var_declaration(TT::Const, id, None, ty, Some(value))?;
                    }
                    CommonDeclarationNamedKind::Struct(struct_decl) => {
                        let &StructDeclaration {
                            span: _,
                            span_body: _,
                            id,
                            ref params,
                            ref fields,
                        } = struct_decl;
                        self.push(TT::Struct)?;
                        self.push_space();
                        self.format_maybe_id(id)?;
                        if let Some(params) = params {
                            self.format_params(params)?;
                        }
                        self.push_space();
                        self.push(TT::OpenC)?;

                        self.format_extra_list_always_wrap(fields, &|slf, field: &StructField| {
                            let &StructField { span: _, id, ty } = field;
                            slf.format_id(id)?;
                            slf.push(TT::Colon)?;
                            slf.push_space();
                            slf.format_expr(ty, true)?;
                            Ok(())
                        })?;

                        self.push(TT::CloseC)?;
                        self.push_newline();
                    }
                    CommonDeclarationNamedKind::Enum(enum_decl) => {
                        let &EnumDeclaration {
                            span: _,
                            id,
                            ref params,
                            ref variants,
                        } = enum_decl;
                        self.push(TT::Enum)?;
                        self.push_space();
                        self.format_maybe_id(id)?;
                        if let Some(params) = params {
                            self.format_params(params)?;
                        }
                        self.push_space();
                        self.push(TT::OpenC)?;

                        self.format_extra_list_always_wrap(variants, &|slf, variant: &EnumVariant| {
                            let &EnumVariant { span: _, id, content } = variant;
                            slf.format_id(id)?;
                            if let Some(content) = content {
                                slf.push(TT::OpenR)?;
                                slf.format_expr(content, true)?;
                                slf.push(TT::CloseR)?;
                            }
                            Ok(())
                        })?;

                        self.push(TT::CloseC)?;
                        self.push_newline();
                    }
                    CommonDeclarationNamedKind::Function(func_decl) => {
                        let &FunctionDeclaration {
                            span: _,
                            id,
                            ref params,
                            ret_ty,
                            ref body,
                        } = func_decl;
                        self.push(TT::Function)?;
                        self.push_space();
                        self.format_maybe_id(id)?;
                        self.format_params(params)?;

                        if let Some(ret_ty) = ret_ty {
                            self.push_space();
                            self.push(TT::Arrow)?;
                            self.push_space();
                            self.format_expr(ret_ty, true)?;
                        }

                        self.push_space();
                        self.format_block(body)?;
                    }
                }
            }
            CommonDeclaration::ConstBlock(block) => {
                let ConstBlock { span_keyword: _, block } = block;
                self.push(TT::Const)?;
                self.push_space();
                self.format_block(block)?;
                self.push_newline();
            }
        }

        Ok(())
    }

    fn format_var_declaration(
        &mut self,
        kind: TT,
        id: MaybeIdentifier,
        params: Option<&Parameters>,
        ty: Option<Expression>,
        value: Option<Expression>,
    ) -> DiagResult {
        // TODO this can be improved, there's still some weird wrapping behavior

        // format kind, id, params
        self.push(kind)?;
        self.push_space();
        self.format_maybe_id(id)?;
        if let Some(params) = params {
            self.format_params(params)?;
        }

        // try single line
        let check_before_ty = self.checkpoint();
        if let Some(ty) = ty {
            self.push(TT::Colon)?;
            self.push_space();
            self.format_expr(ty, false)?;
        }
        let check_before_value = if let Some(value) = value {
            self.push_space();
            self.push(TT::Eq)?;
            self.push_space();
            let check_before_value = self.checkpoint();
            self.format_expr(value, false)?;
            check_before_value
        } else {
            self.checkpoint()
        };
        self.push(TT::Semi)?;
        self.push_newline();
        if !self.overflow_since(check_before_ty) {
            return Ok(());
        }

        // fallback, wrap value
        if let Some(value) = value {
            self.restore(check_before_value);
            self.format_expr(value, true)?;
            self.push(TT::Semi)?;
            self.push_newline();
        }
        if !self.overflow_since(check_before_ty) {
            return Ok(());
        }

        // fallback, type and value on new line
        self.restore(check_before_ty);
        if let Some(ty) = ty {
            self.indent_newline(|slf| {
                slf.push(TT::Colon)?;
                slf.push_space();
                slf.format_expr(ty, true)?;
                Ok(())
            })?;
        }
        if let Some(value) = value {
            self.indent_newline(|slf| {
                slf.push(TT::Eq)?;
                slf.push_space();
                slf.format_expr(value, true)?;
                Ok(())
            })?;
        }
        self.push(TT::Semi)?;
        self.push_newline();
        Ok(())
    }

    fn format_params(&mut self, params: &Parameters) -> DiagResult {
        let Parameters { span: _, items } = params;
        self.push(TT::OpenR)?;
        self.format_extra_list_maybe_wrap(items, &Self::format_param)?;
        self.push(TT::CloseR)?;
        Ok(())
    }

    fn format_extra_list_maybe_wrap<T: HasSpan>(
        &mut self,
        extra_list: &ExtraList<T>,
        f: &impl Fn(&mut Self, &T, bool) -> DiagResult,
    ) -> DiagResult {
        self.format_extra_list_impl(extra_list, true, f)
    }

    fn format_extra_list_always_wrap<T: HasSpan>(
        &mut self,
        extra_list: &ExtraList<T>,
        f: &impl Fn(&mut Self, &T) -> DiagResult,
    ) -> DiagResult {
        self.format_extra_list_impl(
            extra_list,
            false,
            &(|slf, item, allow_wrap| {
                assert!(allow_wrap);
                f(slf, item)
            }),
        )
    }

    fn format_extra_list_impl<T: HasSpan>(
        &mut self,
        extra_list: &ExtraList<T>,
        try_single_line: bool,
        f: &impl Fn(&mut Self, &T, bool) -> DiagResult,
    ) -> DiagResult {
        let ExtraList { span: _, items } = extra_list;

        // prevent spurious empty newlines
        if items.is_empty() {
            return Ok(());
        }

        // maybe try a single line
        if try_single_line && items.iter().all(|item| matches!(item, ExtraItem::Inner(_))) {
            let check = self.checkpoint();

            for (item, last) in items.iter().with_last() {
                let item = match item {
                    ExtraItem::Inner(item) => item,
                    _ => unreachable!(),
                };
                f(self, item, false)?;
                if !last {
                    self.push(TT::Comma)?;
                    self.push_space();
                }
            }

            if !self.overflow_since(check) || items.is_empty() {
                return Ok(());
            }
            self.restore(check);
        }

        // multiple lines, one item per line
        self.indent_newline(|slf| {
            for (i, item) in enumerate(items) {
                match item {
                    ExtraItem::Inner(param) => {
                        f(slf, param, true)?;
                        slf.push(TT::Comma)?;
                        slf.push_newline();
                    }
                    ExtraItem::Declaration(decl) => {
                        slf.format_common_declaration(decl)?;
                    }
                    ExtraItem::If(if_) => {
                        slf.format_if(if_, |slf, b| {
                            slf.push(TT::OpenC)?;
                            slf.format_extra_list_impl(b, false, f)?;
                            slf.push(TT::CloseC)?;
                            Ok(())
                        })?;
                    }
                }

                if let Some(next) = items.get(i + 1) {
                    slf.preserve_blank_line(item.span(), next.span());
                }
            }
            Ok(())
        })
    }

    fn format_param(&mut self, param: &Parameter, allow_wrap: bool) -> DiagResult {
        let &Parameter {
            span: _,
            id,
            ty,
            default,
        } = param;
        self.format_id(id)?;
        self.push(TT::Colon)?;
        self.push_space();
        self.format_expr(ty, allow_wrap)?;
        if let Some(default) = default {
            self.push_space();
            self.push(TT::Eq)?;
            self.push_space();
            self.format_expr(default, allow_wrap)?;
        }
        Ok(())
    }

    fn format_if<B>(&mut self, if_: &IfStatement<B>, f: impl Fn(&mut Self, &B) -> DiagResult) -> DiagResult {
        let format_pair = |slf: &mut Self, pair: &IfCondBlockPair<B>| {
            let &IfCondBlockPair {
                span: _,
                span_if: _,
                cond,
                ref block,
            } = pair;
            slf.push(TT::If)?;
            slf.push_space();
            slf.format_condition(cond)?;
            slf.push_space();
            f(slf, block)?;
            slf.push_newline();
            Ok(())
        };

        let IfStatement {
            span: _,
            initial_if,
            else_ifs,
            final_else,
        } = if_;
        format_pair(self, initial_if)?;
        for else_if in else_ifs {
            self.push(TT::Else)?;
            self.push_space();
            format_pair(self, else_if)?;
        }
        if let Some(block) = final_else {
            self.push(TT::Else)?;
            self.push_space();
            f(self, block)?;
            self.push_newline();
        }
        Ok(())
    }

    fn format_condition(&mut self, cond: Expression) -> DiagResult {
        self.push(TT::OpenR)?;

        // try single line
        let check = self.checkpoint();
        self.format_expr(cond, false)?;
        self.push(TT::CloseR)?;

        // (simulate the following curly to check the final line length)
        let check_dummy = self.checkpoint();
        self.push_space();
        self.push(TT::OpenC)?;
        let overflow = self.overflow_since(check);
        self.restore(check_dummy);

        // maybe fallback to multi-line
        if overflow {
            self.restore(check);
            self.indent_newline(|slf| slf.format_expr(cond, true))?;
            self.push_newline();
            self.push(TT::CloseR)?;
        }

        Ok(())
    }

    fn format_for_statement<B>(
        &mut self,
        for_: &ForStatement<B>,
        f: impl Fn(&mut Self, &Block<B>) -> DiagResult,
    ) -> DiagResult {
        let &ForStatement {
            span_keyword: _,
            index,
            index_ty,
            iter,
            ref body,
        } = for_;
        self.push(TT::For)?;
        self.push_space();
        self.push(TT::OpenR)?;

        // try single line
        let check = self.checkpoint();
        self.format_maybe_id(index)?;
        if let Some(ty) = index_ty {
            self.push(TT::Colon)?;
            self.push_space();
            self.format_expr(ty, false)?;
        }
        self.push_space();
        self.push(TT::In)?;
        self.push_space();
        self.format_expr(iter, false)?;
        self.push(TT::CloseR)?;
        self.push_space();

        let check_dummy = self.checkpoint();
        self.push(TT::OpenC)?;
        let overflow = self.overflow_since(check);
        self.restore(check_dummy);

        // maybe fallback to multi-line
        if overflow {
            self.restore(check);
            self.indent_newline(|slf| {
                slf.format_maybe_id(index)?;
                if let Some(ty) = index_ty {
                    slf.push(TT::Colon)?;
                    slf.push_space();
                    slf.format_expr(ty, true)?;
                }
                slf.push_newline();
                slf.push(TT::In)?;
                slf.push_space();
                slf.format_expr(iter, true)?;
                slf.push(TT::CloseR)?;
                slf.push_space();
                Ok(())
            })?;
        }

        f(self, body)?;
        self.push_newline();
        Ok(())
    }

    fn format_match<B>(&mut self, match_: &MatchStatement<B>, f: impl Fn(&mut Self, &B) -> DiagResult) -> DiagResult {
        let &MatchStatement {
            target,
            span_branches: _,
            ref branches,
        } = match_;
        self.push(TT::Match)?;
        self.push_space();
        self.format_condition(target)?;
        self.push_space();
        self.push(TT::OpenC)?;

        self.indent_newline(|slf| {
            for branch in branches {
                let MatchBranch { pattern, block } = branch;
                match &pattern.inner {
                    MatchPattern::Wildcard => slf.push(TT::Underscore)?,
                    &MatchPattern::Equal(expr) => slf.format_expr(expr, true)?,
                    &MatchPattern::Val(id) => {
                        slf.push(TT::Identifier)?;
                        slf.format_id(id)?;
                    }
                    &MatchPattern::In(expr) => {
                        slf.push(TT::In)?;
                        slf.format_expr(expr, true)?;
                    }
                    &MatchPattern::EnumVariant(variant, data) => {
                        slf.push(TT::Dot)?;
                        slf.format_id(variant)?;
                        if let Some(data) = data {
                            slf.push(TT::OpenR)?;
                            slf.format_maybe_id(data)?;
                            slf.push(TT::CloseR)?;
                        }
                    }
                }
                slf.push_space();
                slf.push(TT::DoubleArrow)?;
                slf.push_space();
                f(slf, block)?;
                slf.push_newline();
            }
            Ok(())
        })?;
        self.push(TT::CloseC)?;
        self.push_newline();
        Ok(())
    }

    fn format_port_item(&mut self, port: &ModulePortItem) -> DiagResult {
        match port {
            ModulePortItem::Single(port) => {
                let &ModulePortSingle { span: _, id, ref kind } = port;
                self.format_id(id)?;
                self.push(TT::Colon)?;
                self.push_space();
                match kind {
                    ModulePortSingleKind::Port { direction, kind } => {
                        self.push(direction.inner.token())?;
                        self.push_space();
                        match kind {
                            PortSingleKindInner::Clock { span_clock: _ } => self.push(TT::Clock)?,
                            &PortSingleKindInner::Normal { domain, ty } => {
                                self.format_domain(domain.inner, true)?;
                                self.push_space();
                                self.format_expr(ty, true)?;
                            }
                        }
                    }
                    &ModulePortSingleKind::Interface {
                        span_keyword: _,
                        domain,
                        interface,
                    } => {
                        self.push(TT::Interface)?;
                        self.push_space();
                        self.format_domain(domain.inner, true)?;
                        self.push_space();
                        self.format_expr(interface, true)?;
                    }
                }
            }
            ModulePortItem::Block(port_block) => {
                let ModulePortBlock { span: _, domain, ports } = port_block;
                self.format_domain(domain.inner, true)?;
                self.push_space();
                self.push(TT::OpenC)?;
                self.format_extra_list_always_wrap(ports, &Self::format_port_in_block)?;
                self.push(TT::CloseC)?;
            }
        }

        Ok(())
    }

    fn format_port_in_block(&mut self, port: &ModulePortInBlock) -> DiagResult {
        let &ModulePortInBlock { span: _, id, ref kind } = port;
        self.format_id(id)?;
        self.push(TT::Colon)?;
        self.push_space();
        match *kind {
            ModulePortInBlockKind::Port { direction, ty } => {
                self.push(direction.inner.token())?;
                self.push_space();
                self.format_expr(ty, true)?;
            }
            ModulePortInBlockKind::Interface {
                span_keyword: _,
                interface,
            } => {
                self.push(TT::Interface)?;
                self.push_space();
                self.format_expr(interface, true)?;
            }
        }
        Ok(())
    }

    fn format_domain(&mut self, domain: DomainKind<Expression>, allow_wrap: bool) -> DiagResult {
        match domain {
            DomainKind::Const => self.push(TT::Const)?,
            DomainKind::Async => self.push(TT::Async)?,
            DomainKind::Sync(SyncDomain { clock, reset }) => {
                self.push(TT::Sync)?;
                self.push(TT::OpenR)?;

                // try single line
                let check = self.checkpoint();
                self.format_expr(clock, false)?;
                if let Some(reset) = reset {
                    self.push(TT::Comma)?;
                    self.push_space();
                    self.format_expr(reset, false)?;
                    self.push(TT::CloseR)?;

                    // maybe fallback to multi-line (only if there's a reset, otherwise we're not saving much)
                    if allow_wrap && self.overflow_since(check) {
                        self.restore(check);
                        self.indent_newline(|slf| {
                            slf.format_expr(clock, true)?;
                            slf.push(TT::Comma)?;
                            slf.push_newline();
                            slf.format_expr(reset, true)?;
                            slf.push_newline();
                            Ok(())
                        })?;
                        self.push(TT::CloseR)?;
                    }
                } else {
                    self.push(TT::CloseR)?;
                }
            }
        }
        Ok(())
    }

    fn format_block(&mut self, block: &Block<BlockStatement>) -> DiagResult {
        self.format_block_general(block, Self::format_block_statement)
    }

    fn format_block_general<T: HasSpan>(
        &mut self,
        block: &Block<T>,
        f: impl Fn(&mut Self, &T) -> DiagResult,
    ) -> DiagResult {
        let Block { span: _, statements } = block;
        self.push(TT::OpenC)?;
        self.indent_newline(|slf| slf.format_block_statements_general(statements, f))?;
        self.push(TT::CloseC)?;
        Ok(())
    }

    fn format_block_statements_general<T: HasSpan>(
        &mut self,
        statements: &[T],
        f: impl Fn(&mut Self, &T) -> DiagResult,
    ) -> DiagResult {
        for (i, stmt) in enumerate(statements) {
            f(self, stmt)?;
            if let Some(next) = statements.get(i + 1) {
                self.preserve_blank_line(stmt.span(), next.span());
            }
        }
        Ok(())
    }

    fn format_block_statement(&mut self, stmt: &BlockStatement) -> DiagResult {
        match &stmt.inner {
            BlockStatementKind::CommonDeclaration(decl) => self.format_common_declaration(decl)?,
            BlockStatementKind::VariableDeclaration(decl) => {
                let &VariableDeclaration {
                    span: _,
                    mutable,
                    id,
                    ty,
                    init,
                } = decl;
                let kind = if mutable { TT::Var } else { TT::Val };
                self.format_var_declaration(kind, id, None, ty, init)?;
            }
            BlockStatementKind::Assignment(stmt) => {
                let &Assignment {
                    span: _,
                    op,
                    target,
                    value,
                } = stmt;
                let op_token = op.inner.map_or(TT::Eq, AssignBinaryOp::token);

                self.format_expr(target, false)?;

                // try single line
                let check = self.checkpoint();
                self.push_space();
                self.push(op_token)?;
                self.push_space();
                self.format_expr(value, true)?;
                self.push(TT::Semi)?;
                self.push_newline();

                // maybe fallback to multi-line
                if self.overflow_since(check) {
                    self.restore(check);
                    self.indent_newline(|slf| {
                        slf.push(op_token)?;
                        slf.push_space();
                        slf.format_expr(value, true)?;
                        slf.push(TT::Semi)?;
                        slf.push_newline();
                        Ok(())
                    })?;
                }
            }
            &BlockStatementKind::Expression(expr) => {
                self.format_expr(expr, true)?;
                self.push(TT::Semi)?;
                self.push_newline();
            }
            BlockStatementKind::Block(block) => {
                self.format_block(block)?;
            }
            BlockStatementKind::If(if_) => {
                self.format_if(if_, Self::format_block)?;
            }
            BlockStatementKind::Match(match_) => {
                self.format_match(match_, Self::format_block)?;
            }
            BlockStatementKind::For(for_) => {
                self.format_for_statement(for_, Self::format_block)?;
            }
            BlockStatementKind::While(while_) => {
                let &WhileStatement {
                    span_keyword: _,
                    cond,
                    ref body,
                } = while_;
                self.push(TT::While)?;
                self.push_space();
                self.push(TT::OpenR)?;

                // try single line
                let check = self.checkpoint();
                self.format_expr(cond, false)?;
                self.push(TT::CloseR)?;
                self.push_space();

                // maybe fallback to multi-line
                // TODO if body is empty, the closing curly should be included in the overflow check
                if self.overflow_since(check) {
                    self.restore(check);
                    self.indent_newline(|slf| {
                        slf.format_expr(cond, true)?;
                        slf.push_newline();
                        Ok(())
                    })?;
                    self.push(TT::CloseR)?;
                    self.push_space();
                }

                self.format_block(body)?;
                self.push_newline();
            }
            BlockStatementKind::Return(stmt) => {
                let &ReturnStatement { span_return: _, value } = stmt;
                self.push(TT::Return)?;
                if let Some(value) = value {
                    self.push_space();
                    self.format_expr(value, true)?;
                }
                self.push(TT::Semi)?;
                self.push_newline();
            }
            &BlockStatementKind::Break(_span) => {
                self.push(TT::Break)?;
                self.push(TT::Semi)?;
                self.push_newline();
            }
            &BlockStatementKind::Continue(_span) => {
                self.push(TT::Continue)?;
                self.push(TT::Semi)?;
                self.push_newline();
            }
        }
        Ok(())
    }

    fn format_expr(&mut self, expr: Expression, allow_wrap: bool) -> DiagResult {
        match &self.source_expressions[expr.inner] {
            ExpressionKind::Dummy => self.push(TT::Underscore)?,
            ExpressionKind::Undefined => self.push(TT::Undefined)?,
            ExpressionKind::Type => self.push(TT::Type)?,
            ExpressionKind::TypeFunction => self.push(TT::Function)?,
            ExpressionKind::Wrapped(inner) => {
                self.push(TT::OpenR)?;
                self.format_expr(*inner, allow_wrap)?;
                self.push(TT::CloseR)?;
            }
            ExpressionKind::Block(block) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = block;
                self.push(TT::OpenC)?;
                self.indent_newline(|slf| {
                    slf.format_block_statements_general(statements, Self::format_block_statement)?;
                    slf.format_expr(expression, true)?;
                    slf.push_newline();
                    Ok(())
                })?;
                self.push(TT::CloseC)?;
            }
            &ExpressionKind::Id(id) => self.format_general_id(id)?,
            ExpressionKind::IntLiteral(lit) => match *lit {
                IntLiteral::Binary(_span) => self.push(TT::IntLiteralBinary)?,
                IntLiteral::Decimal(_span) => self.push(TT::IntLiteralDecimal)?,
                IntLiteral::Hexadecimal(_span) => self.push(TT::IntLiteralHexadecimal)?,
            },
            &ExpressionKind::BoolLiteral(bool) => match bool {
                false => self.push(TT::False)?,
                true => self.push(TT::True)?,
            },
            ExpressionKind::StringLiteral(pieces) => {
                self.push(TT::StringStart)?;

                // try single line
                let check = self.checkpoint();
                let mut any_sub = false;
                for piece in pieces {
                    match piece {
                        StringPiece::Literal(_span) => self.push(TT::StringMiddle)?,
                        &StringPiece::Substitute(expr) => {
                            any_sub = true;
                            self.push(TT::StringSubStart)?;
                            self.format_expr(expr, false)?;
                            self.push(TT::StringSubEnd)?;
                        }
                    }
                }
                self.push(TT::StringEnd)?;

                // maybe fallback to multi-line, one sub per line
                if allow_wrap && any_sub && self.overflow_since(check) {
                    self.restore(check);

                    for piece in pieces {
                        match piece {
                            StringPiece::Literal(_span) => self.push(TT::StringMiddle)?,
                            &StringPiece::Substitute(expr) => {
                                self.push(TT::StringSubStart)?;
                                self.indent_newline(|slf| slf.format_expr(expr, true))?;
                                self.push_newline();
                                self.push(TT::StringSubEnd)?;
                            }
                        }
                    }
                    self.push(TT::StringEnd)?;
                }
            }
            ExpressionKind::ArrayLiteral(elements) => {
                self.push(TT::OpenS)?;
                self.format_comma_list(elements, allow_wrap, Self::format_array_element)?;
                self.push(TT::CloseS)?;
            }
            ExpressionKind::TupleLiteral(elements) => {
                self.push(TT::OpenR)?;
                match elements.as_slice() {
                    &[] => {}
                    &[single] => {
                        self.format_expr(single, allow_wrap)?;
                        self.push(TT::Comma)?;
                    }
                    multiple => {
                        self.format_comma_list_copy(multiple, allow_wrap, Self::format_expr)?;
                    }
                }
                self.push(TT::CloseR)?;
            }
            ExpressionKind::RangeLiteral(range) => match *range {
                RangeLiteral::ExclusiveEnd { op_span: _, start, end } => {
                    self.format_maybe_binary_op(TT::Dots, false, start, end, allow_wrap)?;
                }
                RangeLiteral::InclusiveEnd { op_span: _, start, end } => {
                    self.format_maybe_binary_op(TT::DotsEq, false, start, Some(end), allow_wrap)?;
                }
                RangeLiteral::Length { op_span: _, start, len } => {
                    self.format_binary_op(TT::PlusDots, false, start, len, allow_wrap)?;
                }
            },
            ExpressionKind::ArrayComprehension(expr) => {
                let &ArrayComprehension {
                    body,
                    index,
                    span_keyword: _,
                    iter,
                } = expr;

                // try single line
                let check = self.checkpoint();
                self.push(TT::OpenS)?;
                self.format_array_element(&body, false)?;
                self.push_space();
                self.push(TT::For)?;
                self.push_space();
                self.format_maybe_id(index)?;
                self.push_space();
                self.push(TT::In)?;
                self.push_space();
                self.format_expr(iter, false)?;
                self.push(TT::CloseS)?;

                // maybe fallback to multi-line
                if allow_wrap && self.overflow_since(check) {
                    self.restore(check);
                    self.push(TT::OpenS)?;
                    self.indent_newline(|slf| {
                        slf.format_array_element(&body, true)?;
                        slf.push_newline();
                        slf.push(TT::For)?;
                        slf.push_space();
                        slf.format_maybe_id(index)?;
                        slf.push_space();
                        slf.push(TT::In)?;
                        slf.push_space();
                        slf.format_expr(iter, true)?;
                        Ok(())
                    })?;
                    self.push_newline();
                    self.push(TT::CloseS)?;
                }
            }
            &ExpressionKind::UnaryOp(op, inner) => {
                self.push(op.inner.token())?;
                self.format_expr(inner, allow_wrap)?;
            }
            &ExpressionKind::BinaryOp(op, left, right) => {
                self.format_binary_op(op.inner.token(), true, left, right, allow_wrap)?
            }
            &ExpressionKind::ArrayType(ref elements, inner) => {
                self.push(TT::OpenS)?;
                self.format_comma_list(&elements.inner, allow_wrap, Self::format_array_element)?;
                self.push(TT::CloseS)?;
                self.format_expr(inner, allow_wrap)?;
            }
            &ExpressionKind::ArrayIndex(base, ref indices) => {
                self.format_expr(base, allow_wrap)?;
                self.push(TT::OpenS)?;
                self.format_comma_list_copy(&indices.inner, allow_wrap, Self::format_expr)?;
                self.push(TT::CloseS)?;
            }
            &ExpressionKind::DotIndex(initial_base, initial_index) => {
                // deep collect
                // TODO collect together with DotIntIndex
                let mut base = initial_base;
                let mut indices = vec![initial_index];
                while let &ExpressionKind::DotIndex(new_base, new_index) = &self.source_expressions[base.inner] {
                    base = new_base;
                    indices.push(new_index);
                }

                // try single line
                let check = self.checkpoint();
                self.format_expr(base, false)?;
                for &index in indices.iter().rev() {
                    self.push(TT::Dot)?;
                    self.format_dot_index_kind(index)?;
                }

                // maybe fallback to multi-line
                if allow_wrap && self.overflow_since(check) {
                    self.restore(check);
                    self.format_expr(base, true)?;
                    self.indent_no_newline(|slf| {
                        for (&index, last) in indices.iter().rev().with_last() {
                            slf.push(TT::Dot)?;
                            slf.format_dot_index_kind(index)?;
                            if !last {
                                slf.push_newline();
                            }
                        }
                        Ok(())
                    })?;
                }
            }
            &ExpressionKind::Call(target, ref args) => {
                // TODO allow wrapping target?
                self.format_expr(target, false)?;
                self.format_args(args, allow_wrap)?;
            }
            ExpressionKind::Builtin(args) => {
                self.push(TT::Builtin)?;
                self.push(TT::OpenR)?;
                self.format_comma_list_copy(&args.inner, allow_wrap, Self::format_expr)?;
                self.push(TT::CloseR)?;
            }
            &ExpressionKind::UnsafeValueWithDomain(inner, domain) => {
                self.push(TT::UnsafeValueWithDomain)?;
                self.push(TT::OpenR)?;
                self.format_comma_list_copy(
                    &[Either::Left(inner), Either::Right(domain.inner)],
                    allow_wrap,
                    |slf, item, allow_wrap| match item {
                        Either::Left(inner) => slf.format_expr(inner, allow_wrap),
                        Either::Right(domain) => slf.format_domain(domain, allow_wrap),
                    },
                )?;
                self.push(TT::CloseR)?;
            }
            ExpressionKind::RegisterDelay(expr) => {
                let &RegisterDelay {
                    span_keyword: _,
                    value,
                    init,
                } = expr;
                self.push(TT::Reg)?;
                self.push(TT::OpenR)?;
                self.format_comma_list_copy(&[value, init], allow_wrap, Self::format_expr)?;
                self.push(TT::CloseR)?;
            }
        }
        Ok(())
    }

    fn format_dot_index_kind(&mut self, index: DotIndexKind) -> DiagResult {
        match index {
            DotIndexKind::Id(index) => self.format_id(index),
            DotIndexKind::Int(_span) => self.push(TT::IntLiteralDecimal),
        }
    }

    fn format_maybe_binary_op(
        &mut self,
        op: TT,
        op_space: bool,
        left: Option<Expression>,
        right: Option<Expression>,
        allow_wrap: bool,
    ) -> DiagResult {
        if let (Some(left), Some(right)) = (left, right) {
            self.format_binary_op(op, op_space, left, right, allow_wrap)?
        } else {
            if let Some(left) = left {
                self.format_expr(left, allow_wrap)?;
                if op_space {
                    self.push_space();
                }
            }
            self.push(op)?;
            if let Some(right) = right {
                if op_space {
                    self.push_space();
                }
                self.format_expr(right, allow_wrap)?;
            }
        }
        Ok(())
    }

    fn format_binary_op(
        &mut self,
        op: TT,
        op_space: bool,
        left: Expression,
        right: Expression,
        allow_wrap: bool,
    ) -> DiagResult {
        // try single line
        let check = self.checkpoint();
        self.format_expr(left, false)?;
        if op_space {
            self.push_space();
        }
        self.push(op)?;
        if op_space {
            self.push_space();
        }
        self.format_expr(right, allow_wrap)?;

        // maybe fallback to multi-line
        if allow_wrap && self.overflow_since(check) {
            self.restore(check);
            self.format_expr(left, true)?;
            self.indent_newline(|slf| {
                slf.push(op)?;
                if op_space {
                    slf.push_space();
                }
                slf.format_expr(right, allow_wrap)?;
                Ok(())
            })?;
        }

        Ok(())
    }

    fn format_array_element(&mut self, elem: &ArrayLiteralElement<Expression>, allow_wrap: bool) -> DiagResult {
        match *elem {
            ArrayLiteralElement::Single(elem) => self.format_expr(elem, allow_wrap),
            ArrayLiteralElement::Spread(_span, elem) => {
                self.push(TT::Star)?;
                self.format_expr(elem, allow_wrap)
            }
        }
    }

    fn format_args(&mut self, args: &Args, allow_wrap: bool) -> DiagResult {
        let Args { span: _, inner } = args;
        self.push(TT::OpenR)?;
        self.format_comma_list(inner, allow_wrap, Self::format_arg)?;
        self.push(TT::CloseR)?;
        Ok(())
    }

    fn format_arg(&mut self, arg: &Arg, allow_wrap: bool) -> DiagResult {
        let Arg { span: _, name, value } = *arg;
        if let Some(name) = name {
            self.format_id(name)?;
            self.push(TT::Eq)?;
        }
        self.format_expr(value, allow_wrap)?;
        Ok(())
    }

    fn format_comma_list<T>(
        &mut self,
        list: &[T],
        allow_wrap: bool,
        mut f: impl FnMut(&mut Self, &T, bool) -> DiagResult,
    ) -> DiagResult {
        // try single line
        let check = self.checkpoint();
        for (item, last) in list.iter().with_last() {
            f(self, item, false)?;
            if !last {
                self.push(TT::Comma)?;
                self.push_space();
            }
        }

        // maybe fallback to multi-line, one item per line
        if allow_wrap && self.overflow_since(check) {
            self.restore(check);
            self.indent_newline(|slf| {
                for item in list {
                    f(slf, item, true)?;
                    slf.push(TT::Comma)?;
                    slf.push_newline();
                }
                Ok(())
            })?;
        }

        Ok(())
    }

    fn format_comma_list_copy<T: Copy>(
        &mut self,
        list: &[T],
        allow_wrap: bool,
        mut f: impl FnMut(&mut Self, T, bool) -> DiagResult,
    ) -> DiagResult {
        self.format_comma_list(list, allow_wrap, |c, item, allow_wrap| f(c, *item, allow_wrap))
    }

    fn format_visibility<V: FormatVisibility>(&mut self, vis: &V) -> DiagResult {
        V::format_visibility(self, vis)
    }

    fn format_general_id(&mut self, id: GeneralIdentifier) -> DiagResult {
        match id {
            GeneralIdentifier::Simple(id) => self.format_id(id),
            GeneralIdentifier::FromString(_span, inner) => {
                // TODO support wrapping?
                self.push(TT::IdFromStr)?;
                self.push(TT::OpenR)?;
                self.format_expr(inner, false)?;
                self.push(TT::CloseR)?;
                Ok(())
            }
        }
    }

    fn format_maybe_general_id(&mut self, id: MaybeGeneralIdentifier) -> DiagResult {
        match id {
            MaybeIdentifier::Dummy(_span) => self.push(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.format_general_id(id),
        }
    }

    fn format_maybe_id(&mut self, id: MaybeIdentifier) -> DiagResult {
        match id {
            MaybeIdentifier::Dummy(_span) => self.push(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.format_id(id),
        }
    }

    fn format_id(&mut self, id: Identifier) -> DiagResult {
        let Identifier { span: _ } = id;
        self.push(TT::Identifier)
    }
}

trait FormatVisibility {
    fn format_visibility(c: &mut FormatContext, vis: &Self) -> DiagResult;
}

impl FormatVisibility for Visibility {
    fn format_visibility(c: &mut FormatContext, vis: &Visibility) -> DiagResult {
        match *vis {
            Visibility::Public(_span) => {
                c.push(TT::Public)?;
                c.push_space();
            }
            Visibility::Private => {
                // do nothing, private visibility is the default
            }
        }
        Ok(())
    }
}

impl FormatVisibility for () {
    fn format_visibility(_: &mut FormatContext, _: &()) -> DiagResult {
        // do nothing, no visibility
        Ok(())
    }
}
