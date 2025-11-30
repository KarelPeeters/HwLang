use crate::syntax::ast::{
    ArenaExpressions, Arg, ArrayComprehension, ArrayLiteralElement, Assignment, Block, BlockExpression, BlockStatement,
    BlockStatementKind, ClockedBlock, ClockedBlockReset, CombinatorialBlock, CommonDeclaration, CommonDeclarationNamed,
    CommonDeclarationNamedKind, ConstBlock, ConstDeclaration, DomainKind, EnumDeclaration, EnumVariant, Expression,
    ExpressionKind, ExtraItem, ExtraList, FileContent, ForStatement, FunctionDeclaration, GeneralIdentifier,
    IfCondBlockPair, IfStatement, ImportEntry, ImportFinalKind, InterfaceView, Item, ItemDefInterface,
    ItemDefModuleExternal, ItemDefModuleInternal, MatchBranch, MatchPattern, MatchStatement, MaybeGeneralIdentifier,
    MaybeIdentifier, ModuleInstance, ModulePortBlock, ModulePortInBlock, ModulePortInBlockKind, ModulePortItem,
    ModulePortSingle, ModulePortSingleKind, ModuleStatement, ModuleStatementKind, Parameter, Parameters,
    PortConnection, PortSingleKindInner, RangeLiteral, RegDeclaration, RegOutPortMarker, RegisterDelay,
    ReturnStatement, StringPiece, StructDeclaration, StructField, SyncDomain, TypeDeclaration, VariableDeclaration,
    Visibility, WhileStatement, WireDeclaration, WireDeclarationDomainTyKind, WireDeclarationKind,
};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::syntax::source::SourceDatabase;
use crate::syntax::token::apply_string_literal_escapes;
use crate::util::regex::RegexDfa;
use indexmap::IndexMap;
use std::ops::ControlFlow;

pub fn syntax_visit<V: SyntaxVisitor>(
    source: &SourceDatabase,
    file: &FileContent,
    visitor: &mut V,
) -> ControlFlow<V::Break> {
    let FileContent {
        span: _,
        items,
        arena_expressions,
    } = file;
    let mut ctx = VisitContext {
        source,
        arena_expressions,
        visitor,
    };
    ctx.visit_file(items)
}

pub enum FoldRangeKind {
    Comment,
    Imports,
    Region,
}

// TODO document all of this
pub trait SyntaxVisitor {
    type Break;
    const SCOPE_DECLARE: bool;

    fn should_visit_span(&self, span: Span) -> bool;

    fn report_id_declare(&mut self, id: GeneralIdentifier) -> ControlFlow<Self::Break, ()> {
        let _ = id;
        ControlFlow::Continue(())
    }

    fn report_id_use(
        &mut self,
        scope: &DeclScope<'_>,
        id: GeneralIdentifier,
        id_eval: impl Fn(&SourceDatabase, GeneralIdentifier) -> EvaluatedId<&str>,
    ) -> ControlFlow<Self::Break, ()> {
        let _ = (scope, id, id_eval);
        ControlFlow::Continue(())
    }

    fn report_range(&mut self, range: Span, fold: Option<FoldRangeKind>) {
        let _ = (range, fold);
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Conditional {
    No,
    Yes,
}

#[derive(Debug)]
pub struct DeclScope<'p> {
    parent: Option<&'p DeclScope<'p>>,
    content: DeclScopeContent,
}

#[derive(Debug)]
struct DeclScopeContent {
    simple: IndexMap<String, Vec<(Span, Conditional)>>,
    pattern: Vec<(RegexDfa, Span, Conditional)>,
}

#[derive(Debug)]
pub enum EvaluatedId<S> {
    Simple(S),
    Pattern(RegexDfa),
}

impl EvaluatedId<&str> {
    pub fn into_owned(self) -> EvaluatedId<String> {
        match self {
            EvaluatedId::Simple(s) => EvaluatedId::Simple(s.to_owned()),
            EvaluatedId::Pattern(regex) => EvaluatedId::Pattern(regex),
        }
    }
}

struct VisitContext<'a, 'v, V> {
    source: &'a SourceDatabase,
    arena_expressions: &'a ArenaExpressions,
    visitor: &'v mut V,
}

impl<'p> DeclScope<'p> {
    fn new_root() -> Self {
        Self {
            parent: None,
            content: DeclScopeContent {
                simple: IndexMap::new(),
                pattern: vec![],
            },
        }
    }

    fn new_child(&'p self) -> Self {
        Self {
            parent: Some(self),
            content: DeclScopeContent {
                simple: IndexMap::new(),
                pattern: vec![],
            },
        }
    }

    fn merge_conditional_child(&mut self, content: DeclScopeContent) {
        let DeclScopeContent {
            simple: fixed,
            pattern: regex,
        } = content;
        for (k, v) in fixed {
            for (span, _) in v {
                self.content
                    .simple
                    .entry(k.clone())
                    .or_default()
                    .push((span, Conditional::Yes));
            }
        }
        for (k, v, _) in regex {
            self.content.pattern.push((k, v, Conditional::Yes));
        }
    }

    fn declare(&mut self, cond: Conditional, span: Span, id: EvaluatedId<&str>) {
        match id {
            EvaluatedId::Simple(id_inner) => {
                self.content
                    .simple
                    .entry(id_inner.to_owned())
                    .or_default()
                    .push((span, cond));
            }
            EvaluatedId::Pattern(pattern) => self.content.pattern.push((pattern, span, cond)),
        }
    }

    pub fn find(&self, id: &EvaluatedId<&str>) -> Vec<Span> {
        let mut curr = self;
        let mut result = vec![];

        loop {
            let mut any_certain = false;

            let DeclScopeContent {
                simple: fixed,
                pattern: patterns,
            } = &curr.content;

            // check simple
            match id {
                &EvaluatedId::Simple(id) => {
                    if let Some(entries) = fixed.get(id) {
                        for &(span, cond) in entries {
                            result.push(span);
                            match cond {
                                Conditional::Yes => {}
                                Conditional::No => any_certain = true,
                            }
                        }
                    }
                }
                EvaluatedId::Pattern(id) => {
                    for (key, entries) in fixed {
                        if id.could_match_str(key) {
                            for &(span, cond) in entries {
                                result.push(span);
                                // pattern matches are never certain
                                let _ = cond;
                            }
                        }
                    }
                }
            }

            // check patterns
            for &(ref pattern, span, c) in patterns {
                let could_match = match id {
                    EvaluatedId::Simple(id) => pattern.could_match_str(id),
                    EvaluatedId::Pattern(id) => pattern.could_match_pattern(id),
                };
                if could_match {
                    result.push(span);
                    // pattern matches are never certain
                    let _ = c;
                }
            }

            if any_certain {
                break;
            }

            curr = match &curr.parent {
                Some(parent) => parent,
                None => break,
            }
        }

        result
    }
}

macro_rules! check_skip {
    ($ctx: expr, $span: expr) => {
        if !$ctx.visitor.should_visit_span($span) {
            return ControlFlow::Continue(());
        }
    };
}

impl<V: SyntaxVisitor> VisitContext<'_, '_, V> {
    fn visit_file(&mut self, items: &[Item]) -> ControlFlow<V::Break> {
        // first pass: declare all items in scope
        // TODO should we filter based on span here or not? maybe make an enum for how declarations should behave
        //   at least document this properly
        let mut scope = DeclScope::new_root();
        for item in items {
            if let Some(info) = item.info().declaration {
                self.scope_declare(&mut scope, Conditional::No, info.id.into())?;
            }

            if let Item::Import(item) = item {
                let mut visit_entry = |slf: &mut VisitContext<V>, entry: &ImportEntry| {
                    let &ImportEntry { span, id, as_ } = entry;
                    slf.visitor.report_range(span, None);
                    let final_id = as_.unwrap_or(MaybeIdentifier::Identifier(id));
                    slf.scope_declare(&mut scope, Conditional::Yes, final_id.into())
                };
                match &item.entry.inner {
                    ImportFinalKind::Single(entry) => {
                        visit_entry(self, entry)?;
                    }
                    ImportFinalKind::Multi(entries) => {
                        if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
                            self.visitor.report_range(first.span.join(last.span), None);
                        }

                        for entry in entries {
                            visit_entry(self, entry)?;
                        }
                    }
                }
            }
        }

        // second pass: visit the items themselves
        for item in items {
            self.visit_file_item(&scope, item)?;
        }

        ControlFlow::Continue(())
    }

    fn visit_file_item(&mut self, scope_file: &DeclScope, item: &Item) -> ControlFlow<V::Break> {
        let item_span = item.info().span_full;
        check_skip!(self, item_span);

        // TODO report single fold range for consecutive imports
        // TODO report single fold range for multi-line comments? that's really more of a token thing, not really AST
        let fold = if let Item::Import(_) = item {
            FoldRangeKind::Imports
        } else {
            FoldRangeKind::Region
        };
        self.visitor.report_range(item_span, Some(fold));

        match item {
            Item::Import(_) => {
                // TODO implement "go to definition" for paths, entries, ...
                ControlFlow::Continue(())
            }
            Item::CommonDeclaration(decl) => {
                // we don't need a real scope here, the declaration would already have happened in the first pass
                let mut scope_dummy = scope_file.new_child();
                self.visit_common_declaration(&mut scope_dummy, &decl.inner)
            }
            Item::ModuleInternal(decl) => {
                let ItemDefModuleInternal {
                    span: _,
                    vis: _,
                    id: _,
                    params,
                    ports,
                    body,
                } = decl;

                let mut scope_params = scope_file.new_child();
                if let Some(params) = params {
                    self.visit_parameters(&mut scope_params, params)?;
                }

                let mut scope_ports = scope_params.new_child();
                self.visit_extra_list(&mut scope_ports, &ports.inner, &mut |slf, scope_ports, port| {
                    slf.visit_port_item(scope_ports, port)
                })?;

                let mut scope_body = scope_ports.new_child();
                self.collect_module_body_pub_declarations(&mut scope_body, Conditional::Yes, body)?;

                self.visit_block_module_inner(&mut scope_body, body)?;
                ControlFlow::Continue(())
            }
            Item::ModuleExternal(decl) => {
                let ItemDefModuleExternal {
                    span: _,
                    span_ext: _,
                    vis: _,
                    id: _,
                    params,
                    ports,
                } = decl;

                let mut scope_params = scope_file.new_child();
                if let Some(params) = params {
                    self.visit_parameters(&mut scope_params, params)?;
                }

                let mut scope_ports = scope_params.new_child();
                self.visit_extra_list(&mut scope_ports, &ports.inner, &mut |slf, scope_ports, port| {
                    slf.visit_port_item(scope_ports, port)
                })?;

                ControlFlow::Continue(())
            }
            Item::Interface(decl) => {
                let ItemDefInterface {
                    span: _,
                    vis: _,
                    id: _,
                    params,
                    span_body: _,
                    port_types,
                    views,
                } = decl;

                let mut scope_params = scope_file.new_child();
                if let Some(params) = params {
                    self.visit_parameters(&mut scope_params, params)?;
                }

                let mut scope_body = scope_params.new_child();
                self.visit_extra_list(
                    &mut scope_body,
                    port_types,
                    &mut |slf, scope_body, &(port_name, port_ty)| {
                        slf.visit_expression(scope_body, port_ty)?;
                        slf.scope_declare(scope_body, Conditional::No, port_name.into())?;
                        ControlFlow::Continue(())
                    },
                )?;

                for view in views {
                    let &InterfaceView {
                        span,
                        id,
                        ref port_dirs,
                    } = view;

                    self.visitor.report_range(span, Some(FoldRangeKind::Region));

                    self.visit_extra_list(
                        &mut scope_body,
                        port_dirs,
                        &mut |slf, scope_body, &(port_name, _port_dir)| {
                            slf.visit_id_usage(scope_body, port_name.into())
                        },
                    )?;
                    self.scope_declare(&mut scope_body, Conditional::No, id.into())?;
                }

                ControlFlow::Continue(())
            }
        }
    }

    fn visit_port_item(&mut self, scope_ports: &mut DeclScope, port: &ModulePortItem) -> ControlFlow<V::Break> {
        self.visitor.report_range(port.span(), None);

        match port {
            ModulePortItem::Single(port) => {
                let &ModulePortSingle { span: _, id, ref kind } = port;
                match kind {
                    ModulePortSingleKind::Port { direction: _, kind } => match kind {
                        PortSingleKindInner::Clock { span_clock: _ } => {}
                        &PortSingleKindInner::Normal { domain, ty } => {
                            self.visit_domain(scope_ports, domain)?;
                            self.visit_expression(scope_ports, ty)?;
                        }
                    },
                    &ModulePortSingleKind::Interface {
                        span_keyword: _,
                        domain,
                        interface,
                    } => {
                        self.visit_domain(scope_ports, domain)?;
                        self.visit_expression(scope_ports, interface)?;
                    }
                }
                self.scope_declare(scope_ports, Conditional::No, id.into())?;
                ControlFlow::Continue(())
            }
            ModulePortItem::Block(block) => {
                let &ModulePortBlock {
                    span,
                    domain,
                    ref ports,
                } = block;

                self.visitor.report_range(span, Some(FoldRangeKind::Region));

                self.visit_domain(scope_ports, domain)?;
                self.visit_extra_list(scope_ports, ports, &mut |slf, scope_ports, port| {
                    let &ModulePortInBlock { span, id, ref kind } = port;
                    slf.visitor.report_range(span, None);

                    match *kind {
                        ModulePortInBlockKind::Port { direction: _, ty } => {
                            slf.visit_expression(scope_ports, ty)?;
                        }
                        ModulePortInBlockKind::Interface {
                            span_keyword: _,
                            interface,
                        } => {
                            slf.visit_expression(scope_ports, interface)?;
                        }
                    }
                    slf.scope_declare(scope_ports, Conditional::No, id.into())?;
                    ControlFlow::Continue(())
                })?;

                ControlFlow::Continue(())
            }
        }
    }

    fn visit_domain(&mut self, scope: &DeclScope, domain: Spanned<DomainKind<Expression>>) -> ControlFlow<V::Break> {
        self.visitor.report_range(domain.span, None);

        match domain.inner {
            DomainKind::Const => ControlFlow::Continue(()),
            DomainKind::Async => ControlFlow::Continue(()),
            DomainKind::Sync(domain) => self.visit_domain_sync(scope, domain),
        }
    }

    fn visit_domain_sync(&mut self, scope: &DeclScope, domain: SyncDomain<Expression>) -> ControlFlow<V::Break> {
        let SyncDomain { clock, reset } = domain;
        self.visit_expression(scope, clock)?;
        if let Some(reset) = reset {
            self.visit_expression(scope, reset)?;
        }
        ControlFlow::Continue(())
    }

    fn visit_common_declaration<S>(
        &mut self,
        scope_parent: &mut DeclScope,
        decl: &CommonDeclaration<S>,
    ) -> ControlFlow<V::Break> {
        self.visitor.report_range(decl.span(), Some(FoldRangeKind::Region));

        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis: _, kind } = decl;
                let id = match kind {
                    CommonDeclarationNamedKind::Type(decl) => {
                        let &TypeDeclaration {
                            span: _,
                            id,
                            ref params,
                            body,
                        } = decl;

                        let mut scope_params = scope_parent.new_child();
                        if let Some(params) = params {
                            self.visit_parameters(&mut scope_params, params)?;
                        }
                        self.visit_expression(&scope_params, body)?;

                        id
                    }
                    CommonDeclarationNamedKind::Const(decl) => {
                        let &ConstDeclaration { span: _, id, ty, value } = decl;

                        if let Some(ty) = ty {
                            self.visit_expression(scope_parent, ty)?;
                        }
                        self.visit_expression(scope_parent, value)?;

                        id
                    }
                    CommonDeclarationNamedKind::Struct(decl) => {
                        let &StructDeclaration {
                            span: _,
                            span_body: _,
                            id,
                            ref params,
                            ref fields,
                        } = decl;

                        let mut scope_params = scope_parent.new_child();
                        if let Some(params) = params {
                            self.visit_parameters(&mut scope_params, params)?;
                        }
                        // TODO why the weird scoping here?
                        self.visit_extra_list(&mut scope_params, fields, &mut |slf, _scope, field| {
                            let &StructField { span: _, id: _, ty } = field;
                            slf.visit_expression(scope_parent, ty)?;
                            ControlFlow::Continue(())
                        })?;

                        id
                    }
                    CommonDeclarationNamedKind::Enum(decl) => {
                        let &EnumDeclaration {
                            span: _,
                            id,
                            ref params,
                            ref variants,
                        } = decl;

                        let mut scope_params = scope_parent.new_child();
                        if let Some(params) = params {
                            self.visit_parameters(&mut scope_params, params)?;
                        }

                        self.visit_extra_list(&mut scope_params, variants, &mut |slf, scope_params, variant| {
                            let &EnumVariant { span, id: _, content } = variant;
                            slf.visitor.report_range(span, None);

                            // TODO declare variant name
                            if let Some(content) = content {
                                slf.visit_expression(scope_params, content)?;
                            }
                            ControlFlow::Continue(())
                        })?;

                        id
                    }
                    CommonDeclarationNamedKind::Function(decl) => {
                        let &FunctionDeclaration {
                            span: _,
                            id,
                            ref params,
                            ret_ty,
                            ref body,
                        } = decl;

                        let mut scope_params = scope_parent.new_child();
                        self.visit_parameters(&mut scope_params, params)?;
                        if let Some(ret_ty) = ret_ty {
                            self.visit_expression(&scope_params, ret_ty)?;
                        }
                        self.visit_block_statements(&scope_params, body)?;

                        id
                    }
                };

                self.scope_declare(scope_parent, Conditional::No, id.into())?;
            }
            CommonDeclaration::ConstBlock(block) => {
                let ConstBlock { span_keyword: _, block } = block;
                self.visit_block_statements(scope_parent, block)?;
            }
        }

        ControlFlow::Continue(())
    }

    fn visit_parameters(&mut self, scope: &mut DeclScope, params: &Parameters) -> ControlFlow<V::Break> {
        let &Parameters { span, ref items } = params;
        self.visitor.report_range(span, Some(FoldRangeKind::Region));

        self.visit_extra_list(scope, items, &mut |slf, scope, param| {
            let &Parameter { span, id, ty, default } = param;

            slf.visitor.report_range(span, None);

            slf.visit_expression(scope, ty)?;
            if let Some(default) = default {
                slf.visit_expression(scope, default)?;
            }
            slf.scope_declare(scope, Conditional::No, id.into())?;
            ControlFlow::Continue(())
        })
    }

    fn visit_extra_list<I: HasSpan>(
        &mut self,
        scope_parent: &mut DeclScope,
        extra: &ExtraList<I>,
        f: &mut impl FnMut(&mut Self, &mut DeclScope, &I) -> ControlFlow<V::Break>,
    ) -> ControlFlow<V::Break> {
        let &ExtraList { span, ref items } = extra;

        self.visitor.report_range(span, Some(FoldRangeKind::Region));

        for item in items {
            match item {
                ExtraItem::Inner(item) => f(self, scope_parent, item)?,
                ExtraItem::Declaration(decl) => self.visit_common_declaration(scope_parent, decl)?,
                ExtraItem::If(if_stmt) => {
                    let mut scope_inner = scope_parent.new_child();
                    self.visit_if_stmt(&mut scope_inner, if_stmt, &mut |slf, s: &mut DeclScope, b| {
                        slf.visit_extra_list(s, b, f)
                    })?;

                    // TODO this is overly pessimistic, if/else combo can become non-conditional,
                    //   which we don't model correctly
                    scope_parent.merge_conditional_child(scope_inner.content);
                }
            }
        }

        ControlFlow::Continue(())
    }

    fn visit_if_stmt<I>(
        &mut self,
        scope: &mut DeclScope,
        if_stmt: &IfStatement<I>,
        f: &mut impl FnMut(&mut Self, &mut DeclScope, &I) -> ControlFlow<V::Break>,
    ) -> ControlFlow<V::Break> {
        let &IfStatement {
            span,
            ref initial_if,
            ref else_ifs,
            ref final_else,
        } = if_stmt;

        self.visitor.report_range(span, None);

        let mut visit_pair = |slf: &mut Self, pair: &IfCondBlockPair<I>| {
            let &IfCondBlockPair {
                span: _,
                span_if: _,
                cond,
                ref block,
            } = pair;

            slf.visit_expression(scope, cond)?;
            f(slf, scope, block)?;

            ControlFlow::Continue(())
        };

        visit_pair(self, initial_if)?;
        for else_if in else_ifs {
            visit_pair(self, else_if)?;
        }
        if let Some(final_else) = final_else {
            f(self, scope, final_else)?;
        }
        ControlFlow::Continue(())
    }

    fn visit_for_stmt<S>(
        &mut self,
        scope: &DeclScope,
        stmt: &ForStatement<S>,
        f: impl FnOnce(&mut Self, &DeclScope, &Block<S>) -> ControlFlow<V::Break>,
    ) -> ControlFlow<V::Break> {
        let &ForStatement {
            span_keyword: _,
            index,
            index_ty,
            iter,
            ref body,
        } = stmt;

        self.visitor.report_range(stmt.span(), None);

        self.visit_expression(scope, iter)?;

        if let Some(index_ty) = index_ty {
            self.visit_expression(scope, index_ty)?;
        }

        let mut scope_inner = scope.new_child();
        self.scope_declare(&mut scope_inner, Conditional::No, index.into())?;

        f(self, &scope_inner, body)?;

        ControlFlow::Continue(())
    }

    fn visit_block_statements(
        &mut self,
        scope_parents: &DeclScope,
        block: &Block<BlockStatement>,
    ) -> ControlFlow<V::Break> {
        let &Block { span, ref statements } = block;

        // declarations inside a block can't leak outside, so we can skip here
        check_skip!(self, span);

        self.visitor.report_range(span, Some(FoldRangeKind::Region));

        let mut scope = scope_parents.new_child();
        for stmt in statements {
            self.visit_statement(&mut scope, stmt)?;
        }

        ControlFlow::Continue(())
    }

    fn visit_statement(&mut self, scope: &mut DeclScope, stmt: &BlockStatement) -> ControlFlow<V::Break> {
        let stmt_span = stmt.span;

        self.visitor.report_range(stmt_span, None);

        match &stmt.inner {
            BlockStatementKind::CommonDeclaration(decl) => {
                self.visit_common_declaration(scope, decl)?;
            }
            BlockStatementKind::VariableDeclaration(decl) => {
                let &VariableDeclaration {
                    span: _,
                    mutable: _,
                    id,
                    ty,
                    init,
                } = decl;
                if let Some(ty) = ty {
                    self.visit_expression(scope, ty)?;
                }
                if let Some(init) = init {
                    self.visit_expression(scope, init)?;
                }
                self.scope_declare(scope, Conditional::No, id.into())?;
            }
            BlockStatementKind::Assignment(stmt) => {
                let &Assignment {
                    span: _,
                    op: _,
                    target,
                    value,
                } = stmt;
                self.visit_expression(scope, target)?;
                self.visit_expression(scope, value)?;
            }
            &BlockStatementKind::Expression(expr) => {
                self.visit_expression(scope, expr)?;
            }
            BlockStatementKind::Block(block) => {
                self.visit_block_statements(scope, block)?;
            }
            BlockStatementKind::If(stmt) => {
                check_skip!(self, stmt_span);
                self.visit_if_stmt(scope, stmt, &mut |slf, s, b| slf.visit_block_statements(s, b))?;
            }
            BlockStatementKind::Match(stmt) => {
                check_skip!(self, stmt_span);

                let &MatchStatement {
                    target,
                    span_branches: _,
                    ref branches,
                } = stmt;
                self.visit_expression(scope, target)?;

                for branch in branches {
                    let MatchBranch { pattern, block } = branch;

                    let mut scope_inner = scope.new_child();
                    match &pattern.inner {
                        MatchPattern::Wildcard => {}
                        &MatchPattern::Val(id) => {
                            self.scope_declare(&mut scope_inner, Conditional::No, id.into())?;
                        }
                        &MatchPattern::Equal(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                        &MatchPattern::In(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                        &MatchPattern::EnumVariant(_variant, id) => {
                            if let Some(id) = id {
                                self.scope_declare(&mut scope_inner, Conditional::No, id.into())?;
                            }
                        }
                    }

                    self.visit_block_statements(&scope_inner, block)?;
                }
            }
            BlockStatementKind::For(stmt) => {
                check_skip!(self, stmt_span);
                self.visit_for_stmt(scope, stmt, |slf, s, b| slf.visit_block_statements(s, b))?;
            }
            BlockStatementKind::While(stmt) => {
                check_skip!(self, stmt_span);
                let &WhileStatement {
                    span_keyword: _,
                    cond,
                    ref body,
                } = stmt;
                self.visit_expression(scope, cond)?;
                self.visit_block_statements(scope, body)?;
            }
            BlockStatementKind::Return(stmt) => {
                let &ReturnStatement { span_return: _, value } = stmt;
                if let Some(value) = value {
                    self.visit_expression(scope, value)?;
                }
            }
            BlockStatementKind::Break { span: _ } | BlockStatementKind::Continue { span: _ } => {}
        }

        ControlFlow::Continue(())
    }

    fn collect_module_body_pub_declarations(
        &mut self,
        scope_body: &mut DeclScope,
        cond: Conditional,
        curr_block: &Block<ModuleStatement>,
    ) -> ControlFlow<V::Break> {
        for stmt in &curr_block.statements {
            match &stmt.inner {
                // control flow
                ModuleStatementKind::Block(block) => {
                    self.collect_module_body_pub_declarations(scope_body, cond, block)?;
                }
                ModuleStatementKind::If(if_stmt) => {
                    let IfStatement {
                        span: _,
                        initial_if,
                        else_ifs,
                        final_else,
                    } = if_stmt;

                    let IfCondBlockPair {
                        span: _,
                        span_if: _,
                        cond: _,
                        block,
                    } = initial_if;
                    self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, block)?;

                    for else_if in else_ifs {
                        let IfCondBlockPair {
                            span: _,
                            span_if: _,
                            cond: _,
                            block,
                        } = else_if;
                        self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, block)?;
                    }

                    if let Some(final_else) = final_else {
                        self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, final_else)?;
                    }
                }
                ModuleStatementKind::For(for_stmt) => {
                    let ForStatement {
                        span_keyword: _,
                        index: _,
                        index_ty: _,
                        iter: _,
                        body,
                    } = for_stmt;

                    self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, body)?;
                }

                // potentially public declarations
                // TODO it's annoying that this messes up visiting order a bit, maybe we should only call the visitor
                //   for these right next to the non-public visits
                ModuleStatementKind::RegDeclaration(decl) => {
                    let &RegDeclaration {
                        vis,
                        id,
                        sync: _,
                        ty: _,
                        init: _,
                    } = decl;
                    match vis {
                        Visibility::Public { span: _ } => self.scope_declare(scope_body, cond, id)?,
                        Visibility::Private => {}
                    }
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let &WireDeclaration {
                        vis,
                        span_keyword: _,
                        id,
                        kind: _,
                    } = decl;
                    match vis {
                        Visibility::Public { span: _ } => self.scope_declare(scope_body, cond, id)?,
                        Visibility::Private => {}
                    }
                }

                // no public declarations
                ModuleStatementKind::CommonDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
            }
        }

        ControlFlow::Continue(())
    }

    fn visit_block_module(
        &mut self,
        scope_parent: &DeclScope,
        block: &Block<ModuleStatement>,
    ) -> ControlFlow<V::Break> {
        let mut scope = scope_parent.new_child();
        self.visit_block_module_inner(&mut scope, block)
    }

    fn visit_block_module_inner(
        &mut self,
        scope: &mut DeclScope,
        block: &Block<ModuleStatement>,
    ) -> ControlFlow<V::Break> {
        let &Block { span, ref statements } = block;
        check_skip!(self, span);
        self.visitor.report_range(span, Some(FoldRangeKind::Region));

        // match the two-pass system from the main compiler
        for stmt in statements {
            self.visitor.report_range(stmt.span, None);

            match &stmt.inner {
                ModuleStatementKind::CommonDeclaration(decl) => {
                    self.visit_common_declaration(scope, decl)?;
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let &RegDeclaration {
                        vis,
                        id,
                        sync,
                        ty,
                        init,
                    } = decl;

                    if let Some(sync) = sync {
                        self.visit_domain_sync(scope, sync.inner)?;
                    }
                    self.visit_expression(scope, ty)?;
                    self.visit_expression(scope, init)?;

                    self.visit_maybe_general(scope, id)?;
                    match vis {
                        Visibility::Public { span: _ } => {}
                        Visibility::Private => self.scope_declare(scope, Conditional::No, id)?,
                    }
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let &WireDeclaration {
                        vis,
                        span_keyword: _,
                        id,
                        kind,
                    } = decl;

                    match kind {
                        WireDeclarationKind::Normal {
                            domain_ty,
                            assign_span_and_value,
                        } => {
                            match domain_ty {
                                WireDeclarationDomainTyKind::Clock { span_clock: _ } => {}
                                WireDeclarationDomainTyKind::Normal { domain, ty } => {
                                    if let Some(domain) = domain {
                                        self.visit_domain(scope, domain)?;
                                    }
                                    if let Some(ty) = ty {
                                        self.visit_expression(scope, ty)?;
                                    }
                                }
                            }

                            if let Some((_, value)) = assign_span_and_value {
                                self.visit_expression(scope, value)?;
                            }
                        }
                        WireDeclarationKind::Interface {
                            domain,
                            span_keyword: _,
                            interface,
                        } => {
                            if let Some(domain) = domain {
                                self.visit_domain(scope, domain)?;
                            }
                            self.visit_expression(scope, interface)?;
                        }
                    }

                    self.visit_maybe_general(scope, id)?;
                    match vis {
                        Visibility::Public { span: _ } => {}
                        Visibility::Private => self.scope_declare(scope, Conditional::No, id)?,
                    }
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    let &RegOutPortMarker { id, init } = decl;
                    self.visit_expression(scope, init)?;
                    self.visit_id_usage(scope, id.into())?;
                }

                // no declarations, handled in the second pass
                ModuleStatementKind::Block(_) => {}
                ModuleStatementKind::If(_) => {}
                ModuleStatementKind::For(_) => {}
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
            }
        }

        for stmt in statements {
            match &stmt.inner {
                ModuleStatementKind::Block(block) => {
                    self.visit_block_module(scope, block)?;
                }
                ModuleStatementKind::If(stmt) => {
                    self.visit_if_stmt(scope, stmt, &mut |slf, s, b| slf.visit_block_module(s, b))?;
                }
                ModuleStatementKind::For(stmt) => {
                    self.visit_for_stmt(scope, stmt, |slf, s, b| slf.visit_block_module(s, b))?
                }
                ModuleStatementKind::CombinatorialBlock(stmt) => {
                    let CombinatorialBlock { span_keyword: _, block } = stmt;
                    self.visit_block_statements(scope, block)?;
                }
                ModuleStatementKind::ClockedBlock(stmt) => {
                    let &ClockedBlock {
                        span_keyword: _,
                        span_domain: _,
                        clock,
                        reset,
                        ref block,
                    } = stmt;
                    self.visit_expression(scope, clock)?;
                    if let Some(reset) = reset {
                        let ClockedBlockReset { kind: _, signal } = reset.inner;
                        self.visit_expression(scope, signal)?;
                    }
                    self.visit_block_statements(scope, block)?;
                }
                ModuleStatementKind::Instance(stmt) => {
                    let &ModuleInstance {
                        name: _,
                        span_keyword: _,
                        module,
                        ref port_connections,
                    } = stmt;

                    self.visit_expression(scope, module)?;

                    for conn in &port_connections.inner {
                        let &PortConnection { id, expr } = &conn.inner;
                        // TODO try resolving port name, needs type info
                        let _ = id;
                        self.visit_expression(scope, expr.expr())?;
                    }
                }

                // declarations, already handled in the first pass
                ModuleStatementKind::CommonDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
            }
        }

        ControlFlow::Continue(())
    }

    fn visit_array_literal_element(
        &mut self,
        scope: &DeclScope,
        elem: ArrayLiteralElement<Expression>,
    ) -> ControlFlow<V::Break> {
        self.visitor.report_range(elem.span(), None);

        match elem {
            ArrayLiteralElement::Single(elem) => self.visit_expression(scope, elem),
            ArrayLiteralElement::Spread(_, elem) => self.visit_expression(scope, elem),
        }
    }

    fn visit_expression(&mut self, scope: &DeclScope, expr: Expression) -> ControlFlow<V::Break> {
        // expressions can't contain declarations that leak outside, so we can skip here
        check_skip!(self, expr.span);

        self.visitor.report_range(expr.span, None);

        match &self.arena_expressions[expr.inner] {
            &ExpressionKind::Wrapped(inner) => {
                self.visit_expression(scope, inner)?;
            }
            ExpressionKind::Block(block) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = block;
                let mut scope_inner = scope.new_child();
                for stmt in statements {
                    self.visit_statement(&mut scope_inner, stmt)?
                }
                self.visit_expression(&scope_inner, expression)?;
            }
            &ExpressionKind::Id(id) => {
                self.visit_id_usage(scope, id.into())?;
            }
            ExpressionKind::StringLiteral(pieces) => {
                for &piece in pieces {
                    match piece {
                        StringPiece::Literal(span) => {
                            self.visitor.report_range(span, None);
                        }
                        StringPiece::Substitute(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                    }
                }
            }
            ExpressionKind::ArrayLiteral(elems) => {
                // report extra selection range covering only the inside of the literal
                if let (Some(first), Some(last)) = (elems.first(), elems.last()) {
                    self.visitor.report_range(first.span().join(last.span()), None);
                }

                for &elem in elems {
                    self.visit_array_literal_element(scope, elem)?;
                }
            }
            ExpressionKind::TupleLiteral(elems) => {
                // report extra selection range covering only the inside of the literal
                if let (Some(first), Some(last)) = (elems.first(), elems.last()) {
                    self.visitor.report_range(first.span().join(last.span()), None);
                }

                for &elem in elems {
                    self.visit_expression(scope, elem)?;
                }
            }
            ExpressionKind::RangeLiteral(expr) => match *expr {
                RangeLiteral::ExclusiveEnd { op_span: _, start, end } => {
                    if let Some(start) = start {
                        self.visit_expression(scope, start)?;
                    }
                    if let Some(end) = end {
                        self.visit_expression(scope, end)?;
                    }
                }
                RangeLiteral::InclusiveEnd { op_span: _, start, end } => {
                    if let Some(start) = start {
                        self.visit_expression(scope, start)?;
                    }
                    self.visit_expression(scope, end)?;
                }
                RangeLiteral::Length {
                    op_span: _,
                    start,
                    length,
                } => {
                    self.visit_expression(scope, start)?;
                    self.visit_expression(scope, length)?;
                }
            },
            ExpressionKind::ArrayComprehension(expr) => {
                let &ArrayComprehension {
                    body,
                    index,
                    span_keyword: _,
                    iter,
                } = expr;

                self.visitor.report_range(body.span().join(iter.span), None);

                self.visit_expression(scope, iter)?;

                let mut scope_inner = scope.new_child();
                self.scope_declare(&mut scope_inner, Conditional::No, index.into())?;

                self.visit_array_literal_element(&scope_inner, body)?;
            }
            &ExpressionKind::UnaryOp(_, inner) => {
                self.visit_expression(scope, inner)?;
            }
            &ExpressionKind::BinaryOp(_, left, right) => {
                self.visit_expression(scope, left)?;
                self.visit_expression(scope, right)?;
            }
            &ExpressionKind::ArrayType(ref lens, inner) => {
                for &len in &lens.inner {
                    self.visit_array_literal_element(scope, len)?;
                }
                self.visit_expression(scope, inner)?;
            }
            &ExpressionKind::ArrayIndex(base, ref indices) => {
                self.visit_expression(scope, base)?;
                for &index in &indices.inner {
                    self.visit_expression(scope, index)?;
                }
            }
            &ExpressionKind::DotIndex(base, _) => {
                // TODO try resolving index, needs type info
                self.visit_expression(scope, base)?;
            }
            &ExpressionKind::Call(target, ref args) => {
                self.visit_expression(scope, target)?;

                self.visitor.report_range(args.span, None);

                for arg in &args.inner {
                    let &Arg { span, ref name, value } = arg;
                    self.visitor.report_range(span, None);
                    // TODO try resolving name, needs type info
                    let _ = name;
                    self.visit_expression(scope, value)?;
                }
            }
            &ExpressionKind::UnsafeValueWithDomain(value, domain) => {
                self.visit_expression(scope, value)?;
                self.visit_domain(scope, domain)?;
            }
            ExpressionKind::RegisterDelay(expr) => {
                let &RegisterDelay {
                    span_keyword: _,
                    value,
                    init,
                } = expr;

                self.visitor.report_range(value.span.join(init.span), None);

                self.visit_expression(scope, value)?;
                self.visit_expression(scope, init)?;
            }

            ExpressionKind::Dummy
            | ExpressionKind::Undefined
            | ExpressionKind::Type
            | ExpressionKind::TypeFunction
            | ExpressionKind::Builtin
            | ExpressionKind::IntLiteral(_)
            | ExpressionKind::BoolLiteral(_) => {}
        }

        ControlFlow::Continue(())
    }

    // TODO what is the difference between this and visit_id_usage?
    fn visit_maybe_general(&mut self, scope: &DeclScope, id: MaybeGeneralIdentifier) -> ControlFlow<V::Break> {
        self.visitor.report_range(id.span(), None);

        match id {
            MaybeGeneralIdentifier::Dummy { span: _ } => {}
            MaybeGeneralIdentifier::Identifier(id) => match id {
                GeneralIdentifier::Simple(_id) => {}
                GeneralIdentifier::FromString(_span, expr) => {
                    self.visit_expression(scope, expr)?;
                }
            },
        }

        ControlFlow::Continue(())
    }

    fn visit_id_usage(&mut self, scope: &DeclScope, id: MaybeGeneralIdentifier) -> ControlFlow<V::Break> {
        check_skip!(self, id.span());

        self.visitor.report_range(id.span(), None);

        match id {
            MaybeGeneralIdentifier::Dummy { .. } => ControlFlow::Continue(()),
            MaybeGeneralIdentifier::Identifier(id) => {
                // first visit the inner expressions if any
                match id {
                    GeneralIdentifier::Simple(_id) => {}
                    GeneralIdentifier::FromString(_span, expr) => {
                        self.visit_expression(scope, expr)?;
                    }
                }

                // report the id usage itself
                self.visitor.report_id_use(scope, id, |source, id| {
                    eval_general_id(source, self.arena_expressions, id)
                })
            }
        }
    }

    fn scope_declare(
        &mut self,
        scope: &mut DeclScope,
        cond: Conditional,
        id: MaybeGeneralIdentifier,
    ) -> ControlFlow<V::Break> {
        match id {
            MaybeGeneralIdentifier::Dummy { .. } => {}
            MaybeGeneralIdentifier::Identifier(id) => {
                self.visitor.report_id_declare(id)?;

                if V::SCOPE_DECLARE {
                    let id_eval = eval_general_id(self.source, self.arena_expressions, id);
                    scope.declare(cond, id.span(), id_eval);
                }
            }
        }

        ControlFlow::Continue(())
    }
}

fn eval_general_id<'s>(
    source: &'s SourceDatabase,
    arena: &ArenaExpressions,
    id: GeneralIdentifier,
) -> EvaluatedId<&'s str> {
    match id {
        GeneralIdentifier::Simple(id) => EvaluatedId::Simple(id.str(source)),
        GeneralIdentifier::FromString(_, expr) => {
            // TODO look even further through eg. constants, values, ...
            match &arena[expr.inner] {
                ExpressionKind::StringLiteral(pieces) => {
                    // build a pattern that matches all possible substitutions
                    let mut pattern = String::new();
                    pattern.push('^');

                    for piece in pieces {
                        match piece {
                            &StringPiece::Literal(literal_span) => {
                                let literal = apply_string_literal_escapes(source.span_str(literal_span));
                                pattern.push_str(&regex::escape(literal.as_ref()));
                            }
                            StringPiece::Substitute(_expr) => {
                                pattern.push_str(".*");
                            }
                        }
                    }

                    pattern.push('$');
                    EvaluatedId::Pattern(RegexDfa::new(&pattern).unwrap())
                }
                _ => {
                    // default to a regex that matches everything
                    // TODO maybe it's even faster to have a separate enum branch for this case
                    EvaluatedId::Pattern(RegexDfa::new(".*").unwrap())
                }
            }
        }
    }
}
