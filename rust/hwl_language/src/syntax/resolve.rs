use crate::syntax::ast::{
    Arg, ArrayComprehension, ArrayLiteralElement, Assignment, Block, BlockExpression, BlockStatement,
    BlockStatementKind, ClockedBlock, ClockedBlockReset, CombinatorialBlock, CommonDeclaration, CommonDeclarationNamed,
    CommonDeclarationNamedKind, ConstBlock, ConstDeclaration, DomainKind, EnumDeclaration, EnumVariant, Expression,
    ExpressionKind, ExpressionKindIndex, ExtraItem, ExtraList, FileContent, ForStatement, FunctionDeclaration,
    GeneralIdentifier, Identifier, IfCondBlockPair, IfStatement, ImportEntry, ImportFinalKind, InterfaceView, Item,
    ItemDefInterface, ItemDefModuleExternal, ItemDefModuleInternal, MatchBranch, MatchPattern, MatchStatement,
    MaybeGeneralIdentifier, MaybeIdentifier, ModuleInstance, ModulePortBlock, ModulePortInBlock, ModulePortInBlockKind,
    ModulePortItem, ModulePortSingle, ModulePortSingleKind, ModuleStatement, ModuleStatementKind, Parameter,
    Parameters, PortConnection, PortSingleKindInner, RangeLiteral, RegDeclaration, RegOutPortMarker, RegisterDelay,
    ReturnStatement, StringPiece, StructDeclaration, StructField, SyncDomain, TypeDeclaration, VariableDeclaration,
    WhileStatement, WireDeclaration, WireDeclarationKind,
};
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::SourceDatabase;
use crate::syntax::token::apply_string_literal_escapes;
use crate::util::arena::Arena;
use indexmap::IndexMap;
use regex::Regex;

#[derive(Debug, Eq, PartialEq)]
pub enum FindDefinition<S = Vec<Span>> {
    Found(S),
    PosNotOnIdentifier,
    DefinitionNotFound,
}

type FindDefinitionResult = Result<(), FindDefinition>;

// TODO implement the opposite direction, "find usages"
// TODO follow imports instead of jumping to them
// TODO we can do better: in `if(_) { a } else { a }` a is not really conditional any more
// TODO generalize this "visitor", we also want to collect all usages, find the next selection span, find folding ranges, ...
pub fn find_definition(source: &SourceDatabase, ast: &FileContent, pos: Pos) -> FindDefinition {
    let FileContent {
        span: _,
        items,
        arena_expressions,
    } = ast;

    let ctx = ResolveContext {
        pos,
        arena_expressions,
        source,
    };

    match ctx.visit_file_items(items) {
        // TODO maybe this should be an internal error?
        Ok(()) => FindDefinition::PosNotOnIdentifier,
        Err(e) => e,
    }
}

struct ResolveContext<'a> {
    pos: Pos,
    arena_expressions: &'a Arena<ExpressionKindIndex, ExpressionKind>,
    source: &'a SourceDatabase,
}

macro_rules! check_skip {
    ($ctx: expr, $span: expr) => {
        if !$span.touches_pos($ctx.pos) {
            return Ok(());
        }
    };
}

#[derive(Debug)]
struct DeclScope<'p> {
    parent: Option<&'p DeclScope<'p>>,
    content: DeclScopeContent,
}

#[derive(Debug)]
struct DeclScopeContent {
    fixed: IndexMap<String, Vec<(Span, Conditional)>>,
    patterns: Vec<(Regex, Span, Conditional)>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Conditional {
    Yes,
    No,
}

impl<'p> DeclScope<'p> {
    fn new_root() -> Self {
        Self {
            parent: None,
            content: DeclScopeContent {
                fixed: IndexMap::new(),
                patterns: vec![],
            },
        }
    }

    fn new_child(parent: &'p DeclScope<'p>) -> Self {
        Self {
            parent: Some(parent),
            content: DeclScopeContent {
                fixed: IndexMap::new(),
                patterns: vec![],
            },
        }
    }

    fn merge_conditional_child(&mut self, content: DeclScopeContent) {
        let DeclScopeContent { fixed, patterns: regex } = content;
        for (k, v) in fixed {
            for (span, _) in v {
                self.content
                    .fixed
                    .entry(k.clone())
                    .or_default()
                    .push((span, Conditional::Yes));
            }
        }
        for (k, v, _) in regex {
            self.content.patterns.push((k, v, Conditional::Yes));
        }
    }

    fn declare(&mut self, source: &SourceDatabase, id: Identifier) {
        self.content
            .fixed
            .entry(id.str(source).to_owned())
            .or_default()
            .push((id.span, Conditional::No));
    }

    fn maybe_declare(&mut self, source: &SourceDatabase, id: MaybeIdentifier) {
        match id {
            MaybeIdentifier::Dummy(_) => {}
            MaybeIdentifier::Identifier(id) => self.declare(source, id),
        }
    }

    fn declare_general(&mut self, key: Regex, span: Span) {
        self.content.patterns.push((key, span, Conditional::No));
    }

    fn find(&self, id: &str) -> FindDefinition {
        let mut curr = self;
        let mut result = vec![];

        loop {
            let mut any_certain = false;

            let DeclScopeContent { fixed, patterns } = &curr.content;
            if let Some(entries) = fixed.get(id) {
                for &(span, cond) in entries {
                    result.push(span);
                    match cond {
                        Conditional::Yes => {}
                        Conditional::No => any_certain = true,
                    }
                }
            }
            for &(ref regex, span, c) in patterns {
                if regex.is_match(id) {
                    // pattern matches are never certain, even if they're not conditional
                    let _ = c;
                    result.push(span);
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

        if result.is_empty() {
            FindDefinition::DefinitionNotFound
        } else {
            FindDefinition::Found(result)
        }
    }
}

impl ResolveContext<'_> {
    // TODO variant without early exits that finds all usages
    fn visit_file_items(&self, items: &[Item]) -> FindDefinitionResult {
        // find item containing the pos
        let item_index = match items.binary_search_by(|item| item.info().span_full.cmp_touches_pos(self.pos)) {
            Ok(index) => index,
            Err(_) => return Ok(()),
        };

        // declare all items in scope
        let mut scope_file = DeclScope::new_root();
        for item in items {
            if let Some(info) = item.info().declaration {
                scope_file.maybe_declare(self.source, info.id);
            }

            if let Item::Import(item) = item {
                let mut visit_entry = |entry: &ImportEntry| {
                    let &ImportEntry { span: _, id, as_ } = entry;
                    let result = as_.unwrap_or(MaybeIdentifier::Identifier(id));
                    scope_file.maybe_declare(self.source, result);
                };
                match &item.entry.inner {
                    ImportFinalKind::Single(entry) => {
                        visit_entry(entry);
                    }
                    ImportFinalKind::Multi(entries) => {
                        for entry in entries {
                            visit_entry(entry);
                        }
                    }
                }
            }
        }

        // visit the relevant item
        match &items[item_index] {
            Item::Import(_) => {
                // TODO implement "go to definition" for paths, entries, ...
                Ok(())
            }
            Item::CommonDeclaration(decl) => {
                // we don't need a real scope here, the declaration has already happened in the first pass
                let mut scope_dummy = DeclScope::new_child(&scope_file);
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

                let mut scope_params = DeclScope::new_child(&scope_file);
                if let Some(params) = params {
                    self.visit_parameters(&mut scope_params, params)?;
                }

                let mut scope_ports = DeclScope::new_child(&scope_params);
                self.visit_extra_list(&mut scope_ports, &ports.inner, &mut |scope_ports, port| {
                    self.visit_port_item(scope_ports, port)?
                })?;

                self.visit_block_module(&scope_ports, body)?;
                Ok(())
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

                let mut scope_params = DeclScope::new_child(&scope_file);
                if let Some(params) = params {
                    self.visit_parameters(&mut scope_params, params)?;
                }

                let mut scope_ports = DeclScope::new_child(&scope_params);
                self.visit_extra_list(&mut scope_ports, &ports.inner, &mut |scope_ports, port| {
                    self.visit_port_item(scope_ports, port)?
                })?;

                Ok(())
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

                let mut scope_params = DeclScope::new_child(&scope_file);
                if let Some(params) = params {
                    self.visit_parameters(&mut scope_params, params)?;
                }

                let mut scope_body = DeclScope::new_child(&scope_params);
                self.visit_extra_list(&mut scope_body, port_types, &mut |scope_body, &(port_name, port_ty)| {
                    self.visit_expression(scope_body, port_ty)?;
                    scope_body.declare(self.source, port_name);
                    Ok(())
                })?;

                for view in views {
                    let &InterfaceView { id, ref port_dirs } = view;
                    self.visit_extra_list(
                        &mut scope_body,
                        port_dirs,
                        &mut |scope_body, &(port_name, _port_dir)| self.visit_id_usage(scope_body, port_name),
                    )?;
                    scope_body.maybe_declare(self.source, id);
                }

                Ok(())
            }
        }
    }

    fn visit_port_item(
        &self,
        scope_ports: &mut DeclScope,
        port: &ModulePortItem,
    ) -> Result<FindDefinitionResult, FindDefinition> {
        Ok(match port {
            ModulePortItem::Single(port) => {
                let &ModulePortSingle { span: _, id, ref kind } = port;
                match kind {
                    ModulePortSingleKind::Port { direction: _, kind } => match kind {
                        PortSingleKindInner::Clock { span_clock: _ } => {}
                        &PortSingleKindInner::Normal { domain, ty } => {
                            self.visit_domain(scope_ports, domain.inner)?;
                            self.visit_expression(scope_ports, ty)?;
                        }
                    },
                    &ModulePortSingleKind::Interface {
                        span_keyword: _,
                        domain,
                        interface,
                    } => {
                        self.visit_domain(scope_ports, domain.inner)?;
                        self.visit_expression(scope_ports, interface)?;
                    }
                }
                scope_ports.declare(self.source, id);
                Ok(())
            }
            ModulePortItem::Block(block) => {
                let ModulePortBlock { span: _, domain, ports } = block;

                self.visit_domain(scope_ports, domain.inner)?;
                self.visit_extra_list(scope_ports, ports, &mut |scope_ports, port| {
                    let &ModulePortInBlock { span: _, id, ref kind } = port;
                    match *kind {
                        ModulePortInBlockKind::Port { direction: _, ty } => {
                            self.visit_expression(scope_ports, ty)?;
                        }
                        ModulePortInBlockKind::Interface {
                            span_keyword: _,
                            interface,
                        } => {
                            self.visit_expression(scope_ports, interface)?;
                        }
                    }
                    scope_ports.declare(self.source, id);
                    Ok(())
                })?;

                Ok(())
            }
        })
    }

    fn visit_domain(&self, scope: &DeclScope, domain: DomainKind<Expression>) -> FindDefinitionResult {
        match domain {
            DomainKind::Const => Ok(()),
            DomainKind::Async => Ok(()),
            DomainKind::Sync(domain) => self.visit_domain_sync(scope, domain),
        }
    }

    fn visit_domain_sync(&self, scope: &DeclScope, domain: SyncDomain<Expression>) -> FindDefinitionResult {
        let SyncDomain { clock, reset } = domain;
        self.visit_expression(scope, clock)?;
        if let Some(reset) = reset {
            self.visit_expression(scope, reset)?;
        }
        Ok(())
    }

    fn visit_common_declaration<V>(
        &self,
        scope_parent: &mut DeclScope,
        decl: &CommonDeclaration<V>,
    ) -> FindDefinitionResult {
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

                        let mut scope_params = DeclScope::new_child(scope_parent);
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

                        let mut scope_params = DeclScope::new_child(scope_parent);
                        if let Some(params) = params {
                            self.visit_parameters(&mut scope_params, params)?;
                        }
                        self.visit_extra_list(&mut scope_params, fields, &mut |_, field| {
                            let &StructField { span: _, id: _, ty } = field;
                            self.visit_expression(scope_parent, ty)?;
                            Ok(())
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

                        let mut scope_params = DeclScope::new_child(scope_parent);
                        if let Some(params) = params {
                            self.visit_parameters(&mut scope_params, params)?;
                        }

                        self.visit_extra_list(&mut scope_params, variants, &mut |scope_params, variant| {
                            let &EnumVariant {
                                span: _,
                                id: _,
                                content,
                            } = variant;
                            // TODO declare variant name
                            if let Some(content) = content {
                                self.visit_expression(scope_params, content)?;
                            }
                            Ok(())
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

                        let mut scope_params = DeclScope::new_child(scope_parent);
                        self.visit_parameters(&mut scope_params, params)?;
                        if let Some(ret_ty) = ret_ty {
                            self.visit_expression(&scope_params, ret_ty)?;
                        }
                        self.visit_block_statements(&scope_params, body)?;

                        id
                    }
                };

                scope_parent.maybe_declare(self.source, id);
            }
            CommonDeclaration::ConstBlock(block) => {
                let ConstBlock { span_keyword: _, block } = block;
                self.visit_block_statements(scope_parent, block)?;
            }
        }

        Ok(())
    }

    fn visit_parameters(&self, scope: &mut DeclScope, params: &Parameters) -> FindDefinitionResult {
        let Parameters { span: _, items } = params;

        self.visit_extra_list(scope, items, &mut |scope, param| {
            let &Parameter { id, ty, default } = param;
            self.visit_expression(scope, ty)?;
            if let Some(default) = default {
                self.visit_expression(scope, default)?;
            }
            scope.declare(self.source, id);
            Ok(())
        })
    }

    fn visit_extra_list<I>(
        &self,
        scope_parent: &mut DeclScope,
        extra: &ExtraList<I>,
        f: &mut impl FnMut(&mut DeclScope, &I) -> FindDefinitionResult,
    ) -> FindDefinitionResult {
        let ExtraList { span: _, items } = extra;

        for item in items {
            match item {
                ExtraItem::Inner(item) => f(scope_parent, item)?,
                ExtraItem::Declaration(decl) => self.visit_common_declaration(scope_parent, decl)?,
                ExtraItem::If(if_stmt) => {
                    let mut scope_inner = DeclScope::new_child(scope_parent);
                    self.visit_if_stmt(&mut scope_inner, if_stmt, &mut |s: &mut DeclScope, b| {
                        self.visit_extra_list(s, b, f)
                    })?;
                    scope_parent.merge_conditional_child(scope_inner.content);
                }
            }
        }

        Ok(())
    }

    fn visit_if_stmt<I>(
        &self,
        scope: &mut DeclScope,
        if_stmt: &IfStatement<I>,
        f: &mut impl FnMut(&mut DeclScope, &I) -> FindDefinitionResult,
    ) -> FindDefinitionResult {
        let mut visit_pair = |pair: &IfCondBlockPair<I>| {
            let &IfCondBlockPair {
                span: _,
                span_if: _,
                cond,
                ref block,
            } = pair;

            self.visit_expression(scope, cond)?;
            f(scope, block)?;

            Ok(())
        };

        let IfStatement {
            initial_if,
            else_ifs,
            final_else,
        } = if_stmt;
        visit_pair(initial_if)?;
        for else_if in else_ifs {
            visit_pair(else_if)?;
        }
        if let Some(final_else) = final_else {
            f(scope, final_else)?;
        }
        Ok(())
    }

    fn visit_for_stmt<S>(
        &self,
        scope: &DeclScope,
        stmt: &ForStatement<S>,
        f: impl FnOnce(&DeclScope, &Block<S>) -> FindDefinitionResult,
    ) -> FindDefinitionResult {
        let &ForStatement {
            span_keyword: _,
            index,
            index_ty,
            iter,
            ref body,
        } = stmt;
        self.visit_expression(scope, iter)?;

        if let Some(index_ty) = index_ty {
            self.visit_expression(scope, index_ty)?;
        }

        let mut scope_inner = DeclScope::new_child(scope);
        scope_inner.maybe_declare(self.source, index);

        f(&scope_inner, body)?;

        Ok(())
    }

    fn visit_block_statements(&self, scope_parents: &DeclScope, block: &Block<BlockStatement>) -> FindDefinitionResult {
        let Block { span, statements } = block;

        // declarations inside a block can't leak outside, so we can skip here
        check_skip!(self, span);

        let mut scope = DeclScope::new_child(scope_parents);
        for stmt in statements {
            self.visit_statement(&mut scope, stmt)?;
        }

        Ok(())
    }

    fn visit_statement(&self, scope: &mut DeclScope, stmt: &BlockStatement) -> FindDefinitionResult {
        let stmt_span = stmt.span;
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
                scope.maybe_declare(self.source, id);
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
                self.visit_if_stmt(scope, stmt, &mut |s, b| self.visit_block_statements(s, b))?;
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

                    let mut scope_inner = DeclScope::new_child(scope);
                    match &pattern.inner {
                        MatchPattern::Dummy => {}
                        &MatchPattern::Val(id) => {
                            scope_inner.declare(self.source, id);
                        }
                        &MatchPattern::Equal(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                        &MatchPattern::In(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                        &MatchPattern::EnumVariant(_variant, id) => {
                            if let Some(id) = id {
                                scope_inner.maybe_declare(self.source, id);
                            }
                        }
                    }

                    self.visit_block_statements(&scope_inner, block)?;
                }
            }
            BlockStatementKind::For(stmt) => {
                check_skip!(self, stmt_span);
                self.visit_for_stmt(scope, stmt, |s, b| self.visit_block_statements(s, b))?;
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
            BlockStatementKind::Break(_span) | BlockStatementKind::Continue(_span) => {}
        }

        Ok(())
    }

    fn visit_block_module(&self, scope_parent: &DeclScope, block: &Block<ModuleStatement>) -> FindDefinitionResult {
        let Block { span, statements } = block;
        check_skip!(self, span);

        // match the two-pass system from the main compiler
        let mut scope = DeclScope::new_child(scope_parent);
        for stmt in statements {
            match &stmt.inner {
                ModuleStatementKind::CommonDeclaration(decl) => {
                    self.visit_common_declaration(&mut scope, decl)?;
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let &RegDeclaration { id, sync, ty, init } = decl;

                    if let Some(sync) = sync {
                        self.visit_domain_sync(&scope, sync.inner)?;
                    }
                    self.visit_expression(&scope, ty)?;
                    self.visit_expression(&scope, init)?;
                    self.visit_and_declare_maybe_general(&mut scope, id)?;
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let &WireDeclaration { id, ref kind } = decl;

                    match *kind {
                        WireDeclarationKind::Clock {
                            span_clock: _,
                            span_assign_and_value,
                        } => {
                            if let Some((_, value)) = span_assign_and_value {
                                self.visit_expression(&scope, value)?;
                            }
                        }
                        WireDeclarationKind::NormalWithValue {
                            domain,
                            ty,
                            span_assign: _,
                            value,
                        } => {
                            if let Some(domain) = domain {
                                self.visit_domain(&scope, domain.inner)?;
                            }
                            if let Some(ty) = ty {
                                self.visit_expression(&scope, ty)?;
                            }
                            self.visit_expression(&scope, value)?;
                        }
                        WireDeclarationKind::NormalWithoutValue { domain, ty } => {
                            if let Some(domain) = domain {
                                self.visit_domain(&scope, domain.inner)?;
                            }
                            self.visit_expression(&scope, ty)?;
                        }
                    }

                    self.visit_and_declare_maybe_general(&mut scope, id)?;
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    let &RegOutPortMarker { id, init } = decl;
                    self.visit_expression(&scope, init)?;
                    self.visit_id_usage(&scope, id)?;
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
                    self.visit_block_module(&scope, block)?;
                }
                ModuleStatementKind::If(stmt) => {
                    self.visit_if_stmt(&mut scope, stmt, &mut |s, b| self.visit_block_module(s, b))?;
                }
                ModuleStatementKind::For(stmt) => {
                    self.visit_for_stmt(&scope, stmt, |s, b| self.visit_block_module(s, b))?
                }
                ModuleStatementKind::CombinatorialBlock(stmt) => {
                    let CombinatorialBlock { span_keyword: _, block } = stmt;
                    self.visit_block_statements(&scope, block)?;
                }
                ModuleStatementKind::ClockedBlock(stmt) => {
                    let &ClockedBlock {
                        span_keyword: _,
                        span_domain: _,
                        clock,
                        reset,
                        ref block,
                    } = stmt;
                    self.visit_expression(&scope, clock)?;
                    if let Some(reset) = reset {
                        let ClockedBlockReset { kind: _, signal } = reset.inner;
                        self.visit_expression(&scope, signal)?;
                    }
                    self.visit_block_statements(&scope, block)?;
                }
                ModuleStatementKind::Instance(stmt) => {
                    let &ModuleInstance {
                        name: _,
                        span_keyword: _,
                        module,
                        ref port_connections,
                    } = stmt;

                    self.visit_expression(&scope, module)?;

                    check_skip!(self, port_connections.span);
                    for conn in &port_connections.inner {
                        let &PortConnection { id, expr } = &conn.inner;
                        // TODO try resolving port name, needs type info
                        let _ = id;
                        self.visit_expression(&scope, expr)?;
                    }
                }

                // declarations, already handled in the first pass
                ModuleStatementKind::CommonDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
            }
        }

        Ok(())
    }

    fn visit_array_literal_element(
        &self,
        scope: &DeclScope,
        elem: ArrayLiteralElement<Expression>,
    ) -> FindDefinitionResult {
        match elem {
            ArrayLiteralElement::Single(elem) => self.visit_expression(scope, elem),
            ArrayLiteralElement::Spread(_, elem) => self.visit_expression(scope, elem),
        }
    }

    fn visit_expression(&self, scope: &DeclScope, expr: Expression) -> FindDefinitionResult {
        // expressions can't contain declarations that leak outside, so we can skip here
        check_skip!(self, expr.span);

        match &self.arena_expressions[expr.inner] {
            &ExpressionKind::Wrapped(inner) => {
                self.visit_expression(scope, inner)?;
            }
            ExpressionKind::Block(block) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = block;
                let mut scope_inner = DeclScope::new_child(scope);
                for stmt in statements {
                    self.visit_statement(&mut scope_inner, stmt)?
                }
                self.visit_expression(&scope_inner, expression)?;
            }
            &ExpressionKind::Id(id) => {
                match id {
                    GeneralIdentifier::Simple(id) => {
                        self.visit_id_usage(scope, id)?;
                    }
                    GeneralIdentifier::FromString(span, expr) => {
                        self.visit_expression(scope, expr)?;

                        // TODO support find on general ids?
                        //   For now this should find all ids in any scope, which is not very useful.
                        //   Maybe with regexes this becomes slightly more useful.
                        let _ = span;
                    }
                }
            }
            ExpressionKind::StringLiteral(pieces) => {
                for piece in pieces {
                    match piece {
                        StringPiece::Literal(_span) => {}
                        &StringPiece::Substitute(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                    }
                }
            }
            ExpressionKind::ArrayLiteral(elems) => {
                for &elem in elems {
                    self.visit_array_literal_element(scope, elem)?;
                }
            }
            ExpressionKind::TupleLiteral(elems) => {
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
                RangeLiteral::Length { op_span: _, start, len } => {
                    self.visit_expression(scope, start)?;
                    self.visit_expression(scope, len)?;
                }
            },
            ExpressionKind::ArrayComprehension(expr) => {
                let &ArrayComprehension {
                    body,
                    index,
                    span_keyword: _,
                    iter,
                } = expr;
                self.visit_expression(scope, iter)?;

                let mut scope_inner = DeclScope::new_child(scope);
                scope_inner.maybe_declare(self.source, index);

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
            &ExpressionKind::DotIdIndex(base, _) => {
                // TODO try resolving index, needs type info
                self.visit_expression(scope, base)?;
            }
            &ExpressionKind::DotIntIndex(base, _) => {
                // TODO try resolving index, needs type info
                self.visit_expression(scope, base)?;
            }
            &ExpressionKind::Call(target, ref args) => {
                self.visit_expression(scope, target)?;
                for arg in &args.inner {
                    let &Arg {
                        span: _,
                        ref name,
                        value,
                    } = arg;
                    // TODO try resolving name, needs type info
                    let _ = name;
                    self.visit_expression(scope, value)?;
                }
            }
            ExpressionKind::Builtin(args) => {
                for &arg in &args.inner {
                    self.visit_expression(scope, arg)?;
                }
            }
            &ExpressionKind::UnsafeValueWithDomain(value, domain) => {
                self.visit_expression(scope, value)?;
                self.visit_domain(scope, domain.inner)?;
            }
            ExpressionKind::RegisterDelay(expr) => {
                let &RegisterDelay {
                    span_keyword: _,
                    value,
                    init,
                } = expr;
                self.visit_expression(scope, value)?;
                self.visit_expression(scope, init)?;
            }

            ExpressionKind::Dummy
            | ExpressionKind::Undefined
            | ExpressionKind::Type
            | ExpressionKind::TypeFunction
            | ExpressionKind::IntLiteral(_)
            | ExpressionKind::BoolLiteral(_) => {}
        }

        Ok(())
    }

    fn visit_and_declare_general(&self, scope: &mut DeclScope, id: GeneralIdentifier) -> FindDefinitionResult {
        match id {
            GeneralIdentifier::Simple(id) => {
                scope.declare(self.source, id);
            }
            GeneralIdentifier::FromString(span, expr) => {
                self.visit_expression(scope, expr)?;

                // build a pattern that matches all possible substitutions
                let pattern = match &self.arena_expressions[expr.inner] {
                    ExpressionKind::StringLiteral(pieces) => {
                        let mut pattern = String::new();
                        pattern.push('^');

                        for piece in pieces {
                            match piece {
                                &StringPiece::Literal(s) => {
                                    let literal = apply_string_literal_escapes(self.source.span_str(s));
                                    pattern.push_str(&regex::escape(literal.as_ref()));
                                }
                                StringPiece::Substitute(_expr) => {
                                    pattern.push_str(".*");
                                }
                            }
                        }

                        pattern.push('$');
                        Regex::new(&pattern).unwrap()
                    }

                    // default to a regex that matches everything
                    _ => Regex::new("").unwrap(),
                };

                scope.declare_general(pattern, span);
            }
        }
        Ok(())
    }

    fn visit_and_declare_maybe_general(
        &self,
        scope: &mut DeclScope,
        id: MaybeGeneralIdentifier,
    ) -> FindDefinitionResult {
        match id {
            MaybeGeneralIdentifier::Dummy(_) => Ok(()),
            MaybeGeneralIdentifier::Identifier(id) => self.visit_and_declare_general(scope, id),
        }
    }

    fn visit_id_usage(&self, scope: &DeclScope, id: Identifier) -> FindDefinitionResult {
        // TODO support visiting general IDs too
        check_skip!(self, id.span);
        Err(scope.find(id.str(self.source)))
    }
}
