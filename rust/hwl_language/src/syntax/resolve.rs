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
    ReturnStatement, Spanned, StringPiece, StructDeclaration, StructField, SyncDomain, TypeDeclaration,
    VariableDeclaration, Visibility, WhileStatement, WireDeclaration, WireDeclarationDomainTyKind, WireDeclarationKind,
};
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::SourceDatabase;
use crate::syntax::token::apply_string_literal_escapes;
use crate::util::arena::Arena;
use indexmap::IndexMap;
use regex_automata::dfa::Automaton;
use std::collections::HashSet;

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
// TODO maybe this should be moved to the LSP, the compiler itself really doesn't need this
// TODO use the real Scope for the file root, to reduce duplication and get a guaranteed match
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
    simple: IndexMap<String, Vec<(Span, Conditional)>>,
    pattern: Vec<(RegexDfa, Span, Conditional)>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Conditional {
    Yes,
    No,
}

#[derive(Debug)]
enum EvaluatedId<S> {
    Simple(S),
    Pattern(RegexDfa),
}

#[derive(Debug)]
struct RegexDfa {
    dfa: regex_automata::dfa::sparse::DFA<Vec<u8>>,
}

impl EvaluatedId<&str> {
    pub fn into_owned(self) -> EvaluatedId<String> {
        match self {
            EvaluatedId::Simple(s) => EvaluatedId::Simple(s.to_owned()),
            EvaluatedId::Pattern(regex) => EvaluatedId::Pattern(regex),
        }
    }
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

    fn new_child(parent: &'p DeclScope<'p>) -> Self {
        Self {
            parent: Some(parent),
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

    fn declare_impl(&mut self, cond: Conditional, id: Spanned<EvaluatedId<String>>) {
        match id.inner {
            EvaluatedId::Simple(id_inner) => {
                self.content.simple.entry(id_inner).or_default().push((id.span, cond));
            }
            EvaluatedId::Pattern(pattern) => self.content.pattern.push((pattern, id.span, cond)),
        }
    }

    fn declare(&mut self, source: &SourceDatabase, cond: Conditional, id: Identifier) {
        let id_eval = EvaluatedId::Simple(id.str(source).to_owned());
        self.declare_impl(cond, Spanned::new(id.span, id_eval));
    }

    fn maybe_declare(&mut self, source: &SourceDatabase, cond: Conditional, id: MaybeIdentifier) {
        match id {
            MaybeIdentifier::Dummy(_) => {}
            MaybeIdentifier::Identifier(id) => {
                self.declare(source, cond, id);
            }
        }
    }

    fn find(&self, id: &EvaluatedId<&str>) -> FindDefinition {
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

        if result.is_empty() {
            FindDefinition::DefinitionNotFound
        } else {
            FindDefinition::Found(result)
        }
    }
}

impl RegexDfa {
    pub fn new(pattern: &str) -> Result<Self, regex_automata::dfa::dense::BuildError> {
        let dfa = regex_automata::dfa::sparse::DFA::new(pattern)?;
        Ok(Self { dfa })
    }

    pub fn could_match_str(&self, input: &str) -> bool {
        self.dfa
            .try_search_fwd(&regex_automata::Input::new(input))
            .unwrap()
            .is_some()
    }

    pub fn could_match_pattern(&self, other: &RegexDfa) -> bool {
        // regex intersection, based on https://users.rust-lang.org/t/detect-regex-conflict/57184/13
        let dfa_0 = &self.dfa;
        let dfa_1 = &other.dfa;

        let start_config = regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes);
        let start_0 = dfa_0.start_state(&start_config).unwrap();
        let start_1 = dfa_1.start_state(&start_config).unwrap();

        if dfa_0.is_match_state(start_0) && dfa_1.is_match_state(start_1) {
            return true;
        }

        let mut visited_states = HashSet::new();
        let mut to_process = vec![(start_0, start_1)];
        visited_states.insert((start_0, start_1));

        while let Some((curr_0, curr_1)) = to_process.pop() {
            let mut handle_next = |next_0, next_1| {
                if dfa_0.is_match_state(next_0) && dfa_1.is_match_state(next_1) {
                    return true;
                }
                if visited_states.insert((next_0, next_1)) {
                    to_process.push((next_0, next_1));
                }
                false
            };

            // TODO is there a good way to only iterate over the bytes that appear in either pattern?
            for input in 0..u8::MAX {
                let next_0 = dfa_0.next_state(curr_0, input);
                let next_1 = dfa_1.next_state(curr_1, input);
                if handle_next(next_0, next_1) {
                    return true;
                }
            }

            let next_0 = dfa_0.next_eoi_state(curr_0);
            let next_1 = dfa_1.next_eoi_state(curr_1);
            if handle_next(next_0, next_1) {
                return true;
            }
        }

        false
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
                scope_file.maybe_declare(self.source, Conditional::No, info.id);
            }

            if let Item::Import(item) = item {
                let mut visit_entry = |entry: &ImportEntry| {
                    let &ImportEntry { span: _, id, as_ } = entry;
                    let result = as_.unwrap_or(MaybeIdentifier::Identifier(id));
                    scope_file.maybe_declare(self.source, Conditional::No, result);
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

                let mut scope_body = DeclScope::new_child(&scope_ports);
                self.collect_module_body_pub_declarations(&mut scope_body, Conditional::Yes, body);

                self.visit_block_module_inner(&mut scope_body, body)?;
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
                    scope_body.declare(self.source, Conditional::No, port_name);
                    Ok(())
                })?;

                for view in views {
                    let &InterfaceView { id, ref port_dirs } = view;
                    self.visit_extra_list(
                        &mut scope_body,
                        port_dirs,
                        &mut |scope_body, &(port_name, _port_dir)| self.visit_id_usage(scope_body, port_name),
                    )?;
                    scope_body.maybe_declare(self.source, Conditional::No, id);
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
                scope_ports.declare(self.source, Conditional::No, id);
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
                    scope_ports.declare(self.source, Conditional::No, id);
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

                scope_parent.maybe_declare(self.source, Conditional::No, id);
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
            scope.declare(self.source, Conditional::No, id);
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
        scope_inner.maybe_declare(self.source, Conditional::No, index);

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
                scope.maybe_declare(self.source, Conditional::No, id);
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
                            scope_inner.declare(self.source, Conditional::No, id);
                        }
                        &MatchPattern::Equal(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                        &MatchPattern::In(expr) => {
                            self.visit_expression(scope, expr)?;
                        }
                        &MatchPattern::EnumVariant(_variant, id) => {
                            if let Some(id) = id {
                                scope_inner.maybe_declare(self.source, Conditional::No, id);
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

    fn collect_module_body_pub_declarations(
        &self,
        scope_body: &mut DeclScope,
        cond: Conditional,
        curr_block: &Block<ModuleStatement>,
    ) {
        for stmt in &curr_block.statements {
            match &stmt.inner {
                // control flow
                ModuleStatementKind::Block(block) => {
                    self.collect_module_body_pub_declarations(scope_body, cond, block);
                }
                ModuleStatementKind::If(if_stmt) => {
                    let IfStatement {
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
                    self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, block);

                    for else_if in else_ifs {
                        let IfCondBlockPair {
                            span: _,
                            span_if: _,
                            cond: _,
                            block,
                        } = else_if;
                        self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, block);
                    }

                    if let Some(final_else) = final_else {
                        self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, final_else);
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

                    self.collect_module_body_pub_declarations(scope_body, Conditional::Yes, body);
                }

                // potentially public declarations
                ModuleStatementKind::RegDeclaration(decl) => {
                    let &RegDeclaration {
                        vis,
                        id,
                        sync: _,
                        ty: _,
                        init: _,
                    } = decl;
                    match vis {
                        Visibility::Public(_) => self.declare_maybe_general(scope_body, cond, id),
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
                        Visibility::Public(_) => self.declare_maybe_general(scope_body, cond, id),
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
    }

    fn visit_block_module(&self, scope_parent: &DeclScope, block: &Block<ModuleStatement>) -> FindDefinitionResult {
        let mut scope = DeclScope::new_child(scope_parent);
        self.visit_block_module_inner(&mut scope, block)
    }

    fn visit_block_module_inner(&self, scope: &mut DeclScope, block: &Block<ModuleStatement>) -> FindDefinitionResult {
        let Block { span, statements } = block;
        check_skip!(self, span);

        // match the two-pass system from the main compiler
        for stmt in statements {
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
                        Visibility::Public(_) => {}
                        Visibility::Private => self.declare_maybe_general(scope, Conditional::No, id),
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
                                        self.visit_domain(scope, domain.inner)?;
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
                                self.visit_domain(scope, domain.inner)?;
                            }
                            self.visit_expression(scope, interface)?;
                        }
                    }

                    self.visit_maybe_general(scope, id)?;
                    match vis {
                        Visibility::Public(_) => {}
                        Visibility::Private => self.declare_maybe_general(scope, Conditional::No, id),
                    }
                }
                ModuleStatementKind::RegOutPortMarker(decl) => {
                    let &RegOutPortMarker { id, init } = decl;
                    self.visit_expression(scope, init)?;
                    self.visit_id_usage(scope, id)?;
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
                    self.visit_if_stmt(scope, stmt, &mut |s, b| self.visit_block_module(s, b))?;
                }
                ModuleStatementKind::For(stmt) => {
                    self.visit_for_stmt(scope, stmt, |s, b| self.visit_block_module(s, b))?
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
                        self.visit_expression(scope, expr)?;
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
                self.visit_general_id_usage(scope, id)?;
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
                scope_inner.maybe_declare(self.source, Conditional::No, index);

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

    fn visit_maybe_general(&self, scope: &DeclScope, id: MaybeGeneralIdentifier) -> FindDefinitionResult {
        match id {
            MaybeGeneralIdentifier::Dummy(_span) => {}
            MaybeGeneralIdentifier::Identifier(id) => match id {
                GeneralIdentifier::Simple(_id) => {}
                GeneralIdentifier::FromString(_span, expr) => {
                    self.visit_expression(scope, expr)?;
                }
            },
        }

        Ok(())
    }

    fn declare_maybe_general(&self, scope: &mut DeclScope, cond: Conditional, id: MaybeGeneralIdentifier) {
        match id {
            MaybeGeneralIdentifier::Dummy(_) => {}
            MaybeGeneralIdentifier::Identifier(id) => {
                let id = self.eval_general(id);
                scope.declare_impl(cond, id.map_inner(|id| id.into_owned()));
            }
        }
    }

    fn visit_id_usage(&self, scope: &DeclScope, id: Identifier) -> FindDefinitionResult {
        check_skip!(self, id.span);
        Err(scope.find(&EvaluatedId::Simple(id.str(self.source))))
    }

    fn visit_general_id_usage(&self, scope: &DeclScope, id: GeneralIdentifier) -> FindDefinitionResult {
        // visit inner expressions first
        match id {
            GeneralIdentifier::Simple(_id) => {}
            GeneralIdentifier::FromString(_span, expr) => {
                self.visit_expression(scope, expr)?;
            }
        }

        // find the id itself
        check_skip!(self, id.span());
        Err(scope.find(&self.eval_general(id).inner))
    }

    fn eval_general(&self, id: GeneralIdentifier) -> Spanned<EvaluatedId<&str>> {
        let eval = match id {
            GeneralIdentifier::Simple(id) => EvaluatedId::Simple(id.str(self.source)),
            GeneralIdentifier::FromString(_, expr) => {
                // TODO look even further through eg. constants, values, ...
                match &self.arena_expressions[expr.inner] {
                    ExpressionKind::StringLiteral(pieces) => {
                        // build a pattern that matches all possible substitutions
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
                        EvaluatedId::Pattern(RegexDfa::new(&pattern).unwrap())
                    }
                    _ => {
                        // default to a regex that matches everything
                        // TODO maybe it's even faster to have a separate enum branch for this case
                        EvaluatedId::Pattern(RegexDfa::new(".*").unwrap())
                    }
                }
            }
        };

        Spanned::new(id.span(), eval)
    }
}

#[cfg(test)]
mod test {
    use crate::syntax::resolve::RegexDfa;
    use itertools::Itertools;

    // tests based on https://users.rust-lang.org/t/detect-regex-conflict/57184/13
    fn regex_overlap(a: &str, b: &str) -> bool {
        let a = RegexDfa::new(a).unwrap();
        let b = RegexDfa::new(b).unwrap();
        a.could_match_pattern(&b)
    }

    #[test]
    fn overlapping_regexes() {
        let pattern1 = r"[a-zA-Z]+";
        let pattern2 = r"[a-z]+";
        assert!(regex_overlap(pattern1, pattern2));
        let pattern1 = r"a*";
        let pattern2 = r"b*";
        assert!(regex_overlap(pattern1, pattern2));
        let pattern1 = r"a*bba+";
        let pattern2 = r"b*aaab+a";
        assert!(regex_overlap(pattern1, pattern2));
        let pattern1 = r" ";
        let pattern2 = r"\s";
        assert!(regex_overlap(pattern1, pattern2));
        let pattern1 = r"[A-Z]+";
        let pattern2 = r"[a-z]+";
        assert!(!regex_overlap(pattern1, pattern2));
        let pattern1 = r"a";
        let pattern2 = r"b";
        assert!(!regex_overlap(pattern1, pattern2));
        let pattern1 = r"a*bba+";
        let pattern2 = r"b*aaabbb+a";
        assert!(!regex_overlap(pattern1, pattern2));
        let pattern1 = r"\s+";
        let pattern2 = r"a+";
        assert!(!regex_overlap(pattern1, pattern2));
    }

    #[test]
    fn all_overlapping_regexes() {
        let patterns = [
            r"[a-zA-Z]+",
            r"[a-z]+",
            r"a*",
            r"b*",
            r"a*bba+",
            r"b*aaab+a",
            r" ",
            r"\s",
            r"[A-Z]+",
            r"[a-z]+",
            r"a",
            r"b",
            r"a*bba+",
            r"b*aaabbb+a",
            r"\s+",
            r"a+",
        ];

        let patterns = patterns.iter().map(|&s| RegexDfa::new(s).unwrap()).collect_vec();

        let mut match_count = 0;
        for a in &patterns {
            for b in &patterns {
                if a.could_match_pattern(b) {
                    match_count += 1;
                }
            }
        }
        assert_eq!(match_count, 102);
    }

    #[test]
    fn test_basic() {
        let a = RegexDfa::new("^a$").unwrap();
        assert!(a.could_match_str("a"));
        assert!(!a.could_match_str("ab"));

        let empty = RegexDfa::new("").unwrap();
        assert!(empty.could_match_str(""));
        assert!(empty.could_match_str("abc"));

        let start = RegexDfa::new("^a.*b$").unwrap();
        assert!(start.could_match_str("ab"));
        assert!(!start.could_match_str("abc"));
        assert!(start.could_match_str("acb"));
        assert!(!start.could_match_str("acbd"));
    }

    #[test]
    fn test_self() {
        let a = RegexDfa::new("^a$").unwrap();
        assert!(a.could_match_pattern(&a));
    }
}
