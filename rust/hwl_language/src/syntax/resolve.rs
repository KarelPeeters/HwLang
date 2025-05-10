use crate::syntax::ast::{
    Arg, ArrayComprehension, ArrayLiteralElement, Assignment, Block, BlockExpression, BlockStatement,
    BlockStatementKind, ClockedBlock, ClockedBlockReset, CombinatorialBlock, CommonDeclaration, CommonDeclarationNamed,
    CommonDeclarationNamedKind, ConstBlock, ConstDeclaration, DomainKind, EnumDeclaration, EnumVariant, Expression,
    ExpressionKind, ExtraItem, ExtraList, FileContent, ForStatement, FunctionDeclaration, Identifier, IfCondBlockPair,
    IfStatement, ImportEntry, ImportFinalKind, InterfaceView, Item, ItemDefInterface, ItemDefModule, MatchBranch,
    MatchPattern, MatchStatement, MaybeIdentifier, ModuleInstance, ModulePortBlock, ModulePortInBlock,
    ModulePortInBlockKind, ModulePortItem, ModulePortSingle, ModulePortSingleKind, ModuleStatement,
    ModuleStatementKind, Parameter, Parameters, PortConnection, PortSingleKindInner, RangeLiteral, RegDeclaration,
    RegOutPortMarker, RegisterDelay, ReturnStatement, StructDeclaration, StructField, SyncDomain, TypeDeclaration,
    VariableDeclaration, WhileStatement, WireDeclaration, WireDeclarationKind,
};
use crate::syntax::pos::{Pos, Span};
use indexmap::IndexMap;
use std::borrow::Borrow;

// TODO implement the opposite direction, "find usages"
// TODO follow imports instead of jumping to them
// TODO we can do better: in `if(_) { a } else { a }` a is not really conditional any more

#[derive(Debug, Eq, PartialEq)]
pub enum FindDefinition<S = Vec<Span>> {
    Found(S),
    PosNotOnIdentifier,
    DefinitionNotFound,
}

type FindDefinitionResult = Result<(), FindDefinition>;

macro_rules! check_skip {
    ($span: expr, $pos: expr) => {
        if !$span.touches_pos($pos) {
            return Ok(());
        }
    };
}

#[derive(Debug)]
struct DeclScope<'p> {
    parent: Option<&'p DeclScope<'p>>,
    map: ScopeMap,
}

type ScopeMap = IndexMap<String, Vec<(Span, Conditional)>>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Conditional {
    Yes,
    No,
}

impl<'p> DeclScope<'p> {
    fn new_root() -> Self {
        Self {
            parent: None,
            map: IndexMap::new(),
        }
    }

    fn new_child(parent: &'p DeclScope<'p>) -> Self {
        Self {
            parent: Some(parent),
            map: IndexMap::new(),
        }
    }

    fn merge_conditional_child(&mut self, map: ScopeMap) {
        for (k, v) in map {
            for (span, _) in v {
                self.map.entry(k.clone()).or_default().push((span, Conditional::Yes));
            }
        }
    }

    fn declare(&mut self, id: &Identifier) {
        self.map
            .entry(id.string.clone())
            .or_default()
            .push((id.span, Conditional::No));
    }

    fn maybe_declare(&mut self, id: MaybeIdentifier<&Identifier>) {
        match id {
            MaybeIdentifier::Dummy(_) => {}
            MaybeIdentifier::Identifier(id) => self.declare(id),
        }
    }

    fn find(&self, id: &Identifier) -> FindDefinition {
        let mut curr = self;
        let mut result = vec![];

        loop {
            let mut any_certain = false;

            if let Some(entries) = curr.map.get(&id.string) {
                for &(span, cond) in entries {
                    result.push(span);
                    match cond {
                        Conditional::Yes => {}
                        Conditional::No => any_certain = true,
                    }
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

pub fn find_definition(ast: &FileContent, pos: Pos) -> FindDefinition {
    match visit_file(ast, pos) {
        // TODO maybe this should be an internal error?
        Ok(()) => FindDefinition::PosNotOnIdentifier,
        Err(e) => e,
    }
}

// TODO variant without early exits that finds all usages
fn visit_file(ast: &FileContent, pos: Pos) -> FindDefinitionResult {
    let FileContent { span, items } = ast;
    check_skip!(span, pos);

    // find item containing the pos
    let item_index = match items.binary_search_by(|item| item.info().span_full.cmp_touches_pos(pos)) {
        Ok(index) => index,
        Err(_) => return Ok(()),
    };

    // declare all items in scope
    let mut scope_file = DeclScope::new_root();
    for item in items {
        if let Some(info) = item.info().declaration {
            scope_file.maybe_declare(info.id);
        }

        if let Item::Import(item) = item {
            let mut visit_entry = |entry: &ImportEntry| {
                let ImportEntry { span: _, id, as_ } = entry;
                let result = as_
                    .as_ref()
                    .map(MaybeIdentifier::as_ref)
                    .unwrap_or(MaybeIdentifier::Identifier(id));
                scope_file.maybe_declare(result);
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
            visit_common_declaration(&mut scope_dummy, pos, &decl.inner)
        }
        Item::Module(decl) => {
            let ItemDefModule {
                span: _,
                vis: _,
                id: _,
                params,
                ports,
                body,
            } = decl;

            let mut scope_params = DeclScope::new_child(&scope_file);
            if let Some(params) = params {
                visit_parameters(&mut scope_params, pos, params)?;
            }

            let mut scope_ports = DeclScope::new_child(&scope_params);
            visit_extra_list(&mut scope_ports, pos, &ports.inner, &mut |scope_ports, port| {
                visit_port_item(scope_ports, pos, port)?
            })?;

            visit_block_module(&scope_ports, pos, body)?;
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
                visit_parameters(&mut scope_params, pos, params)?;
            }

            let mut scope_body = DeclScope::new_child(&scope_params);
            visit_extra_list(
                &mut scope_body,
                pos,
                port_types,
                &mut |scope_body, (port_name, port_ty)| {
                    visit_expression(scope_body, pos, port_ty)?;
                    scope_body.declare(port_name);
                    Ok(())
                },
            )?;

            for view in views {
                let InterfaceView { id, port_dirs } = view;
                visit_extra_list(
                    &mut scope_body,
                    pos,
                    port_dirs,
                    &mut |scope_body, (port_name, _port_dir)| visit_id_usage(scope_body, pos, port_name),
                )?;
                scope_body.maybe_declare(id.as_ref());
            }

            Ok(())
        }
    }
}

fn visit_port_item(
    scope_ports: &mut DeclScope,
    pos: Pos,
    port: &ModulePortItem,
) -> Result<FindDefinitionResult, FindDefinition> {
    Ok(match port {
        ModulePortItem::Single(port) => {
            let ModulePortSingle { span: _, id, kind } = port;
            match kind {
                ModulePortSingleKind::Port { direction: _, kind } => match kind {
                    PortSingleKindInner::Clock { span_clock: _ } => {}
                    PortSingleKindInner::Normal { domain, ty } => {
                        visit_domain(scope_ports, pos, &domain.inner)?;
                        visit_expression(scope_ports, pos, ty)?;
                    }
                },
                ModulePortSingleKind::Interface {
                    span_keyword: _,
                    domain,
                    interface,
                } => {
                    visit_domain(scope_ports, pos, &domain.inner)?;
                    visit_expression(scope_ports, pos, interface)?;
                }
            }
            scope_ports.declare(id);
            Ok(())
        }
        ModulePortItem::Block(block) => {
            let ModulePortBlock { span: _, domain, ports } = block;

            visit_domain(scope_ports, pos, &domain.inner)?;
            visit_extra_list(scope_ports, pos, ports, &mut |scope_ports, port| {
                let ModulePortInBlock { span: _, id, kind } = port;
                match kind {
                    ModulePortInBlockKind::Port { direction: _, ty } => {
                        visit_expression(scope_ports, pos, ty)?;
                    }
                    ModulePortInBlockKind::Interface {
                        span_keyword: _,
                        interface,
                    } => {
                        visit_expression(scope_ports, pos, interface)?;
                    }
                }
                scope_ports.declare(id);
                Ok(())
            })?;

            Ok(())
        }
    })
}

fn visit_domain(scope: &DeclScope, pos: Pos, domain: &DomainKind<Box<Expression>>) -> FindDefinitionResult {
    match domain {
        DomainKind::Const => Ok(()),
        DomainKind::Async => Ok(()),
        DomainKind::Sync(domain) => visit_domain_sync(scope, pos, domain),
    }
}

fn visit_domain_sync(scope: &DeclScope, pos: Pos, domain: &SyncDomain<Box<Expression>>) -> FindDefinitionResult {
    let SyncDomain { clock, reset } = domain;
    visit_expression(scope, pos, clock)?;
    if let Some(reset) = reset {
        visit_expression(scope, pos, reset)?;
    }
    Ok(())
}

fn visit_common_declaration<'a, V>(
    scope_parent: &mut DeclScope,
    pos: Pos,
    decl: &'a CommonDeclaration<V>,
) -> FindDefinitionResult {
    match decl {
        CommonDeclaration::Named(decl) => {
            let CommonDeclarationNamed { vis: _, kind } = decl;
            let id: MaybeIdentifier<&'a Identifier> = match kind {
                CommonDeclarationNamedKind::Type(decl) => {
                    let TypeDeclaration {
                        span: _,
                        id,
                        params,
                        body,
                    } = decl;

                    let mut scope_params = DeclScope::new_child(scope_parent);
                    if let Some(params) = params {
                        visit_parameters(&mut scope_params, pos, params)?;
                    }
                    visit_expression(&scope_params, pos, body)?;

                    id.as_ref()
                }
                CommonDeclarationNamedKind::Const(decl) => {
                    let ConstDeclaration { span: _, id, ty, value } = decl;

                    if let Some(ty) = ty {
                        visit_expression(scope_parent, pos, ty)?;
                    }
                    visit_expression(scope_parent, pos, value)?;

                    id.as_ref()
                }
                CommonDeclarationNamedKind::Struct(decl) => {
                    let StructDeclaration {
                        span: _,
                        span_body: _,
                        id,
                        params,
                        fields,
                    } = decl;

                    let mut scope_params = DeclScope::new_child(scope_parent);
                    if let Some(params) = params {
                        visit_parameters(&mut scope_params, pos, params)?;
                    }
                    visit_extra_list(&mut scope_params, pos, fields, &mut |_, field| {
                        let StructField { span: _, id: _, ty } = field;
                        visit_expression(scope_parent, pos, ty)?;
                        Ok(())
                    })?;

                    id.as_ref()
                }
                CommonDeclarationNamedKind::Enum(decl) => {
                    let EnumDeclaration {
                        span: _,
                        id,
                        params,
                        variants,
                    } = decl;

                    let mut scope_params = DeclScope::new_child(scope_parent);
                    if let Some(params) = params {
                        visit_parameters(&mut scope_params, pos, params)?;
                    }

                    visit_extra_list(&mut scope_params, pos, variants, &mut |scope_params, variant| {
                        let EnumVariant {
                            span: _,
                            id: _,
                            content,
                        } = variant;
                        if let Some(content) = content {
                            visit_expression(scope_params, pos, content)?;
                        }
                        Ok(())
                    })?;

                    id.as_ref()
                }
                CommonDeclarationNamedKind::Function(decl) => {
                    let FunctionDeclaration {
                        span: _,
                        id,
                        params,
                        ret_ty,
                        body,
                    } = decl;

                    let mut scope_params = DeclScope::new_child(scope_parent);
                    visit_parameters(&mut scope_params, pos, params)?;
                    if let Some(ret_ty) = ret_ty {
                        visit_expression(&scope_params, pos, ret_ty)?;
                    }
                    visit_block_statements(&scope_params, pos, body)?;

                    id.as_ref()
                }
            };

            scope_parent.maybe_declare(id);
        }
        CommonDeclaration::ConstBlock(block) => {
            let ConstBlock { span_keyword: _, block } = block;
            visit_block_statements(scope_parent, pos, block)?;
        }
    }

    Ok(())
}

fn visit_parameters(scope: &mut DeclScope, pos: Pos, params: &Parameters) -> FindDefinitionResult {
    let Parameters { span: _, items } = params;

    visit_extra_list(scope, pos, items, &mut |scope, param| {
        let Parameter { id, ty, default } = param;
        visit_expression(scope, pos, ty)?;
        if let Some(default) = default {
            visit_expression(scope, pos, default)?;
        }
        scope.declare(id);
        Ok(())
    })
}

fn visit_extra_list<I>(
    scope_parent: &mut DeclScope,
    pos: Pos,
    extra: &ExtraList<I>,
    f: &mut impl FnMut(&mut DeclScope, &I) -> FindDefinitionResult,
) -> FindDefinitionResult {
    let ExtraList { span: _, items } = extra;

    for item in items {
        match item {
            ExtraItem::Inner(item) => f(scope_parent, item)?,
            ExtraItem::Declaration(decl) => visit_common_declaration(scope_parent, pos, decl)?,
            ExtraItem::If(if_stmt) => {
                let mut scope_inner = DeclScope::new_child(scope_parent);
                visit_if_stmt(&mut scope_inner, pos, if_stmt, &mut |s: &mut DeclScope, b| {
                    visit_extra_list(s, pos, b, f)
                })?;
                scope_parent.merge_conditional_child(scope_inner.map);
            }
        }
    }

    Ok(())
}

fn visit_if_stmt<I>(
    scope: &mut DeclScope,
    pos: Pos,
    if_stmt: &IfStatement<I>,
    f: &mut impl FnMut(&mut DeclScope, &I) -> FindDefinitionResult,
) -> FindDefinitionResult {
    let mut visit_pair = |pair: &IfCondBlockPair<I>| {
        let IfCondBlockPair {
            span: _,
            span_if: _,
            cond,
            block,
        } = pair;

        visit_expression(scope, pos, cond)?;
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
    scope: &DeclScope,
    pos: Pos,
    stmt: &ForStatement<S>,
    f: impl FnOnce(&DeclScope, &Block<S>) -> FindDefinitionResult,
) -> FindDefinitionResult {
    let ForStatement {
        span_keyword: _,
        index,
        index_ty,
        iter,
        body,
    } = stmt;
    visit_expression(scope, pos, iter)?;

    if let Some(index_ty) = index_ty {
        visit_expression(scope, pos, index_ty)?;
    }

    let mut scope_inner = DeclScope::new_child(scope);
    scope_inner.maybe_declare(index.as_ref());

    f(&scope_inner, body)?;

    Ok(())
}

fn visit_block_statements(scope_parents: &DeclScope, pos: Pos, block: &Block<BlockStatement>) -> FindDefinitionResult {
    let Block { span, statements } = block;

    // declarations inside a block can't leak outside, so we can skip here
    check_skip!(span, pos);

    let mut scope = DeclScope::new_child(scope_parents);
    for stmt in statements {
        visit_statement(&mut scope, pos, stmt)?;
    }

    Ok(())
}

fn visit_statement(scope: &mut DeclScope, pos: Pos, stmt: &BlockStatement) -> Result<(), FindDefinition> {
    let stmt_span = stmt.span;
    match &stmt.inner {
        BlockStatementKind::CommonDeclaration(decl) => {
            visit_common_declaration(scope, pos, decl)?;
        }
        BlockStatementKind::VariableDeclaration(decl) => {
            let VariableDeclaration {
                span: _,
                mutable: _,
                id,
                ty,
                init,
            } = decl;
            if let Some(ty) = ty {
                visit_expression(scope, pos, ty)?;
            }
            if let Some(init) = init {
                visit_expression(scope, pos, init)?;
            }
            scope.maybe_declare(id.as_ref());
        }
        BlockStatementKind::Assignment(stmt) => {
            let Assignment {
                span: _,
                op: _,
                target,
                value,
            } = stmt;
            visit_expression(scope, pos, target)?;
            visit_expression(scope, pos, value)?;
        }
        BlockStatementKind::Expression(expr) => {
            visit_expression(scope, pos, expr)?;
        }
        BlockStatementKind::Block(block) => {
            visit_block_statements(scope, pos, block)?;
        }
        BlockStatementKind::If(stmt) => {
            check_skip!(stmt_span, pos);
            visit_if_stmt(scope, pos, stmt, &mut |s, b| visit_block_statements(s, pos, b))?;
        }
        BlockStatementKind::Match(stmt) => {
            check_skip!(stmt_span, pos);

            let MatchStatement {
                target,
                span_branches: _,
                branches,
            } = stmt;
            visit_expression(scope, pos, target)?;

            for branch in branches {
                let MatchBranch { pattern, block } = branch;

                let mut scope_inner = DeclScope::new_child(scope);
                match &pattern.inner {
                    MatchPattern::Dummy => {}
                    MatchPattern::Val(id) => {
                        scope_inner.declare(id);
                    }
                    MatchPattern::Equal(expr) => {
                        visit_expression(scope, pos, expr)?;
                    }
                    MatchPattern::In(expr) => {
                        visit_expression(scope, pos, expr)?;
                    }
                    MatchPattern::EnumVariant(_variant, id) => {
                        if let Some(id) = id {
                            scope_inner.maybe_declare(id.as_ref());
                        }
                    }
                }

                visit_block_statements(&scope_inner, pos, block)?;
            }
        }
        BlockStatementKind::For(stmt) => {
            check_skip!(stmt_span, pos);
            visit_for_stmt(scope, pos, stmt, |s, b| visit_block_statements(s, pos, b))?;
        }
        BlockStatementKind::While(stmt) => {
            check_skip!(stmt_span, pos);
            let WhileStatement {
                span_keyword: _,
                cond,
                body,
            } = stmt;
            visit_expression(scope, pos, cond)?;
            visit_block_statements(scope, pos, body)?;
        }
        BlockStatementKind::Return(stmt) => {
            let ReturnStatement { span_return: _, value } = stmt;
            if let Some(value) = value {
                visit_expression(scope, pos, value)?;
            }
        }
        BlockStatementKind::Break(_span) | BlockStatementKind::Continue(_span) => {}
    }

    Ok(())
}

fn visit_block_module(scope_parent: &DeclScope, pos: Pos, block: &Block<ModuleStatement>) -> FindDefinitionResult {
    let Block { span, statements } = block;
    check_skip!(span, pos);

    // match the two-pass system from the main compiler
    let mut scope = DeclScope::new_child(scope_parent);
    for stmt in statements {
        match &stmt.inner {
            ModuleStatementKind::CommonDeclaration(decl) => {
                visit_common_declaration(&mut scope, pos, decl)?;
            }
            ModuleStatementKind::RegDeclaration(decl) => {
                let RegDeclaration { id, sync, ty, init } = decl;

                if let Some(sync) = sync {
                    visit_domain_sync(&scope, pos, &sync.inner)?;
                }
                visit_expression(&scope, pos, ty)?;
                visit_expression(&scope, pos, init)?;
                scope.maybe_declare(id.as_ref());
            }
            ModuleStatementKind::WireDeclaration(decl) => {
                let WireDeclaration { id, kind } = decl;

                match kind {
                    WireDeclarationKind::Clock { span_clock: _, value } => {
                        if let Some(value) = value {
                            visit_expression(&scope, pos, value)?;
                        }
                    }
                    WireDeclarationKind::NormalWithValue { domain, ty, value } => {
                        if let Some(domain) = domain {
                            visit_domain(&scope, pos, &domain.inner)?;
                        }
                        if let Some(ty) = ty {
                            visit_expression(&scope, pos, ty)?;
                        }
                        visit_expression(&scope, pos, value)?;
                    }
                    WireDeclarationKind::NormalWithoutValue { domain, ty } => {
                        if let Some(domain) = domain {
                            visit_domain(&scope, pos, &domain.inner)?;
                        }
                        visit_expression(&scope, pos, ty)?;
                    }
                }

                scope.maybe_declare(id.as_ref());
            }
            ModuleStatementKind::RegOutPortMarker(decl) => {
                let RegOutPortMarker { id, init } = decl;
                visit_expression(&scope, pos, init)?;
                visit_id_usage(&scope, pos, id)?;
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
                visit_block_module(&scope, pos, block)?;
            }
            ModuleStatementKind::If(stmt) => {
                visit_if_stmt(&mut scope, pos, stmt, &mut |s, b| visit_block_module(s, pos, b))?;
            }
            ModuleStatementKind::For(stmt) => visit_for_stmt(&scope, pos, stmt, |s, b| visit_block_module(s, pos, b))?,
            ModuleStatementKind::CombinatorialBlock(stmt) => {
                let CombinatorialBlock { span_keyword: _, block } = stmt;
                visit_block_statements(&scope, pos, block)?;
            }
            ModuleStatementKind::ClockedBlock(stmt) => {
                let ClockedBlock {
                    span_keyword: _,
                    span_domain: _,
                    clock,
                    reset,
                    block,
                } = stmt;
                visit_expression(&scope, pos, clock)?;
                if let Some(reset) = reset {
                    let ClockedBlockReset { kind: _, signal } = &reset.inner;
                    visit_expression(&scope, pos, signal)?;
                }
                visit_block_statements(&scope, pos, block)?;
            }
            ModuleStatementKind::Instance(stmt) => {
                let ModuleInstance {
                    name: _,
                    span_keyword: _,
                    module,
                    port_connections,
                } = stmt;

                visit_expression(&scope, pos, module)?;
                for conn in &port_connections.inner {
                    let PortConnection { id, expr } = &conn.inner;

                    // TODO try resolving port name, needs type info
                    let _ = id;

                    if let Some(expr) = expr {
                        visit_expression(&scope, pos, expr)?;
                    } else {
                        visit_id_usage(&scope, pos, id)?;
                    }
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

fn visit_array_literal_element<E: Borrow<Expression>>(
    scope: &DeclScope,
    pos: Pos,
    elem: &ArrayLiteralElement<E>,
) -> FindDefinitionResult {
    match elem {
        ArrayLiteralElement::Single(elem) => visit_expression(scope, pos, elem.borrow()),
        ArrayLiteralElement::Spread(_, elem) => visit_expression(scope, pos, elem.borrow()),
    }
}

fn visit_expression(scope: &DeclScope, pos: Pos, expr: &Expression) -> FindDefinitionResult {
    let Expression { span, inner } = expr;

    // expressions can't contain declarations that leak outside, so we can skip here
    if !span.touches_pos(pos) {
        return Ok(());
    }

    match inner {
        ExpressionKind::Wrapped(inner) => {
            visit_expression(scope, pos, inner)?;
        }
        ExpressionKind::Block(block) => {
            let BlockExpression { statements, expression } = block;
            let mut scope_inner = DeclScope::new_child(scope);
            for stmt in statements {
                visit_statement(&mut scope_inner, pos, stmt)?
            }
            visit_expression(&scope_inner, pos, expression)?;
        }
        ExpressionKind::Id(id) => {
            return visit_id_usage(scope, pos, id);
        }
        ExpressionKind::ArrayLiteral(elems) => {
            for elem in elems {
                visit_array_literal_element(scope, pos, elem)?;
            }
        }
        ExpressionKind::TupleLiteral(elems) => {
            for elem in elems {
                visit_expression(scope, pos, elem)?;
            }
        }
        ExpressionKind::RangeLiteral(expr) => match expr {
            RangeLiteral::ExclusiveEnd { op_span: _, start, end } => {
                if let Some(start) = start {
                    visit_expression(scope, pos, start)?;
                }
                if let Some(end) = end {
                    visit_expression(scope, pos, end)?;
                }
            }
            RangeLiteral::InclusiveEnd { op_span: _, start, end } => {
                if let Some(start) = start {
                    visit_expression(scope, pos, start)?;
                }
                visit_expression(scope, pos, end)?;
            }
            RangeLiteral::Length { op_span: _, start, len } => {
                visit_expression(scope, pos, start)?;
                visit_expression(scope, pos, len)?;
            }
        },
        ExpressionKind::ArrayComprehension(expr) => {
            let ArrayComprehension {
                body,
                index,
                span_keyword: _,
                iter,
            } = expr;
            visit_expression(scope, pos, iter)?;

            let mut scope_inner = DeclScope::new_child(scope);
            scope_inner.maybe_declare(index.as_ref());

            visit_array_literal_element(&scope_inner, pos, body)?;
        }
        ExpressionKind::UnaryOp(_, inner) => {
            visit_expression(scope, pos, inner)?;
        }
        ExpressionKind::BinaryOp(_, left, right) => {
            visit_expression(scope, pos, left)?;
            visit_expression(scope, pos, right)?;
        }
        ExpressionKind::ArrayType(lens, inner) => {
            for len in &lens.inner {
                visit_array_literal_element(scope, pos, len)?;
            }
            visit_expression(scope, pos, inner)?;
        }
        ExpressionKind::ArrayIndex(base, indices) => {
            visit_expression(scope, pos, base)?;
            for index in &indices.inner {
                visit_expression(scope, pos, index)?;
            }
        }
        ExpressionKind::DotIdIndex(base, _) => {
            // TODO try resolving index, needs type info
            visit_expression(scope, pos, base)?;
        }
        ExpressionKind::DotIntIndex(base, _) => {
            // TODO try resolving index, needs type info
            visit_expression(scope, pos, base)?;
        }
        ExpressionKind::Call(target, args) => {
            visit_expression(scope, pos, target)?;
            for arg in &args.inner {
                let Arg { span: _, name, value } = arg;
                // TODO try resolving name, needs type info
                let _ = name;
                visit_expression(scope, pos, value)?;
            }
        }
        ExpressionKind::Builtin(args) => {
            for arg in &args.inner {
                visit_expression(scope, pos, arg)?;
            }
        }
        ExpressionKind::UnsafeValueWithDomain(value, domain) => {
            visit_expression(scope, pos, value)?;
            visit_domain(scope, pos, &domain.inner)?;
        }
        ExpressionKind::RegisterDelay(expr) => {
            let RegisterDelay {
                span_keyword: _,
                value,
                init,
            } = expr;
            visit_expression(scope, pos, value)?;
            visit_expression(scope, pos, init)?;
        }

        ExpressionKind::Dummy
        | ExpressionKind::Undefined
        | ExpressionKind::Type
        | ExpressionKind::TypeFunction
        | ExpressionKind::IntLiteral(_)
        | ExpressionKind::BoolLiteral(_)
        | ExpressionKind::StringLiteral(_) => {}
    }

    Ok(())
}

fn visit_id_usage(scope: &DeclScope, pos: Pos, id: &Identifier) -> Result<(), FindDefinition> {
    check_skip!(id.span, pos);
    Err(scope.find(id))
}
