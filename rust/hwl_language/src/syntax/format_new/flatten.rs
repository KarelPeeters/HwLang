use crate::syntax::ast::{
    ArenaExpressions, Arg, Args, ArrayComprehension, ArrayLiteralElement, AssignBinaryOp, Assignment, Block,
    BlockExpression, BlockStatement, BlockStatementKind, ClockedBlock, ClockedBlockReset, CombinatorialBlock,
    CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration, DomainKind, DotIndexKind,
    Expression, ExpressionKind, ExtraItem, ExtraList, FileContent, ForStatement, FunctionDeclaration,
    GeneralIdentifier, Identifier, IfCondBlockPair, IfStatement, ImportEntry, ImportFinalKind, IntLiteral, Item,
    ItemDefModuleExternal, ItemDefModuleInternal, ItemImport, MaybeGeneralIdentifier, MaybeIdentifier, ModulePortBlock,
    ModulePortInBlock, ModulePortInBlockKind, ModulePortItem, ModulePortSingle, ModulePortSingleKind, ModuleStatement,
    ModuleStatementKind, Parameter, Parameters, PortSingleKindInner, RangeLiteral, RegDeclaration, RegOutPortMarker,
    RegisterDelay, SyncDomain, VariableDeclaration, Visibility, WireDeclaration, WireDeclarationDomainTyKind,
    WireDeclarationKind,
};
use crate::syntax::format_new::high::HNode;
use crate::syntax::token::TokenType as TT;
use crate::util::iter::IterExt;
use itertools::Either;

pub fn ast_to_node(file: &FileContent) -> HNode {
    let FileContent {
        span: _,
        items,
        arena_expressions,
    } = file;
    let ctx = Context { arena_expressions };
    ctx.fmt_file_items(items)
}

// TODO if this stays this single field, maybe we don't need this struct at all?
struct Context<'a> {
    arena_expressions: &'a ArenaExpressions,
}

impl Context<'_> {
    fn fmt_file_items(&self, items: &[Item]) -> HNode {
        // special case to deal with empty files
        let mut nodes = vec![HNode::PreserveBlankLines { last: items.is_empty() }];

        for (item, last) in items.iter().with_last() {
            let item_node = match item {
                Item::Import(import) => self.fmt_import(import),
                Item::CommonDeclaration(decl) => self.fmt_common_decl(&decl.inner),
                Item::ModuleInternal(decl) => {
                    let &ItemDefModuleInternal {
                        span: _,
                        vis,
                        id,
                        ref params,
                        ref ports,
                        ref body,
                    } = decl;
                    self.fmt_module_decl(vis, false, id, params.as_ref(), &ports.inner, Some(body))
                }
                Item::ModuleExternal(decl) => {
                    let &ItemDefModuleExternal {
                        span,
                        vis,
                        span_ext,
                        id,
                        ref params,
                        ref ports,
                    } = decl;
                    self.fmt_module_decl(
                        vis,
                        true,
                        MaybeIdentifier::Identifier(id),
                        params.as_ref(),
                        &ports.inner,
                        None,
                    )
                }
                Item::Interface(_) => todo!(),
            };
            nodes.push(item_node);

            nodes.push(HNode::PreserveBlankLines { last })
        }

        HNode::Sequence(nodes)
    }

    fn fmt_import(&self, import: &ItemImport) -> HNode {
        let ItemImport {
            span: _,
            parents,
            entry,
        } = import;

        let mut nodes = vec![];
        nodes.push(token(TT::Import));
        nodes.push(HNode::Space);

        for &parent in &parents.inner {
            nodes.push(self.fmt_id(parent));
            nodes.push(token(TT::Dot));
        }

        match &entry.inner {
            ImportFinalKind::Single(entry) => {
                nodes.push(self.fmt_import_entry(entry));
            }
            ImportFinalKind::Multi(entries) => {
                nodes.push(token(TT::OpenS));

                let mut nodes_fill = vec![];
                for (entry, last) in entries.iter().with_last() {
                    let mut seq = vec![self.fmt_import_entry(entry)];
                    push_comma_nodes(last, &mut seq);
                    nodes_fill.push(HNode::Sequence(seq));
                }

                nodes.push(HNode::Fill(nodes_fill));
                nodes.push(token(TT::CloseS));
            }
        }

        nodes.push(token(TT::Semi));
        nodes.push(HNode::AlwaysNewline);
        HNode::Sequence(nodes)
    }

    fn fmt_import_entry(&self, entry: &ImportEntry) -> HNode {
        let &ImportEntry { span: _, id, as_ } = entry;
        let mut nodes = vec![];
        nodes.push(self.fmt_id(id));
        if let Some(as_) = as_ {
            // TODO allow wrapping here?
            nodes.push(HNode::Space);
            nodes.push(token(TT::As));
            nodes.push(HNode::Space);
            nodes.push(self.fmt_maybe_id(as_));
        }
        HNode::Sequence(nodes)
    }

    fn fmt_common_decl<V: FormatVisibility>(&self, decl: &CommonDeclaration<V>) -> HNode {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                let node_vis = vis.token();
                let node_kind = match kind {
                    CommonDeclarationNamedKind::Type(_) => todo!(),
                    CommonDeclarationNamedKind::Const(decl) => {
                        let &ConstDeclaration { span: _, id, ty, value } = decl;
                        self.fmt_variable_decl(TT::Const, id, ty, Some(value))
                    }
                    CommonDeclarationNamedKind::Struct(_) => todo!(),
                    CommonDeclarationNamedKind::Enum(_) => todo!(),
                    CommonDeclarationNamedKind::Function(decl) => {
                        let &FunctionDeclaration {
                            span: _,
                            id,
                            ref params,
                            ret_ty,
                            ref body,
                        } = decl;

                        let mut nodes = vec![
                            token(TT::Fn),
                            HNode::Space,
                            self.fmt_maybe_id(id),
                            self.fmt_parameters(params),
                        ];

                        if let Some(ret_ty) = ret_ty {
                            nodes.push(HNode::Space);
                            nodes.push(token(TT::Arrow));
                            nodes.push(HNode::Space);
                            nodes.push(self.fmt_expr(ret_ty));
                        }

                        nodes.push(HNode::Space);
                        nodes.push(self.fmt_block(body, false));
                        nodes.push(HNode::AlwaysNewline);

                        HNode::Sequence(nodes)
                    }
                };

                match node_vis {
                    None => node_kind,
                    Some(token_vis) => HNode::Sequence(vec![token(token_vis), node_kind]),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }
    }

    fn fmt_module_decl(
        &self,
        vis: Visibility,
        external: bool,
        id: MaybeIdentifier,
        params: Option<&Parameters>,
        ports: &ExtraList<ModulePortItem>,
        body: Option<&Block<ModuleStatement>>,
    ) -> HNode {
        let mut seq = vec![];

        if let Some(vis) = vis.token() {
            seq.push(token(vis));
            seq.push(HNode::Space);
        }

        if external {
            seq.push(token(TT::External));
            seq.push(HNode::Space);
        }

        seq.push(token(TT::Module));
        seq.push(HNode::Space);
        seq.push(self.fmt_maybe_id(id));

        if let Some(params) = params {
            seq.push(self.fmt_parameters(params));
        }

        seq.push(HNode::Space);
        seq.push(token(TT::Ports));
        seq.push(token(TT::OpenR));
        seq.push(fmt_extra_list(ports, &|port| self.fmt_module_port_item(port)));
        seq.push(token(TT::CloseR));

        if let Some(body) = body {
            let Block { span: _, statements } = body;
            let node_block = fmt_block_impl(statements, None, false, |stmt| self.fmt_module_statement(stmt));

            seq.push(HNode::Space);
            seq.push(node_block);
        }

        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_parameters(&self, params: &Parameters) -> HNode {
        let Parameters { span: _, items } = params;
        HNode::Sequence(vec![
            token(TT::OpenR),
            fmt_extra_list(items, &|p| self.fmt_parameter(p)),
            token(TT::CloseR),
        ])
    }

    fn fmt_parameter(&self, param: &Parameter) -> HNode {
        let &Parameter {
            span: _,
            id,
            ty,
            default,
        } = param;

        let mut seq = vec![self.fmt_id(id)];
        push_wrapping_type(&mut seq, self.fmt_expr(ty));
        if let Some(default) = default {
            push_wrapping_assign(&mut seq, self.fmt_expr(default));
        }
        HNode::Sequence(seq)
    }

    fn fmt_module_port_item(&self, item: &ModulePortItem) -> HNode {
        match item {
            ModulePortItem::Single(single) => {
                let &ModulePortSingle { span: _, id, ref kind } = single;

                let node_kind = match kind {
                    ModulePortSingleKind::Port {
                        direction,
                        kind: kind_inner,
                    } => {
                        let node_kind_inner = match kind_inner {
                            PortSingleKindInner::Clock { span_clock: _ } => token(TT::Clock),
                            &PortSingleKindInner::Normal { domain, ty } => {
                                HNode::Sequence(vec![self.fmt_domain(domain.inner), HNode::Space, self.fmt_expr(ty)])
                            }
                        };
                        HNode::Sequence(vec![token(direction.inner.token()), HNode::Space, node_kind_inner])
                    }
                    &ModulePortSingleKind::Interface {
                        span_keyword: _,
                        domain,
                        interface,
                    } => HNode::Sequence(vec![
                        token(TT::Interface),
                        HNode::Space,
                        self.fmt_domain(domain.inner),
                        HNode::Space,
                        self.fmt_expr(interface),
                    ]),
                };

                let mut seq = vec![self.fmt_id(id)];
                push_wrapping_type(&mut seq, node_kind);
                HNode::Sequence(seq)
            }
            ModulePortItem::Block(block) => {
                let ModulePortBlock { span: _, domain, ports } = block;

                let node_domain = self.fmt_domain(domain.inner);
                let node_ports = fmt_extra_list(ports, &|port| {
                    let &ModulePortInBlock { span: _, id, kind } = port;
                    let node_kind = match kind {
                        ModulePortInBlockKind::Port { direction, ty } => {
                            HNode::Sequence(vec![token(direction.inner.token()), HNode::Space, self.fmt_expr(ty)])
                        }
                        ModulePortInBlockKind::Interface {
                            span_keyword: _,
                            interface,
                        } => HNode::Sequence(vec![token(TT::Interface), HNode::Space, self.fmt_expr(interface)]),
                    };
                    let mut seq = vec![self.fmt_id(id)];
                    push_wrapping_type(&mut seq, node_kind);
                    HNode::Sequence(seq)
                });

                HNode::Sequence(vec![
                    node_domain,
                    HNode::Space,
                    token(TT::OpenC),
                    node_ports,
                    token(TT::CloseC),
                ])
            }
        }
    }

    fn fmt_module_block(&self, block: &Block<ModuleStatement>, force_wrap: bool) -> HNode {
        let Block { span: _, statements } = block;
        fmt_block_impl(statements, None, force_wrap, |stmt| self.fmt_module_statement(stmt))
    }

    fn fmt_block(&self, block: &Block<BlockStatement>, force_wrap: bool) -> HNode {
        let Block { span: _, statements } = block;
        self.fmt_block_ext(statements, None, force_wrap)
    }

    fn fmt_block_ext(
        &self,
        statements: &[BlockStatement],
        final_expression: Option<Expression>,
        force_wrap: bool,
    ) -> HNode {
        let node_final_expression = final_expression.map(|expr| self.fmt_expr(expr));
        let f = |stmt: &BlockStatement| self.fmt_block_statement(stmt);
        fmt_block_impl(statements, node_final_expression, force_wrap, f)
    }

    fn fmt_module_statement(&self, stmt: &ModuleStatement) -> HNode {
        match &stmt.inner {
            ModuleStatementKind::Block(block) => {
                let node_block = self.fmt_module_block(block, false);
                HNode::Sequence(vec![node_block, HNode::AlwaysNewline])
            }
            ModuleStatementKind::If(stmt) => self.fmt_if(stmt, |b| self.fmt_module_block(b, true)),
            ModuleStatementKind::For(stmt) => self.fmt_for(stmt, |b| self.fmt_module_block(b, false)),
            ModuleStatementKind::CommonDeclaration(decl) => self.fmt_common_decl(decl),
            ModuleStatementKind::RegDeclaration(decl) => {
                let &RegDeclaration {
                    vis,
                    id,
                    sync,
                    ty,
                    init,
                } = decl;

                let mut seq = vec![];
                if let Some(vis) = vis.token() {
                    seq.push(token(vis));
                }

                seq.push(token(TT::Reg));
                seq.push(HNode::Space);
                seq.push(self.fmt_maybe_general_id(id));

                let mut sync_ty_seq = vec![];
                if let Some(sync) = sync {
                    sync_ty_seq.push(self.fmt_domain(DomainKind::Sync(sync.inner)));
                    sync_ty_seq.push(HNode::Space);
                }
                sync_ty_seq.push(self.fmt_expr(ty));
                push_wrapping_type(&mut seq, HNode::Sequence(sync_ty_seq));

                push_wrapping_assign(&mut seq, self.fmt_expr(init));

                seq.push(token(TT::Semi));
                seq.push(HNode::AlwaysNewline);
                HNode::Sequence(seq)
            }
            ModuleStatementKind::WireDeclaration(decl) => {
                let &WireDeclaration {
                    vis,
                    span_keyword: _,
                    id,
                    kind,
                } = decl;

                let mut seq = vec![];
                if let Some(vis) = vis.token() {
                    seq.push(token(vis));
                    seq.push(HNode::Space);
                }

                seq.push(token(TT::Wire));
                seq.push(HNode::Space);
                seq.push(self.fmt_maybe_general_id(id));

                match kind {
                    WireDeclarationKind::Normal {
                        domain_ty,
                        assign_span_and_value,
                    } => {
                        // TODO wrapping, similar to the type of constants or variables
                        let domain_ty_node = match domain_ty {
                            WireDeclarationDomainTyKind::Clock { span_clock: _ } => Some(token(TT::Clock)),
                            WireDeclarationDomainTyKind::Normal { domain, ty } => {
                                if domain.is_some() || ty.is_some() {
                                    let mut domain_ty_seq = vec![];
                                    if let Some(domain) = domain {
                                        domain_ty_seq.push(HNode::Space);
                                        domain_ty_seq.push(self.fmt_domain(domain.inner));
                                    }
                                    if let Some(ty) = ty {
                                        domain_ty_seq.push(HNode::Space);
                                        domain_ty_seq.push(self.fmt_expr(ty));
                                    }
                                    Some(HNode::Sequence(domain_ty_seq))
                                } else {
                                    None
                                }
                            }
                        };

                        if let Some(domain_ty_node) = domain_ty_node {
                            push_wrapping_type(&mut seq, domain_ty_node);
                        }
                        if let Some((_span, value)) = assign_span_and_value {
                            push_wrapping_assign(&mut seq, self.fmt_expr(value));
                        }
                    }
                    WireDeclarationKind::Interface { .. } => todo!(),
                };

                seq.push(token(TT::Semi));
                seq.push(HNode::AlwaysNewline);
                HNode::Sequence(seq)
            }
            ModuleStatementKind::RegOutPortMarker(marker) => {
                let &RegOutPortMarker { id, init } = marker;
                HNode::Sequence(vec![
                    token(TT::Reg),
                    HNode::Space,
                    token(TT::Out),
                    HNode::Space,
                    self.fmt_id(id),
                    HNode::Space,
                    token(TT::Eq),
                    HNode::Space,
                    self.fmt_expr(init),
                    token(TT::Semi),
                    HNode::AlwaysNewline,
                ])
            }
            ModuleStatementKind::CombinatorialBlock(block) => {
                let CombinatorialBlock { span_keyword: _, block } = block;
                HNode::Sequence(vec![
                    token(TT::Comb),
                    HNode::Space,
                    self.fmt_block(block, false),
                    HNode::AlwaysNewline,
                ])
            }
            ModuleStatementKind::ClockedBlock(block) => {
                let &ClockedBlock {
                    span_keyword: _,
                    span_domain: _,
                    clock,
                    reset,
                    ref block,
                } = block;

                let args: &[_] = match reset {
                    None => &[Either::Left(clock)],
                    Some(reset) => &[Either::Left(clock), Either::Right(reset)],
                };
                let node_domain = fmt_call(token(TT::Clocked), args, |&arg| match arg {
                    Either::Left(clock) => self.fmt_expr(clock),
                    Either::Right(reset) => {
                        let ClockedBlockReset { kind, signal } = reset.inner;
                        HNode::Sequence(vec![token(kind.inner.token()), HNode::Space, self.fmt_expr(signal)])
                    }
                });

                HNode::Sequence(vec![
                    node_domain,
                    HNode::Space,
                    self.fmt_block(block, false),
                    HNode::AlwaysNewline,
                ])
            }
            ModuleStatementKind::Instance(_) => todo!(),
        }
    }

    fn fmt_block_statement(&self, stmt: &BlockStatement) -> HNode {
        match &stmt.inner {
            BlockStatementKind::CommonDeclaration(_) => todo!(),
            BlockStatementKind::VariableDeclaration(decl) => {
                let &VariableDeclaration {
                    span: _,
                    mutable,
                    id,
                    ty,
                    init,
                } = decl;
                let kind = if mutable { TT::Var } else { TT::Val };
                self.fmt_variable_decl(kind, id, ty, init)
            }
            BlockStatementKind::Assignment(stmt) => {
                let &Assignment {
                    span: _,
                    op,
                    target,
                    value,
                } = stmt;

                let mut seq = vec![self.fmt_expr(target)];
                push_wrapping_assign_op(
                    &mut seq,
                    op.inner.map_or(TT::Eq, AssignBinaryOp::token),
                    self.fmt_expr(value),
                );
                seq.push(token(TT::Semi));
                seq.push(HNode::AlwaysNewline);
                HNode::Sequence(seq)
            }
            &BlockStatementKind::Expression(expr) => {
                HNode::Sequence(vec![self.fmt_expr(expr), token(TT::Semi), HNode::AlwaysNewline])
            }
            BlockStatementKind::Block(_) => todo!(),
            BlockStatementKind::If(stmt) => self.fmt_if(stmt, |block| self.fmt_block(block, true)),
            BlockStatementKind::Match(_) => todo!(),
            BlockStatementKind::For(_) => todo!(),
            BlockStatementKind::While(_) => todo!(),
            BlockStatementKind::Return(_) => todo!(),
            BlockStatementKind::Break(_) => todo!(),
            BlockStatementKind::Continue(_) => todo!(),
        }
    }

    fn fmt_variable_decl(
        &self,
        kind: TT,
        id: MaybeIdentifier,
        ty: Option<Expression>,
        value: Option<Expression>,
    ) -> HNode {
        let mut seq = vec![];
        seq.push(token(kind));
        seq.push(HNode::Space);
        seq.push(self.fmt_maybe_id(id));
        if let Some(ty) = ty {
            push_wrapping_type(&mut seq, self.fmt_expr(ty));
        }
        if let Some(value) = value {
            push_wrapping_assign(&mut seq, self.fmt_expr(value));
        }
        seq.push(token(TT::Semi));
        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_if<B>(&self, stmt: &IfStatement<B>, f_force_wrap: impl Fn(&B) -> HNode) -> HNode {
        fn fmt_else(seq: &mut Vec<HNode>) {
            seq.push(HNode::Space);
            seq.push(token(TT::Else));
            seq.push(HNode::Space);
        }

        fn fmt_pair<B>(
            slf: &Context,
            f_force_wrap: impl Fn(&B) -> HNode,
            seq: &mut Vec<HNode>,
            pair: &IfCondBlockPair<B>,
        ) {
            let &IfCondBlockPair {
                span: _,
                span_if: _,
                cond,
                ref block,
            } = pair;
            seq.push(token(TT::If));
            seq.push(HNode::Space);
            seq.push(surrounded_group_indent(SurroundKind::Round, slf.fmt_expr(cond)));
            seq.push(HNode::Space);
            seq.push(f_force_wrap(block));
        }

        let IfStatement {
            span: _,
            initial_if,
            else_ifs,
            final_else,
        } = stmt;

        let mut seq = vec![];
        fmt_pair(self, &f_force_wrap, &mut seq, initial_if);
        for else_if in else_ifs {
            fmt_else(&mut seq);
            fmt_pair(self, &f_force_wrap, &mut seq, else_if);
        }
        if let Some(final_else) = final_else {
            fmt_else(&mut seq);
            seq.push(f_force_wrap(final_else));
        }

        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_for<S>(&self, stmt: &ForStatement<S>, f: impl Fn(&Block<S>) -> HNode) -> HNode {
        let &ForStatement {
            span_keyword: _,
            index,
            index_ty,
            iter,
            ref body,
        } = stmt;

        let mut seq = vec![self.fmt_maybe_id(index)];
        if let Some(index_ty) = index_ty {
            push_wrapping_type(&mut seq, self.fmt_expr(index_ty));
        }

        seq.push(group_indent_seq(vec![
            HNode::WrapNewline,
            HNode::Space,
            token(TT::In),
            HNode::Space,
            self.fmt_expr(iter),
        ]));

        HNode::Sequence(vec![
            token(TT::For),
            HNode::Space,
            surrounded_group_indent(SurroundKind::Round, HNode::Sequence(seq)),
            HNode::Space,
            f(body),
            HNode::AlwaysNewline,
        ])
    }

    fn fmt_domain(&self, domain: DomainKind<Expression>) -> HNode {
        match domain {
            DomainKind::Const => token(TT::Const),
            DomainKind::Async => token(TT::Async),
            DomainKind::Sync(domain) => {
                let SyncDomain { clock, reset } = domain;
                let args: &[Expression] = match reset {
                    None => &[clock],
                    Some(reset) => &[clock, reset],
                };
                fmt_call(token(TT::Sync), args, |&expr| self.fmt_expr(expr))
            }
        }
    }

    fn fmt_expr(&self, expr: Expression) -> HNode {
        match &self.arena_expressions[expr.inner] {
            ExpressionKind::Dummy => token(TT::Underscore),
            ExpressionKind::Undefined => token(TT::Undef),
            ExpressionKind::Type => token(TT::Type),
            ExpressionKind::TypeFunction => token(TT::Fn),
            ExpressionKind::Builtin => token(TT::Builtin),
            &ExpressionKind::Wrapped(inner) => surrounded_group_indent(SurroundKind::Round, self.fmt_expr(inner)),
            ExpressionKind::Block(expr) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = expr;
                self.fmt_block_ext(statements, Some(expression), false)
            }
            &ExpressionKind::Id(id) => self.fmt_general_id(id),
            ExpressionKind::IntLiteral(literal) => {
                let tt = match literal {
                    IntLiteral::Binary(_span) => TT::IntLiteralBinary,
                    IntLiteral::Decimal(_span) => TT::IntLiteralDecimal,
                    IntLiteral::Hexadecimal(_span) => TT::IntLiteralHexadecimal,
                };
                token(tt)
            }
            &ExpressionKind::BoolLiteral(value) => {
                let tt = if value { TT::True } else { TT::False };
                token(tt)
            }
            ExpressionKind::StringLiteral(pieces) => {
                todo!()
            }
            ExpressionKind::ArrayLiteral(elements) => {
                let node_elements = fmt_comma_list(elements, |&elem| self.fmt_array_literal_element(elem));
                HNode::Sequence(vec![token(TT::OpenS), node_elements, token(TT::CloseS)])
            }
            ExpressionKind::TupleLiteral(elements) => {
                match elements.as_slice() {
                    // don't allow breaking for empty tuple
                    [] => HNode::Sequence(vec![token(TT::OpenR), token(TT::CloseR)]),
                    // force trailing comma for single-element tuple
                    &[element] => surrounded_group_indent(
                        SurroundKind::Round,
                        HNode::Sequence(vec![self.fmt_expr(element), token(TT::Comma)]),
                    ),
                    // general case
                    elements => {
                        let node_elements = fmt_comma_list(elements, |&elem| self.fmt_expr(elem));
                        HNode::Sequence(vec![token(TT::OpenR), node_elements, token(TT::CloseR)])
                    }
                }
            }
            &ExpressionKind::RangeLiteral(expr) => {
                let mut seq = vec![];
                match expr {
                    RangeLiteral::ExclusiveEnd { op_span: _, start, end } => {
                        if let Some(start) = start {
                            seq.push(self.fmt_expr(start));
                        }
                        seq.push(token(TT::DotDot));
                        if let Some(end) = end {
                            seq.push(self.fmt_expr(end));
                        }
                    }
                    RangeLiteral::InclusiveEnd { op_span: _, start, end } => {
                        if let Some(start) = start {
                            seq.push(self.fmt_expr(start));
                        }
                        seq.push(token(TT::DotDotEq));
                        seq.push(self.fmt_expr(end));
                    }
                    RangeLiteral::Length { op_span: _, start, len } => {
                        seq.push(self.fmt_expr(start));
                        seq.push(token(TT::PlusDotDot));
                        seq.push(self.fmt_expr(len));
                    }
                }
                HNode::Sequence(seq)
            }
            ExpressionKind::ArrayComprehension(expr) => {
                let &ArrayComprehension {
                    body,
                    index,
                    span_keyword: _,
                    iter,
                } = expr;

                let seq = vec![
                    self.fmt_array_literal_element(body),
                    HNode::WrapNewline,
                    HNode::Space,
                    token(TT::For),
                    HNode::Space,
                    self.fmt_maybe_id(index),
                    HNode::Space,
                    token(TT::In),
                    HNode::Space,
                    self.fmt_expr(iter),
                ];

                surrounded_group_indent(SurroundKind::Square, HNode::Sequence(seq))
            }
            &ExpressionKind::UnaryOp(op, inner) => HNode::Sequence(vec![token(op.inner.token()), self.fmt_expr(inner)]),
            &ExpressionKind::BinaryOp(op, left, right) => {
                let node_left = self.fmt_expr(left);
                let node_right = HNode::Sequence(vec![
                    HNode::Space,
                    token(op.inner.token()),
                    HNode::Space,
                    self.fmt_expr(right),
                ]);
                wrapping_binary_op(node_left, node_right)
            }
            &ExpressionKind::ArrayType(ref lengths, base) => {
                let node_lengths = fmt_comma_list(&lengths.inner, |&len| self.fmt_array_literal_element(len));
                HNode::Sequence(vec![
                    token(TT::OpenS),
                    node_lengths,
                    token(TT::CloseS),
                    self.fmt_expr(base),
                ])
            }
            &ExpressionKind::ArrayIndex(base, ref indices) => {
                let node_indices = fmt_comma_list(&indices.inner, |&index| self.fmt_expr(index));
                HNode::Sequence(vec![
                    self.fmt_expr(base),
                    token(TT::OpenS),
                    node_indices,
                    token(TT::CloseS),
                ])
            }
            ExpressionKind::DotIndex(_, _) => {
                // repeated dot indices should wrap together
                let mut base = expr;
                let mut indices = vec![];
                while let &ExpressionKind::DotIndex(curr_base, curr_index) = &self.arena_expressions[base.inner] {
                    base = curr_base;
                    indices.push(curr_index);
                }

                let node_base = self.fmt_expr(base);

                let mut seq = vec![];
                for (index, last) in indices.into_iter().with_last() {
                    let node_index = match index {
                        DotIndexKind::Id(id) => self.fmt_id(id),
                        DotIndexKind::Int(_span) => token(TT::IntLiteralDecimal),
                    };

                    seq.push(token(TT::Dot));
                    seq.push(node_index);

                    if !last {
                        seq.push(HNode::WrapNewline);
                    }
                }
                let node_indices = HNode::Sequence(seq);

                wrapping_binary_op(node_base, node_indices)
            }
            &ExpressionKind::Call(target, ref args) => {
                let Args { span: _, inner: args } = args;
                fmt_call(self.fmt_expr(target), args, |arg| {
                    let &Arg { span: _, name, value } = arg;
                    let node_value = self.fmt_expr(value);
                    if let Some(name) = name {
                        // TODO allow wrapping before/after "="?
                        HNode::Sequence(vec![self.fmt_id(name), token(TT::Eq), node_value])
                    } else {
                        node_value
                    }
                })
            }
            &ExpressionKind::UnsafeValueWithDomain(value, domain) => fmt_call(
                token(TT::UnsafeValueWithDomain),
                &[Either::Left(value), Either::Right(domain)],
                |&either| match either {
                    Either::Left(value) => self.fmt_expr(value),
                    Either::Right(domain) => self.fmt_domain(domain.inner),
                },
            ),
            ExpressionKind::RegisterDelay(expr) => {
                let &RegisterDelay {
                    span_keyword: _,
                    value,
                    init,
                } = expr;
                fmt_call(token(TT::Reg), &[value, init], |&expr| self.fmt_expr(expr))
            }
        }
    }

    fn fmt_array_literal_element(&self, elem: ArrayLiteralElement<Expression>) -> HNode {
        match elem {
            ArrayLiteralElement::Single(elem) => self.fmt_expr(elem),
            ArrayLiteralElement::Spread(_span, elem) => HNode::Sequence(vec![token(TT::Star), self.fmt_expr(elem)]),
        }
    }

    fn fmt_maybe_general_id(&self, id: MaybeGeneralIdentifier) -> HNode {
        match id {
            MaybeGeneralIdentifier::Dummy(_span) => token(TT::Underscore),
            MaybeGeneralIdentifier::Identifier(id) => self.fmt_general_id(id),
        }
    }

    fn fmt_general_id(&self, id: GeneralIdentifier) -> HNode {
        match id {
            GeneralIdentifier::Simple(id) => self.fmt_id(id),
            GeneralIdentifier::FromString(_span, expr) => {
                fmt_call(token(TT::IdFromStr), &[expr], |&expr| self.fmt_expr(expr))
            }
        }
    }

    fn fmt_maybe_id(&self, id: MaybeIdentifier) -> HNode {
        match id {
            MaybeIdentifier::Dummy(_span) => token(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.fmt_id(id),
        }
    }

    fn fmt_id(&self, id: Identifier) -> HNode {
        let _ = id;
        token(TT::Identifier)
    }
}

fn fmt_block_impl<T>(
    statements: &[T],
    final_expression: Option<HNode>,
    force_wrap: bool,
    f: impl Fn(&T) -> HNode,
) -> HNode {
    let mut seq = vec![HNode::WrapNewline];
    if force_wrap {
        seq.push(HNode::ForceWrap);
    }

    for (stmt, last_stmt) in statements.iter().with_last() {
        seq.push(f(stmt));
        let last = last_stmt && final_expression.is_none();
        seq.push(HNode::PreserveBlankLines { last });
    }

    if let Some(final_expression) = final_expression {
        todo!()
    }

    HNode::Group(Box::new(HNode::Sequence(vec![
        token(TT::OpenC),
        HNode::WrapIndent(Box::new(HNode::Sequence(seq))),
        token(TT::CloseC),
    ])))
}

fn fmt_extra_list<T>(list: &ExtraList<T>, f: &impl Fn(&T) -> HNode) -> HNode {
    // TODO variant that always wraps if there is any item, for eg. module ports? or not, maybe it's more elegant if we don't
    let ExtraList { span: _, items } = list;

    let mut nodes = vec![];
    nodes.push(HNode::WrapNewline);

    for (item, last) in items.iter().with_last() {
        match item {
            ExtraItem::Inner(item) => {
                nodes.push(f(item));
                push_comma_nodes(last, &mut nodes);
                nodes.push(HNode::WrapNewline);
            }
            ExtraItem::Declaration(_) => todo!(),
            ExtraItem::If(_) => todo!(),
        }
        nodes.push(HNode::PreserveBlankLines { last });
    }

    group_indent_seq(nodes)
}

fn fmt_call<T>(target: HNode, args: &[T], f: impl Fn(&T) -> HNode) -> HNode {
    let node_arg_list = fmt_comma_list(args, f);
    let node_args = HNode::Sequence(vec![token(TT::OpenR), node_arg_list, token(TT::CloseR)]);
    HNode::Sequence(vec![target, node_args])
}

fn fmt_comma_list<T>(items: &[T], f: impl Fn(&T) -> HNode) -> HNode {
    let mut nodes = vec![];
    nodes.push(HNode::WrapNewline);
    for (item, last) in items.iter().with_last() {
        nodes.push(f(item));
        push_comma_nodes(last, &mut nodes);
        nodes.push(HNode::WrapNewline);
        nodes.push(HNode::PreserveBlankLines { last });
    }
    group_indent_seq(nodes)
}

fn push_comma_nodes(last: bool, seq: &mut Vec<HNode>) {
    if !last {
        seq.push(token(TT::Comma));
        seq.push(HNode::Space);
    } else {
        seq.push(HNode::WrapComma);
    }
}

fn push_wrapping_type(seq: &mut Vec<HNode>, ty: HNode) {
    seq.push(group_indent_seq(vec![
        HNode::WrapNewline,
        token(TT::Colon),
        HNode::Space,
    ]));
    seq.push(ty);
}

fn push_wrapping_assign(seq: &mut Vec<HNode>, value: HNode) {
    push_wrapping_assign_op(seq, TT::Eq, value);
}

fn push_wrapping_assign_op(seq: &mut Vec<HNode>, op: TT, value: HNode) {
    seq.push(group_indent_seq(vec![
        HNode::WrapNewline,
        HNode::Space,
        token(op),
        HNode::Space,
    ]));
    seq.push(value);
}

fn wrapping_binary_op(left: HNode, right: HNode) -> HNode {
    HNode::Group(Box::new(HNode::Sequence(vec![
        left,
        HNode::WrapNewline,
        HNode::WrapIndent(Box::new(right)),
    ])))
}

fn group_indent_seq(nodes: Vec<HNode>) -> HNode {
    HNode::Group(Box::new(HNode::WrapIndent(Box::new(HNode::Sequence(nodes)))))
}

enum SurroundKind {
    Round,
    Square,
    Curly,
}

fn surrounded_group_indent(surround: SurroundKind, inner: HNode) -> HNode {
    let (before, after) = match surround {
        SurroundKind::Round => (TT::OpenR, TT::CloseR),
        SurroundKind::Square => (TT::OpenS, TT::CloseS),
        SurroundKind::Curly => (TT::OpenC, TT::CloseC),
    };

    HNode::Group(Box::new(HNode::Sequence(vec![
        token(before),
        HNode::WrapIndent(Box::new(HNode::Sequence(vec![
            HNode::WrapNewline,
            inner,
            HNode::WrapNewline,
        ]))),
        token(after),
    ])))
}

fn token(tt: TT) -> HNode {
    HNode::AlwaysToken(tt)
}

trait FormatVisibility: Copy {
    fn token(self) -> Option<TT>;
}

impl FormatVisibility for Visibility {
    fn token(self) -> Option<TT> {
        match self {
            Visibility::Private => None,
            Visibility::Public(_span) => Some(TT::Pub),
        }
    }
}

impl FormatVisibility for () {
    fn token(self) -> Option<TT> {
        None
    }
}
