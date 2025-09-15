use crate::syntax::ast::{
    ArenaExpressions, Arg, Args, ArrayComprehension, ArrayLiteralElement, AssignBinaryOp, Assignment, BinaryOp,
    BinaryOpLevel, Block, BlockExpression, BlockStatement, BlockStatementKind, ClockedBlock, ClockedBlockReset,
    CombinatorialBlock, CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstBlock,
    ConstDeclaration, DomainKind, DotIndexKind, EnumDeclaration, EnumVariant, Expression, ExpressionKind, ExtraItem,
    ExtraList, FileContent, ForStatement, FunctionDeclaration, GeneralIdentifier, Identifier, IfCondBlockPair,
    IfStatement, ImportEntry, ImportFinalKind, IntLiteral, InterfaceView, Item, ItemDefInterface,
    ItemDefModuleExternal, ItemDefModuleInternal, ItemImport, MatchBranch, MatchPattern, MatchStatement,
    MaybeGeneralIdentifier, MaybeIdentifier, ModuleInstance, ModulePortBlock, ModulePortInBlock, ModulePortInBlockKind,
    ModulePortItem, ModulePortSingle, ModulePortSingleKind, ModuleStatement, ModuleStatementKind, Parameter,
    Parameters, PortConnection, PortConnectionExpression, PortSingleKindInner, RangeLiteral, RegDeclaration,
    RegOutPortMarker, RegisterDelay, ReturnStatement, StringPiece, StructDeclaration, StructField, SyncDomain,
    TypeDeclaration, VariableDeclaration, Visibility, WhileStatement, WireDeclaration, WireDeclarationDomainTyKind,
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
                        span: _,
                        vis,
                        span_ext: _,
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
                Item::Interface(decl) => self.fmt_interface_decl(decl),
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
                    seq.push(comma_nodes(last));
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

    fn fmt_extra_list<T, N: Into<HNodeAndComma>>(
        &self,
        surround: SurroundKind,
        force_wrap: bool,
        list: &ExtraList<T>,
        f: &impl Fn(&T) -> N,
    ) -> HNode {
        surrounded_group_indent(surround, self.fmt_extra_list_inner(force_wrap, list, f))
    }

    fn fmt_extra_list_inner<T, N: Into<HNodeAndComma>>(
        &self,
        force_wrap: bool,
        list: &ExtraList<T>,
        f: &impl Fn(&T) -> N,
    ) -> HNode {
        let ExtraList { span: _, items } = list;

        let mut seq = vec![];
        if force_wrap && !items.is_empty() {
            seq.push(HNode::ForceWrap);
        }

        for (item, last) in items.iter().with_last() {
            match item {
                ExtraItem::Inner(item) => {
                    let HNodeAndComma { node, comma } = N::into(f(item));
                    seq.push(node);
                    if comma {
                        seq.push(comma_nodes(last));
                    } else {
                        seq.push(HNode::ForceWrap);
                    }
                    if !last {
                        seq.push(HNode::WrapNewline);
                    }
                }
                ExtraItem::Declaration(decl) => {
                    seq.push(self.fmt_common_decl(decl));
                }
                ExtraItem::If(stmt) => {
                    seq.push(self.fmt_if(stmt, |inner| self.fmt_extra_list(SurroundKind::Curly, true, inner, f)))
                }
            }
            seq.push(HNode::PreserveBlankLines { last });
        }
        HNode::Sequence(seq)
    }

    fn fmt_common_decl<V: FormatVisibility>(&self, decl: &CommonDeclaration<V>) -> HNode {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                let node_kind = match kind {
                    CommonDeclarationNamedKind::Type(decl) => {
                        let &TypeDeclaration {
                            span: _,
                            id,
                            ref params,
                            body,
                        } = decl;
                        let mut seq = vec![token(TT::Type), HNode::Space, self.fmt_maybe_id(id)];
                        if let Some(params) = params {
                            seq.push(self.fmt_parameters(params));
                        }
                        seq.push(wrapping_assign(self.fmt_expr(body)));

                        seq.push(token(TT::Semi));
                        seq.push(HNode::AlwaysNewline);
                        HNode::Sequence(seq)
                    }
                    CommonDeclarationNamedKind::Const(decl) => {
                        let &ConstDeclaration { span: _, id, ty, value } = decl;
                        self.fmt_variable_decl(TT::Const, id, ty, Some(value))
                    }
                    CommonDeclarationNamedKind::Struct(decl) => {
                        let &StructDeclaration {
                            span: _,
                            span_body: _,
                            id,
                            ref params,
                            ref fields,
                        } = decl;

                        let mut seq = vec![token(TT::Struct), HNode::Space, self.fmt_maybe_id(id)];
                        if let Some(params) = params {
                            seq.push(self.fmt_parameters(params));
                        }
                        seq.push(HNode::Space);
                        seq.push(self.fmt_extra_list(SurroundKind::Curly, true, fields, &|field| {
                            let &StructField { span: _, id, ty } = field;
                            HNode::Sequence(vec![self.fmt_id(id), wrapping_type(self.fmt_expr(ty))])
                        }));
                        seq.push(HNode::AlwaysNewline);
                        HNode::Sequence(seq)
                    }
                    CommonDeclarationNamedKind::Enum(decl) => {
                        let &EnumDeclaration {
                            span: _,
                            id,
                            ref params,
                            ref variants,
                        } = decl;

                        let mut seq = vec![token(TT::Enum), HNode::Space, self.fmt_maybe_id(id)];
                        if let Some(params) = params {
                            seq.push(self.fmt_parameters(params));
                        }
                        seq.push(HNode::Space);
                        seq.push(self.fmt_extra_list(SurroundKind::Curly, true, variants, &|variant| {
                            let &EnumVariant { span: _, id, content } = variant;

                            let node_id = self.fmt_id(id);
                            match content {
                                None => node_id,
                                Some(content) => HNode::Sequence(vec![
                                    node_id,
                                    surrounded_group_indent(SurroundKind::Round, self.fmt_expr(content)),
                                ]),
                            }
                        }));
                        seq.push(HNode::AlwaysNewline);
                        HNode::Sequence(seq)
                    }
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
                        nodes.push(self.fmt_block(body));
                        nodes.push(HNode::AlwaysNewline);

                        HNode::Sequence(nodes)
                    }
                };

                let mut seq = vec![];
                vis.to_seq(&mut seq);

                seq.push(node_kind);
                HNode::Sequence(seq)
            }
            CommonDeclaration::ConstBlock(block) => {
                let ConstBlock { span_keyword: _, block } = block;
                HNode::Sequence(vec![
                    token(TT::Const),
                    HNode::Space,
                    self.fmt_block(block),
                    HNode::AlwaysNewline,
                ])
            }
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
        vis.to_seq(&mut seq);

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
        seq.push(self.fmt_extra_list(SurroundKind::Round, true, ports, &|port| {
            self.fmt_module_port_item(port)
        }));

        if let Some(body) = body {
            let Block { span: _, statements } = body;
            let node_block = fmt_block_impl(statements, None, |stmt| self.fmt_module_statement(stmt));

            seq.push(HNode::Space);
            seq.push(node_block);
        }

        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_interface_decl(&self, decl: &ItemDefInterface) -> HNode {
        let &ItemDefInterface {
            span: _,
            vis,
            id,
            ref params,
            span_body: _,
            ref port_types,
            ref views,
        } = decl;

        let mut seq = vec![];
        vis.to_seq(&mut seq);
        seq.push(token(TT::Interface));
        seq.push(HNode::Space);
        seq.push(self.fmt_maybe_id(id));
        if let Some(params) = params {
            seq.push(self.fmt_parameters(params));
        }

        let body_node = {
            let mut body_seq = vec![];
            body_seq.push(self.fmt_extra_list_inner(true, port_types, &|&(port_id, port_ty)| {
                HNode::Sequence(vec![self.fmt_id(port_id), wrapping_type(self.fmt_expr(port_ty))])
            }));

            if !port_types.items.is_empty() && !views.is_empty() {
                body_seq.push(HNode::AlwaysBlankLine);
                body_seq.push(HNode::PreserveBlankLines { last: false });
            }

            for (view, last) in views.iter().with_last() {
                body_seq.push(self.fmt_interface_view_decl(view));
                if !last {
                    body_seq.push(HNode::AlwaysNewline);
                }
                body_seq.push(HNode::PreserveBlankLines { last });
            }

            surrounded_group_indent(SurroundKind::Curly, HNode::Sequence(body_seq))
        };

        seq.push(HNode::Space);
        seq.push(body_node);
        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_interface_view_decl(&self, view: &InterfaceView) -> HNode {
        let &InterfaceView {
            span: _,
            id: view_id,
            ref port_dirs,
        } = view;

        let token_ports = self.fmt_extra_list(SurroundKind::Curly, true, port_dirs, &|&(port_id, port_dir)| {
            HNode::Sequence(vec![
                self.fmt_id(port_id),
                token(TT::Colon),
                HNode::Space,
                token(port_dir.inner.token()),
            ])
        });

        HNode::Sequence(vec![
            HNode::ForceWrap,
            token(TT::Interface),
            HNode::Space,
            self.fmt_maybe_id(view_id),
            HNode::Space,
            token_ports,
        ])
    }

    fn fmt_parameters(&self, params: &Parameters) -> HNode {
        let Parameters { span: _, items } = params;
        self.fmt_extra_list(SurroundKind::Round, false, items, &|p| self.fmt_parameter(p))
    }

    fn fmt_parameter(&self, param: &Parameter) -> HNode {
        let &Parameter {
            span: _,
            id,
            ty,
            default,
        } = param;

        let mut seq = vec![self.fmt_id(id)];
        seq.push(wrapping_type(self.fmt_expr(ty)));
        if let Some(default) = default {
            seq.push(wrapping_assign(self.fmt_expr(default)));
        }
        HNode::Sequence(seq)
    }

    fn fmt_module_port_item(&self, item: &ModulePortItem) -> HNodeAndComma {
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

                let node = HNode::Sequence(vec![self.fmt_id(id), wrapping_type(node_kind)]);
                HNodeAndComma { node, comma: true }
            }
            ModulePortItem::Block(block) => {
                let ModulePortBlock { span: _, domain, ports } = block;

                let node_domain = self.fmt_domain(domain.inner);
                let node_ports = self.fmt_extra_list(SurroundKind::Curly, true, ports, &|port| {
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
                    HNode::Sequence(vec![self.fmt_id(id), wrapping_type(node_kind)])
                });

                let node = HNode::Sequence(vec![node_domain, HNode::Space, node_ports]);
                HNodeAndComma { node, comma: false }
            }
        }
    }

    fn fmt_module_block(&self, block: &Block<ModuleStatement>) -> HNode {
        let Block { span: _, statements } = block;
        fmt_block_impl(statements, None, |stmt| self.fmt_module_statement(stmt))
    }

    fn fmt_block(&self, block: &Block<BlockStatement>) -> HNode {
        let Block { span: _, statements } = block;
        self.fmt_block_ext(statements, None)
    }

    fn fmt_block_ext(&self, statements: &[BlockStatement], final_expression: Option<Expression>) -> HNode {
        let node_final_expression = final_expression.map(|expr| self.fmt_expr(expr));
        let f = |stmt: &BlockStatement| self.fmt_block_statement(stmt);
        fmt_block_impl(statements, node_final_expression, f)
    }

    fn fmt_module_statement(&self, stmt: &ModuleStatement) -> HNode {
        match &stmt.inner {
            ModuleStatementKind::Block(block) => {
                let node_block = self.fmt_module_block(block);
                HNode::Sequence(vec![node_block, HNode::AlwaysNewline])
            }
            ModuleStatementKind::If(stmt) => self.fmt_if(stmt, |b| self.fmt_module_block(b)),
            ModuleStatementKind::For(stmt) => self.fmt_for(stmt, |b| self.fmt_module_block(b)),
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
                vis.to_seq(&mut seq);

                seq.push(token(TT::Reg));
                seq.push(HNode::Space);
                seq.push(self.fmt_maybe_general_id(id));

                let mut sync_ty_seq = vec![];
                if let Some(sync) = sync {
                    sync_ty_seq.push(self.fmt_domain(DomainKind::Sync(sync.inner)));
                    sync_ty_seq.push(HNode::Space);
                }
                sync_ty_seq.push(self.fmt_expr(ty));
                seq.push(wrapping_type(HNode::Sequence(sync_ty_seq)));

                seq.push(wrapping_assign(self.fmt_expr(init)));

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
                vis.to_seq(&mut seq);

                seq.push(token(TT::Wire));
                seq.push(HNode::Space);
                seq.push(self.fmt_maybe_general_id(id));

                match kind {
                    WireDeclarationKind::Normal {
                        domain_ty,
                        assign_span_and_value,
                    } => {
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
                            seq.push(wrapping_type(domain_ty_node));
                        }
                        if let Some((_span, value)) = assign_span_and_value {
                            seq.push(wrapping_assign(self.fmt_expr(value)));
                        }
                    }
                    WireDeclarationKind::Interface {
                        domain,
                        span_keyword: _,
                        interface,
                    } => {
                        let mut domain_interface_seq = vec![];
                        if let Some(domain) = domain {
                            domain_interface_seq.push(self.fmt_domain(domain.inner));
                            domain_interface_seq.push(HNode::Space);
                        }
                        domain_interface_seq.push(token(TT::Interface));
                        domain_interface_seq.push(HNode::Space);
                        domain_interface_seq.push(self.fmt_expr(interface));

                        seq.push(wrapping_type(HNode::Sequence(domain_interface_seq)))
                    }
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
                    self.fmt_block(block),
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
                    self.fmt_block(block),
                    HNode::AlwaysNewline,
                ])
            }
            ModuleStatementKind::Instance(instance) => {
                let &ModuleInstance {
                    name,
                    span_keyword: _,
                    module,
                    ref port_connections,
                } = instance;

                let mut seq = vec![token(TT::Instance)];
                if let Some(name) = name {
                    seq.push(HNode::Space);
                    seq.push(self.fmt_id(name));
                    seq.push(HNode::Space);
                    seq.push(token(TT::Eq));
                }

                seq.push(HNode::Space);
                seq.push(self.fmt_expr(module));

                seq.push(HNode::Space);
                seq.push(token(TT::Ports));
                seq.push(fmt_comma_list(
                    SurroundKind::Round,
                    &port_connections.inner,
                    |connection| {
                        // TODO if id and expression match, collapse them automatically
                        let PortConnection { id, expr } = connection.inner;

                        let node_id = self.fmt_id(id);
                        match expr {
                            PortConnectionExpression::FakeId(_) => node_id,
                            PortConnectionExpression::Real(expr) => {
                                HNode::Sequence(vec![node_id, token(TT::Eq), self.fmt_expr(expr)])
                            }
                        }
                    },
                ));

                seq.push(token(TT::Semi));
                seq.push(HNode::AlwaysNewline);
                HNode::Sequence(seq)
            }
        }
    }

    fn fmt_block_statement(&self, stmt: &BlockStatement) -> HNode {
        match &stmt.inner {
            BlockStatementKind::CommonDeclaration(decl) => self.fmt_common_decl(decl),
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

                let op_token = op.inner.map_or(TT::Eq, AssignBinaryOp::token);
                HNode::Sequence(vec![
                    self.fmt_expr(target),
                    wrapping_assign_op(op_token, self.fmt_expr(value)),
                    token(TT::Semi),
                    HNode::AlwaysNewline,
                ])
            }
            &BlockStatementKind::Expression(expr) => {
                HNode::Sequence(vec![self.fmt_expr(expr), token(TT::Semi), HNode::AlwaysNewline])
            }
            BlockStatementKind::Block(block) => HNode::Sequence(vec![self.fmt_block(block), HNode::AlwaysNewline]),
            BlockStatementKind::If(stmt) => self.fmt_if(stmt, |block| self.fmt_block(block)),
            BlockStatementKind::Match(stmt) => self.fmt_match(stmt, |block| self.fmt_block(block)),
            BlockStatementKind::For(stmt) => self.fmt_for(stmt, |block| self.fmt_block(block)),
            BlockStatementKind::While(stmt) => self.fmt_while(stmt),
            BlockStatementKind::Return(stmt) => {
                let &ReturnStatement { span_return: _, value } = stmt;
                let mut seq = vec![token(TT::Return)];
                if let Some(value) = value {
                    seq.push(HNode::Space);
                    seq.push(self.fmt_expr(value));
                }
                seq.push(token(TT::Semi));
                seq.push(HNode::AlwaysNewline);
                HNode::Sequence(seq)
            }
            BlockStatementKind::Break { span: _ } => {
                HNode::Sequence(vec![token(TT::Break), token(TT::Semi), HNode::AlwaysNewline])
            }
            BlockStatementKind::Continue { span: _ } => {
                HNode::Sequence(vec![token(TT::Continue), token(TT::Semi), HNode::AlwaysNewline])
            }
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
            seq.push(wrapping_type(self.fmt_expr(ty)));
        }
        if let Some(value) = value {
            seq.push(wrapping_assign(self.fmt_expr(value)));
        }
        seq.push(token(TT::Semi));
        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_if<B>(&self, stmt: &IfStatement<B>, f: impl Fn(&B) -> HNode) -> HNode {
        fn fmt_else(seq: &mut Vec<HNode>) {
            seq.push(HNode::Space);
            seq.push(token(TT::Else));
            seq.push(HNode::Space);
        }

        fn fmt_pair<B>(slf: &Context, f: impl Fn(&B) -> HNode, seq: &mut Vec<HNode>, pair: &IfCondBlockPair<B>) {
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
            seq.push(f(block));
        }

        let IfStatement {
            span: _,
            initial_if,
            else_ifs,
            final_else,
        } = stmt;

        let mut seq = vec![];
        fmt_pair(self, &f, &mut seq, initial_if);
        for else_if in else_ifs {
            fmt_else(&mut seq);
            fmt_pair(self, &f, &mut seq, else_if);
        }
        if let Some(final_else) = final_else {
            fmt_else(&mut seq);
            seq.push(f(final_else));
        }

        seq.push(HNode::AlwaysNewline);
        HNode::Sequence(seq)
    }

    fn fmt_match<B>(&self, stmt: &MatchStatement<B>, f: impl Fn(&B) -> HNode) -> HNode {
        let &MatchStatement {
            target,
            span_branches: _,
            ref branches,
        } = stmt;

        let mut seq_branches = vec![];
        for (branch, last) in branches.iter().with_last() {
            let MatchBranch { pattern, block } = branch;

            let pattern_node = match pattern.inner {
                MatchPattern::Wildcard => token(TT::Underscore),
                MatchPattern::Val(id) => HNode::Sequence(vec![token(TT::Val), HNode::Space, self.fmt_id(id)]),
                MatchPattern::Equal(value) => self.fmt_expr(value),
                MatchPattern::In(value) => HNode::Sequence(vec![token(TT::In), HNode::Space, self.fmt_expr(value)]),
                MatchPattern::EnumVariant(variant, content) => {
                    // TODO check that fuzzing would find a missing `val`
                    let node_variant = HNode::Sequence(vec![token(TT::Dot), self.fmt_id(variant)]);
                    match content {
                        None => node_variant,
                        Some(content) => {
                            let node_content = match content {
                                MaybeIdentifier::Dummy { span: _ } => token(TT::Underscore),
                                MaybeIdentifier::Identifier(id) => {
                                    HNode::Sequence(vec![token(TT::Val), HNode::Space, self.fmt_id(id)])
                                }
                            };
                            HNode::Sequence(vec![
                                node_variant,
                                surrounded_group_indent(SurroundKind::Round, node_content),
                            ])
                        }
                    }
                }
            };

            seq_branches.push(pattern_node);
            seq_branches.push(HNode::Space);
            seq_branches.push(token(TT::DoubleArrow));
            seq_branches.push(HNode::Space);
            seq_branches.push(f(block));
            seq_branches.push(HNode::AlwaysNewline);
            if !last {
                seq_branches.push(HNode::PreserveBlankLines { last: false });
            }
        }

        HNode::Sequence(vec![
            token(TT::Match),
            HNode::Space,
            surrounded_group_indent(SurroundKind::Round, self.fmt_expr(target)),
            HNode::Space,
            surrounded_group_indent(SurroundKind::Curly, HNode::Sequence(seq_branches)),
            HNode::AlwaysNewline,
        ])
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
            seq.push(wrapping_type(self.fmt_expr(index_ty)));
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

    fn fmt_while(&self, stmt: &WhileStatement) -> HNode {
        let &WhileStatement {
            span_keyword: _,
            cond,
            ref body,
        } = stmt;

        HNode::Sequence(vec![
            token(TT::While),
            HNode::Space,
            surrounded_group_indent(SurroundKind::Round, self.fmt_expr(cond)),
            HNode::Space,
            self.fmt_block(body),
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
                self.fmt_block_ext(statements, Some(expression))
            }
            &ExpressionKind::Id(id) => self.fmt_general_id(id),
            ExpressionKind::IntLiteral(literal) => {
                let tt = match literal {
                    IntLiteral::Binary { span: _ } => TT::IntLiteralBinary,
                    IntLiteral::Decimal { span: _ } => TT::IntLiteralDecimal,
                    IntLiteral::Hexadecimal { span: _ } => TT::IntLiteralHexadecimal,
                };
                token(tt)
            }
            &ExpressionKind::BoolLiteral(value) => {
                let tt = if value { TT::True } else { TT::False };
                token(tt)
            }
            ExpressionKind::StringLiteral(pieces) => {
                // dedent all nodes that could follow StringMiddle to avoid any indentation sneaking into the string
                let mut seq = vec![token(TT::StringStart)];
                for piece in pieces {
                    match piece {
                        StringPiece::Literal { span: _ } => {
                            seq.push(HNode::Dedent(Box::new(token(TT::StringMiddle))));
                        }
                        &StringPiece::Substitute(expr) => {
                            seq.push(HNode::Dedent(Box::new(token(TT::StringSubStart))));
                            seq.push(group_indent_seq(vec![
                                HNode::WrapNewline,
                                self.fmt_expr(expr),
                                HNode::WrapNewline,
                            ]));
                            seq.push(token(TT::StringSubEnd));
                        }
                    }
                }
                seq.push(HNode::Dedent(Box::new(token(TT::StringEnd))));
                HNode::Sequence(seq)
            }
            ExpressionKind::ArrayLiteral(elements) => fmt_comma_list(SurroundKind::Square, elements, |&elem| {
                self.fmt_array_literal_element(elem)
            }),
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
                    elements => fmt_comma_list(SurroundKind::Round, elements, |&elem| self.fmt_expr(elem)),
                }
            }
            &ExpressionKind::RangeLiteral(expr) => match expr {
                RangeLiteral::ExclusiveEnd { op_span: _, start, end } => wrapping_binary_op(
                    token(TT::DotDot),
                    start.map(|e| self.fmt_expr(e)),
                    end.map(|e| self.fmt_expr(e)),
                ),
                RangeLiteral::InclusiveEnd { op_span: _, start, end } => wrapping_binary_op(
                    token(TT::DotDotEq),
                    start.map(|e| self.fmt_expr(e)),
                    Some(self.fmt_expr(end)),
                ),
                RangeLiteral::Length { op_span: _, start, len } => wrapping_binary_op(
                    token(TT::PlusDotDot),
                    Some(self.fmt_expr(start)),
                    Some(self.fmt_expr(len)),
                ),
            },
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
            &ExpressionKind::BinaryOp(op, _, _) => {
                let mut seq = vec![];
                let leftmost = self.collect_binary_expr::<LeftmostYes>(&mut seq, op.inner.level(), expr);
                binary_indent_seq(leftmost, seq)
            }
            &ExpressionKind::ArrayType(ref lengths, base) => {
                let node_lengths = fmt_comma_list(SurroundKind::Square, &lengths.inner, |&len| {
                    self.fmt_array_literal_element(len)
                });
                HNode::Sequence(vec![node_lengths, self.fmt_expr(base)])
            }
            &ExpressionKind::ArrayIndex(base, ref indices) => {
                let node_indices = fmt_comma_list(SurroundKind::Square, &indices.inner, |&index| self.fmt_expr(index));
                HNode::Sequence(vec![self.fmt_expr(base), node_indices])
            }
            ExpressionKind::DotIndex(_, _) => {
                // repeated dot indices should wrap together
                let mut base = expr;
                let mut indices_rev = vec![];
                while let &ExpressionKind::DotIndex(curr_base, curr_index) = &self.arena_expressions[base.inner] {
                    base = curr_base;
                    indices_rev.push(curr_index);
                }

                let node_base = self.fmt_expr(base);

                let mut seq = vec![];
                for (index, last) in indices_rev.into_iter().rev().with_last() {
                    let node_index = match index {
                        DotIndexKind::Id(id) => self.fmt_id(id),
                        DotIndexKind::Int { span: _ } => token(TT::IntLiteralDecimal),
                    };

                    seq.push(token(TT::Dot));
                    seq.push(node_index);

                    if !last {
                        seq.push(HNode::WrapNewline);
                    }
                }
                let node_indices = HNode::Sequence(seq);

                wrapping_binary_op(HNode::EMPTY, Some(node_base), Some(node_indices))
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

    fn collect_binary_expr<L: LeftmostMaybe>(
        &self,
        seq: &mut Vec<HNode>,
        level: BinaryOpLevel,
        expr: Expression,
    ) -> L::R {
        if let &ExpressionKind::BinaryOp(op, left, right) = &self.arena_expressions[expr.inner]
            && op.inner.level() == level
        {
            let leftmost = self.collect_binary_expr::<L>(seq, level, left);

            seq.push(HNode::WrapNewline);
            let op_node = token(op.inner.token());
            match op.inner {
                BinaryOp::Pow => {
                    seq.push(op_node);
                }
                _ => {
                    seq.push(HNode::Space);
                    seq.push(op_node);
                    seq.push(HNode::Space);
                }
            };

            self.collect_binary_expr::<LeftmostNo>(seq, level, right);

            leftmost
        } else {
            let expr_node = self.fmt_expr(expr);
            L::push_or_return(seq, expr_node)
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
            MaybeGeneralIdentifier::Dummy { span: _ } => token(TT::Underscore),
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
            MaybeIdentifier::Dummy { span: _ } => token(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.fmt_id(id),
        }
    }

    fn fmt_id(&self, id: Identifier) -> HNode {
        let _ = id;
        token(TT::Identifier)
    }
}

fn fmt_block_impl<T>(statements: &[T], final_expression: Option<HNode>, f: impl Fn(&T) -> HNode) -> HNode {
    let mut seq = vec![];

    for (stmt, last_stmt) in statements.iter().with_last() {
        seq.push(f(stmt));
        let last = last_stmt && final_expression.is_none();
        seq.push(HNode::PreserveBlankLines { last });
    }

    if let Some(final_expression) = final_expression {
        seq.push(HNode::Space);
        seq.push(final_expression);
        seq.push(HNode::WrapNewline);
        seq.push(HNode::Space);
    }

    if statements.is_empty() {
        seq.push(HNode::PreserveBlankLines { last: true });
    }

    surrounded_group_indent(SurroundKind::Curly, HNode::Sequence(seq))
}

fn fmt_call<T>(target: HNode, args: &[T], f: impl Fn(&T) -> HNode) -> HNode {
    HNode::Sequence(vec![target, fmt_comma_list(SurroundKind::Round, args, f)])
}

fn fmt_comma_list<T>(surround: SurroundKind, items: &[T], f: impl Fn(&T) -> HNode) -> HNode {
    let mut seq = vec![];

    for (item, last) in items.iter().with_last() {
        seq.push(f(item));
        seq.push(comma_nodes(last));
        seq.push(HNode::WrapNewline);
        seq.push(HNode::PreserveBlankLines { last });
    }

    surrounded_group_indent(surround, HNode::Sequence(seq))
}

fn comma_nodes(last: bool) -> HNode {
    if !last {
        HNode::Sequence(vec![token(TT::Comma), HNode::Space])
    } else {
        HNode::WrapComma
    }
}

fn wrapping_type(ty: HNode) -> HNode {
    HNode::Sequence(vec![
        group_indent_seq(vec![HNode::WrapNewline, token(TT::Colon), HNode::Space]),
        ty,
    ])
}

fn wrapping_assign(value: HNode) -> HNode {
    wrapping_assign_op(TT::Eq, value)
}

fn wrapping_assign_op(op: TT, value: HNode) -> HNode {
    HNode::Sequence(vec![
        group_indent_seq(vec![HNode::WrapNewline, HNode::Space, token(op), HNode::Space]),
        value,
    ])
}

fn wrapping_binary_op(op: HNode, left: Option<HNode>, right: Option<HNode>) -> HNode {
    match (left, right) {
        (Some(left), Some(right)) => {
            // both left and right hand side, provide some wrapping opportunities
            binary_indent_seq(left, vec![HNode::WrapNewline, op, right])
        }
        (left, right) => {
            // only an operator and one side, just flatten to a sequence without wrapping
            let mut seq = vec![];
            if let Some(left) = left {
                seq.push(left);
            }
            seq.push(op);
            if let Some(right) = right {
                seq.push(right);
            }
            HNode::Sequence(seq)
        }
    }
}

fn group_seq(nodes: Vec<HNode>) -> HNode {
    HNode::Group(Box::new(HNode::Sequence(nodes)))
}

fn group_indent_seq(nodes: Vec<HNode>) -> HNode {
    HNode::Group(Box::new(HNode::WrapIndent(Box::new(HNode::Sequence(nodes)))))
}

fn binary_indent_seq(leftmost: HNode, rest: Vec<HNode>) -> HNode {
    group_seq(vec![leftmost, HNode::WrapIndent(Box::new(HNode::Sequence(rest)))])
}

fn surrounded_group_indent(surround: SurroundKind, inner: HNode) -> HNode {
    // the before/after tokens should not be part of the group,
    //   since then trailing line comments after `after` would force the entire group to wrap
    let (before, after) = surround.before_after();
    group_seq(vec![
        token(before),
        HNode::WrapNewline,
        HNode::WrapIndent(Box::new(inner)),
        HNode::WrapNewline,
        token(after),
    ])
}

fn token(tt: TT) -> HNode {
    HNode::AlwaysToken(tt)
}

enum SurroundKind {
    Round,
    Square,
    Curly,
}

impl SurroundKind {
    pub fn before_after(self) -> (TT, TT) {
        match self {
            SurroundKind::Round => (TT::OpenR, TT::CloseR),
            SurroundKind::Square => (TT::OpenS, TT::CloseS),
            SurroundKind::Curly => (TT::OpenC, TT::CloseC),
        }
    }
}

trait FormatVisibility: Copy {
    fn to_seq(self, seq: &mut Vec<HNode>);
}

impl FormatVisibility for Visibility {
    fn to_seq(self, seq: &mut Vec<HNode>) {
        match self {
            Visibility::Public { span: _ } => {
                seq.push(token(TT::Pub));
                seq.push(HNode::Space);
            }
            Visibility::Private => {
                // this is the default, do nothing
            }
        }
    }
}

impl FormatVisibility for () {
    fn to_seq(self, _: &mut Vec<HNode>) {
        // do nothing
    }
}

struct HNodeAndComma {
    node: HNode,
    comma: bool,
}

impl From<HNode> for HNodeAndComma {
    fn from(node: HNode) -> Self {
        Self { node, comma: true }
    }
}

trait LeftmostMaybe {
    type R;
    fn push_or_return(seq: &mut Vec<HNode>, v: HNode) -> Self::R;
}

struct LeftmostYes;
struct LeftmostNo;
impl LeftmostMaybe for LeftmostYes {
    type R = HNode;
    fn push_or_return(_: &mut Vec<HNode>, v: HNode) -> Self::R {
        v
    }
}
impl LeftmostMaybe for LeftmostNo {
    type R = ();
    fn push_or_return(seq: &mut Vec<HNode>, v: HNode) -> Self::R {
        seq.push(v);
    }
}
