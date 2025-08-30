use crate::syntax::ast::{
    ArenaExpressions, Arg, Args, ArrayComprehension, ArrayLiteralElement, Block, BlockExpression, BlockStatement,
    CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration, DotIndexKind, Expression,
    ExpressionKind, ExtraItem, ExtraList, FileContent, FunctionDeclaration, GeneralIdentifier, Identifier, ImportEntry,
    ImportFinalKind, IntLiteral, Item, ItemImport, MaybeIdentifier, Parameter, Parameters, RangeLiteral, Visibility,
};
use crate::syntax::format_new::high::HNode;
use crate::syntax::token::TokenType as TT;
use crate::util::iter::IterExt;

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
        let mut nodes = vec![];
        for (item, last) in items.iter().with_last() {
            let item_node = match item {
                Item::Import(import) => self.fmt_import(import),
                Item::CommonDeclaration(decl) => self.fmt_common_decl(&decl.inner),
                Item::ModuleInternal(_) => todo!(),
                Item::ModuleExternal(_) => todo!(),
                Item::Interface(_) => todo!(),
            };
            nodes.push(item_node);

            if !last {
                nodes.push(HNode::PreserveBlankLines)
            }
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
                        let mut seq = vec![];

                        seq.push(token(TT::Const));
                        seq.push(HNode::Space);
                        seq.push(self.fmt_maybe_id(id));

                        if let Some(ty) = ty {
                            seq.push(group_indent_seq(vec![
                                HNode::WrapNewline,
                                token(TT::Colon),
                                HNode::Space,
                            ]));
                            seq.push(self.fmt_expr(ty));
                        }

                        seq.push(group_indent_seq(vec![
                            HNode::WrapNewline,
                            HNode::Space,
                            token(TT::Eq),
                            HNode::Space,
                        ]));
                        seq.push(self.fmt_expr(value));

                        seq.push(token(TT::Semi));
                        seq.push(HNode::AlwaysNewline);

                        HNode::Sequence(seq)
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
                        let Block { span: _, statements } = body;
                        nodes.push(self.fmt_block(statements, None, false));
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
        let mut nodes = vec![self.fmt_id(id), token(TT::Colon), HNode::Space, self.fmt_expr(ty)];
        if let Some(default) = default {
            nodes.push(HNode::Space);
            nodes.push(token(TT::Eq));
            nodes.push(HNode::Space);
            nodes.push(self.fmt_expr(default));
        }
        HNode::Sequence(nodes)
    }

    fn fmt_block(
        &self,
        statements: &[BlockStatement],
        final_expression: Option<Expression>,
        force_wrap: bool,
    ) -> HNode {
        // TODO for else-if blocks, force a newline inside the block to avoid infinite clutter?
        if !statements.is_empty() {
            todo!()
        }
        if final_expression.is_some() {
            todo!()
        }
        if force_wrap {
            todo!()
        }
        HNode::Sequence(vec![token(TT::OpenC), token(TT::CloseC)])
    }

    fn fmt_expr(&self, expr: Expression) -> HNode {
        match &self.arena_expressions[expr.inner] {
            ExpressionKind::Dummy => token(TT::Underscore),
            ExpressionKind::Undefined => token(TT::Undef),
            ExpressionKind::Type => token(TT::Type),
            ExpressionKind::TypeFunction => token(TT::Fn),
            ExpressionKind::Builtin => token(TT::Builtin),
            &ExpressionKind::Wrapped(inner) => surrounded_group_indent(TT::OpenR, self.fmt_expr(inner), TT::CloseR),
            ExpressionKind::Block(expr) => {
                let &BlockExpression {
                    ref statements,
                    expression,
                } = expr;
                self.fmt_block(statements, Some(expression), false)
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
                let seq = match elements.as_slice() {
                    // don't allow breaking for empty tuple
                    [] => HNode::Sequence(vec![token(TT::OpenR), token(TT::CloseR)]),
                    // force trailing comma for single-element tuple
                    &[element] => surrounded_group_indent(
                        TT::OpenR,
                        HNode::Sequence(vec![self.fmt_expr(element), token(TT::Comma)]),
                        TT::CloseR,
                    ),
                    // general case
                    elements => {
                        todo!()
                    }
                };
                (seq)
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

                surrounded_group_indent(TT::OpenS, HNode::Sequence(seq), TT::CloseS)
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
                let node_target = self.fmt_expr(target);

                let Args { span: _, inner } = args;
                let node_arg_list = fmt_comma_list(inner, |arg| {
                    let &Arg { span: _, name, value } = arg;
                    let node_value = self.fmt_expr(value);
                    if let Some(name) = name {
                        // TODO allow wrapping before/after "="?
                        HNode::Sequence(vec![self.fmt_id(name), token(TT::Eq), node_value])
                    } else {
                        node_value
                    }
                });
                let node_args = HNode::Sequence(vec![token(TT::OpenR), node_arg_list, token(TT::CloseR)]);

                HNode::Sequence(vec![node_target, node_args])
            }
            ExpressionKind::UnsafeValueWithDomain(_, _) => todo!(),
            ExpressionKind::RegisterDelay(_) => todo!(),
        }
    }

    fn fmt_array_literal_element(&self, elem: ArrayLiteralElement<Expression>) -> HNode {
        match elem {
            ArrayLiteralElement::Single(elem) => self.fmt_expr(elem),
            ArrayLiteralElement::Spread(_span, elem) => HNode::Sequence(vec![token(TT::Star), self.fmt_expr(elem)]),
        }
    }

    fn fmt_general_id(&self, id: GeneralIdentifier) -> HNode {
        match id {
            GeneralIdentifier::Simple(id) => self.fmt_id(id),
            GeneralIdentifier::FromString(_, _) => todo!(),
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

fn fmt_extra_list<T>(list: &ExtraList<T>, f: &impl Fn(&T) -> HNode) -> HNode {
    // TODO variant that always wraps, for eg. module ports? or not, maybe it's more elegant if we don't
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
        if !last {
            nodes.push(HNode::PreserveBlankLines);
        }
    }

    group_indent_seq(nodes)
}

fn fmt_comma_list<T>(items: &[T], f: impl Fn(&T) -> HNode) -> HNode {
    let mut nodes = vec![];
    nodes.push(HNode::WrapNewline);
    for (item, last) in items.iter().with_last() {
        nodes.push(f(item));
        push_comma_nodes(last, &mut nodes);
        nodes.push(HNode::WrapNewline);
        if !last {
            nodes.push(HNode::PreserveBlankLines);
        }
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

fn surrounded_group_indent(before: TT, inner: HNode, after: TT) -> HNode {
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
