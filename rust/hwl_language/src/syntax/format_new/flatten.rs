use crate::syntax::ast::{
    ArenaExpressions, Arg, Args, ArrayLiteralElement, Block, BlockStatement, CommonDeclaration, CommonDeclarationNamed,
    CommonDeclarationNamedKind, ConstDeclaration, Expression, ExpressionKind, ExtraItem, ExtraList, FileContent,
    FunctionDeclaration, GeneralIdentifier, Identifier, ImportEntry, ImportFinalKind, IntLiteral, Item, ItemImport,
    MaybeIdentifier, Parameter, Parameters, Visibility,
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
        nodes.push(HNode::AlwaysToken(TT::Import));
        nodes.push(HNode::Space);

        for &parent in &parents.inner {
            nodes.push(self.fmt_id(parent));
            nodes.push(HNode::AlwaysToken(TT::Dot));
        }

        match &entry.inner {
            ImportFinalKind::Single(entry) => {
                nodes.push(self.fmt_import_entry(entry));
            }
            ImportFinalKind::Multi(entries) => {
                nodes.push(HNode::AlwaysToken(TT::OpenS));

                let mut nodes_fill = vec![];
                for (entry, last) in entries.iter().with_last() {
                    let mut seq = vec![self.fmt_import_entry(entry)];
                    push_comma_nodes(last, &mut seq);
                    nodes_fill.push(HNode::Sequence(seq));
                }

                nodes.push(HNode::Fill(nodes_fill));
                nodes.push(HNode::AlwaysToken(TT::CloseS));
            }
        }

        nodes.push(HNode::AlwaysToken(TT::Semi));
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
            nodes.push(HNode::AlwaysToken(TT::As));
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

                        seq.push(HNode::AlwaysToken(TT::Const));
                        seq.push(HNode::Space);
                        seq.push(self.fmt_maybe_id(id));

                        if let Some(ty) = ty {
                            seq.push(group_indent_sequence(vec![
                                HNode::WrapNewline,
                                HNode::AlwaysToken(TT::Colon),
                                HNode::Space,
                            ]));
                            seq.push(self.fmt_expr(ty));
                        }

                        seq.push(group_indent_sequence(vec![
                            HNode::WrapNewline,
                            HNode::Space,
                            HNode::AlwaysToken(TT::Eq),
                            HNode::Space,
                        ]));
                        seq.push(self.fmt_expr(value));

                        seq.push(HNode::AlwaysToken(TT::Semi));
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
                            HNode::AlwaysToken(TT::Function),
                            HNode::Space,
                            self.fmt_maybe_id(id),
                            self.fmt_parameters(params),
                        ];
                        if let Some(ret_ty) = ret_ty {
                            nodes.push(HNode::Space);
                            nodes.push(HNode::AlwaysToken(TT::Arrow));
                            nodes.push(HNode::Space);
                            nodes.push(self.fmt_expr(ret_ty));
                        }
                        nodes.push(HNode::Space);
                        nodes.push(self.fmt_block(body));
                        nodes.push(HNode::AlwaysNewline);
                        HNode::Sequence(nodes)
                    }
                };

                match node_vis {
                    None => node_kind,
                    Some(token_vis) => HNode::Sequence(vec![HNode::AlwaysToken(token_vis), node_kind]),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }
    }

    fn fmt_parameters(&self, params: &Parameters) -> HNode {
        let Parameters { span: _, items } = params;
        HNode::Sequence(vec![
            HNode::AlwaysToken(TT::OpenR),
            fmt_extra_list(items, &|p| self.fmt_parameter(p)),
            HNode::AlwaysToken(TT::CloseR),
        ])
    }

    fn fmt_parameter(&self, param: &Parameter) -> HNode {
        let &Parameter {
            span: _,
            id,
            ty,
            default,
        } = param;
        let mut nodes = vec![
            self.fmt_id(id),
            HNode::AlwaysToken(TT::Colon),
            HNode::Space,
            self.fmt_expr(ty),
        ];
        if let Some(default) = default {
            nodes.push(HNode::Space);
            nodes.push(HNode::AlwaysToken(TT::Eq));
            nodes.push(HNode::Space);
            nodes.push(self.fmt_expr(default));
        }
        HNode::Sequence(nodes)
    }

    fn fmt_block(&self, block: &Block<BlockStatement>) -> HNode {
        // TODO for else-if blocks, force a newline inside the block to avoid infinite clutter?
        if !block.statements.is_empty() {
            todo!()
        }
        HNode::Sequence(vec![HNode::AlwaysToken(TT::OpenC), HNode::AlwaysToken(TT::CloseC)])
    }

    fn fmt_expr(&self, expr: Expression) -> HNode {
        match &self.arena_expressions[expr.inner] {
            ExpressionKind::Dummy => todo!(),
            ExpressionKind::Undefined => todo!(),
            ExpressionKind::Type => todo!(),
            ExpressionKind::TypeFunction => todo!(),
            ExpressionKind::Wrapped(_) => todo!(),
            ExpressionKind::Block(_) => todo!(),
            &ExpressionKind::Id(id) => self.fmt_general_id(id),
            ExpressionKind::IntLiteral(literal) => {
                let tt = match literal {
                    IntLiteral::Binary(_span) => TT::IntLiteralBinary,
                    IntLiteral::Decimal(_span) => TT::IntLiteralDecimal,
                    IntLiteral::Hexadecimal(_span) => TT::IntLiteralHexadecimal,
                };
                HNode::AlwaysToken(tt)
            }
            ExpressionKind::BoolLiteral(_) => todo!(),
            ExpressionKind::StringLiteral(_) => todo!(),
            ExpressionKind::ArrayLiteral(elements) => {
                let node_elements = fmt_comma_list(elements, |elem| match elem {
                    &ArrayLiteralElement::Single(elem) => self.fmt_expr(elem),
                    ArrayLiteralElement::Spread(_, _) => todo!(),
                });
                HNode::Sequence(vec![
                    HNode::AlwaysToken(TT::OpenS),
                    node_elements,
                    HNode::AlwaysToken(TT::CloseS),
                ])
            }
            ExpressionKind::TupleLiteral(_) => todo!(),
            ExpressionKind::RangeLiteral(_) => todo!(),
            ExpressionKind::ArrayComprehension(_) => todo!(),
            ExpressionKind::UnaryOp(_, _) => todo!(),
            &ExpressionKind::BinaryOp(op, left, right) => {
                let seq = vec![
                    self.fmt_expr(left),
                    HNode::Space,
                    HNode::WrapNewline,
                    HNode::WrapIndent(Box::new(HNode::Sequence(vec![
                        HNode::AlwaysToken(op.inner.token()),
                        HNode::Space,
                        self.fmt_expr(right),
                    ]))),
                ];
                HNode::Group(Box::new(HNode::Sequence(seq)))
            }
            ExpressionKind::ArrayType(_, _) => todo!(),
            ExpressionKind::ArrayIndex(_, _) => todo!(),
            ExpressionKind::DotIndex(_, _) => todo!(),
            &ExpressionKind::Call(target, ref args) => {
                let node_target = self.fmt_expr(target);

                let Args { span: _, inner } = args;
                let node_arg_list = fmt_comma_list(inner, |arg| {
                    let &Arg { span: _, name, value } = arg;
                    let node_value = self.fmt_expr(value);
                    if let Some(name) = name {
                        // TODO allow wrapping before/after "="
                        HNode::Sequence(vec![self.fmt_id(name), HNode::AlwaysToken(TT::Eq), node_value])
                    } else {
                        node_value
                    }
                });
                let node_args = HNode::Sequence(vec![
                    HNode::AlwaysToken(TT::OpenR),
                    node_arg_list,
                    HNode::AlwaysToken(TT::CloseR),
                ]);

                HNode::Sequence(vec![node_target, node_args])
            }
            ExpressionKind::Builtin(_) => todo!(),
            ExpressionKind::UnsafeValueWithDomain(_, _) => todo!(),
            ExpressionKind::RegisterDelay(_) => todo!(),
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
            MaybeIdentifier::Dummy(_span) => HNode::AlwaysToken(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.fmt_id(id),
        }
    }

    fn fmt_id(&self, id: Identifier) -> HNode {
        let _ = id;
        HNode::AlwaysToken(TT::Identifier)
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
    group_indent_sequence(nodes)
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
    group_indent_sequence(nodes)
}

fn push_comma_nodes(last: bool, seq: &mut Vec<HNode>) {
    if !last {
        seq.push(HNode::AlwaysToken(TT::Comma));
        seq.push(HNode::Space);
    } else {
        seq.push(HNode::WrapComma);
    }
}

fn group_indent_sequence(nodes: Vec<HNode>) -> HNode {
    HNode::Group(Box::new(HNode::WrapIndent(Box::new(HNode::Sequence(nodes)))))
}

trait FormatVisibility: Copy {
    fn token(self) -> Option<TT>;
}

impl FormatVisibility for Visibility {
    fn token(self) -> Option<TT> {
        match self {
            Visibility::Private => None,
            Visibility::Public(_span) => Some(TT::Public),
        }
    }
}

impl FormatVisibility for () {
    fn token(self) -> Option<TT> {
        None
    }
}
