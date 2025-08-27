use crate::syntax::ast::{
    ArenaExpressions, Arg, Args, Block, BlockStatement, CommonDeclaration, CommonDeclarationNamed,
    CommonDeclarationNamedKind, Expression, ExpressionKind, ExtraItem, ExtraList, FileContent, FunctionDeclaration,
    GeneralIdentifier, Identifier, ImportEntry, ImportFinalKind, Item, ItemImport, MaybeIdentifier, Parameter,
    Parameters, Visibility,
};
use crate::syntax::format_new::high::HCommaList;
use crate::syntax::format_new::high::HNode;
use crate::syntax::token::TokenType as TT;
use itertools::Itertools;

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
        let nodes = items
            .iter()
            .map(|item| match item {
                Item::Import(import) => self.fmt_import(import),
                Item::CommonDeclaration(decl) => self.fmt_common_decl(&decl.inner),
                Item::ModuleInternal(_) => todo!(),
                Item::ModuleExternal(_) => todo!(),
                Item::Interface(_) => todo!(),
            })
            .collect();
        HNode::Vertical(nodes)
    }

    fn fmt_import(&self, import: &ItemImport) -> HNode {
        let ItemImport {
            span: _,
            parents,
            entry,
        } = import;

        let mut nodes = vec![];
        nodes.push(HNode::Token(TT::Import));
        nodes.push(HNode::Space);

        for &parent in &parents.inner {
            nodes.push(self.fmt_id(parent));
            nodes.push(HNode::Token(TT::Dot));
        }

        match &entry.inner {
            ImportFinalKind::Single(entry) => {
                nodes.push(self.fmt_import_entry(entry));
            }
            ImportFinalKind::Multi(entries) => {
                let children = entries.iter().map(|e| self.fmt_import_entry(e)).collect();
                let list = HCommaList { fill: true, children };

                nodes.push(HNode::Token(TT::OpenS));
                nodes.push(HNode::CommaList(list));
                nodes.push(HNode::Token(TT::CloseS));
            }
        }

        nodes.push(HNode::Token(TT::Semi));
        HNode::Horizontal(nodes)
    }

    fn fmt_import_entry(&self, entry: &ImportEntry) -> HNode {
        let &ImportEntry { span: _, id, as_ } = entry;
        let mut nodes = vec![];
        nodes.push(self.fmt_id(id));
        if let Some(as_) = as_ {
            nodes.push(HNode::Space);
            nodes.push(HNode::Token(TT::As));
            nodes.push(HNode::Space);
            nodes.push(self.fmt_maybe_id(as_));
        }
        HNode::Horizontal(nodes)
    }

    fn fmt_common_decl<V: FormatVisibility>(&self, decl: &CommonDeclaration<V>) -> HNode {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis, kind } = decl;
                let node_vis = vis.token();
                let node_kind = match kind {
                    CommonDeclarationNamedKind::Type(_) => todo!(),
                    CommonDeclarationNamedKind::Const(_) => todo!(),
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
                            HNode::Token(TT::Function),
                            HNode::Space,
                            self.fmt_maybe_id(id),
                            self.fmt_parameters(params),
                        ];
                        if let Some(ret_ty) = ret_ty {
                            nodes.push(HNode::Space);
                            nodes.push(HNode::Token(TT::Arrow));
                            nodes.push(HNode::Space);
                            nodes.push(self.fmt_expr(ret_ty));
                        }
                        nodes.push(HNode::Space);
                        nodes.push(self.fmt_block(body));
                        HNode::Horizontal(nodes)
                    }
                };

                match node_vis {
                    None => node_kind,
                    Some(token_vis) => HNode::Horizontal(vec![HNode::Token(token_vis), node_kind]),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }
    }

    fn fmt_parameters(&self, params: &Parameters) -> HNode {
        let Parameters { span: _, items } = params;
        let children = fmt_extra_list(items, &|p| self.fmt_parameter(p));

        let list = HCommaList { fill: false, children };
        let node_list = HNode::CommaList(list);
        HNode::Horizontal(vec![HNode::Token(TT::OpenR), node_list, HNode::Token(TT::CloseR)])
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
            HNode::Token(TT::Colon),
            HNode::Space,
            self.fmt_expr(ty),
        ];
        if let Some(default) = default {
            nodes.push(HNode::Space);
            nodes.push(HNode::Token(TT::Eq));
            nodes.push(HNode::Space);
            nodes.push(self.fmt_expr(default));
        }
        HNode::Horizontal(nodes)
    }

    fn fmt_block(&self, block: &Block<BlockStatement>) -> HNode {
        // TODO for else-if blocks, force a newline to avoid infinite clutter
        if !block.statements.is_empty() {
            todo!()
        }
        HNode::Horizontal(vec![HNode::Token(TT::OpenC), HNode::Token(TT::CloseC)])
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
            ExpressionKind::IntLiteral(_) => todo!(),
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
            ExpressionKind::DotIndex(_, _) => todo!(),
            &ExpressionKind::Call(target, ref args) => {
                let node_target = self.fmt_expr(target);

                let Args { span: _, inner } = args;
                let nodes_arg = inner
                    .iter()
                    .map(|arg| {
                        let &Arg { span: _, name, value } = arg;
                        let node_value = self.fmt_expr(value);
                        if let Some(name) = name {
                            HNode::Horizontal(vec![self.fmt_id(name), HNode::Token(TT::Eq), node_value])
                        } else {
                            node_value
                        }
                    })
                    .collect_vec();

                let node_list = HCommaList {
                    fill: false,
                    children: nodes_arg,
                };
                let node_args = HNode::Horizontal(vec![
                    HNode::Token(TT::OpenR),
                    HNode::CommaList(node_list),
                    HNode::Token(TT::CloseR),
                ]);

                HNode::Horizontal(vec![node_target, node_args])
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
            MaybeIdentifier::Dummy(_span) => HNode::Token(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.fmt_id(id),
        }
    }

    fn fmt_id(&self, id: Identifier) -> HNode {
        let _ = id;
        HNode::Token(TT::Identifier)
    }
}

// TODO variant that always wraps, for eg. module ports? or not, maybe it's more elegant if we don't
fn fmt_extra_list<T>(list: &ExtraList<T>, f: &impl Fn(&T) -> HNode) -> Vec<HNode> {
    // TODO if there are no-simple items, force the parent to wrap
    //   or will that happen automatically once a Vertical is a child?
    let ExtraList { span: _, items } = list;
    items
        .iter()
        .map(|item| match item {
            ExtraItem::Inner(inner) => f(inner),
            ExtraItem::Declaration(_) => todo!(),
            ExtraItem::If(_) => todo!(),
        })
        .collect()
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
