use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::syntax::ast::{
    ArenaExpressions, Arg, Args, Block, BlockStatement, CommonDeclaration, CommonDeclarationNamed,
    CommonDeclarationNamedKind, Expression, ExpressionKind, ExtraItem, ExtraList, FileContent, FunctionDeclaration,
    GeneralIdentifier, Identifier, Item, MaybeIdentifier, Parameter, Parameters, Visibility,
};
use crate::syntax::format::FormatSettings;
use crate::syntax::pos::{Pos, Span};
use crate::syntax::source::{FileId, SourceDatabase};
use crate::syntax::token::{Token, TokenCategory, TokenType as TT, tokenize};
use crate::syntax::{parse_error_to_diagnostic, parse_file_content};
use crate::util::iter::IterExt;
use crate::util::{Never, ResultNeverExt};
use itertools::Itertools;

// Sketch of the formatter implementation:
// * convert the AST to a tree of FNodes, which are just tokens with some extra structure
// * cross-reference emitted formatting tokens with the original tokens to get span info
// * bottom-up traverse the format tree to figure out which branches need to wrap due to line comments
// * top-down traverse the format tree, printing the tokens to the final buffer
//    * if any line overflows and we can still make additional wrapping choices, roll back and try
//    * if blank lines between items: insert matching blank line
// TODO add debug mode that slowly decreases the line width and saves every time the output changes
pub fn format(
    diags: &Diagnostics,
    source: &SourceDatabase,
    file: FileId,
    settings: &FormatSettings,
) -> DiagResult<String> {
    // tokenize and parse
    let source_str = &source[file].content;
    let mut source_tokens = tokenize(file, source_str, false).map_err(|e| diags.report(e.to_diagnostic()))?;
    // TODO remove whitespace tokens, everyone just filters them out anyway, we can always recover whitespace as "stuff between other tokens" if we really want to
    source_tokens.retain(|t| t.ty != TT::WhiteSpace);
    let source_ast = parse_file_content(file, source_str).map_err(|e| diags.report(parse_error_to_diagnostic(e)))?;

    println!("Source tokens:");
    for token in &source_tokens {
        println!("    {token:?}");
    }

    // convert to formatting tree
    let ctx = NodeBuilderContext {
        arena_expressions: &source_ast.arena_expressions,
    };
    let node_root = ctx.fmt_file(&source_ast);

    println!("Tree:");
    println!("{:#?}", node_root);

    println!("Tree tokens:");
    node_root
        .try_for_each_token(&mut |node_token_ty, node_token_fixed| {
            println!("    {node_token_ty:?} (fixed: {node_token_fixed})");
            Ok::<(), Never>(())
        })
        .remove_never();

    // cross-reference tokens and figure out which nodes contain newlines
    let source_end_pos = source.full_span(file).end();
    let fixed_token_map = match_tokens(diags, source_end_pos, &source_tokens, &node_root)?;

    // convert to output string
    let mut result_ctx = StringBuilderContext {
        source_str,
        source_tokens: &source_tokens,
        node_token_to_source: &fixed_token_map,

        settings,

        result: String::with_capacity(source_str.len() * 2),
        state: StringState {
            next_node_token_index: 0,
            curr_line_index: 0,
            curr_line_start: 0,
            indent: 0,
        },
    };
    result_ctx.write_node(&node_root);

    if result_ctx.state.next_node_token_index != fixed_token_map.len() {
        return Err(todo!("err"));
    }

    Ok(result_ctx.result)
}

fn match_tokens(
    diags: &Diagnostics,
    source_end_pos: Pos,
    source_tokens: &[Token],
    node_root: &FNode,
) -> DiagResult<Vec<Option<usize>>> {
    let mut next_source_index = 0;
    let mut map = vec![];

    node_root.try_for_each_token(&mut |node_token_ty, node_token_fixed| {
        let source_index = loop {
            // find next real token
            let source_index = next_source_index;
            let Some(source_token) = source_tokens.get(next_source_index) else {
                let e = diags.report_internal_error(
                    Span::empty_at(source_end_pos),
                    format!("failed to match token: node expects {node_token_ty:?} but reached end of source tokens"),
                );
                return Err(e);
            };
            if matches!(
                source_token.ty.category(),
                TokenCategory::WhiteSpace | TokenCategory::Comment
            ) {
                println!("skipping whitespace/comment");
                next_source_index += 1;
                continue;
            }

            // try to match it
            break if source_token.ty == node_token_ty {
                println!("matched token {node_token_ty:?}");
                next_source_index += 1;
                Some(source_index)
            } else if !node_token_fixed {
                println!("skipping non-fixed token {node_token_ty:?}");
                None
            } else {
                // failed to find match
                let e = diags.report_internal_error(
                    source_token.span,
                    format!(
                        "failed to match token: node expects {:?} but source has {:?}",
                        node_token_ty, source_token.ty
                    ),
                );
                return Err(e);
            };
        };

        map.push(source_index);
        Ok(())
    })?;

    Ok(map)
}

#[derive(Debug)]
#[must_use]
enum FNode {
    NonH(FNodeNonHor),

    /// Sequence of tokens that tries hard to stay on a single line.
    /// It is only broken up if is is forced to by line comments between the tokens.
    /// TODO if broken, indent the next line
    /// TODO doc: child tokens are still allowed to individually wrap, it's just that they're joined in a single line
    Horizontal(Vec<FNodeNonHor>),
}

#[derive(Debug)]
#[must_use]
enum FNodeNonHor {
    NonWrap(FNodeNonWrap),
    // TODO variant of this that always wraps? or will wrapping be tracked outside completely,
    //   in which case we can use a meta-node?
    // TODO doc
    // TODO respect newlines
    // TODO rename to "maybewrappinglist"?
    CommaList(FCommaList),
}

#[derive(Debug)]
#[must_use]
enum FNodeNonWrap {
    Space,
    Token(TT),
    // TODO respect newlines, maybe configurable?
    Vertical(Vec<FNode>),
}

#[derive(Debug)]
#[must_use]
struct FCommaList {
    compact: bool,
    nodes: Vec<FNode>,
}

// TODO remove?
fn token(ty: TT) -> FNode {
    FNode::NonH(FNodeNonHor::NonWrap(FNodeNonWrap::Token(ty)))
}
fn space() -> FNode {
    FNode::NonH(FNodeNonHor::NonWrap(FNodeNonWrap::Space))
}
fn vertical(nodes: Vec<FNode>) -> FNode {
    FNode::NonH(FNodeNonHor::NonWrap(FNodeNonWrap::Vertical(nodes)))
}
// TODO many callers are unnecessarily using this
fn horizontal(nodes: Vec<FNode>) -> FNode {
    let mut result = vec![];
    for n in nodes {
        match n {
            FNode::NonH(non_h) => result.push(non_h),
            FNode::Horizontal(children) => result.extend(children.into_iter()),
        }
    }
    FNode::Horizontal(result)
}
fn comma_list(compact: bool, nodes: Vec<FNode>) -> FNode {
    let list = FCommaList { compact, nodes };
    FNode::NonH(FNodeNonHor::CommaList(list))
}

struct StringBuilderContext<'a> {
    source_str: &'a str,
    source_tokens: &'a [Token],
    node_token_to_source: &'a Vec<Option<usize>>,

    settings: &'a FormatSettings,

    result: String,
    state: StringState,
}

#[derive(Debug, Copy, Clone)]
struct CheckPoint {
    result_len: usize,
    state: StringState,
}

#[derive(Debug, Copy, Clone)]
struct StringState {
    next_node_token_index: usize,
    curr_line_index: usize,
    curr_line_start: usize,
    indent: usize,
}

#[must_use]
enum PreferWrap {
    No,
    Yes,
}

impl StringBuilderContext<'_> {
    fn checkpoint(&self) -> CheckPoint {
        CheckPoint {
            result_len: self.result.len(),
            state: self.state,
        }
    }

    fn restore(&mut self, check: CheckPoint) {
        assert!(self.result.len() >= check.result_len);
        self.result.truncate(check.result_len);
        self.state = check.state;
    }

    // TODO asser that this is called in exactly the same order as the first collection pass?
    fn write_token(&mut self, ty: TT) {
        let token_str = loop {
            // TODO replace asserts and unwraps with diag error?
            let node_index = self.state.next_node_token_index;
            self.state.next_node_token_index += 1;
            let source_index = self.node_token_to_source.get(node_index).unwrap();

            // TODO this is slightly duplicate logic
            break match source_index {
                None => {
                    if ty == TT::Comma {
                        ","
                    } else {
                        continue;
                    }
                }
                &Some(source_index) => {
                    let source_token = &self.source_tokens[source_index];
                    assert_eq!(source_token.ty, ty);
                    &self.source_str[source_token.span.range_bytes()]
                }
            };
        };

        println!("pushing token {ty:?}");

        // indent if first token on the line
        if self.state.curr_line_start == self.result.len() {
            for _ in 0..self.state.indent {
                self.result.push_str(&self.settings.indent_str);
            }
        }

        // TODO count newlines (important for comments and multi-line string literals)
        // TODO dedicated, non-ugly multi-line string literals?
        self.result.push_str(token_str);
    }

    fn write_space(&mut self) {
        // TODO err if first thing on line?
        self.result.push(' ');
    }

    fn write_newline(&mut self) {
        self.result.push('\n');
        self.state.curr_line_index += 1;
        self.state.curr_line_start = self.result.len();
    }

    fn line_overflows(&self, check: CheckPoint) -> bool {
        let line_start = check.state.curr_line_start;
        let rest = &self.result[line_start..];
        let line_len = rest.bytes().position(|c| c == b'\n').unwrap_or(rest.len());
        line_len > self.settings.max_line_length
    }

    fn indent(&mut self, f: impl FnOnce(&mut Self)) {
        self.state.indent += 1;
        f(self);
        self.state.indent -= 1;
    }

    fn write_comma_list(&mut self, list: &FCommaList, wrap: bool) {
        let &FCommaList { compact, ref nodes } = list;
        if compact {
            todo!()
        }

        // TODO indent if wrap
        if wrap {
            self.write_newline();
        }
        for (node, last) in nodes.iter().with_last() {
            self.write_node(node);
            if wrap {
                self.write_token(TT::Comma);
                self.write_newline();
            } else if !last {
                self.write_token(TT::Comma);
                self.write_space();
            }
        }
    }

    fn write_node(&mut self, node: &FNode) {
        // TODO indentation stuff?
        match node {
            FNode::NonH(node) => match node {
                FNodeNonHor::NonWrap(non_wrap) => {
                    self.write_non_wrap(non_wrap);
                }
                FNodeNonHor::CommaList(list) => {
                    // TODO can this even happen without being in a horizontal?
                    // self.write_comma_list(list, false);
                    todo!()
                }
            },
            FNode::Horizontal(nodes) => {
                // TODO try each non-wrapped first, including future stuff on the same line, then wrap left-to-right
                // TODO we need to check for overflow on specific lines, not just in general, since otherwise we might wrap too much
                // TODO maye we need to merge nested horizontals for this to fully work

                // let mut checks = vec![self.checkpoint()];
                // let mut i_uncommited = 0;
                // let mut i_next = 0;
                //
                // while i_uncommited < nodes.len() {
                //
                // }
                self.write_horizontal(nodes);
            }
        }
    }

    fn write_non_wrap(&mut self, node: &FNodeNonWrap) {
        match node {
            FNodeNonWrap::Space => {
                // TODO err if first thing on line?
                self.write_space();
            }
            &FNodeNonWrap::Token(ty) => {
                self.write_token(ty);
            }
            FNodeNonWrap::Vertical(nodes) => {
                for n in nodes {
                    // TODO try single first, then go back to multiple? for vertical this feels weird, we don't actually care ourselves
                    // TODO respect blank lines between items
                    // within a vertical, nodes are always allowed to wrap
                    let _ = self.write_node(n);
                    self.write_newline();
                }
            }
        }
    }

    fn write_horizontal(&mut self, nodes: &[FNodeNonHor]) {
        let (node, rest) = match nodes.split_first() {
            None => return,
            Some(p) => p,
        };

        match node {
            FNodeNonHor::NonWrap(node) => {
                // simple non-wrapping node, no decisions to take here
                self.write_non_wrap(node);
                self.write_horizontal(rest);
            }
            FNodeNonHor::CommaList(list) => {
                // comma list, we need to decide whether to wrap or not

                // try without wrapping first
                let check = self.checkpoint();
                self.write_comma_list(list, false);
                // TODO benchmark if this extra check helps a lot
                // if there is no overflow yet, try writing the rest of the nodes
                let mut overflow = self.line_overflows(check);
                if !overflow {
                    self.write_horizontal(rest);
                    overflow = self.line_overflows(check)
                }

                // there was overflow (which could not be fixed by wrapping future elements), try wrapping this one
                if overflow {
                    self.restore(check);
                    self.write_comma_list(list, true);
                    self.write_horizontal(rest);
                }
            }
        }
    }
}

impl FNode {
    // TODO doc bool arg: "fixed"
    fn try_for_each_token<E>(&self, f: &mut impl FnMut(TT, bool) -> Result<(), E>) -> Result<(), E> {
        match self {
            FNode::NonH(slf) => slf.try_for_each_token(f),
            FNode::Horizontal(children) => children.iter().try_for_each(|c| c.try_for_each_token(f)),
        }
    }
}

impl FNodeNonHor {
    // TODO doc bool arg: "fixed"
    fn try_for_each_token<E>(&self, f: &mut impl FnMut(TT, bool) -> Result<(), E>) -> Result<(), E> {
        match self {
            FNodeNonHor::NonWrap(slf) => match slf {
                FNodeNonWrap::Space => Ok(()),
                &FNodeNonWrap::Token(ty) => f(ty, true),
                FNodeNonWrap::Vertical(children) => children.iter().try_for_each(|c| c.try_for_each_token(f)),
            },
            FNodeNonHor::CommaList(FCommaList { compact: _, nodes }) => {
                nodes.iter().with_last().try_for_each(|(n, last)| {
                    n.try_for_each_token(f)?;
                    f(TT::Comma, !last)?;
                    Ok(())
                })
            }
        }
    }
}

// TODO if this stays this single field, maybe we don't need this struct at all?
struct NodeBuilderContext<'a> {
    arena_expressions: &'a ArenaExpressions,
}

impl NodeBuilderContext<'_> {
    fn fmt_file(&self, file: &FileContent) -> FNode {
        let FileContent {
            span,
            items,
            arena_expressions,
        } = file;
        let nodes = items
            .iter()
            .map(|item| match item {
                Item::Import(_) => todo!(),
                Item::CommonDeclaration(decl) => self.fmt_common_decl(&decl.inner),
                Item::ModuleInternal(_) => todo!(),
                Item::ModuleExternal(_) => todo!(),
                Item::Interface(_) => todo!(),
            })
            .collect();
        vertical(nodes)
    }

    fn fmt_common_decl<V: FormatVisibility>(&self, decl: &CommonDeclaration<V>) -> FNode {
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
                            span,
                            id,
                            ref params,
                            ret_ty,
                            ref body,
                        } = decl;
                        let mut nodes = vec![
                            token(TT::Function),
                            space(),
                            self.fmt_maybe_id(id),
                            self.fmt_parameters(params),
                        ];
                        if let Some(ret_ty) = ret_ty {
                            nodes.push(space());
                            nodes.push(token(TT::Arrow));
                            nodes.push(space());
                            nodes.push(self.fmt_expr(ret_ty));
                        }
                        nodes.push(space());
                        nodes.push(self.fmt_block(body));
                        horizontal(nodes)
                    }
                };

                match node_vis {
                    None => node_kind,
                    Some(token_vis) => horizontal(vec![token(token_vis), node_kind]),
                }
            }
            CommonDeclaration::ConstBlock(_) => todo!(),
        }
    }

    fn fmt_parameters(&self, params: &Parameters) -> FNode {
        let Parameters { span, items } = params;
        let items = fmt_extra_list(items, &|p| self.fmt_parameter(p));

        let node_list = comma_list(false, items);
        horizontal(vec![token(TT::OpenR), node_list, token(TT::CloseR)])
    }

    fn fmt_parameter(&self, param: &Parameter) -> FNode {
        let &Parameter {
            span: _,
            id,
            ty,
            default,
        } = param;
        let mut nodes = vec![self.fmt_id(id), token(TT::Colon), space(), self.fmt_expr(ty)];
        if let Some(default) = default {
            nodes.push(space());
            nodes.push(token(TT::Eq));
            nodes.push(space());
            nodes.push(self.fmt_expr(default));
        }
        horizontal(nodes)
    }

    fn fmt_block(&self, block: &Block<BlockStatement>) -> FNode {
        // TODO for else-if blocks, force a newline to avoid infinite clutter
        if !block.statements.is_empty() {
            todo!()
        }
        horizontal(vec![token(TT::OpenC), token(TT::CloseC)])
    }

    fn fmt_expr(&self, expr: Expression) -> FNode {
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
                            horizontal(vec![self.fmt_id(name), token(TT::Eq), node_value])
                        } else {
                            node_value
                        }
                    })
                    .collect_vec();

                let node_list = comma_list(false, nodes_arg);
                let node_args = horizontal(vec![token(TT::OpenR), node_list, token(TT::CloseR)]);

                horizontal(vec![node_target, node_args])
            }
            ExpressionKind::Builtin(_) => todo!(),
            ExpressionKind::UnsafeValueWithDomain(_, _) => todo!(),
            ExpressionKind::RegisterDelay(_) => todo!(),
        }
    }

    fn fmt_general_id(&self, id: GeneralIdentifier) -> FNode {
        match id {
            GeneralIdentifier::Simple(id) => self.fmt_id(id),
            GeneralIdentifier::FromString(_, _) => todo!(),
        }
    }

    fn fmt_maybe_id(&self, id: MaybeIdentifier) -> FNode {
        match id {
            MaybeIdentifier::Dummy(_span) => token(TT::Underscore),
            MaybeIdentifier::Identifier(id) => self.fmt_id(id),
        }
    }

    fn fmt_id(&self, id: Identifier) -> FNode {
        let _ = id;
        token(TT::Identifier)
    }
}

// TODO variant that always wraps, for eg. module ports? or not, maybe it's more elegant if we don't
fn fmt_extra_list<T>(list: &ExtraList<T>, f: &impl Fn(&T) -> FNode) -> Vec<FNode> {
    // TODO if there are no-simple items, force the parent to wrap
    //   or will that happen automatically once a Vertical is a child?
    let ExtraList { span: _, items } = list;
    items
        .iter()
        .map(|item| match item {
            ExtraItem::Inner(inner) => f(inner),
            ExtraItem::Declaration(decl) => todo!(),
            ExtraItem::If(if_item) => todo!(),
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
