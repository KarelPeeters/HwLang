use crate::syntax::format::FormatSettings;
use crate::syntax::pos::{LineOffsets, Span};
use crate::syntax::token::{Token, TokenType as TT, is_whitespace_or_empty};
use crate::util::iter::IterExt;
use hwl_util::{swrite, swriteln};
use itertools::{Itertools, enumerate};

#[derive(Debug)]
pub enum FNode {
    Space,
    Token(TT),
    Horizontal(Vec<FNode>),
    Vertical(Vec<FNode>),
    CommaList(FCommaList),
}

#[derive(Debug)]
pub struct FCommaList {
    pub compact: bool,
    pub children: Vec<FNode>,
}

// TODO this could be turned into a single-pass algorithm,
//   by building the tree and computing the mapping at the same time
#[derive(Debug)]
pub struct SNode<K = SNodeKind> {
    span: Option<Span>,
    force_wrap: bool,
    kind: K,
}

#[derive(Debug)]
pub enum SNodeKind {
    NonHorizontal(SNodeKindNonHorizontal),
    Horizontal(Vec<SNodeKindNonHorizontal>),
}
#[derive(Debug)]
pub enum SNodeKindNonHorizontal {
    NonWrap(SNodeKindNonWrap),
    CommaList(SCommaList),
}

#[derive(Debug)]
pub enum SNodeKindNonWrap {
    Space,
    Token(TT, Option<SourceTokenIndex>),
    Vertical(Vec<SNode>),
}

#[derive(Debug, Copy, Clone)]
pub struct SourceTokenIndex(usize);

#[derive(Debug)]
pub struct SCommaList {
    compact: bool,
    force_wrap: bool,
    children: Vec<(SNode, Option<SourceTokenIndex>)>,
}

// TODO move down
fn swrite_indent(f: &mut String, indent: usize) {
    swrite!(f, "    ");
    for _ in 0..indent {
        swrite!(f, " |  ")
    }
}

impl FNode {
    pub fn tree_string(&self) -> String {
        let mut f = String::new();
        self.tree_string_impl(&mut f, 0);
        f
    }

    fn tree_string_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        let swrite_children = |f: &mut String, cs: &[FNode]| {
            for c in cs {
                c.tree_string_impl(f, indent + 1);
            }
        };
        match self {
            FNode::Space => swriteln!(f, "Space"),
            FNode::Token(ty) => swriteln!(f, "Token({ty:?})"),
            FNode::Horizontal(children) => {
                swriteln!(f, "Horizontal");
                swrite_children(f, children);
            }
            FNode::Vertical(children) => {
                swriteln!(f, "Vertical");
                swrite_children(f, children);
            }
            FNode::CommaList(FCommaList { compact, children }) => {
                swriteln!(f, "CommaList(compact={compact})");
                swrite_children(f, children);
            }
        }
    }
}

impl SNode {
    pub fn tree_string(&self) -> String {
        let mut f = String::new();
        self.tree_string_impl(&mut f, 0);
        f
    }

    fn tree_string_impl(&self, f: &mut String, indent: usize) {
        match &self.kind {
            SNodeKind::NonHorizontal(kind) => kind.tree_string_impl(f, indent),
            SNodeKind::Horizontal(children) => {
                swrite_indent(f, indent);
                swriteln!(f, "Horizontal");
                for c in children {
                    c.tree_string_impl(f, indent + 1);
                }
            }
        }
    }
}

impl SNodeKindNonHorizontal {
    fn tree_string_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        match self {
            SNodeKindNonHorizontal::NonWrap(node) => match node {
                SNodeKindNonWrap::Space => swriteln!(f, "Space"),
                SNodeKindNonWrap::Token(ty, index) => swriteln!(f, "Token({ty:?}, {index:?})"),
                SNodeKindNonWrap::Vertical(children) => {
                    swriteln!(f, "Vertical");
                    for c in children {
                        c.tree_string_impl(f, indent + 1);
                    }
                }
            },
            SNodeKindNonHorizontal::CommaList(list) => {
                swriteln!(f, "CommaList(compact={}, force_wrap={})", list.compact, list.force_wrap);
                for (c, comma_index) in &list.children {
                    c.tree_string_impl(f, indent + 1);
                    swrite_indent(f, indent + 1);
                    swriteln!(f, "{comma_index:?}");
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct TokenMismatch;

// TODO find a better name (for this and for SNode)
pub fn map_nodes(offsets: &LineOffsets, source_tokens: &[Token], node: FNode) -> Result<SNode, TokenMismatch> {
    let mut ctx = MapContext {
        offsets,
        source_tokens,
        next_source_index: 0,
    };
    ctx.map(node)
}

struct MapContext<'a> {
    offsets: &'a LineOffsets,
    source_tokens: &'a [Token],
    next_source_index: usize,
}

impl MapContext<'_> {
    fn map_token(&mut self, ty: TT) -> Result<(Option<SourceTokenIndex>, Option<Span>, bool), TokenMismatch> {
        let mut wrap = false;

        let token_index = loop {
            let curr_index = self.next_source_index;
            break match self.source_tokens.get(curr_index) {
                Some(source_token) => {
                    // skip whitespace and comments, but take them into account for wrapping
                    let skip = match source_token.ty {
                        TT::WhiteSpace => true,
                        TT::LineComment => {
                            wrap = true;
                            true
                        }
                        TT::BlockComment => {
                            let span = self.offsets.expand_span(source_token.span);
                            wrap |= span.end.line_0 > span.start.line_0;
                            true
                        }
                        _ => false,
                    };
                    if skip {
                        self.next_source_index += 1;
                        continue;
                    }

                    if ty == source_token.ty {
                        self.next_source_index += 1;
                        Some(SourceTokenIndex(curr_index))
                    } else if ty == TT::Comma {
                        None
                    } else {
                        return Err(TokenMismatch);
                    }
                }
                None => {
                    if ty == TT::Comma {
                        None
                    } else {
                        return Err(TokenMismatch);
                    }
                }
            };
        };

        // wrap if token spans multiple lines (eg. string literals or multiline comments)
        if let Some(token_index) = token_index {
            let token = &self.source_tokens[token_index.0];
            let span = self.offsets.expand_span(token.span);
            wrap |= span.end.line_0 > span.start.line_0;
        };

        let token_span = token_index.map(|i| self.source_tokens[i.0].span);
        Ok((token_index, token_span, wrap))
    }

    fn map(&mut self, node: FNode) -> Result<SNode, TokenMismatch> {
        let result = match node {
            FNode::Space => {
                let kind = SNodeKind::NonHorizontal(SNodeKindNonHorizontal::NonWrap(SNodeKindNonWrap::Space));
                SNode {
                    span: None,
                    force_wrap: false,
                    kind,
                }
            }
            FNode::Token(ty) => {
                let (index, span, force_wrap) = self.map_token(ty)?;
                let kind =
                    SNodeKind::NonHorizontal(SNodeKindNonHorizontal::NonWrap(SNodeKindNonWrap::Token(ty, index)));
                SNode { span, force_wrap, kind }
            }
            FNode::Horizontal(children) => {
                let mut span = None;
                let mut force_wrap = false;
                let mut children_mapped = vec![];
                for child in children {
                    let child_mapped = self.map(child)?;
                    span = join_maybe_span(span, child_mapped.span);
                    force_wrap |= child_mapped.force_wrap;
                    match child_mapped.kind {
                        SNodeKind::NonHorizontal(non_hor) => children_mapped.push(non_hor),
                        SNodeKind::Horizontal(hor) => children_mapped.extend(hor),
                    }
                }
                let kind_mapped = SNodeKind::Horizontal(children_mapped);
                SNode {
                    span,
                    force_wrap,
                    kind: kind_mapped,
                }
            }
            FNode::Vertical(children) => {
                let mut span = None;
                let children_mapped = children
                    .into_iter()
                    .map(|child| {
                        let child_mapped = self.map(child)?;
                        span = join_maybe_span(span, child_mapped.span);
                        Ok(child_mapped)
                    })
                    .try_collect_vec()?;
                let mapped = SNodeKind::NonHorizontal(SNodeKindNonHorizontal::NonWrap(SNodeKindNonWrap::Vertical(
                    children_mapped,
                )));
                SNode {
                    span,
                    force_wrap: true,
                    kind: mapped,
                }
            }
            FNode::CommaList(list) => {
                let FCommaList { compact, children } = list;

                let mut span = None;
                let mut force_wrap = false;
                let mut children_mapped = vec![];
                for child in children {
                    let child_mapped = self.map(child)?;
                    let (comma_index, comma_span, comma_force_wrap) = self.map_token(TT::Comma)?;

                    span = join_maybe_span(span, child_mapped.span);
                    span = join_maybe_span(span, comma_span);

                    force_wrap |= child_mapped.force_wrap | comma_force_wrap;
                    children_mapped.push((child_mapped, comma_index));
                }

                let list_mapped = SCommaList {
                    compact,
                    force_wrap,
                    children: children_mapped,
                };
                let kind_mapped = SNodeKind::NonHorizontal(SNodeKindNonHorizontal::CommaList(list_mapped));
                SNode {
                    span,
                    force_wrap,
                    kind: kind_mapped,
                }
            }
        };
        Ok(result)
    }
}

fn join_maybe_span(a: Option<Span>, b: Option<Span>) -> Option<Span> {
    match (a, b) {
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (Some(a), Some(b)) => Some(a.join(b)),
    }
}

// TODO move to separate file?
pub fn node_to_string(
    settings: &FormatSettings,
    source_str: &str,
    source_offsets: &LineOffsets,
    source_tokens: &[Token],
    root: &SNode,
) -> String {
    let mut ctx = StringBuilderContext {
        settings,
        source_str,
        source_offsets,
        source_tokens,
        result: String::with_capacity(source_str.len() * 2),
        state: StringState {
            next_node_token_index: 0,
            curr_line_index: 0,
            curr_line_start: 0,
            indent: 0,
        },
    };
    ctx.write_node(root, true).expect(MSG_WRAP);
    ctx.result
}

struct StringBuilderContext<'a> {
    settings: &'a FormatSettings,

    source_str: &'a str,
    source_offsets: &'a LineOffsets,
    source_tokens: &'a [Token],

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

#[derive(Debug)]
struct NeedsWrap;

// TODO convert this into a reportable internal compiler error?
const MSG_WRAP: &str = "should succeed, wrapping is allowed";

#[derive(Debug)]
struct SliceExtra<'a, T> {
    slice: &'a [T],
    extra: Option<&'a T>,
}

impl<T> Copy for SliceExtra<'_, T> {}

impl<T> Clone for SliceExtra<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> SliceExtra<'a, T> {
    fn split_first(&self) -> Option<(&'a T, SliceExtra<'a, T>)> {
        if let Some((first, rest)) = self.slice.split_first() {
            let extra = SliceExtra {
                slice: rest,
                extra: self.extra,
            };
            Some((first, extra))
        } else if let Some(first) = self.extra {
            let extra = SliceExtra {
                slice: &[],
                extra: None,
            };
            Some((first, extra))
        } else {
            None
        }
    }
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

    fn write_space(&mut self) {
        // TODO err if first thing on line?
        // TODO should we write comments here?
        self.result.push(' ');
    }

    fn write_newline(&mut self) {
        // TODO write comments that are on the same line as the previous token?
        self.result.push('\n');
        self.state.curr_line_index += 1;
        self.state.curr_line_start = self.result.len();
    }

    fn line_overflows(&self, check: CheckPoint) -> bool {
        // TODO optimize this?
        let line_start = check.state.curr_line_start;
        let rest = &self.result[line_start..];
        let line_len = rest.bytes().position(|c| c == b'\n').unwrap_or(rest.len());
        line_len > self.settings.max_line_length
    }

    fn indent<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.state.indent += 1;
        let r = f(self);
        self.state.indent -= 1;
        r
    }

    fn write_token(&mut self, ty: TT, index: Option<SourceTokenIndex>) {
        // indent if first token on current line
        if self.state.curr_line_start == self.result.len() {
            for _ in 0..self.state.indent {
                self.result.push_str(&self.settings.indent_str);
            }
        }

        // TODO write any leftover comments here, to make sure they're emitted
        // TODO how do we determine what the comments are if there is no index corresponding to this token?
        //   is it fine if we handle the comments afterwards?
        // TODO should comments after lines be allowed?

        // figure out the token string
        let token_str = if let Some(index) = index {
            let token = &self.source_tokens[index.0];
            assert_eq!(ty, token.ty);
            &self.source_str[token.span.range_bytes()]
        } else {
            assert_eq!(ty, TT::Comma);
            ","
        };

        // write to the result
        // TODO update line and overflow counters, especially if this contains newlines
        self.result.push_str(token_str);
    }

    fn write_comma_list(&mut self, list: &SCommaList, wrap: bool) -> Result<(), NeedsWrap> {
        let &SCommaList {
            compact,
            force_wrap,
            ref children,
        } = list;
        if compact {
            todo!()
        }

        if force_wrap && !wrap {
            // TODO this is really an error condition, the parent should already know that they need to wrap
            return Err(NeedsWrap);
        }

        if wrap {
            self.indent(|slf| {
                slf.write_newline();
                for (child_index, (child, comma_index)) in enumerate(children) {
                    let comma_node = SNodeKindNonHorizontal::NonWrap(SNodeKindNonWrap::Token(TT::Comma, *comma_index));
                    slf.write_node_with_extra_horizontal(child, Some(&comma_node), true)
                        .expect(MSG_WRAP);

                    slf.write_newline();
                    if let Some((next_child, _)) = children.get(child_index + 1) {
                        slf.write_matching_empty_line(child.span, next_child.span);
                    }
                }
            })
        } else {
            for (&(ref child, comma_index), last) in children.iter().with_last() {
                self.write_node(child, false)?;
                if !last {
                    self.write_token(TT::Comma, comma_index);
                    self.write_space();
                }
            }
        }

        Ok(())
    }

    fn write_node(&mut self, node: &SNode, allow_wrap: bool) -> Result<(), NeedsWrap> {
        self.write_node_with_extra_horizontal(node, None, allow_wrap)
    }

    fn write_node_with_extra_horizontal(
        &mut self,
        node: &SNode,
        extra_horizontal: Option<&SNodeKindNonHorizontal>,
        allow_wrap: bool,
    ) -> Result<(), NeedsWrap> {
        match &node.kind {
            SNodeKind::NonHorizontal(node) => {
                if let Some(extra_horizontal) = extra_horizontal {
                    let slice_extra = SliceExtra {
                        slice: std::slice::from_ref(node),
                        extra: Some(extra_horizontal),
                    };
                    self.write_horizontal(slice_extra, allow_wrap)?;
                } else {
                    match node {
                        SNodeKindNonHorizontal::NonWrap(node) => self.write_node_non_wrap(node),
                        SNodeKindNonHorizontal::CommaList(_) => todo!("is this even possible?"),
                    }
                }
            }
            SNodeKind::Horizontal(children) => {
                let slice_extra = SliceExtra {
                    slice: children,
                    extra: extra_horizontal,
                };
                self.write_horizontal(slice_extra, allow_wrap)?
            }
        }
        Ok(())
    }

    fn write_matching_empty_line(&mut self, span: Option<Span>, next_span: Option<Span>) {
        // TODO comments should stick to the correct item (eg. stick to previous should be possible)
        if let Some(child_span) = span
            && let Some(next_span) = next_span
        {
            let child_span = self.source_offsets.expand_span(child_span);
            let next_span = self.source_offsets.expand_span(next_span);

            let any_blank_line = ((child_span.end.line_0 + 1)..next_span.start.line_0).any(|line_0| {
                let line_range = self.source_offsets.line_range(line_0, false);
                let line_str = &self.source_str[line_range];
                is_whitespace_or_empty(line_str)
            });
            if any_blank_line {
                self.write_newline();
            }
        }
    }

    fn write_node_non_wrap(&mut self, node: &SNodeKindNonWrap) {
        match node {
            SNodeKindNonWrap::Space => self.write_space(),
            &SNodeKindNonWrap::Token(ty, index) => self.write_token(ty, index),
            SNodeKindNonWrap::Vertical(children) => {
                for (child_index, child) in enumerate(children) {
                    // within a vertical, nodes are always allowed to wrap
                    let _ = self.write_node(child, true);
                    self.write_newline();

                    if let Some(next_child) = children.get(child_index + 1) {
                        self.write_matching_empty_line(child.span, next_child.span);
                    }
                }
            }
        }
    }

    fn write_horizontal(
        &mut self,
        nodes: SliceExtra<SNodeKindNonHorizontal>,
        allow_wrap: bool,
    ) -> Result<(), NeedsWrap> {
        // pop the next node
        let (node, rest) = match nodes.split_first() {
            Some((node, rest)) => (node, rest),
            None => return Ok(()),
        };

        // write the node
        match node {
            SNodeKindNonHorizontal::NonWrap(node) => {
                // simple non-wrapping node, no decisions to take here
                self.write_node_non_wrap(node);
                self.write_horizontal(rest, allow_wrap)?;
                Ok(())
            }
            SNodeKindNonHorizontal::CommaList(list) => {
                // comma list, we need to decide whether to wrap or not
                // try without wrapping first
                let check = self.checkpoint();
                let result_unwrapped = self.write_comma_list(list, false);

                // check if the elements needs wrapping
                let mut should_wrap = match result_unwrapped {
                    Ok(()) => false,
                    Err(NeedsWrap) => true,
                };
                // check if the line already overflows
                if !should_wrap {
                    should_wrap = self.line_overflows(check);
                }

                // if no wrapping is needed yet, try writing the rest of the list
                // TODO this is just a perf optimization, we could also always do this
                //    and this can also be optimized more, as soon as we overflow deeper we know that we should bail
                if !should_wrap {
                    self.write_horizontal(rest, allow_wrap)?;
                    should_wrap = self.line_overflows(check)
                }

                // if we need to wrap, roll back and re-writing everything with wrapping
                if should_wrap {
                    if !allow_wrap {
                        return Err(NeedsWrap);
                    }
                    self.restore(check);
                    self.write_comma_list(list, true).expect(MSG_WRAP);
                    self.write_horizontal(rest, true).expect(MSG_WRAP);
                }

                Ok(())
            }
        }
    }
}
