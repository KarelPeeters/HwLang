use crate::syntax::format::FormatSettings;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, TokenCategory, TokenType as TT};
use crate::util::iter::IterExt;
use hwl_util::{swrite, swriteln};
use itertools::Itertools;

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

pub enum PNode {
    Literal(&'static str),
    Token(SourceTokenIndex),

    // TODO make all of these just variants of literal
    NewLine,
    IfWrap(Box<PNode>),

    Indent(Box<PNode>),
    Sequence(Vec<PNode>),
    CommaGroup(PCommaGroup),
}

struct PCommaGroup {
    compact: bool,
    children: Vec<PNode>,
}

#[derive(Debug, Copy, Clone)]
pub struct SourceTokenIndex(pub usize);

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

impl PNode {
    pub fn debug_str(&self) -> String {
        let mut f = String::new();
        self.debug_str_impl(&mut f, 0);
        f
    }

    fn debug_str_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        match self {
            PNode::Literal(s) => swriteln!(f, "Literal({s:?})"),
            PNode::Token(index) => swriteln!(f, "Token({index:?})"),
            PNode::NewLine => swriteln!(f, "NewLine"),
            PNode::IfWrap(child) => {
                swriteln!(f, "IfWrap");
                child.debug_str_impl(f, indent + 1);
            }
            PNode::Indent(child) => {
                swriteln!(f, "Indent");
                child.debug_str_impl(f, indent + 1);
            }
            PNode::Sequence(children) => {
                swriteln!(f, "Sequence");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                }
            }
            PNode::CommaGroup(PCommaGroup { compact, children }) => {
                swriteln!(f, "CommaGroup(compact={compact})");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                    swrite_indent(f, indent + 1);
                }
            }
        }
    }

    // TODO calculate this in a single pass instead of repeatedly
    fn contains_forced_newline(&self) -> bool {
        match self {
            // TODO if str contains newline?
            PNode::Literal(_) => false,
            // TODO if str contains newline?
            PNode::Token(_) => false,
            PNode::NewLine => true,
            PNode::IfWrap(child) | PNode::Indent(child) => child.contains_forced_newline(),
            PNode::Sequence(children) | PNode::CommaGroup(PCommaGroup { compact: _, children }) => {
                children.iter().any(PNode::contains_forced_newline)
            }
        }
    }
}

#[derive(Debug)]
pub struct TokenMismatch {
    pub index: Option<SourceTokenIndex>,
    pub expected: TT,
}

// TODO find a better name (for this and for SNode)
pub fn map_nodes(offsets: &LineOffsets, source_tokens: &[Token], node: FNode) -> Result<PNode, TokenMismatch> {
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

// TODO can we express this in the type system?
fn push_to_sequence(seq: &mut Vec<PNode>, node: PNode) {
    match node {
        PNode::Sequence(children) => seq.extend(children),
        _ => seq.push(node),
    }
}

impl MapContext<'_> {
    // fn map_token(&mut self, ty: TT) -> Result<Option<SourceTokenIndex>, TokenMismatch> {
    //     let token_index = SourceTokenIndex(self.next_source_index);
    //     let token = self.source_tokens.get(token_index.0).ok_or(TokenMismatch)?;
    //     self.next_source_index += 1;
    //
    //     if token.ty == ty {
    //         Ok(Some(token_index))
    //     } else if ty == TT::Comma {
    //         Ok(None)
    //     } else {
    //         Err(TokenMismatch)
    //     }
    // }

    fn capture_comments(&mut self) -> Vec<SourceTokenIndex> {
        let mut result = vec![];
        while let Some(token) = self.source_tokens.get(self.next_source_index)
            && token.ty.category() == TokenCategory::Comment
        {
            result.push(SourceTokenIndex(self.next_source_index));
            self.next_source_index += 1;
        }
        result
    }

    fn peek_token(&self) -> Option<&Token> {
        self.source_tokens.get(self.next_source_index)
    }

    fn prev_token(&self) -> Option<&Token> {
        if self.next_source_index == 0 {
            None
        } else {
            self.source_tokens.get(self.next_source_index - 1)
        }
    }

    fn pop_token(&mut self, ty: TT) -> Result<SourceTokenIndex, TokenMismatch> {
        let token = self.source_tokens.get(self.next_source_index).ok_or({
            TokenMismatch {
                index: None,
                expected: ty,
            }
        })?;

        let token_index = SourceTokenIndex(self.next_source_index);
        self.next_source_index += 1;

        if token.ty == ty {
            Ok(token_index)
        } else {
            Err(TokenMismatch {
                index: Some(token_index),
                expected: ty,
            })
        }
    }

    fn find_next_non_comment_token(&self) -> Option<SourceTokenIndex> {
        for i in self.next_source_index..self.source_tokens.len() {
            if self.source_tokens[i].ty.category() != TokenCategory::Comment {
                return Some(SourceTokenIndex(i));
            }
        }
        None
    }

    fn collect_vertical_comments_and_blank_lines(&mut self, result: &mut Vec<PNode>) -> Result<(), TokenMismatch> {
        // TODO preserve blank lines
        let non_comment_start = self
            .find_next_non_comment_token()
            .map(|token| self.offsets.expand_pos(self.source_tokens[token.0].span.start()));

        while let Some(token) = self.peek_token() {
            if let Some(real_token_start) = non_comment_start
                && self.offsets.expand_pos(token.span.end()).line_0 == real_token_start.line_0
            {
                break;
            }

            if let Some(prev_token) = self.prev_token() {
                let prev_end = self.offsets.expand_pos(prev_token.span.end());
                let curr_start = self.offsets.expand_pos(token.span.start());
                if prev_end.line_0 < curr_start.line_0 {
                    result.push(PNode::NewLine);
                }
            }

            match token.ty {
                TT::LineComment => {
                    // TODO space before?
                    result.push(PNode::Token(self.pop_token(token.ty)?));
                    result.push(PNode::NewLine);
                }
                TT::BlockComment => {
                    // TODO spaces before/after?
                    result.push(PNode::Token(self.pop_token(token.ty)?));
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn collect_horizontal_comments(&mut self, result: &mut Vec<PNode>) -> Result<(), TokenMismatch> {
        while let Some(token) = self.peek_token() {
            match token.ty {
                TT::LineComment => {
                    // TODO space before?
                    result.push(PNode::Token(self.pop_token(token.ty)?));
                    result.push(PNode::NewLine);
                }
                TT::BlockComment => {
                    // TODO space before/after?
                    result.push(PNode::Token(self.pop_token(token.ty)?));
                }
                _ => break,
            }
        }
        Ok(())
    }

    // TODO doc: comments inside this node are captures, before/after are the responsibility of the caller
    fn map(&mut self, node: FNode) -> Result<PNode, TokenMismatch> {
        let result = match node {
            FNode::Space => PNode::Literal(" "),
            FNode::Token(ty) => {
                let mut comments = vec![];
                self.collect_horizontal_comments(&mut comments)?;

                let node = PNode::Token(self.pop_token(ty)?);

                if comments.is_empty() {
                    node
                } else {
                    comments.push(node);
                    PNode::Sequence(comments)
                }
            }
            FNode::Horizontal(children) => {
                // TODO capture comments between nodes
                // TODO remember blank lines

                let mut children_mapped = vec![];
                for c in children {
                    let c_mapped = self.map(c)?;
                    push_to_sequence(&mut children_mapped, c_mapped);
                }
                PNode::Sequence(children_mapped)
            }
            FNode::Vertical(children) => {
                // TODO capture comments between nodes
                // TODO remember blank lines between both items and comments
                let mut children_mapped = vec![];
                for c in children {
                    self.collect_vertical_comments_and_blank_lines(&mut children_mapped)?;
                    let c_mapped = self.map(c)?;
                    push_to_sequence(&mut children_mapped, c_mapped);
                    children_mapped.push(PNode::NewLine);
                }
                self.collect_vertical_comments_and_blank_lines(&mut children_mapped)?;
                PNode::Sequence(children_mapped)
            }
            FNode::CommaList(list) => {
                // TODO capture comments between nodes
                // TODO here we also want to capture before/after to match indent?
                // TODO remember blank lines
                let FCommaList { compact, children } = list;
                let mut children_mapped = vec![];

                for child in children {
                    let child_mapped = self.map(child)?;
                    // TODO fix formatting for this
                    if let Some(t) = self.peek_token()
                        && t.ty == TT::Comma
                    {
                        let _ = self.pop_token(TT::Comma)?;
                    }

                    children_mapped.push(child_mapped);
                }

                PNode::CommaGroup(PCommaGroup {
                    compact,
                    children: children_mapped,
                })
            }
        };
        Ok(result)
    }
}

// TODO move to separate file?
pub fn node_to_string(
    settings: &FormatSettings,
    source_str: &str,
    source_offsets: &LineOffsets,
    source_tokens: &[Token],
    root: &PNode,
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

type WrapResult = Result<(), NeedsWrap>;

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

    fn ensure_indent(&mut self) {
        // indent if first token on current line
        if self.state.curr_line_start == self.result.len() {
            for _ in 0..self.state.indent {
                self.result.push_str(&self.settings.indent_str);
            }
        }
    }

    // fn write_comment(&mut self, index: SourceTokenIndex) {
    //     let token = &self.source_tokens[index.0];
    //     let token_str = &self.source_str[token.span.range_bytes()];
    //
    //     let line = match token.ty {
    //         TT::LineComment => true,
    //         TT::BlockComment => false,
    //         _ => todo!("err"),
    //     };
    //
    //     self.ensure_indent();
    //     self.result.push_str(token_str);
    //     if line {
    //         self.write_newline();
    //     }
    // }

    // fn write_token(&mut self, ty: TT, index: Option<SourceTokenIndex>) {
    //     self.ensure_indent();
    //
    //     // TODO write any leftover comments here, to make sure they're emitted
    //     // TODO how do we determine what the comments are if there is no index corresponding to this token?
    //     //   is it fine if we handle the comments afterwards?
    //     // TODO should comments after lines be allowed?
    //
    //     // figure out the token string
    //     let token_str = if let Some(index) = index {
    //         let token = &self.source_tokens[index.0];
    //         assert_eq!(ty, token.ty);
    //         &self.source_str[token.span.range_bytes()]
    //     } else {
    //         assert_eq!(ty, TT::Comma);
    //         ","
    //     };
    //
    //     // write to the result
    //     // TODO update line and overflow counters, especially if this contains newlines
    //     self.result.push_str(token_str);
    // }

    // fn write_comma_list(&mut self, list: &SCommaList, wrap: bool) -> WrapResult {
    //     let &SCommaList {
    //         compact,
    //         force_wrap,
    //         ref children,
    //     } = list;
    //     if compact {
    //         todo!()
    //     }
    //
    //     if force_wrap && !wrap {
    //         // TODO this is really an error condition, the parent should already know that they need to wrap
    //         return Err(NeedsWrap);
    //     }
    //
    //     if wrap {
    //         self.indent(|slf| {
    //             slf.write_newline();
    //             for (child_index, child) in enumerate(children) {
    //                 match child {
    //                     &SCommaListChild::Comment(child) => {
    //                         slf.write_comment(child);
    //                     }
    //                     SCommaListChild::Child(child, comma_index) => {
    //                         let comma_node =
    //                             SNodeKindNonHorizontal::NonWrap(SNodeKindNonWrap::Token(TT::Comma, *comma_index));
    //                         slf.write_node_with_extra_horizontal(child, Some(&comma_node), true)
    //                             .expect(MSG_WRAP);
    //
    //                         slf.write_newline();
    //                         if let Some(next_child) = children.get(child_index + 1) {
    //                             let next_span = match next_child {
    //                                 SCommaListChild::Comment(index) => Some(self.source_tokens[index.0].span),
    //                                 SCommaListChild::Child(next_child, _) => next_child.span,
    //                             };
    //                             slf.write_matching_empty_line(child.span, next_span);
    //                         }
    //                     }
    //                 }
    //             }
    //         })
    //     } else {
    //         for (child, last) in children.iter().with_last() {
    //             match child {
    //                 SCommaListChild::Comment(child) => {
    //                     // TODO error if line comment
    //                     self.write_comment(*child);
    //                 }
    //                 SCommaListChild::Child(child, comma_index) => {
    //                     self.write_node(child, false)?;
    //                     if !last {
    //                         self.write_token(TT::Comma, *comma_index);
    //                         self.write_space();
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //
    //     Ok(())
    // }

    fn write_str(&mut self, s: &str) -> WrapResult {
        // TODO if multiline, require wrap?
        // TODO if multiline, increment line state
        self.ensure_indent();
        self.result.push_str(s);
        Ok(())
    }

    fn write_node(&mut self, node: &PNode, allow_wrap: bool) -> WrapResult {
        match node {
            PNode::Literal(literal_str) => {
                self.write_str(literal_str)?;
            }
            PNode::Token(index) => {
                let token_span = self.source_tokens[index.0].span;
                let token_str = &self.source_str[token_span.range_bytes()];
                self.write_str(token_str)?;
            }
            PNode::NewLine => {
                self.write_newline();
            }
            PNode::IfWrap(_) => todo!(),
            PNode::Indent(_) => todo!(),
            PNode::Sequence(children) => {
                // TODO rollback if overflow etc
                for c in children {
                    self.write_node(c, false)?;
                }
            }
            PNode::CommaGroup(PCommaGroup { compact, children }) => {
                if *compact {
                    todo!()
                }

                // try without wrapping
                let check = self.checkpoint();
                let mut needs_wrap = false;

                for (child, last) in children.iter().with_last() {
                    match self.write_node(child, false) {
                        Ok(_) => {}
                        Err(_) => {}
                    }
                    // TODO if the child is a sequence, merge the comma into that sequence

                    if !last {
                        self.write_str(", ")?;
                    }
                }
            }
        }
        Ok(())
    }

    fn write_comma_group(
        &mut self,
        compact: bool,
        children: &[(PNode, Option<SourceTokenIndex>)],
        wrap: bool,
    ) -> WrapResult {
        todo!()
    }

    // fn write_node_with_extra_horizontal(
    //     &mut self,
    //     node: &SNode,
    //     extra_horizontal: Option<&SNodeKindNonHorizontal>,
    //     allow_wrap: bool,
    // ) -> WrapResult {
    //     match &node.kind {
    //         SNodeKind::NonHorizontal(node) => {
    //             if let Some(extra_horizontal) = extra_horizontal {
    //                 let slice_extra = SliceExtra {
    //                     slice: std::slice::from_ref(node),
    //                     extra: Some(extra_horizontal),
    //                 };
    //                 self.write_horizontal(slice_extra, allow_wrap)?;
    //             } else {
    //                 match node {
    //                     SNodeKindNonHorizontal::NonWrap(node) => self.write_node_non_wrap(node),
    //                     SNodeKindNonHorizontal::CommaList(_) => todo!("is this even possible?"),
    //                 }
    //             }
    //         }
    //         SNodeKind::Horizontal(children) => {
    //             let slice_extra = SliceExtra {
    //                 slice: children,
    //                 extra: extra_horizontal,
    //             };
    //             self.write_horizontal(slice_extra, allow_wrap)?
    //         }
    //     }
    //     Ok(())
    // }
}
