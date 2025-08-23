use crate::syntax::format::FormatSettings;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, TokenCategory, TokenType as TT};
use crate::util::iter::IterExt;
use crate::util::{Never, ResultNeverExt};
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

pub struct PCommaGroup {
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
    ctx.write_node::<AllowWrap>(root).remove_never();
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

trait MaybeWrap {
    type E;
    fn allow_wrap() -> bool;
    fn require_wrap() -> Result<(), Self::E>;
}

struct NoWrap {}
impl MaybeWrap for NoWrap {
    type E = NeedsWrap;
    fn allow_wrap() -> bool {
        false
    }
    fn require_wrap() -> Result<(), NeedsWrap> {
        Err(NeedsWrap)
    }
}

struct AllowWrap {}
impl MaybeWrap for AllowWrap {
    type E = Never;
    fn allow_wrap() -> bool {
        true
    }
    fn require_wrap() -> Result<(), Never> {
        Ok(())
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
        // TODO rename to clarify whether this checks the first or last line, or maybe this should check all lines?
        //   but not comment lines >:(
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

    fn write_str<W: MaybeWrap>(&mut self, s: &str) -> Result<(), W::E> {
        // TODO if multiline, require wrap?
        // TODO if multiline, increment line state
        self.ensure_indent();
        self.result.push_str(s);
        Ok(())
    }

    fn write_node_extra<W: MaybeWrap>(&mut self, node: &PNode, extra: Option<&PNode>) -> Result<(), W::E> {
        if let PNode::Sequence(children) = node {
            self.write_sequence::<W>(children.iter().chain(extra.into_iter()))
        } else if let Some(extra) = extra {
            self.write_sequence::<W>([node, extra].into_iter())
        } else {
            self.write_node::<W>(node)
        }
    }

    fn write_node<W: MaybeWrap>(&mut self, node: &PNode) -> Result<(), W::E> {
        match node {
            PNode::Literal(literal_str) => {
                self.write_str::<W>(literal_str)?;
            }
            PNode::Token(index) => {
                let token_span = self.source_tokens[index.0].span;
                let token_str = &self.source_str[token_span.range_bytes()];
                self.write_str::<W>(token_str)?;
            }
            PNode::NewLine => {
                self.write_newline();
            }
            PNode::IfWrap(_) => todo!(),
            PNode::Indent(_) => todo!(),
            PNode::Sequence(children) => {
                self.write_sequence::<W>(children.iter())?;
            }
            PNode::CommaGroup(PCommaGroup { compact, children }) => {
                if *compact {
                    todo!()
                }

                // try without wrapping
                let check = self.checkpoint();
                let mut needs_wrap = false;
                for (child, last) in children.iter().with_last() {
                    let extra = if last { None } else { Some(&PNode::Literal(", ")) };
                    match self.write_node_extra::<NoWrap>(child, extra) {
                        Ok(_) => {}
                        Err(NeedsWrap) => {
                            needs_wrap = true;
                            break;
                        }
                    }
                    if self.line_overflows(check) {
                        needs_wrap = true;
                        break;
                    }
                }

                // maybe fallback to wrapping
                if needs_wrap {
                    self.restore(check);
                    W::require_wrap()?;
                    self.indent(|slf| {
                        slf.write_newline();
                        for child in children {
                            let extra = Some(&PNode::Literal(","));
                            slf.write_node_extra::<AllowWrap>(child, extra).remove_never();
                            slf.write_newline();
                        }
                    })
                }
            }
        }
        Ok(())
    }

    fn write_sequence<'c, W: MaybeWrap>(
        &mut self,
        mut children: impl Iterator<Item = &'c PNode> + Clone,
    ) -> Result<(), W::E> {
        let Some(child) = children.next() else {
            return Ok(());
        };
        let rest = children;

        match child {
            PNode::Sequence(_) => todo!("err, should be flattened"),
            PNode::Literal(_) | PNode::Token(_) | PNode::NewLine | PNode::IfWrap(_) | PNode::Indent(_) => {
                // simple nodes without a wrapping decision, just write them
                let check = self.checkpoint();
                self.write_node::<W>(child)?;
                self.write_sequence::<W>(rest).inspect_err(|_| self.restore(check))?;
                Ok(())
            }
            PNode::CommaGroup(group) => {
                // for groups we have to make a choice

                // try without wrapping
                let check = self.checkpoint();
                let result_no_wrap = self.write_comma_group::<NoWrap>(group);

                // check check if children need wrapping
                let mut should_wrap = match result_no_wrap {
                    Ok(()) => false,
                    Err(NeedsWrap) => true,
                };

                // check if line overflows
                if !should_wrap {
                    should_wrap |= self.line_overflows(check);
                }

                // if no wrapping is needed yet, try writing the rest of the list
                // TODO this is just a perf optimization, we could also always do this
                //    and this can also be optimized more, as soon as we overflow deeper we know that we should bail
                if !should_wrap {
                    self.write_sequence::<W>(rest.clone())
                        .inspect_err(|_| self.restore(check))?;
                    should_wrap |= self.line_overflows(check);
                }

                // if we need to wrap, roll back and re-writing everything with wrapping
                if should_wrap {
                    self.restore(check);
                    W::require_wrap()?;

                    self.write_comma_group::<AllowWrap>(group).remove_never();
                    self.write_sequence::<AllowWrap>(rest).remove_never();
                }

                Ok(())
            }
        }
    }

    fn write_comma_group<W: MaybeWrap>(&mut self, group: &PCommaGroup) -> Result<(), W::E> {
        let PCommaGroup { compact, children } = group;
        if *compact {
            todo!()
        }

        // TODO here we interpret W as "wrap", not "allow wrap"
        if W::allow_wrap() {
            self.indent(|slf| {
                slf.write_newline();
                for child in children {
                    slf.write_node_extra::<AllowWrap>(child, Some(&PNode::Literal(",")))
                        .remove_never();
                    slf.write_newline();
                }
            });
        } else {
            let check = self.checkpoint();
            for (child, last) in children.iter().with_last() {
                self.write_node::<W>(child).inspect_err(|_| self.restore(check))?;
                if !last {
                    self.write_str::<W>(",").inspect_err(|_| self.restore(check))?;
                    self.write_space();
                }
            }
        }

        Ok(())
    }
}
