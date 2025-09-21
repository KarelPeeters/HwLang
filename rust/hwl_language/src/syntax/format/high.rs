use crate::syntax::format::common::swrite_indent;
use crate::syntax::format::low::LNode;
use crate::syntax::pos::{LineOffsets, SpanFull};
use crate::syntax::token::{Token, TokenCategory as TC, TokenType as TT};
use hwl_util::swriteln;
use itertools::enumerate;

/// High-level formatting nodes.
///
/// These mostly correspond to [LNode]s, except that the leaf nodes are tokens instead of strings,
/// and that newlines and comments are not fully represented here yet, they will be inserted during lowering.
#[must_use]
#[derive(Debug)]
pub enum HNode {
    Space,
    AlwaysToken(TT),
    WrapComma,
    AlwaysNewline,
    WrapNewline,
    AlwaysBlankLine,
    ForceWrap,
    Indent(Box<HNode>),
    Dedent(Box<HNode>),
    Sequence(Vec<HNode>),
    Group(Box<HNode>),
    Fill(Vec<HNode>),
    PreserveBlankLines { last: bool },
}

#[derive(Debug)]
pub struct TokenMismatch {
    pub index: Option<SourceTokenIndex>,
    pub expected: TT,
}

#[derive(Debug, Copy, Clone)]
pub struct SourceTokenIndex(pub usize);

pub fn lower_nodes<'s>(
    source: &'s str,
    offsets: &LineOffsets,
    source_tokens: &[Token],
    node: &HNode,
) -> Result<LNode<'s>, TokenMismatch> {
    let mut ctx = LowerContext {
        source,
        offsets,
        source_tokens,
        next_source_index: 0,
    };
    ctx.map_root(node)
}

impl HNode {
    pub const EMPTY: HNode = HNode::Sequence(vec![]);

    pub fn debug_str(&self) -> String {
        let mut f = String::new();
        self.debug_str_impl(&mut f, 0);
        f
    }

    fn debug_str_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        let swrite_children = |f: &mut String, cs: &[HNode]| {
            for c in cs {
                c.debug_str_impl(f, indent + 1);
            }
        };
        match self {
            HNode::Space => swriteln!(f, "Space"),
            HNode::AlwaysToken(ty) => swriteln!(f, "AlwaysToken({ty:?})"),
            HNode::WrapComma => swriteln!(f, "WrapComma"),
            HNode::AlwaysNewline => swriteln!(f, "AlwaysNewline"),
            HNode::AlwaysBlankLine => swriteln!(f, "BlankLine"),
            HNode::WrapNewline => swriteln!(f, "WrapNewline"),
            HNode::ForceWrap => swriteln!(f, "ForceWrap"),
            HNode::Indent(child) => {
                swriteln!(f, "Indent");
                child.debug_str_impl(f, indent + 1);
            }
            HNode::Dedent(child) => {
                swriteln!(f, "Dedent");
                child.debug_str_impl(f, indent + 1);
            }
            HNode::Sequence(children) => {
                swriteln!(f, "Sequence");
                swrite_children(f, children);
            }
            HNode::Group(child) => {
                swriteln!(f, "Group");
                child.debug_str_impl(f, indent + 1);
            }
            HNode::Fill(children) => {
                swriteln!(f, "Fill");
                swrite_children(f, children);
            }
            HNode::PreserveBlankLines { last: end } => swriteln!(f, "PreserveBlankLines(end={end})"),
        }
    }
}

struct LowerContext<'s, 'r> {
    source: &'s str,
    offsets: &'r LineOffsets,
    source_tokens: &'r [Token],
    next_source_index: usize,
}

// TODO reverse order of functions
impl<'s, 'r> LowerContext<'s, 'r> {
    fn prev_token(&self) -> Option<&Token> {
        if self.next_source_index == 0 {
            None
        } else {
            self.source_tokens.get(self.next_source_index - 1)
        }
    }

    fn peek_token(&self) -> Option<&'r Token> {
        self.source_tokens.get(self.next_source_index)
    }

    fn pop_token(&mut self) -> Option<SourceTokenIndex> {
        if self.next_source_index < self.source_tokens.len() {
            let token_index = SourceTokenIndex(self.next_source_index);
            self.next_source_index += 1;
            Some(token_index)
        } else {
            None
        }
    }

    fn find_next_non_comment_token(&self) -> Option<SourceTokenIndex> {
        for i in self.next_source_index..self.source_tokens.len() {
            if self.source_tokens[i].ty.category() != TC::Comment {
                return Some(SourceTokenIndex(i));
            }
        }
        None
    }

    fn collect_comments(
        &mut self,
        prev_space: bool,
        seq: &mut Vec<LNode<'s>>,
        mut filter: impl FnMut(SpanFull) -> bool,
    ) {
        let prev_space = seq_ends_with_space(seq).unwrap_or(prev_space);

        let mut next_is_first_comment = true;
        let mut escape_index = None;

        let mut report_newline = |index: usize| {
            if escape_index.is_none() {
                escape_index = Some(index);
            }
        };

        while let Some(token) = self.peek_token() {
            // check whether we should include this token
            let is_line_comment = match token.ty {
                TT::LineComment => true,
                TT::BlockComment => false,
                _ => break,
            };
            let token_span = self.offsets.expand_span(token.span);
            if !filter(token_span) {
                break;
            }

            let is_first_comment = next_is_first_comment;
            next_is_first_comment = false;

            // preserve newlines
            let len_before = seq.len();
            if self.preserve_empty_lines(seq, !is_first_comment) {
                report_newline(len_before);
            }

            // pop the comment token itself
            let _ = self.pop_token().unwrap();

            // add the comment nodes, de-dented if necessary
            // TODO start escaping if there is a newline in the comment
            let comment_str_all = &self.source[token.span.range_bytes()];
            let (comment_str_first, comment_str_rest) = match comment_str_all.find(LineOffsets::LINE_ENDING_CHARS) {
                None => (comment_str_all, None),
                Some(pos) => (&comment_str_all[..pos], Some(&comment_str_all[pos..])),
            };

            let node_comment = LNode::Sequence(vec![LNode::Space, LNode::AlwaysStr(comment_str_first)]);
            let node_comment = if token_span.start.col_0 == 0 {
                LNode::Dedent(Box::new(node_comment))
            } else {
                node_comment
            };
            seq.push(node_comment);

            if let Some(comment_str_rest) = comment_str_rest {
                report_newline(seq.len());
                seq.push(LNode::Dedent(Box::new(LNode::AlwaysStr(comment_str_rest))));
            }

            // add a suffix if necessary
            if is_line_comment {
                report_newline(seq.len());
                seq.push(LNode::AlwaysNewline);
            } else if prev_space {
                seq.push(LNode::Space);
            }
        }

        // preserve newlines between comments
        if !next_is_first_comment {
            let len_before = seq.len();
            if self.preserve_empty_lines(seq, false) {
                report_newline(len_before);
            }
        }

        // move escaping nodes into an escape node
        if let Some(escape_index) = escape_index {
            let seq_escaping = seq.split_off(escape_index);
            seq.push(LNode::EscapeGroupIfLast((), Box::new(LNode::Sequence(seq_escaping))));
        }
    }

    fn collect_comments_all(&mut self, prev_space: bool, seq: &mut Vec<LNode<'s>>) {
        self.collect_comments(prev_space, seq, |_| true);
    }

    fn collect_comments_on_prev_line(&mut self, prev_space: bool, seq: &mut Vec<LNode<'s>>) {
        let mut prev_end = self
            .prev_token()
            .map(|prev_token| self.offsets.expand_pos(prev_token.span.end()));
        self.collect_comments(prev_space, seq, |span| {
            let accept = prev_end.is_none_or(|prev_end| span.start.line_0 == prev_end.line_0);
            prev_end = Some(span.end);
            accept
        });
    }

    fn collect_comments_on_lines_before_real_token(&mut self, prev_space: bool, seq: &mut Vec<LNode<'s>>) {
        let non_comment_start = self
            .find_next_non_comment_token()
            .map(|token| self.offsets.expand_pos(self.source_tokens[token.0].span.start()));
        self.collect_comments(prev_space, seq, |span| match non_comment_start {
            Some(non_comment_start) => span.end.line_0 < non_comment_start.line_0,
            None => true,
        })
    }

    fn preserve_empty_lines(&self, seq: &mut Vec<LNode<'s>>, allow_blank: bool) -> bool {
        let mut any_newlines = false;

        if let Some(prev) = self.prev_token()
            && let Some(next) = self.peek_token()
        {
            let prev_end = self.offsets.expand_pos(prev.span.end());
            let next_start = self.offsets.expand_pos(next.span.start());

            let delta = next_start.line_0 - prev_end.line_0;

            if delta > 1 && allow_blank {
                any_newlines = true;
                seq.push(LNode::AlwaysBlankLine);
            } else if delta > 0 {
                any_newlines = true;
                seq.push(LNode::AlwaysNewline);
            }
        }

        any_newlines
    }

    fn map_root(&mut self, node: &HNode) -> Result<LNode<'s>, TokenMismatch> {
        let mut seq = vec![self.map(false, false, node)?];
        self.collect_comments_all(false, &mut seq);
        Ok(LNode::Sequence(seq))
    }

    fn map(&mut self, prev_space: bool, next_wrap_comma: bool, node: &HNode) -> Result<LNode<'s>, TokenMismatch> {
        let result = match node {
            HNode::Space => LNode::Space,
            &HNode::AlwaysToken(expected) => {
                let mut seq = vec![];
                self.collect_comments_all(prev_space, &mut seq);

                let token_index = match self.pop_token() {
                    Some(index) => index,
                    None => {
                        return Err(TokenMismatch { index: None, expected });
                    }
                };

                let token = &self.source_tokens[token_index.0];
                if token.ty != expected {
                    return Err(TokenMismatch {
                        index: Some(token_index),
                        expected,
                    });
                }

                let token_str = &self.source[token.span.range_bytes()];
                seq.push(LNode::AlwaysStr(token_str));

                let capture_trailing_comments = if next_wrap_comma || token.ty == TT::StringSubStart {
                    false
                } else {
                    // capture comments after the current token only if there is no next real token on the same line
                    //   if there is one, prefer to capture comments as prefixes to that token
                    let real_token_on_same_line = match self.find_next_non_comment_token() {
                        None => false,
                        Some(next) => {
                            let token_end = self.offsets.expand_pos(token.span.end());
                            let real_start = self.offsets.expand_pos(self.source_tokens[next.0].span.start());
                            token_end.line_0 == real_start.line_0
                        }
                    };
                    !real_token_on_same_line
                };
                if capture_trailing_comments {
                    self.collect_comments_on_prev_line(prev_space, &mut seq);
                }

                LNode::Sequence(seq)
            }
            HNode::WrapComma => {
                // TODO doc, especially comment handling
                let mut seq = vec![];
                if let Some(token) = self.find_next_non_comment_token()
                    && self.source_tokens[token.0].ty == TT::Comma
                {
                    self.collect_comments_all(prev_space, &mut seq);
                }

                seq.push(LNode::WrapStr(","));

                // if there was a comma in the source, pop it
                if let Some(token) = self.peek_token()
                    && token.ty == TT::Comma
                {
                    // TODO force wrap if there was a trailing comma in the source, like Black?
                    let _ = self.pop_token();
                }

                self.collect_comments_on_prev_line(prev_space, &mut seq);

                LNode::Sequence(seq)
            }
            HNode::AlwaysNewline => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(prev_space, &mut seq);
                seq.push(LNode::AlwaysNewline);
                LNode::Sequence(seq)
            }
            HNode::WrapNewline => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(prev_space, &mut seq);
                seq.push(LNode::WrapNewline);
                LNode::Sequence(seq)
            }
            HNode::AlwaysBlankLine => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(prev_space, &mut seq);
                seq.push(LNode::AlwaysBlankLine);
                LNode::Sequence(seq)
            }
            HNode::ForceWrap => LNode::ForceWrap,
            HNode::Indent(inner) => {
                let mut seq = vec![self.map(prev_space, next_wrap_comma, inner)?];
                if !next_wrap_comma {
                    self.collect_comments_on_lines_before_real_token(prev_space, &mut seq);
                }
                LNode::Indent(Box::new(LNode::Sequence(seq)))
            }
            HNode::Dedent(inner) => LNode::Dedent(Box::new(self.map(prev_space, next_wrap_comma, inner)?)),
            HNode::Sequence(children) => {
                let mut seq = vec![];
                for (child_i, child) in enumerate(children) {
                    let child_prev_space = seq_ends_with_space(&seq).unwrap_or(prev_space);
                    let child_next_wrap_comma =
                        seq_starts_with_wrap_comma(&children[child_i + 1..]).unwrap_or(next_wrap_comma);

                    seq.push(self.map(child_prev_space, child_next_wrap_comma, child)?);
                }
                LNode::Sequence(seq)
            }
            HNode::Group(inner) => LNode::Group(Box::new(self.map(prev_space, next_wrap_comma, inner)?)),
            HNode::Fill(children) => {
                let mut mapped = vec![];
                for child in children {
                    let child_prev_space = fill_ends_with_space(&mapped).unwrap_or(prev_space);
                    mapped.push(self.map(child_prev_space, false, child)?);
                }
                LNode::Fill(mapped)
            }
            HNode::PreserveBlankLines { last: end } => {
                let mut seq_escaping = vec![];

                let next_token_is_comment = self.peek_token().is_some_and(|t| t.ty.category() == TC::Comment);

                if !end || next_token_is_comment {
                    self.preserve_empty_lines(&mut seq_escaping, true);

                    let len_before = seq_escaping.len();
                    self.collect_comments_all(prev_space, &mut seq_escaping);

                    if !end && seq_escaping.len() > len_before {
                        self.preserve_empty_lines(&mut seq_escaping, true);
                    }
                }

                if seq_escaping.is_empty() {
                    LNode::EMPTY
                } else {
                    LNode::EscapeGroupIfLast((), Box::new(LNode::Sequence(seq_escaping)))
                }
            }
        };
        Ok(result)
    }
}

// TODO these should maybe be moved to `low`, we could add a space-capturing structure there
fn node_ends_with_space(node: &LNode) -> Option<bool> {
    match node {
        LNode::Space => Some(true),
        LNode::AlwaysStr(_)
        | LNode::WrapStr(_)
        | LNode::AlwaysNewline
        | LNode::WrapNewline
        | LNode::AlwaysBlankLine => Some(false),
        LNode::ForceWrap => None,
        LNode::Indent(inner) => node_ends_with_space(inner),
        LNode::Dedent(inner) => node_ends_with_space(inner),
        LNode::Sequence(seq) => seq_ends_with_space(seq),
        LNode::Group(inner) => node_ends_with_space(inner),
        LNode::Fill(children) => fill_ends_with_space(children),
        LNode::EscapeGroupIfLast(_, inner) => node_ends_with_space(inner),
    }
}

fn seq_ends_with_space(seq: &[LNode]) -> Option<bool> {
    seq.iter().rev().find_map(node_ends_with_space)
}

fn fill_ends_with_space(children: &[LNode]) -> Option<bool> {
    // there is an implicit [LNode::WrapNewLine] after each child, which counts as "not a space"
    if children.is_empty() { None } else { Some(false) }
}

fn node_starts_with_wrap_comma(node: &HNode) -> Option<bool> {
    match node {
        HNode::WrapComma => Some(true),
        HNode::Space
        | HNode::AlwaysToken(_)
        | HNode::AlwaysNewline
        | HNode::WrapNewline
        | HNode::AlwaysBlankLine
        | HNode::ForceWrap => Some(false),
        HNode::Indent(inner) => node_starts_with_wrap_comma(inner),
        HNode::Dedent(inner) => node_starts_with_wrap_comma(inner),
        HNode::Sequence(seq) => seq_starts_with_wrap_comma(seq),
        HNode::Group(inner) => node_starts_with_wrap_comma(inner),
        HNode::Fill(children) => fill_starts_with_wrap_comma(children),
        HNode::PreserveBlankLines { last: _ } => None,
    }
}

fn seq_starts_with_wrap_comma(seq: &[HNode]) -> Option<bool> {
    seq.iter().find_map(node_starts_with_wrap_comma)
}

fn fill_starts_with_wrap_comma(children: &[HNode]) -> Option<bool> {
    if let Some(first) = children.first() {
        if let Some(result) = node_starts_with_wrap_comma(first) {
            Some(result)
        } else {
            Some(false)
        }
    } else {
        None
    }
}
