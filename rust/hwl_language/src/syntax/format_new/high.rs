use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::format_new::low::LNode;
use crate::syntax::pos::{LineOffsets, SpanFull};
use crate::syntax::token::{Token, TokenCategory as TC, TokenType as TT};
use hwl_util::swriteln;

// TODO doc
// TODO rename "always" variants to just their base name, it's pretty verbose
// TODO add special "preserve blank lines" node?
#[must_use]
#[derive(Debug)]
pub enum HNode {
    Space,
    AlwaysToken(TT),
    WrapComma,
    AlwaysNewline,
    WrapNewline,
    ForceWrap,
    // TODO we don't really need a distinction between these wrapping nodes, right?
    AlwaysIndent(Box<HNode>),
    WrapIndent(Box<HNode>),
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

// TODO find a better name
pub fn lower_nodes<'s>(
    source: &'s str,
    offsets: &LineOffsets,
    source_tokens: &[Token],
    node: HNode,
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

    pub fn tree_string(&self) -> String {
        let mut f = String::new();
        self.tree_string_impl(&mut f, 0);
        f
    }

    fn tree_string_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        let swrite_children = |f: &mut String, cs: &[HNode]| {
            for c in cs {
                c.tree_string_impl(f, indent + 1);
            }
        };
        match self {
            HNode::Space => swriteln!(f, "Space"),
            HNode::AlwaysToken(ty) => swriteln!(f, "AlwaysToken({ty:?})"),
            HNode::WrapComma => swriteln!(f, "WrapComma"),
            HNode::AlwaysNewline => swriteln!(f, "AlwaysNewline"),
            HNode::AlwaysIndent(child) => {
                swriteln!(f, "AlwaysIndent");
                child.tree_string_impl(f, indent + 1);
            }
            HNode::WrapNewline => swriteln!(f, "WrapNewline"),
            HNode::ForceWrap => swriteln!(f, "ForceWrap"),
            HNode::WrapIndent(child) => {
                swriteln!(f, "WrapIndent");
                child.tree_string_impl(f, indent + 1);
            }
            HNode::Sequence(children) => {
                swriteln!(f, "Sequence");
                swrite_children(f, children);
            }
            HNode::Group(child) => {
                swriteln!(f, "Group");
                child.tree_string_impl(f, indent + 1);
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

    fn collect_comments(&mut self, prev_space: bool, seq: &mut Vec<LNode<'s>>, filter: impl Fn(SpanFull) -> bool) {
        let prev_space = seq_ends_with_space(seq).unwrap_or(prev_space);
        let mut first_comment = true;

        while let Some(token) = self.peek_token() {
            // filtering
            if token.ty.category() != TC::Comment {
                break;
            }
            let token_span = self.offsets.expand_span(token.span);
            if !filter(token_span) {
                break;
            }

            // preserve newlines
            if !first_comment {
                self.preserve_blank_lines(seq);
            }
            first_comment = false;

            // copy over string
            let _ = self.pop_token().unwrap();
            let comment_str = &self.source[token.span.range_bytes()];
            let mut seq_inner = vec![LNode::AlwaysStr(comment_str)];

            // add suffix
            match token.ty {
                TT::LineComment => {
                    seq_inner.push(LNode::AlwaysNewline);
                }
                TT::BlockComment => {
                    if prev_space {
                        seq_inner.push(LNode::Space);
                    }
                }
                _ => unreachable!(),
            };

            // dedent or prefix space
            let token_node = if token_span.start.col_0 == 0 {
                LNode::Dedent(Box::new(LNode::Sequence(seq_inner)))
            } else {
                seq_inner.insert(0, LNode::Space);
                LNode::Sequence(seq_inner)
            };
            seq.push(token_node);
        }
    }

    fn collect_comments_all(&mut self, prev_space: bool, seq: &mut Vec<LNode<'s>>) {
        self.collect_comments(prev_space, seq, |_| true);
    }

    fn collect_comments_on_prev_line(&mut self, prev_space: bool, seq: &mut Vec<LNode<'s>>) {
        let prev_end = self
            .prev_token()
            .map(|prev_token| self.offsets.expand_pos(prev_token.span.end()));
        self.collect_comments(prev_space, seq, |span| {
            prev_end.is_none_or(|prev_end| span.end.line_0 == prev_end.line_0)
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

    fn preserve_blank_lines(&self, seq: &mut Vec<LNode<'s>>) {
        if let Some(prev) = self.prev_token()
            && let Some(next) = self.peek_token()
        {
            let prev_end = self.offsets.expand_pos(prev.span.end());
            let next_start = self.offsets.expand_pos(next.span.start());

            if next_start.line_0 > prev_end.line_0 + 1 {
                seq.push(LNode::AlwaysNewline);
            }
        }
    }

    fn map_root(&mut self, node: HNode) -> Result<LNode<'s>, TokenMismatch> {
        let mut seq = vec![self.map(false, node)?];
        self.collect_comments_on_lines_before_real_token(false, &mut seq);
        Ok(LNode::Sequence(seq))
    }

    fn map(&mut self, prev_space: bool, node: HNode) -> Result<LNode<'s>, TokenMismatch> {
        let result = match node {
            HNode::Space => LNode::Space,
            HNode::AlwaysToken(expected) => {
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
                LNode::Sequence(seq)
            }
            HNode::WrapComma => {
                // TODO skip comments (_if_ the next non-comment token is a comma?)
                // TODO force wrap if there was a trailing comma in the source?
                if let Some(token) = self.peek_token()
                    && token.ty == TT::Comma
                {
                    let _ = self.pop_token();
                }
                LNode::WrapStr(",")
            }
            HNode::AlwaysNewline => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(prev_space, &mut seq);
                if !matches!(seq.last(), Some(LNode::AlwaysNewline)) {
                    seq.push(LNode::AlwaysNewline);
                }
                LNode::Sequence(seq)
            }
            HNode::WrapNewline => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(prev_space, &mut seq);
                if !matches!(seq.last(), Some(LNode::AlwaysNewline)) {
                    seq.push(LNode::WrapNewline);
                }
                LNode::Sequence(seq)
            }
            HNode::ForceWrap => LNode::ForceWrap,
            HNode::AlwaysIndent(inner) => {
                let mut seq = vec![self.map(prev_space, *inner)?];
                self.collect_comments_on_lines_before_real_token(prev_space, &mut seq);
                LNode::AlwaysIndent(Box::new(LNode::Sequence(seq)))
            }
            HNode::WrapIndent(inner) => {
                let mut seq = vec![self.map(prev_space, *inner)?];
                self.collect_comments_on_lines_before_real_token(prev_space, &mut seq);
                LNode::WrapIndent(Box::new(LNode::Sequence(seq)))
            }
            HNode::Sequence(children) => {
                let mut seq = vec![];
                for child in children {
                    let child_prev_space = seq_ends_with_space(&seq).unwrap_or(prev_space);
                    seq.push(self.map(child_prev_space, child)?);
                }
                LNode::Sequence(seq)
            }
            HNode::Group(inner) => LNode::Group(Box::new(self.map(prev_space, *inner)?)),
            HNode::Fill(children) => {
                let mut mapped = vec![];
                for child in children {
                    let child_prev_space = fill_ends_with_space(&mapped).unwrap_or(prev_space);
                    mapped.push(self.map(child_prev_space, child)?);
                }
                LNode::Fill(mapped)
            }
            HNode::PreserveBlankLines { last: end } => {
                let mut seq = vec![];

                let next_token_is_comment = self.peek_token().is_some_and(|t| t.ty.category() == TC::Comment);

                if !end || next_token_is_comment {
                    self.preserve_blank_lines(&mut seq);

                    let len_before = seq.len();
                    self.collect_comments_all(prev_space, &mut seq);

                    if !end && seq.len() > len_before {
                        self.preserve_blank_lines(&mut seq);
                    }
                }

                LNode::Sequence(seq)
            }
        };
        Ok(result)
    }
}

fn node_ends_with_space(node: &LNode) -> Option<bool> {
    match node {
        LNode::Space => Some(true),
        LNode::AlwaysStr(_) | LNode::WrapStr(_) | LNode::AlwaysNewline | LNode::WrapNewline => Some(false),
        LNode::ForceWrap => None,
        LNode::AlwaysIndent(inner) => node_ends_with_space(inner),
        LNode::WrapIndent(inner) => node_ends_with_space(inner),
        LNode::Dedent(inner) => node_ends_with_space(inner),
        LNode::Sequence(seq) => seq_ends_with_space(seq),
        LNode::Group(inner) => node_ends_with_space(inner),
        LNode::Fill(children) => fill_ends_with_space(children),
    }
}

fn seq_ends_with_space(seq: &[LNode]) -> Option<bool> {
    seq.iter().rev().find_map(node_ends_with_space)
}

fn fill_ends_with_space(children: &[LNode]) -> Option<bool> {
    // there is an implicit [LNode::WrapNewLine] after each child, which counts as "not a space"
    if children.is_empty() { None } else { Some(false) }
}
