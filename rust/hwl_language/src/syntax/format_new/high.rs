use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::format_new::low::LNode;
use crate::syntax::pos::{LineOffsets, SpanFull};
use crate::syntax::token::{Token, TokenCategory, TokenType as TT};
use hwl_util::swriteln;
use itertools::Itertools;

// TODO doc
// TODO rename "always" variants to just their base name, it's pretty verbose
// TODO add special "preserve blank lines" node?
#[derive(Debug)]
pub enum HNode {
    Space,

    AlwaysToken(TT),
    WrapComma,

    AlwaysNewline,
    WrapNewline,

    // TODO we don't really need a distinction between these, right?
    AlwaysIndent(Box<HNode>),
    WrapIndent(Box<HNode>),

    Sequence(Vec<HNode>),
    Group(Box<HNode>),
    Fill(Vec<HNode>),

    PreserveBlankLines,
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
            HNode::PreserveBlankLines => swriteln!(f, "PreserveBlankLines"),
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
        if self.next_source_index <= self.source_tokens.len() {
            let token_index = SourceTokenIndex(self.next_source_index);
            self.next_source_index += 1;
            Some(token_index)
        } else {
            None
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

    fn collect_comments(&mut self, seq: &mut Vec<LNode<'s>>, filter: impl Fn(SpanFull) -> bool) {
        // TODO preserve blank lines before/after/between comments?
        let mut first_comment = true;

        while let Some(token) = self.peek_token() {
            if token.ty.category() != TokenCategory::Comment {
                break;
            }
            let token_span = self.offsets.expand_span(token.span);
            if !filter(token_span) {
                break;
            }

            if !first_comment {
                self.preserve_blank_lines(seq);
            }
            first_comment = false;

            let _ = self.pop_token().unwrap();
            let token_str = &self.source[token.span.range_bytes()];

            match token.ty {
                TT::LineComment => {
                    seq.push(LNode::Space);
                    seq.push(LNode::AlwaysStr(token_str));
                    seq.push(LNode::AlwaysNewline);
                }
                TT::BlockComment => {
                    seq.push(LNode::Space);
                    seq.push(LNode::AlwaysStr(token_str));
                    seq.push(LNode::Space);
                }
                _ => break,
            }
        }
    }

    fn collect_comments_all(&mut self, seq: &mut Vec<LNode<'s>>) {
        self.collect_comments(seq, |_| true);
    }

    fn collect_comments_on_prev_line(&mut self, seq: &mut Vec<LNode<'s>>) {
        let prev_end = self
            .prev_token()
            .map(|prev_token| self.offsets.expand_pos(prev_token.span.end()));
        self.collect_comments(seq, |span| {
            prev_end.is_none_or(|prev_end| span.end.line_0 == prev_end.line_0)
        });
    }

    fn collect_comments_on_lines_before_real_token(&mut self, seq: &mut Vec<LNode<'s>>) {
        let non_comment_start = self
            .find_next_non_comment_token()
            .map(|token| self.offsets.expand_pos(self.source_tokens[token.0].span.start()));
        self.collect_comments(seq, |span| match non_comment_start {
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
        let mut seq = vec![self.map(node)?];
        self.collect_comments_on_lines_before_real_token(&mut seq);
        Ok(LNode::Sequence(seq))
    }

    // TODO doc: comments inside this node are captures, before/after are the responsibility of the caller
    // TODO collect comments before/after
    fn map(&mut self, node: HNode) -> Result<LNode<'s>, TokenMismatch> {
        let result = match node {
            HNode::Space => LNode::Space,
            HNode::AlwaysToken(expected) => {
                let mut seq = vec![];
                self.collect_comments_all(&mut seq);

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
                if let Some(token) = self.peek_token()
                    && token.ty == TT::Comma
                {
                    let _ = self.pop_token();
                }
                LNode::WrapStr(",")
            }
            HNode::AlwaysNewline => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(&mut seq);
                if !matches!(seq.last(), Some(LNode::AlwaysNewline)) {
                    seq.push(LNode::AlwaysNewline);
                }
                LNode::Sequence(seq)
            }
            HNode::WrapNewline => {
                let mut seq = vec![];
                self.collect_comments_on_prev_line(&mut seq);
                if !matches!(seq.last(), Some(LNode::AlwaysNewline)) {
                    seq.push(LNode::WrapNewline);
                }
                LNode::Sequence(seq)
            }

            HNode::AlwaysIndent(inner) => {
                let mut seq = vec![self.map(*inner)?];
                self.collect_comments_on_lines_before_real_token(&mut seq);
                LNode::AlwaysIndent(Box::new(LNode::Sequence(seq)))
            }
            HNode::WrapIndent(inner) => {
                let mut seq = vec![self.map(*inner)?];
                self.collect_comments_on_lines_before_real_token(&mut seq);
                LNode::WrapIndent(Box::new(LNode::Sequence(seq)))
            }
            HNode::Sequence(children) => {
                LNode::Sequence(children.into_iter().map(|child| self.map(child)).try_collect()?)
            }
            HNode::Group(inner) => LNode::Group(Box::new(self.map(*inner)?)),
            HNode::Fill(children) => LNode::Fill(children.into_iter().map(|child| self.map(child)).try_collect()?),
            HNode::PreserveBlankLines => {
                let mut seq = vec![];
                self.preserve_blank_lines(&mut seq);
                let len_before = seq.len();
                self.collect_comments_all(&mut seq);
                if seq.len() > len_before {
                    self.preserve_blank_lines(&mut seq);
                }
                LNode::Sequence(seq)
            }
        };
        Ok(result)
    }
}
