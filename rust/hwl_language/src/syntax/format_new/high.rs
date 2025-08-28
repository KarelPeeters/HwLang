use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::format_new::low::LNode;
use crate::syntax::pos::{LineOffsets, SpanFull};
use crate::syntax::token::{Token, TokenCategory, TokenType as TT};
use crate::util::iter::IterExt;
use hwl_util::swriteln;

#[derive(Debug)]
pub enum HNode {
    Space,
    Token(TT),
    Horizontal(Vec<HNode>),
    Vertical(Vec<HNode>),
    CommaList(HCommaList),

    WrapNewline,
    WrapIndent(Box<HNode>),
    Group(Box<HNode>),
}

#[derive(Debug)]
pub struct HCommaList {
    pub fill: bool,
    pub children: Vec<HNode>,
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
    ctx.map(node)
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
            HNode::Token(ty) => swriteln!(f, "Token({ty:?})"),
            HNode::Horizontal(children) => {
                swriteln!(f, "Horizontal");
                swrite_children(f, children);
            }
            HNode::Vertical(children) => {
                swriteln!(f, "Vertical");
                swrite_children(f, children);
            }
            HNode::CommaList(HCommaList { fill, children }) => {
                swriteln!(f, "CommaList(fill={fill})");
                swrite_children(f, children);
            }
            HNode::WrapNewline => swriteln!(f, "WrapNewline"),
            HNode::WrapIndent(child) => {
                swriteln!(f, "WrapIndent");
                child.tree_string_impl(f, indent + 1);
            }
            HNode::Group(child) => {
                swriteln!(f, "Group");
                child.tree_string_impl(f, indent + 1);
            }
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
impl<'s> LowerContext<'s, '_> {
    fn prev_token(&self) -> Option<&Token> {
        if self.next_source_index == 0 {
            None
        } else {
            self.source_tokens.get(self.next_source_index - 1)
        }
    }

    fn peek_token(&self) -> Option<&Token> {
        self.source_tokens.get(self.next_source_index)
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
            todo!("expected {ty:?} at {token_index:?}, got {:?}", token.ty);
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

    fn preserve_blank_lines(&self, seq: &mut Vec<LNode<'s>>) {
        if let (Some(prev_token), Some(next_token)) = (self.prev_token(), self.peek_token()) {
            let prev_end = self.offsets.expand_pos(prev_token.span.end());
            let next_start = self.offsets.expand_pos(next_token.span.start());
            // TODO why +1 here and not in other places?
            if next_start.line_0 > prev_end.line_0 + 1 {
                seq.push(LNode::AlwaysNewline);
            }
        }
    }

    fn collect_comments(&mut self, seq: &mut Vec<LNode<'s>>, filter: impl Fn(SpanFull) -> bool) {
        while let Some(token) = self.peek_token()
            && filter(self.offsets.expand_span(token.span))
        {
            let token_span = token.span;
            match token.ty {
                TT::LineComment => {
                    // TODO preserve - 1?
                    // self.preserve_blank_lines(seq);
                    seq.push(LNode::Space);

                    let token_index = self.pop_token(token.ty).unwrap();
                    let token_str = &self.source[self.source_tokens[token_index.0].span.range_bytes()];
                    seq.push(LNode::AlwaysStr(token_str));

                    seq.push(LNode::AlwaysNewline);
                }
                TT::BlockComment => {
                    // TODO preserve - 1?
                    // TODO spaces before/after?
                    // self.preserve_blank_lines(seq);
                    seq.push(LNode::Space);

                    let token_index = self.pop_token(token.ty).unwrap();
                    let token_str = &self.source[self.source_tokens[token_index.0].span.range_bytes()];
                    seq.push(LNode::AlwaysStr(token_str));

                    seq.push(LNode::Space);

                    // TODO should preserving happen before or after tokens?
                    if let Some(next_token) = self.peek_token() {
                        let token_end_pos = self.offsets.expand_pos(token_span.end());
                        let next_start_pos = self.offsets.expand_pos(next_token.span.start());
                        if next_start_pos.line_0 > token_end_pos.line_0 {
                            seq.push(LNode::AlwaysNewline);
                        }
                    }
                }
                _ => break,
            }
        }
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

    fn collect_comments_all(&mut self, seq: &mut Vec<LNode<'s>>) {
        // TODO make newline recording configurable? eg. for comma lists, the first newline doesn't count
        //   but if there are multiple (_between items_) then force wrap
        self.collect_comments(seq, |_| true);
    }

    fn collect_comments_on_prev_line(&mut self, seq: &mut Vec<LNode<'s>>) {
        if let Some(prev_token) = self.prev_token() {
            let prev_end = self.offsets.expand_pos(prev_token.span.end());
            self.collect_comments(seq, |span| span.end.line_0 == prev_end.line_0);
        }
    }

    // TODO doc: comments inside this node are captures, before/after are the responsibility of the caller
    fn map(&mut self, node: HNode) -> Result<LNode<'s>, TokenMismatch> {
        let result = match node {
            HNode::Space => {
                let mut seq = vec![];
                self.collect_comments_all(&mut seq);
                seq.push(LNode::Space);
                self.collect_comments_on_prev_line(&mut seq);
                LNode::Sequence(seq)
            }
            HNode::Token(ty) => {
                let mut seq = vec![];
                self.collect_comments_all(&mut seq);

                let token_index = self.pop_token(ty)?;
                let token_str = &self.source[self.source_tokens[token_index.0].span.range_bytes()];
                seq.push(LNode::AlwaysStr(token_str));

                self.collect_comments_on_prev_line(&mut seq);
                LNode::Sequence(seq)
            }
            HNode::Horizontal(children) => {
                let mut seq = vec![];
                self.collect_comments_all(&mut seq);
                for child in children {
                    seq.push(self.map(child)?);
                    self.collect_comments_on_prev_line(&mut seq);
                }
                LNode::Sequence(seq)
            }
            HNode::Vertical(children) => {
                // TODO capture comments between nodes
                // TODO remember blank lines between both items and comments
                let mut seq = vec![];
                self.collect_comments_on_lines_before_real_token(&mut seq);
                for child in children {
                    self.preserve_blank_lines(&mut seq);
                    seq.push(self.map(child)?);
                    seq.push(LNode::AlwaysNewline);
                    self.collect_comments_on_lines_before_real_token(&mut seq);
                }
                LNode::Sequence(seq)
            }
            HNode::CommaList(list) => {
                // TODO remember blank lines between (but not before) items, and if there are any then force wrap
                let HCommaList { fill, children } = list;

                let mut seq = vec![];
                seq.push(LNode::WrapNewline);

                for (child, last) in children.into_iter().with_last() {
                    seq.push(self.map(child)?);

                    self.collect_comments_all(&mut seq);

                    // handle comma
                    if last {
                        if let Some(t) = self.peek_token()
                            && t.ty == TT::Comma
                        {
                            self.pop_token(TT::Comma)?;
                        }
                        seq.push(LNode::WrapStr(","));
                    } else {
                        self.pop_token(TT::Comma)?;
                        seq.push(LNode::AlwaysStr(","));
                        seq.push(LNode::Space);
                    }

                    self.collect_comments_on_prev_line(&mut seq);

                    if !fill {
                        seq.push(LNode::WrapNewline);
                    }
                }

                // TODO capture comments

                if fill {
                    LNode::Fill(seq)
                } else {
                    LNode::Group(Box::new(LNode::WrapIndent(Box::new(LNode::Sequence(seq)))))
                }
            }
            HNode::WrapNewline => LNode::WrapNewline,
            HNode::WrapIndent(child) => LNode::WrapIndent(Box::new(self.map(*child)?)),
            HNode::Group(child) => LNode::Group(Box::new(self.map(*child)?)),
        };
        Ok(result)
    }
}
