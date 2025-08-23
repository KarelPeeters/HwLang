use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::format_new::low::{LCommaList, LNode};
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, TokenCategory, TokenType as TT};
use hwl_util::swriteln;

#[derive(Debug)]
pub enum HNode {
    Space,
    Token(TT),
    Horizontal(Vec<HNode>),
    Vertical(Vec<HNode>),
    CommaList(HCommaList),
}

#[derive(Debug)]
pub struct HCommaList {
    pub compact: bool,
    pub children: Vec<HNode>,
}

#[derive(Debug)]
pub struct TokenMismatch {
    pub index: Option<SourceTokenIndex>,
    pub expected: TT,
}

// TODO find a better name
pub fn lower_nodes(offsets: &LineOffsets, source_tokens: &[Token], node: HNode) -> Result<LNode, TokenMismatch> {
    let mut ctx = LowerContext {
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
            HNode::CommaList(HCommaList { compact, children }) => {
                swriteln!(f, "CommaList(compact={compact})");
                swrite_children(f, children);
            }
        }
    }
}

struct LowerContext<'a> {
    offsets: &'a LineOffsets,
    source_tokens: &'a [Token],
    next_source_index: usize,
}

// TODO reverse order of functions
impl LowerContext<'_> {
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

    fn collect_vertical_comments_and_blank_lines(&mut self, result: &mut Vec<LNode>) -> Result<(), TokenMismatch> {
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
                    result.push(LNode::NewLine);
                }
            }

            match token.ty {
                TT::LineComment => {
                    // TODO space before?
                    result.push(LNode::Token(self.pop_token(token.ty)?));
                    result.push(LNode::NewLine);
                }
                TT::BlockComment => {
                    // TODO spaces before/after?
                    result.push(LNode::Token(self.pop_token(token.ty)?));
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn collect_horizontal_comments(&mut self, result: &mut Vec<LNode>) -> Result<(), TokenMismatch> {
        while let Some(token) = self.peek_token() {
            match token.ty {
                TT::LineComment => {
                    // TODO space before?
                    result.push(LNode::Token(self.pop_token(token.ty)?));
                    result.push(LNode::NewLine);
                }
                TT::BlockComment => {
                    // TODO space before/after?
                    result.push(LNode::Token(self.pop_token(token.ty)?));
                }
                _ => break,
            }
        }
        Ok(())
    }

    // TODO doc: comments inside this node are captures, before/after are the responsibility of the caller
    fn map(&mut self, node: HNode) -> Result<LNode, TokenMismatch> {
        let result = match node {
            HNode::Space => LNode::Literal(" "),
            HNode::Token(ty) => {
                let mut comments = vec![];
                self.collect_horizontal_comments(&mut comments)?;

                let node = LNode::Token(self.pop_token(ty)?);

                if comments.is_empty() {
                    node
                } else {
                    comments.push(node);
                    LNode::Sequence(comments)
                }
            }
            HNode::Horizontal(children) => {
                // TODO capture comments between nodes
                // TODO remember blank lines

                let mut children_mapped = vec![];
                for c in children {
                    let c_mapped = self.map(c)?;
                    push_to_sequence(&mut children_mapped, c_mapped);
                }
                LNode::Sequence(children_mapped)
            }
            HNode::Vertical(children) => {
                // TODO capture comments between nodes
                // TODO remember blank lines between both items and comments
                let mut children_mapped = vec![];
                for c in children {
                    self.collect_vertical_comments_and_blank_lines(&mut children_mapped)?;
                    let c_mapped = self.map(c)?;
                    push_to_sequence(&mut children_mapped, c_mapped);
                    children_mapped.push(LNode::NewLine);
                }
                self.collect_vertical_comments_and_blank_lines(&mut children_mapped)?;
                LNode::Sequence(children_mapped)
            }
            HNode::CommaList(list) => {
                // TODO capture comments between nodes
                // TODO here we also want to capture before/after to match indent?
                // TODO remember blank lines
                let HCommaList { compact, children } = list;
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
                LNode::CommaGroup(LCommaList {
                    compact,
                    children: children_mapped,
                })
            }
        };
        Ok(result)
    }
}

// TODO can we express this in the type system?
fn push_to_sequence(seq: &mut Vec<LNode>, node: LNode) {
    match node {
        LNode::Sequence(children) => seq.extend(children),
        _ => seq.push(node),
    }
}
