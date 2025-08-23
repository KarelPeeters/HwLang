use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::format_new::low::{LCommaList, LNode};
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, TokenCategory, TokenType as TT};
use crate::util::data::VecExt;
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
            panic!("expected {ty:?} at {token_index:?}, got {:?}", token.ty);
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

    fn collect_comments_on_lines_before_real_token(&mut self, seq: &mut SequenceBuilder) -> Result<(), TokenMismatch> {
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
                    seq.push(LNode::NewLine);
                }
            }

            match token.ty {
                TT::LineComment => {
                    // TODO space before?
                    seq.push(LNode::Token(self.pop_token(token.ty)?));
                    seq.push(LNode::NewLine);
                }
                TT::BlockComment => {
                    // TODO spaces before/after?
                    seq.push(LNode::Token(self.pop_token(token.ty)?));
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn collect_comments_all(&mut self, seq: &mut SequenceBuilder) {
        // TODO preserve blank lines between comments
        while let Some(token) = self.peek_token() {
            match token.ty {
                TT::LineComment => {
                    // TODO space before?
                    seq.push(LNode::Token(self.pop_token(token.ty).unwrap()));
                    seq.push(LNode::NewLine);
                }
                TT::BlockComment => {
                    // TODO space before/after?
                    seq.push(LNode::Token(self.pop_token(token.ty).unwrap()));
                }
                _ => break,
            }
        }
    }

    fn collect_comments_on_prev_line(&mut self, seq: &mut SequenceBuilder) {
        if let Some(prev_token) = self.prev_token() {
            let prev_end = self.offsets.expand_pos(prev_token.span.end());

            while let Some(token) = self.peek_token() {
                if self.offsets.expand_pos(token.span.start()).line_0 != prev_end.line_0 {
                    break;
                }
                match token.ty {
                    TT::LineComment => {
                        // TODO space before?
                        seq.push(LNode::Token(self.pop_token(token.ty).unwrap()));
                        seq.push(LNode::NewLine);
                    }
                    TT::BlockComment => {
                        // TODO space before/after?
                        seq.push(LNode::Token(self.pop_token(token.ty).unwrap()));
                    }
                    _ => break,
                }
            }
        }
    }

    // TODO doc: comments inside this node are captures, before/after are the responsibility of the caller
    fn map(&mut self, node: HNode) -> Result<LNode, TokenMismatch> {
        let result = match node {
            HNode::Space => {
                let mut seq = SequenceBuilder::new();
                self.collect_comments_all(&mut seq);
                seq.push(LNode::Literal(" "));
                self.collect_comments_on_prev_line(&mut seq);
                seq.build()
            }
            HNode::Token(ty) => {
                let mut seq = SequenceBuilder::new();
                self.collect_comments_all(&mut seq);
                seq.push(LNode::Token(self.pop_token(ty)?));
                self.collect_comments_on_prev_line(&mut seq);
                seq.build()
            }
            HNode::Horizontal(children) => {
                let mut seq = SequenceBuilder::new();
                self.collect_comments_all(&mut seq);
                for child in children {
                    seq.push(self.map(child)?);
                    self.collect_comments_on_prev_line(&mut seq);
                }
                seq.build()
            }
            HNode::Vertical(children) => {
                // TODO capture comments between nodes
                // TODO remember blank lines between both items and comments
                let mut seq = SequenceBuilder::new();
                self.collect_comments_on_lines_before_real_token(&mut seq)?;
                for child in children {
                    seq.push(self.map(child)?);
                    seq.push(LNode::NewLine);
                    self.collect_comments_on_lines_before_real_token(&mut seq)?;
                }
                seq.build()
            }
            HNode::CommaList(list) => {
                // TODO capture comments between nodes
                // TODO here we also want to capture before/after to match indent?
                // TODO remember blank lines
                let HCommaList { compact, children } = list;
                let mut mapped = vec![];

                for (i, child) in children.into_iter().enumerate() {
                    let child_mapped = self.map(child)?;

                    // TODO capture comments here

                    // TODO fix formatting for this
                    if let Some(t) = self.peek_token()
                        && t.ty == TT::Comma
                    {
                        let _ = self.pop_token(TT::Comma)?;
                    }

                    mapped.push(child_mapped);
                }

                // let mut suffix = vec![];
                // // TODO drop trailing newlines
                // self.collect_comments_on_lines_before_real_token(&mut suffix)?;
                // let suffix = if suffix.is_empty() {
                //     None
                // } else {
                //     Some(Box::new(LNode::Sequence(suffix)))
                // };

                LNode::CommaGroup(LCommaList {
                    compact,
                    children: mapped,
                    suffix: None,
                })
            }
        };
        Ok(result)
    }
}

struct SequenceBuilder {
    nodes: Vec<LNode>,
}
impl SequenceBuilder {
    fn new() -> Self {
        Self { nodes: vec![] }
    }

    fn push(&mut self, node: LNode) {
        match node {
            LNode::Sequence(children) => self.nodes.extend(children),
            _ => self.nodes.push(node),
        }
    }

    fn build(self) -> LNode {
        // TODO should we special-case single or not? we risk creating extra edge cases either way
        self.nodes.single().unwrap_or_else(LNode::Sequence)
    }
}
