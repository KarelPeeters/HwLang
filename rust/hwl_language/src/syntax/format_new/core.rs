use crate::syntax::format::FormatSettings;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, TokenType as TT};
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

// TODO this could be turned into a single-pass algorithm,
//   by building the tree and computing the mapping at the same time
#[derive(Debug)]
pub enum SNode {
    NonHorizontal(SNodeNonHorizontal),
    Horizontal(Vec<SNodeNonHorizontal>),
}

#[derive(Debug, Copy, Clone)]
struct SourceTokenIndex(usize);

#[derive(Debug)]
enum SNodeNonHorizontal {
    NonWrap(SNodeNonWrap),
    CommaList(SCommaList),
}

#[derive(Debug)]
enum SNodeNonWrap {
    Space,
    Token(TT, Option<SourceTokenIndex>),
    Vertical(Vec<SNode>),
}

#[derive(Debug)]
struct SCommaList {
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
        match self {
            SNode::NonHorizontal(non_hor) => non_hor.tree_string_impl(f, indent),
            SNode::Horizontal(children) => {
                swrite_indent(f, indent);
                swriteln!(f, "Horizontal");
                for c in children {
                    c.tree_string_impl(f, indent + 1);
                }
            }
        }
    }
}

impl SNodeNonHorizontal {
    fn tree_string_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        match self {
            SNodeNonHorizontal::NonWrap(node) => match node {
                SNodeNonWrap::Space => swriteln!(f, "Space"),
                SNodeNonWrap::Token(ty, index) => swriteln!(f, "Token({ty:?}, {index:?})"),
                SNodeNonWrap::Vertical(children) => {
                    swriteln!(f, "Vertical");
                    for c in children {
                        c.tree_string_impl(f, indent + 1);
                    }
                }
            },
            SNodeNonHorizontal::CommaList(list) => {
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

// TODO find a better name
pub fn simplify_and_connect_nodes(
    offsets: &LineOffsets,
    source_tokens: &[Token],
    node: FNode,
) -> Result<SNode, TokenMismatch> {
    let mut ctx = MatchContext {
        offsets,
        source_tokens,
        next_source_index: 0,
    };
    let (node, _) = ctx.map(node)?;
    Ok(node)
}

struct MatchContext<'a> {
    offsets: &'a LineOffsets,
    source_tokens: &'a [Token],
    next_source_index: usize,
}

impl MatchContext<'_> {
    fn map_token(&mut self, ty: TT) -> Result<(Option<SourceTokenIndex>, bool), TokenMismatch> {
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

        Ok((token_index, wrap))
    }

    fn map(&mut self, node: FNode) -> Result<(SNode, bool), TokenMismatch> {
        let result = match node {
            FNode::Space => (
                SNode::NonHorizontal(SNodeNonHorizontal::NonWrap(SNodeNonWrap::Space)),
                false,
            ),
            FNode::Token(ty) => {
                let (token_index, wrap) = self.map_token(ty)?;
                let mapped = SNode::NonHorizontal(SNodeNonHorizontal::NonWrap(SNodeNonWrap::Token(ty, token_index)));
                (mapped, wrap)
            }
            FNode::Horizontal(children) => {
                let mut children_mapped = vec![];
                let mut wrap = false;
                for child in children {
                    let (c_mapped, c_wrap) = self.map(child)?;
                    match c_mapped {
                        SNode::NonHorizontal(non_hor) => children_mapped.push(non_hor),
                        SNode::Horizontal(hor) => children_mapped.extend(hor),
                    }
                    wrap |= c_wrap;
                }
                (SNode::Horizontal(children_mapped), wrap)
            }
            FNode::Vertical(children) => {
                let children_mapped = children
                    .into_iter()
                    .map(|c| {
                        let (c, _) = self.map(c)?;
                        Ok(c)
                    })
                    .try_collect_vec()?;
                let mapped = SNode::NonHorizontal(SNodeNonHorizontal::NonWrap(SNodeNonWrap::Vertical(children_mapped)));
                (mapped, true)
            }
            FNode::CommaList(list) => {
                let FCommaList { compact, children } = list;
                let mut mapped = vec![];
                let mut wrap = false;
                for child in children {
                    let (child_mapped, child_wrap) = self.map(child)?;
                    let (comma_index, comma_wrap) = self.map_token(TT::Comma)?;
                    mapped.push((child_mapped, comma_index));
                    wrap |= child_wrap | comma_wrap;
                }
                let list_mapped = SCommaList {
                    compact,
                    force_wrap: wrap,
                    children: mapped,
                };
                (SNode::NonHorizontal(SNodeNonHorizontal::CommaList(list_mapped)), wrap)
            }
        };
        Ok(result)
    }
}

// TODO move to separate file?
pub fn node_to_string(settings: &FormatSettings, source_str: &str, source_tokens: &[Token], root: &SNode) -> String {
    let mut ctx = StringBuilderContext {
        settings,
        source_str,
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
        self.result.push(' ');
    }

    fn write_newline(&mut self) {
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
                for &(ref child, comma_index) in children {
                    slf.write_node(child, true).expect(MSG_WRAP);
                    slf.write_token(TT::Comma, comma_index);
                    slf.write_newline();
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
        match node {
            SNode::NonHorizontal(node) => match node {
                SNodeNonHorizontal::NonWrap(node) => self.write_node_non_wrap(node),
                SNodeNonHorizontal::CommaList(_) => todo!("is this even possible?"),
            },
            SNode::Horizontal(children) => self.write_horizontal(children, allow_wrap)?,
        }
        Ok(())
    }

    fn write_node_non_wrap(&mut self, node: &SNodeNonWrap) {
        match node {
            SNodeNonWrap::Space => self.write_space(),
            &SNodeNonWrap::Token(ty, index) => self.write_token(ty, index),
            SNodeNonWrap::Vertical(children) => {
                for child in children {
                    // TODO try single first, then go back to multiple? for vertical this feels weird, we don't actually care ourselves
                    // TODO respect blank lines between items
                    // within a vertical, nodes are always allowed to wrap
                    let _ = self.write_node(child, true);
                    self.write_newline();
                }
            }
        }
    }

    fn write_horizontal(&mut self, nodes: &[SNodeNonHorizontal], allow_wrap: bool) -> Result<(), NeedsWrap> {
        let (node, rest) = match nodes.split_first() {
            None => return Ok(()),
            Some(p) => p,
        };

        match node {
            SNodeNonHorizontal::NonWrap(node) => {
                // simple non-wrapping node, no decisions to take here
                self.write_node_non_wrap(node);
                self.write_horizontal(rest, allow_wrap)?;
                Ok(())
            }
            SNodeNonHorizontal::CommaList(list) => {
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
