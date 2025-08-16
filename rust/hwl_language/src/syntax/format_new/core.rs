use crate::syntax::format::FormatSettings;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, TokenType as TT};
use crate::util::iter::IterExt;
use hwl_util::{swrite, swriteln};
use itertools::Itertools;

pub enum FNode {
    Space,
    Token(TT),
    Horizontal(Vec<FNode>),
    Vertical(Vec<FNode>),
    CommaList(FCommaList),
}

pub struct FCommaList {
    pub compact: bool,
    pub children: Vec<FNode>,
}

// TODO this could be turned into a single-pass algorithm,
//   by building the tree and computing the mapping at the same time
pub enum SNode {
    NonHorizontal(SNodeNonHorizontal),
    Horizontal(Vec<SNodeNonHorizontal>),
}

pub enum SNodeNonHorizontal {
    Space,
    Token(TT, Option<usize>),
    Vertical(Vec<SNode>),
    CommaList(SCommaList),
}

pub struct SCommaList {
    pub compact: bool,
    pub force_wrap: bool,
    pub children: Vec<SNode>,
}

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
            SNodeNonHorizontal::Space => swriteln!(f, "Space"),
            SNodeNonHorizontal::Token(ty, token_index) => {
                swriteln!(f, "Token({ty:?}, index={token_index:?})");
            }
            SNodeNonHorizontal::Vertical(children) => {
                swriteln!(f, "Vertical");
                for c in children {
                    c.tree_string_impl(f, indent + 1);
                }
            }
            SNodeNonHorizontal::CommaList(list) => {
                swriteln!(f, "CommaList(compact={}, force_wrap={})", list.compact, list.force_wrap);
                for c in &list.children {
                    c.tree_string_impl(f, indent + 1);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct TokenMismatch;

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
    let (node, _) = simplify_and_connect_nodes_impl(&mut ctx, node)?;
    Ok(node)
}

struct MatchContext<'a> {
    offsets: &'a LineOffsets,
    source_tokens: &'a [Token],
    next_source_index: usize,
}

// TODO find a better name
// TODO if we encounter any line comments, set force_wrap
fn simplify_and_connect_nodes_impl<'t>(ctx: &mut MatchContext, node: FNode) -> Result<(SNode, bool), TokenMismatch> {
    let result = match node {
        FNode::Space => (SNode::NonHorizontal(SNodeNonHorizontal::Space), false),
        FNode::Token(ty) => {
            let mut wrap = false;

            let token_index = loop {
                let curr_index = ctx.next_source_index;
                break match ctx.source_tokens.get(curr_index) {
                    Some(token) => {
                        // skip whitespace and comments, but take them into account for wrapping
                        let skip = match token.ty {
                            TT::WhiteSpace => true,
                            TT::LineComment => {
                                wrap = true;
                                true
                            }
                            TT::BlockComment => {
                                let span = ctx.offsets.expand_span(token.span);
                                wrap |= span.end.line_0 > span.start.line_0;
                                true
                            }
                            _ => false,
                        };
                        if skip {
                            ctx.next_source_index += 1;
                            continue;
                        }

                        if ty == token.ty {
                            ctx.next_source_index += 1;
                            Some(curr_index)
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
                let token = &ctx.source_tokens[token_index];
                let span = ctx.offsets.expand_span(token.span);
                wrap |= span.end.line_0 > span.start.line_0;
            };

            (SNode::NonHorizontal(SNodeNonHorizontal::Token(ty, token_index)), wrap)
        }
        FNode::Horizontal(children) => {
            let mut mapped = vec![];
            let mut wrap = false;
            for c in children {
                let (c_mapped, c_wrap) = simplify_and_connect_nodes_impl(ctx, c)?;
                match c_mapped {
                    SNode::NonHorizontal(non_hor) => mapped.push(non_hor),
                    SNode::Horizontal(hor) => mapped.extend(hor),
                }
                wrap |= c_wrap;
            }
            (SNode::Horizontal(mapped), wrap)
        }
        FNode::Vertical(children) => {
            let mapped = children
                .into_iter()
                .map(|c| {
                    let (c, _) = simplify_and_connect_nodes_impl(ctx, c)?;
                    Ok(c)
                })
                .try_collect_vec()?;
            (SNode::NonHorizontal(SNodeNonHorizontal::Vertical(mapped)), true)
        }
        FNode::CommaList(list) => {
            let FCommaList { compact, children } = list;
            let mut mapped = vec![];
            let mut wrap = false;
            for c in children {
                let (c_mapped, c_wrap) = simplify_and_connect_nodes_impl(ctx, c)?;
                mapped.push(c_mapped);
                wrap |= c_wrap;
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

// // TODO completely rework this into mapping from FNode into SNode
// fn match_tokens(
//     diags: &Diagnostics,
//     source_end_pos: Pos,
//     source_tokens: &[Token],
//     node_root: &FNode,
// ) -> DiagResult<Vec<Option<usize>>> {
//     let mut next_source_index = 0;
//     let mut map = vec![];
//
//     node_root.try_for_each_token(&mut |node_token_ty| {
//         let source_index = loop {
//             // find next real non-whitespace/comment token
//             let source_index = next_source_index;
//             let Some(source_token) = source_tokens.get(next_source_index) else {
//                 let e = diags.report_internal_error(
//                     Span::empty_at(source_end_pos),
//                     format!("failed to match token: node expects {node_token_ty:?} but reached end of source tokens"),
//                 );
//                 return Err(e);
//             };
//             if matches!(
//                 source_token.ty.category(),
//                 TokenCategory::WhiteSpace | TokenCategory::Comment
//             ) {
//                 // println!("skipping whitespace/comment");
//                 next_source_index += 1;
//                 continue;
//             }
//
//             // try to match it
//             // TODO this this logic
//             break if source_token.ty == node_token_ty {
//                 // println!("matched token {node_token_ty:?}");
//                 next_source_index += 1;
//                 Some(source_index)
//             } else if node_token_ty == TT::Comma {
//                 // skip over comma that was not present in source
//                 None
//             } else {
//                 // failed to find match
//                 let e = diags.report_internal_error(
//                     source_token.span,
//                     format!(
//                         "failed to match token: node expects {:?} but source has {:?}",
//                         node_token_ty, source_token.ty
//                     ),
//                 );
//                 return Err(e);
//             };
//         };
//
//         map.push(source_index);
//         Ok(())
//     })?;
//
//     Ok(map)
// }

struct StringBuilderContext<'a> {
    source_str: &'a str,
    source_tokens: &'a [Token],
    node_token_to_source_token: &'a Vec<Option<usize>>,

    settings: &'a FormatSettings,

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

// impl StringBuilderContext<'_> {
//     fn checkpoint(&self) -> CheckPoint {
//         CheckPoint {
//             result_len: self.result.len(),
//             state: self.state,
//         }
//     }
//
//     fn restore(&mut self, check: CheckPoint) {
//         assert!(self.result.len() >= check.result_len);
//         self.result.truncate(check.result_len);
//         self.state = check.state;
//     }
//
//     fn write_token(&mut self, ty: TT) {
//         let token_str = loop {
//             let node_index = self.state.next_node_token_index;
//             self.state.next_node_token_index += 1;
//
//             // TODO guard against out-of-bounds?
//             let source_token =
//                 self.node_token_to_source_token[node_index].map(|source_index| &self.source_tokens[source_index]);
//
//             // TODO this is slightly duplicate logic
//             break match source_token {
//                 None => {
//                     if ty == TT::Comma {
//                         ","
//                     } else {
//                         continue;
//                     }
//                 }
//                 Some(source_token) => {
//                     if source_token.ty == ty {
//                         &self.source_str[source_token.span.range_bytes()]
//                     } else if source_token.ty == TT::Comma {
//                         continue;
//                     } else {
//                         todo!("source token {:?}, token {:?}", source_token.ty, ty);
//                     }
//                 }
//             };
//         };
//
//         // indent if first token on current line
//         if self.state.curr_line_start == self.result.len() {
//             for _ in 0..self.state.indent {
//                 self.result.push_str(&self.settings.indent_str);
//             }
//         }
//
//         // TODO count newlines (important for comments and multi-line string literals)
//         // TODO dedicated, non-ugly multi-line string literals?
//         self.result.push_str(token_str);
//     }
//
//     fn write_space(&mut self) {
//         // TODO err if first thing on line?
//         self.result.push(' ');
//     }
//
//     fn write_newline(&mut self) {
//         self.result.push('\n');
//         self.state.curr_line_index += 1;
//         self.state.curr_line_start = self.result.len();
//     }
//
//     fn line_overflows(&self, check: CheckPoint) -> bool {
//         let line_start = check.state.curr_line_start;
//         let rest = &self.result[line_start..];
//         let line_len = rest.bytes().position(|c| c == b'\n').unwrap_or(rest.len());
//         line_len > self.settings.max_line_length
//     }
//
//     fn indent<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
//         self.state.indent += 1;
//         let r = f(self);
//         self.state.indent -= 1;
//         r
//     }
//
//     fn write_comma_list(&mut self, list: &FCommaList, wrap: bool) -> Result<(), NeedsWrap> {
//         let &FCommaList { compact, ref children } = list;
//         if compact {
//             todo!()
//         }
//
//         if wrap {
//             self.indent(|slf| {
//                 slf.write_newline();
//                 for child in children {
//                     slf.write_node(child, true).expect(MSG_WRAP);
//                     slf.write_token(TT::Comma);
//                     slf.write_newline();
//                 }
//             })
//         } else {
//             for (child, last) in children.iter().with_last() {
//                 self.write_node(child, false)?;
//                 if !last {
//                     self.write_token(TT::Comma);
//                     self.write_space();
//                 }
//             }
//         }
//
//         Ok(())
//     }
//
//     fn write_node(&mut self, node: &FNode, allow_wrap: bool) -> Result<(), NeedsWrap> {
//         // TODO indentation stuff?
//         match node {
//             FNode::NonH(node) => match node {
//                 FNodeNonHor::NonWrap(non_wrap) => {
//                     self.write_non_wrap(non_wrap);
//                     Ok(())
//                 }
//                 FNodeNonHor::CommaList(list) => {
//                     // TODO can this even happen without being in a horizontal?
//                     // self.write_comma_list(list, false);
//                     todo!()
//                 }
//             },
//             FNode::Horizontal(nodes) => {
//                 self.write_horizontal(nodes, allow_wrap)?;
//                 Ok(())
//             }
//         }
//     }
//
//     fn write_non_wrap(&mut self, node: &FNodeNonWrap) {
//         match node {
//             FNodeNonWrap::Space => {
//                 // TODO err if first thing on line?
//                 self.write_space();
//             }
//             &FNodeNonWrap::Token(ty) => {
//                 self.write_token(ty);
//             }
//             FNodeNonWrap::Vertical(nodes) => {
//                 for n in nodes {
//                     // TODO try single first, then go back to multiple? for vertical this feels weird, we don't actually care ourselves
//                     // TODO respect blank lines between items
//                     // within a vertical, nodes are always allowed to wrap
//                     let _ = self.write_node(n, true);
//                     self.write_newline();
//                 }
//             }
//         }
//     }
//
//     fn write_horizontal(&mut self, nodes: &[FNodeNonHor], allow_wrap: bool) -> Result<(), NeedsWrap> {
//         let (node, rest) = match nodes.split_first() {
//             None => return Ok(()),
//             Some(p) => p,
//         };
//
//         match node {
//             FNodeNonHor::NonWrap(node) => {
//                 // simple non-wrapping node, no decisions to take here
//                 self.write_non_wrap(node);
//                 self.write_horizontal(rest, allow_wrap)?;
//                 Ok(())
//             }
//             FNodeNonHor::CommaList(list) => {
//                 // comma list, we need to decide whether to wrap or not
//
//                 // try without wrapping first
//                 let check = self.checkpoint();
//                 let result_unwrapped = self.write_comma_list(list, false);
//
//                 // check if the elements needs wrapping
//                 let mut should_wrap = match result_unwrapped {
//                     Ok(()) => false,
//                     Err(NeedsWrap) => true,
//                 };
//                 // check if the line already overflows
//                 if !should_wrap {
//                     should_wrap = self.line_overflows(check);
//                 }
//
//                 // if no wrapping is needed yet, try writing the rest of the list
//                 // TODO this is just a perf optimization, we could also always do this
//                 //    and this can also be optimized more, as soon as we overflow deeper we know that we should bail
//                 if !should_wrap {
//                     self.write_horizontal(rest, allow_wrap)?;
//                     should_wrap = self.line_overflows(check)
//                 }
//
//                 // if we need to wrap, roll back and re-writing everything with wrapping
//                 if should_wrap {
//                     if !allow_wrap {
//                         return Err(NeedsWrap);
//                     }
//                     self.restore(check);
//                     self.write_comma_list(list, true).expect(MSG_WRAP);
//                     self.write_horizontal(rest, true).expect(MSG_WRAP);
//                 }
//
//                 Ok(())
//             }
//         }
//     }
// }
