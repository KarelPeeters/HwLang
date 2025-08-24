use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::Token;
use crate::util::{Never, ResultNeverExt};
use hwl_util::swriteln;

pub enum LNode {
    Literal(&'static str),
    Token(SourceTokenIndex),

    // TODO make all of these just variants of literal
    NewLine,

    Indent(Box<LNode>),
    Sequence(Vec<LNode>),

    Group(LGroup),
    BranchWrap { no_wrap: &'static str, wrap: &'static str },
    IfNotFirstOnLine(&'static str),
}

pub struct LGroup {
    pub compact: bool,
    pub children: Vec<LNode>,
}

// TODO move to separate file?
pub fn node_to_string(
    settings: &FormatSettings,
    source_str: &str,
    source_offsets: &LineOffsets,
    source_tokens: &[Token],
    root: &LNode,
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

impl LNode {
    pub fn debug_str(&self) -> String {
        let mut f = String::new();
        self.debug_str_impl(&mut f, 0);
        f
    }

    fn debug_str_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        match self {
            LNode::Literal(s) => swriteln!(f, "Literal({s:?})"),
            LNode::Token(index) => swriteln!(f, "Token({index:?})"),
            LNode::NewLine => swriteln!(f, "NewLine"),
            LNode::Indent(child) => {
                swriteln!(f, "Indent");
                child.debug_str_impl(f, indent + 1);
            }
            LNode::Sequence(children) => {
                swriteln!(f, "Sequence");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                }
            }
            LNode::Group(group) => {
                let LGroup { compact, children } = group;
                swriteln!(f, "Group(compact={compact})");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                }
            }
            LNode::BranchWrap { no_wrap, wrap } => {
                swriteln!(f, "BranchWrap(no_wrap={no_wrap:?}, wrap={wrap:?})");
            }
            LNode::IfNotFirstOnLine(s) => {
                swriteln!(f, "IfNotFirstOnLine({s:?})");
            }
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

    fn write_node<W: MaybeWrap>(&mut self, node: &LNode) -> Result<(), W::E> {
        match node {
            LNode::Literal(literal_str) => {
                self.write_str::<W>(literal_str)?;
            }
            LNode::Token(index) => {
                let token_span = self.source_tokens[index.0].span;
                let token_str = &self.source_str[token_span.range_bytes()];
                self.write_str::<W>(token_str)?;
            }
            LNode::NewLine => {
                W::require_wrap()?;
                self.write_newline();
            }
            LNode::Indent(_) => todo!(),
            LNode::Sequence(children) => {
                self.write_sequence::<W>(children)?;
            }
            LNode::Group(_) => {
                todo!("is this even reachable?")
            }
            LNode::BranchWrap { no_wrap, wrap } => {
                // TODO is this condition correct, or should we pass this as a separate param from above?
                if !W::allow_wrap() {
                    self.write_str::<W>(no_wrap)?;
                } else {
                    self.write_str::<W>(wrap)?;
                }
            }
            LNode::IfNotFirstOnLine(s) => {
                if self.state.curr_line_start != self.result.len() {
                    self.write_str::<W>(s)?;
                }
            }
        }
        Ok(())
    }

    fn write_sequence<W: MaybeWrap>(&mut self, children: &[LNode]) -> Result<(), W::E> {
        let Some((child, rest)) = children.split_first() else {
            return Ok(());
        };

        match child {
            LNode::Sequence(_) => todo!("err, should be flattened"),
            // simple nodes without a wrapping decision, just write them
            LNode::Literal(_)
            | LNode::Token(_)
            | LNode::NewLine
            | LNode::Indent(_)
            | LNode::BranchWrap { .. }
            | LNode::IfNotFirstOnLine(_) => {
                let check = self.checkpoint();
                self.write_node::<W>(child)?;
                self.write_sequence::<W>(rest).inspect_err(|_| self.restore(check))?;
                Ok(())
            }
            // for groups we have to make a choice
            LNode::Group(group) => {
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

                // if we need to wrap, roll back and re-write everything with wrapping
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

    fn write_comma_group<W: MaybeWrap>(&mut self, group: &LGroup) -> Result<(), W::E> {
        let LGroup { compact, children } = group;
        if *compact {
            todo!()
        }

        // here we interpret W as "wrap", not "allow wrap"
        let wrap = W::allow_wrap();

        if !wrap {
            let check = self.checkpoint();
            for child in children {
                self.write_node::<W>(child).inspect_err(|_| self.restore(check))?;
            }
        } else {
            self.indent(|slf| {
                slf.write_newline();
                for child in children {
                    slf.write_node::<AllowWrap>(child).remove_never();
                    slf.write_newline();
                }
            });
        }

        Ok(())
    }
}
