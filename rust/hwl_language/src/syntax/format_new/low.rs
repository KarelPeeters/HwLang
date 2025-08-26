use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::common::{SourceTokenIndex, swrite_indent};
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, is_whitespace_or_empty};
use crate::util::{Never, ResultNeverExt};
use hwl_util::swriteln;

// TODO more docs
pub enum LNode {
    Token(SourceTokenIndex),

    // TODO rename this to either "space", "hardspace" or "space if touching"
    //   is this actually used?
    Space,

    AlwaysStr(&'static str),
    WrapStr(&'static str),

    /// Always emit a newline, this forces any containing groups to wrap.
    AlwaysNewline,
    /// If the containing group is wrapping, emit a newline. If not, emit a space.
    WrapNewline,

    // TODO does this make sense?
    // TODO maybe we don't need separate indents
    /// Always indent the inner node.
    AlwaysIndent(Box<LNode>),
    /// If the containing group is wrapping, indent the inner node.
    WrapIndent(Box<LNode>),

    Sequence(Vec<LNode>),
    Group(Box<LNode>),
    Fill(Vec<LNode>),
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
            curr_line_start: 0,
            indent: 0,
            emit_space: false,
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
    curr_line_start: usize,
    indent: usize,
    emit_space: bool,
}

// TODO rename all of these to be more in the spirit of "is the parent wrapping"
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
            LNode::Token(index) => swriteln!(f, "Token({index:?})"),
            LNode::Space => swriteln!(f, "Space"),
            LNode::AlwaysStr(s) => swriteln!(f, "AlwaysStr({s:?})"),
            LNode::WrapStr(s) => swriteln!(f, "WrapStr({s:?})"),
            LNode::AlwaysNewline => swriteln!(f, "AlwaysNewline"),
            LNode::WrapNewline => swriteln!(f, "WrapNewline"),
            LNode::AlwaysIndent(child) => {
                swriteln!(f, "AlwaysIndent");
                child.debug_str_impl(f, indent + 1);
            }
            LNode::WrapIndent(child) => {
                swriteln!(f, "WrapIndent");
                child.debug_str_impl(f, indent + 1);
            }
            LNode::Sequence(children) => {
                swriteln!(f, "Sequence");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                }
            }
            LNode::Group(child) => {
                swriteln!(f, "Group");
                child.debug_str_impl(f, indent + 1);
            }
            LNode::Fill(children) => {
                swriteln!(f, "Fill");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                }
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

    fn write_newline<W: MaybeWrap>(&mut self) -> Result<(), W::E> {
        self.write_str::<W>(&self.settings.newline_str)
    }

    fn write_str<W: MaybeWrap>(&mut self, s: &str) -> Result<(), W::E> {
        // strings containing newlines need the parent to wrap
        let last_line_end = s.rfind(LineOffsets::LINE_ENDING_CHARS);
        if last_line_end.is_some() {
            W::require_wrap()?;
        }

        // emit indent if this is the first text on the line
        let at_line_start = self.state.curr_line_start == self.result.len();
        if at_line_start && !is_whitespace_or_empty(s) {
            for _ in 0..self.state.indent {
                self.result.push_str(&self.settings.indent_str);
            }
        }

        // emit space if asked and both the previous and next charactor are not whitespace
        if self.state.emit_space
            && self.state.curr_line_start != self.result.len()
            && let Some(prev_char) = self.result.chars().last()
            && !char_is_whitespace(prev_char)
            && let Some(next_char) = s.chars().next()
            && !char_is_whitespace(next_char)
        {
            self.result.push(' ');
        }
        if !s.is_empty() {
            self.state.emit_space = false;
        }

        // emit str itself
        // TODO if multiline, require wrap?
        // TODO if multiline, increment line state
        self.result.push_str(s);

        // update line state
        if let Some(last_line_end) = last_line_end {
            self.state.curr_line_start = self.result.len() - s.len() + last_line_end + 1;
        }

        Ok(())
    }

    fn write_node<W: MaybeWrap>(&mut self, node: &LNode) -> Result<(), W::E> {
        match node {
            LNode::Token(index) => {
                let token_span = self.source_tokens[index.0].span;
                let token_str = &self.source_str[token_span.range_bytes()];
                self.write_str::<W>(token_str)?;
            }
            LNode::Space => {
                self.state.emit_space = true;
            }
            LNode::AlwaysStr(s) => {
                self.write_str::<W>(s)?;
            }
            LNode::WrapStr(s) => {
                if W::allow_wrap() {
                    self.write_str::<W>(s)?;
                }
            }
            LNode::AlwaysNewline => {
                self.write_newline::<W>()?;
            }
            LNode::WrapNewline => {
                if W::allow_wrap() {
                    self.write_newline::<W>()?;
                }
            }
            LNode::AlwaysIndent(child) => {
                self.indent(|ctx| ctx.write_node::<W>(child))?;
            }
            LNode::WrapIndent(child) => {
                if W::allow_wrap() {
                    self.indent(|ctx| ctx.write_node::<W>(child))?;
                } else {
                    self.write_node::<W>(child)?;
                }
            }
            LNode::Sequence(children) => {
                self.write_sequence::<W>(children)?;
            }
            LNode::Group(_) => {
                todo!("is this even reachable?")
            }
            LNode::Fill(_) => {
                todo!("is this even reachable?")
            }
        }
        Ok(())
    }

    fn write_sequence<W: MaybeWrap>(&mut self, children: &[LNode]) -> Result<(), W::E> {
        let Some((child, rest)) = children.split_first() else {
            return Ok(());
        };

        match child {
            // for groups we have to make a choice
            // TODO it's strange that we're doing the group wrapping here instead of in group itself,
            //   is there a way to improve that?
            LNode::Group(child) => {
                // try without wrapping
                let check = self.checkpoint();
                let result_no_wrap = self.write_node::<NoWrap>(child);

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

                    self.write_node::<AllowWrap>(child).remove_never();
                    self.write_sequence::<AllowWrap>(rest).remove_never();
                }

                Ok(())
            }
            // nested sequences should have been flattened already
            LNode::Sequence(_) => todo!("err, should be flattened"),
            // simple nodes without a wrapping decision, just write them
            _ => {
                let check = self.checkpoint();
                self.write_node::<W>(child)?;
                self.write_sequence::<W>(rest).inspect_err(|_| self.restore(check))?;
                Ok(())
            }
        }
    }
}

fn char_is_whitespace(c: char) -> bool {
    let mut buffer = [0; 4];
    is_whitespace_or_empty(c.encode_utf8(&mut buffer))
}
