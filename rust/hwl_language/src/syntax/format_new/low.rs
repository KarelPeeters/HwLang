use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::common::swrite_indent;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{Token, is_whitespace_or_empty};
use crate::util::data::VecExt;
use crate::util::{Never, ResultNeverExt};
use hwl_util::swriteln;

// TODO rename "always" variants to just their base name, it's pretty verbose
/// Low-level formatting nodes.
/// Based on the [Prettier commands](https://github.com/prettier/prettier/blob/main/commands.md).
pub enum LNode<'s> {
    /// Emit a space if the previous and next characters on the same line exist and are not whitespace.
    Space,

    /// Emit the given string.
    AlwaysStr(&'s str),
    /// Emit the given string if the containing group is wrapping.
    WrapStr(&'s str),

    /// Emit a newline. This forces any containing groups to wrap.
    AlwaysNewline,
    // TODO make this emit a space if not wrapping? or let callers add space nodes themselves?
    /// Emit a newline if the containing group is wrapping.
    WrapNewline,

    /// Indent the inner node.
    AlwaysIndent(Box<LNode<'s>>),
    /// Indent the inner node if the containing group is wrapping, if not do nothing.
    WrapIndent(Box<LNode<'s>>),

    /// A sequence of nodes to be emitted in order.
    /// Children can each individually decide to wrap or not,
    /// based on whether their contants overflow the start or end line.
    /// Any child wrapping forces the parent groups to wrap too.
    Sequence(Vec<LNode<'s>>),

    /// Groups are the mechanism to control wrapping.
    /// A group either wraps or does not wrap, which recursively affects all child nodes.
    /// Groups can be nested, in which case inner groups can only wrap if the outer group is wrapping,
    ///   but inner groups are allowed to not wrap even if the outer group is wrapping.
    Group(Box<LNode<'s>>),

    /// Similar to [LNode::Group], except that as many children as possible are placed on each line,
    /// instead of a single global wrapping decision for the entire group.
    ///
    /// After each child there's an implicit [LNode::WrapNewLine] after each child
    /// and an implicit [LNode::Indent] around all children.
    Fill(Vec<LNode<'s>>),
}

pub fn node_to_string(settings: &FormatSettings, source_str: &str, source_tokens: &[Token], root: &LNode) -> String {
    let mut ctx = StringBuilderContext {
        settings,
        source_str,
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

// TODO remove unused fields
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
    curr_line_start: usize,
    indent: usize,
    emit_space: bool,
}

// TODO rename all of these to be more in the spirit of "is the parent wrapping"
#[derive(Debug)]
struct NeedsWrap;

trait MaybeWrap {
    type E;
    fn is_wrapping() -> bool;
    fn require_wrapping() -> Result<(), Self::E>;
}

struct NoWrap {}
impl MaybeWrap for NoWrap {
    type E = NeedsWrap;
    fn is_wrapping() -> bool {
        false
    }
    fn require_wrapping() -> Result<(), NeedsWrap> {
        Err(NeedsWrap)
    }
}

struct AllowWrap {}
impl MaybeWrap for AllowWrap {
    type E = Never;
    fn is_wrapping() -> bool {
        true
    }
    fn require_wrapping() -> Result<(), Never> {
        Ok(())
    }
}

impl<'s> LNode<'s> {
    pub fn debug_str(&self) -> String {
        let mut f = String::new();
        self.debug_str_impl(&mut f, 0);
        f
    }

    fn debug_str_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        match self {
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

    pub fn simplify(self) -> LNode<'s> {
        match self {
            // actual simplification+
            LNode::Sequence(children) => {
                let mut result = Vec::with_capacity(children.len());
                for c in children {
                    match c.simplify() {
                        LNode::Sequence(inner) => result.extend(inner),
                        c_simple => result.push(c_simple),
                    }
                }
                result.single().unwrap_or_else(LNode::Sequence)
            }
            // just simplify children
            LNode::AlwaysIndent(child) => LNode::AlwaysIndent(Box::new(child.simplify())),
            LNode::WrapIndent(child) => LNode::WrapIndent(Box::new(child.simplify())),
            LNode::Group(child) => LNode::Group(Box::new(child.simplify())),
            LNode::Fill(children) => LNode::Fill(children.into_iter().map(LNode::simplify).collect()),
            // trivial cases
            LNode::Space => LNode::Space,
            LNode::AlwaysStr(s) => LNode::AlwaysStr(s),
            LNode::WrapStr(s) => LNode::WrapStr(s),
            LNode::AlwaysNewline => LNode::AlwaysNewline,
            LNode::WrapNewline => LNode::WrapNewline,
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
        // TODO use proper LineOffsets line-ending logic
        let line_start = check.state.curr_line_start;
        let rest = &self.result[line_start..];
        let line_len = rest.find(LineOffsets::LINE_ENDING_CHARS).unwrap_or(rest.len());
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
            W::require_wrapping()?;
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
        self.result.push_str(s);

        // update line state
        if let Some(last_line_end) = last_line_end {
            self.state.curr_line_start = self.result.len() - s.len() + last_line_end + 1;
        }

        Ok(())
    }

    fn write_node<W: MaybeWrap>(&mut self, node: &LNode) -> Result<(), W::E> {
        match node {
            LNode::Space => {
                self.state.emit_space = true;
            }
            LNode::AlwaysStr(s) => {
                self.write_str::<W>(s)?;
            }
            LNode::WrapStr(s) => {
                if W::is_wrapping() {
                    self.write_str::<W>(s)?;
                }
            }
            LNode::AlwaysNewline => {
                self.write_newline::<W>()?;
            }
            LNode::WrapNewline => {
                if W::is_wrapping() {
                    self.write_newline::<W>()?;
                }
            }
            LNode::AlwaysIndent(child) => {
                self.indent(|ctx| ctx.write_node::<W>(child))?;
            }
            LNode::WrapIndent(child) => {
                if W::is_wrapping() {
                    self.indent(|ctx| ctx.write_node::<W>(child))?;
                } else {
                    self.write_node::<W>(child)?;
                }
            }
            LNode::Sequence(children) => {
                self.write_sequence::<W>(children)?;
            }
            LNode::Group(_) | LNode::Fill(_) => {
                // If we meet a top-level group we just pretend it's a single-item sequence
                //   and delegate to the existing sequence logic for group wrapping.
                self.write_sequence::<W>(std::slice::from_ref(node))?;
            }
        }
        Ok(())
    }

    fn write_sequence<'s, W: MaybeWrap>(&mut self, children: &[LNode<'s>]) -> Result<(), W::E> {
        // flatten if necessary
        if children.iter().all(|c| !matches!(c, LNode::Sequence(_))) {
            self.write_sequence_impl::<W>(children.iter())
        } else {
            fn f<'s, 'c>(flat: &mut Vec<&'c LNode<'s>>, node: &'c LNode<'s>) {
                match node {
                    LNode::Sequence(children) => {
                        for child in children {
                            f(flat, child);
                        }
                    }
                    _ => flat.push(node),
                }
            }

            let mut flat = vec![];
            for c in children {
                f(&mut flat, c);
            }
            self.write_sequence_impl::<W>(flat.iter().copied())
        }
    }

    fn write_sequence_impl<'c, 's: 'c, W: MaybeWrap>(
        &mut self,
        mut children: impl Iterator<Item = &'c LNode<'s>> + Clone,
    ) -> Result<(), W::E> {
        let (child, rest) = match children.next() {
            None => return Ok(()),
            Some(child) => (child, children),
        };

        // wrapping logic is here instead of inside the group nodes,
        //   so we can take items that follow the group on the same line into account to check for line overflow
        match child {
            LNode::Group(child) => {
                self.write_sequence_branch::<W>(
                    rest,
                    |slf| slf.write_node::<NoWrap>(child),
                    |slf| slf.write_node::<AllowWrap>(child).remove_never(),
                )?;
            }
            LNode::Fill(children) => {
                self.write_sequence_branch::<W>(
                    rest,
                    |slf| {
                        for c in children {
                            slf.write_node::<NoWrap>(c)?;
                        }
                        Ok(())
                    },
                    |slf| {
                        slf.indent(|slf| {
                            slf.write_newline::<AllowWrap>().remove_never();
                            let mut first_after_break = true;
                            for c in children {
                                let check = slf.checkpoint();
                                slf.write_node::<AllowWrap>(c).remove_never();
                                if slf.line_overflows(check) && !first_after_break {
                                    slf.restore(check);
                                    slf.write_newline::<AllowWrap>().remove_never();
                                    slf.write_node::<AllowWrap>(c).remove_never();
                                    first_after_break = true;
                                } else {
                                    first_after_break = false;
                                }
                            }
                        });
                        slf.write_newline::<AllowWrap>().remove_never();
                    },
                )?;
            }
            // nested sequences should have been flattened already
            LNode::Sequence(_) => unreachable!(),
            // simple nodes without a wrapping decision, just write them
            _ => {
                let check = self.checkpoint();
                self.write_node::<W>(child)?;
                self.write_sequence_impl::<W>(rest)
                    .inspect_err(|_| self.restore(check))?;
            }
        }
        Ok(())
    }

    fn write_sequence_branch<'c, 's: 'c, W: MaybeWrap>(
        &mut self,
        rest: impl Iterator<Item = &'c LNode<'s>> + Clone,
        f_single: impl FnOnce(&mut Self) -> Result<(), NeedsWrap>,
        f_wrap: impl FnOnce(&mut Self),
    ) -> Result<(), W::E> {
        // try without wrapping
        let check = self.checkpoint();
        let result_no_wrap = f_single(self);

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
        // TODO this can also be optimized more, as soon as we overflow deeper we know that we should bail
        if !should_wrap {
            self.write_sequence_impl::<W>(rest.clone())
                .inspect_err(|_| self.restore(check))?;
            should_wrap |= self.line_overflows(check);
        }

        // if we need to wrap, roll back and re-write everything with wrapping
        if should_wrap {
            self.restore(check);
            W::require_wrapping()?;

            f_wrap(self);
            self.write_sequence_impl::<AllowWrap>(rest).remove_never();
        }

        Ok(())
    }
}

fn char_is_whitespace(c: char) -> bool {
    let mut buffer = [0; 4];
    is_whitespace_or_empty(c.encode_utf8(&mut buffer))
}
