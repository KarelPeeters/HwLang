use crate::syntax::format::FormatSettings;
use crate::syntax::format::common::swrite_indent;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{char_is_whitespace, str_is_whitespace_or_empty};
use crate::util::Never;
use crate::util::data::VecExt;
use crate::util::iter::IterExt;
use hwl_util::swriteln;
use std::fmt::Debug;

pub type LNode<'s> = LNodeImpl<'s, ()>;
pub type LNodeSimple<'s> = LNodeImpl<'s, Never>;

/// Low-level formatting nodes.
/// Based on the [Prettier commands](https://github.com/prettier/prettier/blob/main/commands.md).
#[derive(Debug)]
pub enum LNodeImpl<'s, E> {
    /// Emit a space if the previous and next characters on the same line exist and are not whitespace.
    Space,

    /// Emit the given string.
    AlwaysStr(&'s str),
    /// Emit the given string if the containing group is wrapping.
    WrapStr(&'s str),

    /// Ensure there is a newline here. This forces any containing groups to wrap.
    /// Newlines are automatically deduplicated, use [LNode::AlwaysBlankLine] to force a full empty line.
    AlwaysNewline,
    /// Ensure there is a newline here, if the the containing group is wrapping.
    /// Newlines are automatically deduplicated, use [LNode::AlwaysBlankLine] to force a full empty line.
    WrapNewline,
    /// Ensure there is a full blank like here (= two newlines).
    AlwaysBlankLine,

    /// Force the enclosing group to wrap, without emitting anything itself.
    ForceWrap(E),

    /// Indent the inner node by one level.
    /// The indent only applies to new lines emitted by the inner node, the current line is not retroactively indented.
    Indent(Box<LNodeImpl<'s, E>>),

    /// Dedent the inner node all the way to indentation 0, independently of the current indentation level.
    /// Nodes that come after this node are indented normally again.
    Dedent(Box<LNodeImpl<'s, E>>),

    /// A sequence of nodes to be emitted in order.
    /// Children can each individually decide to wrap or not,
    /// based on whether their contants overflow the start or end line.
    /// Any child wrapping forces the parent groups to wrap too.
    Sequence(Vec<LNodeImpl<'s, E>>),

    /// Groups are the mechanism to control wrapping.
    /// A group either wraps or does not wrap, which recursively affects all child nodes.
    /// Groups can be nested, in which case inner groups can only wrap if the outer group is wrapping,
    ///   but inner groups are allowed to not wrap even if the outer group is wrapping.
    Group {
        force_wrap: bool,
        child: Box<LNodeImpl<'s, E>>,
    },

    // TODO doc
    EscapeGroupIfLast(E, Box<LNodeImpl<'s, E>>),
}

pub struct StringOutput {
    pub stats: StringsStats,
    pub string: String,
}

pub fn node_to_string(settings: &FormatSettings, source_str: &str, root: &LNodeSimple) -> StringOutput {
    // TODO match field order
    let mut state = NewState {
        builder: StringBuilder {
            settings,
            buffer: String::with_capacity(2 * source_str.len()),
            state: StringBuilderState {
                curr_line_start: 0,
                indent: 0,
                emit_space: false,
            },
        },
        stats: StringsStats::default(),
        queue: vec![],
        queue_fill: vec![],
        group_no_wrap_count: 0,
    };

    state.push_iter(std::iter::once(Command::Node(root)));
    new_loop(&mut state);

    StringOutput {
        stats: state.stats,
        string: state.builder.buffer,
    }
}

#[derive(Debug, Default)]
pub struct StringsStats {
    // TODO update/rename/reduce these
    pub checkpoint: usize,
    pub restore: usize,
    pub restore_chars: usize,
    pub check_overflow: usize,

    pub iter_loop: usize,
    pub iter_fits: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct Line {
    start: usize,
}

impl<E: Debug> LNodeImpl<'_, E> {
    pub fn debug_str(&self) -> String {
        let mut f = String::new();
        self.debug_str_impl(&mut f, 0);
        f
    }

    fn debug_str_impl(&self, f: &mut String, indent: usize) {
        swrite_indent(f, indent);
        match self {
            LNodeImpl::Space => swriteln!(f, "Space"),
            LNodeImpl::AlwaysStr(s) => swriteln!(f, "AlwaysStr({s:?})"),
            LNodeImpl::WrapStr(s) => swriteln!(f, "WrapStr({s:?})"),
            LNodeImpl::AlwaysNewline => swriteln!(f, "AlwaysNewline"),
            LNodeImpl::WrapNewline => swriteln!(f, "WrapNewline"),
            LNodeImpl::AlwaysBlankLine => swriteln!(f, "AlwaysBlankLine"),
            LNodeImpl::ForceWrap(e) => swriteln!(f, "ForceWrap({e:?})"),
            LNodeImpl::Indent(child) => {
                swriteln!(f, "Indent");
                child.debug_str_impl(f, indent + 1);
            }
            LNodeImpl::Dedent(child) => {
                swriteln!(f, "Dedent");
                child.debug_str_impl(f, indent + 1);
            }
            LNodeImpl::Sequence(children) => {
                swriteln!(f, "Sequence");
                for child in children {
                    child.debug_str_impl(f, indent + 1);
                }
            }
            LNodeImpl::Group { force_wrap, child } => {
                swriteln!(f, "Group(force_wrap={force_wrap})");
                child.debug_str_impl(f, indent + 1);
            }
            LNodeImpl::EscapeGroupIfLast(e, child) => {
                swriteln!(f, "EscapeGroup({e:?})");
                child.debug_str_impl(f, indent + 1);
            }
        }
    }
}

impl<E> LNodeImpl<'static, E> {
    pub const EMPTY: LNodeImpl<'static, E> = LNodeImpl::Sequence(vec![]);
}

struct SimplifyResult<'s> {
    node: LNodeSimple<'s>,
    force_wrap: bool,
}

impl<'s> LNode<'s> {
    pub fn simplify(&self) -> LNodeSimple<'s> {
        self.simplify_impl(None).node
    }

    // TODO doc
    fn simplify_impl(&self, mut escape_group: Option<&mut (Vec<LNodeSimple<'s>>, bool)>) -> SimplifyResult<'s> {
        match self {
            // flatten sequences
            LNode::Sequence(children) => {
                let mut result = Vec::with_capacity(children.len());
                let mut force_wrap = false;
                for (child, last) in children.iter().with_last() {
                    let child_escape_group = if last { escape_group.as_deref_mut() } else { None };
                    let child_simple = child.simplify_impl(child_escape_group);
                    match child_simple.node {
                        LNodeSimple::Sequence(inner) => result.extend(inner),
                        c_simple => result.push(c_simple),
                    }
                    force_wrap |= child_simple.force_wrap;
                }
                SimplifyResult {
                    node: result.single().unwrap_or_else(LNodeSimple::Sequence),
                    force_wrap,
                }
            }
            // TODO doc
            LNodeImpl::EscapeGroupIfLast((), inner) => {
                let inner = inner.simplify_impl(escape_group.as_deref_mut());
                if let Some(escape_group) = escape_group.as_deref_mut() {
                    escape_group.0.push(inner.node);
                    escape_group.1 |= inner.force_wrap;
                    SimplifyResult {
                        node: LNodeSimple::EMPTY,
                        force_wrap: false,
                    }
                } else {
                    inner
                }
            }
            // simplify children
            LNode::Indent(child) => simplify_container(child, escape_group, LNodeSimple::Indent),
            LNode::Dedent(child) => simplify_container(child, escape_group, LNodeSimple::Dedent),
            LNode::Group { force_wrap, child } => {
                let mut result = match escape_group {
                    None => {
                        let mut escape_group = (vec![], false);
                        let mut result = simplify_group(child, Some(&mut escape_group));

                        result.force_wrap |= escape_group.1;
                        if !escape_group.0.is_empty() {
                            escape_group.0.insert(0, result.node);
                            result.node = LNodeSimple::Sequence(escape_group.0);
                        }

                        result
                    }
                    Some(escape_group) => simplify_group(child, Some(escape_group)),
                };

                result.force_wrap |= *force_wrap;
                result
            }
            // trivial cases
            LNode::Space => SimplifyResult {
                node: LNodeSimple::Space,
                force_wrap: false,
            },
            LNode::AlwaysStr(s) => SimplifyResult {
                node: LNodeSimple::AlwaysStr(s),
                force_wrap: s.contains(LineOffsets::LINE_ENDING_CHARS),
            },
            LNode::WrapStr(s) => SimplifyResult {
                node: LNodeSimple::WrapStr(s),
                force_wrap: false,
            },
            LNode::AlwaysNewline => SimplifyResult {
                node: LNodeSimple::AlwaysNewline,
                force_wrap: true,
            },
            LNode::WrapNewline => SimplifyResult {
                node: LNodeSimple::WrapNewline,
                force_wrap: false,
            },
            LNode::AlwaysBlankLine => SimplifyResult {
                node: LNodeSimple::AlwaysBlankLine,
                force_wrap: true,
            },
            LNode::ForceWrap(()) => SimplifyResult {
                node: LNodeSimple::EMPTY,
                force_wrap: true,
            },
        }
    }
}

fn simplify_container<'s>(
    child: &LNode<'s>,
    group_escape_slot: Option<&mut (Vec<LNodeSimple<'s>>, bool)>,
    f: impl FnOnce(Box<LNodeSimple<'s>>) -> LNodeSimple<'s>,
) -> SimplifyResult<'s> {
    simplify_container_impl(child, group_escape_slot, |child, _| f(child))
}

fn simplify_group<'s>(
    child: &LNode<'s>,
    group_escape_slot: Option<&mut (Vec<LNodeSimple<'s>>, bool)>,
) -> SimplifyResult<'s> {
    simplify_container_impl(child, group_escape_slot, |child, force_wrap| LNodeImpl::Group {
        force_wrap,
        child,
    })
}

fn simplify_container_impl<'s>(
    child: &LNode<'s>,
    group_escape_slot: Option<&mut (Vec<LNodeSimple<'s>>, bool)>,
    f: impl FnOnce(Box<LNodeSimple<'s>>, bool) -> LNodeSimple<'s>,
) -> SimplifyResult<'s> {
    let child = child.simplify_impl(group_escape_slot);

    let node = if let LNodeImpl::Sequence(inner) = &child.node
        && inner.is_empty()
    {
        LNodeSimple::EMPTY
    } else {
        f(Box::new(child.node), child.force_wrap)
    };

    SimplifyResult {
        node,
        force_wrap: child.force_wrap,
    }
}

// TODO rename
struct NewState<'n, 's, 'f> {
    stats: StringsStats,
    queue: Vec<Command<'n, 's>>,
    queue_fill: Vec<Command<'n, 's>>,
    group_no_wrap_count: usize,
    builder: StringBuilder<'f>,
}

struct StringBuilder<'f> {
    settings: &'f FormatSettings,
    buffer: String,
    state: StringBuilderState,
}

#[derive(Debug, Copy, Clone)]
struct StringBuilderState {
    curr_line_start: usize,
    indent: usize,
    emit_space: bool,
}

#[derive(Debug)]
struct StringBuilderCheckpoint {
    result_len: usize,
    state: StringBuilderState,
}

impl StringBuilderCheckpoint {
    pub fn line(&self) -> Line {
        Line {
            start: self.state.curr_line_start,
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum MaybeNewline {
    No,
    Yes,
}

impl StringBuilder<'_> {
    pub fn checkpoint(&self) -> StringBuilderCheckpoint {
        StringBuilderCheckpoint {
            result_len: self.buffer.len(),
            state: self.state,
        }
    }

    pub fn restore(&mut self, checkpoint: StringBuilderCheckpoint) {
        assert!(self.buffer.len() >= checkpoint.result_len);
        self.buffer.truncate(checkpoint.result_len);
        self.state = checkpoint.state;
    }

    fn at_line_start(&self) -> bool {
        self.state.curr_line_start == self.buffer.len()
    }

    fn line_overflows(&mut self, line: Line) -> bool {
        let end = line.start + self.settings.max_line_length + 1;
        if end <= self.buffer.len() {
            !self.buffer[line.start..end].contains(LineOffsets::LINE_ENDING_CHARS)
        } else {
            false
        }
    }

    #[must_use]
    fn write_str(&mut self, s: &str) -> MaybeNewline {
        // strings containing newlines need the parent to wrap
        let last_line_end = s.rfind(LineOffsets::LINE_ENDING_CHARS);

        // emit indent if this is the first text on the line
        if self.at_line_start() && !str_is_whitespace_or_empty(s) {
            for _ in 0..self.state.indent {
                self.buffer.push_str(&self.settings.indent_str);
            }
        }

        // emit space if asked and both the previous and next charactor are not whitespace
        if self.state.emit_space
            && !self.at_line_start()
            && let Some(prev_char) = self.buffer.chars().last()
            && !char_is_whitespace(prev_char)
            && let Some(next_char) = s.chars().next()
            && !char_is_whitespace(next_char)
        {
            self.buffer.push(' ');
        }
        if !s.is_empty() {
            self.state.emit_space = false;
        }

        // emit str itself
        self.buffer.push_str(s);

        // update line state
        if let Some(last_line_end) = last_line_end {
            self.state.curr_line_start = self.buffer.len() - s.len() + last_line_end + 1;
        }

        if last_line_end.is_some() {
            MaybeNewline::Yes
        } else {
            MaybeNewline::No
        }
    }

    /// Ensure the buffer ends in at least `n` newlines.
    /// This is used to avoid duplicate newlines being emitted where not necessary.
    fn ensure_newlines(&mut self, n: usize) {
        // count current ending newlines
        let mut curr = self.buffer.as_str();
        let mut curr_count = 0;
        while curr_count < n
            && let Some(before) = curr.strip_suffix(&self.settings.newline_str)
        {
            curr = before;
            curr_count += 1;
        }

        // add newlines if necessary
        for _ in curr_count..n {
            let _: MaybeNewline = self.write_str(&self.settings.newline_str);
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum Command<'n, 's> {
    Node(&'n LNodeSimple<'s>),
    EndGroupNoWrap,
    RestoreIndent { indent: usize },
}

// TODO reorder functions
impl<'n, 's> NewState<'n, 's, '_> {
    fn push(&mut self, cmd: Command<'n, 's>) {
        self.push_iter(std::iter::once(cmd));
    }

    fn push_iter<I: IntoIterator<Item = Command<'n, 's>>>(&mut self, iter: I)
    where
        I::IntoIter: DoubleEndedIterator,
    {
        self.queue.extend(iter.into_iter().rev())
    }
}

// TODO if everything is a single method, we don't need this context "class" any more
// TODO document all of this a bit more
fn new_loop(state: &mut NewState) {
    loop {
        let Some(next) = state.queue.pop() else { break };
        state.stats.iter_loop += 1;

        // handle the next command
        let next = match next {
            Command::Node(next) => next,
            Command::EndGroupNoWrap => {
                assert!(state.group_no_wrap_count > 0);
                state.group_no_wrap_count -= 1;
                continue;
            }
            Command::RestoreIndent { indent } => {
                state.builder.state.indent = indent;
                continue;
            }
        };

        // handle the next node
        // TODO fix code duplication with fits
        let can_wrap = state.group_no_wrap_count == 0;
        match next {
            LNodeSimple::Space => {
                state.builder.state.emit_space = true;
            }
            LNodeSimple::AlwaysStr(s) => match state.builder.write_str(s) {
                MaybeNewline::No => {}
                MaybeNewline::Yes => {
                    assert!(can_wrap);
                }
            },
            LNodeSimple::WrapStr(s) => {
                if can_wrap {
                    let _: MaybeNewline = state.builder.write_str(s);
                }
            }
            LNodeSimple::AlwaysNewline => {
                assert!(can_wrap);
                state.builder.ensure_newlines(1);
            }
            LNodeSimple::WrapNewline => {
                if can_wrap {
                    state.builder.ensure_newlines(1);
                }
            }
            LNodeSimple::AlwaysBlankLine => {
                assert!(can_wrap);
                state.builder.ensure_newlines(2);
            }
            LNodeSimple::Indent(child) => {
                let indent = state.builder.state.indent;
                state.builder.state.indent += 1;
                state.push_iter([Command::Node(child), Command::RestoreIndent { indent }]);
            }
            LNodeSimple::Dedent(child) => {
                let indent = state.builder.state.indent;
                state.builder.state.indent = 0;
                state.push_iter([Command::Node(child), Command::RestoreIndent { indent }]);
            }
            LNodeSimple::Sequence(seq) => {
                // TODO we flatten here, so maybe we can remove simplify
                state.push_iter(seq.iter().map(Command::Node));
            }
            &LNodeSimple::Group { force_wrap, ref child } => {
                let group_wrap = force_wrap || {
                    state.group_no_wrap_count += 1;
                    state.push_iter([Command::Node(child), Command::EndGroupNoWrap]);

                    let group_fits = fits(
                        &mut state.builder,
                        state.group_no_wrap_count,
                        &state.queue,
                        &mut state.queue_fill,
                    );

                    state.group_no_wrap_count -= 1;
                    state.queue.pop().unwrap();
                    state.queue.pop().unwrap();

                    !group_fits
                };

                if group_wrap {
                    state.push(Command::Node(child));
                } else {
                    state.group_no_wrap_count += 1;
                    state.push_iter([Command::Node(child), Command::EndGroupNoWrap]);
                }
            }

            LNodeSimple::ForceWrap(never) | LNodeSimple::EscapeGroupIfLast(never, _) => never.unreachable(),
        }
    }
}

// TODO turn asserts into errors?
// TODO re-use fits-queue vec to avoid redundant allocations
// TODO reduce code duplication with the main loop
fn fits<'n, 's>(
    builder: &mut StringBuilder,
    mut no_wrap_count: usize,
    mut queue_outer: &[Command<'n, 's>],
    queue: &mut Vec<Command<'n, 's>>,
) -> bool {
    let check = builder.checkpoint();
    queue.clear();

    loop {
        // stop once we've overflown the line
        if builder.line_overflows(check.line()) {
            break;
        }

        // find the next command
        // (to avoid fully copying the outer queue, we instead use it as a separate second level of the stack)
        let cmd = match queue.pop() {
            Some(cmd) => cmd,
            None => match queue_outer.split_last() {
                Some((&cmd, rest)) => {
                    queue_outer = rest;
                    cmd
                }
                None => break,
            },
        };

        // process the next command
        let node = match cmd {
            Command::Node(node) => node,
            Command::EndGroupNoWrap => {
                no_wrap_count -= 1;
                continue;
            }
            Command::RestoreIndent { indent } => {
                builder.state.indent = indent;
                continue;
            }
        };

        // process the next node
        // we encounter a newline, we can stop since future commands can't influence the current line anymore
        let can_wrap = no_wrap_count == 0;
        match node {
            LNodeSimple::Space => {
                builder.state.emit_space = true;
            }
            LNodeSimple::AlwaysStr(s) => match builder.write_str(s) {
                MaybeNewline::No => {}
                MaybeNewline::Yes => {
                    assert!(can_wrap);
                    break;
                }
            },
            LNodeSimple::WrapStr(s) => {
                if can_wrap {
                    match builder.write_str(s) {
                        MaybeNewline::No => {}
                        MaybeNewline::Yes => {
                            assert!(can_wrap);
                            break;
                        }
                    }
                }
            }
            LNodeSimple::AlwaysNewline => {
                assert!(can_wrap);
                break;
            }
            LNodeSimple::WrapNewline => {
                if can_wrap {
                    break;
                }
            }
            LNodeSimple::AlwaysBlankLine => {
                assert!(can_wrap);
                break;
            }
            LNodeSimple::Indent(child) => {
                let indent = builder.state.indent;
                builder.state.indent += 1;

                queue.push(Command::RestoreIndent { indent });
                queue.push(Command::Node(child));
            }
            LNodeSimple::Dedent(child) => {
                let indent = builder.state.indent;
                builder.state.indent = 0;

                queue.push(Command::RestoreIndent { indent });
                queue.push(Command::Node(child));
            }
            LNodeSimple::Sequence(children) => {
                queue.extend(children.iter().rev().map(Command::Node));
            }
            &LNodeSimple::Group { force_wrap, ref child } => {
                if force_wrap {
                    assert!(can_wrap);
                }

                // assume wrapping if allowed by the parent, this just requires _not_ incrementing no_wrap_count
                queue.push(Command::Node(child));
            }
            LNodeSimple::ForceWrap(never) | LNodeSimple::EscapeGroupIfLast(never, _) => never.unreachable(),
        };
    }

    queue.clear();

    // check for final overflow and restore the builder
    let fits = !builder.line_overflows(check.line());
    builder.restore(check);
    fits
}
