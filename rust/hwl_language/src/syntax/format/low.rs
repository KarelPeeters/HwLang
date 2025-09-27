use crate::syntax::format::FormatSettings;
use crate::syntax::format::common::swrite_indent;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{char_is_whitespace, str_is_whitespace_or_empty};
use crate::util::Never;
use crate::util::data::VecExt;
use crate::util::iter::IterExt;
use hwl_util::swriteln;
use itertools::{Itertools, enumerate};
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
    ForceWrap,

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
    Group(Box<LNodeImpl<'s, E>>),

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
        settings,
        result: String::with_capacity(2 * source_str.len()),
        state: StringState {
            curr_line_start: 0,
            indent: 0,
            emit_space: false,
        },
        stats: StringsStats {
            checkpoint: 0,
            restore: 0,
            restore_chars: 0,
            check_overflow: 0,
        },
        queue: vec![],
        queue_next: 0,
        stack_group_no_wrap: vec![],
    };

    state.push_iter(std::iter::once(Command::Node(root)));
    new_loop(&mut state);

    StringOutput {
        stats: state.stats,
        string: state.result,
    }
}

#[derive(Debug)]
pub struct StringsStats {
    // TODO update these
    pub checkpoint: usize,
    pub restore: usize,
    pub restore_chars: usize,
    pub check_overflow: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Line {
    start: usize,
}

#[derive(Debug, Copy, Clone)]
struct StringState {
    curr_line_start: usize,
    indent: usize,
    emit_space: bool,
}

#[derive(Debug)]
struct NeedsWrap;

trait WrapMaybe {
    type E;
    fn is_wrapping() -> bool;
    fn require_wrapping() -> Result<(), Self::E>;
}

struct WrapNo {}
impl WrapMaybe for WrapNo {
    type E = NeedsWrap;
    fn is_wrapping() -> bool {
        false
    }
    fn require_wrapping() -> Result<(), NeedsWrap> {
        Err(NeedsWrap)
    }
}
struct WrapYes {}
impl WrapMaybe for WrapYes {
    type E = Never;
    fn is_wrapping() -> bool {
        true
    }
    fn require_wrapping() -> Result<(), Never> {
        Ok(())
    }
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
            LNodeImpl::ForceWrap => swriteln!(f, "ForceWrap"),
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
            LNodeImpl::Group(child) => {
                swriteln!(f, "Group");
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

impl<'s> LNode<'s> {
    // TODO rename, this is no longer only simplification
    pub fn simplify(&self) -> LNodeSimple<'s> {
        self.simplify_impl(None)
    }

    // TODO instead of creating a messy tree and then simplifying it,
    //   avoid creating it in the first place by adding some convenient sequence builder
    // TODO try moving newlines around here
    fn simplify_impl(&self, mut escape_group: Option<&mut Vec<LNodeSimple<'s>>>) -> LNodeSimple<'s> {
        match self {
            // flatten sequences
            LNode::Sequence(children) => {
                let mut result = Vec::with_capacity(children.len());
                for (child, last) in children.iter().with_last() {
                    let child_escape_group = if last { escape_group.as_deref_mut() } else { None };
                    match child.simplify_impl(child_escape_group) {
                        LNodeSimple::Sequence(inner) => result.extend(inner),
                        c_simple => result.push(c_simple),
                    }
                }
                result.single().unwrap_or_else(LNodeSimple::Sequence)
            }
            // TODO doc
            LNodeImpl::EscapeGroupIfLast((), inner) => {
                let inner = inner.simplify_impl(escape_group.as_deref_mut());
                if let Some(escape_group) = escape_group.as_deref_mut() {
                    escape_group.push(inner);
                    LNodeSimple::EMPTY
                } else {
                    inner
                }
            }
            // simplify children
            LNode::Indent(child) => simplify_container(child, escape_group, LNodeSimple::Indent),
            LNode::Dedent(child) => simplify_container(child, escape_group, LNodeSimple::Dedent),
            LNode::Group(child) => {
                let mut seq = vec![];

                let escape_group = match escape_group.as_deref_mut() {
                    Some(escape_group) => escape_group,
                    None => &mut seq,
                };
                let result = simplify_container(child, Some(escape_group), LNodeSimple::Group);

                if seq.is_empty() {
                    result
                } else {
                    seq.insert(0, result);
                    LNodeSimple::Sequence(seq)
                }
            }
            // trivial cases
            LNode::Space => LNodeSimple::Space,
            LNode::AlwaysStr(s) => LNodeSimple::AlwaysStr(s),
            LNode::WrapStr(s) => LNodeSimple::WrapStr(s),
            LNode::AlwaysNewline => LNodeSimple::AlwaysNewline,
            LNode::WrapNewline => LNodeSimple::WrapNewline,
            LNode::AlwaysBlankLine => LNodeSimple::AlwaysBlankLine,
            LNode::ForceWrap => LNodeSimple::ForceWrap,
        }
    }
}

fn simplify_container<'s>(
    child: &LNode<'s>,
    group_escape_slot: Option<&mut Vec<LNodeSimple<'s>>>,
    f: impl FnOnce(Box<LNodeSimple<'s>>) -> LNodeSimple<'s>,
) -> LNodeSimple<'s> {
    let child = child.simplify_impl(group_escape_slot);
    if let LNodeImpl::Sequence(inner) = &child
        && inner.is_empty()
    {
        LNodeSimple::EMPTY
    } else {
        f(Box::new(child))
    }
}

// TODO rename (and merge?)
struct NewState<'n, 's, 'f> {
    settings: &'f FormatSettings,
    stats: StringsStats,

    queue: Vec<Command<'n, 's>>,
    queue_next: usize,
    stack_group_no_wrap: Vec<CheckGroupNoWrap<'n, 's>>,

    state: StringState,
    result: String,
}

#[derive(Debug, Copy, Clone)]
enum Command<'n, 's> {
    Node(&'n LNodeSimple<'s>),
    EndGroupNoWrap { group_index: usize },
    RestoreIndent { indent: usize },
}

#[derive(Debug)]
struct CheckGroupNoWrap<'n, 's> {
    result_len: usize,
    state: StringState,

    // TODO this fundamentally doesn't work if the next group is expanded before the current one is
    //   maybe instead of this we can add some sort of depth (no) or time (maybe?) value and filter based on those
    queue_next: usize,
    // queue_after: usize,

    // TODO avoid storing this entirely, find some clever delta-encoding idea maybe with age or depth prefixes
    //   or with a BTree
    queue: Vec<Command<'n, 's>>,

    first_active: usize,

    group_inner: &'n LNodeSimple<'s>,

    // TODO remove from this struct, this should only be present in the stack?
    active: bool,
}

impl<'n, 's> CheckGroupNoWrap<'n, 's> {
    pub fn line(&self) -> Line {
        Line {
            start: self.state.curr_line_start,
        }
    }
}

struct CausedWrap;

// TODO reorder functions
impl<'n, 's> NewState<'n, 's, '_> {
    fn push(&mut self, cmd: Command<'n, 's>) {
        self.queue.insert(self.queue_next, cmd);
    }

    fn push_iter(&mut self, iter: impl IntoIterator<Item = Command<'n, 's>>) {
        self.queue.insert_iter(self.queue_next, iter);
    }

    fn at_line_start(&self) -> bool {
        self.state.curr_line_start == self.result.len()
    }

    fn line(&self) -> Line {
        Line {
            start: self.state.curr_line_start,
        }
    }

    /// Check if the line at which the checkpoint was taken overflows the max line length.
    fn line_overflows(&mut self, line: Line) -> bool {
        self.stats.check_overflow += 1;

        // TODO only overflow if the length of the line larger than the indent, otherwise wrapping will never help
        //   (that's only true if the result of wrapping wil cause extra indents, is that always true?)
        let rest = &self.result[line.start..];
        let line_len = rest.find(LineOffsets::LINE_ENDING_CHARS).unwrap_or(rest.len());
        line_len > self.settings.max_line_length
    }

    fn write_str(&mut self, s: &str) -> Result<(), CausedWrap> {
        // strings containing newlines need the parent to wrap
        let last_line_end = s.rfind(LineOffsets::LINE_ENDING_CHARS);
        if last_line_end.is_some() {
            self.force_wrap_now()?;
        }

        // emit indent if this is the first text on the line
        // TODO instead of actually emitting the indent, we can also add yet another symbolic layer:
        //   first abstractly push indent/str nodes, then later actually convert to a full string
        //   this should speed up the somewhat bruteforce backtracking we do now, especially when indents are deep
        if self.at_line_start() && !str_is_whitespace_or_empty(s) {
            for _ in 0..self.state.indent {
                self.result.push_str(&self.settings.indent_str);
            }
        }

        // emit space if asked and both the previous and next charactor are not whitespace
        if self.state.emit_space
            && !self.at_line_start()
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

    /// Ensure the buffer ends in at least `n` newlines.
    /// This is used to avoid duplicate newlines being emitted where not necessary.
    fn ensure_newlines(&mut self, n: usize) -> Result<(), CausedWrap> {
        // TODO should we always force wrap, even if we don't emit any new newlines here?

        // count current ending newlines
        let mut curr_count = 0;
        let mut curr = self.result.as_str();
        while curr_count < n
            && let Some(before) = curr.strip_suffix(&self.settings.newline_str)
        {
            curr_count += 1;
            curr = before;
        }

        // add newlines if necessary
        for _ in curr_count..n {
            self.write_str(&self.settings.newline_str)?;
        }

        Ok(())
    }

    fn can_wrap(&self) -> bool {
        // TODO update this incrementally
        !self.stack_group_no_wrap.iter().any(|info| info.active)
    }

    fn force_wrap_now(&mut self) -> Result<(), CausedWrap> {
        let first_active_wrap = self.stack_group_no_wrap.iter().position(|info| info.active);
        if let Some(pos) = first_active_wrap {
            self.restore_and_wrap(pos);
            Err(CausedWrap)
        } else {
            Ok(())
        }
    }

    fn try_group_without_wrap(&mut self, inner: &'n LNodeSimple<'s>) {
        let group_index = self.stack_group_no_wrap.len();
        // println!("event try_no_wrap {group_index}");

        let first_active = self
            .stack_group_no_wrap
            .iter()
            .position(|info| info.active)
            .unwrap_or(group_index);

        let check = CheckGroupNoWrap {
            // TODO match field order
            result_len: self.result.len(),
            state: self.state,
            queue_next: self.queue_next,
            // queue_after: self.queue.len() - self.queue_next,
            queue: self.queue.clone(),
            first_active,
            group_inner: inner,
            active: true,
        };

        // println!("{:?}", check);

        self.stack_group_no_wrap.push(check);
        self.push_iter([Command::Node(inner), Command::EndGroupNoWrap { group_index }]);

        self.stats.checkpoint += 1;
    }

    // TODO make this more type-safe, without bare usize index
    // TODO create variant of this that automatically finds the containing group
    // TODO make some utility that automatically wraps and _currently active_ groups instead of taking in an index
    fn restore_and_wrap(&mut self, group_index: usize) {
        // get first active group, this is the outermost group which (also) needs to wrap
        let first_active = self.stack_group_no_wrap[group_index].first_active;

        // println!("event force_wrap {group_index}, {first_active}");

        // println!("{:?}", self.stack_group_no_wrap[first_active]);

        let CheckGroupNoWrap {
            result_len,
            state,
            queue,
            queue_next,
            first_active,
            group_inner,
            active: _,
        } = self.stack_group_no_wrap.swap_remove(first_active);

        self.stats.restore += 1;

        // restore stack
        self.stack_group_no_wrap.truncate(first_active);
        // TODO disable in non-debug mode?
        for i in 0..first_active {
            assert!(!self.stack_group_no_wrap[i].active);
        }

        // restore string
        self.stats.restore_chars += self.result.len() - result_len;
        self.result.truncate(result_len);
        self.state = state;

        // restore queue
        // TODO retore the queue to _after_ the insert, saving some duplicate effort here (benchmark)
        self.queue_next = queue_next;
        // drop(self.queue.drain(queue_next..(self.queue.len() - queue_after)));
        self.queue = queue;
        self.push(Command::Node(group_inner));
    }
}

// TODO if everything is a single method, we don't need this context "class" any more
// TODO document all of this a bit more
fn new_loop(state: &mut NewState) {
    loop {
        // println!("iter");
        // println!("  queue:");
        // for (i, item) in enumerate(&state.queue) {
        //     print!("   {i:4}");
        //
        //     if i < state.queue_next {
        //         print!(" [X] ");
        //     } else {
        //         print!(" [ ] ");
        //     }
        //     println!("{item:?}");
        // }
        // println!("  output\n    {:?}", state.str_ctx.result);

        // check for line overflow
        let curr_line = state.line();
        if state.line_overflows(curr_line) {
            let info = enumerate(&state.stack_group_no_wrap)
                .rev()
                .find(|(_, info)| info.line() == curr_line);
            if let Some((info_pos, _)) = info {
                state.restore_and_wrap(info_pos);
                // TODO do we need this continue?
                continue;
            }
        }

        // pop the next command
        let Some(next) = state.queue.get(state.queue_next) else {
            break;
        };
        state.queue_next += 1;

        let next = match *next {
            Command::Node(next) => next,
            Command::EndGroupNoWrap { group_index } => {
                // TODO discard items on the stack that are no longer active and end on a previous line
                let info = &mut state.stack_group_no_wrap[group_index];
                assert!(info.active);
                info.active = false;
                continue;
            }
            Command::RestoreIndent { indent } => {
                state.state.indent = indent;
                continue;
            }
        };

        // handle the next node
        // TODO extract function?
        let res = match next {
            LNodeSimple::Space => {
                state.state.emit_space = true;
                Ok(())
            }
            LNodeSimple::AlwaysStr(s) => state.write_str(s),
            LNodeSimple::WrapStr(s) => {
                if state.can_wrap() {
                    state.write_str(s)
                } else {
                    Ok(())
                }
            }
            LNodeSimple::AlwaysNewline => state.ensure_newlines(1),
            LNodeSimple::WrapNewline => {
                if state.can_wrap() {
                    state.ensure_newlines(1)
                } else {
                    Ok(())
                }
            }
            LNodeSimple::AlwaysBlankLine => state.ensure_newlines(2),
            LNodeSimple::ForceWrap => state.force_wrap_now(),
            LNodeSimple::Indent(inner) => {
                let indent = state.state.indent;
                state.state.indent += 1;
                state.push_iter([Command::Node(inner), Command::RestoreIndent { indent }]);
                Ok(())
            }
            LNodeSimple::Dedent(inner) => {
                let indent = state.state.indent;
                state.state.indent = 0;
                state.push_iter([Command::Node(inner), Command::RestoreIndent { indent }]);
                Ok(())
            }
            LNodeSimple::Sequence(seq) => {
                // TODO we flatten here, so maybe we can remove simplify
                state.push_iter(seq.iter().map(Command::Node));
                Ok(())
            }
            LNodeSimple::Group(inner) => {
                // TODO maybe we can implement group escaping here, removing the "simplify" step entirely
                state.try_group_without_wrap(inner);
                Ok(())
            }
            LNodeSimple::EscapeGroupIfLast(never, _) => never.unreachable(),
        };

        // both cases are correctly handled by continuing the loop:
        // * If Ok, we succeeded in writing the node and we can move on to the next one.
        // * If Err, a wrap happened which restored an earlier state, and the next iteration will re-try with wrapping.
        let _: Result<(), CausedWrap> = res;
    }
}
