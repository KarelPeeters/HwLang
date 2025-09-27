use crate::syntax::format::FormatSettings;
use crate::syntax::format::common::swrite_indent;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::{char_is_whitespace, str_is_whitespace_or_empty};
use crate::util::Never;
use crate::util::data::VecExt;
use crate::util::iter::IterExt;
use hwl_util::swriteln;
use itertools::enumerate;
use std::fmt::Debug;
use std::time::{Duration, Instant};

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
        settings,
        result: String::with_capacity(2 * source_str.len()),
        state: StringState {
            curr_line_start: 0,
            indent: 0,
            emit_space: false,
        },
        stats: StringsStats::default(),
        queue: vec![],
        queue_next: 0,
        group_no_wrap_stack: vec![],
        group_no_wrap_active: vec![],
    };

    state.push_iter(std::iter::once(CommandKind::Node(root)));
    new_loop(&mut state);

    StringOutput {
        stats: state.stats,
        string: state.result,
    }
}

#[derive(Debug, Default)]
pub struct StringsStats {
    // TODO update/rename/reduce these
    pub checkpoint: usize,
    pub restore: usize,
    pub restore_chars: usize,
    pub check_overflow: usize,

    pub loop_iterations: usize,
    pub loop_commands: usize,
    pub loop_nodes: usize,

    pub retain_calls: usize,
    pub retain_count: usize,

    pub time_retain: Duration,
    pub time_active: Duration,
    pub time_active2: Duration,
    pub time_line: Duration,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct Line {
    start: usize,
}

#[derive(Debug, Copy, Clone)]
struct StringState {
    curr_line_start: usize,
    indent: usize,
    emit_space: bool,
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

    // TODO instead of creating a messy tree and then simplifying it,
    //   avoid creating it in the first place by adding some convenient sequence builder
    // TODO try moving newlines around here
    // TODO doc what this actually does
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

// TODO rename (and merge?)
struct NewState<'n, 's, 'f> {
    settings: &'f FormatSettings,
    stats: StringsStats,

    queue: Vec<Command<'n, 's>>,
    queue_next: usize,

    // TODO maybe active groups can be a separate VecDeque or maybe even linked list?
    group_no_wrap_stack: Vec<CheckGroupNoWrap<'n, 's>>,
    group_no_wrap_active: Vec<usize>,

    state: StringState,
    result: String,
}

#[derive(Debug, Copy, Clone)]
struct Command<'n, 's> {
    stack_group_no_wrap_len: usize,
    kind: CommandKind<'n, 's>,
}

#[derive(Debug, Copy, Clone)]
enum CommandKind<'n, 's> {
    Node(&'n LNodeSimple<'s>),
    EndGroupNoWrap,
    RestoreIndent { indent: usize },
}

// TODO document all of this
#[derive(Debug)]
struct CheckGroupNoWrap<'n, 's> {
    result_len: usize,
    string_state: StringState,

    queue_next: usize,
    queue_len: usize,

    first_active: usize,

    group_inner: &'n LNodeSimple<'s>,
}

impl<'n, 's> CheckGroupNoWrap<'n, 's> {
    pub fn line(&self) -> Line {
        Line {
            start: self.string_state.curr_line_start,
        }
    }
}

struct CausedWrap;

// TODO reorder functions
impl<'n, 's> NewState<'n, 's, '_> {
    fn push(&mut self, cmd: CommandKind<'n, 's>) {
        self.push_iter(std::iter::once(cmd));
    }

    fn push_iter(&mut self, iter: impl IntoIterator<Item = CommandKind<'n, 's>>) {
        let stack_group_no_wrap_len = self.group_no_wrap_stack.len();
        self.queue.insert_iter(
            self.queue_next,
            iter.into_iter().map(|cmd| Command {
                stack_group_no_wrap_len,
                kind: cmd,
            }),
        );
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
        self.group_no_wrap_active.is_empty()
    }

    fn force_wrap_now(&mut self) -> Result<(), CausedWrap> {
        if let Some(&first) = self.group_no_wrap_active.first() {
            self.restore_and_wrap(first);
            Err(CausedWrap)
        } else {
            Ok(())
        }
    }

    fn try_group_without_wrap(&mut self, inner: &'n LNodeSimple<'s>) {
        let group_index = self.group_no_wrap_stack.len();
        let first_active = self.group_no_wrap_active.first().copied().unwrap_or(group_index);

        let check = CheckGroupNoWrap {
            // TODO match field order
            result_len: self.result.len(),
            string_state: self.state,
            queue_next: self.queue_next,
            queue_len: self.queue.len(),
            // queue_after: self.queue.len() - self.queue_next,
            // queue: self.queue.clone(),
            first_active,
            group_inner: inner,
        };

        // println!("{:?}", check);

        self.group_no_wrap_stack.push(check);
        self.group_no_wrap_active.push(group_index);

        self.push_iter([CommandKind::Node(inner), CommandKind::EndGroupNoWrap]);

        self.stats.checkpoint += 1;
    }

    // TODO make this more type-safe, without bare usize index
    // TODO create variant of this that automatically finds the containing group
    // TODO make some utility that automatically wraps and _currently active_ groups instead of taking in an index
    fn restore_and_wrap(&mut self, group_index: usize) {
        // get first active group, this is the outermost group which (also) needs to wrap
        let first_active = self.group_no_wrap_stack[group_index].first_active;

        // println!("event force_wrap {group_index}, {first_active}");

        // println!("{:?}", self.stack_group_no_wrap[first_active]);

        let CheckGroupNoWrap {
            result_len,
            string_state: state,
            queue_next,
            queue_len,
            first_active,
            group_inner,
        } = self.group_no_wrap_stack.swap_remove(first_active);

        self.stats.restore += 1;

        // restore stack
        self.group_no_wrap_stack.truncate(first_active);
        self.group_no_wrap_active.clear();

        // restore string
        self.stats.restore_chars += self.result.len() - result_len;
        self.result.truncate(result_len);
        self.state = state;

        // restore queue
        // TODO retore the queue to _after_ the insert, saving some duplicate effort here (benchmark)
        // TODO also remember how many items were at the end of the queue, we know we don't have to discard those
        //   but then the mem move will still cause O(n**2) issues, no? what we really need is a fancier vec index pattern that allows more linear clearing
        self.queue_next = queue_next;
        let start = Instant::now();
        // print!("retain pattern: ");
        self.queue.retain_range(queue_next..self.queue.len(), |cmd| {
            self.stats.retain_count += 1;
            let r = cmd.stack_group_no_wrap_len <= first_active;
            // print!("{}", if r { 'r' } else { 'd' });
            r
        });
        // println!();
        self.stats.retain_calls += 1;
        self.stats.time_retain += start.elapsed();
        assert_eq!(self.queue.len(), queue_len);

        self.push(CommandKind::Node(group_inner));
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

        state.stats.loop_iterations += 1;

        // check for line overflow
        let curr_line = state.line();
        if state.line_overflows(curr_line) {
            let info = enumerate(&state.group_no_wrap_stack)
                .rev()
                .take_while(|(_, info)| info.line() >= curr_line)
                .find(|(_, info)| info.line() == curr_line);
            if let Some((info_pos, _)) = info {
                state.restore_and_wrap(info_pos);
                continue;
            }
        }

        // pop the next command
        let Some(next) = state.queue.get(state.queue_next) else {
            break;
        };
        let next = &next.kind;
        state.queue_next += 1;

        state.stats.loop_commands += 1;

        let next = match *next {
            CommandKind::Node(next) => next,
            CommandKind::EndGroupNoWrap => {
                state.group_no_wrap_active.pop().unwrap();
                continue;
            }
            CommandKind::RestoreIndent { indent } => {
                state.state.indent = indent;
                continue;
            }
        };

        // handle the next node
        state.stats.loop_nodes += 1;

        // TODO extract function?
        // TODO discard this whole branching idea again and do something more similar to prettier?
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
            LNodeSimple::Indent(child) => {
                let indent = state.state.indent;
                state.state.indent += 1;
                state.push_iter([CommandKind::Node(child), CommandKind::RestoreIndent { indent }]);
                Ok(())
            }
            LNodeSimple::Dedent(child) => {
                let indent = state.state.indent;
                state.state.indent = 0;
                state.push_iter([CommandKind::Node(child), CommandKind::RestoreIndent { indent }]);
                Ok(())
            }
            LNodeSimple::Sequence(seq) => {
                // TODO we flatten here, so maybe we can remove simplify
                state.push_iter(seq.iter().map(CommandKind::Node));
                Ok(())
            }
            &LNodeSimple::Group { force_wrap, ref child } => {
                // TODO maybe we can implement group escaping here, removing the "simplify" step entirely
                state.try_group_without_wrap(child);
                if force_wrap { state.force_wrap_now() } else { Ok(()) }
            }
            LNodeSimple::EscapeGroupIfLast(never, _) => never.unreachable(),
        };

        // both cases are correctly handled by continuing the loop:
        // * If Ok, we succeeded in writing the node and we can move on to the next one.
        // * If Err, a wrap happened which restored an earlier state, and the next iteration will re-try with wrapping.
        let _: Result<(), CausedWrap> = res;
    }
}
