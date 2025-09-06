use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::common::swrite_indent;
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::is_whitespace_or_empty;
use crate::util::data::VecExt;
use crate::util::{Never, ResultNeverExt};
use hwl_util::swriteln;
use itertools::Either;

/// Low-level formatting nodes.
/// Based on the [Prettier commands](https://github.com/prettier/prettier/blob/main/commands.md).
pub enum LNode<'s> {
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

    /// Indent the inner node.
    Indent(Box<LNode<'s>>),
    /// Dedent the inner node all the way to indentation 0, independently of the current indentation level.
    /// Nodes that come after this node are indented normally again.
    Dedent(Box<LNode<'s>>),

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
    /// There's an implicit [LNode::WrapNewLine] after each child
    /// and an implicit [LNode::Indent] around all children.
    Fill(Vec<LNode<'s>>),
}

pub struct StringOutput {
    pub stats: StringsStats,
    pub string: String,
}

pub fn node_to_string(settings: &FormatSettings, source_str: &str, root: &LNode) -> StringOutput {
    let mut ctx = StringBuilderContext {
        settings,
        result: String::with_capacity(source_str.len() * 2),
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
    };
    ctx.write_node::<WrapYes>(root).remove_never();

    StringOutput {
        stats: ctx.stats,
        string: ctx.result,
    }
}

struct StringBuilderContext<'a> {
    settings: &'a FormatSettings,

    result: String,
    state: StringState,
    stats: StringsStats,
}

#[derive(Debug)]
pub struct StringsStats {
    pub checkpoint: usize,
    pub restore: usize,
    pub restore_chars: usize,
    pub check_overflow: usize,
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

#[derive(Debug)]
struct ShouldWrapEarlier;

trait EarlierBranchMaybe: Copy {
    type E;
    fn check_overflow(&self, ctx: &mut StringBuilderContext) -> Result<(), Self::E>;
}

#[derive(Copy, Clone)]
struct EarlierBranchNo;
impl EarlierBranchMaybe for EarlierBranchNo {
    type E = Never;
    fn check_overflow(&self, _: &mut StringBuilderContext) -> Result<(), Never> {
        Ok(())
    }
}

#[derive(Copy, Clone)]
struct EarlierBranchYes(CheckPoint);
impl EarlierBranchMaybe for EarlierBranchYes {
    type E = ShouldWrapEarlier;
    fn check_overflow(&self, ctx: &mut StringBuilderContext) -> Result<(), ShouldWrapEarlier> {
        if ctx.line_overflows(self.0) {
            Err(ShouldWrapEarlier)
        } else {
            Ok(())
        }
    }
}

impl<'s> LNode<'s> {
    pub const EMPTY: LNode<'static> = LNode::Sequence(vec![]);

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
            LNode::AlwaysBlankLine => swriteln!(f, "AlwaysBlankLine"),
            LNode::ForceWrap => swriteln!(f, "ForceWrap"),
            LNode::Indent(child) => {
                swriteln!(f, "Indent");
                child.debug_str_impl(f, indent + 1);
            }
            LNode::Dedent(child) => {
                swriteln!(f, "Dedent");
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

    // TODO instead of creating a messy tree and then simplifying it,
    //   avoid creating it in the first place by adding some convenient sequence builder
    pub fn simplify(self) -> LNode<'s> {
        fn simplify_container<'s>(f: impl FnOnce(Box<LNode<'s>>) -> LNode<'s>, mut child: Box<LNode<'s>>) -> LNode<'s> {
            *child = child.simplify();
            if let LNode::Sequence(inner) = &*child
                && inner.is_empty()
            {
                return LNode::EMPTY;
            }
            f(child)
        }

        match self {
            // flatten sequences
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
            // simplify children
            LNode::Indent(child) => simplify_container(LNode::Indent, child),
            LNode::Dedent(child) => simplify_container(LNode::Dedent, child),
            LNode::Group(child) => simplify_container(LNode::Group, child),
            LNode::Fill(children) => {
                if children.is_empty() {
                    LNode::EMPTY
                } else {
                    LNode::Fill(children.into_iter().map(LNode::simplify).collect())
                }
            }
            // trivial cases
            LNode::Space => LNode::Space,
            LNode::AlwaysStr(s) => LNode::AlwaysStr(s),
            LNode::WrapStr(s) => LNode::WrapStr(s),
            LNode::AlwaysNewline => LNode::AlwaysNewline,
            LNode::WrapNewline => LNode::WrapNewline,
            LNode::AlwaysBlankLine => LNode::AlwaysBlankLine,
            LNode::ForceWrap => LNode::ForceWrap,
        }
    }
}

impl StringBuilderContext<'_> {
    /// Save the current state into a checkpoint that can later be used to roll back to the current state.
    fn checkpoint(&mut self) -> CheckPoint {
        self.stats.checkpoint += 1;
        CheckPoint {
            result_len: self.result.len(),
            state: self.state,
        }
    }

    /// Roll back the state to a previous checkpoint.
    fn restore(&mut self, check: CheckPoint) {
        assert!(self.result.len() >= check.result_len);

        self.stats.restore += 1;
        self.stats.restore_chars += self.result.len() - check.result_len;

        self.result.truncate(check.result_len);
        self.state = check.state;
    }

    /// Check if the line at which the checkpoint was taken overflows the max line length.
    fn line_overflows(&mut self, check: CheckPoint) -> bool {
        self.stats.check_overflow += 1;

        // TODO only overflow if the length of the line larger than the indent, otherwise wrapping will never help
        //   (that's only true if the result of wrapping wil cause extra indents, is that always true?)
        let rest = &self.result[check.state.curr_line_start..];
        let line_len = rest.find(LineOffsets::LINE_ENDING_CHARS).unwrap_or(rest.len());
        line_len > self.settings.max_line_length
    }

    fn indent<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.state.indent += 1;
        let r = f(self);
        self.state.indent -= 1;
        r
    }

    fn dedent<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev_indent = self.state.indent;
        self.state.indent = 0;
        let r = f(self);
        self.state.indent = prev_indent;
        r
    }

    fn write_newline<W: WrapMaybe>(&mut self) -> Result<(), W::E> {
        self.write_str::<W>(&self.settings.newline_str)
    }

    fn write_str<W: WrapMaybe>(&mut self, s: &str) -> Result<(), W::E> {
        // strings containing newlines need the parent to wrap
        let last_line_end = s.rfind(LineOffsets::LINE_ENDING_CHARS);
        if last_line_end.is_some() {
            W::require_wrapping()?;
        }

        // emit indent if this is the first text on the line
        // TODO instead of actually emitting the indent, we can also add yet another symbolic layer:
        //   first abstractly push indent/str nodes, then later actually convert to a full string
        //   this should speed up the somewhat bruteforce backtracking we do now, especially when indents are deep
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

    fn write_node<W: WrapMaybe>(&mut self, node: &LNode) -> Result<(), W::E> {
        match node {
            LNode::Space => {
                self.state.emit_space = true;
            }
            &LNode::AlwaysStr(s) => {
                self.write_str::<W>(s)?;
            }
            &LNode::WrapStr(s) => {
                if W::is_wrapping() {
                    self.write_str::<W>(s)?;
                }
            }
            LNode::AlwaysNewline => {
                W::require_wrapping()?;

                // only add a newline if there isn't one already to avoid duplicates
                if !self.result.ends_with(&self.settings.newline_str) {
                    self.write_newline::<W>()?;
                }
            }
            LNode::WrapNewline => {
                // only add a newline if wrapping and there isn't one already to avoid duplicates
                if W::is_wrapping() && !self.result.ends_with(&self.settings.newline_str) {
                    self.write_newline::<W>()?;
                }
            }
            LNode::AlwaysBlankLine => {
                W::require_wrapping()?;

                // count ending newlines
                let mut left: &str = &self.result;
                let mut newline_count = 0;
                while let Some(before) = left.strip_suffix(&self.settings.newline_str) {
                    newline_count += 1;
                    left = before;
                }

                // ensure at least two newlines
                for _ in newline_count..2 {
                    self.write_newline::<W>()?;
                }
            }
            LNode::ForceWrap => {
                W::require_wrapping()?;
            }
            LNode::Indent(child) => {
                self.indent(|ctx| ctx.write_node::<W>(child))?;
            }
            LNode::Dedent(child) => {
                self.dedent(|ctx| ctx.write_node::<W>(child))?;
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

    fn write_sequence<'s, W: WrapMaybe>(&mut self, children: &[LNode<'s>]) -> Result<(), W::E> {
        // flatten if necessary
        if children.iter().all(|c| !matches!(c, LNode::Sequence(_))) {
            self.write_sequence_flat::<W, _>(EarlierBranchNo, children.iter())
                .map_err(remove_never_right)
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
            self.write_sequence_flat::<W, _>(EarlierBranchNo, flat.iter().copied())
                .map_err(remove_never_right)
        }
    }

    fn write_sequence_flat<'c, 's: 'c, W: WrapMaybe, B: EarlierBranchMaybe>(
        &mut self,
        last_branch: B,
        mut children: impl Iterator<Item = &'c LNode<'s>> + Clone,
    ) -> Result<(), Either<W::E, B::E>> {
        let (child, rest) = match children.next() {
            None => return Ok(()),
            Some(child) => (child, children),
        };

        // wrapping logic is here instead of inside the group nodes,
        //   so we can take items that follow the group on the same line into account to check for line overflow
        match child {
            LNode::Group(child) => {
                self.write_sequence_branch::<W, B>(
                    last_branch,
                    rest,
                    |slf| slf.write_node::<WrapNo>(child),
                    |slf| slf.write_node::<WrapYes>(child).remove_never(),
                )?;
            }
            LNode::Fill(children) => {
                self.write_sequence_branch::<W, B>(
                    last_branch,
                    rest,
                    |slf| {
                        for c in children {
                            slf.write_node::<WrapNo>(c)?;
                        }
                        Ok(())
                    },
                    |slf| {
                        slf.indent(|slf| {
                            slf.write_newline::<WrapYes>().remove_never();
                            let mut first_after_break = true;
                            for c in children {
                                let check = slf.checkpoint();
                                slf.write_node::<WrapYes>(c).remove_never();
                                if slf.line_overflows(check) && !first_after_break {
                                    slf.restore(check);
                                    slf.write_newline::<WrapYes>().remove_never();
                                    slf.write_node::<WrapYes>(c).remove_never();
                                    first_after_break = true;
                                } else {
                                    first_after_break = false;
                                }
                            }
                        });
                        slf.write_newline::<WrapYes>().remove_never();
                    },
                )?;
            }
            // nested sequences should have been flattened already
            LNode::Sequence(_) => unreachable!(),
            // simple nodes without a wrapping decision, just write them
            _ => {
                let check = self.checkpoint();
                self.write_node::<W>(child).map_err(Either::Left)?;

                // if there was an earlier branching point and we've now overflowed it, bail immediately
                //   this avoids terrible exponential backtracking behavior
                last_branch.check_overflow(self).map_err(Either::Right)?;

                self.write_sequence_flat::<W, _>(last_branch, rest)
                    .inspect_err(|_| self.restore(check))?;
            }
        }
        Ok(())
    }

    fn write_sequence_branch<'c, 's: 'c, W: WrapMaybe, B: EarlierBranchMaybe>(
        &mut self,
        last_branch: B,
        rest: impl Iterator<Item = &'c LNode<'s>> + Clone,
        f_single: impl FnOnce(&mut Self) -> Result<(), NeedsWrap>,
        f_wrap: impl FnOnce(&mut Self),
    ) -> Result<(), Either<W::E, B::E>> {
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
        if !should_wrap {
            let result_rest = self.write_sequence_flat::<W, _>(EarlierBranchYes(check), rest.clone());

            match result_rest {
                Ok(()) => {}
                Err(Either::Left(e)) => {
                    self.restore(check);
                    return Err(Either::Left(e));
                }
                Err(Either::Right(ShouldWrapEarlier)) => {
                    should_wrap = true;
                }
            }

            should_wrap |= self.line_overflows(check);
        }

        // if we need to wrap, roll back and re-write everything with wrapping
        if should_wrap {
            self.restore(check);
            W::require_wrapping().map_err(Either::Left)?;

            f_wrap(self);
            self.write_sequence_flat::<WrapYes, _>(last_branch, rest)
                .map_err(|e| Either::Right(remove_never_left(e)))?;
        }

        Ok(())
    }
}

fn char_is_whitespace(c: char) -> bool {
    let mut buffer = [0; 4];
    is_whitespace_or_empty(c.encode_utf8(&mut buffer))
}

fn remove_never_left<T>(r: Either<Never, T>) -> T {
    match r {
        Either::Left(n) => n.unreachable(),
        Either::Right(t) => t,
    }
}
fn remove_never_right<T>(r: Either<T, Never>) -> T {
    match r {
        Either::Left(t) => t,
        Either::Right(n) => n.unreachable(),
    }
}
