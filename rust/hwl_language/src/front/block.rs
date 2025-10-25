use crate::front::assignment::store_ir_expression_in_new_variable;
use crate::front::check::{
    TypeContainsReason, check_type_contains_compile_value, check_type_contains_value, check_type_is_bool,
    check_type_is_bool_compile,
};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::exit::{ExitStack, LoopEntry, ReturnEntryKind};
use crate::front::flow::{Flow, FlowHardware, VariableId};
use crate::front::flow::{FlowKind, VariableInfo};
use crate::front::function::check_function_return_type_and_set_value;
use crate::front::implication::HardwareValueWithImplications;
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::{HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::{
    IrBlock, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIfStatement, IrIntCompareOp, IrStatement,
};
use crate::syntax::ast::{
    Block, BlockStatement, BlockStatementKind, ConstBlock, ExtraItem, ExtraList, ForStatement, Identifier,
    IfCondBlockPair, IfStatement, MatchBranch, MatchPattern, MatchStatement, MaybeIdentifier, ReturnStatement,
    VariableDeclaration, WhileStatement,
};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::throw;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::iter::IterExt;
use crate::util::{ResultExt, result_pair};
use annotate_snippets::Level;
use itertools::{Itertools, enumerate, zip_eq};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use unwrap_match::unwrap_match;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[must_use]
pub enum BlockEnd {
    /// No early exit, proceed normally.
    Normal,
    /// Definite early exit, known at compile time.
    CompileExit(EarlyExitKind),
    /// Definite early exit, but the kind depends on hardware conditions.
    /// The mask indicates which kinds of exits are possible.
    HardwareExit(ExitMask<2>),
    /// Maybe early exit, but the kind and whether there is actually an exit depends on hardware conditions.
    /// The mask indicates which kinds of exits are possible.
    HardwareMaybeExit(ExitMask<1>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum EarlyExitKind {
    Return,
    Break,
    Continue,
}

/// A set of possible exit kinds.
/// The const generic [N] indicates the minimum number of exit kinds that are set.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExitMask<const N: u8> {
    can_return: bool,
    can_break: bool,
    can_continue: bool,
}

impl BlockEnd {
    /// Construct the correct [BlockEnd] from `certain_exit` and `mask`.
    ///
    /// * `certain_exit` is whether we know for sure there has been an early exit.
    /// * `mask` is the set of possible early exists that can have happened.
    fn new(certain_exit: bool, mask: ExitMask<0>) -> BlockEnd {
        #[allow(clippy::collapsible_else_if)]
        if certain_exit {
            if let Some(mask) = ExitMask::<2>::from_base(mask) {
                BlockEnd::HardwareExit(mask)
            } else {
                if mask.can_return {
                    BlockEnd::CompileExit(EarlyExitKind::Return)
                } else if mask.can_break {
                    BlockEnd::CompileExit(EarlyExitKind::Break)
                } else if mask.can_continue {
                    BlockEnd::CompileExit(EarlyExitKind::Continue)
                } else {
                    BlockEnd::Normal
                }
            }
        } else {
            if let Some(mask) = ExitMask::<1>::from_base(mask) {
                BlockEnd::HardwareMaybeExit(mask)
            } else {
                BlockEnd::Normal
            }
        }
    }

    pub fn is_certain_exit(self) -> bool {
        match self {
            BlockEnd::Normal | BlockEnd::HardwareMaybeExit(_) => false,
            BlockEnd::CompileExit(_) | BlockEnd::HardwareExit(_) => true,
        }
    }

    pub fn possible_exit_mask(self) -> ExitMask<0> {
        match self {
            BlockEnd::Normal => ExitMask::default(),
            BlockEnd::CompileExit(kind) => ExitMask::from_kind(kind).into_base(),
            BlockEnd::HardwareExit(mask) => mask.into_base(),
            BlockEnd::HardwareMaybeExit(mask) => mask.into_base(),
        }
    }

    pub fn unwrap_normal(self, diags: &Diagnostics, span: Span) -> DiagResult {
        match self {
            BlockEnd::Normal => Ok(()),
            BlockEnd::CompileExit(_) | BlockEnd::HardwareExit(_) | BlockEnd::HardwareMaybeExit(_) => {
                Err(diags.report_internal_error(span, "unexpected early exit"))
            }
        }
    }

    pub fn remove(self, mask_remove: ExitMask<0>) -> BlockEnd {
        let mask_self = self.possible_exit_mask();
        let mask_result = ExitMask {
            can_return: mask_self.can_return & !mask_remove.can_return,
            can_break: mask_self.can_break & !mask_remove.can_break,
            can_continue: mask_self.can_continue & !mask_remove.can_continue,
        };
        Self::new(self.is_certain_exit(), mask_result)
    }
}

impl<const N: u8> ExitMask<N> {
    pub fn into_base(self) -> ExitMask<0> {
        ExitMask {
            can_return: self.can_return,
            can_break: self.can_break,
            can_continue: self.can_continue,
        }
    }

    pub fn from_base(mask: ExitMask<0>) -> Option<Self> {
        let count = mask.can_return as u8 + mask.can_break as u8 + mask.can_continue as u8;
        if count >= N {
            Some(ExitMask {
                can_return: mask.can_return,
                can_break: mask.can_break,
                can_continue: mask.can_continue,
            })
        } else {
            None
        }
    }

    pub fn absorb_loop(self) -> ExitMask<0> {
        ExitMask {
            can_return: self.can_return,
            can_break: false,
            can_continue: false,
        }
    }
}

impl<const N: u8> std::ops::BitOr for ExitMask<N> {
    type Output = ExitMask<N>;
    fn bitor(self, rhs: Self) -> Self::Output {
        ExitMask {
            can_return: self.can_return || rhs.can_return,
            can_break: self.can_break || rhs.can_break,
            can_continue: self.can_continue || rhs.can_continue,
        }
    }
}

impl Default for ExitMask<0> {
    fn default() -> Self {
        ExitMask {
            can_return: false,
            can_break: false,
            can_continue: false,
        }
    }
}

impl ExitMask<1> {
    pub fn from_kind(kind: EarlyExitKind) -> Self {
        let mut res = ExitMask {
            can_return: false,
            can_break: false,
            can_continue: false,
        };
        match kind {
            EarlyExitKind::Return => res.can_return = true,
            EarlyExitKind::Break => res.can_break = true,
            EarlyExitKind::Continue => res.can_continue = true,
        }
        res
    }
}

enum BranchMatched {
    Yes(Option<(MaybeIdentifier, CompileValue)>),
    No,
}

impl BranchMatched {
    fn from_bool(b: bool) -> Self {
        if b { BranchMatched::Yes(None) } else { BranchMatched::No }
    }
}

enum PatternEqual {
    Bool(bool),
    Int(BigInt),
    String(Arc<String>),
}

type CheckedMatchPattern<'a> = MatchPattern<PatternEqual, IncRange<BigInt>, usize, Identifier>;

impl CompileItemContext<'_, '_> {
    pub fn elaborate_const_block(&mut self, scope: &Scope, flow: &mut impl Flow, block: &ConstBlock) -> DiagResult {
        let diags = self.refs.diags;
        let &ConstBlock {
            span_keyword,
            ref block,
        } = block;

        let span = span_keyword.join(block.span);
        let mut flow_inner = flow.new_child_compile(span, "const block");

        let mut stack = ExitStack::new_root();
        let block_end = self.elaborate_block(scope, &mut flow_inner, &mut stack, block)?;
        block_end.unwrap_normal(diags, block.span)?;

        Ok(())
    }

    pub fn elaborate_block(
        &mut self,
        scope_parent: &Scope,
        flow_parent: &mut impl Flow,
        stack: &mut ExitStack,
        block: &Block<BlockStatement>,
    ) -> DiagResult<BlockEnd> {
        let &Block { span, ref statements } = block;

        let mut scope = Scope::new_child(span, scope_parent);

        // this is actually static dispatch on the flow type, but we can't easily express that
        match flow_parent.kind_mut() {
            FlowKind::Compile(flow_parent) => {
                let mut flow = flow_parent.new_child_scoped();
                self.elaborate_block_statements(&mut scope, &mut flow, stack, statements)
            }
            FlowKind::Hardware(flow_parent) => {
                let mut flow = flow_parent.new_child_scoped();
                self.elaborate_block_statements(&mut scope, &mut flow, stack, statements)
            }
        }
    }

    pub fn elaborate_block_statements(
        &mut self,
        scope: &mut Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        statements: &[BlockStatement],
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let span = if statements.is_empty() {
            return Ok(BlockEnd::Normal);
        } else {
            statements
                .first()
                .unwrap()
                .span()
                .join(statements.last().unwrap().span())
        };

        let end = match stack.early_exit_condition(diags, &mut self.large, flow, span)? {
            Value::Compile(exit_cond) => {
                if exit_cond {
                    return Err(diags
                        .report_internal_error(span, "compile-time early exit condition should be handled elsewhere"));
                }
                self.elaborate_block_statements_without_immediate_exit_check(scope, flow, stack, statements)?
            }
            Value::Hardware(exit_cond) => {
                let flow = flow.check_hardware(span, "hardware exit conditions")?;

                let exit_cond = Spanned::new(span, exit_cond);
                let (_, run_end) =
                    self.elaborate_hardware_branch(flow, span, exit_cond, |slf, branch_flow, branch_cond| {
                        if branch_cond {
                            // early exit, do nothing
                            Ok(BlockEnd::Normal)
                        } else {
                            // not exiting, actually run the statements
                            slf.elaborate_block_statements_without_immediate_exit_check(
                                scope,
                                branch_flow,
                                stack,
                                statements,
                            )
                        }
                    })?;

                // we don't need to join ends here: if we got here,
                //   this means that the run branch will be taken and it's only that end case that counts
                run_end
            }
        };
        Ok(end)
    }

    pub fn elaborate_block_statements_without_immediate_exit_check(
        &mut self,
        scope: &mut Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        statements: &[BlockStatement],
    ) -> DiagResult<BlockEnd> {
        let mut end_joined = BlockEnd::Normal;

        for (stmt_index, stmt) in enumerate(statements) {
            let end_curr = self.elaborate_statement(scope, flow, stack, stmt)?;
            end_joined = join_block_ends_sequence(end_joined, end_curr);

            if end_joined.is_certain_exit() {
                break;
            }

            let recheck_exit = match end_curr {
                BlockEnd::Normal | BlockEnd::CompileExit(_) | BlockEnd::HardwareExit(_) => false,
                BlockEnd::HardwareMaybeExit(_) => true,
            };
            if recheck_exit {
                let statements_rest = &statements[stmt_index + 1..];
                let end_rest = self.elaborate_block_statements(scope, flow, stack, statements_rest)?;
                return Ok(join_block_ends_sequence(end_joined, end_rest));
            } else {
                continue;
            }
        }

        Ok(end_joined)
    }

    fn elaborate_statement(
        &mut self,
        scope: &mut Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt: &BlockStatement,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let stmt_span = stmt.span;

        let end = match &stmt.inner {
            BlockStatementKind::CommonDeclaration(decl) => {
                self.eval_and_declare_declaration(scope, flow, decl);
                BlockEnd::Normal
            }
            BlockStatementKind::VariableDeclaration(decl) => {
                let &VariableDeclaration {
                    span: _,
                    mutable,
                    id,
                    ty,
                    init,
                } = decl;

                // eval ty
                let ty = ty.map(|ty| self.eval_expression_as_ty(scope, flow, ty)).transpose();

                // eval init
                let init = ty.as_ref_ok().and_then(|ty| {
                    let init_expected_ty = ty.as_ref().map_or(&Type::Any, |ty| &ty.inner);
                    init.map(|init| self.eval_expression(scope, flow, init_expected_ty, init))
                        .transpose()
                });

                let entry = result_pair(ty, init).and_then(|(ty, init)| {
                    // check that init fits in type
                    if let Some(ty) = &ty
                        && let Some(init) = &init
                    {
                        let reason = TypeContainsReason::Assignment {
                            span_target: id.span(),
                            span_target_ty: ty.span,
                        };
                        check_type_contains_value(diags, reason, &ty.inner, init.as_ref(), true, true)?;
                    }

                    // build variable
                    let info = VariableInfo {
                        span_decl: id.span(),
                        id: VariableId::Id(id),
                        mutable,
                        ty,
                        use_ir_variable: None,
                    };
                    let var = flow.var_new(info);

                    // store initial value if there is one
                    //   for hardware values, also store them in an IR variable to avoid duplicate expressions
                    //   and to keep the generated RTL somewhat similar to the source
                    if let Some(init) = init {
                        let init = init.inner.try_map_hardware(|init_inner| {
                            let flow = flow.check_hardware(init.span, "hardware value")?;
                            let debug_info_id = id.spanned_string(self.refs.fixed.source).inner;
                            store_ir_expression_in_new_variable(self.refs, flow, id.span(), debug_info_id, init_inner)
                                .map(HardwareValue::to_general_expression)
                        })?;
                        flow.var_set(var, decl.span, Ok(init));
                    }

                    Ok(ScopedEntry::Named(NamedValue::Variable(var)))
                });

                let id = Ok(id.spanned_str(self.refs.fixed.source));
                scope.maybe_declare(diags, id, entry);
                BlockEnd::Normal
            }
            BlockStatementKind::Assignment(stmt) => {
                self.elaborate_assignment(scope, flow, stmt)?;
                BlockEnd::Normal
            }
            &BlockStatementKind::Expression(expr) => {
                let _: Spanned<Value> = self.eval_expression(scope, flow, &Type::Any, expr)?;
                BlockEnd::Normal
            }
            BlockStatementKind::Block(inner_block) => self.elaborate_block(scope, flow, stack, inner_block)?,
            BlockStatementKind::If(stmt) => {
                let IfStatement {
                    span: _,
                    initial_if,
                    else_ifs,
                    final_else,
                } = stmt;

                self.elaborate_if_statement(scope, flow, stack, Some((initial_if, else_ifs)), final_else)?
            }
            BlockStatementKind::Match(stmt) => {
                let stmt = Spanned::new(stmt_span, stmt);
                self.elaborate_match_statement(scope, flow, stack, stmt)?
            }
            BlockStatementKind::While(stmt) => {
                let stmt = Spanned::new(stmt_span, stmt);
                self.elaborate_while_statement(scope, flow, stack, stmt)?
            }
            BlockStatementKind::For(stmt) => {
                let stmt = Spanned::new(stmt_span, stmt);
                self.elaborate_for_statement(scope, flow, stack, stmt)?
            }
            BlockStatementKind::Return(stmt) => {
                let &ReturnStatement { span_return, ref value } = stmt;

                // set flag
                let entry = stack.return_info(diags, span_return)?;
                if let ReturnEntryKind::Hardware(entry) = &mut entry.kind {
                    entry.return_flag.set(flow, span_return);
                }

                // check type and store value
                let type_unit = Type::unit();
                let expected_ty = entry.return_type.map_or(&type_unit, |ty| ty.inner);
                let value = value
                    .map(|value| self.eval_expression(scope, flow, expected_ty, value))
                    .transpose()?;

                check_function_return_type_and_set_value(diags, flow, entry, stmt_span, span_return, value)?;

                BlockEnd::CompileExit(EarlyExitKind::Return)
            }
            &BlockStatementKind::Break { span } => {
                let entry = stack.innermost_loop(diags, span, "break")?;
                if let LoopEntry::Hardware(entry) = entry {
                    entry.break_flag.set(flow, span);
                }
                BlockEnd::CompileExit(EarlyExitKind::Break)
            }
            &BlockStatementKind::Continue { span } => {
                let entry = stack.innermost_loop(diags, span, "continue")?;
                if let LoopEntry::Hardware(entry) = entry {
                    entry.continue_flag.set(flow, span);
                }
                BlockEnd::CompileExit(EarlyExitKind::Continue)
            }
        };
        Ok(end)
    }

    // TODO make this non-recursive for very deep if chains?
    fn elaborate_if_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        ifs: Option<(
            &IfCondBlockPair<Block<BlockStatement>>,
            &[IfCondBlockPair<Block<BlockStatement>>],
        )>,
        final_else: &Option<Block<BlockStatement>>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;

        let (initial_if, remaining_ifs) = match ifs {
            Some(p) => p,
            None => {
                return match final_else {
                    None => Ok(BlockEnd::Normal),
                    Some(final_else) => {
                        return self.elaborate_block(scope, flow, stack, final_else);
                    }
                };
            }
        };

        let &IfCondBlockPair {
            span: _,
            span_if,
            cond,
            ref block,
        } = initial_if;

        let cond = self.eval_expression_with_implications(scope, flow, &Type::Bool, cond)?;

        let reason = TypeContainsReason::IfCondition(span_if);
        let cond = check_type_is_bool(diags, reason, cond)?;

        match cond.inner {
            // evaluate the if at compile-time
            Value::Compile(cond_eval) => {
                // only visit the selected branch
                if cond_eval {
                    self.elaborate_block(scope, flow, stack, block)
                } else {
                    self.elaborate_if_statement(scope, flow, stack, remaining_ifs.split_first(), final_else)
                }
            }
            // evaluate the if in hardware, generating IR
            Value::Hardware(cond_value) => {
                let flow = flow.check_hardware(cond.span, "hardware value")?;

                let cond_value = Spanned::new(cond.span, cond_value);
                let (then_end, else_end) =
                    self.elaborate_hardware_branch(flow, span_if, cond_value, |slf, branch_flow, branch_cond| {
                        if branch_cond {
                            slf.elaborate_block(scope, branch_flow, stack, block)
                        } else {
                            slf.elaborate_if_statement(
                                scope,
                                branch_flow,
                                stack,
                                remaining_ifs.split_first(),
                                final_else,
                            )
                        }
                    })?;

                // join ends
                let end = join_block_ends_branches(&[then_end, else_end]);
                Ok(end)
            }
        }
    }

    fn elaborate_match_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt: Spanned<&MatchStatement<Block<BlockStatement>>>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let &MatchStatement {
            target,
            span_branches,
            ref branches,
        } = stmt.inner;

        // eval target
        let target = self.eval_expression(scope, flow, &Type::Any, target)?;
        let target_ty = target.inner.ty();

        // track pattern coverage
        // TODO handle coverage checking of empty enums properly
        // TODO don't check coverage for compile-time cases, it's weird and not that useful
        let mut cover_all = false;
        let mut cover_bool_false = false;
        let mut cover_bool_true = false;
        let mut cover_enum_variant: HashMap<usize, Span> = HashMap::new();

        // some type-specific handling
        let cover_enum_count = if let &Type::Enum(elab) = &target_ty {
            let info = self.refs.shared.elaboration_arenas.enum_info(elab);
            Some(info.variants.len())
        } else {
            None
        };
        let eq_expected_ty = if matches!(&target_ty, Type::Int(_)) && matches!(&target.inner, Value::Compile(_)) {
            &Type::Int(IncRange::OPEN)
        } else {
            &target_ty
        };

        // eval all branch patterns before visiting any bodies, to check for coverage and to get nice error messages
        let reason = "match pattern";
        let branch_patterns = branches
            .iter()
            .map(|branch| -> DiagResult<CheckedMatchPattern> {
                if cover_all {
                    // TODO turn into warning
                    let diag = Diagnostic::new("redundant match branch")
                        .add_error(branch.pattern.span, "this branch is unreachable")
                        .finish();
                    diags.report(diag);
                }

                match &branch.pattern.inner {
                    MatchPattern::Wildcard => {
                        cover_all = true;
                        Ok(MatchPattern::Wildcard)
                    }
                    &MatchPattern::Val(i) => {
                        cover_all = true;
                        Ok(MatchPattern::Val(i))
                    }
                    &MatchPattern::Equal(value) => {
                        // TODO support tuples, arrays, structs, enums (by value), all recursively
                        let value = self.eval_expression_as_compile(scope, flow, &target_ty, value, reason)?;
                        check_type_contains_compile_value(
                            diags,
                            TypeContainsReason::MatchPattern(target.span),
                            eq_expected_ty,
                            value.as_ref(),
                            false,
                        )?;

                        let pattern = match value.inner {
                            CompileValue::Bool(value) => {
                                cover_bool_true |= value;
                                cover_bool_false |= !value;
                                cover_all |= cover_bool_true && cover_bool_false;
                                PatternEqual::Bool(value)
                            }
                            CompileValue::Int(value) => {
                                // TODO track covered int ranges
                                PatternEqual::Int(value)
                            }
                            CompileValue::String(value) => PatternEqual::String(value),
                            _ => {
                                return Err(diags.report_simple(
                                    "unsupported match type",
                                    value.span,
                                    format!("pattern has type `{}`", value.inner.ty().diagnostic_string()),
                                ));
                            }
                        };

                        Ok(MatchPattern::Equal(pattern))
                    }
                    &MatchPattern::In(value) => {
                        if !matches!(target_ty, Type::Int(_)) {
                            return Err(diags.report_simple(
                                "range patterns are only supported for int values",
                                value.span,
                                format!("value has type `{}`", target_ty.diagnostic_string()),
                            ));
                        }

                        let value = self.eval_expression_as_compile(scope, flow, &target_ty, value, reason)?;
                        let value = match value.inner {
                            CompileValue::IntRange(range) => range,
                            _ => {
                                return Err(diags.report_simple(
                                    "expected range for in pattern",
                                    value.span,
                                    format!("pattern has type `{}`", value.inner.ty().diagnostic_string()),
                                ));
                            }
                        };

                        Ok(MatchPattern::In(value))
                    }
                    &MatchPattern::EnumVariant(variant, id_content) => {
                        let elab = match target_ty {
                            Type::Enum(elab) => elab,
                            _ => {
                                return Err(diags.report_simple(
                                    "expected enum type for enum variant pattern",
                                    variant.span,
                                    format!("value has type `{}`", target_ty.diagnostic_string()),
                                ));
                            }
                        };
                        let info = self.refs.shared.elaboration_arenas.enum_info(elab);

                        let variant_str = variant.str(self.refs.fixed.source);
                        let variant_index = info.find_variant(diags, Spanned::new(variant.span, variant_str))?;

                        // check reachable
                        match cover_enum_variant.entry(variant_index) {
                            Entry::Occupied(entry) => {
                                let prev = *entry.get();
                                let diag = Diagnostic::new("redundant match branch")
                                    .add_error(branch.pattern.span, "this branch is unreachable")
                                    .add_info(prev, "this enum variant was already handled here")
                                    .finish();
                                return Err(diags.report(diag));
                            }
                            Entry::Vacant(entry) => {
                                entry.insert(branch.pattern.span);

                                if cover_enum_count == Some(cover_enum_variant.len()) {
                                    cover_all = true;
                                }
                            }
                        }

                        // check content
                        let (variant_decl, variant_content) = &info.variants[variant_index];
                        match (variant_content, id_content) {
                            (Some(_), Some(_)) | (None, None) => {}
                            (Some(variant_content), None) => {
                                let diag = Diagnostic::new("mismatch between enum and match content")
                                    .add_info(variant_content.span, "enum variant declared with content here")
                                    .add_error(branch.pattern.span, "match pattern without content here")
                                    .footer(Level::Help, "use (_) to ignore the content")
                                    .finish();
                                return Err(diags.report(diag));
                            }
                            (None, Some(id_content)) => {
                                let diag = Diagnostic::new("mismatch between enum and match content")
                                    .add_info(variant_decl.span, "enum variant declared without content here")
                                    .add_error(id_content.span(), "match pattern with content here")
                                    .finish();
                                return Err(diags.report(diag));
                            }
                        }

                        Ok(MatchPattern::EnumVariant(variant_index, id_content))
                    }
                }
            })
            .try_collect_all_vec()?;

        // check that all cases have been handled
        if !cover_all {
            let msg;
            let msg = match target_ty {
                Type::Bool => match (cover_bool_false, cover_bool_true) {
                    (false, false) => "values not covered: false, true",
                    (false, true) => "value not covered: false",
                    (true, false) => "value not covered: true",
                    _ => unreachable!(),
                },
                Type::Enum(elab) => {
                    let info = self.refs.shared.elaboration_arenas.enum_info(elab);

                    let mut not_covered = vec![];
                    for (i, (id, _)) in info.variants.iter().enumerate() {
                        if !cover_enum_variant.contains_key(&i) {
                            not_covered.push(id);
                        }
                    }
                    let prefix = if not_covered.len() > 1 { "variant" } else { "variants" };
                    msg = format!("{prefix} not covered: {}", not_covered.iter().join(","));
                    &msg
                }
                _ => "not all values are covered",
            };

            let diag = Diagnostic::new("match does not cover all values")
                .add_error(span_branches, msg)
                .add_info(
                    target.span,
                    format!("value has type `{}`", target_ty.diagnostic_string()),
                )
                .footer(Level::Help, "add missing cases, or")
                .footer(
                    Level::Help,
                    "add a default case using `_` to cover all remaining values",
                )
                .finish();
            return Err(diags.report(diag));
        }

        // evaluate match itself
        match target.inner {
            Value::Compile(target_inner) => self.elaborate_match_statement_compile(
                scope,
                flow,
                stack,
                stmt.span,
                target_inner,
                branches,
                branch_patterns,
            ),
            Value::Hardware(target_inner) => {
                let flow = flow.check_hardware(target.span, "hardware value")?;
                self.elaborate_match_statement_hardware(
                    scope,
                    flow,
                    stack,
                    stmt.span,
                    Spanned::new(target.span, target_inner),
                    branches,
                    branch_patterns,
                )
            }
        }
    }

    fn elaborate_match_statement_compile(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt_span: Span,
        target: CompileValue,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<CheckedMatchPattern>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;

        for (branch, pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch {
                pattern: pattern_raw,
                block,
            } = branch;
            let pattern_span = pattern_raw.span;

            let matched: BranchMatched = match pattern {
                MatchPattern::Wildcard => BranchMatched::Yes(None),
                MatchPattern::Val(id) => BranchMatched::Yes(Some((MaybeIdentifier::Identifier(id), target.clone()))),
                MatchPattern::Equal(pattern) => {
                    let c = match (&pattern, &target) {
                        (PatternEqual::Bool(p), CompileValue::Bool(v)) => p == v,
                        (PatternEqual::Int(p), CompileValue::Int(v)) => p == v,
                        (PatternEqual::String(p), CompileValue::String(v)) => p == v,
                        _ => return Err(diags.report_internal_error(pattern_span, "unexpected pattern/value")),
                    };

                    BranchMatched::from_bool(c)
                }
                MatchPattern::In(pattern) => match &target {
                    CompileValue::Int(value) => BranchMatched::from_bool(pattern.contains(value)),
                    _ => return Err(diags.report_internal_error(pattern_span, "unexpected range/value")),
                },
                MatchPattern::EnumVariant(pattern_index, id_content) => match &target {
                    CompileValue::Enum(_, (value_index, value_content)) => {
                        if pattern_index == *value_index {
                            let declare_content = match (id_content, value_content) {
                                (Some(id_content), Some(value_content)) => {
                                    Some((id_content, (**value_content).clone()))
                                }
                                (None, None) => None,
                                _ => unreachable!(),
                            };
                            BranchMatched::Yes(declare_content)
                        } else {
                            BranchMatched::No
                        }
                    }
                    _ => return Err(diags.report_internal_error(pattern_span, "unexpected enum/value")),
                },
            };

            match matched {
                BranchMatched::No => continue,
                BranchMatched::Yes(declare) => {
                    let mut scope_inner = Scope::new_child(pattern_span.join(block.span), scope);

                    let scoped_used = if let Some((declare_id, declare_value)) = declare {
                        let var = flow.var_new_immutable_init(
                            declare_id.span(),
                            VariableId::Id(declare_id),
                            pattern_span,
                            Ok(Value::Compile(declare_value)),
                        );
                        scope_inner.maybe_declare(
                            diags,
                            Ok(declare_id.spanned_str(self.refs.fixed.source)),
                            Ok(ScopedEntry::Named(NamedValue::Variable(var))),
                        );
                        &scope_inner
                    } else {
                        scope
                    };

                    return self.elaborate_block(scoped_used, flow, stack, block);
                }
            }
        }

        // we should never get here, we already checked that all cases are handled
        Err(diags.report_internal_error(stmt_span, "reached end of match statement"))
    }

    // TODO write some tests for this
    fn elaborate_match_statement_hardware(
        &mut self,
        scope_parent: &Scope,
        flow: &mut FlowHardware,
        stack: &mut ExitStack,
        stmt_span: Span,
        target: Spanned<HardwareValue>,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<CheckedMatchPattern>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;

        let mut if_branch_conditions = vec![];
        let mut if_branch_flows = vec![];
        let mut if_branch_ends = vec![];

        for (branch, pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch {
                pattern: pattern_raw,
                block,
            } = branch;
            let pattern_span = pattern_raw.span;
            let large = &mut self.large;

            // evaluate the pattern as a boolean condition and collect the variable to declare if any
            let (cond, declare): (Option<IrExpression>, Option<(MaybeIdentifier, HardwareValue)>) = match pattern {
                MatchPattern::Wildcard => (None, None),
                MatchPattern::Val(id) => (None, Some((MaybeIdentifier::Identifier(id), target.inner.clone()))),
                MatchPattern::Equal(pattern) => {
                    let cond = match (&target.inner.ty, pattern) {
                        (HardwareType::Bool, PatternEqual::Bool(pattern)) => {
                            let cond_expr = target.inner.expr.clone();
                            if pattern {
                                cond_expr
                            } else {
                                large.push_expr(IrExpressionLarge::BoolNot(cond_expr))
                            }
                        }
                        (HardwareType::Int(_), PatternEqual::Int(pattern)) => {
                            // TODO expand int range?
                            large.push_expr(IrExpressionLarge::IntCompare(
                                IrIntCompareOp::Eq,
                                target.inner.expr.clone(),
                                IrExpression::Int(pattern),
                            ))
                        }
                        _ => return Err(diags.report_internal_error(pattern_span, "unexpected hw pattern/value")),
                    };

                    (Some(cond), None)
                }
                MatchPattern::In(range) => {
                    let IncRange { start_inc, end_inc } = range;

                    let start_inc = start_inc.map(|start_inc| {
                        large.push_expr(IrExpressionLarge::IntCompare(
                            IrIntCompareOp::Lte,
                            IrExpression::Int(start_inc),
                            target.inner.expr.clone(),
                        ))
                    });
                    let end_inc = end_inc.map(|end_inc| {
                        large.push_expr(IrExpressionLarge::IntCompare(
                            IrIntCompareOp::Lte,
                            target.inner.expr.clone(),
                            IrExpression::Int(end_inc),
                        ))
                    });

                    let cond = match (start_inc, end_inc) {
                        (None, None) => None,
                        (Some(single), None) => Some(single),
                        (None, Some(single)) => Some(single),
                        (Some(start), Some(end)) => {
                            let cond = large.push_expr(IrExpressionLarge::BoolBinary(IrBoolBinaryOp::And, start, end));
                            Some(cond)
                        }
                    };

                    (cond, None)
                }
                MatchPattern::EnumVariant(pattern_index, id_content) => {
                    let target_ty = match &target.inner.ty {
                        HardwareType::Enum(cond_ty) => cond_ty,
                        _ => return Err(diags.report_internal_error(pattern_span, "unexpected hw enum/value")),
                    };

                    let info = self.refs.shared.elaboration_arenas.enum_info(target_ty.inner());
                    let info_hw = info.hw.as_ref().unwrap();

                    let ty_content = &info_hw.content_types[pattern_index];

                    let target_tag = large.push_expr(IrExpressionLarge::TupleIndex {
                        base: target.inner.expr.clone(),
                        index: BigUint::ZERO,
                    });

                    let cond = large.push_expr(IrExpressionLarge::IntCompare(
                        IrIntCompareOp::Eq,
                        target_tag,
                        IrExpression::Int(BigInt::from(pattern_index)),
                    ));
                    let declare = match (ty_content, id_content) {
                        (Some(ty_content), Some(id_content)) => {
                            let target_content_all_bits = large.push_expr(IrExpressionLarge::TupleIndex {
                                base: target.inner.expr.clone(),
                                index: BigUint::ONE,
                            });
                            let target_content_bits = large.push_expr(IrExpressionLarge::ArraySlice {
                                base: target_content_all_bits,
                                start: IrExpression::Int(BigInt::ZERO),
                                len: ty_content.size_bits(self.refs),
                            });
                            let target_content = large.push_expr(IrExpressionLarge::FromBits(
                                ty_content.as_ir(self.refs),
                                target_content_bits,
                            ));

                            let declare_value = HardwareValue {
                                ty: ty_content.clone(),
                                domain: target.inner.domain,
                                expr: target_content,
                            };
                            Some((id_content, declare_value))
                        }
                        (None, None) => None,
                        _ => unreachable!(),
                    };

                    (Some(cond), declare)
                }
            };

            // create child flow and scope
            // TODO push implications for integer ranges and for booleans
            let target_domain = Spanned::new(target.span, target.inner.domain);
            let mut flow_branch = flow.new_child_branch(target_domain, vec![]);
            let mut flow_branch_flow = flow_branch.as_flow();

            let mut scope_inner = Scope::new_child(pattern_span.join(block.span), scope_parent);
            if let Some((declare_id, declare_value)) = declare {
                let var = flow_branch_flow.var_new_immutable_init(
                    declare_id.span(),
                    VariableId::Id(declare_id),
                    pattern_span,
                    Ok(Value::Hardware(declare_value)),
                );
                scope_inner.maybe_declare(
                    diags,
                    Ok(declare_id.spanned_str(self.refs.fixed.source)),
                    Ok(ScopedEntry::Named(NamedValue::Variable(var))),
                );
            };

            // evaluate the child block
            let end = self.elaborate_block(&scope_inner, &mut flow_branch_flow, stack, block)?;

            // build the if stack
            let (cond, fully_covered) = match cond {
                Some(cond) => (cond, false),
                None => (IrExpression::Bool(true), true),
            };
            if_branch_conditions.push(cond);
            if_branch_flows.push(flow_branch.finish());
            if_branch_ends.push(end);

            if fully_covered {
                break;
            }
        }

        // merge flows
        assert_eq!(if_branch_conditions.len(), if_branch_flows.len());
        let if_branch_blocks = flow.join_child_branches(self.refs, &mut self.large, stmt_span, if_branch_flows)?;

        // merge ends
        assert_eq!(if_branch_conditions.len(), if_branch_ends.len());
        let end = join_block_ends_branches(&if_branch_ends);

        // build complete if chain
        let mut else_ir_block = None;
        for (curr_cond, curr_ir_block) in zip_eq(
            if_branch_conditions.into_iter().rev(),
            if_branch_blocks.into_iter().rev(),
        ) {
            let else_next = match else_ir_block {
                Some(else_ir_block) => {
                    // build if statement
                    // TODO use same simplification logic as used in if statements
                    //   (maybe even handle constant true/false)
                    let if_stmt = IrIfStatement {
                        condition: curr_cond,
                        then_block: curr_ir_block,
                        else_block: Some(else_ir_block),
                    };
                    let if_stmt = Spanned::new(stmt_span, IrStatement::If(if_stmt));
                    IrBlock {
                        statements: vec![if_stmt],
                    }
                }
                None => {
                    // this is the final branch, which means that the condition can be ignored
                    //   (this is easier to reason about for var merging and synthesis tools)
                    let _ = curr_cond;
                    curr_ir_block
                }
            };
            else_ir_block = Some(else_next);
        }

        // push the complete if statement
        if let Some(else_ir_block) = else_ir_block {
            flow.push_ir_statement(Spanned::new(stmt_span, IrStatement::Block(else_ir_block)));
        }

        Ok(end)
    }

    fn elaborate_while_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt: Spanned<&WhileStatement>,
    ) -> DiagResult<BlockEnd> {
        let &WhileStatement {
            span_keyword,
            cond,
            ref body,
        } = stmt.inner;
        let diags = self.refs.diags;

        self.elaborate_loop(
            flow,
            stack,
            span_keyword,
            std::iter::repeat(()),
            |slf, flow, stack, ()| {
                // eval condition
                let cond = slf.eval_expression_as_compile(scope, flow, &Type::Bool, cond, "while loop condition")?;

                // typecheck condition
                let reason = TypeContainsReason::WhileCondition(span_keyword);
                check_type_contains_compile_value(diags, reason, &Type::Bool, cond.as_ref(), false)?;
                let cond = match &cond.inner {
                    &CompileValue::Bool(b) => b,
                    _ => throw!(
                        diags.report_internal_error(cond.span, "expected bool, should have been checked already")
                    ),
                };

                // check condition
                if !cond {
                    return Ok(BlockEnd::CompileExit(EarlyExitKind::Break));
                }

                // elaborate body itself
                slf.elaborate_block(scope, flow, stack, body)
            },
        )
    }

    // TODO code reuse between this and module
    fn elaborate_for_statement(
        &mut self,
        scope_parent: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt: Spanned<&ForStatement<BlockStatement>>,
    ) -> DiagResult<BlockEnd> {
        let &ForStatement {
            span_keyword,
            index: index_id,
            index_ty,
            iter,
            ref body,
        } = stmt.inner;
        let diags = self.refs.diags;

        // header
        let index_ty = index_ty
            .map(|index_ty| self.eval_expression_as_ty(scope_parent, flow, index_ty))
            .transpose();
        let iter = self.eval_expression_as_for_iterator(scope_parent, flow, iter);

        let index_ty = index_ty?;
        let iter = iter?;

        // create variable and scope for the index
        let index_var = flow.var_new(VariableInfo {
            span_decl: index_id.span(),
            id: VariableId::Id(index_id),
            mutable: false,
            ty: None,
            use_ir_variable: None,
        });
        let mut scope_index = Scope::new_child(stmt.span, scope_parent);
        scope_index.maybe_declare(
            diags,
            Ok(index_id.spanned_str(self.refs.fixed.source)),
            Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
        );

        self.elaborate_loop(flow, stack, span_keyword, iter, |slf, flow, stack, index_value| {
            let index_value = index_value.to_maybe_compile(&mut slf.large);

            // typecheck index
            if let Some(index_ty) = &index_ty {
                let curr_spanned = Spanned {
                    span: stmt.inner.iter.span,
                    inner: &index_value,
                };
                let reason = TypeContainsReason::ForIndexType(index_ty.span);
                check_type_contains_value(diags, reason, &index_ty.inner, curr_spanned, false, true)?;
            }

            // set index
            flow.var_set(index_var, span_keyword, Ok(index_value));

            // elaborate body
            slf.elaborate_block(&scope_index, flow, stack, body)
        })
    }

    fn elaborate_loop<F: Flow, T>(
        &mut self,
        flow: &mut F,
        stack: &mut ExitStack,
        span_keyword: Span,
        iter: impl Iterator<Item = T>,
        mut body: impl FnMut(&mut Self, &mut F, &mut ExitStack, T) -> DiagResult<BlockEnd>,
    ) -> DiagResult<BlockEnd> {
        let entry = LoopEntry::new(flow, span_keyword);
        stack.with_loop_entry(entry, |stack| {
            let mut end_joined = BlockEnd::Normal;

            for iter_item in iter {
                self.refs.check_should_stop(span_keyword)?;

                // clear continue flag
                if let FlowKind::Hardware(flow) = flow.kind_mut() {
                    let entry =
                        unwrap_match!(stack.innermost_loop_option().unwrap(), LoopEntry::Hardware(entry) => entry);
                    entry.continue_flag.clear(flow, span_keyword);
                }

                // elaborate body
                //   the body is responsible for checking the exit flags, that's typically handled in elaborate_block
                let end_body_raw = body(self, flow, stack, iter_item)?;

                // stop continue, those don't leak out of the loop or into the next iteration
                let end_body = end_body_raw.remove(ExitMask::from_kind(EarlyExitKind::Continue).into_base());

                // handle end and stop elaborating if possible
                // TODO absorb continue
                end_joined = join_block_ends_sequence(end_joined, end_body);
                if end_joined.is_certain_exit() {
                    break;
                }
            }

            // stop break/continue, they don't leak out of the loop
            let mask_loop = ExitMask {
                can_break: true,
                can_continue: true,
                can_return: false,
            };
            Ok(end_joined.remove(mask_loop))
        })
    }

    fn elaborate_hardware_branch<R>(
        &mut self,
        flow: &mut FlowHardware,
        span: Span,
        cond: Spanned<HardwareValueWithImplications<()>>,
        mut f: impl FnMut(&mut Self, &mut FlowHardware, bool) -> DiagResult<R>,
    ) -> DiagResult<(R, R)> {
        let cond_domain = Spanned::new(cond.span, cond.inner.value.domain);

        // lower then
        let mut then_flow = flow.new_child_branch(cond_domain, cond.inner.implications.if_true);
        let then_result = f(self, &mut then_flow.as_flow(), true);
        let then_flow = then_flow.finish();

        // lower else
        let mut else_flow = flow.new_child_branch(cond_domain, cond.inner.implications.if_false);
        let else_result = f(self, &mut else_flow.as_flow(), false);
        let else_flow = else_flow.finish();

        // join
        let (then_block, else_block) =
            flow.join_child_branches_pair(self.refs, &mut self.large, span, (then_flow, else_flow))?;

        // build if, rewriting it to remove empty blocks
        let ir_if = match (then_block.statements.is_empty(), else_block.statements.is_empty()) {
            // both empty, emit nothing
            (true, true) => None,
            // only then, drop else
            (false, true) => Some(IrIfStatement {
                condition: cond.inner.value.expr,
                then_block,
                else_block: None,
            }),
            // only else, drop then and invert condition
            (true, false) => {
                let inverted_cond = self.large.push_expr(IrExpressionLarge::BoolNot(cond.inner.value.expr));
                Some(IrIfStatement {
                    condition: inverted_cond,
                    then_block: else_block,
                    else_block: None,
                })
            }
            // both, emit the full if
            (false, false) => Some(IrIfStatement {
                condition: cond.inner.value.expr,
                then_block,
                else_block: Some(else_block),
            }),
        };
        if let Some(ir_if) = ir_if {
            flow.push_ir_statement(Spanned::new(span, IrStatement::If(ir_if)));
        }

        Ok((then_result?, else_result?))
    }

    pub fn compile_elaborate_extra_list<'a, F: Flow, I: HasSpan>(
        &mut self,
        scope: &mut Scope,
        flow: &mut F,
        list: &'a ExtraList<I>,
        f: &mut impl FnMut(&mut Self, &mut Scope, &mut F, &'a I) -> DiagResult,
    ) -> DiagResult {
        let ExtraList { span: _, items } = list;
        for item in items {
            match item {
                ExtraItem::Inner(inner) => f(self, scope, flow, inner)?,
                ExtraItem::Declaration(decl) => {
                    self.eval_and_declare_declaration(scope, flow, decl);
                }
                ExtraItem::If(if_stmt) => {
                    let list_inner = self.compile_if_statement_choose_block(scope, flow, if_stmt)?;
                    if let Some(list_inner) = list_inner {
                        self.compile_elaborate_extra_list(scope, flow, list_inner, f)?;
                    }
                }
            }
        }
        Ok(())
    }

    // TODO share code with normal if statement?
    pub fn compile_if_statement_choose_block<'a, B>(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        if_stmt: &'a IfStatement<B>,
    ) -> DiagResult<Option<&'a B>> {
        let diags = self.refs.diags;
        let IfStatement {
            span: _,
            initial_if,
            else_ifs,
            final_else,
        } = if_stmt;

        let mut eval_pair = |pair: &'a IfCondBlockPair<B>| {
            let &IfCondBlockPair {
                span: _,
                span_if,
                cond,
                ref block,
            } = pair;

            let cond = self.eval_expression_as_compile(scope, flow, &Type::Bool, cond, "compile-time if condition")?;

            let reason = TypeContainsReason::IfCondition(span_if);
            let cond = check_type_is_bool_compile(diags, reason, cond)?;

            if cond { Ok(Some(block)) } else { Ok(None) }
        };

        if let Some(block) = eval_pair(initial_if)? {
            return Ok(Some(block));
        }
        for else_if in else_ifs {
            if let Some(block) = eval_pair(else_if)? {
                return Ok(Some(block));
            }
        }
        Ok(final_else.as_ref())
    }
}

fn join_block_ends_sequence(first: BlockEnd, second: BlockEnd) -> BlockEnd {
    if first.is_certain_exit() {
        first
    } else {
        BlockEnd::new(
            second.is_certain_exit(),
            first.possible_exit_mask() | second.possible_exit_mask(),
        )
    }
}

fn join_block_ends_branches(ends: &[BlockEnd]) -> BlockEnd {
    // handle simple cases
    match ends.iter().all_equal_value() {
        // all ends are the same, just return that
        Ok(&end) => return end,
        // array is empty, this shouldn't really happen and if it does it doesn't matter what we return here
        Err(None) => return BlockEnd::Normal,
        // different ends found, fallthrough into hardware merge
        Err(Some((_, _))) => {}
    }

    // hardware merge
    let mut all_certain = true;
    let mut any_mask = ExitMask::<0>::default();

    for &end in ends {
        let (is_certain, mask) = match end {
            BlockEnd::Normal => (false, ExitMask::default()),
            BlockEnd::CompileExit(kind) => (true, ExitMask::from_kind(kind).into_base()),
            BlockEnd::HardwareExit(mask) => (true, mask.into_base()),
            BlockEnd::HardwareMaybeExit(mask) => (false, mask.into_base()),
        };
        all_certain &= is_certain;
        any_mask = any_mask | mask;
    }

    BlockEnd::new(all_certain, any_mask)
}
