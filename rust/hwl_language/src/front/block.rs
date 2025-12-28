use crate::front::check::{
    TypeContainsReason, check_type_contains_value, check_type_is_bool, check_type_is_bool_compile,
    check_type_is_range_compile,
};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::exit::{ExitStack, LoopEntry, ReturnEntryKind};
use crate::front::flow::{Flow, FlowHardware, ImplicationContradiction, VariableId};
use crate::front::flow::{FlowKind, VariableInfo};
use crate::front::function::check_function_return_type_and_set_value;
use crate::front::implication::{HardwareValueWithImplications, Implication, ValueWithImplications};
use crate::front::item::{ElaboratedEnum, HardwareChecked};
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::{HardwareType, NonHardwareType, Type, TypeBool, Typed};
use crate::front::value::{
    CompileCompoundValue, CompileValue, HardwareValue, MaybeCompile, NotCompile, SimpleCompileValue, Value, ValueCommon,
};
use crate::mid::ir::{
    IrBlock, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIfStatement, IrIntCompareOp, IrLargeArena, IrStatement,
};
use crate::syntax::ast::{
    Block, BlockStatement, BlockStatementKind, ConstBlock, ExtraItem, ExtraList, ForStatement, IfCondBlockPair,
    IfStatement, MatchBranch, MatchPattern, MatchStatement, MaybeIdentifier, ReturnStatement, VariableDeclaration,
    WhileStatement,
};
use crate::syntax::pos::{HasSpan, Pos, Span, Spanned};
use crate::throw;
use crate::util::big_int::BigInt;
use crate::util::data::VecExt;
use crate::util::iter::IterExt;
use crate::util::range::Range;
use crate::util::range_multi::{AnyMultiRange, ClosedMultiRange, MultiRange};
use crate::util::{ResultExt, result_pair};
use itertools::{Either, Itertools, enumerate, zip_eq};
use unwrap_match::unwrap_match;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[must_use]
pub enum BlockEnd {
    /// No early exit, proceed normally.
    Normal,

    /// This block turns out to have been unreachable. This can happen for conditions that are in hindsight always
    /// false, for `assert(false)`-like constructs or in dead code after early exits.
    /// This [BlockEnd] is the bottom value of the lattice,
    /// for branch joining it should always decay and result in the other branches.
    Unreachable,

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

/// Another encoding of [BlockEnd] that's easier to use for merging and joining.
#[derive(Debug, Copy, Clone)]
pub enum BlockEndFlags {
    /// Corresponds to [BlockEnd::Unreachable]
    Unreachable,
    /// Represents other [BlockEnd] variants.
    /// `certain_exit` indicates whether the block definitely exits early,
    /// `mask` indicates which kinds of exits are possible.
    Normal { certain_exit: bool, mask: ExitMask<0> },
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
    fn from_flags(flags: BlockEndFlags) -> BlockEnd {
        #[allow(clippy::collapsible_else_if)]
        match flags {
            BlockEndFlags::Unreachable => BlockEnd::Unreachable,
            BlockEndFlags::Normal { certain_exit, mask } => {
                if certain_exit {
                    if let Some(mask) = ExitMask::<2>::from_base(mask) {
                        // at least two options, so hardware-dependant exit
                        BlockEnd::HardwareExit(mask)
                    } else {
                        // at most one option, so definite compile-known exit
                        if mask.can_return {
                            BlockEnd::CompileExit(EarlyExitKind::Return)
                        } else if mask.can_break {
                            BlockEnd::CompileExit(EarlyExitKind::Break)
                        } else if mask.can_continue {
                            BlockEnd::CompileExit(EarlyExitKind::Continue)
                        } else {
                            unreachable!("certain exit but empty mask")
                        }
                    }
                } else {
                    if let Some(mask) = ExitMask::<1>::from_base(mask) {
                        // at least one option, so maybe exit
                        BlockEnd::HardwareMaybeExit(mask)
                    } else {
                        // no exit options, so normal
                        BlockEnd::Normal
                    }
                }
            }
        }
    }

    fn into_flags(self) -> BlockEndFlags {
        match self {
            BlockEnd::Normal => BlockEndFlags::Normal {
                certain_exit: false,
                mask: ExitMask::default(),
            },
            BlockEnd::Unreachable => BlockEndFlags::Unreachable,
            BlockEnd::CompileExit(kind) => BlockEndFlags::Normal {
                certain_exit: true,
                mask: ExitMask::from_kind(kind).into_base(),
            },
            BlockEnd::HardwareExit(mask) => BlockEndFlags::Normal {
                certain_exit: true,
                mask: mask.into_base(),
            },
            BlockEnd::HardwareMaybeExit(mask) => BlockEndFlags::Normal {
                certain_exit: false,
                mask: mask.into_base(),
            },
        }
    }

    pub fn should_stop_sequence(self) -> bool {
        match self {
            BlockEnd::Unreachable | BlockEnd::CompileExit(_) | BlockEnd::HardwareExit(_) => true,
            BlockEnd::Normal | BlockEnd::HardwareMaybeExit(_) => false,
        }
    }

    pub fn should_recheck_exit_flags(self) -> bool {
        match self {
            BlockEnd::HardwareMaybeExit(_) => true,
            BlockEnd::Normal | BlockEnd::Unreachable | BlockEnd::CompileExit(_) | BlockEnd::HardwareExit(_) => false,
        }
    }

    pub fn unwrap_normal(self, diags: &Diagnostics, span: Span) -> DiagResult {
        match self {
            BlockEnd::Normal => Ok(()),
            _ => Err(diags.report_internal_error(span, "unexpected early exit")),
        }
    }

    pub fn remove(self, mask_remove: ExitMask<0>) -> BlockEnd {
        match self.into_flags() {
            BlockEndFlags::Unreachable => BlockEnd::Unreachable,
            BlockEndFlags::Normal { certain_exit, mask } => {
                let result_mask = ExitMask {
                    can_return: mask.can_return & !mask_remove.can_return,
                    can_break: mask.can_break & !mask_remove.can_break,
                    can_continue: mask.can_continue & !mask_remove.can_continue,
                };

                let result_certain_exit = certain_exit && ExitMask::<1>::from_base(result_mask).is_some();

                BlockEnd::from_flags(BlockEndFlags::Normal {
                    certain_exit: result_certain_exit,
                    mask: result_mask,
                })
            }
        }
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

#[derive(Debug)]
enum CompileBranchMatched {
    Yes(Option<BranchDeclare<CompileValue>>),
    No,
}

#[derive(Debug)]
struct HardwareBranchMatched {
    cond: Option<IrExpression>,
    declare: Option<BranchDeclare<HardwareValue>>,
    implications: Vec<Implication>,
}

#[derive(Debug)]
struct BranchDeclare<V> {
    id: MaybeIdentifier,
    value: V,
}

#[derive(Debug)]
pub enum EvaluatedMatchPattern {
    Wildcard,
    WildcardVal(MaybeIdentifier),
    EqualTo(Spanned<CompileValue>),
    InRange(Spanned<Range<BigInt>>),
    IsEnumVariant {
        variant_index: usize,
        payload_id: Option<MaybeIdentifier>,
    },
}

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

        // Create a new scope, necessary to ensure declarations don't leak into the parent scope.
        let mut scope = Scope::new_child(span, scope_parent);

        // Create a new scoped flow, not strictly necessary for correctness,
        //   but allows dropping variables that go out of scope immediately.
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

        let end = match stack.early_exit_condition(self.refs, diags, &mut self.large, flow, span)? {
            MaybeCompile::Compile(exit_cond) => {
                if exit_cond {
                    return Err(diags
                        .report_internal_error(span, "compile-time early exit condition should be handled elsewhere"));
                }
                self.elaborate_block_statements_without_immediate_exit_check(scope, flow, stack, statements)?
            }
            MaybeCompile::Hardware(exit_cond) => {
                let flow = flow.require_hardware(span, "hardware exit condition")?;

                let exit_cond = Spanned::new(span, exit_cond);

                self.elaborate_hardware_if(flow, span, exit_cond, |slf, branch_flow, branch_cond| {
                    if branch_cond {
                        // early exit, do nothing
                        //   as far as block end merging is concerned this branch is not actually reachable,
                        //   since we should have exited earlier already
                        Ok(BlockEnd::Unreachable)
                    } else {
                        // no early exit, so actually run the statements
                        slf.elaborate_block_statements_without_immediate_exit_check(
                            scope,
                            branch_flow,
                            stack,
                            statements,
                        )
                    }
                })?
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

            if end_joined.should_stop_sequence() {
                break;
            }

            if end_curr.should_recheck_exit_flags() {
                // recurse and return, instead of continuing the loop
                let statements_rest = &statements[stmt_index + 1..];
                let end_rest = self.elaborate_block_statements(scope, flow, stack, statements_rest)?;
                return Ok(join_block_ends_sequence(end_joined, end_rest));
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
        let elab = &self.refs.shared.elaboration_arenas;

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
                        check_type_contains_value(diags, elab, reason, &ty.inner, init.as_ref())?;
                    }

                    // build variable
                    let info = VariableInfo {
                        span_decl: id.span(),
                        id: VariableId::Id(id),
                        mutable,
                        ty,
                        join_ir_variable: None,
                    };
                    let var = flow.var_new(info);

                    // store initial value if there is one
                    if let Some(init) = init {
                        flow.var_set(self.refs, var, decl.span, Ok(init.inner))?;
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
                    entry.return_flag.set(flow, span_return)?;
                }

                // check type and store value
                let type_unit = Type::unit();
                let expected_ty = entry.return_type.map_or(&type_unit, |ty| ty.inner);
                let value = value
                    .map(|value| self.eval_expression(scope, flow, expected_ty, value))
                    .transpose()?;

                check_function_return_type_and_set_value(self.refs, flow, entry, stmt_span, span_return, value)?;

                BlockEnd::CompileExit(EarlyExitKind::Return)
            }
            &BlockStatementKind::Break { span } => {
                let entry = stack.innermost_loop(diags, span, "break")?;
                if let LoopEntry::Hardware(entry) = entry {
                    entry.break_flag.set(flow, span)?;
                }
                BlockEnd::CompileExit(EarlyExitKind::Break)
            }
            &BlockStatementKind::Continue { span } => {
                let entry = stack.innermost_loop(diags, span, "continue")?;
                if let LoopEntry::Hardware(entry) = entry {
                    entry.continue_flag.set(flow, span)?;
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
        let elab = &self.refs.shared.elaboration_arenas;

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
        let cond_span = cond.span;

        let cond = self.eval_expression_with_implications(scope, flow, &Type::Bool, cond)?;

        let reason = TypeContainsReason::IfCondition(span_if);
        let cond = check_type_is_bool(diags, elab, reason, cond)?;

        match cond {
            // evaluate the if at compile-time
            MaybeCompile::Compile(cond_eval) => {
                // only visit the selected branch
                if cond_eval {
                    self.elaborate_block(scope, flow, stack, block)
                } else {
                    self.elaborate_if_statement(scope, flow, stack, remaining_ifs.split_first(), final_else)
                }
            }
            // evaluate the if in hardware, generating IR
            MaybeCompile::Hardware(cond_value) => {
                let flow = flow.require_hardware(cond_span, "hardware value")?;

                let cond_value = Spanned::new(cond_span, cond_value);
                self.elaborate_hardware_if(flow, span_if, cond_value, |slf, branch_flow, branch_cond| {
                    if branch_cond {
                        slf.elaborate_block(scope, branch_flow, stack, block)
                    } else {
                        slf.elaborate_if_statement(scope, branch_flow, stack, remaining_ifs.split_first(), final_else)
                    }
                })
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
        let elab = &self.refs.shared.elaboration_arenas;

        let &MatchStatement {
            target,
            pos_end,
            ref branches,
        } = stmt.inner;

        // eval target
        let target = self.eval_expression_with_implications(scope, flow, &Type::Any, target)?;
        let target_ty = target.inner.ty();

        // eval branches
        let branch_patterns = branches
            .iter()
            .map(|branch| self.eval_match_pattern(scope, flow, target.span, &target_ty, branch.pattern.as_ref()))
            .try_collect_all_vec()?;

        match CompileValue::try_from(&target.inner) {
            Ok(target_inner) => {
                // compile-time target value, we can handle the entire match at compile-time
                let target = Spanned::new(target.span, target_inner);
                self.elaborate_match_statement_compile(scope, flow, stack, target, pos_end, branches, branch_patterns)
            }
            Err(NotCompile) => {
                // hardware target value (at least partially), convert the target to full hardware
                //   and handle the match at hardware time
                let flow = flow.require_hardware(stmt.span, "match on hardware target")?;

                let target_ty = target_ty.as_hardware_type(elab).map_err(|_: NonHardwareType| {
                    diags.report_simple(
                        "failed to fully convert non-compile match target to hardware",
                        target.span,
                        format!("match target has non-hardware type {}", target_ty.value_string(elab)),
                    )
                })?;

                let target_inner = match target.inner {
                    ValueWithImplications::Simple(t) => HardwareValueWithImplications::simple(
                        t.as_hardware_value_unchecked(self.refs, &mut self.large, target.span, target_ty.clone())?,
                    ),
                    ValueWithImplications::Compound(t) => HardwareValueWithImplications::simple(
                        t.as_hardware_value_unchecked(self.refs, &mut self.large, target.span, target_ty.clone())?,
                    ),
                    ValueWithImplications::Hardware(t) => t,
                };
                let target = Spanned::new(target.span, target_inner);

                self.elaborate_match_statement_hardware(
                    scope,
                    flow,
                    stack,
                    stmt.span,
                    target,
                    pos_end,
                    branches,
                    branch_patterns,
                )
            }
        }
    }

    fn eval_match_pattern(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        target_span: Span,
        target_ty: &Type,
        pattern: Spanned<&MatchPattern>,
    ) -> DiagResult<EvaluatedMatchPattern> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        match *pattern.inner {
            MatchPattern::Wildcard => Ok(EvaluatedMatchPattern::Wildcard),
            MatchPattern::WildcardVal(id) => Ok(EvaluatedMatchPattern::WildcardVal(id)),
            MatchPattern::EqualTo(value) => {
                let value = self.eval_expression_as_compile(
                    scope,
                    flow,
                    target_ty,
                    value,
                    Spanned::new(pattern.span, "match branch"),
                )?;
                Ok(EvaluatedMatchPattern::EqualTo(value))
            }
            MatchPattern::InRange { span_in, range } => {
                let value = self.eval_expression_as_compile(
                    scope,
                    flow,
                    target_ty,
                    range,
                    Spanned::new(pattern.span, "match branch"),
                )?;
                let value = check_type_is_range_compile(diags, elab, TypeContainsReason::Operator(span_in), value)?;
                Ok(EvaluatedMatchPattern::InRange(Spanned::new(range.span, value)))
            }
            MatchPattern::IsEnumVariant { variant, payload_id } => {
                if let &Type::Enum(target_ty) = target_ty {
                    let enum_info = self.refs.shared.elaboration_arenas.enum_info(target_ty);

                    let variant_str = variant.spanned_str(self.refs.fixed.source);
                    let variant_index = enum_info.find_variant(diags, variant_str)?;
                    let variant_info = &enum_info.variants[variant_index];

                    match (payload_id, &variant_info.payload_ty) {
                        (None, None) | (Some(_), Some(_)) => {}
                        (None, Some(variant_pyload_ty)) => {
                            let diag = Diagnostic::new("enum variant payload mismatch")
                                .add_info(variant_pyload_ty.span, "declared with a payload here")
                                .add_error(pattern.span, "matched without a payload here")
                                .finish();
                            return Err(diags.report(diag));
                        }
                        (Some(payload), None) => {
                            let diag = Diagnostic::new("enum variant payload mismatch")
                                .add_info(variant_info.id.span, "declared without a payload here")
                                .add_error(payload.span(), "matched with a payload here")
                                .finish();
                            return Err(diags.report(diag));
                        }
                    }

                    Ok(EvaluatedMatchPattern::IsEnumVariant {
                        variant_index,
                        payload_id,
                    })
                } else {
                    let diag = Diagnostic::new("enum match pattern with non-enum target type")
                        .add_info(target_span, format!("target has type {}", target_ty.value_string(elab)))
                        .add_error(pattern.span, "enum match pattern used here")
                        .finish();
                    Err(diags.report(diag))
                }
            }
        }
    }

    fn elaborate_match_statement_compile(
        &mut self,
        scope_parent: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        target: Spanned<CompileValue>,
        pos_end: Pos,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<EvaluatedMatchPattern>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        // compile-time match, just check each pattern in sequence with early exit
        for (branch, pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch { pattern: _, block } = branch;
            let pattern_span = branch.pattern.span;

            let matched = match pattern {
                EvaluatedMatchPattern::Wildcard => CompileBranchMatched::Yes(None),
                EvaluatedMatchPattern::WildcardVal(id) => CompileBranchMatched::Yes(Some(BranchDeclare {
                    id,
                    value: target.inner.clone(),
                })),
                EvaluatedMatchPattern::EqualTo(value) => {
                    if target.inner == value.inner {
                        CompileBranchMatched::Yes(None)
                    } else {
                        CompileBranchMatched::No
                    }
                }
                EvaluatedMatchPattern::InRange(range) => {
                    if let CompileValue::Simple(SimpleCompileValue::Int(target)) = &target.inner
                        && range.inner.contains(target)
                    {
                        CompileBranchMatched::Yes(None)
                    } else {
                        CompileBranchMatched::No
                    }
                }
                EvaluatedMatchPattern::IsEnumVariant {
                    variant_index,
                    payload_id,
                } => {
                    if let CompileValue::Compound(CompileCompoundValue::Enum(target)) = &target.inner
                        && target.variant == variant_index
                    {
                        let declare = match (payload_id, &target.payload) {
                            (None, None) => None,
                            (Some(payload_id), Some(target_payload)) => Some(BranchDeclare {
                                id: payload_id,
                                value: target_payload.as_ref().clone(),
                            }),
                            (None, Some(_)) | (Some(_), None) => {
                                return Err(diags.report_internal_error(pattern_span, "payload mismatch"));
                            }
                        };

                        CompileBranchMatched::Yes(declare)
                    } else {
                        CompileBranchMatched::No
                    }
                }
            };

            match matched {
                CompileBranchMatched::Yes(declare) => {
                    // declare any pattern variables
                    let mut scope_branch = Scope::new_child(pattern_span.join(block.span), scope_parent);
                    if let Some(BranchDeclare {
                        id: declare_id,
                        value: declare_value,
                    }) = declare
                    {
                        let var = flow.var_new_immutable_init(
                            self.refs,
                            declare_id.span(),
                            VariableId::Id(declare_id),
                            pattern_span,
                            Ok(Value::from(declare_value)),
                        )?;
                        scope_branch.maybe_declare(
                            diags,
                            Ok(declare_id.spanned_str(self.refs.fixed.source)),
                            Ok(ScopedEntry::Named(NamedValue::Variable(var))),
                        );
                    }

                    // evaluate the branch and exit
                    return self.elaborate_block(&scope_branch, flow, stack, block);
                }
                CompileBranchMatched::No => {
                    // continue to next branch
                }
            }
        }

        let diag = Diagnostic::new("match statement reached end without matching any branch")
            .add_info(
                target.span,
                format!("target value `{}`", target.inner.value_string(elab)),
            )
            .add_error(Span::empty_at(pos_end), "did not match any branch")
            .finish();
        Err(diags.report(diag))
    }

    fn elaborate_match_statement_hardware(
        &mut self,
        scope_parent: &Scope,
        flow_parent: &mut FlowHardware,
        stack: &mut ExitStack,
        span_match: Span,
        target: Spanned<HardwareValueWithImplications>,
        pos_end: Pos,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<EvaluatedMatchPattern>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let target_version = target.inner.version;
        let target_value = target.inner.value;
        let target_domain = Spanned::new(target.span, target_value.domain);

        // TODO extract function
        let mut coverage_remaining = match &target_value.ty {
            HardwareType::Bool => MatchCoverage::Bool {
                rem_false: true,
                rem_true: true,
            },
            HardwareType::Int(ty) => MatchCoverage::Int {
                rem_range: ClosedMultiRange::from(ty.clone()),
            },
            &HardwareType::Enum(ty) => {
                let elab_enum = elab.enum_info(ty.inner());
                MatchCoverage::Enum {
                    target_ty: ty,
                    rem_variants: vec![true; elab_enum.variants.len()],
                }
            }
            _ => {
                return Err(diags.report_todo(
                    target.span,
                    format!(
                        "hardware matching for target type {}",
                        target_value.ty.value_string(elab)
                    ),
                ));
            }
        };

        let mut all_conditions = vec![];
        let mut all_contents = vec![];
        let mut all_ends = vec![];

        // TODO emit warning if parts are already covered
        for (branch, branch_pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch {
                pattern: _,
                block: branch_block,
            } = branch;
            let branch_pattern_span = branch.pattern.span;

            // TODO extract function
            let matched = match branch_pattern {
                EvaluatedMatchPattern::Wildcard => {
                    coverage_remaining.clear();
                    HardwareBranchMatched {
                        cond: None,
                        declare: None,
                        implications: vec![],
                    }
                }
                EvaluatedMatchPattern::WildcardVal(id) => {
                    coverage_remaining.clear();
                    HardwareBranchMatched {
                        cond: None,
                        declare: Some(BranchDeclare {
                            id,
                            value: target_value.clone(),
                        }),
                        implications: vec![],
                    }
                }
                EvaluatedMatchPattern::EqualTo(value) => {
                    check_type_contains_value(
                        diags,
                        elab,
                        TypeContainsReason::MatchPattern(value.span),
                        &target_value.ty.as_type(),
                        value.as_ref(),
                    )?;

                    let (cond, implications) = match &mut coverage_remaining {
                        MatchCoverage::Bool { rem_false, rem_true } => {
                            let value = unwrap_match!(value.inner, CompileValue::Simple(SimpleCompileValue::Bool(value)) => value);

                            // TODO should we also imply that the bool itself is true/false? How does this work for ifs?
                            //  Or do the implications already cover that?
                            if value {
                                *rem_true = false;
                                (target_value.expr.clone(), target.inner.implications.if_true.clone())
                            } else {
                                *rem_false = false;
                                let cond = self
                                    .large
                                    .push_expr(IrExpressionLarge::BoolNot(target_value.expr.clone()));
                                (cond, target.inner.implications.if_false.clone())
                            }
                        }
                        MatchCoverage::Int { rem_range } => {
                            let value = unwrap_match!(value.inner, CompileValue::Simple(SimpleCompileValue::Int(value)) => value);
                            let value_range = MultiRange::from(Range::single(value.clone()));
                            *rem_range = rem_range.subtract(&value_range);

                            let cond = self.large.push_expr(IrExpressionLarge::IntCompare(
                                IrIntCompareOp::Eq,
                                target_value.expr.clone(),
                                IrExpression::Int(value.clone()),
                            ));

                            let implications = if let Some(target_version) = target_version {
                                vec![Implication::new_int(target_version, value_range)]
                            } else {
                                vec![]
                            };
                            (cond, implications)
                        }
                        MatchCoverage::Enum { .. } => {
                            return Err(diags.report_todo(branch_pattern_span, "matching enum value by equality"));
                        }
                    };

                    HardwareBranchMatched {
                        cond: Some(cond),
                        declare: None,
                        implications,
                    }
                }
                EvaluatedMatchPattern::InRange(range) => match &mut coverage_remaining {
                    MatchCoverage::Int { rem_range } => {
                        // TODO warn if parts of range already covered
                        // TODO share code with ordinary comparisons?
                        let range_multi = MultiRange::from(range.inner.clone());
                        *rem_range = rem_range.subtract(&range_multi);

                        let cond = build_ir_int_in_range(&mut self.large, &target_value.expr, range.inner.clone());

                        let implications = if let Some(target_version) = target_version {
                            vec![Implication::new_int(
                                target_version,
                                MultiRange::from(range.inner.clone()),
                            )]
                        } else {
                            vec![]
                        };

                        HardwareBranchMatched {
                            cond,
                            declare: None,
                            implications,
                        }
                    }
                    _ => {
                        let diag = Diagnostic::new("range match pattern with non-integer target type")
                            .add_info(
                                target.span,
                                format!("target has type {}", target_value.ty.value_string(elab)),
                            )
                            .add_error(range.span, "integer range match pattern used here")
                            .finish();
                        return Err(diags.report(diag));
                    }
                },
                EvaluatedMatchPattern::IsEnumVariant {
                    variant_index,
                    payload_id,
                } => match &mut coverage_remaining {
                    MatchCoverage::Enum {
                        target_ty,
                        rem_variants,
                    } => {
                        rem_variants[variant_index] = false;

                        let enum_info = elab.enum_info(target_ty.inner());
                        let enum_info_hw = enum_info.hw.as_ref().unwrap();

                        let cond =
                            enum_info_hw.check_tag_matches(&mut self.large, target_value.expr.clone(), variant_index);
                        let payload_hw = enum_info_hw.extract_payload(&mut self.large, &target_value, variant_index);
                        let declare = match (payload_id, payload_hw) {
                            (None, None) => None,
                            (Some(id), Some(value)) => Some(BranchDeclare { id, value }),
                            (None, Some(_)) | (Some(_), None) => {
                                return Err(diags.report_internal_error(branch_pattern_span, "payload mismatch"));
                            }
                        };

                        HardwareBranchMatched {
                            cond: Some(cond),
                            declare,
                            implications: vec![],
                        }
                    }
                    _ => {
                        let diag = Diagnostic::new("enum variant match pattern with non-enum target type")
                            .add_info(
                                target.span,
                                format!("target has type {}", target_value.ty.value_string(elab)),
                            )
                            .add_error(branch_pattern_span, "enum variant match pattern used here")
                            .finish();
                        return Err(diags.report(diag));
                    }
                },
            };

            let HardwareBranchMatched {
                cond,
                declare,
                implications,
            } = matched;

            let mut branch_scope = Scope::new_child(branch.span(), scope_parent);
            let mut branch_flow =
                match flow_parent.new_child_branch(self, branch.span(), target_domain, implications)? {
                    Ok(branch_flow) => branch_flow,
                    Err(ImplicationContradiction) => continue,
                };

            if let Some(declare) = declare {
                let BranchDeclare { id, value } = declare;
                if let MaybeIdentifier::Identifier(id) = id {
                    let var = branch_flow.as_flow().var_new_immutable_init(
                        self.refs,
                        id.span(),
                        VariableId::Id(MaybeIdentifier::Identifier(id)),
                        id.span(),
                        Ok(Value::Hardware(value)),
                    )?;
                    branch_scope.declare(
                        diags,
                        Ok(id.spanned_str(self.refs.fixed.source)),
                        Ok(ScopedEntry::Named(NamedValue::Variable(var))),
                    );
                }
            }

            let branch_end = self.elaborate_block(&branch_scope, &mut branch_flow.as_flow(), stack, branch_block)?;
            let content = branch_flow.finish();

            all_conditions.push(cond);
            all_contents.push(content);
            all_ends.push(branch_end);
        }

        // check coverage
        // TODO move to separate function
        let span_end = Span::empty_at(pos_end);
        let title_non_exhaustive = "hardware match statement is not exhaustive";
        let msg_target_type = format!("target has type {}", target_value.ty.value_string(elab));
        match coverage_remaining {
            MatchCoverage::Bool { rem_false, rem_true } => {
                let values_not_covered = match (rem_false, rem_true) {
                    (false, false) => None,
                    (true, false) => Some("[false]"),
                    (false, true) => Some("[true]"),
                    (true, true) => Some("[false, true]"),
                };
                if let Some(values_not_covered) = values_not_covered {
                    let diag = Diagnostic::new(title_non_exhaustive)
                        .add_info(target.span, msg_target_type)
                        .add_error(span_end, format!("values not covered: {}", values_not_covered))
                        .finish();
                    return Err(diags.report(diag));
                }
            }
            MatchCoverage::Int { rem_range } => {
                if !rem_range.is_empty() {
                    let diag = Diagnostic::new(title_non_exhaustive)
                        .add_info(target.span, msg_target_type)
                        .add_error(span_end, format!("ranges not covered: {rem_range}"))
                        .finish();
                    return Err(diags.report(diag));
                }
            }
            MatchCoverage::Enum {
                target_ty,
                rem_variants,
            } => {
                if rem_variants.iter().any(|&x| x) {
                    let enum_info = elab.enum_info(target_ty.inner());
                    let variants_not_covered = enum_info
                        .variants
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| rem_variants[i])
                        .map(|(_, (name, _))| format!(".{name}"))
                        .join(", ");

                    let diag = Diagnostic::new(title_non_exhaustive)
                        .add_info(target.span, msg_target_type)
                        .add_info(enum_info.unique.id().span(), "enum declared here")
                        .add_error(span_end, format!("variants not covered: [{}]", variants_not_covered))
                        .finish();
                    return Err(diags.report(diag));
                }
            }
        }

        // join things
        let all_blocks = flow_parent.join_child_branches(self.refs, &mut self.large, span_match, all_contents)?;
        let joined_end = join_block_ends_branches(&all_ends);

        // build the if statement
        // TODO flatten this into single if/else-if/else structure or even a match?
        let mut joined_statement = None;
        for (cond, block) in zip_eq(all_conditions.into_iter().rev(), all_blocks.into_iter().rev()) {
            joined_statement = match cond {
                None => Some(IrStatement::Block(block)),
                Some(cond) => Some(IrStatement::If(IrIfStatement {
                    condition: cond,
                    then_block: block,
                    else_block: joined_statement.map(|curr| IrBlock::new_single(span_match, curr)),
                })),
            };
        }

        if let Some(curr) = joined_statement {
            flow_parent.push_ir_statement(Spanned::new(span_match, curr));
        }

        Ok(joined_end)
    }

    fn elaborate_while_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        stack: &mut ExitStack,
        stmt: Spanned<&WhileStatement>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let &WhileStatement {
            span_keyword,
            cond,
            ref body,
        } = stmt.inner;

        self.elaborate_loop(
            flow,
            stack,
            span_keyword,
            std::iter::repeat(()),
            |slf, flow, stack, ()| {
                // eval condition
                let cond = slf.eval_expression_as_compile(
                    scope,
                    flow,
                    &Type::Bool,
                    cond,
                    Spanned::new(span_keyword, "while loop condition"),
                )?;

                // typecheck condition
                let reason = TypeContainsReason::WhileCondition(span_keyword);
                check_type_contains_value(diags, elab, reason, &Type::Bool, cond.as_ref())?;
                let cond = match &cond.inner {
                    &CompileValue::Simple(SimpleCompileValue::Bool(b)) => b,
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
        let diags = self.refs.diags;
        let elab = &self.refs.shared.elaboration_arenas;

        let &ForStatement {
            span_keyword,
            index: index_id,
            index_ty,
            iter,
            ref body,
        } = stmt.inner;

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
            join_ir_variable: None,
        });
        let mut scope_index = Scope::new_child(stmt.span, scope_parent);
        scope_index.maybe_declare(
            diags,
            Ok(index_id.spanned_str(self.refs.fixed.source)),
            Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
        );

        self.elaborate_loop(flow, stack, span_keyword, iter, |slf, flow, stack, index_value| {
            let index_value = index_value.map_hardware(|h| h.map_expression(|h| slf.large.push_expr(h)));

            // typecheck index (if specified)
            if let Some(index_ty) = &index_ty {
                let curr_spanned = Spanned {
                    span: stmt.inner.iter.span,
                    inner: &index_value,
                };
                let reason = TypeContainsReason::ForIndexType(index_ty.span);
                check_type_contains_value(diags, elab, reason, &index_ty.inner, curr_spanned)?;
            }

            // set index and elaborate body
            flow.var_set(slf.refs, index_var, span_keyword, Ok(index_value))?;
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
        let entry = LoopEntry::new(flow, span_keyword)?;
        stack.with_loop_entry(entry, |stack| {
            let mut end_joined = BlockEnd::Normal;

            for iter_item in iter {
                self.refs.check_should_stop(span_keyword)?;

                // clear continue flag
                if let FlowKind::Hardware(flow) = flow.kind_mut() {
                    let entry =
                        unwrap_match!(stack.innermost_loop_option().unwrap(), LoopEntry::Hardware(entry) => entry);
                    entry.continue_flag.clear(flow, span_keyword)?;
                }

                // elaborate body
                //   the body is responsible for checking the exit flags, that's typically handled in elaborate_block
                let end_body_raw = body(self, flow, stack, iter_item)?;

                // stop continue, it shouldn't leak into the next iteration
                let end_body = end_body_raw.remove(ExitMask::from_kind(EarlyExitKind::Continue).into_base());

                // handle end and stop elaborating if possible
                end_joined = join_block_ends_sequence(end_joined, end_body);
                if end_joined.should_stop_sequence() {
                    break;
                }
            }

            // absorb break/continue, they don't leak out of the loop
            let mask_loop = ExitMask {
                can_break: true,
                can_continue: true,
                can_return: false,
            };
            Ok(end_joined.remove(mask_loop))
        })
    }

    fn elaborate_hardware_if(
        &mut self,
        flow: &mut FlowHardware,
        span: Span,
        cond: Spanned<HardwareValueWithImplications<TypeBool>>,
        mut f: impl FnMut(&mut Self, &mut FlowHardware, bool) -> DiagResult<BlockEnd>,
    ) -> DiagResult<BlockEnd> {
        let cond_domain = Spanned::new(cond.span, cond.inner.value.domain);

        // lower then
        let then_flow_end = match flow.new_child_branch(self, span, cond_domain, cond.inner.implications.if_true)? {
            Ok(mut then_flow) => {
                let then_end = f(self, &mut then_flow.as_flow(), true)?;
                let then_flow = then_flow.finish();
                Some((then_flow, then_end))
            }
            Err(ImplicationContradiction) => None,
        };

        // lower else
        let else_flow_end = match flow.new_child_branch(self, span, cond_domain, cond.inner.implications.if_false)? {
            Ok(mut else_flow) => {
                let else_end = f(self, &mut else_flow.as_flow(), false)?;
                let else_flow = else_flow.finish();
                Some((else_flow, else_end))
            }
            Err(ImplicationContradiction) => None,
        };

        // join
        match (then_flow_end, else_flow_end) {
            (Some(then_flow_end), Some(else_flow_end)) => {
                // both branches are possible
                let (then_flow, then_end) = then_flow_end;
                let (else_flow, else_end) = else_flow_end;

                // join flows
                let (then_block, else_block) =
                    flow.join_child_branches_pair(self.refs, &mut self.large, span, (then_flow, else_flow))?;

                // build if
                let ir_if = build_ir_if_statement(
                    &mut self.large,
                    cond.inner.value.expr,
                    Some(then_block),
                    Some(else_block),
                );
                if let Some(ir_if) = ir_if {
                    let ir_if = match ir_if {
                        Either::Left(ir_if) => IrStatement::Block(ir_if),
                        Either::Right(ir_if) => IrStatement::If(ir_if),
                    };
                    flow.push_ir_statement(Spanned::new(span, ir_if));
                }

                // join ends
                Ok(join_block_ends_branches(&[then_end, else_end]))
            }
            (Some(flow_end), None) | (None, Some(flow_end)) => {
                // only one branch is possible, so no need to build an if statement
                let (case_flow, case_end) = flow_end;

                let block = flow
                    .join_child_branches(self.refs, &mut self.large, span, vec![case_flow])?
                    .single()
                    .unwrap();

                flow.push_ir_statement(Spanned::new(span, IrStatement::Block(block)));

                Ok(case_end)
            }
            (None, None) => {
                // both branches contradict, this basically means this entire flow is unreachable,
                //  no need to do anything at all
                Ok(BlockEnd::Unreachable)
            }
        }
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
        let elab = &self.refs.shared.elaboration_arenas;

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

            let cond = self.eval_expression_as_compile(
                scope,
                flow,
                &Type::Bool,
                cond,
                Spanned::new(span_if, "compile-time if condition"),
            )?;

            let reason = TypeContainsReason::IfCondition(span_if);
            let cond = check_type_is_bool_compile(diags, elab, reason, cond)?;

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

#[derive(Debug)]
enum MatchCoverage {
    Bool {
        rem_false: bool,
        rem_true: bool,
    },
    Int {
        rem_range: ClosedMultiRange<BigInt>,
    },
    Enum {
        target_ty: HardwareChecked<ElaboratedEnum>,
        rem_variants: Vec<bool>,
    },
}

impl MatchCoverage {
    fn clear(&mut self) {
        match self {
            MatchCoverage::Bool { rem_false, rem_true } => {
                *rem_false = false;
                *rem_true = false;
            }
            MatchCoverage::Int { rem_range } => *rem_range = ClosedMultiRange::EMPTY,
            MatchCoverage::Enum {
                target_ty: _,
                rem_variants,
            } => {
                rem_variants.clear();
            }
        }
    }
}

fn join_block_ends_sequence(first: BlockEnd, second: BlockEnd) -> BlockEnd {
    match first.into_flags() {
        BlockEndFlags::Unreachable => BlockEnd::Unreachable,
        BlockEndFlags::Normal {
            certain_exit: first_certain_exit,
            mask: first_mask,
        } => {
            if first_certain_exit {
                first
            } else {
                match second.into_flags() {
                    BlockEndFlags::Unreachable => first,
                    BlockEndFlags::Normal {
                        certain_exit: second_certain_exit,
                        mask: second_mask,
                    } => BlockEnd::from_flags(BlockEndFlags::Normal {
                        certain_exit: second_certain_exit,
                        mask: first_mask | second_mask,
                    }),
                }
            }
        }
    }
}

fn join_block_ends_branches(ends: &[BlockEnd]) -> BlockEnd {
    let mut all_certain = true;
    let mut any_mask = ExitMask::<0>::default();
    let mut any_non_unreachable = false;

    for &end in ends {
        match end.into_flags() {
            BlockEndFlags::Unreachable => {
                // ignore unreachable branches, it's as if they don't exist
            }
            BlockEndFlags::Normal { certain_exit, mask } => {
                all_certain &= certain_exit;
                any_mask = any_mask | mask;
                any_non_unreachable = true;
            }
        }
    }

    if any_non_unreachable {
        BlockEnd::from_flags(BlockEndFlags::Normal {
            certain_exit: all_certain,
            mask: any_mask,
        })
    } else {
        BlockEnd::Unreachable
    }
}

fn build_ir_if_statement(
    large: &mut IrLargeArena,
    condition: IrExpression,
    then_block: Option<IrBlock>,
    else_block: Option<IrBlock>,
) -> Option<Either<IrBlock, IrIfStatement>> {
    // discard empty blocks
    let then_block = then_block.filter(|b| !b.statements.is_empty());
    let else_block = else_block.filter(|b| !b.statements.is_empty());

    // simplify constant condition
    if let IrExpression::Bool(condition) = condition {
        let block = if condition { then_block } else { else_block };
        return block.map(Either::Left);
    }

    // simplify if/else setup
    let res = match (then_block, else_block) {
        // both empty, emit nothing
        (None, None) => None,
        // only then, drop else
        (Some(then_block), None) => Some(IrIfStatement {
            condition,
            then_block,
            else_block: None,
        }),
        // only else, drop then and invert condition
        (None, Some(else_block)) => {
            let inverted_cond = large.push_expr(IrExpressionLarge::BoolNot(condition));
            Some(IrIfStatement {
                condition: inverted_cond,
                then_block: else_block,
                else_block: None,
            })
        }
        // both, emit the full if
        (Some(then_block), Some(else_block)) => Some(IrIfStatement {
            condition,
            then_block,
            else_block: Some(else_block),
        }),
    };

    res.map(Either::Right)
}

fn build_ir_int_in_range(large: &mut IrLargeArena, value: &IrExpression, range: Range<BigInt>) -> Option<IrExpression> {
    range.assert_valid();
    let Range { start, end } = range;

    let cond_start = start.map(|start| {
        large.push_expr(IrExpressionLarge::IntCompare(
            IrIntCompareOp::Lte,
            IrExpression::Int(start),
            value.clone(),
        ))
    });
    let cond_end = end.map(|end| {
        large.push_expr(IrExpressionLarge::IntCompare(
            IrIntCompareOp::Lt,
            value.clone(),
            IrExpression::Int(end),
        ))
    });

    match (cond_start, cond_end) {
        (None, None) => None,
        (Some(cond), None) | (None, Some(cond)) => Some(cond),
        (Some(cond_start), Some(cond_end)) => {
            Some(large.push_expr(IrExpressionLarge::BoolBinary(IrBoolBinaryOp::And, cond_start, cond_end)))
        }
    }
}
