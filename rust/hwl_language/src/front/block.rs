use crate::front::assignment::store_ir_expression_in_new_variable;
use crate::front::check::{
    check_type_contains_compile_value, check_type_contains_value, check_type_is_bool_compile, TypeContainsReason,
};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::flow::{Flow, FlowHardware};
use crate::front::flow::{FlowKind, VariableInfo};
use crate::front::implication::ValueWithImplications;
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::{HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::{
    IrBlock, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIfStatement, IrIntCompareOp, IrStatement,
};
use crate::syntax::ast::{
    Block, BlockStatement, BlockStatementKind, ConstBlock, ExpressionKind, ExtraItem, ExtraList, ForStatement,
    Identifier, IfCondBlockPair, IfStatement, MatchBranch, MatchPattern, MatchStatement, MaybeIdentifier,
    ReturnStatement, Spanned, VariableDeclaration, WhileStatement,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::iter::IterExt;
use crate::util::{result_pair, ResultExt};
use annotate_snippets::Level;
use itertools::{zip_eq, Itertools};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
#[must_use]
pub enum BlockEnd<S = BlockEndStopping> {
    Normal,
    Stopping(S),
}

#[derive(Debug)]
pub enum BlockEndStopping {
    FunctionReturn(BlockEndReturn),
    LoopBreak(Span),
    LoopContinue(Span),
}

#[derive(Debug)]
pub struct BlockEndReturn {
    pub span_keyword: Span,
    pub value: Option<Spanned<Value>>,
}

impl<S> BlockEnd<S> {
    pub fn map_stopping<T>(self, f: impl FnOnce(S) -> T) -> BlockEnd<T> {
        match self {
            BlockEnd::Normal => BlockEnd::Normal,
            BlockEnd::Stopping(end) => BlockEnd::Stopping(f(end)),
        }
    }
}

impl BlockEnd<BlockEndStopping> {
    pub fn unwrap_normal_todo_in_conditional(self, diags: &Diagnostics, span_cond: Span) -> DiagResult<()> {
        match self {
            BlockEnd::Normal => Ok(()),
            BlockEnd::Stopping(end) => {
                let (span, kind) = match end {
                    BlockEndStopping::FunctionReturn(ret) => (ret.span_keyword, "return"),
                    BlockEndStopping::LoopBreak(span) => (span, "break"),
                    BlockEndStopping::LoopContinue(span) => (span, "continue"),
                };

                let diag = Diagnostic::new_todo(format!("{kind} in conditional statement with runtime condition"))
                    .add_error(span, "used here")
                    .add_info(span_cond, "runtime condition here")
                    .finish();
                Err(diags.report(diag))
            }
        }
    }

    pub fn unwrap_normal_or_return_in_function(self, diags: &Diagnostics) -> DiagResult<BlockEnd<BlockEndReturn>> {
        match self {
            BlockEnd::Normal => Ok(BlockEnd::Normal),
            BlockEnd::Stopping(end) => match end {
                BlockEndStopping::FunctionReturn(ret) => Ok(BlockEnd::Stopping(BlockEndReturn {
                    span_keyword: ret.span_keyword,
                    value: ret.value,
                })),
                BlockEndStopping::LoopBreak(span) => {
                    Err(diags.report_simple("break outside loop", span, "break statement here"))
                }
                BlockEndStopping::LoopContinue(span) => {
                    Err(diags.report_simple("continue outside loop", span, "continue statement here"))
                }
            },
        }
    }

    pub fn unwrap_outside_function_and_loop(self, diags: &Diagnostics) -> DiagResult<()> {
        match self {
            BlockEnd::Normal => Ok(()),
            BlockEnd::Stopping(end) => match end {
                BlockEndStopping::FunctionReturn(ret) => {
                    Err(diags.report_simple("return outside function", ret.span_keyword, "return statement here"))
                }
                BlockEndStopping::LoopBreak(span) => {
                    Err(diags.report_simple("break outside loop", span, "break statement here"))
                }
                BlockEndStopping::LoopContinue(span) => {
                    Err(diags.report_simple("continue outside loop", span, "continue statement here"))
                }
            },
        }
    }
}

enum BranchMatched {
    Yes(Option<(MaybeIdentifier, CompileValue)>),
    No,
}

impl BranchMatched {
    fn from_bool(b: bool) -> Self {
        if b {
            BranchMatched::Yes(None)
        } else {
            BranchMatched::No
        }
    }
}

enum PatternEqual {
    Bool(bool),
    Int(BigInt),
    String(Arc<String>),
}

type CheckedMatchPattern<'a> = MatchPattern<PatternEqual, IncRange<BigInt>, usize, Identifier>;

impl CompileItemContext<'_, '_> {
    pub fn elaborate_const_block(&mut self, scope: &Scope, flow: &mut impl Flow, block: &ConstBlock) -> DiagResult<()> {
        let diags = self.refs.diags;
        let &ConstBlock {
            span_keyword,
            ref block,
        } = block;

        let span = span_keyword.join(block.span);
        let mut flow_inner = flow.new_child_compile(span, "const block");

        let block_end = self.elaborate_block(scope, &mut flow_inner, None, block)?;
        block_end.unwrap_outside_function_and_loop(diags)?;

        Ok(())
    }

    pub fn elaborate_block(
        &mut self,
        scope_parent: &Scope,
        flow_parent: &mut impl Flow,
        expected_return_ty: Option<&Type>,
        block: &Block<BlockStatement>,
    ) -> DiagResult<BlockEnd> {
        let &Block { span, ref statements } = block;

        let mut scope = Scope::new_child(span, scope_parent);

        // this is actually static dispatch, but we can't easily express that
        match flow_parent.kind_mut() {
            FlowKind::Compile(flow_parent) => {
                let mut flow = flow_parent.new_child_scoped();
                self.elaborate_block_statements(&mut scope, &mut flow, expected_return_ty, statements)
            }
            FlowKind::Hardware(flow_parent) => {
                let mut flow = flow_parent.new_child_scoped();
                self.elaborate_block_statements(&mut scope, &mut flow, expected_return_ty, statements)
            }
        }
    }

    pub fn elaborate_block_statements(
        &mut self,
        scope: &mut Scope,
        flow: &mut impl Flow,
        expected_return_ty: Option<&Type>,
        statements: &[BlockStatement],
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;

        for stmt in statements {
            let stmt_span = stmt.span;
            let stmt_end = match &stmt.inner {
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
                        if let Some(ty) = &ty {
                            if let Some(init) = &init {
                                let reason = TypeContainsReason::Assignment {
                                    span_target: id.span(),
                                    span_target_ty: ty.span,
                                };
                                check_type_contains_value(diags, reason, &ty.inner, init.as_ref(), true, true)?;
                            }
                        }

                        // build variable
                        let info = VariableInfo { id, mutable, ty };
                        let var = flow.var_new(info);

                        // store initial value if there is one
                        //   for hardware values, also store them in an IR variable to avoid duplicate expressions
                        //   and to keep the generated RTL somewhat similar to the source
                        if let Some(init) = init {
                            let init = init.inner.try_map_hardware(|init_inner| {
                                let flow = flow.check_hardware(init.span, "hardware value")?;
                                store_ir_expression_in_new_variable(self.refs, flow, id, init_inner)
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
                BlockStatementKind::Block(inner_block) => {
                    self.elaborate_block(scope, flow, expected_return_ty, inner_block)?
                }
                BlockStatementKind::If(stmt) => {
                    let IfStatement {
                        initial_if,
                        else_ifs,
                        final_else,
                    } = stmt;

                    self.elaborate_if_statement(
                        scope,
                        flow,
                        expected_return_ty,
                        Some((initial_if, else_ifs)),
                        final_else,
                    )?
                }
                BlockStatementKind::Match(stmt) => {
                    let stmt = Spanned::new(stmt_span, stmt);
                    self.elaborate_match_statement(scope, flow, expected_return_ty, stmt)?
                }
                BlockStatementKind::While(stmt) => {
                    let stmt = Spanned::new(stmt_span, stmt);
                    self.elaborate_while_statement(scope, flow, expected_return_ty, stmt)?
                        .map_stopping(BlockEndStopping::FunctionReturn)
                }
                BlockStatementKind::For(stmt) => {
                    let stmt = Spanned::new(stmt_span, stmt);
                    self.elaborate_for_statement(scope, flow, expected_return_ty, stmt)?
                        .map_stopping(BlockEndStopping::FunctionReturn)
                }
                BlockStatementKind::Return(stmt) => {
                    let &ReturnStatement {
                        span_return: span_keyword,
                        ref value,
                    } = stmt;

                    let expected_return_ty = expected_return_ty.ok_or_else(|| {
                        diags.report_simple(
                            "return is not allowed outside a function body",
                            span_keyword,
                            "return statement here",
                        )
                    })?;
                    let value = value
                        .map(|value| self.eval_expression(scope, flow, expected_return_ty, value))
                        .transpose()?;

                    BlockEnd::Stopping(BlockEndStopping::FunctionReturn(BlockEndReturn { span_keyword, value }))
                }
                &BlockStatementKind::Break(span) => BlockEnd::Stopping(BlockEndStopping::LoopBreak(span)),
                &BlockStatementKind::Continue(span) => BlockEnd::Stopping(BlockEndStopping::LoopContinue(span)),
            };

            match stmt_end {
                BlockEnd::Normal => {}
                BlockEnd::Stopping(end) => return Ok(BlockEnd::Stopping(end)),
            }
        }

        Ok(BlockEnd::Normal)
    }

    fn elaborate_if_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_return_ty: Option<&Type>,
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
                        return self.elaborate_block(scope, flow, expected_return_ty, final_else);
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
        check_type_contains_value(
            diags,
            reason,
            &Type::Bool,
            cond.clone().map_inner(ValueWithImplications::into_value).as_ref(),
            false,
            false,
        )?;

        match cond.inner {
            // evaluate the if at compile-time
            Value::Compile(cond_eval) => {
                let cond_eval = match cond_eval {
                    CompileValue::Bool(b) => b,
                    _ => throw!(diags.report_internal_error(cond.span, "expected bool value")),
                };

                // only visit the selected branch
                if cond_eval {
                    self.elaborate_block(scope, flow, expected_return_ty, block)
                } else {
                    self.elaborate_if_statement(
                        scope,
                        flow,
                        expected_return_ty,
                        remaining_ifs.split_first(),
                        final_else,
                    )
                }
            }
            // evaluate the if in hardware, generating IR
            Value::Hardware(cond_value) => {
                let flow = flow.check_hardware(cond.span, "hardware value")?;
                let cond_domain = Spanned::new(cond.span, cond_value.value.domain);

                // lower then
                let then_flow = {
                    let mut then_flow_branch = flow.new_child_branch(cond_domain, cond_value.implications.if_true);
                    let end =
                        { self.elaborate_block(scope, &mut then_flow_branch.as_flow(), expected_return_ty, block)? };
                    end.unwrap_normal_todo_in_conditional(diags, cond.span)?;
                    Ok(then_flow_branch.finish())
                };

                // lower else
                let else_flow = {
                    let mut else_flow_branch = flow.new_child_branch(cond_domain, cond_value.implications.if_false);
                    self.elaborate_if_statement(
                        scope,
                        &mut else_flow_branch.as_flow(),
                        expected_return_ty,
                        remaining_ifs.split_first(),
                        final_else,
                    )
                    .and_then(|end| end.unwrap_normal_todo_in_conditional(diags, cond.span))
                    .map(|()| else_flow_branch.finish())
                };

                let then_flow = then_flow?;
                let else_flow = else_flow?;

                // join flows
                let blocks =
                    flow.join_child_branches(self.refs, &mut self.large, span_if, vec![then_flow, else_flow])?;
                assert_eq!(blocks.len(), 2);
                let (then_block, else_block) = blocks.into_iter().collect_tuple().unwrap();

                // build the if statement
                let ir_if = IrStatement::If(IrIfStatement {
                    condition: cond_value.value.expr,
                    then_block,
                    else_block: Some(else_block),
                });
                // TODO is this span correct?
                flow.push_ir_statement(Spanned::new(span_if, ir_if));

                Ok(BlockEnd::Normal)
            }
        }
    }

    fn elaborate_match_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_return_ty: Option<&Type>,
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
                    MatchPattern::Dummy => {
                        cover_all = true;
                        Ok(MatchPattern::Dummy)
                    }
                    &MatchPattern::Val(i) => {
                        cover_all = true;
                        Ok(MatchPattern::Val(i))
                    }
                    &MatchPattern::Equal(value) => {
                        // TODO support tuples, arrays, structs, enums (by value), all recursively
                        if let ExpressionKind::Dummy = self.refs.get_expr(value) {
                            cover_all = true;
                            Ok(MatchPattern::Dummy)
                        } else {
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
                                    ))
                                }
                            };

                            Ok(MatchPattern::Equal(pattern))
                        }
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
                                ))
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
                                ))
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
                expected_return_ty,
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
                    expected_return_ty,
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
        expected_return_ty: Option<&Type>,
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
                MatchPattern::Dummy => BranchMatched::Yes(None),
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
                        let var =
                            flow.var_new_immutable_init(declare_id, pattern_span, Ok(Value::Compile(declare_value)));
                        scope_inner.maybe_declare(
                            diags,
                            Ok(declare_id.spanned_str(self.refs.fixed.source)),
                            Ok(ScopedEntry::Named(NamedValue::Variable(var))),
                        );
                        &scope_inner
                    } else {
                        scope
                    };

                    return self.elaborate_block(scoped_used, flow, expected_return_ty, block);
                }
            }
        }

        // we should never get here, we already checked that all cases are handled
        Err(diags.report_internal_error(stmt_span, "reached end of match statement"))
    }

    fn elaborate_match_statement_hardware(
        &mut self,
        scope_parent: &Scope,
        flow: &mut FlowHardware,
        expected_return_ty: Option<&Type>,
        stmt_span: Span,
        target: Spanned<HardwareValue>,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<CheckedMatchPattern>,
    ) -> DiagResult<BlockEnd> {
        let diags = self.refs.diags;

        let mut if_branch_conditions = vec![];
        let mut if_branch_flows = vec![];

        for (branch, pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch {
                pattern: pattern_raw,
                block,
            } = branch;
            let pattern_span = pattern_raw.span;
            let large = &mut self.large;

            // evaluate the pattern as a boolean condition and collect the variable to declare if any
            let (cond, declare): (Option<IrExpression>, Option<(MaybeIdentifier, HardwareValue)>) = match pattern {
                MatchPattern::Dummy => (None, None),
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
            // TODO push implications for integer ranges
            let target_domain = Spanned::new(target.span, target.inner.domain);
            let mut flow_branch = flow.new_child_branch(target_domain, vec![]);
            let mut flow_branch_flow = flow_branch.as_flow();

            let mut scope_inner = Scope::new_child(pattern_span.join(block.span), scope_parent);
            if let Some((declare_id, declare_value)) = declare {
                let var = flow_branch_flow.var_new_immutable_init(
                    declare_id,
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
            let end = self.elaborate_block(&scope_inner, &mut flow_branch_flow, expected_return_ty, block)?;
            end.unwrap_normal_todo_in_conditional(diags, stmt_span)?;

            // build the if stack
            let (cond, fully_covered) = match cond {
                Some(cond) => (cond, false),
                None => (IrExpression::Bool(true), true),
            };
            if_branch_conditions.push(cond);
            if_branch_flows.push(flow_branch.finish());
            if fully_covered {
                break;
            }
        }

        // merge flows
        assert_eq!(if_branch_conditions.len(), if_branch_flows.len());
        let if_branch_blocks = flow.join_child_branches(self.refs, &mut self.large, stmt_span, if_branch_flows)?;

        // build complete if chain
        let mut else_ir_block = None;
        for (curr_cond, curr_ir_block) in zip_eq(if_branch_conditions, if_branch_blocks) {
            let else_next = match else_ir_block {
                Some(else_ir_block) => {
                    // build if statement
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

        Ok(BlockEnd::Normal)
    }

    fn elaborate_while_statement(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expected_return_ty: Option<&Type>,
        stmt: Spanned<&WhileStatement>,
    ) -> DiagResult<BlockEnd<BlockEndReturn>> {
        let &WhileStatement {
            span_keyword,
            cond,
            ref body,
        } = stmt.inner;
        let diags = self.refs.diags;

        loop {
            self.refs.check_should_stop(span_keyword)?;

            // eval cond
            let cond = self.eval_expression_as_compile(scope, flow, &Type::Bool, cond, "while loop condition")?;

            let reason = TypeContainsReason::WhileCondition(span_keyword);
            check_type_contains_compile_value(diags, reason, &Type::Bool, cond.as_ref(), false)?;
            let cond = match &cond.inner {
                &CompileValue::Bool(b) => b,
                _ => throw!(diags.report_internal_error(cond.span, "expected bool, should have been checked already")),
            };

            // check cond
            if !cond {
                break;
            }

            // visit body
            let end = self.elaborate_block(scope, flow, expected_return_ty, body)?;

            // handle end
            match end {
                BlockEnd::Normal => {}
                BlockEnd::Stopping(end) => match end {
                    BlockEndStopping::FunctionReturn(ret) => {
                        return Ok(BlockEnd::Stopping(ret));
                    }
                    BlockEndStopping::LoopBreak(_span) => break,
                    BlockEndStopping::LoopContinue(_span) => continue,
                },
            }
        }

        Ok(BlockEnd::Normal)
    }

    // TODO code reuse between this and module
    fn elaborate_for_statement(
        &mut self,
        scope_parent: &Scope,
        flow: &mut impl Flow,
        expected_return_ty: Option<&Type>,
        stmt: Spanned<&ForStatement<BlockStatement>>,
    ) -> DiagResult<BlockEnd<BlockEndReturn>> {
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
            id: index_id,
            mutable: false,
            ty: None,
        });
        let mut scope_index = Scope::new_child(stmt.span, scope_parent);
        scope_index.maybe_declare(
            diags,
            Ok(index_id.spanned_str(self.refs.fixed.source)),
            Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
        );

        // loop
        for index_value in iter {
            self.refs.check_should_stop(span_keyword)?;
            let index_value = index_value.to_maybe_compile(&mut self.large);

            // typecheck index
            if let Some(index_ty) = &index_ty {
                let curr_spanned = Spanned {
                    span: stmt.inner.iter.span,
                    inner: &index_value,
                };
                let reason = TypeContainsReason::ForIndexType(index_ty.span);
                check_type_contains_value(diags, reason, &index_ty.inner, curr_spanned, false, true)?;
            }

            // set index and run body
            flow.var_set(index_var, span_keyword, Ok(index_value));
            let body_end = self.elaborate_block(&scope_index, flow, expected_return_ty, body)?;

            // handle possible loop termination
            match body_end {
                BlockEnd::Normal => {}
                BlockEnd::Stopping(end) => match end {
                    BlockEndStopping::FunctionReturn(end) => return Ok(BlockEnd::Stopping(end)),
                    BlockEndStopping::LoopBreak(_span) => break,
                    BlockEndStopping::LoopContinue(_span) => continue,
                },
            }
        }

        Ok(BlockEnd::Normal)
    }

    pub fn compile_elaborate_extra_list<'a, F: Flow, I>(
        &mut self,
        scope: &mut Scope,
        flow: &mut F,
        list: &'a ExtraList<I>,
        f: &mut impl FnMut(&mut Self, &mut Scope, &mut F, &'a I) -> DiagResult<()>,
    ) -> DiagResult<()> {
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

            if cond {
                Ok(Some(block))
            } else {
                Ok(None)
            }
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
