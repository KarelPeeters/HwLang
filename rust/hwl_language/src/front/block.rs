use crate::front::assignment::store_ir_expression_in_new_variable;
use crate::front::check::{
    check_type_contains_compile_value, check_type_contains_value, check_type_is_bool_compile, TypeContainsReason,
};
use crate::front::compile::{CompileItemContext, VariableInfo};
use crate::front::context::{CompileTimeExpressionContext, ExpressionContext};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::domain::{BlockDomain, ValueDomain};
use crate::front::expression::{ForIterator, ValueWithImplications};
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::{HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::front::variables::{merge_variable_branches, VariableValues};
use crate::mid::ir::{IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIfStatement, IrIntCompareOp, IrStatement};
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

impl BlockEnd<BlockEndStopping> {
    pub fn unwrap_normal_todo_in_conditional(
        self,
        diags: &Diagnostics,
        span_cond: Span,
    ) -> Result<(), ErrorGuaranteed> {
        match self {
            BlockEnd::Normal => Ok(()),
            BlockEnd::Stopping(end) => {
                let (span, kind) = match end {
                    BlockEndStopping::FunctionReturn(ret) => (ret.span_keyword, "return"),
                    BlockEndStopping::LoopBreak(span) => (span, "break"),
                    BlockEndStopping::LoopContinue(span) => (span, "continue"),
                };

                let diag = Diagnostic::new_todo(format!("{} in conditional statement with runtime condition", kind))
                    .add_error(span, "used here")
                    .add_info(span_cond, "runtime condition here")
                    .finish();
                Err(diags.report(diag))
            }
        }
    }

    pub fn unwrap_normal_or_return_in_function(
        self,
        diags: &Diagnostics,
    ) -> Result<BlockEnd<BlockEndReturn>, ErrorGuaranteed> {
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

    pub fn unwrap_outside_function_and_loop(self, diags: &Diagnostics) -> Result<(), ErrorGuaranteed> {
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
    pub fn elaborate_const_block(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        block: &ConstBlock,
    ) -> Result<(), ErrorGuaranteed> {
        // TODO assignments in this block should be not allowed to leak outside
        //   we can fix this generally on the next scope/vars refactor,
        //   if we provide a "child with immutable view on parent" child mode
        let diags = self.refs.diags;
        let &ConstBlock {
            span_keyword,
            ref block,
        } = block;

        let mut ctx = CompileTimeExpressionContext {
            span: span_keyword.join(block.span),
            reason: "const block".to_owned(),
        };
        let ((), block_end) = self.elaborate_block_raw(&mut ctx, scope, vars, None, block)?;
        block_end.unwrap_outside_function_and_loop(diags)?;

        Ok(())
    }

    pub fn elaborate_and_push_block<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope_parent: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        block: &Block<BlockStatement>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let (ctx_block_inner, block_end) =
            self.elaborate_block_raw(ctx, scope_parent, vars, expected_return_ty, block)?;

        let block_ir_spanned = Spanned {
            span: block.span,
            inner: ctx_block_inner,
        };
        ctx.push_ir_statement_block(ctx_block, block_ir_spanned);

        Ok(block_end)
    }

    // TODO rename back to elaborate_block
    pub fn elaborate_block_raw<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        scope_parent: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        block: &Block<BlockStatement>,
    ) -> Result<(C::Block, BlockEnd), ErrorGuaranteed> {
        let &Block { span, ref statements } = block;

        let mut scope = Scope::new_child(span, scope_parent);
        self.elaborate_block_statements(ctx, &mut scope, vars, expected_return_ty, statements)
    }

    pub fn elaborate_block_statements<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        scope: &mut Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        statements: &[BlockStatement],
    ) -> Result<(C::Block, BlockEnd), ErrorGuaranteed> {
        let diags = self.refs.diags;
        let mut ctx_block = ctx.new_ir_block();

        for stmt in statements {
            let stmt_end = match &stmt.inner {
                BlockStatementKind::CommonDeclaration(decl) => {
                    self.eval_and_declare_declaration(scope, vars, decl);
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
                    let ty = ty.map(|ty| self.eval_expression_as_ty(scope, vars, ty)).transpose();

                    // eval init
                    let init = ty.as_ref_ok().and_then(|ty| {
                        let init_expected_ty = ty.as_ref().map_or(&Type::Any, |ty| &ty.inner);
                        init.map(|init| self.eval_expression(ctx, &mut ctx_block, scope, vars, init_expected_ty, init))
                            .transpose()
                    });

                    let entry = result_pair(ty, init).and_then(|(ty, init)| {
                        // check init fits in type
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
                        let var = vars.var_new(&mut self.variables, info);

                        // store initial value if there is one
                        if let Some(init) = init {
                            let init = init.inner.try_map_other(|init| {
                                store_ir_expression_in_new_variable(self.refs, ctx, &mut ctx_block, id, init)
                                    .map(HardwareValue::to_general_expression)
                            })?;
                            vars.var_set(diags, var, decl.span, init)?;
                        }

                        Ok(ScopedEntry::Named(NamedValue::Variable(var)))
                    });

                    let id = Ok(id.spanned_str(self.refs.fixed.source));
                    scope.maybe_declare(diags, id, entry);
                    BlockEnd::Normal
                }
                BlockStatementKind::Assignment(stmt) => {
                    self.elaborate_assignment(ctx, &mut ctx_block, scope, vars, stmt)?;
                    BlockEnd::Normal
                }
                &BlockStatementKind::Expression(expr) => {
                    let _: Spanned<Value> = self.eval_expression(ctx, &mut ctx_block, scope, vars, &Type::Any, expr)?;
                    BlockEnd::Normal
                }
                BlockStatementKind::Block(inner_block) => {
                    self.elaborate_and_push_block(ctx, &mut ctx_block, scope, vars, expected_return_ty, inner_block)?
                }
                BlockStatementKind::If(stmt_if) => {
                    let IfStatement {
                        initial_if,
                        else_ifs,
                        final_else,
                    } = stmt_if;

                    self.elaborate_if_statement(
                        ctx,
                        &mut ctx_block,
                        scope,
                        vars,
                        expected_return_ty,
                        Some((initial_if, else_ifs)),
                        final_else,
                    )?
                }
                BlockStatementKind::Match(stmt_match) => self.elaborate_match_statement(
                    ctx,
                    &mut ctx_block,
                    scope,
                    vars,
                    expected_return_ty,
                    stmt.span,
                    stmt_match,
                )?,
                BlockStatementKind::While(stmt_while) => {
                    let &WhileStatement {
                        span_keyword,
                        cond,
                        ref body,
                    } = stmt_while;

                    loop {
                        self.refs.check_should_stop(span_keyword)?;

                        // eval cond
                        let cond =
                            self.eval_expression_as_compile(scope, vars, &Type::Bool, cond, "while loop condition")?;

                        let reason = TypeContainsReason::WhileCondition(span_keyword);
                        check_type_contains_compile_value(diags, reason, &Type::Bool, cond.as_ref(), false)?;
                        let cond = match &cond.inner {
                            &CompileValue::Bool(b) => b,
                            _ => throw!(diags
                                .report_internal_error(cond.span, "expected bool, should have been checked already")),
                        };

                        // check cond
                        if !cond {
                            break BlockEnd::Normal;
                        }

                        // visit body
                        let end =
                            self.elaborate_and_push_block(ctx, &mut ctx_block, scope, vars, expected_return_ty, body)?;

                        // handle end
                        match end {
                            BlockEnd::Normal => {}
                            BlockEnd::Stopping(end) => match end {
                                BlockEndStopping::FunctionReturn(ret) => {
                                    break BlockEnd::Stopping(BlockEndStopping::FunctionReturn(ret));
                                }
                                BlockEndStopping::LoopBreak(_span) => break BlockEnd::Normal,
                                BlockEndStopping::LoopContinue(_span) => continue,
                            },
                        }
                    }
                }
                BlockStatementKind::For(stmt_for) => {
                    let stmt_for = Spanned {
                        span: stmt.span,
                        inner: stmt_for,
                    };
                    let (block, end) =
                        self.elaborate_for_statement(ctx, ctx_block, scope, vars, expected_return_ty, stmt_for)?;
                    ctx_block = block;

                    match end {
                        BlockEnd::Normal => BlockEnd::Normal,
                        BlockEnd::Stopping(ret) => BlockEnd::Stopping(BlockEndStopping::FunctionReturn(ret)),
                    }
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
                        .map(|value| self.eval_expression(ctx, &mut ctx_block, scope, vars, expected_return_ty, value))
                        .transpose()?;

                    BlockEnd::Stopping(BlockEndStopping::FunctionReturn(BlockEndReturn { span_keyword, value }))
                }
                &BlockStatementKind::Break(span) => BlockEnd::Stopping(BlockEndStopping::LoopBreak(span)),
                &BlockStatementKind::Continue(span) => BlockEnd::Stopping(BlockEndStopping::LoopContinue(span)),
            };

            match stmt_end {
                BlockEnd::Normal => {}
                BlockEnd::Stopping(end) => return Ok((ctx_block, BlockEnd::Stopping(end))),
            }
        }

        Ok((ctx_block, BlockEnd::Normal))
    }

    fn elaborate_if_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        ifs: Option<(
            &IfCondBlockPair<Block<BlockStatement>>,
            &[IfCondBlockPair<Block<BlockStatement>>],
        )>,
        final_else: &Option<Block<BlockStatement>>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let (initial_if, remaining_ifs) = match ifs {
            Some(p) => p,
            None => {
                return match final_else {
                    None => Ok(BlockEnd::Normal),
                    Some(final_else) => {
                        let (final_else_ir, final_else_end) =
                            self.elaborate_block_raw(ctx, scope, vars, expected_return_ty, final_else)?;
                        let final_else_spanned = Spanned {
                            span: final_else.span,
                            inner: final_else_ir,
                        };
                        ctx.push_ir_statement_block(ctx_block, final_else_spanned);
                        Ok(final_else_end)
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
        let cond = self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Bool, cond)?;

        let reason = TypeContainsReason::IfCondition(span_if);
        check_type_contains_value(
            diags,
            reason,
            &Type::Bool,
            cond.as_ref().map_inner(ValueWithImplications::value_cloned).as_ref(),
            false,
            true,
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
                    self.elaborate_and_push_block(ctx, ctx_block, scope, vars, expected_return_ty, block)
                } else {
                    self.elaborate_if_statement(
                        ctx,
                        ctx_block,
                        scope,
                        vars,
                        expected_return_ty,
                        remaining_ifs.split_first(),
                        final_else,
                    )
                }
            }
            // evaluate the if at runtime, generating IR
            Value::Hardware(cond_value) => {
                // check condition domain
                let cond_domain = Spanned {
                    span: cond.span,
                    inner: cond_value.value.domain,
                };
                // TODO extract this to a common function?
                match ctx.block_domain() {
                    BlockDomain::CompileTime => {
                        throw!(diags
                            .report_internal_error(cond.span, "non-compile-time condition in compile-time context"))
                    }
                    BlockDomain::Combinatorial => {}
                    BlockDomain::Clocked(block_domain) => {
                        let block_domain = block_domain.map_inner(ValueDomain::Sync);
                        self.check_valid_domain_crossing(
                            cond.span,
                            block_domain,
                            cond_domain,
                            "condition used in clocked block",
                        )?;
                    }
                };

                // record condition domain
                let (mut then_ir, then_vars, then_end, mut else_ir, else_vars, else_end) =
                    ctx.with_condition_domain(diags, cond_domain, |ctx| {
                        // lower then
                        let mut then_vars = VariableValues::new_child(vars);
                        let (then_ir, then_end) =
                            ctx.with_implications(diags, cond_value.implications.if_true, |ctx_inner| {
                                self.elaborate_block_raw(ctx_inner, scope, &mut then_vars, expected_return_ty, block)
                            })?;

                        // lower else
                        let mut else_ir = ctx.new_ir_block();
                        let mut else_vars = VariableValues::new_child(vars);
                        let else_end = ctx.with_implications(diags, cond_value.implications.if_false, |ctx_inner| {
                            self.elaborate_if_statement(
                                ctx_inner,
                                &mut else_ir,
                                scope,
                                &mut else_vars,
                                expected_return_ty,
                                remaining_ifs.split_first(),
                                final_else,
                            )
                        })?;

                        Ok((then_ir, then_vars, then_end, else_ir, else_vars, else_end))
                    })?;

                let then_end_err = then_end.unwrap_normal_todo_in_conditional(diags, cond.span);
                let else_end_err = else_end.unwrap_normal_todo_in_conditional(diags, cond.span);
                then_end_err?;
                else_end_err?;

                merge_variable_branches(
                    self.refs,
                    ctx,
                    &mut self.large,
                    &self.variables,
                    vars,
                    span_if,
                    vec![
                        (&mut then_ir, then_vars.into_content()),
                        (&mut else_ir, else_vars.into_content()),
                    ],
                )?;

                // actually record the if statement
                let ir_if = IrStatement::If(IrIfStatement {
                    condition: cond_value.value.expr,
                    then_block: ctx.unwrap_ir_block(diags, span_if, then_ir)?,
                    else_block: Some(ctx.unwrap_ir_block(diags, span_if, else_ir)?),
                });
                // TODO is this span correct?
                ctx.push_ir_statement(
                    diags,
                    ctx_block,
                    Spanned {
                        span: span_if,
                        inner: ir_if,
                    },
                )?;

                Ok(BlockEnd::Normal)
            }
        }
    }

    fn elaborate_match_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        stmt_span: Span,
        stmt: &MatchStatement<Block<BlockStatement>>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let &MatchStatement {
            target,
            span_branches,
            ref branches,
        } = stmt;

        // eval target
        let target = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, target)?;
        let target_ty = target.inner.ty();

        // track pattern coverage
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
            .map(|branch| -> Result<CheckedMatchPattern, ErrorGuaranteed> {
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
                            let value = self.eval_expression_as_compile(scope, vars, &target_ty, value, reason)?;

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

                        let value = self.eval_expression_as_compile(scope, vars, &target_ty, value, reason)?;
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
                .footer(Level::Help, "add missing cases")
                .footer(
                    Level::Help,
                    "add a default case using `_` or `val _` to cover all remaining values",
                )
                .finish();
            return Err(diags.report(diag));
        }

        // evaluate match itself
        match target.inner {
            Value::Compile(target_inner) => self.elaborate_match_statement_compile(
                ctx,
                ctx_block,
                scope,
                vars,
                expected_return_ty,
                stmt_span,
                target_inner,
                branches,
                branch_patterns,
            ),
            Value::Hardware(target_inner) => {
                let domain = Spanned::new(target.span, target_inner.domain);
                ctx.with_condition_domain(diags, domain, |ctx| {
                    self.elaborate_match_statement_hardware(
                        ctx,
                        ctx_block,
                        scope,
                        vars,
                        expected_return_ty,
                        stmt_span,
                        target_inner,
                        branches,
                        branch_patterns,
                    )
                })
            }
        }
    }

    fn elaborate_match_statement_compile<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        stmt_span: Span,
        cond: CompileValue,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<CheckedMatchPattern>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let diags = self.refs.diags;

        for (branch, pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch {
                pattern: pattern_raw,
                block,
            } = branch;
            let pattern_span = pattern_raw.span;

            let matched: BranchMatched = match pattern {
                MatchPattern::Dummy => BranchMatched::Yes(None),
                MatchPattern::Val(id) => BranchMatched::Yes(Some((MaybeIdentifier::Identifier(id), cond.clone()))),
                MatchPattern::Equal(pattern) => {
                    let c = match (&pattern, &cond) {
                        (PatternEqual::Bool(p), CompileValue::Bool(v)) => p == v,
                        (PatternEqual::Int(p), CompileValue::Int(v)) => p == v,
                        (PatternEqual::String(p), CompileValue::String(v)) => p == v,
                        _ => return Err(diags.report_internal_error(pattern_span, "unexpected pattern/value")),
                    };

                    BranchMatched::from_bool(c)
                }
                MatchPattern::In(pattern) => match &cond {
                    CompileValue::Int(value) => BranchMatched::from_bool(pattern.contains(value)),
                    _ => return Err(diags.report_internal_error(pattern_span, "unexpected range/value")),
                },
                MatchPattern::EnumVariant(pattern_index, id_content) => match &cond {
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
                        let var = vars.var_new_immutable_init(
                            &mut self.variables,
                            declare_id,
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

                    return self.elaborate_and_push_block(ctx, ctx_block, scoped_used, vars, expected_return_ty, block);
                }
            }
        }

        // we should never get here, we already checked that all cases are handled
        Err(diags.report_internal_error(stmt_span, "reached end of match statement"))
    }

    fn elaborate_match_statement_hardware<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope_outer: &Scope,
        vars_outer: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        stmt_span: Span,
        target: HardwareValue,
        branches: &Vec<MatchBranch<Block<BlockStatement>>>,
        branch_patterns: Vec<CheckedMatchPattern>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let mut if_stack_cond_block = vec![];
        let mut if_stack_vars = vec![];

        for (branch, pattern) in zip_eq(branches, branch_patterns) {
            let MatchBranch {
                pattern: pattern_raw,
                block,
            } = branch;
            let pattern_span = pattern_raw.span;
            let large = &mut self.large;

            let (cond, declare): (Option<IrExpression>, Option<(MaybeIdentifier, HardwareValue)>) = match pattern {
                MatchPattern::Dummy => (None, None),
                MatchPattern::Val(id) => (None, Some((MaybeIdentifier::Identifier(id), target.clone()))),
                MatchPattern::Equal(pattern) => {
                    let cond = match (&target.ty, pattern) {
                        (HardwareType::Bool, PatternEqual::Bool(pattern)) => {
                            let cond_expr = target.expr.clone();
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
                                target.expr.clone(),
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
                            target.expr.clone(),
                        ))
                    });
                    let end_inc = end_inc.map(|end_inc| {
                        large.push_expr(IrExpressionLarge::IntCompare(
                            IrIntCompareOp::Lte,
                            target.expr.clone(),
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
                    let target_ty = match &target.ty {
                        HardwareType::Enum(cond_ty) => cond_ty,
                        _ => return Err(diags.report_internal_error(pattern_span, "unexpected hw enum/value")),
                    };

                    let info = self.refs.shared.elaboration_arenas.enum_info(target_ty.inner());
                    let info_hw = info.hw.as_ref_ok().unwrap();

                    let ty_content = &info_hw.content_types[pattern_index];

                    let target_tag = large.push_expr(IrExpressionLarge::TupleIndex {
                        base: target.expr.clone(),
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
                                base: target.expr.clone(),
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
                                domain: target.domain,
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

            let mut scope_inner = Scope::new_child(pattern_span.join(block.span), scope_outer);
            let mut vars_inner = VariableValues::new_child(vars_outer);

            let scoped_used = if let Some((declare_id, declare_value)) = declare {
                let var = vars_inner.var_new_immutable_init(
                    &mut self.variables,
                    declare_id,
                    pattern_span,
                    Ok(Value::Hardware(declare_value)),
                );
                scope_inner.maybe_declare(
                    diags,
                    Ok(declare_id.spanned_str(self.refs.fixed.source)),
                    Ok(ScopedEntry::Named(NamedValue::Variable(var))),
                );
                &scope_inner
            } else {
                scope_outer
            };

            let (ctx_block_inner, end) =
                self.elaborate_block_raw(ctx, scoped_used, &mut vars_inner, expected_return_ty, block)?;
            end.unwrap_normal_todo_in_conditional(diags, stmt_span)?;

            match cond {
                Some(cond) => {
                    if_stack_cond_block.push((cond, ctx_block_inner));
                    if_stack_vars.push(vars_inner);
                }
                None => {
                    if_stack_cond_block.push((IrExpression::Bool(true), ctx_block_inner));
                    if_stack_vars.push(vars_inner);
                    break;
                }
            }
        }

        // merge vars
        let merge_children = zip_eq(&mut if_stack_cond_block, if_stack_vars)
            .map(|((_, b), vars)| (b, vars.into_content()))
            .collect_vec();
        merge_variable_branches(
            self.refs,
            ctx,
            &mut self.large,
            &self.variables,
            vars_outer,
            stmt_span,
            merge_children,
        )?;

        // build complete if chain
        let mut else_inner = None;
        for (curr_cond, curr_block) in if_stack_cond_block.into_iter().rev() {
            let else_next = match else_inner {
                Some(else_inner) => {
                    let curr_block_ir = ctx.unwrap_ir_block(diags, stmt_span, curr_block)?;
                    let else_block_ir = ctx.unwrap_ir_block(diags, stmt_span, else_inner)?;

                    // build if statement
                    let if_stmt = IrIfStatement {
                        condition: curr_cond,
                        then_block: curr_block_ir,
                        else_block: Some(else_block_ir),
                    };
                    let mut if_block = ctx.new_ir_block();
                    ctx.push_ir_statement(diags, &mut if_block, Spanned::new(stmt_span, IrStatement::If(if_stmt)))?;
                    if_block
                }
                None => {
                    // this is the final branch, which means that the condition can be ignored
                    //   (this is easier to reason about for var merging and synthesis tools)
                    let _ = curr_cond;
                    curr_block
                }
            };
            else_inner = Some(else_next);
        }

        if let Some(else_inner) = else_inner {
            ctx.push_ir_statement_block(ctx_block, Spanned::new(stmt_span, else_inner));
        }

        Ok(BlockEnd::Normal)
    }

    fn elaborate_for_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        mut result_block: C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        stmt: Spanned<&ForStatement<BlockStatement>>,
    ) -> Result<(C::Block, BlockEnd<BlockEndReturn>), ErrorGuaranteed> {
        let ctx_block = &mut result_block;

        let &ForStatement {
            span_keyword: _,
            index: _,
            index_ty,
            iter,
            body: _,
        } = stmt.inner;

        let index_ty = index_ty
            .map(|index_ty| self.eval_expression_as_ty(scope, vars, index_ty))
            .transpose();
        let iter = self.eval_expression_as_for_iterator(ctx, ctx_block, scope, vars, iter);

        let index_ty = index_ty?;
        let iter = iter?;

        let end = self.run_for_statement(ctx, ctx_block, scope, vars, expected_return_ty, stmt, index_ty, iter)?;
        Ok((result_block, end))
    }

    // TODO code reuse between this and module?
    fn run_for_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope_parent: &Scope,
        vars: &mut VariableValues,
        expected_return_ty: Option<&Type>,
        stmt: Spanned<&ForStatement<BlockStatement>>,
        index_ty: Option<Spanned<Type>>,
        iter: ForIterator,
    ) -> Result<BlockEnd<BlockEndReturn>, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let &ForStatement {
            span_keyword,
            index: index_id,
            index_ty: _,
            iter: _,
            ref body,
        } = stmt.inner;

        // run the actual loop
        for index_value in iter {
            self.refs.check_should_stop(span_keyword)?;
            let index_value = index_value.to_maybe_compile(&mut self.large);

            // typecheck index
            // TODO we can also this this once at the start instead, but that's slightly less flexible
            if let Some(index_ty) = &index_ty {
                let curr_spanned = Spanned {
                    span: stmt.inner.iter.span,
                    inner: &index_value,
                };
                let reason = TypeContainsReason::ForIndexType(index_ty.span);
                check_type_contains_value(diags, reason, &index_ty.inner, curr_spanned, false, true)?;
            }

            // create scope and set index
            let index_var = vars.var_new_immutable_init(&mut self.variables, index_id, span_keyword, Ok(index_value));

            let mut scope_index = Scope::new_child(stmt.span, scope_parent);
            scope_index.maybe_declare(
                diags,
                Ok(index_id.spanned_str(self.refs.fixed.source)),
                Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
            );

            // run body
            let body_end =
                self.elaborate_and_push_block(ctx, ctx_block, &scope_index, vars, expected_return_ty, body)?;

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

    pub fn compile_elaborate_extra_list<'a, I>(
        &mut self,
        scope: &mut Scope,
        vars: &mut VariableValues,
        list: &'a ExtraList<I>,
        f: &mut impl FnMut(&mut Self, &mut Scope, &mut VariableValues, &'a I) -> Result<(), ErrorGuaranteed>,
    ) -> Result<(), ErrorGuaranteed> {
        let ExtraList { span: _, items } = list;
        for item in items {
            match item {
                ExtraItem::Inner(inner) => f(self, scope, vars, inner)?,
                ExtraItem::Declaration(decl) => {
                    self.eval_and_declare_declaration(scope, vars, decl);
                }
                ExtraItem::If(if_stmt) => {
                    if let Some(list_inner) = self.compile_if_statement_choose_block(scope, vars, if_stmt)? {
                        self.compile_elaborate_extra_list(scope, vars, list_inner, f)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn compile_if_statement_choose_block<'a, B>(
        &mut self,
        scope: &Scope,
        vars: &VariableValues,
        if_stmt: &'a IfStatement<B>,
    ) -> Result<Option<&'a B>, ErrorGuaranteed> {
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

            let mut vars_inner = VariableValues::new_child(vars);
            let cond = self.eval_expression_as_compile(
                scope,
                &mut vars_inner,
                &Type::Bool,
                cond,
                "compile-time if condition",
            )?;

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
