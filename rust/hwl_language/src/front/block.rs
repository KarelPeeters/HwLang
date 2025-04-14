use crate::front::assignment::store_ir_expression_in_new_variable;
use crate::front::check::{check_type_contains_compile_value, check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileItemContext, VariableInfo};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::expression::ForIterator;
use crate::front::ir::{IrExpression, IrExpressionLarge, IrIfStatement, IrLargeArena, IrStatement, IrVariable};
use crate::front::misc::{DomainSignal, ScopedEntry, ValueDomain};
use crate::front::scope::Scope;
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{CompileValue, MaybeCompile, NamedValue};
use crate::front::variables::{merge_variable_branches, VariableValues};
use crate::syntax::ast::{
    Block, BlockStatement, BlockStatementKind, ForStatement, IfCondBlockPair, IfStatement, ReturnStatement, Spanned,
    SyncDomain, VariableDeclaration, WhileStatement,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::{result_pair, ResultExt};

// TODO move
// TODO rename to HardwareValue
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TypedIrExpression<T = HardwareType, E = IrExpression> {
    pub ty: T,
    pub domain: ValueDomain,
    pub expr: E,
}

impl<E> Typed for TypedIrExpression<HardwareType, E> {
    fn ty(&self) -> Type {
        self.ty.as_type()
    }
}

impl<T> TypedIrExpression<T, IrVariable> {
    pub fn to_general_expression(self) -> TypedIrExpression<T, IrExpression> {
        TypedIrExpression {
            ty: self.ty,
            domain: self.domain,
            expr: IrExpression::Variable(self.expr),
        }
    }
}

impl TypedIrExpression {
    pub fn soft_expand_to_type(self, large: &mut IrLargeArena, ty: &HardwareType) -> Self {
        match (&self.ty, ty) {
            (HardwareType::Int(expr_ty), HardwareType::Int(target)) => {
                if target != expr_ty && target.contains_range(expr_ty) {
                    TypedIrExpression {
                        ty: HardwareType::Int(target.clone()),
                        domain: self.domain,
                        expr: large.push_expr(IrExpressionLarge::ExpandIntRange(target.clone(), self.expr)),
                    }
                } else {
                    self
                }
            }
            _ => self,
        }
    }
}

#[derive(Debug)]
pub enum BlockDomain {
    CompileTime,
    Combinatorial,
    Clocked(Spanned<SyncDomain<DomainSignal>>),
}

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
    pub value: Option<Spanned<MaybeCompile<TypedIrExpression>>>,
}

impl BlockEnd<BlockEndStopping> {
    pub fn unwrap_normal_todo_in_if(self, diags: &Diagnostics, span_cond: Span) -> Result<(), ErrorGuaranteed> {
        match self {
            BlockEnd::Normal => Ok(()),
            BlockEnd::Stopping(end) => {
                let (span, kind) = match end {
                    BlockEndStopping::FunctionReturn(ret) => (ret.span_keyword, "return"),
                    BlockEndStopping::LoopBreak(span) => (span, "break"),
                    BlockEndStopping::LoopContinue(span) => (span, "continue"),
                };

                let diag = Diagnostic::new_todo(span, format!("{} in if statement with runtime condition", kind))
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

impl CompileItemContext<'_, '_> {
    pub fn elaborate_block<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        scope_parent: &Scope,
        vars: &mut VariableValues,
        block: &Block<BlockStatement>,
    ) -> Result<(C::Block, BlockEnd), ErrorGuaranteed> {
        let &Block { span, ref statements } = block;

        let mut scope = Scope::new_child(span, scope_parent);
        self.elaborate_block_statements(ctx, &mut scope, vars, statements)
    }

    pub fn elaborate_block_statements<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        scope: &mut Scope,
        vars: &mut VariableValues,
        statements: &[BlockStatement],
    ) -> Result<(C::Block, BlockEnd), ErrorGuaranteed> {
        let diags = self.refs.diags;
        let mut ctx_block = ctx.new_ir_block();

        for stmt in statements {
            match &stmt.inner {
                BlockStatementKind::CommonDeclaration(decl) => self.eval_and_declare_declaration(scope, vars, decl),
                BlockStatementKind::VariableDeclaration(decl) => {
                    let VariableDeclaration {
                        span: _,
                        mutable,
                        id,
                        ty,
                        init,
                    } = decl;
                    let mutable = *mutable;

                    // eval ty
                    let ty = ty
                        .as_ref()
                        .map(|ty| self.eval_expression_as_ty(scope, vars, ty))
                        .transpose();

                    // eval init
                    let init = ty.as_ref_ok().and_then(|ty| {
                        let init_expected_ty = ty.as_ref().map_or(&Type::Type, |ty| &ty.inner);
                        init.as_ref()
                            .map(|init| self.eval_expression(ctx, &mut ctx_block, scope, vars, init_expected_ty, init))
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
                        let info = VariableInfo {
                            id: id.clone(),
                            mutable,
                            ty,
                        };
                        let var = vars.var_new(&mut self.variables, info);

                        // store initial value if there is one
                        if let Some(init) = init {
                            let init = init.inner.try_map_other(|init| {
                                store_ir_expression_in_new_variable(diags, ctx, &mut ctx_block, id.clone(), init)
                                    .map(TypedIrExpression::to_general_expression)
                            })?;
                            vars.var_set(diags, var, decl.span, init)?;
                        }

                        Ok(ScopedEntry::Named(NamedValue::Variable(var)))
                    });

                    scope.maybe_declare(diags, id.as_ref(), entry);
                }
                BlockStatementKind::Assignment(stmt) => {
                    self.elaborate_assignment(ctx, &mut ctx_block, scope, vars, stmt)?;
                }
                BlockStatementKind::Expression(expr) => {
                    let _: Spanned<MaybeCompile<TypedIrExpression>> =
                        self.eval_expression(ctx, &mut ctx_block, scope, vars, &Type::Type, expr)?;
                }
                BlockStatementKind::Block(inner_block) => {
                    let (inner_block_ir, block_end) = self.elaborate_block(ctx, scope, vars, inner_block)?;

                    let inner_block_spanned = Spanned {
                        span: inner_block.span,
                        inner: inner_block_ir,
                    };
                    ctx.push_ir_statement_block(&mut ctx_block, inner_block_spanned);

                    match block_end {
                        BlockEnd::Normal => {}
                        BlockEnd::Stopping(end) => return Ok((ctx_block, BlockEnd::Stopping(end))),
                    }
                }
                BlockStatementKind::If(stmt_if) => {
                    let IfStatement {
                        initial_if,
                        else_ifs,
                        final_else,
                    } = stmt_if;

                    let block_end = self.elaborate_if_statement(
                        ctx,
                        &mut ctx_block,
                        scope,
                        vars,
                        Some((initial_if, else_ifs)),
                        final_else,
                    )?;
                    match block_end {
                        BlockEnd::Normal => {}
                        BlockEnd::Stopping(end) => return Ok((ctx_block, BlockEnd::Stopping(end))),
                    }
                }
                BlockStatementKind::While(stmt_while) => {
                    let &WhileStatement {
                        span_keyword,
                        ref cond,
                        ref body,
                    } = stmt_while;

                    loop {
                        self.refs.check_should_stop(span_keyword)?;

                        // eval cond
                        let cond = self.eval_expression_as_compile(scope, vars, cond, "while loop condition")?;

                        let reason = TypeContainsReason::WhileCondition(span_keyword);
                        check_type_contains_compile_value(diags, reason, &Type::Bool, cond.as_ref(), false)?;
                        let cond = match &cond.inner {
                            &CompileValue::Bool(b) => b,
                            _ => throw!(diags
                                .report_internal_error(cond.span, "expected bool, should have been checked already")),
                        };

                        // handle cond
                        if !cond {
                            break;
                        }

                        // visit body
                        let (body_ir, end) = self.elaborate_block(ctx, scope, vars, body)?;
                        let body_ir_spanned = Spanned {
                            span: body.span,
                            inner: body_ir,
                        };
                        ctx.push_ir_statement_block(&mut ctx_block, body_ir_spanned);

                        // handle end
                        match end {
                            BlockEnd::Normal => {}
                            BlockEnd::Stopping(end) => match end {
                                BlockEndStopping::FunctionReturn(ret) => {
                                    return Ok((ctx_block, BlockEnd::Stopping(BlockEndStopping::FunctionReturn(ret))));
                                }
                                BlockEndStopping::LoopBreak(_span) => break,
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
                    let (block, end) = self.elaborate_for_statement(ctx, ctx_block, scope, vars, stmt_for)?;
                    ctx_block = block;

                    match end {
                        BlockEnd::Normal => {}
                        BlockEnd::Stopping(ret) => {
                            return Ok((ctx_block, BlockEnd::Stopping(BlockEndStopping::FunctionReturn(ret))))
                        }
                    }
                }
                BlockStatementKind::Return(stmt) => {
                    let &ReturnStatement {
                        span_return: span_keyword,
                        ref value,
                    } = stmt;

                    // we don't use the return type for the expected type here,
                    //  checking happens in the function call, and expanding at the call expression
                    let value = value
                        .as_ref()
                        .map(|value| self.eval_expression(ctx, &mut ctx_block, scope, vars, &Type::Type, value))
                        .transpose()?;

                    let end =
                        BlockEnd::Stopping(BlockEndStopping::FunctionReturn(BlockEndReturn { span_keyword, value }));
                    return Ok((ctx_block, end));
                }
                &BlockStatementKind::Break(span) => {
                    let end = BlockEnd::Stopping(BlockEndStopping::LoopBreak(span));
                    return Ok((ctx_block, end));
                }
                &BlockStatementKind::Continue(span) => {
                    let end = BlockEnd::Stopping(BlockEndStopping::LoopContinue(span));
                    return Ok((ctx_block, end));
                }
            };
        }

        Ok((ctx_block, BlockEnd::Normal))
    }

    fn elaborate_if_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        ifs: Option<(&IfCondBlockPair<BlockStatement>, &[IfCondBlockPair<BlockStatement>])>,
        final_else: &Option<Block<BlockStatement>>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let (initial_if, remaining_ifs) = match ifs {
            Some(p) => p,
            None => {
                return match final_else {
                    None => Ok(BlockEnd::Normal),
                    Some(final_else) => {
                        let (final_else_ir, final_else_end) = self.elaborate_block(ctx, scope, vars, final_else)?;
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
            ref cond,
            ref block,
        } = initial_if;
        let cond = self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Bool, cond)?;

        let reason = TypeContainsReason::IfCondition(span_if);
        check_type_contains_value(
            diags,
            reason,
            &Type::Bool,
            cond.as_ref().map_inner(|cond| &cond.value),
            false,
            true,
        )?;

        match cond.inner.value {
            // evaluate the if at compile-time
            MaybeCompile::Compile(cond_eval) => {
                let cond_eval = match cond_eval {
                    CompileValue::Bool(b) => b,
                    _ => throw!(diags.report_internal_error(cond.span, "expected bool value")),
                };

                // only visit the selected branch
                if cond_eval {
                    let (block_ir, block_end) = self.elaborate_block(ctx, scope, vars, block)?;
                    let block_ir_spanned = Spanned {
                        span: block.span,
                        inner: block_ir,
                    };
                    ctx.push_ir_statement_block(ctx_block, block_ir_spanned);
                    Ok(block_end)
                } else {
                    self.elaborate_if_statement(ctx, ctx_block, scope, vars, remaining_ifs.split_first(), final_else)
                }
            }
            // evaluate the if at runtime, generating IR
            MaybeCompile::Other(cond_eval) => {
                // check condition domain
                // TODO extract this to a common function?
                match ctx.block_domain() {
                    BlockDomain::CompileTime => {
                        throw!(diags
                            .report_internal_error(cond.span, "non-compile-time condition in compile-time context"))
                    }
                    BlockDomain::Combinatorial => {}
                    BlockDomain::Clocked(block_domain) => {
                        let cond_domain = Spanned {
                            span: cond.span,
                            inner: &cond_eval.domain,
                        };
                        let block_domain = block_domain.as_ref().map_inner(|&d| ValueDomain::Sync(d));
                        self.check_valid_domain_crossing(
                            cond.span,
                            block_domain.as_ref(),
                            cond_domain,
                            "condition used in clocked block",
                        )?;
                    }
                };

                // record condition domain
                let cond_domain = Spanned {
                    span: cond.span,
                    inner: cond_eval.domain,
                };

                let (mut then_ir, then_vars, then_end, mut else_ir, else_vars, else_end) =
                    ctx.with_condition_domain(diags, cond_domain, |ctx| {
                        // lower then
                        let mut then_vars = VariableValues::new_child(vars);
                        let (then_ir, then_end) =
                            ctx.with_implications(diags, cond.inner.implications.if_true, |ctx_inner| {
                                self.elaborate_block(ctx_inner, scope, &mut then_vars, block)
                            })?;

                        // lower else
                        let mut else_ir = ctx.new_ir_block();
                        let mut else_vars = VariableValues::new_child(vars);
                        let else_end = ctx.with_implications(diags, cond.inner.implications.if_false, |ctx_inner| {
                            self.elaborate_if_statement(
                                ctx_inner,
                                &mut else_ir,
                                scope,
                                &mut else_vars,
                                remaining_ifs.split_first(),
                                final_else,
                            )
                        })?;

                        Ok((then_ir, then_vars, then_end, else_ir, else_vars, else_end))
                    })?;

                let then_end_err = then_end.unwrap_normal_todo_in_if(diags, cond.span);
                let else_end_err = else_end.unwrap_normal_todo_in_if(diags, cond.span);
                then_end_err?;
                else_end_err?;

                merge_variable_branches(
                    diags,
                    ctx,
                    &mut self.large,
                    &self.variables,
                    vars,
                    span_if,
                    &mut then_ir,
                    then_vars.into_content(),
                    &mut else_ir,
                    else_vars.into_content(),
                )?;

                // actually record the if statement
                let ir_if = IrStatement::If(IrIfStatement {
                    condition: cond_eval.expr,
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

    fn elaborate_for_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        mut result_block: C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        stmt: Spanned<&ForStatement<BlockStatement>>,
    ) -> Result<(C::Block, BlockEnd<BlockEndReturn>), ErrorGuaranteed> {
        let ctx_block = &mut result_block;

        let ForStatement {
            span_keyword: _,
            index: _,
            index_ty,
            iter,
            body: _,
        } = stmt.inner;

        let index_ty = index_ty
            .as_ref()
            .map(|index_ty| self.eval_expression_as_ty(scope, vars, index_ty))
            .transpose();
        let iter = self.eval_expression_as_for_iterator(ctx, ctx_block, scope, vars, iter);

        let index_ty = index_ty?;
        let iter = iter?;

        // TODO (deterministic!) timeout?
        let end = self.run_for_statement(ctx, ctx_block, scope, vars, stmt, index_ty, iter)?;
        Ok((result_block, end))
    }

    // TODO code reuse between this and module?
    fn run_for_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope_parent: &Scope,
        vars: &mut VariableValues,
        stmt: Spanned<&ForStatement<BlockStatement>>,
        index_ty: Option<Spanned<Type>>,
        iter: ForIterator,
    ) -> Result<BlockEnd<BlockEndReturn>, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let &ForStatement {
            span_keyword,
            index: ref index_id,
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
            let index_var =
                vars.var_new_immutable_init(&mut self.variables, index_id.clone(), span_keyword, index_value);

            let mut scope_index = Scope::new_child(stmt.span, scope_parent);
            scope_index.maybe_declare(
                diags,
                index_id.as_ref(),
                Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
            );

            // run body
            let (body_block, body_end) = self.elaborate_block(ctx, &scope_index, vars, body)?;

            let body_block_spanned = Spanned {
                span: body.span,
                inner: body_block,
            };
            ctx.push_ir_statement_block(ctx_block, body_block_spanned);

            // handle possible termination
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
}
