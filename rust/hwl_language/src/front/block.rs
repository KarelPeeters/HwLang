use crate::front::check::{check_type_contains_compile_value, check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileState, Variable, VariableInfo};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrAssignmentTarget, IrExpression, IrIfStatement, IrStatement, IrVariable, IrVariableInfo};
use crate::front::misc::{DomainSignal, ScopedEntry, ValueDomain};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{AssignmentTarget, CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{
    Assignment, Block, BlockStatement, BlockStatementKind, Expression, ForStatement, IfCondBlockPair, IfStatement,
    ReturnStatement, Spanned, SyncDomain, VariableDeclaration, WhileStatement,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::data::IndexMapExt;
use crate::util::{result_pair, ResultExt};
use indexmap::IndexMap;
use num_bigint::{BigInt, BigUint};
use num_traits::{CheckedSub, One};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TypedIrExpression<T = HardwareType, E = IrExpression> {
    pub ty: T,
    pub domain: ValueDomain,
    pub expr: E,
}

impl Typed for TypedIrExpression {
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

#[derive(Debug)]
pub enum BlockDomain {
    CompileTime,
    Combinatorial,
    Clocked(Spanned<SyncDomain<DomainSignal>>),
}

// TODO move
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MaybeAssignedValue {
    Assigned(AssignedValue),
    NotYetAssigned,
    PartiallyAssigned,
    FailedIfMerge(Span),
}

// TODO move
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AssignedValue {
    pub event: AssignmentEvent,
    pub value: MaybeCompile<TypedIrExpression>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AssignmentEvent {
    assigned_value_span: Span,
}

// TODO move
#[derive(Debug, Clone)]
pub struct VariableValues {
    map: Option<IndexMap<Variable, MaybeAssignedValue>>,
}

impl VariableValues {
    pub fn new() -> Self {
        Self {
            map: Some(IndexMap::new()),
        }
    }

    pub fn new_no_vars() -> Self {
        Self { map: None }
    }

    pub fn get(&self, diags: &Diagnostics, span_use: Span, var: Variable) -> Result<&AssignedValue, ErrorGuaranteed> {
        let map = self
            .map
            .as_ref()
            .ok_or_else(|| diags.report_internal_error(span_use, "variable are not allowed in this context"))?;
        let value = map
            .get(&var)
            .ok_or_else(|| diags.report_internal_error(span_use, "variable used here has not been declared"))?;

        match value {
            MaybeAssignedValue::Assigned(value) => Ok(value),
            MaybeAssignedValue::NotYetAssigned => Err(diags.report_simple(
                "variable has not yet been assigned a value",
                span_use,
                "variable used here",
            )),
            // TODO point to examples of assignment and non-assignment
            MaybeAssignedValue::PartiallyAssigned => Err(diags.report_simple(
                "variable has not yet been assigned a value in all preceding branches",
                span_use,
                "variable used here",
            )),
            &MaybeAssignedValue::FailedIfMerge(span_if) => {
                // TODO include more specific reason and/or hint
                //   in particular, point to the runtime condition and the if keyword,
                //   similar to the return/break/continue error message
                let diag = Diagnostic::new("variable value from different `if` branches could not be merged")
                    .add_error(span_use, "variable used here")
                    .add_info(span_if, "if that caused the merge failure here")
                    .finish();
                Err(diags.report(diag))
            }
        }
    }

    pub fn set(
        &mut self,
        diags: &Diagnostics,
        span_set: Span,
        var: Variable,
        value: MaybeAssignedValue,
    ) -> Result<(), ErrorGuaranteed> {
        match &mut self.map {
            None => Err(diags.report_internal_error(span_set, "variables are not allowed in this context")),
            Some(map) => {
                map.insert(var, value);
                Ok(())
            }
        }
    }
}

#[derive(Debug)]
pub enum BlockEnd<S = BlockEndStopping> {
    Normal(VariableValues),
    Stopping(S),
}

#[derive(Debug)]
pub enum BlockEndStopping {
    Return(BlockEndReturn),
    Break(Span, VariableValues),
    Continue(Span, VariableValues),
}

#[derive(Debug)]
pub struct BlockEndReturn {
    pub span_keyword: Span,
    pub value: Option<Spanned<MaybeCompile<TypedIrExpression>>>,
}

impl BlockEnd<BlockEndStopping> {
    pub fn unwrap_normal_todo_in_if(
        self,
        diags: &Diagnostics,
        span_cond: Span,
    ) -> Result<VariableValues, ErrorGuaranteed> {
        match self {
            BlockEnd::Normal(vars) => Ok(vars),
            BlockEnd::Stopping(end) => {
                let (span, kind) = match end {
                    BlockEndStopping::Return(ret) => (ret.span_keyword, "return"),
                    BlockEndStopping::Break(span, _vars) => (span, "break"),
                    BlockEndStopping::Continue(span, _vars) => (span, "continue"),
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
            BlockEnd::Normal(vars) => Ok(BlockEnd::Normal(vars)),
            BlockEnd::Stopping(end) => match end {
                BlockEndStopping::Return(ret) => Ok(BlockEnd::Stopping(BlockEndReturn {
                    span_keyword: ret.span_keyword,
                    value: ret.value,
                })),
                BlockEndStopping::Break(span, _vars) => {
                    Err(diags.report_simple("break outside loop", span, "break statement here"))
                }
                BlockEndStopping::Continue(span, _vars) => {
                    Err(diags.report_simple("continue outside loop", span, "continue statement here"))
                }
            },
        }
    }

    pub fn unwrap_normal_in_process(self, diags: &Diagnostics) -> Result<VariableValues, ErrorGuaranteed> {
        match self {
            BlockEnd::Normal(vars) => Ok(vars),
            BlockEnd::Stopping(end) => match end {
                BlockEndStopping::Return(ret) => {
                    Err(diags.report_simple("return outside function", ret.span_keyword, "return statement here"))
                }
                BlockEndStopping::Break(span, _vars) => {
                    Err(diags.report_simple("break outside loop", span, "break statement here"))
                }
                BlockEndStopping::Continue(span, _vars) => {
                    Err(diags.report_simple("continue outside loop", span, "continue statement here"))
                }
            },
        }
    }
}

impl CompileState<'_> {
    pub fn elaborate_block<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        mut vars: VariableValues,
        scope_parent: Scope,
        block: &Block<BlockStatement>,
    ) -> Result<(C::Block, BlockEnd), ErrorGuaranteed> {
        let diags = self.diags;
        let Block { span: _, statements } = block;

        let scope = self.scopes.new_child(scope_parent, block.span, Visibility::Private);
        let mut ctx_block = ctx.new_ir_block();

        for stmt in statements {
            match &stmt.inner {
                BlockStatementKind::ConstDeclaration(decl) => self.const_eval_and_declare(scope, decl),
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
                        .map(|ty| self.eval_expression_as_ty(scope, &vars, ty))
                        .transpose();

                    // eval init
                    let eval_expected_ty = ty
                        .as_ref_ok()
                        .ok()
                        .and_then(|t| t.as_ref().map(|t| &t.inner))
                        .unwrap_or(&Type::Any);
                    let init = init
                        .as_ref()
                        .map(|init| self.eval_expression(ctx, &mut ctx_block, scope, &vars, eval_expected_ty, init))
                        .transpose();

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
                        let variable = self.variables.push(info);

                        // store value
                        let assigned = match init {
                            None => MaybeAssignedValue::NotYetAssigned,
                            Some(init) => MaybeAssignedValue::Assigned(AssignedValue {
                                event: AssignmentEvent {
                                    assigned_value_span: init.span,
                                },
                                value: init.inner,
                            }),
                        };
                        vars.set(diags, id.span(), variable, assigned)?;

                        Ok(ScopedEntry::Direct(NamedValue::Variable(variable)))
                    });

                    self.scopes[scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(stmt) => {
                    let new_vars = self.elaborate_assignment(ctx, &mut ctx_block, vars, scope, stmt)?;
                    vars = new_vars;
                }
                BlockStatementKind::Expression(expr) => {
                    let _: Spanned<MaybeCompile<TypedIrExpression>> =
                        self.eval_expression(ctx, &mut ctx_block, scope, &vars, &Type::Type, expr)?;
                }
                BlockStatementKind::Block(inner_block) => {
                    let (inner_block_ir, block_end) = self.elaborate_block(ctx, vars, scope, inner_block)?;

                    let inner_block_spanned = Spanned {
                        span: inner_block.span,
                        inner: inner_block_ir,
                    };
                    ctx.push_ir_statement_block(&mut ctx_block, inner_block_spanned);

                    match block_end {
                        BlockEnd::Normal(new_vars) => vars = new_vars,
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
                        vars,
                        scope,
                        Some((initial_if, else_ifs)),
                        final_else,
                    )?;
                    match block_end {
                        BlockEnd::Normal(new_vars) => vars = new_vars,
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
                        // eval cond
                        let cond = self.eval_expression_as_compile(scope, &vars, cond, "while loop condition")?;

                        let reason = TypeContainsReason::WhileCondition { span_keyword };
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
                        let (body_ir, end) = self.elaborate_block(ctx, vars, scope, body)?;
                        let body_ir_spanned = Spanned {
                            span: body.span,
                            inner: body_ir,
                        };
                        ctx.push_ir_statement_block(&mut ctx_block, body_ir_spanned);

                        // handle end
                        match end {
                            BlockEnd::Normal(new_vars) => vars = new_vars,
                            BlockEnd::Stopping(end) => match end {
                                BlockEndStopping::Return(ret) => {
                                    return Ok((ctx_block, BlockEnd::Stopping(BlockEndStopping::Return(ret))));
                                }
                                BlockEndStopping::Break(_, new_vars) => {
                                    vars = new_vars;
                                    break;
                                }
                                BlockEndStopping::Continue(_, new_vars) => {
                                    vars = new_vars;
                                    continue;
                                }
                            },
                        }
                    }
                }
                BlockStatementKind::For(stmt_for) => {
                    let stmt_for = Spanned {
                        span: stmt.span,
                        inner: stmt_for,
                    };
                    let (block, end) = self.elaborate_for_statement(ctx, ctx_block, vars, scope, stmt_for)?;
                    ctx_block = block;

                    match end {
                        BlockEnd::Normal(new_vars) => vars = new_vars,
                        BlockEnd::Stopping(ret) => {
                            return Ok((ctx_block, BlockEnd::Stopping(BlockEndStopping::Return(ret))))
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
                        .map(|value| self.eval_expression(ctx, &mut ctx_block, scope, &vars, &Type::Type, value))
                        .transpose()?;

                    let end = BlockEnd::Stopping(BlockEndStopping::Return(BlockEndReturn { span_keyword, value }));
                    return Ok((ctx_block, end));
                }
                &BlockStatementKind::Break(span) => {
                    let end = BlockEnd::Stopping(BlockEndStopping::Break(span, vars));
                    return Ok((ctx_block, end));
                }
                &BlockStatementKind::Continue(span) => {
                    let end = BlockEnd::Stopping(BlockEndStopping::Continue(span, vars));
                    return Ok((ctx_block, end));
                }
            };
        }

        Ok((ctx_block, BlockEnd::Normal(vars)))
    }

    fn elaborate_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        mut vars: VariableValues,
        scope: Scope,
        stmt: &Assignment,
    ) -> Result<VariableValues, ErrorGuaranteed> {
        let diags = self.diags;
        let Assignment {
            span: _,
            op,
            target: target_expr,
            value,
        } = stmt;

        // TODO stop evaluating `left` twice, if there are side effects that is observably wrong
        //   instead, add a way to turn targets into their "read value" form
        let target = self.eval_expression_as_assign_target(scope, target_expr);

        let expected_ty = target
            .as_ref_ok()
            .ok()
            .and_then(|target| target.inner.ty(self).map(|ty| ty.inner))
            .unwrap_or(Type::Any);
        let value_right = self.eval_expression(ctx, ctx_block, scope, &vars, &expected_ty, value);

        let value = if let Some(op) = op.transpose() {
            let value_left = self.eval_expression(ctx, ctx_block, scope, &vars, &expected_ty, target_expr);

            result_pair(value_left, value_right).and_then(|(left, right)| {
                let result = self.eval_binary_expression(stmt.span, op, left, right)?;
                Ok(Spanned {
                    span: stmt.span,
                    inner: result,
                })
            })
        } else {
            value_right
        };

        let target = target?;
        let value = value?;

        let condition_domains = ctx.condition_domains();

        let (target_ty, target_domain, ir_target) = match target.inner {
            AssignmentTarget::Port(port) => {
                let info = &self.ports[port];
                let domain = ValueDomain::from_port_domain(info.domain.inner);
                (&info.ty, domain, IrAssignmentTarget::Port(info.ir))
            }
            AssignmentTarget::Wire(wire) => {
                let info = &self.wires[wire];
                let domain = info.domain.inner.clone();
                (&info.ty, domain, IrAssignmentTarget::Wire(info.ir))
            }
            AssignmentTarget::Register(reg) => {
                let info = &self.registers[reg];
                let domain = ValueDomain::Sync(info.domain.inner);
                (&info.ty, domain, IrAssignmentTarget::Register(self.registers[reg].ir))
            }
            AssignmentTarget::Variable(var) => {
                // variable assignments are handled separately, they are evaluated at compile-time as much as possible
                let VariableInfo { id, mutable, ty } = &mut self.variables[var];

                // check mutable
                if !*mutable {
                    let diag = Diagnostic::new("assignment to immutable variable")
                        .add_error(target.span, "variable assigned to here")
                        .add_info(id.span(), "variable declared as immutable here")
                        .finish();
                    return Err(diags.report(diag));
                }

                // check type
                if let Some(ty) = ty {
                    let reason = TypeContainsReason::Assignment {
                        span_target: target.span,
                        span_target_ty: ty.span,
                    };
                    check_type_contains_value(diags, reason, &ty.inner, value.as_ref(), true, true)?;
                }

                // save the value
                let assigned = AssignedValue {
                    event: AssignmentEvent {
                        assigned_value_span: value.span,
                    },
                    value: value.inner,
                };
                vars.set(diags, target.span, var, MaybeAssignedValue::Assigned(assigned))?;

                return Ok(vars);
            }
        };

        // check type
        let reason = TypeContainsReason::Assignment {
            span_target: target.span,
            span_target_ty: target_ty.span,
        };
        let check_ty =
            check_type_contains_value(diags, reason, &target_ty.inner.as_type(), value.as_ref(), true, false);

        // check domains
        let value_domain = value.inner.domain();
        // TODO better error messages with more explanation
        let target_domain = Spanned {
            span: target.span,
            inner: &target_domain,
        };
        let value_domain = Spanned {
            span: value.span,
            inner: value_domain,
        };

        let check_domains = match ctx.block_domain() {
            BlockDomain::CompileTime => {
                for d in [&target_domain, &value_domain] {
                    if d.inner != &ValueDomain::CompileTime {
                        throw!(diags.report_internal_error(d.span, "non-compile-time domain in compile-time context"))
                    }
                }
                Ok(())
            }
            BlockDomain::Combinatorial => {
                let mut check = self.check_valid_domain_crossing(
                    stmt.span,
                    target_domain,
                    value_domain,
                    "value to target in combinatorial block",
                );

                for condition_domain in condition_domains {
                    let c = self.check_valid_domain_crossing(
                        stmt.span,
                        target_domain,
                        condition_domain.as_ref(),
                        "condition to target in combinatorial block",
                    );
                    check = check.and(c);
                }

                check
            }
            BlockDomain::Clocked(block_domain) => {
                let block_domain = block_domain.as_ref().map_inner(|&d| ValueDomain::Sync(d));

                let check_target_domain = self.check_valid_domain_crossing(
                    stmt.span,
                    target_domain,
                    block_domain.as_ref(),
                    "clocked block to target",
                );
                let check_value_domain = self.check_valid_domain_crossing(
                    stmt.span,
                    block_domain.as_ref(),
                    value_domain,
                    "value to clocked block",
                );

                check_target_domain.and(check_value_domain)
            }
        };

        check_domains?;
        check_ty?;

        // convert to value
        let ir_value = value.inner.as_ir_expression(diags, value.span, &target_ty.inner)?;

        ctx.report_assignment(target.as_ref())?;

        let stmt_ir = IrStatement::Assign(ir_target, ir_value.expr);
        ctx.push_ir_statement(
            diags,
            ctx_block,
            Spanned {
                span: stmt.span,
                inner: stmt_ir,
            },
        )?;

        Ok(vars)
    }

    fn elaborate_if_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        vars: VariableValues,
        scope: Scope,
        ifs: Option<(
            &IfCondBlockPair<Box<Expression>, Block<BlockStatement>>,
            &[IfCondBlockPair<Box<Expression>, Block<BlockStatement>>],
        )>,
        final_else: &Option<Block<BlockStatement>>,
    ) -> Result<BlockEnd, ErrorGuaranteed> {
        let diags = self.diags;

        let (initial_if, remaining_ifs) = match ifs {
            Some(p) => p,
            None => {
                return match final_else {
                    None => Ok(BlockEnd::Normal(vars)),
                    Some(final_else) => {
                        let (final_else_ir, final_else_end) = self.elaborate_block(ctx, vars, scope, final_else)?;
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
        let cond = self.eval_expression(ctx, ctx_block, scope, &vars, &Type::Bool, cond)?;

        let reason = TypeContainsReason::Operator(span_if);
        check_type_contains_value(diags, reason, &Type::Bool, cond.as_ref(), false, true)?;

        match cond.inner {
            // evaluate the if at compile-time
            MaybeCompile::Compile(cond_eval) => {
                let cond_eval = match cond_eval {
                    CompileValue::Bool(b) => b,
                    _ => throw!(diags.report_internal_error(cond.span, "expected bool value")),
                };

                // only visit the selected branch
                if cond_eval {
                    let (block_ir, block_end) = self.elaborate_block(ctx, vars, scope, block)?;
                    let block_ir_spanned = Spanned {
                        span: block.span,
                        inner: block_ir,
                    };
                    ctx.push_ir_statement_block(ctx_block, block_ir_spanned);
                    Ok(block_end)
                } else {
                    self.elaborate_if_statement(ctx, ctx_block, vars, scope, remaining_ifs.split_first(), final_else)
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
                let mut else_ir = ctx.new_ir_block();
                let (mut then_ir, then_end, mut else_ir, else_end) =
                    ctx.with_condition_domain(diags, cond_domain, |ctx_inner| {
                        // lower then
                        let (then_ir, then_end) = self.elaborate_block(ctx_inner, vars.clone(), scope, block)?;

                        // lower else
                        let else_end = self.elaborate_if_statement(
                            ctx_inner,
                            &mut else_ir,
                            vars,
                            scope,
                            remaining_ifs.split_first(),
                            final_else,
                        )?;

                        Ok((then_ir, then_end, else_ir, else_end))
                    })?;

                let then_vars = then_end.unwrap_normal_todo_in_if(diags, cond.span);
                let else_vars = else_end.unwrap_normal_todo_in_if(diags, cond.span);
                let then_vars = then_vars?;
                let else_vars = else_vars?;

                // TODO is unwrapping the maps fine here? can we encore that in the type system through C/ctx?
                // TODO re-use one of the maps to avoid an extra allocation
                let then_map = then_vars.map.as_ref().unwrap();
                let else_map = else_vars.map.as_ref().unwrap();

                let mut combined_map = IndexMap::new();
                for &var in itertools::chain(then_map.keys(), else_map.keys()) {
                    if combined_map.contains_key(&var) {
                        // already visited
                        continue;
                    }

                    // if only one of the branches contains the variable, it was locally declared in that branch
                    //   and should now be out of scope
                    let (then_info, else_info) = match (then_map.get(&var), else_map.get(&var)) {
                        (None, _) | (_, None) => continue,
                        (Some(then_info), Some(else_info)) => (then_info, else_info),
                    };

                    let combined_info = match (then_info, else_info) {
                        // any error
                        (&MaybeAssignedValue::FailedIfMerge(span), _)
                        | (_, &MaybeAssignedValue::FailedIfMerge(span)) => MaybeAssignedValue::FailedIfMerge(span),
                        // both the same state
                        (MaybeAssignedValue::Assigned(then_value), MaybeAssignedValue::Assigned(else_value)) => {
                            // TODO maybe it's better to just check whether either has been re-assigned,
                            //   then we don't need to rely on equality
                            if then_value == else_value {
                                MaybeAssignedValue::Assigned(then_value.clone())
                            } else {
                                let (assign_then, assign_else, result) =
                                    self.merge_variable_assigned_values(ctx, span_if, var, then_value, else_value)?;
                                ctx.push_ir_statement(diags, &mut then_ir, assign_then)?;
                                ctx.push_ir_statement(diags, &mut else_ir, assign_else)?;
                                MaybeAssignedValue::Assigned(result)
                            }
                        }
                        (MaybeAssignedValue::NotYetAssigned, MaybeAssignedValue::NotYetAssigned) => {
                            MaybeAssignedValue::NotYetAssigned
                        }
                        // mix of assigned and not assigned
                        (
                            MaybeAssignedValue::Assigned(_)
                            | MaybeAssignedValue::PartiallyAssigned
                            | MaybeAssignedValue::NotYetAssigned,
                            MaybeAssignedValue::Assigned(_)
                            | MaybeAssignedValue::PartiallyAssigned
                            | MaybeAssignedValue::NotYetAssigned,
                        ) => MaybeAssignedValue::PartiallyAssigned,
                    };
                    combined_map.insert_first(var, combined_info);
                }

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

                let combined_vars = VariableValues {
                    map: Some(combined_map),
                };
                Ok(BlockEnd::Normal(combined_vars))
            }
        }
    }

    fn merge_variable_assigned_values<C: ExpressionContext>(
        &self,
        ctx: &mut C,
        span_if: Span,
        var: Variable,
        then_value: &AssignedValue,
        else_value: &AssignedValue,
    ) -> Result<(Spanned<IrStatement>, Spanned<IrStatement>, AssignedValue), ErrorGuaranteed> {
        let diags = self.diags;

        // figure out the hardware type
        let then_ty = then_value.value.ty();
        let else_ty = else_value.value.ty();
        let ty = then_ty.union(&else_ty, false);
        let var_info = &self.variables[var];

        let ty_hw = ty.as_hardware_type().ok_or_else(|| {
            let diag = Diagnostic::new(
                "merging `if` assignments is only possible for variables with types that are representable in hardware",
            )
            .add_error(
                span_if,
                format!("merge necessary here, combined type `{}`", ty.to_diagnostic_string()),
            )
            .add_info(var_info.id.span(), "for variable declared here")
            .add_info(
                then_value.event.assigned_value_span,
                format!(
                    "source value with type `{}` assigned here",
                    then_ty.to_diagnostic_string()
                ),
            )
            .add_info(
                else_value.event.assigned_value_span,
                format!(
                    "source value with type `{}` assigned here",
                    else_ty.to_diagnostic_string()
                ),
            )
            .finish();
            diags.report(diag)
        })?;

        // create ir variable
        let var_ir_info = IrVariableInfo {
            ty: ty_hw.to_ir(),
            debug_info_id: var_info.id.clone(),
        };
        let var_ir = ctx.new_ir_variable(diags, span_if, var_ir_info)?;

        // add assignments to all block
        // TODO include reason for conversion in potential error message
        let then_value_ir = then_value
            .value
            .as_ir_expression(diags, then_value.event.assigned_value_span, &ty_hw)?;
        let else_value_ir = else_value
            .value
            .as_ir_expression(diags, else_value.event.assigned_value_span, &ty_hw)?;

        let assign_then = Spanned {
            span: span_if,
            inner: IrStatement::Assign(IrAssignmentTarget::Variable(var_ir), then_value_ir.expr),
        };
        let assign_else = Spanned {
            span: span_if,
            inner: IrStatement::Assign(IrAssignmentTarget::Variable(var_ir), else_value_ir.expr),
        };

        // return the resulting variable
        let result = AssignedValue {
            // TODO this span is probably not correct
            event: AssignmentEvent {
                assigned_value_span: span_if,
            },
            value: MaybeCompile::Other(TypedIrExpression {
                ty: ty_hw,
                domain: then_value_ir.domain.join(&else_value_ir.domain),
                expr: IrExpression::Variable(var_ir),
            }),
        };
        Ok((assign_then, assign_else, result))
    }

    fn elaborate_for_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        mut result_block: C::Block,
        vars: VariableValues,
        scope: Scope,
        stmt: Spanned<&ForStatement>,
    ) -> Result<(C::Block, BlockEnd<BlockEndReturn>), ErrorGuaranteed> {
        // TODO (deterministic!) timeout
        let diags = self.diags;
        let ctx_block = &mut result_block;

        let ForStatement {
            index: _,
            index_ty,
            iter,
            body: _,
        } = stmt.inner;

        let iter = self.eval_expression(ctx, ctx_block, scope, &vars, &Type::Any, iter);
        let index_ty = index_ty
            .as_ref()
            .map(|index_ty| self.eval_expression_as_ty(scope, &vars, index_ty))
            .transpose();

        let iter = iter?;
        let index_ty = index_ty?;

        let iter_span = iter.span;

        let end = match iter.inner {
            MaybeCompile::Compile(CompileValue::IntRange(iter)) => {
                let IncRange { start_inc, end_inc } = iter;
                let start_inc = match start_inc {
                    Some(start_inc) => start_inc,
                    None => {
                        return Err(diags.report_simple(
                            "for loop iterator range must have start value",
                            iter_span,
                            format!(
                                "got range `{}`",
                                IncRange {
                                    start_inc: None,
                                    end_inc
                                }
                            ),
                        ))
                    }
                };

                let iter = {
                    let mut next = start_inc;
                    std::iter::from_fn(move || {
                        if let Some(end_inc) = &end_inc {
                            if &next > end_inc {
                                return None;
                            }
                        }
                        let curr = MaybeCompile::Compile(CompileValue::Int(next.clone()));
                        next += 1;
                        Some(curr)
                    })
                };

                self.run_for_statement(ctx, ctx_block, vars, scope, stmt, index_ty, iter)?
            }
            MaybeCompile::Compile(CompileValue::Array(iter)) => {
                let iter = iter.into_iter().map(MaybeCompile::Compile);
                self.run_for_statement(ctx, ctx_block, vars, scope, stmt, index_ty, iter)?
            }
            MaybeCompile::Other(TypedIrExpression {
                ty: HardwareType::Array(ty_inner, len),
                domain,
                expr: array_expr,
            }) => {
                // TODO simplify this once empty ranges are representable
                match len.checked_sub(&BigUint::one()) {
                    None => {
                        // empty loop, do nothing
                        BlockEnd::Normal(vars)
                    }
                    Some(end_inc) => {
                        let range = ClosedIncRange {
                            start_inc: BigUint::ZERO,
                            end_inc,
                        };
                        let iter = range.iter().map(|v| {
                            let index_expr = IrExpression::Int(BigInt::from(v));
                            let element_expr = IrExpression::ArrayIndex {
                                base: Box::new(array_expr.clone()),
                                index: Box::new(index_expr),
                            };
                            MaybeCompile::Other(TypedIrExpression {
                                ty: (*ty_inner).clone(),
                                domain: domain.clone(),
                                expr: element_expr,
                            })
                        });
                        self.run_for_statement(ctx, ctx_block, vars, scope, stmt, index_ty, iter)?
                    }
                }
            }
            _ => {
                throw!(diags.report_simple(
                    "invalid for loop iterator type, must be range or array",
                    iter.span,
                    format!("iterator has type `{}`", iter.inner.ty().to_diagnostic_string())
                ))
            }
        };

        Ok((result_block, end))
    }

    fn run_for_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        mut vars: VariableValues,
        scope_parent: Scope,
        stmt: Spanned<&ForStatement>,
        index_ty: Option<Spanned<Type>>,
        iter: impl Iterator<Item = MaybeCompile<TypedIrExpression>>,
    ) -> Result<BlockEnd<BlockEndReturn>, ErrorGuaranteed> {
        let diags = self.diags;
        let ForStatement {
            index: index_id,
            index_ty: _,
            iter: _,
            body,
        } = stmt.inner;

        // create inner scope with index variable
        let scope_index = self.scopes.new_child(scope_parent, stmt.span, Visibility::Private);
        let index_var = self.variables.push(VariableInfo {
            id: index_id.clone(),
            mutable: false,
            ty: None,
        });
        self.scopes[scope_index].maybe_declare(
            diags,
            index_id.as_ref(),
            Ok(ScopedEntry::Direct(NamedValue::Variable(index_var))),
            Visibility::Private,
        );

        // run the actual loop
        for index_value in iter {
            // typecheck index
            // TODO we can also this this once at the start instead, but that's slightly less flexible
            if let Some(index_ty) = &index_ty {
                let curr_spanned = Spanned {
                    span: stmt.inner.iter.span,
                    inner: &index_value,
                };
                let reason = TypeContainsReason::ForIndexType { span_ty: index_ty.span };
                check_type_contains_value(diags, reason, &index_ty.inner, curr_spanned, false, true)?;
            }

            // set index value
            // TODO this span is a bit weird
            let assigned = AssignedValue {
                event: AssignmentEvent {
                    assigned_value_span: index_id.span(),
                },
                value: index_value,
            };
            vars.set(
                diags,
                index_id.span(),
                index_var,
                MaybeAssignedValue::Assigned(assigned),
            )?;

            // run body
            let (body_block, body_end) = self.elaborate_block(ctx, vars, scope_index, body)?;

            let body_block_spanned = Spanned {
                span: body.span,
                inner: body_block,
            };
            ctx.push_ir_statement_block(ctx_block, body_block_spanned);

            // handle possible termination
            match body_end {
                BlockEnd::Normal(new_vars) => vars = new_vars,
                BlockEnd::Stopping(end) => match end {
                    BlockEndStopping::Return(end) => return Ok(BlockEnd::Stopping(end)),
                    BlockEndStopping::Break(_, new_vars) => {
                        vars = new_vars;
                        break;
                    }
                    BlockEndStopping::Continue(_, new_vars) => {
                        vars = new_vars;
                        continue;
                    }
                },
            }
        }

        Ok(BlockEnd::Normal(vars))
    }
}
