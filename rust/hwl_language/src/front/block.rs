use crate::front::check::{check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileState, Variable, VariableInfo};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrAssignmentTarget, IrExpression, IrIfStatement, IrStatement, IrVariable, IrVariableInfo};
use crate::front::misc::{DomainSignal, ScopedEntry, ValueDomain};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{AssignmentTarget, CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{
    Assignment, Block, BlockStatement, BlockStatementKind, Expression, IfCondBlockPair, IfStatement, ReturnStatement,
    Spanned, SyncDomain, VariableDeclaration,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::result_pair;
use indexmap::IndexMap;
use itertools::Itertools;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TypedIrExpression<T = HardwareType> {
    pub ty: T,
    pub domain: ValueDomain,
    pub expr: IrExpression,
}

impl Typed for TypedIrExpression {
    fn ty(&self) -> Type {
        self.ty.as_type()
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
pub enum AssignmentEvent {
    SimpleAssignment(Span),
    IfMerge(Span),
}

impl AssignmentEvent {
    pub fn span(&self) -> Span {
        match self {
            &AssignmentEvent::SimpleAssignment(span) => span,
            &AssignmentEvent::IfMerge(span) => span,
        }
    }
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
            MaybeAssignedValue::PartiallyAssigned => Err(diags.report_simple(
                "variable has not yet been assigned a value in all preceding branches",
                span_use,
                "variable used here",
            )),
            &MaybeAssignedValue::FailedIfMerge(span_if) => {
                // TODO include more specific reason and/or hint
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
pub enum BlockEnd {
    Normal(VariableValues),
    Return {
        span_keyword: Span,
        value: Option<Spanned<MaybeCompile<TypedIrExpression>>>,
    },
}

impl BlockEnd {
    pub fn unwrap_normal_in_if(self, diags: &Diagnostics) -> Result<VariableValues, ErrorGuaranteed> {
        match self {
            BlockEnd::Normal(vars) => Ok(vars),
            BlockEnd::Return { span_keyword, value: _ } => {
                Err(diags.report_todo(span_keyword, "return in non-compile-time `if` statement"))
            }
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
        let &Block {
            span: _,
            ref statements,
        } = block;

        let scope = self.scopes.new_child(scope_parent, block.span, Visibility::Private);
        let mut result_block = ctx.new_ir_block();
        let ctx_block = &mut result_block;

        for stmt in statements {
            let stmt_span = stmt.span;
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

                    let ty = ty
                        .as_ref()
                        .map(|ty| self.eval_expression_as_ty(scope, &vars, ty))
                        .transpose();
                    let init = init
                        .as_ref()
                        .map(|init| self.eval_expression(ctx, ctx_block, scope, &vars, init))
                        .transpose();

                    // check init fits in type
                    let entry = result_pair(ty, init).and_then(|(ty, init)| {
                        // check init fits in type
                        if let Some(ty) = &ty {
                            if let Some(init) = &init {
                                let reason = TypeContainsReason::Assignment {
                                    span_target: id.span(),
                                    span_target_ty: ty.span,
                                };
                                check_type_contains_value(diags, reason, &ty.inner, init.as_ref(), true)?;
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
                                event: AssignmentEvent::SimpleAssignment(decl.span),
                                value: init.inner,
                            }),
                        };
                        vars.set(diags, id.span(), variable, assigned)?;

                        Ok(ScopedEntry::Direct(NamedValue::Variable(variable)))
                    });

                    self.scopes[scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(stmt) => {
                    let (stmt_ir, new_vars) = self.elaborate_assignment(ctx, ctx_block, vars, scope, stmt)?;
                    vars = new_vars;

                    if let Some(stmt_ir) = stmt_ir {
                        let stmt_ir = Spanned {
                            span: stmt_span,
                            inner: stmt_ir,
                        };
                        ctx.push_ir_statement(diags, ctx_block, stmt_ir)?;
                    }
                }
                BlockStatementKind::Expression(expr) => {
                    let _: Spanned<MaybeCompile<TypedIrExpression>> =
                        self.eval_expression(ctx, ctx_block, scope, &vars, expr)?;
                }
                BlockStatementKind::Block(inner_block) => {
                    let (inner_block_ir, block_end) = self.elaborate_block(ctx, vars, scope, inner_block)?;

                    let inner_block_spanned = Spanned {
                        span: inner_block.span,
                        inner: inner_block_ir,
                    };
                    ctx.push_ir_statement_block(ctx_block, inner_block_spanned);

                    match block_end {
                        BlockEnd::Normal(new_vars) => vars = new_vars,
                        ret @ BlockEnd::Return { .. } => return Ok((result_block, ret)),
                    }
                }
                BlockStatementKind::If(stmt_if) => {
                    let IfStatement {
                        initial_if,
                        else_ifs,
                        final_else,
                    } = stmt_if;

                    // TODO avoid allocation here
                    let mut all_ifs = vec![];
                    all_ifs.push(initial_if);
                    all_ifs.extend(else_ifs.iter());

                    let lowered =
                        self.elaborate_if_statement(ctx, ctx_block, vars.clone(), scope, &all_ifs, final_else)?;

                    let next_vars = match lowered {
                        LoweredIfOutcome::Nothing(next_vars) => next_vars,
                        LoweredIfOutcome::SingleBlock(inner_block_ir, block_end) => {
                            ctx.push_ir_statement_block(ctx_block, inner_block_ir);
                            match block_end {
                                BlockEnd::Normal(next_vars) => next_vars,
                                ret @ BlockEnd::Return { .. } => return Ok((result_block, ret)),
                            }
                        }
                        LoweredIfOutcome::IfStatement(lowered) => {
                            let (ir_if, next_vars) =
                                self.convert_if_to_ir_by_merging_variables(ctx, vars, stmt.span, lowered)?;
                            let ir_stmt = Spanned {
                                span: stmt.span,
                                inner: IrStatement::If(ir_if),
                            };
                            ctx.push_ir_statement(diags, ctx_block, ir_stmt)?;
                            next_vars
                        }
                    };
                    vars = next_vars;
                }
                BlockStatementKind::While(_) => throw!(diags.report_todo(stmt.span, "statement kind While")),
                BlockStatementKind::For(_) => throw!(diags.report_todo(stmt.span, "statement kind For")),
                BlockStatementKind::Return(stmt) => {
                    let &ReturnStatement {
                        span_return: span_keyword,
                        ref value,
                    } = stmt;

                    let value = value
                        .as_ref()
                        .map(|value| self.eval_expression(ctx, ctx_block, scope, &vars, value))
                        .transpose()?;

                    let end = BlockEnd::Return { span_keyword, value };
                    return Ok((result_block, end));
                }
                BlockStatementKind::Break(_) => throw!(diags.report_todo(stmt.span, "statement kind Break")),
                BlockStatementKind::Continue => throw!(diags.report_todo(stmt.span, "statement kind Continue")),
            };
        }

        Ok((result_block, BlockEnd::Normal(vars)))
    }

    fn elaborate_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        mut vars: VariableValues,
        scope: Scope,
        stmt: &Assignment,
    ) -> Result<(Option<IrStatement>, VariableValues), ErrorGuaranteed> {
        let diags = self.diags;
        let Assignment {
            span: _,
            op,
            target,
            value,
        } = stmt;

        if op.inner.is_some() {
            throw!(diags.report_todo(stmt.span, "compound assignment"));
        }

        let target = self.eval_expression_as_assign_target(scope, target);
        let value = self.eval_expression(ctx, ctx_block, scope, &vars, value);

        let target = target?;
        let value = value?;

        let condition_domains = ctx.condition_domains();

        let (target_ty, target_domain, ir_target) = match target.inner {
            AssignmentTarget::Port(port) => {
                let info = &self.ports[port];
                let domain = ValueDomain::from_port_domain(info.domain.inner.clone());
                (&info.ty, domain, IrAssignmentTarget::Port(info.ir))
            }
            AssignmentTarget::Wire(wire) => {
                let info = &self.wires[wire];
                let domain = info.domain.inner.clone();
                (&info.ty, domain, IrAssignmentTarget::Wire(info.ir))
            }
            AssignmentTarget::Register(reg) => {
                let info = &self.registers[reg];
                let domain = ValueDomain::Sync(info.domain.inner.clone());
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
                        span_target: id.span(),
                        span_target_ty: ty.span,
                    };
                    check_type_contains_value(diags, reason, &ty.inner, value.as_ref(), true)?;
                }

                // implement by-value semantics
                let (ir_statement, stored_value) = match value.inner {
                    MaybeCompile::Compile(value) => (None, MaybeCompile::Compile(value)),
                    MaybeCompile::Other(value) => {
                        // store value to an ir variable, to turn this assignment into "by value"
                        //   instead of some weird "by reference"
                        let ir_variable_info = IrVariableInfo {
                            ty: value.ty.to_ir(),
                            debug_info_id: id.clone(),
                        };
                        let ir_variable = ctx.new_ir_variable(diags, stmt.span, ir_variable_info)?;
                        let ir_statement = IrStatement::Assign(IrAssignmentTarget::Variable(ir_variable), value.expr);

                        let stored_value = TypedIrExpression {
                            ty: value.ty,
                            domain: value.domain,
                            expr: IrExpression::Variable(ir_variable),
                        };

                        (Some(ir_statement), MaybeCompile::Other(stored_value))
                    }
                };

                // save the value
                let assigned = AssignedValue {
                    event: AssignmentEvent::SimpleAssignment(stmt.span),
                    value: stored_value,
                };
                vars.set(diags, target.span, var, MaybeAssignedValue::Assigned(assigned))?;

                return Ok((ir_statement, vars));
            }
        };

        // check type
        let reason = TypeContainsReason::Assignment {
            span_target: target.span,
            span_target_ty: target_ty.span,
        };
        let target_ty = target_ty.inner.as_type();
        let check_ty = check_type_contains_value(diags, reason, &target_ty, value.as_ref(), true);

        // convert to value
        let value_domain = value.inner.domain();
        let ir_value = value.inner.to_ir_expression(diags, value.span);

        // check domains
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
                let block_domain = block_domain.as_ref().map_inner(|d| ValueDomain::Sync(d.clone()));

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

        let ir_value = ir_value?;
        check_domains?;
        check_ty?;

        ctx.report_assignment(target.as_ref())?;
        let stmt_ir = IrStatement::Assign(ir_target, ir_value.expr);
        Ok((Some(stmt_ir), vars))
    }

    fn elaborate_if_statement<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_cond_eval_block: &mut C::Block,
        vars: VariableValues,
        scope: Scope,
        ifs: &[&IfCondBlockPair<Box<Expression>, Block<BlockStatement>>],
        final_else: &Option<Block<BlockStatement>>,
    ) -> Result<LoweredIfOutcome<C::Block>, ErrorGuaranteed> {
        let diags = self.diags;

        let (initial_if, remaining_ifs) = match ifs.split_first() {
            Some(p) => p,
            None => {
                return match final_else {
                    None => Ok(LoweredIfOutcome::Nothing(vars)),
                    Some(final_else) => {
                        let (final_else_ir, final_else_end) = self.elaborate_block(ctx, vars, scope, final_else)?;
                        let final_else_spanned = Spanned {
                            span: final_else.span,
                            inner: final_else_ir,
                        };
                        Ok(LoweredIfOutcome::SingleBlock(final_else_spanned, final_else_end))
                    }
                };
            }
        };

        let IfCondBlockPair {
            span: _,
            span_if,
            cond,
            block,
        } = initial_if;
        // TODO is this cond_eval_block concept correct? are conditions always evaluated in the right place?
        let cond = self.eval_expression(ctx, ctx_cond_eval_block, scope, &vars, cond)?;

        let reason = TypeContainsReason::Operator(*span_if);
        check_type_contains_value(diags, reason, &Type::Bool, cond.as_ref(), false)?;

        match cond.inner {
            // evaluate the if at compile-time
            MaybeCompile::Compile(cond_eval) => {
                let cond_eval = match cond_eval {
                    CompileValue::Bool(b) => b,
                    _ => throw!(diags.report_internal_error(cond.span, "expected bool value")),
                };

                // only visit the selected branch
                if cond_eval {
                    let (block_ir, next_vars) = self.elaborate_block(ctx, vars, scope, block)?;
                    let block_ir_spanned = Spanned {
                        span: block.span,
                        inner: block_ir,
                    };
                    Ok(LoweredIfOutcome::SingleBlock(block_ir_spanned, next_vars))
                } else {
                    self.elaborate_if_statement(ctx, ctx_cond_eval_block, vars, scope, remaining_ifs, final_else)
                }
            }
            // evaluate the if at runtime, generating IR
            MaybeCompile::Other(cond_eval) => {
                // check condition domain
                // TODO extract this to a common function?
                let check_cond_domain = match ctx.block_domain() {
                    BlockDomain::CompileTime => {
                        throw!(diags
                            .report_internal_error(cond.span, "non-compile-time condition in compile-time context"))
                    }
                    BlockDomain::Combinatorial => Ok(()),
                    BlockDomain::Clocked(block_domain) => {
                        let cond_domain = Spanned {
                            span: cond.span,
                            inner: &cond_eval.domain,
                        };
                        let block_domain = block_domain.as_ref().map_inner(|d| ValueDomain::Sync(d.clone()));
                        self.check_valid_domain_crossing(
                            cond.span,
                            block_domain.as_ref(),
                            cond_domain,
                            "condition used in clocked block",
                        )
                    }
                };

                // record condition domain
                let cond_domain = Spanned {
                    span: cond.span,
                    inner: cond_eval.domain,
                };
                let (then_ir, then_end, else_lowered) = ctx.with_condition_domain(diags, cond_domain, |ctx_inner| {
                    // lower both branches
                    let (then_ir, then_end) = self.elaborate_block(ctx_inner, vars.clone(), scope, block)?;
                    let else_lowered = self.elaborate_if_statement(
                        ctx_inner,
                        ctx_cond_eval_block,
                        vars,
                        scope,
                        remaining_ifs,
                        final_else,
                    )?;

                    Ok((then_ir, then_end, else_lowered))
                })?;

                check_cond_domain?;

                let then_vars = then_end.unwrap_normal_in_if(diags)?;
                let initial_if = IfCondBlockPair {
                    span: cond.span,
                    span_if: *span_if,
                    cond: cond_eval.expr,
                    block: (then_ir, then_vars),
                };

                let stmt: LoweredIfStatement<C::Block> = match else_lowered {
                    LoweredIfOutcome::Nothing(else_vars) => {
                        // new simple if statement without any else-s
                        IfStatement {
                            initial_if,
                            else_ifs: vec![],
                            final_else: (None, else_vars),
                        }
                    }
                    LoweredIfOutcome::SingleBlock(else_block, else_end) => {
                        // new simple if statement with opaque else
                        let else_vars = else_end.unwrap_normal_in_if(diags)?;
                        IfStatement {
                            initial_if,
                            else_ifs: vec![],
                            final_else: (Some(else_block.inner), else_vars),
                        }
                    }
                    LoweredIfOutcome::IfStatement(else_if_stmt) => {
                        // merge into a single bigger if statement
                        let IfStatement {
                            initial_if: else_initial_if,
                            else_ifs: mut combined_else_ifs,
                            final_else,
                        } = else_if_stmt;
                        combined_else_ifs.insert(0, else_initial_if);

                        IfStatement {
                            initial_if,
                            else_ifs: combined_else_ifs,
                            final_else,
                        }
                    }
                };

                Ok(LoweredIfOutcome::IfStatement(stmt))
            }
        }
    }

    // TODO this seems overcomplicated,
    //   maybe a better approach is to just always generate variable writes in IR if possible
    fn convert_if_to_ir_by_merging_variables<C: ExpressionContext>(
        &self,
        ctx: &mut C,
        vars_before: VariableValues,
        span_stmt: Span,
        lowered_if: LoweredIfStatement<C::Block>,
    ) -> Result<(IrIfStatement, VariableValues), ErrorGuaranteed> {
        let diags = self.diags;
        let LoweredIfStatement {
            initial_if,
            mut else_ifs,
            final_else,
        } = lowered_if;

        // collect merge info for all variables
        let mut info: IndexMap<Variable, VariableMergeInfo> = IndexMap::new();

        let (mut initial_if_block, initial_if_vars) = initial_if.block;
        record_merge_info(&mut info, &vars_before, &initial_if_vars);

        for else_if in &else_ifs {
            let IfCondBlockPair {
                span: _,
                span_if: _,
                cond: _,
                block: (_else_if_block, else_if_vars),
            } = else_if;
            record_merge_info(&mut info, &vars_before, &else_if_vars);
        }

        let (mut final_else_block, final_else_vars) = final_else;
        record_merge_info(&mut info, &vars_before, &final_else_vars);

        // do the actual merge
        let mut vars_after = VariableValues::new();

        for (&var, info) in &info {
            let assigned = if info.any_unassigned {
                if info.any_assigned {
                    MaybeAssignedValue::PartiallyAssigned
                } else {
                    MaybeAssignedValue::NotYetAssigned
                }
            } else if let Some(failed_span) = info.any_failed_merge {
                MaybeAssignedValue::FailedIfMerge(failed_span)
            } else if info.any_newly_assigned {
                // TODO infer type if not specified, as the join of all values
                // figure out the hardware type
                let var_info = &self.variables[var];
                let ty = var_info.ty.as_ref().ok_or_else(|| {
                    let diag = Diagnostic::new("cannot merge `if` assignments for variables without explicit type")
                        .add_error(span_stmt, "merge if necessary here")
                        .add_info(var_info.id.span(), "for variable declared here")
                        .finish();
                    diags.report(diag)
                })?;
                let ty_hw = ty.inner.as_hardware_type().ok_or_else(|| {
                    let diag = Diagnostic::new("merging `if` assignments is only possible for variables with types that are representable in hardware")
                        .add_error(span_stmt, "merge necessary here")
                        .add_info(var_info.id.span(), "for variable declared here")
                        .add_info(ty.span, format!("with type {}", ty.inner.to_diagnostic_string()))
                        .finish();
                    diags.report(diag)
                })?;

                // create ir variable
                let var_ir_info = IrVariableInfo {
                    ty: ty_hw.to_ir(),
                    debug_info_id: var_info.id.clone(),
                };
                let var_ir = ctx.new_ir_variable(diags, span_stmt, var_ir_info)?;

                // add assignments to all blocks, and join domains
                let mut joined_domain = ValueDomain::CompileTime;
                let initial_if_merge_store =
                    build_merge_store(diags, span_stmt, var, var_ir, &initial_if_vars, &mut joined_domain)?;
                ctx.push_ir_statement(diags, &mut initial_if_block, initial_if_merge_store)?;

                for else_if in &mut else_ifs {
                    let IfCondBlockPair {
                        span: _,
                        span_if: _,
                        cond: _,
                        block: (else_if_block, else_if_vars),
                    } = else_if;
                    let else_if_merge_store =
                        build_merge_store(diags, span_stmt, var, var_ir, else_if_vars, &mut joined_domain)?;
                    ctx.push_ir_statement(diags, else_if_block, else_if_merge_store)?;
                }

                let final_else_block = final_else_block.get_or_insert_with(|| ctx.new_ir_block());
                let final_else_merge_store =
                    build_merge_store(diags, span_stmt, var, var_ir, &final_else_vars, &mut joined_domain)?;
                ctx.push_ir_statement(diags, final_else_block, final_else_merge_store)?;

                // return the resulting variable
                MaybeAssignedValue::Assigned(AssignedValue {
                    event: AssignmentEvent::IfMerge(span_stmt),
                    value: MaybeCompile::Other(TypedIrExpression {
                        ty: ty_hw,
                        domain: joined_domain,
                        expr: IrExpression::Variable(var_ir),
                    }),
                })
            } else {
                let prev = vars_before.map.as_ref().unwrap().get(&var).ok_or_else(|| {
                    diags.report_internal_error(span_stmt, "variable was not assigned but didn't exist beforehand")
                })?;
                prev.clone()
            };

            vars_after.set(diags, span_stmt, var, assigned)?
        }

        // return finished if statement
        // (we unwrap all blocks, once we're in this function we know we're really in an IR context)
        let stmt: IrIfStatement = IfStatement {
            initial_if: IfCondBlockPair {
                span: initial_if.span,
                span_if: initial_if.span_if,
                cond: initial_if.cond,
                block: ctx.unwrap_ir_block(diags, initial_if.span, initial_if_block)?,
            },
            else_ifs: else_ifs
                .into_iter()
                .map(|else_if| {
                    let (else_if_block, _else_if_vars) = else_if.block;
                    Ok(IfCondBlockPair {
                        span: else_if.span,
                        span_if: else_if.span_if,
                        cond: else_if.cond,
                        block: ctx.unwrap_ir_block(diags, else_if.span, else_if_block)?,
                    })
                })
                .try_collect()?,
            // TODO allow creating new block, just for merging
            final_else: final_else_block
                .map(|final_else_block| ctx.unwrap_ir_block(diags, span_stmt, final_else_block))
                .transpose()?,
        };
        Ok((stmt, vars_after))
    }
}

struct VariableMergeInfo {
    any_assigned: bool,
    any_unassigned: bool,

    any_failed_merge: Option<Span>,
    any_newly_assigned: bool,
}

fn record_merge_info(
    info: &mut IndexMap<Variable, VariableMergeInfo>,
    vars_before: &VariableValues,
    vars_after: &VariableValues,
) {
    // TODO get rid of the unwraps here
    for (&variable, value) in vars_after.map.as_ref().unwrap() {
        let info_slot = info.entry(variable).or_insert_with(|| VariableMergeInfo {
            any_assigned: false,
            any_unassigned: false,
            any_failed_merge: None,
            any_newly_assigned: false,
        });

        match value {
            MaybeAssignedValue::Assigned(_) => {
                info_slot.any_assigned = true;
                if Some(value) != vars_before.map.as_ref().unwrap().get(&variable) {
                    info_slot.any_newly_assigned = true;
                }
            }
            MaybeAssignedValue::NotYetAssigned => {
                info_slot.any_unassigned = true;
            }
            MaybeAssignedValue::PartiallyAssigned => {
                info_slot.any_assigned = true;
                info_slot.any_unassigned = true;
            }
            &MaybeAssignedValue::FailedIfMerge(failed_span) => {
                info_slot.any_failed_merge = Some(failed_span);
            }
        }
    }
}

fn build_merge_store(
    diags: &Diagnostics,
    span_stmt: Span,
    var: Variable,
    var_ir: IrVariable,
    vars: &VariableValues,
    domain: &mut ValueDomain,
) -> Result<Spanned<IrStatement>, ErrorGuaranteed> {
    // TODO get rid of this "unwrap" match by storing this during infor recording
    let value = match vars.map.as_ref().unwrap().get(&var) {
        Some(MaybeAssignedValue::Assigned(value)) => value,
        _ => throw!(diags.report_internal_error(span_stmt, "expected assigned value")),
    };
    // TODO use better span here
    let value_ir = value.value.to_ir_expression(diags, span_stmt)?;

    *domain = domain.join(&value.value.domain());

    let target = IrAssignmentTarget::Variable(var_ir);
    let assign = IrStatement::Assign(target, value_ir.expr);
    let assign_spanned = Spanned {
        span: span_stmt,
        inner: assign,
    };

    Ok(assign_spanned)
}

#[derive(Debug)]
enum LoweredIfOutcome<B> {
    Nothing(VariableValues),
    SingleBlock(Spanned<B>, BlockEnd),
    IfStatement(LoweredIfStatement<B>),
}

type LoweredIfStatement<B> = IfStatement<IrExpression, (B, VariableValues), (Option<B>, VariableValues)>;
