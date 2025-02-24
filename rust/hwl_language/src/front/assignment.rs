use crate::front::array::{
    check_range_hardware_index, check_range_hardware_slice, check_range_hardware_slice_start, check_range_mixed_index,
    handle_array_variable_assignment_steps, ArrayAccessStep, SliceLength, VarAssignSpans,
};
use crate::front::block::{BlockDomain, TypedIrExpression};
use crate::front::check::{check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileState, Port, Register, Variable, VariableInfo, Wire};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrAssignmentTarget, IrAssignmentTargetStep, IrExpression, IrStatement};
use crate::front::misc::{Signal, ValueDomain};
use crate::front::scope::Scope;
use crate::front::types::{ClosedIncRange, HardwareType, Type};
use crate::front::value::MaybeCompile;
use crate::syntax::ast::{Assignment, BinaryOp, Expression, Spanned};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::result_pair;
use indexmap::IndexMap;

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
    // TODO this is weird, especially for partial assigns (eg. array[0] = 1)
    pub assigned_value_span: Span,
}

#[derive(Debug, Clone)]
pub struct AssignmentTarget<B = AssignmentTargetBase> {
    pub base: Spanned<B>,
    pub steps: Vec<Spanned<AssignmentTargetStep>>,
}

impl AssignmentTarget {
    pub fn simple(base: Spanned<AssignmentTargetBase>) -> Self {
        Self { base, steps: vec![] }
    }
}

#[derive(Debug, Clone)]
pub enum AssignmentTargetBase {
    Port(Port),
    Wire(Wire),
    Register(Register),
    Variable(Variable),
}

#[derive(Debug, Clone)]
pub enum AssignmentTargetStep {
    ArrayAccess(ArrayAccessStep),
}

#[derive(Debug, Clone)]
pub struct VariableValues {
    // TODO make private again
    pub map: Option<IndexMap<Variable, MaybeAssignedValue>>,
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

impl CompileState<'_> {
    pub fn elaborate_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        vars: VariableValues,
        scope: Scope,
        stmt: &Assignment,
    ) -> Result<VariableValues, ErrorGuaranteed> {
        let Assignment {
            span: _,
            op,
            target: target_expr,
            value,
        } = stmt;

        // evaluate target
        let target = self.eval_expression_as_assign_target(ctx, ctx_block, scope, &vars, target_expr)?;

        // TODO report exact range/sub-access that is being assigned
        ctx.report_assignment(target.as_ref())?;

        let target_base_signal = match target.inner.base.inner {
            AssignmentTargetBase::Port(port) => Signal::Port(port),
            AssignmentTargetBase::Wire(wire) => Signal::Wire(wire),
            AssignmentTargetBase::Register(reg) => Signal::Register(reg),
            AssignmentTargetBase::Variable(var) => {
                let new_vars = self
                    .elaborate_variable_assignment(ctx, ctx_block, vars, scope, stmt.span, target, var, *op, value)?;
                return Ok(new_vars);
            }
        };

        let new_vars = self.elaborate_hardware_assignment(
            ctx,
            ctx_block,
            vars,
            scope,
            stmt.span,
            target_base_signal,
            target,
            *op,
            value,
        )?;
        Ok(new_vars)
    }

    fn elaborate_variable_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        mut vars: VariableValues,
        scope: Scope,

        assignment_span: Span,
        target: Spanned<AssignmentTarget>,
        target_var: Variable,
        op: Spanned<Option<BinaryOp>>,
        assignment_value: &Expression,
    ) -> Result<VariableValues, ErrorGuaranteed> {
        let diags = self.diags;

        let AssignmentTarget {
            base: target_base,
            steps: target_steps,
        } = target.inner;

        let VariableInfo {
            id,
            mutable,
            ty: var_ty,
        } = &self.variables[target_var];
        let var_ty = var_ty.clone();

        // check mutable
        if !*mutable {
            let diag = Diagnostic::new("assignment to immutable variable")
                .add_error(target.span, "variable assigned to here")
                .add_info(id.span(), "variable declared as immutable here")
                .finish();
            return Err(diags.report(diag));
        }

        let new_value = if target_steps.is_empty() {
            // simple full variable assignment
            // TODO can this be merged with the other branch? avoid evaluating the target if not necessary!

            // evaluate value
            let result = match op.transpose() {
                None => {
                    // evaluate with expected type
                    let expected_ty = var_ty.as_ref().map_or(&Type::Any, |ty| &ty.inner);
                    self.eval_expression(ctx, ctx_block, scope, &vars, expected_ty, assignment_value)?
                }
                Some(op) => {
                    // evaluate both
                    let left = Spanned {
                        span: target.span,
                        inner: vars.get(diags, target.span, target_var)?.value.clone(),
                    };
                    let right = self.eval_expression(ctx, ctx_block, scope, &vars, &Type::Any, assignment_value)?;

                    // evaluate operator
                    let result = self.eval_binary_expression(assignment_span, op, left, right)?;

                    // expand to expected type
                    let expected_ty = var_ty.as_ref().and_then(|ty| ty.inner.as_hardware_type());
                    let result_expanded = if let Some(expected_ty) = expected_ty {
                        match result {
                            MaybeCompile::Compile(result) => MaybeCompile::Compile(result),
                            MaybeCompile::Other(result) => {
                                MaybeCompile::Other(result.soft_expand_to_type(&expected_ty))
                            }
                        }
                    } else {
                        result
                    };

                    Spanned {
                        span: assignment_span,
                        inner: result_expanded,
                    }
                }
            };

            // check type
            if let Some(var_ty) = var_ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target.span,
                    span_target_ty: var_ty.span,
                };
                // TODO allow compound subtype or not?
                check_type_contains_value(diags, reason, &var_ty.inner, result.as_ref(), true, true)?;
            }

            result
        } else {
            // we can evaluate here, we know there are steps so it is required anyway
            let target_initial_value = vars.get(diags, target_base.span, target_var)?.value.clone();

            let ty_ref = var_ty.as_ref().map(Spanned::as_ref);

            let spans = VarAssignSpans {
                variable_declaration: self.variables[target_var].id.span(),
                assigned_value: assignment_value.span,
                assignment_operator: op.span,
            };

            let result = handle_array_variable_assignment_steps(
                diags,
                Spanned {
                    span: target_base.span,
                    inner: target_initial_value,
                },
                ty_ref,
                &target_steps,
                spans,
                |left, expected_ty| match op.transpose() {
                    None => Ok(self
                        .eval_expression(ctx, ctx_block, scope, &vars, expected_ty, assignment_value)?
                        .inner),
                    Some(op) => {
                        let right = self.eval_expression(ctx, ctx_block, scope, &vars, &Type::Any, assignment_value)?;
                        self.eval_binary_expression(assignment_span, op, left, right)
                    }
                },
            )?;

            Spanned {
                span: assignment_span,
                inner: result,
            }
        };

        // do assignment
        let assigned = MaybeAssignedValue::Assigned(AssignedValue {
            event: AssignmentEvent {
                assigned_value_span: new_value.span,
            },
            value: new_value.inner,
        });
        vars.set(diags, target.span, target_var, assigned)?;

        Ok(vars)
    }

    fn elaborate_hardware_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        vars: VariableValues,
        scope: Scope,
        assignment_span: Span,
        target_base_signal: Signal,
        target: Spanned<AssignmentTarget>,
        op: Spanned<Option<BinaryOp>>,
        assignment_value: &Expression,
    ) -> Result<VariableValues, ErrorGuaranteed> {
        let diags = self.diags;
        let AssignmentTarget {
            base: target_base,
            steps: target_steps,
        } = target.inner.clone();

        // take steps
        let mut steps_ir = vec![];

        let mut curr_span = target_base.span;
        let target_base_ty = target_base_signal.ty(self).map_inner(Clone::clone);
        let mut curr_inner_ty = &target_base_ty.inner;
        let mut expected_ranges = vec![];

        // TODO check types here
        for step in target_steps {
            curr_inner_ty = match step.inner {
                AssignmentTargetStep::ArrayAccess(step_inner) => match curr_inner_ty {
                    HardwareType::Array(array_inner, array_len) => {
                        let array_len = Spanned {
                            span: curr_span,
                            inner: array_len,
                        };

                        let (step_ir, slice_len) = match step_inner {
                            ArrayAccessStep::IndexOrSliceLen { start, slice_len } => {
                                let start = match start {
                                    MaybeCompile::Compile(start) => {
                                        match &slice_len {
                                            None => check_range_mixed_index(diags, &start, step.span, array_len)?,
                                            Some(slice_len) => {
                                                let start = ClosedIncRange::single(start.clone());
                                                check_range_hardware_slice(
                                                    diags, &start, slice_len, step.span, array_len,
                                                )?;
                                            }
                                        }
                                        IrExpression::Int(start)
                                    }
                                    MaybeCompile::Other(start) => {
                                        match &slice_len {
                                            None => check_range_hardware_index(diags, &start.ty, step.span, array_len)?,
                                            Some(slice_len) => {
                                                check_range_hardware_slice(
                                                    diags, &start.ty, slice_len, step.span, array_len,
                                                )?;
                                            }
                                        }
                                        start.expr
                                    }
                                };
                                (
                                    IrAssignmentTargetStep::ArrayAccess {
                                        start,
                                        slice_len: slice_len.clone(),
                                    },
                                    slice_len,
                                )
                            }
                            ArrayAccessStep::SliceUntilEnd { start } => {
                                let SliceLength(slice_len) =
                                    check_range_hardware_slice_start(diags, &start, step.span, array_len)?;
                                let start = IrExpression::Int(start);
                                (
                                    IrAssignmentTargetStep::ArrayAccess {
                                        start,
                                        slice_len: Some(slice_len.clone()),
                                    },
                                    Some(slice_len),
                                )
                            }
                        };

                        steps_ir.push(step_ir);
                        if let Some(slice_len) = slice_len {
                            expected_ranges.push(slice_len);
                        }

                        array_inner
                    }
                    _ => {
                        let diag = Diagnostic::new("tried to index non-array type")
                            .add_error(
                                step.span,
                                format!(
                                    "attempting to index value of type `{}` here",
                                    curr_inner_ty.to_diagnostic_string()
                                ),
                            )
                            .add_info(
                                target_base.span,
                                format!("base has type `{}`", target_base_ty.inner.to_diagnostic_string()),
                            )
                            .add_info(target_base_ty.span, "base type set here")
                            .finish();
                        return Err(diags.report(diag));
                    }
                },
            };
            curr_span = curr_span.join(step.span);
        }

        // by going through the steps we've figured out the inner type and built the ir target
        let expected_ty = expected_ranges
            .into_iter()
            .rev()
            .fold(curr_inner_ty.clone(), |a, r| HardwareType::Array(Box::new(a), r));

        let ir_target = IrAssignmentTarget {
            base: target_base_signal.ir_assignment_target(self),
            steps: steps_ir,
        };

        // evaluate value
        let value = if let Some(op) = op.transpose() {
            // compound assignment, re-evaluate target and don't pass expected type for right
            let target = Spanned {
                span: target.span,
                inner: AssignmentTarget {
                    base: Spanned {
                        span: target.inner.base.span,
                        inner: target_base_signal,
                    },
                    steps: target.inner.steps,
                },
            };
            let value_left = self.eval_assignment_target_as_expression_unchecked(target.as_ref());
            let value_right = self.eval_expression(ctx, ctx_block, scope, &vars, &Type::Any, assignment_value);

            result_pair(value_left, value_right).and_then(|(left, right)| {
                let left = left.map_inner(|left| MaybeCompile::Other(left));
                let result = self.eval_binary_expression(assignment_span, op, left, right)?;
                Ok(Spanned {
                    span: assignment_span,
                    inner: result,
                })
            })?
        } else {
            // simple assignment, pass expected type along
            self.eval_expression(ctx, ctx_block, scope, &vars, &expected_ty.as_type(), assignment_value)?
        };

        // check type
        // TODO expand value to type?
        let reason = TypeContainsReason::Assignment {
            span_target: target.span,
            span_target_ty: target_base_ty.span,
        };
        let check_ty = check_type_contains_value(diags, reason, &expected_ty.as_type(), value.as_ref(), true, false);

        // check domains
        // TODO double-check if/how step/index domains should influence this
        let value_domain = value.inner.domain();
        // TODO better error messages with more explanation
        let target_domain = Spanned {
            span: target.span,
            inner: &target_base_signal.domain(self).inner,
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
                    assignment_span,
                    target_domain,
                    value_domain,
                    "value to target in combinatorial block",
                );

                for condition_domain in ctx.condition_domains() {
                    let c = self.check_valid_domain_crossing(
                        assignment_span,
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
                    assignment_span,
                    target_domain,
                    block_domain.as_ref(),
                    "clocked block to target",
                );
                let check_value_domain = self.check_valid_domain_crossing(
                    assignment_span,
                    block_domain.as_ref(),
                    value_domain,
                    "value to clocked block",
                );

                check_target_domain.and(check_value_domain)
            }
        };

        check_domains?;
        check_ty?;

        // convert to IR and store
        let ir_value = value.inner.as_ir_expression(diags, value.span, &target_base_ty.inner)?;

        let stmt_ir = IrStatement::Assign(ir_target, ir_value.expr);
        ctx.push_ir_statement(
            diags,
            ctx_block,
            Spanned {
                span: assignment_span,
                inner: stmt_ir,
            },
        )?;

        Ok(vars)
    }
}
