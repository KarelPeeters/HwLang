use crate::front::block::{BlockDomain, TypedIrExpression};
use crate::front::check::{check_type_contains_compile_value, check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileState, Port, Register, Variable, Wire};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{
    IrAssignmentTarget, IrAssignmentTargetBase, IrExpression, IrStatement, IrVariable, IrVariableInfo,
};
use crate::front::misc::{Signal, ValueDomain};
use crate::front::scope::Scope;
use crate::front::steps::ArraySteps;
use crate::front::types::{HardwareType, Type};
use crate::front::value::MaybeCompile;
use crate::syntax::ast::{Assignment, BinaryOp, Spanned};
use crate::syntax::pos::Span;
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
    // TODO add struct/tuple dot indices here
    pub array_steps: ArraySteps,
}

// TODO general name for this, eg. "SignalOrVariable"
#[derive(Debug, Clone)]
pub enum AssignmentTargetBase {
    Port(Port),
    Wire(Wire),
    Register(Register),
    Variable(Variable),
}

impl AssignmentTarget {
    pub fn simple(base: Spanned<AssignmentTargetBase>) -> Self {
        Self {
            base,
            array_steps: ArraySteps::new(vec![]),
        }
    }
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
        mut vars: VariableValues,
        scope: Scope,
        stmt: &Assignment,
    ) -> Result<VariableValues, ErrorGuaranteed> {
        let diags = self.diags;
        let Assignment {
            span: _,
            op,
            target: target_expr,
            value: right_expr,
        } = stmt;

        // evaluate target
        let target = self.eval_expression_as_assign_target(ctx, ctx_block, scope, &vars, target_expr)?;
        let AssignmentTarget {
            base: target_base,
            array_steps: target_steps,
        } = &target.inner;

        // figure out expected type if any
        let right_expected_ty = if op.inner.is_some() {
            Type::Any
        } else {
            let target_base_ty = match target_base.inner {
                AssignmentTargetBase::Port(port) => Some(self.ports[port].ty.as_ref().map_inner(HardwareType::as_type)),
                AssignmentTargetBase::Wire(wire) => Some(self.wires[wire].ty.as_ref().map_inner(HardwareType::as_type)),
                AssignmentTargetBase::Register(reg) => {
                    Some(self.registers[reg].ty.as_ref().map_inner(HardwareType::as_type))
                }
                AssignmentTargetBase::Variable(var) => self.variables[var].ty.clone(),
            };

            // TODO allow Type::Any passthrough
            match target_base_ty {
                None => Type::Any,
                Some(target_base_ty) => {
                    target_steps.apply_to_expected_type(diags, Spanned::new(target_base.span, target_base_ty.inner))?
                }
            }
        };

        // evaluate right value
        let right_eval = self.eval_expression(ctx, ctx_block, scope, &vars, &right_expected_ty, right_expr)?;

        // figure out if we need a compile or hardware assignment
        let target_base_signal = match target_base.inner {
            AssignmentTargetBase::Port(port) => Signal::Port(port),
            AssignmentTargetBase::Wire(wire) => Signal::Wire(wire),
            AssignmentTargetBase::Register(reg) => Signal::Register(reg),
            AssignmentTargetBase::Variable(var) => {
                self.elaborate_variable_assignment(
                    ctx,
                    ctx_block,
                    &mut vars,
                    stmt.span,
                    target,
                    var,
                    *op,
                    right_eval,
                    right_expected_ty,
                )?;
                return Ok(vars);
            }
        };

        // handle hardware signal assignment
        // TODO report exact range/sub-access that is being assigned
        ctx.report_assignment(Spanned::new(target_base.span, target_base_signal))?;

        // evaluate the full value
        let value = match op.inner {
            None => right_eval,
            Some(_) => todo!("hardware signal op assign"),
        };

        // check type (and convert the steps to IR)
        let target_base_ty = target_base_signal.ty(self).map_inner(Clone::clone);

        let diags = self.diags;
        let (target_ty, target_steps_ir) = target_steps.apply_to_hardware_type(diags, target_base_ty.as_ref())?;

        let reason = TypeContainsReason::Assignment {
            span_target: target.span,
            span_target_ty: target_base_ty.span,
        };
        check_type_contains_value(diags, reason, &target_ty.as_type(), value.as_ref(), true, false)?;

        // convert value to hardware
        let value_ir = value.inner.as_ir_expression(diags, value.span, &target_ty)?;

        // check domains
        let value_domain = Spanned {
            span: value.span,
            inner: &value_ir.domain,
        };
        self.check_assignment_domains(
            ctx.block_domain(),
            ctx.condition_domains(),
            op.span,
            target_base_signal.domain(self).as_ref(),
            target_steps,
            value_domain,
        )?;

        // get ir target
        let target_ir = IrAssignmentTarget {
            base: target_base_signal.as_ir_target_base(self),
            steps: target_steps_ir,
        };

        // push ir statement
        let stmt_ir = IrStatement::Assign(target_ir, value_ir.expr);
        ctx.push_ir_statement(
            diags,
            ctx_block,
            Spanned {
                span: stmt.span,
                inner: stmt_ir,
            },
        )?;

        // vars have not changed
        Ok(vars)
    }

    fn elaborate_variable_assignment<C: ExpressionContext>(
        &self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        vars: &mut VariableValues,
        stmt_span: Span,
        target: Spanned<AssignmentTarget>,
        var: Variable,
        op: Spanned<Option<BinaryOp>>,
        right_eval: Spanned<MaybeCompile<TypedIrExpression>>,
        right_expected_ty: Type,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;
        let AssignmentTarget {
            base: target_base,
            array_steps: target_steps,
        } = target.inner;

        // TODO move all of this into a function
        // check mutable
        if !self.variables[var].mutable {
            let diag = Diagnostic::new("assignment to immutable variable")
                .add_error(target_base.span, "variable assigned to here")
                .add_info(self.variables[var].id.span(), "variable declared as immutable here")
                .finish();
            return Err(diags.report(diag));
        }

        // If no op and no steps, we can just do the full assignment immediately.
        // This is separate because in this case the current target value should not be evaluated.
        if op.inner.is_none() && target_steps.is_empty() {
            // check type
            if let Some(ty) = &self.variables[var].ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target.span,
                    span_target_ty: ty.span,
                };
                check_type_contains_value(diags, reason, &ty.inner, right_eval.as_ref(), true, false)?;
            }

            // set variable
            vars.set(
                diags,
                stmt_span,
                var,
                MaybeAssignedValue::Assigned(AssignedValue {
                    event: AssignmentEvent {
                        assigned_value_span: right_eval.span,
                    },
                    value: right_eval.inner,
                }),
            )?;
            return Ok(());
        }

        // at this point the current target value needs to be evaluated
        let target_base_eval = Spanned::new(target_base.span, &vars.get(diags, target_base.span, var)?.value);

        // check if we will stay compile-time or be forced to convert to hardware
        let mut any_hardware = false;
        any_hardware |= matches!(target_base_eval.inner, MaybeCompile::Other(_));
        any_hardware |= target_steps.any_hardware();
        any_hardware |= matches!(right_eval.inner, MaybeCompile::Other(_));

        if any_hardware {
            // implement assignment as hardware
            let value = match op.inner {
                None => right_eval,
                Some(_) => todo!("hardware variable op assign"),
            };

            let (target_base_ty, target_base_domain, var_ir) = self.convert_variable_to_ir_variable(
                ctx,
                ctx_block,
                target.span,
                target_base.span,
                var,
                target_base_eval.inner,
            )?;

            // figure out the inner type and steps
            let (target_inner_ty, target_steps_ir) =
                target_steps.apply_to_hardware_type(diags, target_base_ty.as_ref())?;
            let reason = TypeContainsReason::Assignment {
                span_target: target.span,
                span_target_ty: target_base_ty.span,
            };
            check_type_contains_value(diags, reason, &target_inner_ty.as_type(), value.as_ref(), true, false)?;

            // do the current assignment
            let value_ir = value.inner.as_ir_expression(diags, value.span, &target_inner_ty)?;
            let target_ir = IrAssignmentTarget {
                base: IrAssignmentTargetBase::Variable(var_ir),
                steps: target_steps_ir,
            };
            let stmt_store = IrStatement::Assign(target_ir, value_ir.expr);
            ctx.push_ir_statement(diags, ctx_block, Spanned::new(stmt_span, stmt_store))?;

            // map variable to ir variable
            let mut combined_domain = target_base_domain.join(value.inner.domain());
            target_steps.for_each_domain(|d| combined_domain = combined_domain.join(d.inner));

            let assigned = AssignedValue {
                event: AssignmentEvent {
                    assigned_value_span: value.span,
                },
                value: MaybeCompile::Other(TypedIrExpression {
                    ty: target_base_ty.inner,
                    domain: combined_domain,
                    expr: IrExpression::Variable(var_ir),
                }),
            };
            vars.set(diags, stmt_span, var, MaybeAssignedValue::Assigned(assigned))?;
        } else {
            // everything is compile-time, do assignment at compile-time
            let target_base_eval = target_base_eval.map_inner(|m| m.as_ref().unwrap_compile());
            let target_steps = target_steps.unwrap_compile();
            let right_eval = right_eval.map_inner(MaybeCompile::unwrap_compile);

            match op.inner {
                None => {
                    // check type
                    if let Some(var_ty) = &self.variables[var].ty {
                        let reason = TypeContainsReason::Assignment {
                            span_target: target.span,
                            span_target_ty: var_ty.span,
                        };
                        check_type_contains_compile_value(
                            diags,
                            reason,
                            &right_expected_ty,
                            right_eval.as_ref(),
                            true,
                        )?;
                    }

                    // do assignment
                    let event = AssignmentEvent {
                        assigned_value_span: right_eval.span,
                    };
                    let result_value = target_steps.set_compile_value(
                        diags,
                        target_base_eval.map_inner(Clone::clone),
                        op.span,
                        right_eval,
                    )?;
                    vars.set(
                        diags,
                        stmt_span,
                        var,
                        MaybeAssignedValue::Assigned(AssignedValue {
                            event,
                            value: MaybeCompile::Compile(result_value),
                        }),
                    )?;
                }
                Some(_) => {
                    // let target_eval = Spanned::new(
                    //     target_expr.span,
                    //     target_steps.get_compile_value(diags, Spanned::new(target_base.span, target_base_eval))?,
                    // );
                    todo!("compile-time op assign")
                }
            }
        }

        Ok(())
    }

    fn convert_variable_to_ir_variable<C: ExpressionContext>(
        &self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        target_span: Span,
        target_base_span: Span,
        var: Variable,
        target_base_eval: &MaybeCompile<TypedIrExpression>,
    ) -> Result<(Spanned<HardwareType>, ValueDomain, IrVariable), ErrorGuaranteed> {
        let diags = self.diags;

        // pick a type and convert the current base value to hardware
        let target_base_ty = self.variables[var].ty.as_ref().ok_or_else(|| {
            let diag = Diagnostic::new("variable needs type annotation")
                .add_error(
                    target_span,
                    "for the assignment here the variable needs to be converted to hardware",
                )
                .add_info(self.variables[var].id.span(), "variable declared without a type here")
                .finish();
            diags.report(diag)
        })?;
        let target_base_ty_hw = target_base_ty.inner.as_hardware_type().ok_or_else(|| {
            let diag = Diagnostic::new("variable needs hardware type because it is assigned a hardware value")
                .add_error(
                    target_span,
                    format!(
                        "actual type `{}` not representable in hardware",
                        target_base_ty.inner.to_diagnostic_string()
                    ),
                )
                .add_info(target_base_ty.span, "variable type set here")
                .finish();
            diags.report(diag)
        })?;
        let target_base_ty_hw = Spanned::new(target_base_ty.span, target_base_ty_hw);

        let target_base_ir_expr =
            target_base_eval.as_ir_expression(diags, target_base_span, &target_base_ty_hw.inner)?;

        // create ir variable and replace the variable with it
        let var_ir_info = IrVariableInfo {
            ty: target_base_ty_hw.inner.to_ir(),
            debug_info_id: self.variables[var].id.clone(),
        };
        let var_ir = ctx.new_ir_variable(diags, target_base_span, var_ir_info)?;

        // store previous value into ir variable
        let stmt_init = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), target_base_ir_expr.expr);
        ctx.push_ir_statement(diags, ctx_block, Spanned::new(target_span, stmt_init))?;

        Ok((target_base_ty_hw, target_base_ir_expr.domain, var_ir))
    }

    fn check_assignment_domains(
        &self,
        block_domain: &BlockDomain,
        condition_domains: &[Spanned<ValueDomain>],
        op_span: Span,
        target_base_domain: Spanned<&ValueDomain>,
        steps: &ArraySteps,
        value_domain: Spanned<&ValueDomain>,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;
        match block_domain {
            BlockDomain::CompileTime => {
                for d in [&target_base_domain, &value_domain] {
                    if d.inner != &ValueDomain::CompileTime {
                        return Err(
                            diags.report_internal_error(d.span, "non-compile-time domain in compile-time context")
                        );
                    }
                }
                Ok(())
            }
            BlockDomain::Combinatorial => {
                let mut check = self.check_valid_domain_crossing(
                    op_span,
                    target_base_domain,
                    value_domain,
                    "value to target in combinatorial block",
                );

                steps.for_each_domain(|d| {
                    let c = self.check_valid_domain_crossing(
                        op_span,
                        target_base_domain,
                        d,
                        "step to target in combinatorial block",
                    );
                    check = check.and(c);
                });

                for condition_domain in condition_domains {
                    let c = self.check_valid_domain_crossing(
                        op_span,
                        target_base_domain,
                        condition_domain.as_ref(),
                        "condition to target in combinatorial block",
                    );
                    check = check.and(c);
                }

                check
            }
            BlockDomain::Clocked(block_domain) => {
                let block_domain = block_domain.as_ref().map_inner(|&d| ValueDomain::Sync(d));

                let mut check = self.check_valid_domain_crossing(
                    op_span,
                    target_base_domain,
                    block_domain.as_ref(),
                    "clocked block to target",
                );

                steps.for_each_domain(|d| {
                    let c =
                        self.check_valid_domain_crossing(op_span, block_domain.as_ref(), d, "step to clocked block");
                    check = check.and(c);
                });

                let c = self.check_valid_domain_crossing(
                    op_span,
                    block_domain.as_ref(),
                    value_domain,
                    "value to clocked block",
                );
                check = check.and(c);

                check
            }
        }
    }
}
