use crate::front::check::{check_type_contains_compile_value, check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::domain::{BlockDomain, ValueDomain};
use crate::front::expression::{eval_binary_expression, ValueWithImplications};
use crate::front::scope::Scope;
use crate::front::signal::{Polarized, Port, Register, Signal, Wire};
use crate::front::steps::ArraySteps;
use crate::front::types::{HardwareType, NonHardwareType, Type, Typed};
use crate::front::value::{HardwareValue, Value};
use crate::front::variables::{MaybeAssignedValue, Variable, VariableValues};
use crate::mid::ir::{
    IrAssignmentTarget, IrAssignmentTargetBase, IrExpression, IrStatement, IrVariable, IrVariableInfo,
};
use crate::syntax::ast::{Assignment, BinaryOp, MaybeIdentifier, Spanned, SyncDomain};
use crate::syntax::pos::Span;

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

impl CompileItemContext<'_, '_> {
    pub fn elaborate_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        stmt: &Assignment,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.refs.diags;
        let &Assignment {
            span: _,
            op,
            target: target_expr,
            value: right_expr,
        } = stmt;

        // evaluate target
        let target = self.eval_expression_as_assign_target(ctx, ctx_block, scope, vars, target_expr)?;
        let AssignmentTarget {
            base: target_base,
            array_steps: target_steps,
        } = &target.inner;

        // figure out expected type if any
        let target_base_ty = match target_base.inner {
            AssignmentTargetBase::Port(port) => Some(self.ports[port].ty.as_ref().map_inner(HardwareType::as_type)),
            AssignmentTargetBase::Wire(wire) => {
                let typed = self.wires[wire].typed_maybe(self.refs, &self.wire_interfaces)?;
                typed.map(|typed| typed.ty.map_inner(|ty| ty.as_type()))
            }
            AssignmentTargetBase::Register(reg) => {
                Some(self.registers[reg].ty.as_ref().map_inner(HardwareType::as_type))
            }
            AssignmentTargetBase::Variable(var) => self.variables[var].ty.clone(),
        };
        let target_expected_ty = match target_base_ty {
            None => Type::Any,
            Some(target_base_ty) => {
                target_steps.apply_to_expected_type(diags, Spanned::new(target_base.span, target_base_ty.inner))?
            }
        };

        // evaluate right value
        let right_expected_ty = if op.inner.is_some() {
            &Type::Any
        } else {
            &target_expected_ty
        };
        let right_eval = self.eval_expression(ctx, ctx_block, scope, vars, right_expected_ty, right_expr)?;

        // figure out if we need a compile or hardware assignment
        let target_base_signal = match target_base.inner {
            AssignmentTargetBase::Port(port) => Signal::Port(port),
            AssignmentTargetBase::Wire(wire) => Signal::Wire(wire),
            AssignmentTargetBase::Register(reg) => Signal::Register(reg),
            AssignmentTargetBase::Variable(var) => {
                self.elaborate_variable_assignment(
                    ctx,
                    ctx_block,
                    vars,
                    stmt.span,
                    target,
                    var,
                    target_expected_ty,
                    op,
                    right_eval,
                )?;
                return Ok(());
            }
        };

        // handle hardware signal assignment
        let clocked_block_domain = match ctx.block_domain() {
            BlockDomain::CompileTime => unreachable!(),
            BlockDomain::Combinatorial => None,
            BlockDomain::Clocked(domain) => Some(domain),
        };

        // TODO report exact range/sub-access that is being assigned
        ctx.report_assignment(self, Spanned::new(target_base.span, target_base_signal), vars)?;

        // suggest target type
        if target_steps.is_empty() && op.inner.is_none() {
            let right_ty = right_eval
                .inner
                .ty()
                .as_hardware_type(self.refs)
                .map_err(|_: NonHardwareType| {
                    let msg_value = format!(
                        "assigned value is non-hardware with type `{}`",
                        right_eval.inner.ty().diagnostic_string()
                    );
                    let diag = Diagnostic::new("assignment to hardware signal requires hardware type")
                        .add_error(right_eval.span, msg_value)
                        .add_info(op.span, "assignment to hardware signal here")
                        .finish();
                    diags.report(diag)
                })?;
            let ir_wires = ctx.get_ir_wires(diags, stmt.span)?;
            target_base_signal.suggest_ty(self, ir_wires, Spanned::new(target_base.span, &right_ty))?;
        }

        // get inner type and steps
        let target_base_ty = target_base_signal.ty(self, target_base.span)?.map_inner(Clone::clone);
        let (target_ty, target_steps_ir) = target_steps.apply_to_hardware_type(self.refs, target_base_ty.as_ref())?;

        // evaluate the full value
        let value = match op.inner {
            None => right_eval,
            Some(op_inner) => {
                // TODO apply implications
                let target_base_eval = target_base_signal.as_hardware_value(self, target_base.span)?;
                let target_eval = target_steps.apply_to_value(
                    self.refs,
                    &mut self.large,
                    Spanned::new(target.span, Value::Hardware(target_base_eval)),
                )?;

                let value_eval = eval_binary_expression(
                    self.refs,
                    &mut self.large,
                    stmt.span,
                    Spanned::new(op.span, op_inner),
                    Spanned::new(target.span, ValueWithImplications::simple(target_eval)),
                    right_eval.map_inner(ValueWithImplications::simple),
                )?
                .value();
                let value_eval = match value_eval {
                    Value::Hardware(value) => value,
                    _ => {
                        return Err(diags.report_internal_error(
                            stmt.span,
                            "binary op on hardware values should result in hardware value again",
                        ))
                    }
                };
                Spanned::new(stmt.span, Value::Hardware(value_eval))
            }
        };

        // check type
        let reason = TypeContainsReason::Assignment {
            span_target: target.span,
            span_target_ty: target_base_ty.span,
        };
        check_type_contains_value(diags, reason, &target_ty.as_type(), value.as_ref(), true, false)?;

        // convert value to hardware
        let value_hw = value
            .inner
            .as_hardware_value(self.refs, &mut self.large, value.span, &target_ty)?;

        // check domains
        let value_domain = Spanned {
            span: value.span,
            inner: value_hw.domain,
        };

        let suggested_domain = match clocked_block_domain {
            Some(domain) => domain.map_inner(ValueDomain::Sync),
            None => value.as_ref().map_inner(Value::domain),
        };
        let target_domain = target_base_signal.suggest_domain(self, suggested_domain)?;

        self.check_assignment_domains(
            clocked_block_domain,
            ctx.condition_domains(),
            op.span,
            target_domain,
            target_steps,
            value_domain,
        )?;

        // get ir target
        let target_ir = IrAssignmentTarget {
            base: target_base_signal.as_ir_target_base(self, target_base.span)?,
            steps: target_steps_ir,
        };

        // push ir statement
        let stmt_ir = IrStatement::Assign(target_ir, value_hw.expr);
        ctx.push_ir_statement(
            diags,
            ctx_block,
            Spanned {
                span: stmt.span,
                inner: stmt_ir,
            },
        )?;

        Ok(())
    }

    fn elaborate_variable_assignment<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        vars: &mut VariableValues,
        stmt_span: Span,
        target: Spanned<AssignmentTarget>,
        var: Variable,
        target_expected_ty: Type,
        op: Spanned<Option<BinaryOp>>,
        right_eval: Spanned<Value>,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.refs.diags;
        let AssignmentTarget {
            base: target_base,
            array_steps: target_steps,
        } = target.inner;

        // TODO move all of this into a function
        // check mutable
        if !self.variables[var].mutable {
            let allow = if target_steps.is_empty() {
                match vars.var_get_maybe(diags, target_base.span, var)? {
                    MaybeAssignedValue::Assigned(_) => false,
                    MaybeAssignedValue::NotYetAssigned => true,
                    MaybeAssignedValue::PartiallyAssigned => false,
                    &MaybeAssignedValue::Error(e) => return Err(e),
                }
            } else {
                false
            };

            if !allow {
                let diag = Diagnostic::new("assignment to immutable variable")
                    .add_error(target_base.span, "variable assigned to here")
                    .add_info(self.variables[var].id.span(), "variable declared as immutable here")
                    .finish();
                return Err(diags.report(diag));
            }
        }

        // If no steps, we can just do the full assignment immediately.
        // This is separate because in this case the current target value should not be evaluated.
        if target_steps.is_empty() {
            // evaluate value
            let value_eval = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    // TODO apply implications
                    let target_eval = Spanned::new(
                        target.span,
                        ValueWithImplications::simple(vars.var_get(diags, target.span, var)?.value()),
                    );
                    let value_eval = eval_binary_expression(
                        self.refs,
                        &mut self.large,
                        stmt_span,
                        Spanned::new(op.span, op_inner),
                        target_eval,
                        right_eval.map_inner(ValueWithImplications::simple),
                    )?
                    .value();
                    Spanned::new(stmt_span, value_eval)
                }
            };

            // check type
            if let Some(ty) = &self.variables[var].ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target.span,
                    span_target_ty: ty.span,
                };
                check_type_contains_value(diags, reason, &ty.inner, value_eval.as_ref(), true, false)?;
            }

            // store hardware expression in IR variable, to avoid generating duplicate code if we end up using it multiple times
            let value_stored = match value_eval.inner {
                Value::Compile(value_inner) => Value::Compile(value_inner),
                Value::Hardware(value_inner) => Value::Hardware(
                    store_ir_expression_in_new_variable(
                        self.refs,
                        ctx,
                        ctx_block,
                        self.variables[var].id,
                        value_inner,
                    )?
                    .to_general_expression(),
                ),
            };

            // set variable
            vars.var_set(diags, var, stmt_span, value_stored)?;
            return Ok(());
        }

        // at this point the current target value needs to be evaluated
        // TODO apply implications
        let target_base_eval = Spanned::new(target_base.span, vars.var_get(diags, target_base.span, var)?.value());

        // check if we will stay compile-time or be forced to convert to hardware
        let mut any_hardware = false;
        any_hardware |= matches!(target_base_eval.inner, Value::Hardware(_));
        any_hardware |= target_steps.any_hardware();
        any_hardware |= matches!(right_eval.inner, Value::Hardware(_));

        let result = if any_hardware {
            // figure out the assigned value
            let value = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let target_eval =
                        target_steps.apply_to_value(self.refs, &mut self.large, target_base_eval.clone())?;
                    let value_eval = eval_binary_expression(
                        self.refs,
                        &mut self.large,
                        stmt_span,
                        Spanned::new(op.span, op_inner),
                        Spanned::new(target.span, ValueWithImplications::simple(target_eval)),
                        right_eval.map_inner(ValueWithImplications::simple),
                    )?
                    .value();
                    Spanned::new(stmt_span, value_eval)
                }
            };

            // create a corresponding ir variable
            let HardwareValue {
                ty: target_base_ty,
                domain: target_base_domain,
                expr: target_base_ir_var,
            } = self.convert_variable_to_new_ir_variable(
                ctx,
                ctx_block,
                target.span,
                target_base.span,
                var,
                &target_base_eval.inner,
            )?;

            // figure out the inner type and steps
            let (target_inner_ty, target_steps_ir) =
                target_steps.apply_to_hardware_type(self.refs, target_base_ty.as_ref())?;
            let reason = TypeContainsReason::Assignment {
                span_target: target.span,
                span_target_ty: target_base_ty.span,
            };
            check_type_contains_value(diags, reason, &target_inner_ty.as_type(), value.as_ref(), true, false)?;

            // do the current assignment
            let value_ir = value
                .inner
                .as_hardware_value(self.refs, &mut self.large, value.span, &target_inner_ty)?;
            let target_ir = IrAssignmentTarget {
                base: IrAssignmentTargetBase::Variable(target_base_ir_var),
                steps: target_steps_ir,
            };
            let stmt_store = IrStatement::Assign(target_ir, value_ir.expr);
            ctx.push_ir_statement(diags, ctx_block, Spanned::new(stmt_span, stmt_store))?;

            // map variable to ir variable
            let mut combined_domain = target_base_domain.join(value.inner.domain());
            target_steps.for_each_domain(|d| combined_domain = combined_domain.join(d.inner));

            let value_assigned = HardwareValue {
                ty: target_base_ty.inner,
                domain: combined_domain,
                expr: IrExpression::Variable(target_base_ir_var),
            };
            Value::Hardware(value_assigned)
        } else {
            // everything is compile-time, do assignment at compile-time
            let target_base_eval = target_base_eval.map_inner(|m| m.unwrap_compile());
            let target_steps = target_steps.unwrap_compile();
            let right_eval = right_eval.map_inner(Value::unwrap_compile);

            // handle op
            let value_eval = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let target_eval =
                        target_steps.get_compile_value(self.refs, &mut self.large, target_base_eval.clone())?;
                    let value = eval_binary_expression(
                        self.refs,
                        &mut self.large,
                        stmt_span,
                        Spanned::new(op.span, op_inner),
                        Spanned::new(target.span, ValueWithImplications::simple(Value::Compile(target_eval))),
                        right_eval.map_inner(ValueWithImplications::Compile),
                    )?
                    .value();
                    let value = match value {
                        Value::Compile(value) => value,
                        _ => {
                            return Err(diags.report_internal_error(
                                stmt_span,
                                "binary op on compile values should result in compile value again",
                            ))
                        }
                    };

                    Spanned::new(stmt_span, value)
                }
            };

            // check type
            if let Some(var_ty) = &self.variables[var].ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target.span,
                    span_target_ty: var_ty.span,
                };
                check_type_contains_compile_value(diags, reason, &target_expected_ty, value_eval.as_ref(), true)?;
            }

            // do assignment
            let result_value = target_steps.set_compile_value(diags, target_base_eval, op.span, value_eval)?;
            Value::Compile(result_value)
        };

        vars.var_set(diags, var, stmt_span, result)?;
        Ok(())
    }

    fn convert_variable_to_new_ir_variable<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        target_span: Span,
        target_base_span: Span,
        var: Variable,
        target_base_eval: &Value,
    ) -> Result<HardwareValue<Spanned<HardwareType>, IrVariable>, ErrorGuaranteed> {
        let refs = self.refs;
        let diags = refs.diags;

        // pick a type and convert the current base value to hardware
        // TODO allow just inferring types, the user can specify one if they really want to
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
        let target_base_ty_hw = target_base_ty.inner.as_hardware_type(refs).map_err(|_| {
            let diag = Diagnostic::new("variable needs hardware type because it is assigned a hardware value")
                .add_error(
                    target_span,
                    format!(
                        "actual type `{}` not representable in hardware",
                        target_base_ty.inner.diagnostic_string()
                    ),
                )
                .add_info(target_base_ty.span, "variable type set here")
                .finish();
            diags.report(diag)
        })?;

        let target_base_ir_expr =
            target_base_eval.as_hardware_value(refs, &mut self.large, target_base_span, &target_base_ty_hw)?;
        let result =
            store_ir_expression_in_new_variable(refs, ctx, ctx_block, self.variables[var].id, target_base_ir_expr)?;

        Ok(HardwareValue {
            ty: Spanned::new(target_base_ty.span, result.ty),
            domain: result.domain,
            expr: result.expr,
        })
    }

    fn check_assignment_domains(
        &self,
        clocked_block_domain: Option<Spanned<SyncDomain<Polarized<Signal>>>>,
        condition_domains: &[Spanned<ValueDomain>],
        op_span: Span,
        target_base_domain: Spanned<ValueDomain>,
        steps: &ArraySteps,
        value_domain: Spanned<ValueDomain>,
    ) -> Result<(), ErrorGuaranteed> {
        match clocked_block_domain {
            None => {
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

                for &condition_domain in condition_domains {
                    let c = self.check_valid_domain_crossing(
                        op_span,
                        target_base_domain,
                        condition_domain,
                        "condition to target in combinatorial block",
                    );
                    check = check.and(c);
                }

                check
            }
            Some(block_domain) => {
                let block_domain = block_domain.map_inner(ValueDomain::Sync);
                let mut check = self.check_valid_domain_crossing(
                    op_span,
                    target_base_domain,
                    block_domain,
                    "clocked block to target",
                );

                steps.for_each_domain(|d| {
                    let c = self.check_valid_domain_crossing(op_span, block_domain, d, "step to clocked block");
                    check = check.and(c);
                });

                let c = self.check_valid_domain_crossing(op_span, block_domain, value_domain, "value to clocked block");
                check = check.and(c);

                check
            }
        }
    }
}

// TODO move to better place?
pub fn store_ir_expression_in_new_variable<C: ExpressionContext>(
    refs: CompileRefs,
    ctx: &mut C,
    ctx_block: &mut C::Block,
    debug_info_id: MaybeIdentifier,
    expr: HardwareValue,
) -> Result<HardwareValue<HardwareType, IrVariable>, ErrorGuaranteed> {
    let diags = refs.diags;

    let span = debug_info_id.span();
    let var_ir_info = IrVariableInfo {
        ty: expr.ty.as_ir(refs),
        debug_info_id: debug_info_id.spanned_string(refs.fixed.source),
    };
    let var_ir = ctx.new_ir_variable(diags, span, var_ir_info)?;

    let stmt_store = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), expr.expr);
    ctx.push_ir_statement(diags, ctx_block, Spanned::new(span, stmt_store))?;

    Ok(HardwareValue {
        ty: expr.ty,
        domain: expr.domain,
        expr: var_ir,
    })
}
