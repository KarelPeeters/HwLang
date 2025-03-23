use crate::front::block::{BlockDomain, TypedIrExpression};
use crate::front::check::{check_type_contains_compile_value, check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileState, Port, Register, Variable, Wire};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::expression::{eval_binary_expression, ExpressionWithImplications};
use crate::front::ir::{
    IrAssignmentTarget, IrAssignmentTargetBase, IrExpression, IrStatement, IrVariable, IrVariableInfo,
};
use crate::front::misc::{Signal, SignalOrVariable, ValueDomain};
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

// TODO move this into context, together with the ir block being written and the scope
#[derive(Debug, Clone)]
pub struct VariableValues {
    pub inner: Option<VariableValuesInner>,
}

#[derive(Debug, Clone)]
pub struct VariableValuesInner {
    // TODO for combinatorial blocks: include which bits have been written/read on signals,
    //   so that we can report an error if not bits are written at the end and if bits are read before they are written
    //   (if any other bit of the respective signal is written later)
    pub versions: IndexMap<SignalOrVariable, u64>,
    pub map: IndexMap<Variable, MaybeAssignedValue>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ValueVersioned {
    pub value: SignalOrVariable,
    pub version: ValueVersion,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ValueVersion(u64);

impl VariableValues {
    pub fn new() -> Self {
        Self {
            inner: Some(VariableValuesInner {
                versions: IndexMap::new(),
                map: IndexMap::new(),
            }),
        }
    }

    pub fn new_no_vars() -> Self {
        Self { inner: None }
    }

    pub fn get(&self, diags: &Diagnostics, span_use: Span, var: Variable) -> Result<&AssignedValue, ErrorGuaranteed> {
        let map = &self
            .inner
            .as_ref()
            .ok_or_else(|| diags.report_internal_error(span_use, "variable are not allowed in this context"))?
            .map;
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
        match &mut self.inner {
            None => Err(diags.report_internal_error(span_set, "variables are not allowed in this context")),
            Some(inner) => {
                inner.map.insert(var, value);
                *inner.versions.entry(SignalOrVariable::Variable(var)).or_insert(0) += 1;
                Ok(())
            }
        }
    }

    pub fn value_versioned(&self, value: SignalOrVariable) -> Option<ValueVersioned> {
        let inner = self.inner.as_ref()?;
        let version = ValueVersion(*inner.versions.get(&value).unwrap_or(&0));
        Some(ValueVersioned { value, version })
    }

    pub fn report_signal_assignment(
        &mut self,
        diags: &Diagnostics,
        signal: Spanned<Signal>,
    ) -> Result<(), ErrorGuaranteed> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            diags.report_internal_error(
                signal.span,
                "signal versioning doesn't make sense in contexts without variables",
            )
        })?;
        *inner
            .versions
            .entry(SignalOrVariable::Signal(signal.inner))
            .or_insert(0) += 1;
        Ok(())
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
        let target_base_ty = match target_base.inner {
            AssignmentTargetBase::Port(port) => {
                Some(self.state.ports[port].ty.as_ref().map_inner(HardwareType::as_type))
            }
            AssignmentTargetBase::Wire(wire) => {
                Some(self.state.wires[wire].ty.as_ref().map_inner(HardwareType::as_type))
            }
            AssignmentTargetBase::Register(reg) => {
                Some(self.state.registers[reg].ty.as_ref().map_inner(HardwareType::as_type))
            }
            AssignmentTargetBase::Variable(var) => self.state.variables[var].ty.clone(),
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
                    target_expected_ty,
                    *op,
                    right_eval,
                )?;
                return Ok(vars);
            }
        };

        // handle hardware signal assignment
        // TODO report exact range/sub-access that is being assigned
        ctx.report_assignment(diags, Spanned::new(target_base.span, target_base_signal), &mut vars)?;

        // get inner type and steps
        let target_base_ty = target_base_signal.ty(self.state).map_inner(Clone::clone);
        let (target_ty, target_steps_ir) = target_steps.apply_to_hardware_type(diags, target_base_ty.as_ref())?;

        // evaluate the full value
        let value = match op.inner {
            None => right_eval,
            Some(op_inner) => {
                // TODO apply implications
                let target_base_eval = target_base_signal.as_ir_expression(self.state);
                let target_eval = target_steps
                    .apply_to_value(diags, Spanned::new(target.span, MaybeCompile::Other(target_base_eval)))?;

                let value_eval = eval_binary_expression(
                    diags,
                    stmt.span,
                    Spanned::new(op.span, op_inner),
                    Spanned::new(target.span, ExpressionWithImplications::simple(target_eval)),
                    right_eval.map_inner(ExpressionWithImplications::simple),
                )?
                .value;
                let value_eval = match value_eval {
                    MaybeCompile::Compile(_) => {
                        return Err(diags.report_internal_error(
                            stmt.span,
                            "binary op on compile values should result in compile value again",
                        ))
                    }
                    MaybeCompile::Other(value) => value,
                };
                Spanned::new(stmt.span, MaybeCompile::Other(value_eval))
            }
        };

        // check type
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
            target_base_signal.domain(self.state).as_ref(),
            target_steps,
            value_domain,
        )?;

        // get ir target
        let target_ir = IrAssignmentTarget {
            base: target_base_signal.as_ir_target_base(self.state),
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
        target_expected_ty: Type,
        op: Spanned<Option<BinaryOp>>,
        right_eval: Spanned<MaybeCompile<TypedIrExpression>>,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;
        let AssignmentTarget {
            base: target_base,
            array_steps: target_steps,
        } = target.inner;

        // TODO move all of this into a function
        // check mutable
        if !self.state.variables[var].mutable {
            let diag = Diagnostic::new("assignment to immutable variable")
                .add_error(target_base.span, "variable assigned to here")
                .add_info(
                    self.state.variables[var].id.span(),
                    "variable declared as immutable here",
                )
                .finish();
            return Err(diags.report(diag));
        }

        // If no steps, we can just do the full assignment immediately.
        // This is separate because in this case the current target value should not be evaluated.
        if target_steps.is_empty() {
            // evaluate value
            let value_eval = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let target_eval = Spanned::new(
                        target.span,
                        ExpressionWithImplications::simple(vars.get(diags, target.span, var)?.value.clone()),
                    );
                    let value_eval = eval_binary_expression(
                        diags,
                        stmt_span,
                        Spanned::new(op.span, op_inner),
                        target_eval,
                        right_eval.map_inner(ExpressionWithImplications::simple),
                    )?
                    .value;
                    Spanned::new(stmt_span, value_eval)
                }
            };

            // check type
            if let Some(ty) = &self.state.variables[var].ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target.span,
                    span_target_ty: ty.span,
                };
                check_type_contains_value(diags, reason, &ty.inner, value_eval.as_ref(), true, false)?;
            }

            // set variable
            vars.set(
                diags,
                stmt_span,
                var,
                MaybeAssignedValue::Assigned(AssignedValue {
                    event: AssignmentEvent {
                        assigned_value_span: value_eval.span,
                    },
                    value: value_eval.inner,
                }),
            )?;
            return Ok(());
        }

        // at this point the current target value needs to be evaluated
        // TODO apply implications
        let target_base_eval = Spanned::new(target_base.span, &vars.get(diags, target_base.span, var)?.value);

        // check if we will stay compile-time or be forced to convert to hardware
        let mut any_hardware = false;
        any_hardware |= matches!(target_base_eval.inner, MaybeCompile::Other(_));
        any_hardware |= target_steps.any_hardware();
        any_hardware |= matches!(right_eval.inner, MaybeCompile::Other(_));

        if any_hardware {
            // figure out the assigned value
            let value = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let target_eval = target_steps.apply_to_value(diags, target_base_eval.cloned())?;
                    let value_eval = eval_binary_expression(
                        diags,
                        stmt_span,
                        Spanned::new(op.span, op_inner),
                        Spanned::new(target.span, ExpressionWithImplications::simple(target_eval)),
                        right_eval.map_inner(ExpressionWithImplications::simple),
                    )?
                    .value;
                    Spanned::new(stmt_span, value_eval)
                }
            };

            // create a corresponding ir variable
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

            // handle op
            let value_eval = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let target_eval = target_steps.get_compile_value(diags, target_base_eval.cloned())?;
                    let value = eval_binary_expression(
                        diags,
                        stmt_span,
                        Spanned::new(op.span, op_inner),
                        Spanned::new(
                            target.span,
                            ExpressionWithImplications::simple(MaybeCompile::Compile(target_eval)),
                        ),
                        right_eval.map_inner(|e| ExpressionWithImplications::simple(MaybeCompile::Compile(e))),
                    )?
                    .value;
                    let value = match value {
                        MaybeCompile::Compile(value) => value,
                        MaybeCompile::Other(_) => {
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
            if let Some(var_ty) = &self.state.variables[var].ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target.span,
                    span_target_ty: var_ty.span,
                };
                check_type_contains_compile_value(diags, reason, &target_expected_ty, value_eval.as_ref(), true)?;
            }

            // do assignment
            let event = AssignmentEvent {
                assigned_value_span: value_eval.span,
            };
            let result_value =
                target_steps.set_compile_value(diags, target_base_eval.map_inner(Clone::clone), op.span, value_eval)?;
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
        let target_base_ty = self.state.variables[var].ty.as_ref().ok_or_else(|| {
            let diag = Diagnostic::new("variable needs type annotation")
                .add_error(
                    target_span,
                    "for the assignment here the variable needs to be converted to hardware",
                )
                .add_info(
                    self.state.variables[var].id.span(),
                    "variable declared without a type here",
                )
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
            debug_info_id: self.state.variables[var].id.clone(),
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
