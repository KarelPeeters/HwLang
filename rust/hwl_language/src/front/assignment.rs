use crate::front::check::{TypeContainsReason, check_type_contains_compile_value, check_type_contains_value};
use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable};
use crate::front::domain::{DomainSignal, ValueDomain};
use crate::front::expression::eval_binary_expression;
use crate::front::flow::{Flow, FlowHardware, HardwareProcessKind, Variable};
use crate::front::implication::{BoolImplications, HardwareValueWithImplications, ValueWithImplications};
use crate::front::scope::Scope;
use crate::front::signal::{Port, Register, Signal, Wire};
use crate::front::steps::ArraySteps;
use crate::front::types::{HardwareType, NonHardwareType, Type, Typed};
use crate::front::value::{HardwareValue, Value};
use crate::mid::ir::{
    IrAssignmentTarget, IrAssignmentTargetBase, IrExpression, IrStatement, IrVariable, IrVariableInfo,
};
use crate::syntax::ast::{AssignBinaryOp, Assignment, SyncDomain};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use annotate_snippets::Level;

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

enum BlockKind {
    Combinatorial,
    Clocked(Spanned<SyncDomain<DomainSignal>>),
}

impl CompileItemContext<'_, '_> {
    // TODO this probably needs yet another refactor, there's a lot of semi-duplicate code here
    pub fn elaborate_assignment(&mut self, scope: &Scope, flow: &mut impl Flow, stmt: &Assignment) -> DiagResult {
        let diags = self.refs.diags;
        let &Assignment {
            span: _,
            op,
            target: target_expr,
            value: right_expr,
        } = stmt;

        // evaluate target
        let target = self.eval_expression_as_assign_target(scope, flow, target_expr)?;
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
            AssignmentTargetBase::Variable(var) => flow.var_info(Spanned::new(target_base.span, var))?.ty.clone(),
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
        let right_eval = self.eval_expression_with_implications(scope, flow, right_expected_ty, right_expr)?;

        // figure out if we need a compile or hardware assignment
        let target_base_signal = match target_base.inner {
            AssignmentTargetBase::Port(port) => Signal::Port(port),
            AssignmentTargetBase::Wire(wire) => Signal::Wire(wire),
            AssignmentTargetBase::Register(reg) => Signal::Register(reg),
            AssignmentTargetBase::Variable(var) => {
                self.elaborate_variable_assignment(
                    flow,
                    stmt.span,
                    target.span,
                    Spanned::new(target_base.span, var),
                    target_steps,
                    target_expected_ty,
                    op,
                    right_eval,
                )?;
                return Ok(());
            }
        };
        let target_base_signal = Spanned::new(target_base.span, target_base_signal);

        // variable assignments have been handled, now we know the target is a signal
        //   check this this is a hardware flow and that we're allowed to write to signals in this context
        // TODO still count as driver or suggested type to supress future errors?
        // TODO maybe type inference for wires based on processes is not actually a good idea,
        //   module instances should be enough
        // TODO maybe elaborate child instances first, before any blocks, since we can infer more info based on them
        let flow = flow.check_hardware(stmt.span, "assignment to hardware signal")?;
        let flow_block_kind = self.check_block_kind_and_driver_type(flow, target_base_signal)?;

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

            let ir_wires = flow.get_ir_wires();
            target_base_signal
                .inner
                .suggest_ty(self, ir_wires, Spanned::new(target_base.span, &right_ty))?;
        }

        // get inner type and steps
        let target_base_ty = target_base_signal
            .inner
            .ty(self, target_base.span)?
            .map_inner(Clone::clone);
        let (target_ty, target_steps_ir) = target_steps.apply_to_hardware_type(self.refs, target_base_ty.as_ref())?;

        // evaluate the full value
        let value = match op.inner {
            None => right_eval,
            Some(op_inner) => {
                let target_base_eval = flow.signal_eval(self, target_base_signal)?;

                let target_eval = if target_steps.is_empty() {
                    HardwareValueWithImplications {
                        value: target_base_eval.value,
                        version: Some(target_base_eval.version),
                        implications: BoolImplications::default(),
                    }
                } else {
                    let target_eval = target_steps.apply_to_hardware_value(
                        self.refs,
                        &mut self.large,
                        Spanned::new(target.span, target_base_eval.value),
                    )?;
                    HardwareValueWithImplications {
                        value: target_eval,
                        version: None,
                        implications: BoolImplications::default(),
                    }
                };

                let value_eval = eval_binary_expression(
                    self.refs,
                    &mut self.large,
                    stmt.span,
                    Spanned::new(op.span, op_inner.to_binary_op()),
                    Spanned::new(target.span, Value::Hardware(target_eval)),
                    right_eval,
                )?;
                let value_eval = match value_eval {
                    Value::Hardware(value) => value,
                    _ => {
                        return Err(diags.report_internal_error(
                            stmt.span,
                            "binary op on hardware values should result in hardware value again",
                        ));
                    }
                };
                Spanned::new(stmt.span, Value::Hardware(value_eval))
            }
        };
        let value = value.map_inner(ValueWithImplications::into_value);

        // report assignment after everything has been evaluated
        // TODO report exact range/sub-access that is being assigned
        // TODO report assigned value type and even the full compile value, so we have more information later on
        flow.signal_assign(target_base_signal, target_steps.is_empty());

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

        // suggest and check domains
        let value_domain = Spanned {
            span: value.span,
            inner: value_hw.domain,
        };
        let suggested_domain = match flow_block_kind {
            BlockKind::Clocked(domain) => domain.map_inner(ValueDomain::Sync),
            BlockKind::Combinatorial => value_domain,
        };
        let target_domain = target_base_signal.inner.suggest_domain(self, suggested_domain)?;

        self.check_assignment_domains(
            flow_block_kind,
            flow.condition_domains(),
            op.span,
            target_domain,
            target_steps,
            value_domain,
        )?;

        // get ir target
        let target_ir = IrAssignmentTarget {
            base: target_base_signal.inner.as_ir_target_base(self, target_base.span)?,
            steps: target_steps_ir,
        };

        // push ir statement
        let stmt_ir = IrStatement::Assign(target_ir, value_hw.expr);
        flow.push_ir_statement(Spanned::new(stmt.span, stmt_ir));

        Ok(())
    }

    fn check_block_kind_and_driver_type(
        &self,
        flow: &mut FlowHardware,
        target_base_signal: Spanned<Signal>,
    ) -> DiagResult<BlockKind> {
        let diags = self.refs.diags;

        let block_kind = match flow.block_kind() {
            HardwareProcessKind::CombinatorialBlockBody {
                span_keyword: _,
                wires_driven,
                ports_driven,
            } => {
                match target_base_signal.inner {
                    Signal::Port(port) => {
                        ports_driven.entry(port).or_insert(target_base_signal.span);
                    }
                    Signal::Wire(wire) => {
                        wires_driven.entry(wire).or_insert(target_base_signal.span);
                    }
                    Signal::Register(reg) => {
                        let reg_info = &self.registers[reg];
                        let diag = Diagnostic::new("registers must be driven by a clocked block")
                            .add_error(target_base_signal.span, "driven incorrectly here")
                            .add_info(reg_info.id.span(), "register declared here")
                            .footer(
                                Level::Help,
                                "drive the register from a clocked block or turn it into a wire",
                            )
                            .finish();
                        return Err(diags.report(diag));
                    }
                }

                BlockKind::Combinatorial
            }
            HardwareProcessKind::ClockedBlockBody {
                span_keyword: _,
                domain,
                registers_driven,
                extra_registers: _,
            } => {
                match target_base_signal.inner {
                    Signal::Register(reg) => {
                        registers_driven.entry(reg).or_insert(target_base_signal.span);
                    }
                    Signal::Port(port) => {
                        let port_info = &self.ports[port];
                        let diag = Diagnostic::new("ports cannot be driven by a clocked block")
                            .add_error(target_base_signal.span, "driven incorrectly here")
                            .add_info(port_info.span, "port declared here")
                            .footer(
                                Level::Help,
                                "mark the port as a register or drive it from a combinatorial block or connection",
                            )
                            .finish();
                        return Err(diags.report(diag));
                    }
                    Signal::Wire(wire) => {
                        let wire_info = &self.wires[wire];
                        let diag = Diagnostic::new("wires cannot be driven by a clocked block")
                            .add_error(target_base_signal.span, "driven incorrectly here")
                            .add_info(wire_info.decl_span(), "wire declared here")
                            .footer(
                                Level::Help,
                                "change the wire to a register or drive it from a combinatorial block or connection",
                            )
                            .finish();
                        return Err(diags.report(diag));
                    }
                }

                BlockKind::Clocked(*domain)
            }
            HardwareProcessKind::WireExpression {
                span_keyword: _,
                span_init,
            } => {
                let diag = Diagnostic::new("assigning to signals is only allowed in processes")
                    .add_error(target_base_signal.span, "assigning to signal here")
                    .add_info(*span_init, "the current context is a wire expression")
                    .finish();
                return Err(diags.report(diag));
            }
            HardwareProcessKind::InstancePortConnection { span_connection } => {
                let diag = Diagnostic::new("assigning to signals is only allowed in processes")
                    .add_error(target_base_signal.span, "assigning to signal here")
                    .add_info(*span_connection, "the current context is an instance connection")
                    .finish();
                return Err(diags.report(diag));
            }
        };
        Ok(block_kind)
    }

    fn elaborate_variable_assignment(
        &mut self,
        flow: &mut impl Flow,
        stmt_span: Span,
        target_span: Span,
        target_base: Spanned<Variable>,
        target_steps: &ArraySteps,
        target_expected_ty: Type,
        op: Spanned<Option<AssignBinaryOp>>,
        right_eval: Spanned<ValueWithImplications>,
    ) -> DiagResult {
        let diags = self.refs.diags;

        // TODO move all of this into a function
        // check mutable
        let target_base_var_info = flow.var_info(target_base)?;
        if !target_base_var_info.mutable {
            let is_simple_first_assignment = target_steps.is_empty() && flow.var_is_not_yet_assigned(target_base)?;
            if !is_simple_first_assignment {
                let diag = Diagnostic::new("assignment to immutable variable that has already been initialized")
                    .add_error(target_base.span, "variable assigned to here")
                    .add_info(target_base_var_info.span_decl, "variable declared as immutable here")
                    .finish();
                return Err(diags.report(diag));
            }
        }

        // If no steps, we can just do the full assignment immediately.
        // This is separate because in this case the current target value should not be evaluated.
        // TODO we still eval if there is an op, can we merge that?
        if target_steps.is_empty() {
            // evaluate value
            let value = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let var_eval = flow.var_eval_unchecked(diags, &mut self.large, target_base)?;
                    let target_eval = Spanned::new(target_span, ValueWithImplications::simple_version(var_eval));
                    let value_eval = eval_binary_expression(
                        self.refs,
                        &mut self.large,
                        stmt_span,
                        Spanned::new(op.span, op_inner.to_binary_op()),
                        target_eval,
                        right_eval,
                    )?;
                    Spanned::new(stmt_span, value_eval)
                }
            };
            let value = value.map_inner(ValueWithImplications::into_value);

            // check type
            let target_base_info = flow.var_info(target_base)?;
            if let Some(ty) = &target_base_info.ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target_span,
                    span_target_ty: ty.span,
                };
                check_type_contains_value(diags, reason, &ty.inner, value.as_ref(), true, false)?;
            }

            // store hardware expression in IR variable, to avoid generating duplicate code if we end up using it multiple times
            let value_stored = match value.inner {
                Value::Compile(value_inner) => Value::Compile(value_inner),
                Value::Hardware(value_inner) => {
                    let debug_info_id = target_base_info.id.str(self.refs.fixed.source).map(str::to_owned);
                    let flow = flow.check_hardware(stmt_span, "assignment involving hardware value")?;
                    let ir_var = store_ir_expression_in_new_variable(
                        self.refs,
                        flow,
                        target_base.span,
                        debug_info_id,
                        value_inner,
                    )?;
                    Value::Hardware(ir_var.to_general_expression())
                }
            };

            // set variable
            flow.var_set(target_base.inner, stmt_span, Ok(value_stored));
            return Ok(());
        }

        // at this point the current target value needs to be evaluated
        let target_base_eval = Spanned::new(
            target_base.span,
            flow.var_eval_unchecked(diags, &mut self.large, target_base)?,
        );

        // check if we will stay compile-time or be forced to convert to hardware
        let mut any_hardware = false;
        any_hardware |= matches!(target_base_eval.inner, Value::Hardware(_));
        any_hardware |= target_steps.any_hardware();
        any_hardware |= matches!(right_eval.inner, Value::Hardware(_));

        let result = if any_hardware {
            let flow = flow.check_hardware(stmt_span, "assignment involving hardware value")?;

            // figure out the assigned value
            // TODO propagate implications?
            let value = match op.inner {
                None => right_eval,
                Some(op_inner) => {
                    let target_eval = target_steps.apply_to_value(
                        self.refs,
                        &mut self.large,
                        target_base_eval.clone().map_inner(|v| v.into_value()),
                    )?;

                    let value_eval = eval_binary_expression(
                        self.refs,
                        &mut self.large,
                        stmt_span,
                        Spanned::new(op.span, op_inner.to_binary_op()),
                        Spanned::new(target_span, ValueWithImplications::simple(target_eval)),
                        right_eval,
                    )?;
                    Spanned::new(stmt_span, value_eval)
                }
            };
            let value = value.map_inner(ValueWithImplications::into_value);

            // create a corresponding ir variable
            let HardwareValue {
                ty: target_base_ty,
                domain: target_base_domain,
                expr: target_base_ir_var,
            } = self.convert_variable_to_new_ir_variable(
                flow,
                target_span,
                target_base,
                &target_base_eval.inner.into_value(),
            )?;

            // figure out the inner type and steps
            let (target_inner_ty, target_steps_ir) =
                target_steps.apply_to_hardware_type(self.refs, target_base_ty.as_ref())?;
            let reason = TypeContainsReason::Assignment {
                span_target: target_span,
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
            flow.push_ir_statement(Spanned::new(stmt_span, stmt_store));

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
                        Spanned::new(op.span, op_inner.to_binary_op()),
                        Spanned::new(target_span, ValueWithImplications::simple(Value::Compile(target_eval))),
                        right_eval.map_inner(ValueWithImplications::Compile),
                    )?
                    .into_value();
                    let value = match value {
                        Value::Compile(value) => value,
                        _ => {
                            return Err(diags.report_internal_error(
                                stmt_span,
                                "binary op on compile values should result in compile value again",
                            ));
                        }
                    };

                    Spanned::new(stmt_span, value)
                }
            };

            // check type
            if let Some(var_ty) = &target_base_var_info.ty {
                let reason = TypeContainsReason::Assignment {
                    span_target: target_span,
                    span_target_ty: var_ty.span,
                };
                check_type_contains_compile_value(diags, reason, &target_expected_ty, value_eval.as_ref(), true)?;
            }

            // do assignment
            let result_value = target_steps.set_compile_value(diags, target_base_eval, op.span, value_eval)?;
            Value::Compile(result_value)
        };

        flow.var_set(target_base.inner, stmt_span, Ok(result));
        Ok(())
    }

    fn convert_variable_to_new_ir_variable(
        &mut self,
        flow: &mut FlowHardware,
        target_span: Span,
        target_base: Spanned<Variable>,
        target_base_eval: &Value,
    ) -> DiagResult<HardwareValue<Spanned<HardwareType>, IrVariable>> {
        let refs = self.refs;
        let diags = refs.diags;

        // pick a type and convert the current base value to hardware
        // TODO allow just inferring types, the user can specify one if they really want to
        let target_var_info = flow.var_info(target_base)?;
        let target_base_ty = target_var_info.ty.as_ref().ok_or_else(|| {
            let diag = Diagnostic::new("variable needs type annotation")
                .add_error(
                    target_span,
                    "for the assignment here the variable needs to be converted to hardware",
                )
                .add_info(target_var_info.span_decl, "variable declared without a type here")
                .finish();
            diags.report(diag)
        })?;
        let target_base_ty_span = target_base_ty.span;

        let target_base_ty_hw = target_base_ty.inner.as_hardware_type(refs).map_err(|_| {
            let diag = Diagnostic::new("variable needs hardware type because it is assigned a hardware value")
                .add_error(
                    target_span,
                    format!(
                        "actual type `{}` not representable in hardware",
                        target_base_ty.inner.diagnostic_string()
                    ),
                )
                .add_info(target_base_ty_span, "variable type set here")
                .finish();
            diags.report(diag)
        })?;

        let target_base_ir_expr =
            target_base_eval.as_hardware_value(refs, &mut self.large, target_base.span, &target_base_ty_hw)?;

        let debug_info_id = target_var_info.id.str(refs.fixed.source).map(str::to_owned);
        let result = store_ir_expression_in_new_variable(refs, flow, target_span, debug_info_id, target_base_ir_expr)?;

        Ok(HardwareValue {
            ty: Spanned::new(target_base_ty_span, result.ty),
            domain: result.domain,
            expr: result.expr,
        })
    }

    fn check_assignment_domains(
        &self,
        block_kind: BlockKind,
        condition_domains: impl Iterator<Item = Spanned<ValueDomain>>,
        op_span: Span,
        target_base_domain: Spanned<ValueDomain>,
        steps: &ArraySteps,
        value_domain: Spanned<ValueDomain>,
    ) -> DiagResult {
        match block_kind {
            BlockKind::Combinatorial => {
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
                        condition_domain,
                        "condition to target in combinatorial block",
                    );
                    check = check.and(c);
                }

                check
            }
            BlockKind::Clocked(block_domain) => {
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

// TODO move to better place, maybe in Flow?
pub fn store_ir_expression_in_new_variable(
    refs: CompileRefs,
    flow: &mut FlowHardware,
    span: Span,
    debug_info_id: Option<String>,
    expr: HardwareValue,
) -> DiagResult<HardwareValue<HardwareType, IrVariable>> {
    let var_ir_info = IrVariableInfo {
        ty: expr.ty.as_ir(refs),
        debug_info_span: span,
        debug_info_id,
    };

    let var_ir = flow.new_ir_variable(var_ir_info);

    let stmt_store = IrStatement::Assign(IrAssignmentTarget::variable(var_ir), expr.expr);
    flow.push_ir_statement(Spanned::new(span, stmt_store));

    Ok(HardwareValue {
        ty: expr.ty,
        domain: expr.domain,
        expr: var_ir,
    })
}
