use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAssignmentTargetBase, IrBlock, IrClockedProcess, IrDatabase,
    IrExpression, IrExpressionLarge, IrIfStatement, IrModuleChild, IrModuleInfo, IrPortConnection, IrPortInfo,
    IrStatement, IrTargetStep, IrType, IrVariables, IrWireOrPort,
};
use crate::syntax::ast::PortDirection;
use crate::syntax::pos::Span;
use crate::util::big_int::BigUint;
use itertools::zip_eq;
use std::borrow::Cow;
use unwrap_match::unwrap_match;

// TODO expand all of this
impl IrDatabase {
    pub fn validate(&self, diags: &Diagnostics) -> Result<(), ErrorGuaranteed> {
        for (_, info) in self.modules.iter() {
            info.validate(self, diags)?;
        }
        Ok(())
    }
}

impl IrModuleInfo {
    pub fn validate(&self, db: &IrDatabase, diags: &Diagnostics) -> Result<(), ErrorGuaranteed> {
        let no_variables = &IrVariables::new();

        for child in &self.children {
            match child {
                IrModuleChild::ClockedProcess(process) => {
                    let IrClockedProcess {
                        locals,
                        clock_signal,
                        clock_block,
                        async_reset_signal_and_block,
                    } = process;

                    clock_signal
                        .inner
                        .validate(diags, self, no_variables, clock_signal.span)?;
                    check_type_match(
                        diags,
                        clock_signal.span,
                        &IrType::Bool,
                        &clock_signal.inner.ty(self, no_variables),
                    )?;
                    clock_block.validate(diags, self, locals)?;

                    if let Some((reset_signal, reset_block)) = async_reset_signal_and_block {
                        reset_signal
                            .inner
                            .validate(diags, self, no_variables, reset_signal.span)?;
                        check_type_match(
                            diags,
                            reset_signal.span,
                            &IrType::Bool,
                            &reset_signal.inner.ty(self, no_variables),
                        )?;
                        reset_block.validate(diags, self, locals)?;
                    }
                }
                IrModuleChild::CombinatorialProcess(process) => {
                    process.block.validate(diags, self, &process.locals)?;
                }
                IrModuleChild::ModuleInstance(instance) => {
                    let child_module_info = &db.modules[instance.module];

                    for ((_, port_info), connection) in zip_eq(&child_module_info.ports, &instance.port_connections) {
                        let IrPortInfo {
                            direction,
                            ref ty,
                            debug_info_id: _,
                            debug_info_ty: _,
                            debug_info_domain: _,
                        } = *port_info;

                        let conn_ty = match &connection.inner {
                            IrPortConnection::Input(expr) => {
                                check_dir_match(diags, connection.span, PortDirection::Input, direction)?;
                                expr.inner.validate(diags, self, no_variables, expr.span)?;
                                expr.inner.ty(self, no_variables)
                            }
                            &IrPortConnection::Output(expr) => {
                                check_dir_match(diags, connection.span, PortDirection::Output, direction)?;
                                match expr {
                                    Some(IrWireOrPort::Wire(wire)) => self.wires[wire].ty.clone(),
                                    Some(IrWireOrPort::Port(port)) => self.ports[port].ty.clone(),
                                    None => continue,
                                }
                            }
                        };
                        check_type_match(diags, connection.span, ty, &conn_ty)?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl IrBlock {
    pub fn validate(
        &self,
        diags: &Diagnostics,
        module: &IrModuleInfo,
        locals: &IrVariables,
    ) -> Result<(), ErrorGuaranteed> {
        for stmt in &self.statements {
            match &stmt.inner {
                IrStatement::Assign(target, expr) => {
                    let target_ty = assignment_target_ty(module, locals, target);

                    expr.validate(diags, module, locals, stmt.span)?;
                    let expr_ty = expr.ty(module, locals);

                    check_type_match(diags, stmt.span, target_ty.as_ref(), &expr_ty)?;
                }
                IrStatement::Block(block) => {
                    block.validate(diags, module, locals)?;
                }
                IrStatement::If(if_stmt) => {
                    let IrIfStatement {
                        condition,
                        then_block,
                        else_block,
                    } = if_stmt;

                    condition.validate(diags, module, locals, stmt.span)?;
                    check_type_match(diags, stmt.span, &IrType::Bool, &condition.ty(module, locals))?;

                    then_block.validate(diags, module, locals)?;

                    if let Some(else_block) = else_block {
                        else_block.validate(diags, module, locals)?;
                    }
                }
                IrStatement::PrintLn(_) => {}
            }
        }
        Ok(())
    }
}

fn assignment_target_ty<'a>(
    module: &'a IrModuleInfo,
    locals: &'a IrVariables,
    target: &IrAssignmentTarget,
) -> Cow<'a, IrType> {
    let IrAssignmentTarget { base, steps } = target;

    let base_ty = match *base {
        IrAssignmentTargetBase::Port(port) => Cow::Borrowed(&module.ports[port].ty),
        IrAssignmentTargetBase::Register(reg) => Cow::Borrowed(&module.registers[reg].ty),
        IrAssignmentTargetBase::Wire(wire) => Cow::Borrowed(&module.wires[wire].ty),
        IrAssignmentTargetBase::Variable(var) => Cow::Borrowed(&locals[var].ty),
    };

    let mut curr_ty = base_ty;
    let mut slice_lens = vec![];

    for step in steps {
        curr_ty = Cow::Owned(unwrap_match!(&*curr_ty, IrType::Array(inner, _) => &**inner).clone());
        match step {
            IrTargetStep::ArrayIndex(_index) => {
                // no slice len
            }
            IrTargetStep::ArraySlice(_start, len) => {
                slice_lens.push(len.clone());
            }
        };
    }

    slice_lens.into_iter().rev().fold(curr_ty, |acc, len| {
        Cow::Owned(IrType::Array(Box::new(acc.into_owned()), len))
    })
}

impl IrExpression {
    pub fn validate(
        &self,
        diags: &Diagnostics,
        module: &IrModuleInfo,
        locals: &IrVariables,
        span: Span,
    ) -> Result<(), ErrorGuaranteed> {
        let large = &module.large;

        // validate operands
        let mut any_err = Ok(());
        self.for_each_expression_operand(large, &mut |op| {
            any_err = any_err.and(op.validate(diags, module, locals, span))
        });
        any_err?;

        // validate self
        match self {
            &IrExpression::Large(expr) => match &large[expr] {
                IrExpressionLarge::ArrayLiteral(inner_ty, len, values) => {
                    let mut actual_len = BigUint::ZERO;
                    for value in values {
                        match value {
                            IrArrayLiteralElement::Single(value) => {
                                let value_ty = value.ty(module, locals);
                                check_type_match(diags, span, inner_ty, &value_ty)?;
                                actual_len += 1u32;
                            }
                            IrArrayLiteralElement::Spread(value) => {
                                let value_ty = value.ty(module, locals);
                                let len = match value.ty(module, locals) {
                                    IrType::Array(_, len) => len,
                                    _ => unreachable!(),
                                };
                                check_type_match(
                                    diags,
                                    span,
                                    &IrType::Array(Box::new(inner_ty.clone()), len.clone()),
                                    &value_ty,
                                )?;
                                actual_len += len;
                            }
                        }
                    }
                    if &actual_len != len {
                        let msg = format!("array literal length mismatch: expected {} but got {}", len, actual_len);
                        return Err(diags.report_internal_error(span, msg));
                    }
                }
                // TODO expand
                _ => {}
            },
            // TODO expand
            _ => {}
        }

        Ok(())
    }
}

fn check_type_match(
    diags: &Diagnostics,
    span: Span,
    expected: &IrType,
    actual: &IrType,
) -> Result<(), ErrorGuaranteed> {
    if expected != actual {
        let msg = format!("ir type mismatch: expected {:?}, got {:?}", expected, actual);
        return Err(diags.report_internal_error(span, msg));
    }
    Ok(())
}

fn check_dir_match(
    diags: &Diagnostics,
    span: Span,
    expected: PortDirection,
    actual: PortDirection,
) -> Result<(), ErrorGuaranteed> {
    if expected != actual {
        let msg = format!("ir port direction mismatch: expected {:?}, got {:?}", expected, actual);
        return Err(diags.report_internal_error(span, msg));
    }
    Ok(())
}
