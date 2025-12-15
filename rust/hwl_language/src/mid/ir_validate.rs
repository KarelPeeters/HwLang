use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::signal::Polarized;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrClockedProcess, IrDatabase, IrExpression,
    IrExpressionLarge, IrForStatement, IrIfStatement, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo,
    IrModuleInternalInstance, IrPortConnection, IrPortInfo, IrStatement, IrStringSubstitution, IrTargetStep, IrType,
    IrVariables, IrWireOrPort,
};
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::syntax::pos::Span;
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::range::ClosedNonEmptyRange;
use indexmap::IndexSet;
use itertools::zip_eq;
use std::borrow::Cow;
use unwrap_match::unwrap_match;

// TODO expand all of this
impl IrDatabase {
    pub fn validate(&self, diags: &Diagnostics) -> DiagResult {
        let IrDatabase {
            top_module: _,
            modules,
            external_modules,
        } = self;
        for (_, info) in self.modules.iter() {
            info.validate(diags, modules, external_modules)?;
        }
        Ok(())
    }
}

impl IrModuleInfo {
    pub fn validate(
        &self,
        diags: &Diagnostics,
        modules: &Arena<IrModule, IrModuleInfo>,
        external_modules: &IndexSet<String>,
    ) -> DiagResult {
        let no_variables = &IrVariables::new();

        for child in &self.children {
            match &child.inner {
                IrModuleChild::ClockedProcess(process) => {
                    let IrClockedProcess {
                        locals,
                        clock_signal,
                        clock_block,
                        async_reset,
                    } = process;
                    let Polarized {
                        inverted: _,
                        signal: clock_signal_inner,
                    } = clock_signal.inner;

                    let clock_signal_inner_expr = IrExpression::Signal(clock_signal_inner);
                    clock_signal_inner_expr.validate(diags, self, no_variables, clock_signal.span)?;
                    check_type_match(
                        diags,
                        clock_signal.span,
                        &IrType::Bool,
                        &clock_signal_inner_expr.ty(self, no_variables),
                    )?;
                    clock_block.validate(diags, self, locals)?;

                    if let Some(async_reset) = async_reset {
                        let IrAsyncResetInfo {
                            signal: reset_signal,
                            resets,
                        } = async_reset;
                        let Polarized {
                            inverted: _,
                            signal: reset_signal_inner,
                        } = reset_signal.inner;

                        let reset_signal_inner_expr = IrExpression::Signal(reset_signal_inner);
                        reset_signal_inner_expr.validate(diags, self, no_variables, reset_signal.span)?;
                        check_type_match(
                            diags,
                            reset_signal.span,
                            &IrType::Bool,
                            &reset_signal_inner_expr.ty(self, no_variables),
                        )?;

                        // TODO check drivers, ie. only driven and reset in one process
                        for reset in resets {
                            let (reg, value) = &reset.inner;
                            let reg_info = &self.registers[*reg];
                            let empty_locals = IrVariables::new();
                            check_type_match(diags, reset.span, &reg_info.ty, &value.ty(self, &empty_locals))?
                        }
                    }
                }
                IrModuleChild::CombinatorialProcess(process) => {
                    process.block.validate(diags, self, &process.locals)?;
                }
                IrModuleChild::ModuleInternalInstance(instance) => {
                    let &IrModuleInternalInstance {
                        name: _,
                        module,
                        ref port_connections,
                    } = instance;
                    // TODO check name unique
                    let child_module_info = &modules[module];

                    for ((_, port_info), connection) in zip_eq(&child_module_info.ports, port_connections) {
                        let IrPortInfo {
                            name: _,
                            direction,
                            ref ty,
                            debug_span: _,
                            debug_info_ty: _,
                            debug_info_domain: _,
                        } = *port_info;

                        let conn_ty = match &connection.inner {
                            IrPortConnection::Input(expr) => {
                                check_dir_match(diags, connection.span, PortDirection::Input, direction)?;
                                let inner_expr = IrExpression::Signal(expr.inner);
                                inner_expr.validate(diags, self, no_variables, expr.span)?;
                                inner_expr.ty(self, no_variables)
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
                IrModuleChild::ModuleExternalInstance(instance) => {
                    // TODO check name unique
                    let IrModuleExternalInstance {
                        name: _,
                        module_name,
                        generic_args,
                        port_names,
                        port_connections,
                    } = instance;

                    if !external_modules.contains(module_name) {
                        let msg = format!("IR external module `{module_name}` not found in external modules");
                        return Err(diags.report_internal_error(child.span, msg));
                    }

                    // TODO ideally we could access the generic and port types here,
                    //   but that would require some generics support in the IR, which we want to avoid
                    let _ = generic_args;
                    let _ = port_connections;

                    if port_names.len() != port_connections.len() {
                        return Err(diags.report_internal_error(child.span, "IR port length mismatch"));
                    }
                }
            }
        }

        Ok(())
    }
}

impl IrBlock {
    pub fn validate(&self, diags: &Diagnostics, module: &IrModuleInfo, locals: &IrVariables) -> DiagResult {
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
                IrStatement::For(for_stmt) => {
                    let &IrForStatement {
                        index,
                        ref range,
                        ref block,
                    } = for_stmt;
                    let index_ty = IrExpression::Variable(index).ty(module, locals);
                    let index_range = check_type_is_int(diags, stmt.span, &index_ty)?;

                    if !index_range.contains_range(range.as_ref()) {
                        let msg = format!("IR for loop range mismatch: variable {index_range:?} but loop {range:?}");
                        return Err(diags.report_internal_error(stmt.span, msg));
                    }

                    block.validate(diags, module, locals)?;
                }
                IrStatement::Print(pieces) => {
                    for p in pieces {
                        match p {
                            StringPiece::Literal(_) => {}
                            StringPiece::Substitute(p) => match p {
                                IrStringSubstitution::Integer(p, _) => {
                                    p.validate(diags, module, locals, stmt.span)?;
                                    check_type_is_int(diags, stmt.span, &p.ty(module, locals))?;
                                }
                            },
                        }
                    }
                }
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

    let base_ty = Cow::Owned(base.as_expression().ty(module, locals));

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
    pub fn validate(&self, diags: &Diagnostics, module: &IrModuleInfo, locals: &IrVariables, span: Span) -> DiagResult {
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
                        let msg = format!("IR array literal length mismatch: expected {len} but got {actual_len}");
                        return Err(diags.report_internal_error(span, msg));
                    }
                }
                IrExpressionLarge::ToBits(ty, expr) => {
                    if ty != &expr.ty(module, locals) {
                        return Err(diags.report_internal_error(span, "IR ToBits type mismatch"));
                    }
                }
                IrExpressionLarge::FromBits(ty, expr) => {
                    if let IrType::Array(element, len) = expr.ty(module, locals)
                        && let IrType::Bool = *element
                        && len == ty.size_bits()
                    {
                        return Ok(());
                    }
                    return Err(diags.report_internal_error(span, "IR FromInt width mismatch"));
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

fn check_type_match(diags: &Diagnostics, span: Span, expected: &IrType, actual: &IrType) -> DiagResult {
    if expected != actual {
        let msg = format!("IR type mismatch: expected {expected:?}, got {actual:?}");
        return Err(diags.report_internal_error(span, msg));
    }
    Ok(())
}

fn check_type_is_int<'t>(
    diags: &Diagnostics,
    span: Span,
    actual: &'t IrType,
) -> DiagResult<&'t ClosedNonEmptyRange<BigInt>> {
    match actual {
        IrType::Int(range) => Ok(range),
        _ => {
            let msg = format!("IR type mismatch: expected int, got {actual:?}");
            Err(diags.report_internal_error(span, msg))
        }
    }
}

fn check_dir_match(diags: &Diagnostics, span: Span, expected: PortDirection, actual: PortDirection) -> DiagResult {
    if expected != actual {
        let msg = format!("IR port direction mismatch: expected {expected:?}, got {actual:?}");
        return Err(diags.report_internal_error(span, msg));
    }
    Ok(())
}
