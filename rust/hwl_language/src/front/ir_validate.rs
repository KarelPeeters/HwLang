use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::ir::{
    IrAssignmentTarget, IrBlock, IrDatabase, IrExpression, IrModuleChild, IrModuleInfo, IrPortConnection, IrStatement,
    IrType, IrVariables, IrWireOrPort,
};
use crate::syntax::ast::{ArrayLiteralElement, IfStatement};
use crate::syntax::pos::Span;

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
        let no_variables = &IrVariables::default();

        for child in &self.children {
            match child {
                IrModuleChild::ClockedProcess(process) => {
                    process
                        .domain
                        .inner
                        .clock
                        .validate(diags, self, no_variables, process.domain.span)?;
                    process
                        .domain
                        .inner
                        .reset
                        .validate(diags, self, no_variables, process.domain.span)?;

                    check_type_match(
                        diags,
                        process.domain.span,
                        &IrType::Bool,
                        &process.domain.inner.clock.ty(self, no_variables),
                    )?;
                    check_type_match(
                        diags,
                        process.domain.span,
                        &IrType::Bool,
                        &process.domain.inner.reset.ty(self, no_variables),
                    )?;

                    process.on_reset.validate(diags, self, &process.locals)?;
                    process.on_clock.validate(diags, self, &process.locals)?;
                }
                IrModuleChild::CombinatorialProcess(process) => {
                    process.block.validate(diags, self, &process.locals)?;
                }
                IrModuleChild::ModuleInstance(instance) => {
                    for (&port, connection) in &instance.port_connections {
                        let port_ty = &db.modules[instance.module].ports[port].ty;
                        let conn_ty = match &connection.inner {
                            IrPortConnection::Input(expr) => {
                                expr.inner.validate(diags, self, no_variables, expr.span)?;
                                expr.inner.ty(self, no_variables)
                            }
                            &IrPortConnection::Output(Some(IrWireOrPort::Wire(wire))) => self.wires[wire].ty.clone(),
                            &IrPortConnection::Output(Some(IrWireOrPort::Port(port))) => self.ports[port].ty.clone(),
                            &IrPortConnection::Output(None) => continue,
                        };
                        check_type_match(diags, connection.span, port_ty, &conn_ty)?;
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
                    let target_ty = match *target {
                        IrAssignmentTarget::Port(port) => &module.ports[port].ty,
                        IrAssignmentTarget::Register(reg) => &module.registers[reg].ty,
                        IrAssignmentTarget::Wire(wire) => &module.wires[wire].ty,
                        IrAssignmentTarget::Variable(var) => &locals[var].ty,
                    };

                    expr.validate(diags, module, locals, stmt.span)?;
                    let expr_ty = expr.ty(module, locals);

                    check_type_match(diags, stmt.span, target_ty, &expr_ty)?;
                }
                IrStatement::Block(block) => {
                    block.validate(diags, module, locals)?;
                }
                IrStatement::If(if_stmt) => {
                    let IfStatement {
                        initial_if,
                        else_ifs,
                        final_else,
                    } = if_stmt;

                    // if
                    initial_if.cond.validate(diags, module, locals, stmt.span)?;
                    check_type_match(diags, stmt.span, &IrType::Bool, &initial_if.cond.ty(module, locals))?;

                    initial_if.block.validate(diags, module, locals)?;

                    // else if
                    for else_if in else_ifs {
                        else_if.cond.validate(diags, module, locals, stmt.span)?;
                        check_type_match(diags, stmt.span, &IrType::Bool, &else_if.cond.ty(module, locals))?;

                        else_if.block.validate(diags, module, locals)?;
                    }

                    // else
                    if let Some(else_block) = &final_else {
                        else_block.validate(diags, module, locals)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl IrExpression {
    pub fn validate(
        &self,
        diags: &Diagnostics,
        module: &IrModuleInfo,
        locals: &IrVariables,
        span: Span,
    ) -> Result<(), ErrorGuaranteed> {
        // validate operands
        let mut any_err = Ok(());
        self.for_each_expression_operand(&mut |op| any_err = any_err.and(op.validate(diags, module, locals, span)));
        any_err?;

        // validate self
        match self {
            IrExpression::TupleLiteral(_) => {}
            IrExpression::ArrayLiteral(inner_ty, values) => {
                for value in values {
                    let ArrayLiteralElement { spread, value } = value;
                    let value_ty = value.ty(module, locals);

                    if spread.is_some() {
                        let len = match value.ty(module, locals) {
                            IrType::Array(_, len) => len,
                            _ => unreachable!(),
                        };
                        check_type_match(diags, span, &IrType::Array(Box::new(inner_ty.clone()), len), &value_ty)?;
                    } else {
                        check_type_match(diags, span, inner_ty, &value_ty)?
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }
}

fn check_type_match(diags: &Diagnostics, span: Span, left: &IrType, right: &IrType) -> Result<(), ErrorGuaranteed> {
    if left == right {
        Ok(())
    } else {
        let msg = format!("ir assignment type mismatch: {:?} vs {:?}", left, right);
        Err(diags.report_internal_error(span, msg))
    }
}
