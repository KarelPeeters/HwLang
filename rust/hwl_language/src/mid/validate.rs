use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::signal::Polarized;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrClockedProcess, IrDatabase, IrEnumType,
    IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrModule, IrModuleChild, IrModuleExternalInstance,
    IrModuleInfo, IrModuleInternalInstance, IrPortConnection, IrPortInfo, IrSignal, IrStatement, IrString,
    IrStringSubstitution, IrStructType, IrTargetStep, IrType, IrVariables,
};
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::syntax::pos::Span;
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::range::{ClosedNonEmptyRange, ClosedRange};
use indexmap::IndexSet;
use itertools::zip_eq;

// TODO expand all of this
impl IrDatabase {
    pub fn validate(&self, diags: &Diagnostics) -> DiagResult {
        let IrDatabase {
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
                        check_type_is_bool(
                            diags,
                            reset_signal.span,
                            &reset_signal_inner_expr.ty(self, no_variables),
                        )?;

                        // TODO check drivers, ie. only driven and reset in one process
                        for reset in resets {
                            let &(reg, ref value) = &reset.inner;

                            let empty_locals = IrVariables::new();
                            let reg_ty = IrExpression::Signal(reg).ty(self, &empty_locals);

                            check_type_match(diags, reset.span, &reg_ty, &value.ty(self, &empty_locals))?
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

                        let conn_ty = match connection.inner {
                            IrPortConnection::Input(expr) => {
                                check_dir_match(diags, connection.span, PortDirection::Input, direction)?;
                                let inner_expr = IrExpression::Signal(expr);
                                inner_expr.validate(diags, self, no_variables, connection.span)?;
                                inner_expr.ty(self, no_variables)
                            }
                            IrPortConnection::Output(expr) => {
                                check_dir_match(diags, connection.span, PortDirection::Output, direction)?;
                                match expr {
                                    Some(IrSignal::Wire(wire)) => self.wires[wire].ty.clone(),
                                    Some(IrSignal::Port(port)) => self.ports[port].ty.clone(),
                                    None => continue,
                                }
                            }
                        };
                        check_type_match(diags, connection.span, ty, &conn_ty)?;
                    }
                }
                IrModuleChild::ModuleExternalInstance(instance) => {
                    let IrModuleExternalInstance {
                        name: _,
                        module_name,
                        generic_args,
                        port_names,
                        port_connections,
                    } = instance;

                    if !external_modules.contains(module_name) {
                        let msg = format!("IR ModuleExternalInstance `{module_name}` not found in external modules");
                        return Err(diags.report_error_internal(child.span, msg));
                    }

                    // the IR does not store type information for external modules, we just have to trust these
                    let _ = generic_args;
                    let _ = port_connections;

                    if port_names.len() != port_connections.len() {
                        return Err(
                            diags.report_error_internal(child.span, "IR ModuleExternalInstance port length mismatch")
                        );
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

                    check_type_match(diags, stmt.span, &target_ty, &expr_ty)?;
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
                    check_type_is_bool(diags, stmt.span, &condition.ty(module, locals))?;

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
                        let msg = format!(
                            "IR IrForStatement variable must contain loop range: variable {index_range:?} but loop {range:?}"
                        );
                        return Err(diags.report_error_internal(stmt.span, msg));
                    }

                    block.validate(diags, module, locals)?;
                }
                IrStatement::Print(pieces) => validate_string(diags, module, locals, stmt.span, pieces)?,
                IrStatement::AssertFailed => {}
            }
        }
        Ok(())
    }
}

fn validate_string(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    locals: &IrVariables,
    span: Span,
    s: &IrString,
) -> DiagResult {
    for p in s {
        match p {
            StringPiece::Literal(_) => {}
            StringPiece::Substitute(p) => match p {
                IrStringSubstitution::Integer(p, _) => {
                    p.validate(diags, module, locals, span)?;
                    check_type_is_int(diags, span, &p.ty(module, locals))?;
                }
            },
        }
    }
    Ok(())
}

fn assignment_target_ty<'a>(module: &'a IrModuleInfo, locals: &'a IrVariables, target: &IrAssignmentTarget) -> IrType {
    let IrAssignmentTarget { base, steps } = target;

    let mut curr_ty = base.as_expression().ty(module, locals);

    for step in steps {
        curr_ty = match step {
            IrTargetStep::ArrayIndex(_index) => {
                let (inner_ty, _len) = curr_ty.unwrap_array();
                inner_ty
            }
            IrTargetStep::ArraySlice { start: _, len: length } => {
                let (inner_ty, _) = curr_ty.unwrap_array();
                IrType::Array(Box::new(inner_ty), length.clone())
            }
        };
    }

    curr_ty
}

impl IrExpression {
    pub fn validate(&self, diags: &Diagnostics, module: &IrModuleInfo, locals: &IrVariables, span: Span) -> DiagResult {
        let large = &module.large;

        // validate operands
        let mut any_err = Ok(());
        self.for_each_operand(large, &mut |op| {
            any_err = any_err.and(op.validate(diags, module, locals, span))
        });
        any_err?;

        // validate self
        match self {
            // always valid
            IrExpression::Bool(_) | IrExpression::Int(_) => {}

            // assert that signal/var exists in this context
            &IrExpression::Signal(sig) => match sig {
                IrSignal::Port(port) => {
                    let _ = module.ports[port];
                }
                IrSignal::Wire(wire) => {
                    let _ = module.wires[wire];
                }
            },
            &IrExpression::Variable(var) => {
                let _ = locals[var];
            }

            // actual type checks
            &IrExpression::Large(expr) => match &large[expr] {
                IrExpressionLarge::Undefined(_ty) => {
                    // always valid
                }
                IrExpressionLarge::BoolNot(inner) => {
                    check_type_is_bool(diags, span, &inner.ty(module, locals))?;
                }
                IrExpressionLarge::BoolBinary(_op, left, right) => {
                    check_type_is_bool(diags, span, &left.ty(module, locals))?;
                    check_type_is_bool(diags, span, &right.ty(module, locals))?;
                }
                IrExpressionLarge::IntArithmetic(_op, _range, left, right) => {
                    check_type_is_int(diags, span, &left.ty(module, locals))?;
                    check_type_is_int(diags, span, &right.ty(module, locals))?;
                }
                IrExpressionLarge::IntCompare(_op, left, right) => {
                    check_type_is_int(diags, span, &left.ty(module, locals))?;
                    check_type_is_int(diags, span, &right.ty(module, locals))?;
                }
                IrExpressionLarge::TupleLiteral(_values) => {
                    // always valid
                }
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
                                let (value_inner_ty, value_len) = check_type_is_array(diags, span, &value_ty)?;
                                check_type_match(diags, span, inner_ty, value_inner_ty)?;
                                actual_len += value_len;
                            }
                        }
                    }
                    if &actual_len != len {
                        return Err(diags.report_error_internal(span, "IR ArrayLiteral length mismatch"));
                    }
                }
                IrExpressionLarge::StructLiteral(ty, values) => {
                    if ty.fields.len() != values.len() {
                        return Err(diags.report_error_internal(span, "IR StructLiteral wrong field count"));
                    }
                    for ((_, field_ty), field_value) in zip_eq(&ty.fields, values) {
                        let value_ty = field_value.ty(module, locals);
                        check_type_match(diags, span, field_ty, &value_ty)?;
                    }
                }
                &IrExpressionLarge::EnumLiteral(ref ty, variant, ref payload) => {
                    if variant >= ty.variants.len() {
                        return Err(diags.report_error_internal(span, "IR EnumLiteral invalid variant"));
                    }

                    let ty_payload = &ty.variants[variant];
                    match (ty_payload, payload) {
                        (None, None) => {}
                        (Some(payload_ty), Some(payload_expr)) => {
                            let value_ty = payload_expr.ty(module, locals);
                            check_type_match(diags, span, payload_ty, &value_ty)?;
                        }
                        _ => return Err(diags.report_error_internal(span, "IR EnumLiteral payload mismatch")),
                    }
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    let base_ty = base.ty(module, locals);
                    let (_, base_len) = check_type_is_array(diags, span, &base_ty)?;

                    let index_ty = index.ty(module, locals);
                    let index_range = check_type_is_int(diags, span, &index_ty)?;

                    let valid_range = ClosedRange {
                        start: BigInt::ZERO,
                        end: BigInt::from(base_len),
                    };
                    if !valid_range.contains_range(index_range.as_ref()) {
                        return Err(diags.report_error_internal(span, "IR ArrayIndex out of bounds"));
                    }
                }
                IrExpressionLarge::ArraySlice { base, start, len } => {
                    let base_ty = base.ty(module, locals);
                    let (_, base_len) = check_type_is_array(diags, span, &base_ty)?;

                    let start_ty = start.ty(module, locals);
                    let start_range = check_type_is_int(diags, span, &start_ty)?;

                    let valid_range = ClosedRange {
                        start: BigInt::ZERO,
                        end: base_len - len + 1,
                    };
                    if !valid_range.contains_range(start_range.as_ref()) {
                        return Err(diags.report_error_internal(span, "IR ArraySlice out of bounds"));
                    }
                }
                &IrExpressionLarge::TupleIndex { ref base, index } => {
                    let base_ty = base.ty(module, locals);
                    let base_ty = check_type_is_tuple(diags, span, &base_ty)?;

                    if index >= base_ty.len() {
                        return Err(diags.report_error_internal(span, "IR TupleIndex index out of bounds"));
                    }
                }
                &IrExpressionLarge::StructField { ref base, field } => {
                    let base_ty = base.ty(module, locals);
                    let base_ty = check_type_is_struct(diags, span, &base_ty)?;

                    if field >= base_ty.fields.len() {
                        return Err(diags.report_error_internal(span, "IR StructField out of bounds"));
                    }
                }
                IrExpressionLarge::EnumTag { base } => {
                    let base_ty = base.ty(module, locals);
                    let _ = check_type_is_enum(diags, span, &base_ty)?;
                }
                &IrExpressionLarge::EnumPayload { ref base, variant } => {
                    let base_ty = base.ty(module, locals);
                    let base_ty = check_type_is_enum(diags, span, &base_ty)?;

                    if variant >= base_ty.variants.len() {
                        return Err(diags.report_error_internal(span, "IR EnumPayload invalid variant"));
                    }

                    if base_ty.variants[variant].is_none() {
                        return Err(diags.report_error_internal(span, "IR EnumPayload variant has no payload"));
                    }
                }
                IrExpressionLarge::ToBits(ty, expr) => {
                    if ty != &expr.ty(module, locals) {
                        return Err(diags.report_error_internal(span, "IR ToBits type mismatch"));
                    }
                }
                IrExpressionLarge::FromBits(ty, expr) => {
                    let expr_ty = expr.ty(module, locals);
                    if expr_ty != IrType::Array(Box::new(IrType::Bool), ty.size_bits()) {
                        return Err(diags.report_error_internal(span, "IR FromBits type mismatch"));
                    }
                }
                IrExpressionLarge::ExpandIntRange(outer, inner) => {
                    let inner_ty = inner.ty(module, locals);
                    let inner_range = check_type_is_int(diags, span, &inner_ty)?;

                    if !outer.contains_range(inner_range.as_ref()) {
                        return Err(diags.report_error_internal(span, "IR ExpandIntRange outer does not contain inner"));
                    }
                }
                IrExpressionLarge::ConstrainIntRange(outer, inner) => {
                    let inner_ty = inner.ty(module, locals);
                    let inner_range = check_type_is_int(diags, span, &inner_ty)?;

                    if !inner_range.contains_range(outer.as_ref()) {
                        return Err(
                            diags.report_error_internal(span, "IR ConstrainIntRange inner does not contain outer")
                        );
                    }
                }
            },
        }

        Ok(())
    }
}

fn check_type_match(diags: &Diagnostics, span: Span, expected: &IrType, actual: &IrType) -> DiagResult {
    if expected != actual {
        let msg = format!("IR type mismatch: expected {expected:?}, got {actual:?}");
        return Err(diags.report_error_internal(span, msg));
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
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_bool(diags: &Diagnostics, span: Span, actual: &IrType) -> DiagResult {
    check_type_match(diags, span, &IrType::Bool, actual)
}

fn check_type_is_array<'t>(
    diags: &Diagnostics,
    span: Span,
    actual: &'t IrType,
) -> DiagResult<(&'t IrType, &'t BigUint)> {
    match actual {
        IrType::Array(inner, len) => Ok((inner, len)),
        _ => {
            let msg = format!("IR type mismatch: expected array, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_tuple<'t>(diags: &Diagnostics, span: Span, actual: &'t IrType) -> DiagResult<&'t [IrType]> {
    match actual {
        IrType::Tuple(elements) => Ok(elements),
        _ => {
            let msg = format!("IR type mismatch: expected tuple, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_struct<'t>(diags: &Diagnostics, span: Span, actual: &'t IrType) -> DiagResult<&'t IrStructType> {
    match actual {
        IrType::Struct(ty) => Ok(ty),
        _ => {
            let msg = format!("IR type mismatch: expected struct, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_enum<'t>(diags: &Diagnostics, span: Span, actual: &'t IrType) -> DiagResult<&'t IrEnumType> {
    match actual {
        IrType::Enum(ty) => Ok(ty),
        _ => {
            let msg = format!("IR type mismatch: expected enum, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_dir_match(diags: &Diagnostics, span: Span, expected: PortDirection, actual: PortDirection) -> DiagResult {
    if expected != actual {
        let msg = format!("IR port direction mismatch: expected {expected:?}, got {actual:?}");
        return Err(diags.report_error_internal(span, msg));
    }
    Ok(())
}
