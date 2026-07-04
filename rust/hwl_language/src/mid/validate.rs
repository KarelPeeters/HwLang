use crate::front::diagnostic::{DiagResult, Diagnostics};
use crate::front::signal::Polarized;
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrClockedProcess, IrDatabase, IrEnumType,
    IrExpression, IrExpressionLarge, IrForStatement, IrIfStatement, IrModule, IrModuleChild, IrModuleExternalInstance,
    IrModuleInfo, IrModuleInternalInstance, IrPortConnection, IrPortInfo, IrSignal, IrSignalOrVariable, IrSignals,
    IrStatement, IrString, IrStringSubstitution, IrStructType, IrType, IrVariables,
};
use crate::mid::steps::{IrTargetStepScalar, IrTargetStepSlice, IrTargetSteps};
use crate::syntax::ast::{PortDirection, StringPiece};
use crate::syntax::pos::Span;
use crate::util::arena::Arena;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::data::{IndexMapExt, VecExt};
use crate::util::range::{ClosedNonEmptyRange, ClosedRange};
use indexmap::IndexSet;
use itertools::zip_eq;

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
        let large = &self.large;
        let signals = &self.signals;
        let no_variables = &IrVariables::new();

        // validate ports
        let id_span = self.debug_info_id.span;
        let IrSignals {
            ports,
            wires: _,
            ports_named,
        } = signals;
        if ports.len() != ports_named.len() {
            return Err(diags.report_error_internal(id_span, "ports_named count mismatch"));
        }
        for (port, port_info) in ports {
            if ports_named.get(&port_info.name) != Some(&port) {
                return Err(diags.report_error_internal(id_span, "ports_named value mismatch"));
            }
        }

        // validate children
        for child in &self.children {
            match &child.inner {
                IrModuleChild::ClockedProcess(process) => {
                    let IrClockedProcess {
                        registers,
                        variables,
                        clock_signal,
                        clock_block,
                        async_reset,
                    } = process;
                    let Polarized {
                        inverted: _,
                        signal: clock_signal_inner,
                    } = clock_signal.inner;

                    // reset
                    if let Some(async_reset) = async_reset {
                        let IrAsyncResetInfo {
                            signal: reset_signal,
                            resets,
                        } = async_reset;
                        let Polarized {
                            inverted: _,
                            signal: reset_signal_inner,
                        } = reset_signal.inner;

                        let reset_signal_inner_expr = reset_signal_inner.as_expression();
                        reset_signal_inner_expr.validate(diags, self, no_variables, reset_signal.span)?;
                        check_type_is_bool(
                            diags,
                            reset_signal.span,
                            &reset_signal_inner_expr.ty(large, signals, no_variables),
                        )?;

                        for reset in resets {
                            let &(reg, ref value) = &reset.inner;

                            if !registers.contains(&reg) {
                                return Err(
                                    diags.report_error_internal(reset.span, "trying to reset non-driven signal")
                                );
                            }

                            let empty_variables = IrVariables::new();
                            let reg_ty = IrExpression::Signal(reg).ty(large, signals, &empty_variables);

                            check_type_match(diags, reset.span, &reg_ty, &value.ty(large, signals, &empty_variables))?
                        }
                    }

                    // clock
                    let clock_signal_inner_expr = clock_signal_inner.as_expression();
                    clock_signal_inner_expr.validate(diags, self, no_variables, clock_signal.span)?;
                    check_type_is_bool(
                        diags,
                        clock_signal.span,
                        &clock_signal_inner_expr.ty(large, signals, no_variables),
                    )?;
                    clock_block.validate(diags, self, Some(registers), variables)?;
                }
                IrModuleChild::CombinatorialProcess(process) => {
                    process.block.validate(diags, self, None, &process.variables)?;
                }
                IrModuleChild::ModuleInternalInstance(instance) => {
                    let &IrModuleInternalInstance {
                        name: _,
                        module,
                        ref port_connections,
                    } = instance;
                    let child_module_info = &modules[module];

                    for ((_, port_info), connection) in zip_eq(&child_module_info.signals.ports, port_connections) {
                        let IrPortInfo {
                            name: _,
                            direction: port_dir,
                            ty: ref port_ty,
                            debug_span: _,
                            debug_info_ty: _,
                            debug_info_domain: _,
                        } = *port_info;

                        let (conn_sig, conn_dir) = match connection.inner {
                            IrPortConnection::Input(signal) => (Some(signal), PortDirection::Input),
                            IrPortConnection::Output(signal) => (signal, PortDirection::Output),
                        };

                        check_dir_match(diags, connection.span, port_dir, conn_dir)?;
                        if let Some(conn_sig) = conn_sig {
                            conn_sig.validate(signals);
                            check_type_match(diags, connection.span, port_ty, conn_sig.ty(signals))?;
                        }
                    }
                }
                IrModuleChild::ModuleExternalInstance(instance) => {
                    let IrModuleExternalInstance {
                        name: _,
                        module_name,
                        generic_args,
                        port_connections,
                    } = instance;

                    if !external_modules.contains(module_name) {
                        let msg = format!("IR ModuleExternalInstance `{module_name}` not found in external modules");
                        return Err(diags.report_error_internal(child.span, msg));
                    }

                    // just trust generic args, the IR does not store type information for them
                    let _ = generic_args;

                    // check port connections
                    // just trust directions, the IR does not separately store them
                    for (port_ty, connection) in port_connections.values() {
                        let conn_sig = match connection.inner {
                            IrPortConnection::Input(signal) => Some(signal),
                            IrPortConnection::Output(signal) => signal,
                        };
                        if let Some(conn_sig) = conn_sig {
                            conn_sig.validate(signals);
                            check_type_match(diags, connection.span, port_ty, conn_sig.ty(signals))?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl IrBlock {
    fn validate(
        &self,
        diags: &Diagnostics,
        module: &IrModuleInfo,
        can_drive: Option<&IndexSet<IrSignal>>,
        variables: &IrVariables,
    ) -> DiagResult {
        let large = &module.large;
        let signals = &module.signals;

        for stmt in &self.statements {
            match &stmt.inner {
                IrStatement::Assign(target, expr) => {
                    let &IrAssignmentTarget { base, ref steps } = target;

                    // check target allowed
                    match base {
                        IrSignalOrVariable::Signal(signal) => {
                            if let Some(can_drive) = can_drive {
                                if !can_drive.contains(&signal) {
                                    return Err(
                                        diags.report_error_internal(stmt.span, "assigning to non-driven signal")
                                    );
                                }
                            }
                        }
                        IrSignalOrVariable::Variable(var) => {
                            let _ = variables[var];
                        }
                    }

                    // check target
                    let base_ty = base.ty(signals, variables);
                    let target_ty = steps.validate(diags, module, variables, stmt.span, base_ty.clone())?;

                    // check expr
                    expr.validate(diags, module, variables, stmt.span)?;
                    let expr_ty = expr.ty(large, signals, variables);

                    // check type match
                    check_type_match(diags, stmt.span, &target_ty, &expr_ty)?;
                }
                IrStatement::Block(block) => {
                    block.validate(diags, module, can_drive, variables)?;
                }
                IrStatement::If(if_stmt) => {
                    let IrIfStatement {
                        condition,
                        then_block,
                        else_block,
                    } = if_stmt;

                    condition.validate(diags, module, variables, stmt.span)?;
                    check_type_is_bool(diags, stmt.span, &condition.ty(large, signals, variables))?;

                    then_block.validate(diags, module, can_drive, variables)?;

                    if let Some(else_block) = else_block {
                        else_block.validate(diags, module, can_drive, variables)?;
                    }
                }
                IrStatement::For(for_stmt) => {
                    let &IrForStatement {
                        index,
                        ref range,
                        ref block,
                    } = for_stmt;
                    let index_ty = IrExpression::Variable(index).ty(large, signals, variables);
                    let index_range = check_type_is_int(diags, stmt.span, &index_ty)?;

                    if !index_range.contains_range(range.as_ref()) {
                        let msg = format!(
                            "IR IrForStatement variable must contain loop range: variable {index_range:?} but loop {range:?}"
                        );
                        return Err(diags.report_error_internal(stmt.span, msg));
                    }

                    block.validate(diags, module, can_drive, variables)?;
                }
                IrStatement::Print(pieces) => validate_string(diags, module, variables, stmt.span, pieces)?,
                IrStatement::AssertFailed => {}
            }
        }
        Ok(())
    }
}

fn validate_string(
    diags: &Diagnostics,
    module: &IrModuleInfo,
    variables: &IrVariables,
    span: Span,
    s: &IrString,
) -> DiagResult {
    let large = &module.large;
    let signals = &module.signals;

    for p in s {
        match p {
            StringPiece::Literal(_) => {}
            StringPiece::Substitute(p) => match p {
                IrStringSubstitution::Integer(p, _) => {
                    p.validate(diags, module, variables, span)?;
                    check_type_is_int(diags, span, &p.ty(large, signals, variables))?;
                }
            },
        }
    }
    Ok(())
}

impl IrTargetSteps {
    fn validate(
        &self,
        diags: &Diagnostics,
        module: &IrModuleInfo,
        variables: &IrVariables,
        span: Span,
        base_ty: IrType,
    ) -> DiagResult<IrType> {
        let IrTargetSteps {
            steps_scalar,
            step_slice,
        } = self;

        let large = &module.large;
        let signals = &module.signals;

        let mut curr_ty = base_ty;

        for step in steps_scalar {
            curr_ty = match step {
                IrTargetStepScalar::ArrayIndex(index) => {
                    let (ty_inner, ty_len) = check_type_is_array(diags, span, curr_ty)?;
                    let index_ty = index.ty(large, signals, variables);
                    let index_ty = check_type_is_int(diags, span, &index_ty)?;

                    let valid_range = ClosedRange {
                        start: BigInt::ZERO,
                        end: BigInt::from(ty_len),
                    };
                    if !valid_range.contains_range(index_ty.as_ref()) {
                        return Err(diags.report_error_internal(span, "IR target ArrayIndex out of bounds"));
                    }

                    ty_inner
                }
                &IrTargetStepScalar::TupleIndex(index) => {
                    let ty_fields = check_type_is_tuple(diags, span, curr_ty)?;
                    if index >= ty_fields.len() {
                        return Err(diags.report_error_internal(span, "IR target TupleIndex out of bounds"));
                    }
                    ty_fields.get_owned(index)
                }
                &IrTargetStepScalar::StructField(field) => {
                    let ty_info = check_type_is_struct(diags, span, curr_ty)?;
                    if field >= ty_info.fields.len() {
                        return Err(diags.report_error_internal(span, "IR target StructField out of bounds"));
                    }
                    ty_info.fields.get_index_owned(field).unwrap().1
                }
            };
        }

        if let Some(step_slice) = step_slice {
            let IrTargetStepSlice { start, len } = step_slice;

            let (ty_inner, ty_len) = check_type_is_array(diags, span, curr_ty)?;
            let start_ty = start.ty(large, signals, variables);
            let start_ty = check_type_is_int(diags, span, &start_ty)?;

            let valid_range = ClosedRange {
                start: BigInt::ZERO,
                end: ty_len - len + 1,
            };
            if !valid_range.contains_range(start_ty.as_ref()) {
                return Err(diags.report_error_internal(span, "IR target StepSlice out of bounds"));
            }

            curr_ty = IrType::Array(Box::new(ty_inner), len.clone());
        }

        Ok(curr_ty)
    }
}

impl IrExpression {
    fn validate(&self, diags: &Diagnostics, module: &IrModuleInfo, variables: &IrVariables, span: Span) -> DiagResult {
        let large = &module.large;
        let signals = &module.signals;

        // validate operands
        let mut any_err = Ok(());
        self.for_each_operand(large, &mut |op| {
            any_err = any_err.and(op.validate(diags, module, variables, span))
        });
        any_err?;

        // validate self
        match self {
            // always valid
            IrExpression::Bool(_) | IrExpression::Int(_) => {}

            // basic existence checks
            &IrExpression::Signal(sig) => sig.validate(signals),
            &IrExpression::Variable(var) => {
                let _ = variables[var];
            }

            // actual type checks
            &IrExpression::Large(expr) => match &large[expr] {
                IrExpressionLarge::Undefined(_ty) => {
                    // always valid
                }
                IrExpressionLarge::BoolNot(inner) => {
                    check_type_is_bool(diags, span, &inner.ty(large, signals, variables))?;
                }
                IrExpressionLarge::BoolBinary(_op, left, right) => {
                    check_type_is_bool(diags, span, &left.ty(large, signals, variables))?;
                    check_type_is_bool(diags, span, &right.ty(large, signals, variables))?;
                }
                IrExpressionLarge::IntArithmetic(_op, _range, left, right) => {
                    check_type_is_int(diags, span, &left.ty(large, signals, variables))?;
                    check_type_is_int(diags, span, &right.ty(large, signals, variables))?;
                }
                IrExpressionLarge::IntCompare(_op, left, right) => {
                    check_type_is_int(diags, span, &left.ty(large, signals, variables))?;
                    check_type_is_int(diags, span, &right.ty(large, signals, variables))?;
                }
                IrExpressionLarge::TupleLiteral(_values) => {
                    // always valid
                }
                IrExpressionLarge::ArrayLiteral(inner_ty, len, values) => {
                    let mut actual_len = BigUint::ZERO;
                    for value in values {
                        match value {
                            IrArrayLiteralElement::Single(value) => {
                                let value_ty = value.ty(large, signals, variables);
                                check_type_match(diags, span, inner_ty, &value_ty)?;
                                actual_len += 1u32;
                            }
                            IrArrayLiteralElement::Spread(value) => {
                                let value_ty = value.ty(large, signals, variables);
                                let (value_inner_ty, value_len) = check_type_is_array(diags, span, value_ty)?;
                                check_type_match(diags, span, inner_ty, &value_inner_ty)?;
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
                        let value_ty = field_value.ty(large, signals, variables);
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
                            let value_ty = payload_expr.ty(large, signals, variables);
                            check_type_match(diags, span, payload_ty, &value_ty)?;
                        }
                        _ => return Err(diags.report_error_internal(span, "IR EnumLiteral payload mismatch")),
                    }
                }
                IrExpressionLarge::Steps { base, steps } => {
                    let base_ty = base.ty(large, signals, variables);
                    steps.validate(diags, module, variables, span, base_ty)?;
                }
                IrExpressionLarge::EnumTag { base } => {
                    let base_ty = base.ty(large, signals, variables);
                    let _ = check_type_is_enum(diags, span, base_ty)?;
                }
                &IrExpressionLarge::EnumPayload { ref base, variant } => {
                    let base_ty = base.ty(large, signals, variables);
                    let base_ty = check_type_is_enum(diags, span, base_ty)?;

                    if variant >= base_ty.variants.len() {
                        return Err(diags.report_error_internal(span, "IR EnumPayload invalid variant"));
                    }

                    if base_ty.variants[variant].is_none() {
                        return Err(diags.report_error_internal(span, "IR EnumPayload variant has no payload"));
                    }
                }
                IrExpressionLarge::ToBits(ty, expr) => {
                    if ty != &expr.ty(large, signals, variables) {
                        return Err(diags.report_error_internal(span, "IR ToBits type mismatch"));
                    }
                }
                IrExpressionLarge::FromBits(ty, expr) => {
                    let expr_ty = expr.ty(large, signals, variables);
                    if expr_ty != IrType::Array(Box::new(IrType::Bool), ty.size_bits()) {
                        return Err(diags.report_error_internal(span, "IR FromBits type mismatch"));
                    }
                }
                IrExpressionLarge::ExpandIntRange(outer, inner) => {
                    let inner_ty = inner.ty(large, signals, variables);
                    let inner_range = check_type_is_int(diags, span, &inner_ty)?;

                    if !outer.contains_range(inner_range.as_ref()) {
                        return Err(diags.report_error_internal(span, "IR ExpandIntRange outer does not contain inner"));
                    }
                }
                IrExpressionLarge::ConstrainIntRange(outer, inner) => {
                    let inner_ty = inner.ty(large, signals, variables);
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

impl IrSignal {
    fn validate(self, signals: &IrSignals) {
        // assert that signal/var exists in this context
        match self {
            IrSignal::Port(port) => {
                let _ = signals.ports[port];
            }
            IrSignal::Wire(wire) => {
                let _ = signals.wires[wire];
            }
        }
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

fn check_type_is_array(diags: &Diagnostics, span: Span, actual: IrType) -> DiagResult<(IrType, BigUint)> {
    match actual {
        IrType::Array(inner, len) => Ok((*inner, len)),
        _ => {
            let msg = format!("IR type mismatch: expected array, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_tuple(diags: &Diagnostics, span: Span, actual: IrType) -> DiagResult<Vec<IrType>> {
    match actual {
        IrType::Tuple(elements) => Ok(elements),
        _ => {
            let msg = format!("IR type mismatch: expected tuple, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_struct(diags: &Diagnostics, span: Span, actual: IrType) -> DiagResult<IrStructType> {
    match actual {
        IrType::Struct(ty) => Ok(ty),
        _ => {
            let msg = format!("IR type mismatch: expected struct, got {actual:?}");
            Err(diags.report_error_internal(span, msg))
        }
    }
}

fn check_type_is_enum(diags: &Diagnostics, span: Span, actual: IrType) -> DiagResult<IrEnumType> {
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
