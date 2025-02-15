use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::ir::{
    IrAssignmentTarget, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrDatabase, IrExpression, IrIfStatement,
    IrModuleChild, IrModuleInfo, IrModuleInstance, IrPort, IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo,
    IrStatement, IrType, IrVariable, IrVariableInfo, IrVariables, IrWire, IrWireInfo, IrWireOrPort,
};
use crate::front::types::ClosedIncRange;
use crate::syntax::ast::{ArrayLiteralElement, Identifier, MaybeIdentifier};
use crate::syntax::pos::Span;
use crate::util::arena::{Idx, IndexType};
use crate::util::Indent;
use crate::{swrite, swriteln};
use itertools::enumerate;
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;
use std::fmt::{Display, Formatter};
use unwrap_match::unwrap_match;

pub fn simulator_codegen(diags: &Diagnostics, ir: &IrDatabase) -> Result<String, ErrorGuaranteed> {
    // TODO split into separate files:
    //   maybe one shared with all structs,
    //   but functions should be split for the compilation speedup
    let mut source = String::new();
    let f = &mut source;
    swriteln!(f, "#include <cstdint>");
    swriteln!(f, "#include <stdlib.h>");
    swriteln!(f, "#include <array>");
    swriteln!(f);

    for (module, module_info) in &ir.modules {
        let IrModuleInfo {
            ports,
            registers,
            wires,
            children,
            debug_info_id: _,
            debug_info_generic_args: _,
        } = module_info;
        let module_index = module.inner().index();

        // TODO include module debug name and add generic args as a comment
        let module_struct = format!("ModuleSignals_{module_index}");

        // TODO convert functions into member functions of the struct?
        //   more generally, think about what the API and memory layout should look like
        swriteln!(f, "struct {module_struct} {{",);

        // factory method that initializes everything
        let mut create_params = String::new();
        let mut create_body = String::new();
        let fp = &mut create_params;
        let fb = &mut create_body;
        swriteln!(
            fb,
            "{I}{I}{module_struct} *result = ({module_struct}*) calloc(1, sizeof({module_struct}));"
        );

        let mut all_body = String::new();
        let fa = &mut all_body;

        // signals
        for (i_port, (port, port_info)) in enumerate(ports) {
            let ty_str = type_to_cpp(diags, port_info.debug_info_id.span, &port_info.ty)?;
            let name = port_str(port, port_info);

            swriteln!(f, "{I}{ty_str} *{name};");
            let end = if i_port == ports.len() - 1 { "" } else { "," };
            swriteln!(fp, "{I}{I}{ty_str} *{name}{end}");
            swriteln!(fb, "{I}{I}result->{name} = {name};");
        }
        for (reg, reg_info) in registers {
            let ty_str = type_to_cpp(diags, reg_info.debug_info_id.span(), &reg_info.ty)?;
            let name = reg_str(reg, reg_info);
            swriteln!(f, "{I}{ty_str} {name};");
        }
        for (wire, wire_info) in wires {
            let ty_str = type_to_cpp(diags, wire_info.debug_info_id.span(), &wire_info.ty)?;
            let name = wire_str(wire, wire_info);
            swriteln!(f, "{I}{ty_str} {name};");
        }
        // children
        for (child_index, child) in enumerate(children) {
            if let IrModuleChild::ModuleInstance(child) = child {
                let &IrModuleInstance {
                    name: _,
                    module: child_module,
                    ref port_connections,
                } = child;

                // create field to store child data
                let child_module_index = child_module.inner().index();
                let child_struct = format!("ModuleSignals_{child_module_index}");
                swriteln!(f, "{I}{child_struct} *child_{child_index};");

                // create child instance
                swriteln!(fb, "{I}{I}result->child_{child_index} = {child_struct}::create(");
                for (i_connection, (_, connection)) in enumerate(port_connections) {
                    let connection_str = match &connection.inner {
                        IrPortConnection::Input(expr) => {
                            // create an additional process
                            match expr.inner {
                                IrExpression::Port(port) => port_str(port, &module_info.ports[port]),
                                IrExpression::Wire(wire) => {
                                    let wire = wire_str(wire, &module_info.wires[wire]);
                                    format!("&result->{wire}")
                                }
                                IrExpression::Register(reg) => {
                                    let reg = reg_str(reg, &module_info.registers[reg]);
                                    format!("&result->{reg}")
                                }
                                // TODO remove this as an option for ports, the frontend can handle this
                                _ => {
                                    return Err(
                                        diags.report_todo(connection.span, "codegen instance input general expression")
                                    )
                                }
                            }
                        }
                        &IrPortConnection::Output(signal) => match signal {
                            None => return Err(diags.report_todo(connection.span, "codegen instance output dummy")),
                            Some(IrWireOrPort::Port(port)) => port_str(port, &module_info.ports[port]),
                            Some(IrWireOrPort::Wire(wire)) => {
                                let wire = wire_str(wire, &module_info.wires[wire]);
                                format!("&result->{wire}")
                            }
                        },
                    };

                    let end = if i_connection == port_connections.len() - 1 {
                        ""
                    } else {
                        ","
                    };
                    swriteln!(fb, "{I}{I}{I}{connection_str}{end}");
                }

                swriteln!(fb, "{I}{I});");
            }
        }

        swriteln!(fb, "{I}{I}return result;");

        swriteln!(f, "{I}static {module_struct} *create(");
        f.push_str(&create_params);
        swriteln!(f, "{I}) {{");
        f.push_str(&create_body);
        swriteln!(f, "{I}}}");

        // make struct non-movable and non-copyable
        swriteln!(f, "{I}{module_struct} &operator=({module_struct}&&) = delete;");

        swriteln!(f, "}};");
        swriteln!(f);

        for (child_index, child) in enumerate(children) {
            match child {
                IrModuleChild::ClockedProcess(proc) => {
                    let IrClockedProcess {
                        domain,
                        locals,
                        on_clock,
                        on_reset,
                    } = proc;

                    // reset function
                    let func_reset_name = format!("module_{module_index}_child_{child_index}_clocked_reset");
                    swriteln!(fa, "{I}{func_reset_name}(prev, next);");

                    let (prev, next) = (Stage::Prev, Stage::Next);
                    swriteln!(
                        f,
                        "void {func_reset_name}({module_struct} &{prev}, {module_struct} &{next}) {{"
                    );
                    let mut ctx = CodegenBlockContext {
                        diags,
                        module_info,
                        locals,
                        f,
                        indent: Indent::new(1),
                        next_temporary_index: 0,
                    };
                    let reset_eval = ctx.eval(domain.span, &domain.inner.reset, Stage::Next)?;
                    swriteln!(ctx.f, "{I}if (!({reset_eval})) {{");
                    swriteln!(ctx.f, "{I}{I}return;");
                    swriteln!(ctx.f, "{I}}}");
                    swriteln!(ctx.f);

                    //   reset doesn't need locals
                    ctx.generate_block(on_reset, Stage::Next)?;

                    swriteln!(f, "}}");
                    swriteln!(f);

                    // clock function
                    let func_clock_name = format!("module_{module_index}_child_{child_index}_clocked_clock");
                    swriteln!(fa, "{I}{func_clock_name}(prev, next);");

                    swriteln!(
                        f,
                        "void {func_clock_name}({module_struct} &{prev}, {module_struct} &{next}) {{"
                    );
                    let mut ctx = CodegenBlockContext {
                        diags,
                        module_info,
                        locals,
                        f,
                        indent: Indent::new(1),
                        next_temporary_index: 0,
                    };
                    let clock_prev_eval = ctx.eval(domain.span, &domain.inner.clock, Stage::Prev)?;
                    let clock_next_eval = ctx.eval(domain.span, &domain.inner.clock, Stage::Next)?;
                    swriteln!(ctx.f, "{I}if (!(!({clock_prev_eval}) && ({clock_next_eval}))) {{");
                    swriteln!(ctx.f, "{I}{I}return;");
                    swriteln!(ctx.f, "{I}}}");
                    swriteln!(ctx.f);

                    //   declare locals
                    for (var, var_info) in locals {
                        let ty_str = type_to_cpp(diags, var_info.debug_info_id.span(), &var_info.ty)?;
                        let name = var_str(var, var_info);
                        swriteln!(ctx.f, "{I}{ty_str} {name};");
                    }
                    ctx.generate_block(on_clock, Stage::Prev)?;

                    swriteln!(f, "}}");
                    swriteln!(f);
                }
                IrModuleChild::CombinatorialProcess(proc) => {
                    let IrCombinatorialProcess { locals, block } = proc;

                    let func_name = format!("module_{module_index}_child_{child_index}_combinatorial");
                    swriteln!(fa, "{I}{func_name}(next);");

                    let next = Stage::Next;
                    swriteln!(f, "void {func_name}({module_struct} &{next}) {{");
                    let mut ctx = CodegenBlockContext {
                        diags,
                        module_info,
                        locals,
                        f,
                        indent: Indent::new(1),
                        next_temporary_index: 0,
                    };
                    for (var, var_info) in locals {
                        let ty_str = type_to_cpp(diags, var_info.debug_info_id.span(), &var_info.ty)?;
                        let name = var_str(var, var_info);
                        swriteln!(ctx.f, "{I}{ty_str} {name};");
                    }
                    ctx.generate_block(block, Stage::Next)?;

                    swriteln!(f, "}}");
                    swriteln!(f);
                }
                IrModuleChild::ModuleInstance(instance) => {
                    // delegate to child
                    let child_module_index = instance.module.inner().index();
                    let field_name = format!("child_{child_index}");
                    swriteln!(
                        fa,
                        "{I}module_{child_module_index}_all(*prev.{field_name}, *next.{field_name});"
                    );
                }
            }
        }

        swriteln!(
            f,
            "void module_{module_index}_all({module_struct} &prev, {module_struct} &next) {{"
        );
        f.push_str(&all_body);
        swriteln!(f, "}}");
        swriteln!(f);
    }

    Ok(source)
}

struct CodegenBlockContext<'a> {
    diags: &'a Diagnostics,
    module_info: &'a IrModuleInfo,
    locals: &'a IrVariables,
    f: &'a mut String,
    indent: Indent,
    next_temporary_index: usize,
}

#[derive(Debug)]
pub enum Evaluated {
    Temporary(usize),
    // TODO avoid string allocation for some common cases
    Simple(String),
}

impl Display for Evaluated {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Evaluated::Temporary(index) => write!(f, "t_{}", index),
            Evaluated::Simple(s) => write!(f, "{}", s),
        }
    }
}

impl CodegenBlockContext<'_> {
    fn eval(&mut self, span: Span, expr: &IrExpression, stage_read: Stage) -> Result<Evaluated, ErrorGuaranteed> {
        let todo = |kind: &str| self.diags.report_todo(span, format!("codegen IrExpression::{kind}"));

        let result = match expr {
            &IrExpression::Bool(b) => Evaluated::Simple(format!("{b}")),
            // TODO support arbitrary sized ints
            IrExpression::Int(v) => Evaluated::Simple(format!("INT64_C({v})")),
            &IrExpression::Port(port) => {
                let name = port_str(port, &self.module_info.ports[port]);
                Evaluated::Simple(format!("*{stage_read}.{name}"))
            }
            &IrExpression::Wire(wire) => {
                let name = wire_str(wire, &self.module_info.wires[wire]);
                Evaluated::Simple(format!("{stage_read}.{name}"))
            }
            &IrExpression::Register(reg) => {
                let name = reg_str(reg, &self.module_info.registers[reg]);
                Evaluated::Simple(format!("{stage_read}.{name}"))
            }
            &IrExpression::Variable(var) => Evaluated::Simple(var_str(var, &self.locals[var])),
            IrExpression::BoolNot(inner) => {
                let inner_eval = self.eval(span, inner, stage_read)?;
                Evaluated::Simple(format!("!({inner_eval})"))
            }
            IrExpression::BoolBinary(_, _, _) => return Err(todo("BoolBinary")),
            IrExpression::IntArithmetic(_, _, _, _) => return Err(todo("IntArithmetic")),
            IrExpression::IntCompare(_, _, _) => return Err(todo("IntCompare")),

            IrExpression::TupleLiteral(_) => return Err(todo("TupleLiteral")),
            IrExpression::ArrayLiteral(inner_ty, elements) => {
                let indent = self.indent;

                let inner_ty_str = type_to_cpp(self.diags, span, inner_ty)?;
                let tmp_result = self.new_temporary();

                swriteln!(
                    self.f,
                    "{indent}std::array<{inner_ty_str}, {len}> {tmp_result};",
                    len = elements.len()
                );

                let mut offset = BigUint::zero();
                for element in elements {
                    let ArrayLiteralElement { spread, value } = element;
                    let value_eval = self.eval(span, value, stage_read)?;
                    if spread.is_some() {
                        let element_len = unwrap_match!(value.ty(self.module_info, &self.locals), IrType::Array(_, element_len) => element_len);
                        swriteln!(self.f, "{indent}std::copy_n({value_eval}.begin(), {value_eval}.size(), {tmp_result}.begin() + {offset});");
                        offset += element_len;
                    } else {
                        swriteln!(self.f, "{indent}{tmp_result}[{offset}] = {value_eval};");
                        offset += 1u32;
                    }
                }

                tmp_result
            }

            IrExpression::ArrayIndex { .. } => return Err(todo("ArrayIndex")),
            IrExpression::ArraySlice { .. } => return Err(todo("ArraySlice")),

            IrExpression::IntToBits(_, _) => return Err(todo("IntToBits")),
            IrExpression::IntFromBits(_, _) => return Err(todo("IntFromBits")),
            IrExpression::ExpandIntRange(range, inner) => {
                // check that result is still representable
                let _ = type_to_cpp(self.diags, span, &IrType::Int(range.clone()))?;
                self.eval(span, inner, stage_read)?
            }
        };
        Ok(result)
    }

    fn generate_nested_block(&mut self, block: &IrBlock, stage_read: Stage) -> Result<(), ErrorGuaranteed> {
        swriteln!(self.f, "{{");
        let indent = self.indent;
        self.indent = self.indent.nest();
        self.generate_block(block, stage_read)?;
        self.indent = indent;
        swrite!(self.f, "{indent}}}");
        Ok(())
    }

    fn generate_block(&mut self, block: &IrBlock, stage_read: Stage) -> Result<(), ErrorGuaranteed> {
        let IrBlock { statements } = block;
        let indent = self.indent;

        for stmt in statements {
            match &stmt.inner {
                IrStatement::Assign(target, expr) => {
                    let next = Stage::Next;
                    let target_str = match *target {
                        IrAssignmentTarget::Port(port) => {
                            format!("*{next}.{}", port_str(port, &self.module_info.ports[port]))
                        }
                        IrAssignmentTarget::Register(reg) => {
                            format!("{next}.{}", reg_str(reg, &self.module_info.registers[reg]))
                        }
                        IrAssignmentTarget::Wire(wire) => {
                            format!("{next}.{}", wire_str(wire, &self.module_info.wires[wire]))
                        }
                        IrAssignmentTarget::Variable(var) => var_str(var, &self.locals[var]),
                    };
                    let value_str = self.eval(stmt.span, expr, stage_read)?;
                    swriteln!(self.f, "{indent}{target_str} = {value_str};");
                }
                IrStatement::Block(inner) => {
                    swriteln!(self.f, "{indent}");
                    self.generate_nested_block(inner, stage_read)?;
                    swriteln!(self.f);
                }
                IrStatement::If(if_stmt) => {
                    let IrIfStatement {
                        initial_if,
                        else_ifs,
                        final_else,
                    } = if_stmt;

                    let initial_cond = self.eval(stmt.span, &initial_if.cond, stage_read)?;
                    swrite!(self.f, "{indent}if ({initial_cond})");
                    self.generate_nested_block(&initial_if.block, stage_read)?;

                    // TODO evaluating the else conditions is probably broken, they might not be a single expression
                    for else_if in else_ifs {
                        let else_if_cond = self.eval(stmt.span, &else_if.cond, stage_read)?;
                        swrite!(self.f, " else if ({else_if_cond})");
                        self.generate_nested_block(&else_if.block, stage_read)?;
                    }

                    if let Some(final_else) = final_else {
                        swrite!(self.f, " else");
                        self.generate_nested_block(final_else, stage_read)?;
                    }

                    swriteln!(self.f);
                }
            }
        }

        Ok(())
    }

    fn new_temporary(&mut self) -> Evaluated {
        let index = self.next_temporary_index;
        self.next_temporary_index += 1;
        Evaluated::Temporary(index)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Stage {
    Prev,
    Next,
}

impl Display for Stage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Stage::Prev => write!(f, "prev"),
            Stage::Next => write!(f, "next"),
        }
    }
}

fn type_to_cpp(diags: &Diagnostics, span: Span, ty: &IrType) -> Result<String, ErrorGuaranteed> {
    let result = match ty {
        IrType::Bool => "bool".to_owned(),
        IrType::Int(range) => {
            let ClosedIncRange { start_inc, end_inc } = range;

            if &BigInt::from(-i64::MAX) <= start_inc && end_inc <= &BigInt::from(i64::MAX) {
                // TODO use smaller types when appropriate
                "int64_t".to_string()
            } else {
                return Err(diags.report_todo(span, format!("codegen wide integer type: {range}")));
            }
        }
        IrType::Tuple(_) => return Err(diags.report_todo(span, "codegen type Tuple")),
        IrType::Array(inner, len) => {
            let inner_str = type_to_cpp(diags, span, inner)?;
            format!("std::array<{inner_str}, {len}>")
        }
    };
    Ok(result)
}

fn port_str(port: IrPort, port_info: &IrPortInfo) -> String {
    name_str(
        "port",
        port.inner(),
        MaybeIdentifier::Identifier(&port_info.debug_info_id),
    )
}

fn wire_str(wire: IrWire, wire_info: &IrWireInfo) -> String {
    name_str("wire", wire.inner(), wire_info.debug_info_id.as_ref())
}

fn reg_str(reg: IrRegister, reg_info: &IrRegisterInfo) -> String {
    name_str("reg", reg.inner(), reg_info.debug_info_id.as_ref())
}

fn var_str(var: IrVariable, var_info: &IrVariableInfo) -> String {
    name_str("var", var.inner(), var_info.debug_info_id.as_ref())
}

fn name_str(prefix: &str, index: Idx, id: MaybeIdentifier<&Identifier>) -> String {
    let index = index.index();
    match id {
        MaybeIdentifier::Identifier(id) => {
            let str_filtered: String = id
                .string
                .chars()
                .filter(|&c| c.is_ascii_alphanumeric() || c == '_')
                .collect();
            format!("{prefix}_{index}_{str_filtered}")
        }
        MaybeIdentifier::Dummy(_) => {
            format!("{prefix}_{index}")
        }
    }
}

const I: &str = Indent::I;
