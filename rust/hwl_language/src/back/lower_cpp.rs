use crate::front::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::front::types::ClosedIncRange;
use crate::mid::ir::{
    ir_modules_topological_sort, IrArrayLiteralElement, IrAssignmentTarget, IrAssignmentTargetBase, IrBlock,
    IrBoolBinaryOp, IrClockedProcess, IrCombinatorialProcess, IrExpression, IrExpressionLarge, IrIfStatement,
    IrIntArithmeticOp, IrIntCompareOp, IrModule, IrModuleChild, IrModuleInfo, IrModuleInstance, IrModules, IrPort,
    IrPortConnection, IrPortInfo, IrRegister, IrRegisterInfo, IrStatement, IrTargetStep, IrType, IrVariable,
    IrVariableInfo, IrVariables, IrWire, IrWireInfo, IrWireOrPort,
};
use crate::syntax::ast::{Identifier, MaybeIdentifier};
use crate::syntax::pos::Span;
use crate::util::arena::{Idx, IndexType};
use crate::util::big_int::{BigInt, BigUint};
use crate::util::int::IntRepresentation;
use crate::util::iter::IterExt;
use crate::util::Indent;
use crate::{swrite, swriteln};
use itertools::enumerate;
use std::fmt::{Display, Formatter};
use unwrap_match::unwrap_match;

// TODO add "undefined" tracking bits
//   only one for each signal, maybe also a separate one for each array element?
//   aggressively propagate it, eg. if (undef) => write undefined for all writes in both branches
//   skip operations on undefined values, we don't want to cause undefined C++ behavior!
pub fn lower_to_cpp(diags: &Diagnostics, modules: &IrModules, top_module: IrModule) -> Result<String, ErrorGuaranteed> {
    // TODO split into separate files:
    //   maybe one shared with all structs,
    //   but functions should be split for the compilation speedup
    let mut f = String::new();
    swriteln!(f, "#include <cstdint>");
    swriteln!(f, "#include <stdlib.h>");
    swriteln!(f, "#include <array>");
    swriteln!(f, "#include <utility>");
    swriteln!(f, "#include <algorithm>");
    swriteln!(f, "#include <iostream>");
    swriteln!(f);

    for (i, module) in enumerate(ir_modules_topological_sort(modules, top_module)) {
        if i != 0 {
            swriteln!(f);
        }

        swrite!(f, "{}", codegen_module(diags, modules, module)?);
    }

    Ok(f)
}

fn codegen_module(diags: &Diagnostics, modules: &IrModules, module: IrModule) -> Result<String, ErrorGuaranteed> {
    let module_info = &modules[module];
    let IrModuleInfo {
        ports,
        registers,
        wires,
        large: _,
        children,
        debug_info_id: _,
        debug_info_generic_args: _,
    } = module_info;
    let module_index = module.inner().index();

    // TODO include module debug name and add generic args as a comment or even in the name
    let struct_signals = format!("ModuleSignals_{module_index}");
    let struct_ports_ptr = format!("ModulePortsPtr_{module_index}");
    let struct_ports_val = format!("ModulePortsVal_{module_index}");

    // ports
    let mut f_structs_ptr = String::new();
    let mut f_structs_val = String::new();
    let mut f_structs_to_ptr = String::new();

    swriteln!(f_structs_ptr, "struct {struct_ports_ptr} {{");
    swriteln!(f_structs_val, "struct {struct_ports_val} {{");
    swriteln!(f_structs_to_ptr, "{I}{struct_ports_ptr} as_ptrs() {{");
    swriteln!(f_structs_to_ptr, "{I}{I}return {struct_ports_ptr} {{");

    for (port_i, (port, port_info)) in enumerate(ports) {
        let ty_str = type_to_cpp(diags, port_info.debug_span, &port_info.ty)?;
        let name = port_str(port, port_info);
        swriteln!(f_structs_ptr, "{I}{ty_str} *{name};");
        swriteln!(f_structs_val, "{I}{ty_str} {name};");
        swriteln!(
            f_structs_to_ptr,
            "{I}{I}{I}&this->{name}{}",
            end_comma(port_i, ports.len())
        );
    }

    swrite!(f_structs_ptr, "}};\n\n");
    swrite!(f_structs_to_ptr, "{I}{I}}};\n{I}}};\n");
    swrite!(f_structs_val, "\n{f_structs_to_ptr}\n}};\n\n");

    let mut f_structs = f_structs_ptr;
    f_structs.push_str(&f_structs_val);
    drop(f_structs_val);

    // signals
    swriteln!(f_structs, "struct {struct_signals} {{");
    for (reg, reg_info) in registers {
        let ty_str = type_to_cpp(diags, reg_info.debug_info_id.span(), &reg_info.ty)?;
        let name = reg_str(reg, reg_info);
        swriteln!(f_structs, "{I}{ty_str} {name};");
    }
    for (wire, wire_info) in wires {
        let ty_str = type_to_cpp(diags, wire_info.debug_info_id.span(), &wire_info.ty)?;
        let name = wire_str(wire, wire_info);
        swriteln!(f_structs, "{I}{ty_str} {name};");
    }

    // children
    let (prev, next) = (Stage::Prev, Stage::Next);
    let step_params = format!("{struct_signals} &{prev}_signals, {struct_ports_ptr} {prev}_ports, {struct_signals} &{next}_signals, {struct_ports_ptr} {next}_ports");
    let step_args: String = format!("{prev}_signals, {prev}_ports, {next}_signals, {next}_ports");

    let mut f_step = String::new();
    let mut f_step_ports = String::new();
    let mut f_step_all = String::new();
    swriteln!(f_step_all, "void module_{module_index}_all({step_params}) {{");

    for (child_index, child) in enumerate(children) {
        let indent = Indent::new(1);

        match child {
            IrModuleChild::ClockedProcess(proc) => {
                let IrClockedProcess {
                    locals,
                    clock_signal,
                    clock_block,
                    async_reset_signal_and_block,
                } = proc;

                // reset function
                if let Some((reset_signal, reset_block)) = async_reset_signal_and_block {
                    let func_step_reset = format!("module_{module_index}_child_{child_index}_clocked_async_reset");
                    swriteln!(f_step_all, "{I}{func_step_reset}({step_args});");

                    swriteln!(f_step, "void {func_step_reset}({step_params}) {{");
                    let mut ctx = CodegenBlockContext {
                        diags,
                        module_info,
                        locals,
                        f: &mut f_step,
                        next_temporary_index: 0,
                    };

                    // TODO this might not be correct, think about which prev/reset combination reset needs to look at
                    let reset_eval = ctx.eval(indent, reset_signal.span, &reset_signal.inner, Stage::Next)?;
                    swriteln!(ctx.f, "{I}if (!({reset_eval})) {{");
                    swriteln!(ctx.f, "{I}{I}return;");
                    swriteln!(ctx.f, "{I}}}");
                    swriteln!(ctx.f);

                    //   reset doesn't need locals
                    ctx.generate_block(indent, reset_block, Stage::Next)?;

                    swriteln!(f_step, "}}");
                    swriteln!(f_step);
                }

                // clock function
                let func_step_clock = format!("module_{module_index}_child_{child_index}_clocked_clock");
                swriteln!(f_step_all, "{I}{func_step_clock}({step_args});");

                swriteln!(f_step, "void {func_step_clock}({step_params}) {{");
                let mut ctx = CodegenBlockContext {
                    diags,
                    module_info,
                    locals,
                    f: &mut f_step,
                    next_temporary_index: 0,
                };
                let clock_prev_eval = ctx.eval(indent, clock_signal.span, &clock_signal.inner, Stage::Prev)?;
                let clock_next_eval = ctx.eval(indent, clock_signal.span, &clock_signal.inner, Stage::Next)?;
                // TODO also skip if reset is active
                // TODO maybe switch to a single function for both?
                //   or maybe that just mixes sensitivities for no good reason
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
                ctx.generate_block(indent, clock_block, Stage::Prev)?;

                swriteln!(f_step, "}}");
                swriteln!(f_step);
            }
            IrModuleChild::CombinatorialProcess(proc) => {
                let IrCombinatorialProcess { locals, block } = proc;

                let func_step = format!("module_{module_index}_child_{child_index}_comb");
                swriteln!(f_step_all, "{I}{func_step}({step_args});");

                swriteln!(f_step, "void {func_step}({step_params}) {{");
                let mut ctx = CodegenBlockContext {
                    diags,
                    module_info,
                    locals,
                    f: &mut f_step,
                    next_temporary_index: 0,
                };
                for (var, var_info) in locals {
                    let ty_str = type_to_cpp(diags, var_info.debug_info_id.span(), &var_info.ty)?;
                    let name = var_str(var, var_info);
                    swriteln!(ctx.f, "{I}{ty_str} {name};");
                }
                ctx.generate_block(indent, block, Stage::Next)?;

                swriteln!(f_step, "}}");
                swriteln!(f_step);
            }
            IrModuleChild::ModuleInstance(instance) => {
                let &IrModuleInstance {
                    name: _,
                    module: child_module,
                    ref port_connections,
                } = instance;
                // create field to store child data
                let child_module_info = &modules[child_module];
                let child_module_index = child_module.inner().index();
                let struct_child_signals = format!("ModuleSignals_{child_module_index}");
                let struct_child_ports = format!("ModulePortsPtr_{child_module_index}");
                swriteln!(f_structs, "{I}{struct_child_signals} child_{child_index};");

                // create port struct constructor, used in step function
                let func_child_ports: String = format!("module_{module_index}_child_{child_module_index}_ports");
                swriteln!(
                    f_step_ports,
                    "{struct_child_ports} {func_child_ports}({struct_signals} &signals, {struct_ports_ptr} ports) {{"
                );
                swriteln!(f_step_ports, "{I}return {struct_child_ports} {{");
                for (connection_index, connection) in enumerate(port_connections) {
                    let port_info = child_module_info.ports.get_by_index(connection_index).unwrap().1;

                    let connection_str = match &connection.inner {
                        IrPortConnection::Input(expr) => match expr.inner {
                            IrExpression::Port(port) => format!("ports.{}", port_str(port, &module_info.ports[port])),
                            IrExpression::Wire(wire) => {
                                format!("&signals.{}", wire_str(wire, &module_info.wires[wire]))
                            }
                            IrExpression::Register(reg) => {
                                format!("&signals.{}", reg_str(reg, &module_info.registers[reg]))
                            }
                            // TODO maybe just remove this from IR,
                            //   we can't guarantee that every expression is lowerable as a single expression
                            ref connection => {
                                return Err(diags.report_todo(
                                    expr.span,
                                    format!("simulator input port general expression {connection:?}"),
                                ))
                            }
                        },
                        &IrPortConnection::Output(expr) => match expr {
                            None => {
                                // connect to dummy "signal"
                                let port_ty = type_to_cpp(diags, connection.span, &port_info.ty)?;
                                swriteln!(f_structs, "{I}{port_ty} dummy_{child_index}_{connection_index};");
                                format!("&signals.dummy_{child_index}_{connection_index}")
                            }
                            Some(IrWireOrPort::Port(port)) => {
                                format!("ports.{}", port_str(port, &module_info.ports[port]))
                            }
                            Some(IrWireOrPort::Wire(wire)) => {
                                format!("&signals.{}", wire_str(wire, &module_info.wires[wire]))
                            }
                        },
                    };

                    let end = end_comma(connection_index, port_connections.len());
                    swriteln!(f_step_ports, "{I}{I}{connection_str}{end}")
                }
                swriteln!(f_step_ports, "{I}}};");
                swriteln!(f_step_ports, "}}");
                swriteln!(f_step_ports);

                // delegate step to child
                let field_name = format!("child_{child_index}");
                swriteln!(f_step_all, "{I}module_{child_module_index}_all(");
                swriteln!(f_step_all, "{I}{I}{prev}_signals.{field_name},");
                swriteln!(f_step_all, "{I}{I}{func_child_ports}({prev}_signals, {prev}_ports),");
                swriteln!(f_step_all, "{I}{I}{next}_signals.{field_name},");
                swriteln!(f_step_all, "{I}{I}{func_child_ports}({next}_signals, {next}_ports)");
                swriteln!(f_step_all, "{I});");
            }
        }
    }

    swriteln!(f_structs, "}};\n");
    swriteln!(f_step_all, "}}");

    // combine all
    let mut f_result = String::new();
    swrite!(f_result, "{f_structs}");
    swrite!(f_result, "{f_step}");
    swrite!(f_result, "{f_step_ports}");
    swrite!(f_result, "{f_step_all}");
    Ok(f_result)
}

enum Evaluated {
    Temporary(Temporary),
    // TODO avoid string allocation for some common cases
    Inline(String),
}

#[derive(Copy, Clone)]
struct Temporary(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Stage {
    Prev,
    Next,
}

impl Display for Evaluated {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Evaluated::Temporary(t) => write!(f, "{}", t),
            Evaluated::Inline(s) => write!(f, "{}", s),
        }
    }
}

impl Display for Temporary {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "t_{}", self.0)
    }
}

impl Display for Stage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Stage::Prev => write!(f, "prev"),
            Stage::Next => write!(f, "next"),
        }
    }
}

struct CodegenBlockContext<'a> {
    diags: &'a Diagnostics,
    module_info: &'a IrModuleInfo,
    locals: &'a IrVariables,
    f: &'a mut String,
    next_temporary_index: usize,
}

impl CodegenBlockContext<'_> {
    fn eval_to_temporary(
        &mut self,
        indent: Indent,
        span: Span,
        expr: &IrExpression,
        stage_read: Stage,
    ) -> Result<Temporary, ErrorGuaranteed> {
        match self.eval(indent, span, expr, stage_read)? {
            Evaluated::Temporary(v) => Ok(v),
            Evaluated::Inline(v) => {
                let tmp = self.new_temporary();
                let ty_str = type_to_cpp(self.diags, span, &expr.ty(self.module_info, self.locals))?;
                swriteln!(self.f, "{indent}{ty_str} {tmp} = {v};");
                Ok(tmp)
            }
        }
    }

    fn eval(
        &mut self,
        indent: Indent,
        span: Span,
        expr: &IrExpression,
        stage_read: Stage,
    ) -> Result<Evaluated, ErrorGuaranteed> {
        let todo = |kind: &str| self.diags.report_todo(span, format!("simulator IrExpression::{kind}"));

        let result = match expr {
            &IrExpression::Bool(b) => Evaluated::Inline(format!("{b}")),
            // TODO support arbitrary sized ints
            IrExpression::Int(v) => Evaluated::Inline(format!("INT64_C({v})")),
            &IrExpression::Port(port) => {
                let name = port_str(port, &self.module_info.ports[port]);
                Evaluated::Inline(format!("(*{stage_read}_ports.{name})"))
            }
            &IrExpression::Wire(wire) => {
                let name = wire_str(wire, &self.module_info.wires[wire]);
                Evaluated::Inline(format!("{stage_read}_signals.{name}"))
            }
            &IrExpression::Register(reg) => {
                let name = reg_str(reg, &self.module_info.registers[reg]);
                Evaluated::Inline(format!("{stage_read}_signals.{name}"))
            }
            &IrExpression::Variable(var) => Evaluated::Inline(var_str(var, &self.locals[var])),
            &IrExpression::Large(expr_large) => match &self.module_info.large[expr_large] {
                IrExpressionLarge::BoolNot(inner) => {
                    let inner_eval = self.eval(indent, span, inner, stage_read)?;
                    Evaluated::Inline(format!("!({inner_eval})"))
                }
                IrExpressionLarge::BoolBinary(op, left, right) => {
                    let op_str = match op {
                        IrBoolBinaryOp::And => "&",
                        IrBoolBinaryOp::Or => "|",
                        IrBoolBinaryOp::Xor => "^",
                    };
                    let left_eval = self.eval(indent, span, left, stage_read)?;
                    let right_eval = self.eval(indent, span, right, stage_read)?;
                    Evaluated::Inline(format!("({left_eval} {op_str} {right_eval})"))
                }
                IrExpressionLarge::IntArithmetic(op, _ty, left, right) => {
                    // TODO types are  wrong
                    let op_str = match op {
                        IrIntArithmeticOp::Add => "+",
                        IrIntArithmeticOp::Sub => "-",
                        IrIntArithmeticOp::Mul => "*",
                        IrIntArithmeticOp::Div => "/",
                        IrIntArithmeticOp::Mod => "%",
                        IrIntArithmeticOp::Pow => return Err(todo("codegen power operator")),
                    };
                    let left_eval = self.eval(indent, span, left, stage_read)?;
                    let right_eval = self.eval(indent, span, right, stage_read)?;
                    Evaluated::Inline(format!("({left_eval} {op_str} {right_eval})"))
                }
                IrExpressionLarge::IntCompare(op, left, right) => {
                    // TODO types are  wrong
                    let op_str = match op {
                        IrIntCompareOp::Eq => "==",
                        IrIntCompareOp::Neq => "!=",
                        IrIntCompareOp::Lt => "<",
                        IrIntCompareOp::Lte => "<=",
                        IrIntCompareOp::Gt => ">",
                        IrIntCompareOp::Gte => ">=",
                    };
                    let left_eval = self.eval(indent, span, left, stage_read)?;
                    let right_eval = self.eval(indent, span, right, stage_read)?;
                    Evaluated::Inline(format!("({left_eval} {op_str} {right_eval})"))
                }

                IrExpressionLarge::TupleLiteral(elements) => {
                    let elements_eval = elements
                        .iter()
                        .map(|e| self.eval(indent, span, e, stage_read))
                        .try_collect_vec()?;

                    let result_ty = expr.ty(self.module_info, self.locals);
                    let result_ty_str = type_to_cpp(self.diags, span, &result_ty)?;
                    let tmp_result = self.new_temporary();

                    swrite!(self.f, "{indent}{result_ty_str} {tmp_result} = {{");
                    for (i, element_eval) in enumerate(elements_eval) {
                        swrite!(self.f, "{}{}", element_eval, end_comma(i, elements.len()));
                    }
                    swriteln!(self.f, "}};");

                    Evaluated::Temporary(tmp_result)
                }
                IrExpressionLarge::ArrayLiteral(inner_ty, len, elements) => {
                    let inner_ty_str = type_to_cpp(self.diags, span, inner_ty)?;
                    let tmp_result = self.new_temporary();

                    swriteln!(self.f, "{indent}std::array<{inner_ty_str}, {len}> {tmp_result};",);

                    let mut offset = BigUint::ZERO;
                    for element in elements {
                        match element {
                            IrArrayLiteralElement::Single(value) => {
                                let value_eval = self.eval(indent, span, value, stage_read)?;
                                swriteln!(self.f, "{indent}{tmp_result}[{offset}] = {value_eval};");
                                offset += 1u32;
                            }
                            IrArrayLiteralElement::Spread(value) => {
                                let value_eval = self.eval(indent, span, value, stage_read)?;
                                let element_len = unwrap_match!(value.ty(self.module_info, self.locals), IrType::Array(_, element_len) => element_len);
                                swriteln!(self.f, "{indent}std::copy_n({value_eval}.begin(), {value_eval}.size(), {tmp_result}.begin() + {offset});");
                                offset += element_len;
                            }
                        }
                    }

                    Evaluated::Temporary(tmp_result)
                }
                IrExpressionLarge::TupleIndex { base, index } => {
                    let base_eval = self.eval(indent, span, base, stage_read)?;
                    Evaluated::Inline(format!("({base_eval}.get({index}))"))
                }
                IrExpressionLarge::ArrayIndex { base, index } => {
                    let base_eval = self.eval(indent, span, base, stage_read)?;
                    let index_eval = self.eval(indent, span, index, stage_read)?;
                    Evaluated::Inline(format!("{base_eval}[{index_eval}]"))
                }
                IrExpressionLarge::ArraySlice { base, start, len } => {
                    let base_eval = self.eval(indent, span, base, stage_read)?;
                    let start_eval = self.eval(indent, span, start, stage_read)?;

                    let tmp_result = self.new_temporary();
                    let result_ty_str = type_to_cpp(self.diags, span, &expr.ty(self.module_info, self.locals))?;

                    swriteln!(self.f, "{indent}{result_ty_str} {tmp_result};");
                    swriteln!(
                        self.f,
                        "{indent}std::copy_n({base_eval}.begin() + {start_eval}, {len}, {tmp_result}.begin());"
                    );

                    Evaluated::Temporary(tmp_result)
                }

                IrExpressionLarge::ToBits(ty, value) => {
                    let value = self.eval_to_temporary(indent, span, value, Stage::Next)?;

                    let size_bits = ty.size_bits();
                    let tmp_result = self.new_temporary();
                    swriteln!(self.f, "{indent}std::array<bool, {size_bits}> {tmp_result};");

                    self.impl_to_bits(indent, ty, tmp_result, "0", &value.to_string())?;

                    Evaluated::Temporary(tmp_result)
                }
                IrExpressionLarge::FromBits(ty, value) => {
                    let value = self.eval_to_temporary(indent, span, value, stage_read)?;

                    let result_ty_str = type_to_cpp(self.diags, span, ty)?;
                    let tmp_result = self.new_temporary();
                    swriteln!(self.f, "{indent}{result_ty_str} {tmp_result};");

                    self.impl_from_bits(indent, ty, &tmp_result.to_string(), value, "0")?;

                    Evaluated::Temporary(tmp_result)
                }

                IrExpressionLarge::ExpandIntRange(range, inner) => {
                    // check that result is still representable
                    let _ = type_to_cpp(self.diags, span, &IrType::Int(range.clone()))?;
                    self.eval(indent, span, inner, stage_read)?
                }
                IrExpressionLarge::ConstrainIntRange(range, inner) => {
                    // check that result is still representable
                    let _ = type_to_cpp(self.diags, span, &IrType::Int(range.clone()))?;
                    self.eval(indent, span, inner, stage_read)?
                }
            },
        };
        Ok(result)
    }

    fn impl_to_bits(
        &mut self,
        indent: Indent,
        ty: &IrType,
        result: Temporary,
        result_offset: &str,
        value: &str,
    ) -> Result<(), ErrorGuaranteed> {
        // TODO maybe it's better to switch to just storing bits for everything in C++ too?
        match ty {
            IrType::Bool => {
                swriteln!(self.f, "{indent}{result}[{result_offset}] = {value};",);
            }
            IrType::Int(range) => {
                // TODO this this pretty inefficient
                // TODO this is probably not correct for signed values
                let tmp_i = self.new_temporary();
                let size_bits = IntRepresentation::for_range(range).size_bits();
                swriteln!(
                    self.f,
                    "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {size_bits}; {tmp_i}++) {{"
                );
                swriteln!(
                    self.f,
                    "{indent}{I}{result}[{result_offset} + {tmp_i}] = ({value} >> {tmp_i}) & 1;"
                );
                swriteln!(self.f, "{indent}}}");
            }
            IrType::Tuple(elements) => {
                let mut offset = BigUint::ZERO;
                for (element_i, element) in enumerate(elements) {
                    self.impl_to_bits(
                        indent,
                        element,
                        result,
                        &format!("{} + {}", result_offset, offset),
                        &format!("std::get<{element_i}>({value})"),
                    )?;
                    offset += element.size_bits();
                }
            }
            IrType::Array(ty_inner, ty_len) => {
                let ty_inner_size = ty_inner.size_bits();
                let tmp_i = self.new_temporary();
                swriteln!(
                    self.f,
                    "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {ty_len}; {tmp_i}++) {{"
                );
                self.impl_to_bits(
                    indent.nest(),
                    ty_inner,
                    result,
                    &format!("{result_offset} + {tmp_i} * {ty_inner_size}"),
                    &format!("{value}[{tmp_i}]"),
                )?;
                swriteln!(self.f, "{indent}}}");
            }
        }

        Ok(())
    }

    fn impl_from_bits(
        &mut self,
        indent: Indent,
        ty: &IrType,
        result: &str,
        value: Temporary,
        value_offset: &str,
    ) -> Result<(), ErrorGuaranteed> {
        match ty {
            IrType::Bool => {
                swriteln!(self.f, "{indent}{result} = {value}[{value_offset}];",);
            }
            IrType::Int(range) => {
                // TODO this this pretty inefficient
                // TODO this is probably not correct for signed values
                let tmp_i = self.new_temporary();
                let size_bits = IntRepresentation::for_range(range).size_bits();

                swriteln!(self.f, "{indent}{result} = 0;");
                swriteln!(
                    self.f,
                    "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {size_bits}; {tmp_i}++) {{"
                );
                swriteln!(
                    self.f,
                    "{indent}{I}{result} |= ({value}[{value_offset} + {tmp_i}] << {tmp_i});"
                );
                swriteln!(self.f, "{indent}}}");
            }
            IrType::Tuple(tys_inner) => {
                let mut offset = BigUint::ZERO;
                for (element_i, element) in enumerate(tys_inner) {
                    self.impl_from_bits(
                        indent,
                        element,
                        &format!("std::get<{element_i}>({result})"),
                        value,
                        &format!("{} + {}", value_offset, offset),
                    )?;
                    offset += element.size_bits();
                }
            }
            IrType::Array(ty_inner, ty_len) => {
                let ty_inner_size = ty_inner.size_bits();
                let tmp_i = self.new_temporary();
                swriteln!(
                    self.f,
                    "{indent}for (std::size_t {tmp_i} = 0; {tmp_i} < {ty_len}; {tmp_i}++) {{"
                );
                self.impl_from_bits(
                    indent.nest(),
                    ty_inner,
                    &format!("{result}[{tmp_i}]"),
                    value,
                    &format!("{value_offset} + {tmp_i} * {ty_inner_size}"),
                )?;
                swriteln!(self.f, "{indent}}}");
            }
        }

        Ok(())
    }

    fn generate_nested_block(
        &mut self,
        indent: Indent,
        block: &IrBlock,
        stage_read: Stage,
    ) -> Result<(), ErrorGuaranteed> {
        swriteln!(self.f, "{{");
        self.generate_block(indent.nest(), block, stage_read)?;
        swrite!(self.f, "{indent}}}");
        Ok(())
    }

    fn generate_block(&mut self, indent: Indent, block: &IrBlock, stage_read: Stage) -> Result<(), ErrorGuaranteed> {
        let IrBlock { statements } = block;

        for stmt in statements {
            match &stmt.inner {
                IrStatement::Assign(target, expr) => {
                    let Target {
                        loops: target_loops,
                        inner: target_inner,
                    } = self.eval_assignment_target(indent, stmt.span, target, stage_read)?;
                    let value_eval = self.eval(indent, stmt.span, expr, stage_read)?;

                    for (i, (tmp_i, start, len)) in enumerate(&target_loops) {
                        let indent_i = indent.nest_n(i);
                        swriteln!(
                            self.f,
                            "{indent_i}for (std::size_t {tmp_i} = {start}; {tmp_i} < {start} + {len}; {tmp_i}++) {{"
                        );
                    }

                    let indent_n = indent.nest_n(target_loops.len());
                    swrite!(self.f, "{indent_n}{target_inner} = {value_eval}");
                    for (tmp_i, _, _) in target_loops.iter().rev() {
                        swrite!(self.f, "[{tmp_i}]");
                    }
                    swriteln!(self.f, ";");

                    for i in 0..target_loops.len() {
                        let indent_i = indent.nest_n(i);
                        swriteln!(self.f, "{indent_i}}}");
                    }
                }
                IrStatement::Block(inner) => {
                    swrite!(self.f, "{indent}");
                    self.generate_nested_block(indent, inner, stage_read)?;
                    swriteln!(self.f);
                }
                IrStatement::If(if_stmt) => {
                    let IrIfStatement {
                        condition,
                        then_block,
                        else_block,
                    } = if_stmt;

                    let cond_eval = self.eval(indent, stmt.span, condition, stage_read)?;

                    swrite!(self.f, "{indent}if ({cond_eval})");
                    self.generate_nested_block(indent, then_block, stage_read)?;

                    if let Some(else_block) = else_block {
                        swrite!(self.f, " else ");
                        self.generate_nested_block(indent, else_block, stage_read)?;
                    }

                    swriteln!(self.f);
                }
                IrStatement::PrintLn(s) => {
                    // TODO properly escape string
                    // TODO write to a user-passed in stream, to save users from capturing hacks
                    //   this should probably live in a general context object that deals with any outside IO
                    swriteln!(self.f, "{indent}std::cout << \"{s}\" << std::endl;");
                }
            }
        }

        Ok(())
    }

    fn eval_assignment_target(
        &mut self,
        indent: Indent,
        span: Span,
        target: &IrAssignmentTarget,
        stage_read: Stage,
    ) -> Result<Target, ErrorGuaranteed> {
        let next = Stage::Next;
        let IrAssignmentTarget { base, steps } = target;

        let base_str = match *base {
            IrAssignmentTargetBase::Port(port) => {
                let port_str = port_str(port, &self.module_info.ports[port]);
                format!("*{next}_ports.{port_str}")
            }
            IrAssignmentTargetBase::Register(reg) => {
                let reg_str = reg_str(reg, &self.module_info.registers[reg]);
                format!("{next}_signals.{reg_str}")
            }
            IrAssignmentTargetBase::Wire(wire) => {
                let wire_str = wire_str(wire, &self.module_info.wires[wire]);
                format!("{next}_signals.{wire_str}")
            }
            IrAssignmentTargetBase::Variable(var) => var_str(var, &self.locals[var]),
        };

        let mut curr_str = base_str;
        let mut loops = vec![];

        for step in steps {
            match step {
                IrTargetStep::ArrayIndex(index) => {
                    let index = self.eval(indent, span, index, stage_read)?;
                    swrite!(&mut curr_str, "[{index}]")
                }

                IrTargetStep::ArraySlice(start, len) => {
                    let start = self.eval(indent, span, start, stage_read)?;

                    let tmp_index = self.new_temporary();
                    loops.push((tmp_index, start, len.clone()));

                    swrite!(&mut curr_str, "[{tmp_index}]");
                }
            }
        }

        Ok(Target { loops, inner: curr_str })
    }

    fn new_temporary(&mut self) -> Temporary {
        let index = self.next_temporary_index;
        self.next_temporary_index += 1;
        Temporary(index)
    }
}

struct Target {
    loops: Vec<(Temporary, Evaluated, BigUint)>,
    inner: String,
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
                return Err(diags.report_todo(span, format!("simulator wide integer type: {range}")));
            }
        }
        IrType::Tuple(inner) => {
            let inner_strs = inner.iter().map(|ty| type_to_cpp(diags, span, ty)).try_collect_vec()?;
            let inner_str = inner_strs.join(", ");
            format!("std::tuple<{inner_str}>")
        }
        IrType::Array(inner, len) => {
            // TODO we want to represent boolean arrays als bitfields, either through a template optimization trick
            //   or through a special case here (and in every other place that interacts with arrays)
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
        MaybeIdentifier::Identifier(&Identifier {
            span: port_info.debug_span,
            string: port_info.name.clone(),
        }),
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

fn end_comma(i: usize, len: usize) -> &'static str {
    if i == len - 1 {
        ""
    } else {
        ", "
    }
}
