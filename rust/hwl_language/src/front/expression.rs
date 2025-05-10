use crate::front::assignment::{AssignmentTarget, AssignmentTargetBase};
use crate::front::check::{
    check_hardware_type_for_bit_operation, check_type_contains_compile_value, check_type_contains_value,
    check_type_is_bool, check_type_is_int, check_type_is_int_compile, check_type_is_int_hardware,
    check_type_is_uint_compile, TypeContainsReason,
};
use crate::front::compile::{CompileItemContext, Port, PortInterface, StackEntry};
use crate::front::context::{CompileTimeExpressionContext, ExpressionContext};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::domain::{BlockDomain, DomainSignal, ValueDomain};
use crate::front::function::{error_unique_mismatch, FunctionBits, FunctionBitsKind, FunctionBody, FunctionValue};
use crate::front::implication::{ClosedIncRangeMulti, Implication, ImplicationOp, Implications};
use crate::front::item::{ElaboratedModule, FunctionItemBody};
use crate::front::scope::{NamedValue, Scope, ScopedEntry};
use crate::front::signal::{Polarized, Signal, SignalOrVariable};
use crate::front::steps::{ArrayStep, ArrayStepCompile, ArrayStepHardware, ArraySteps};
use crate::front::types::{ClosedIncRange, HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, ElaboratedInterfaceView, HardwareValue, Value};
use crate::front::variables::{ValueVersioned, VariableValues};
use crate::mid::ir::{
    IrArrayLiteralElement, IrAssignmentTarget, IrBoolBinaryOp, IrExpression, IrExpressionLarge, IrIntArithmeticOp,
    IrIntCompareOp, IrLargeArena, IrStatement, IrVariableInfo,
};
use crate::syntax::ast::{
    ArrayComprehension, ArrayLiteralElement, BinaryOp, BlockExpression, DomainKind, Expression, ExpressionKind,
    Identifier, IntLiteral, MaybeIdentifier, PortDirection, RangeLiteral, RegisterDelay, Spanned, SyncDomain, UnaryOp,
};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::big_int::{BigInt, BigUint};
use crate::util::data::vec_concat;
use crate::util::iter::IterExt;
use crate::util::{result_pair, Never, ResultDoubleExt, ResultNeverExt};
use annotate_snippets::Level;
use itertools::{enumerate, Either};
use std::borrow::Borrow;
use std::cmp::{max, min};
use std::ops::Sub;
use unwrap_match::unwrap_match;

// TODO better name
#[derive(Debug)]
pub enum ValueInner {
    Value(ValueWithImplications),
    PortInterface(PortInterface),
}

pub type ValueWithImplications = Value<CompileValue, HardwareValueWithImplications>;

#[derive(Debug)]
pub struct HardwareValueWithImplications {
    pub value: HardwareValue,
    pub value_versioned: Option<ValueVersioned>,
    pub implications: Implications,
}

impl ValueWithImplications {
    pub fn simple(value: Value) -> Self {
        value.map_hardware(HardwareValueWithImplications::simple)
    }

    pub fn value(self) -> Value {
        self.map_hardware(|v| v.value)
    }

    pub fn value_cloned(&self) -> Value {
        self.as_ref().map_hardware(|v| &v.value).cloned()
    }
}

impl Typed for ValueWithImplications {
    fn ty(&self) -> Type {
        match self {
            Value::Compile(v) => v.ty(),
            Value::Hardware(v) => v.value.ty(),
        }
    }
}

impl HardwareValueWithImplications {
    pub fn simple(value: HardwareValue) -> Self {
        Self {
            value,
            value_versioned: None,
            implications: Implications::default(),
        }
    }
}

#[derive(Debug)]
pub enum EvaluatedId {
    Named(NamedValue),
    Value(Value),
}

impl CompileItemContext<'_, '_> {
    pub fn eval_id(&mut self, scope: &Scope, id: &Identifier) -> Result<Spanned<EvaluatedId>, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let found = scope.find(diags, id)?;
        let def_span = found.defining_span;
        let result = match *found.value {
            ScopedEntry::Named(value) => EvaluatedId::Named(value),
            ScopedEntry::Item(item) => self
                .recurse(StackEntry::ItemUsage(id.span), |s| {
                    Ok(EvaluatedId::Value(Value::Compile(s.eval_item(item)?.clone())))
                })
                .flatten_err()?,
        };
        Ok(Spanned {
            span: def_span,
            inner: result,
        })
    }

    // TODO make all of these functions on ExpressionContext instead of dragging all these parameters around
    pub fn eval_expression<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<Spanned<Value>, ErrorGuaranteed> {
        assert_eq!(self.variables.check(), vars.check());
        Ok(self
            .eval_expression_with_implications(ctx, ctx_block, scope, vars, expected_ty, expr)?
            .map_inner(|r| match r {
                Value::Compile(v) => Value::Compile(v),
                Value::Hardware(v) => Value::Hardware(v.value),
            }))
    }

    pub fn eval_expression_with_implications<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<Spanned<ValueWithImplications>, ErrorGuaranteed> {
        let value = self.eval_expression_inner(ctx, ctx_block, scope, vars, expected_ty, expr)?;
        match value {
            ValueInner::Value(v) => Ok(Spanned::new(expr.span, v)),
            ValueInner::PortInterface(_) => Err(self.refs.diags.report_simple(
                "interface instance expression not allowed here",
                expr.span,
                "this expression evaluates to an interface instance",
            )),
        }
    }

    // TODO return COW to save some allocations?
    // TODO maybe this should return an abstract expression value,
    //   that can then be written (as target), read (as value), typeof-ed, gotten implications, ...
    //   that's awkward for expressions that create statements though, eg. calls
    //   maybe those should push their statements to virtual blocks, and only actually add them once read?
    pub fn eval_expression_inner<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        expr: &Expression,
    ) -> Result<ValueInner, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let result_simple: Value = match &expr.inner {
            ExpressionKind::Dummy => {
                // if dummy expressions were allowed, the caller would have checked for them already
                return Err(diags.report_simple(
                    "dummy expression not allowed in this context",
                    expr.span,
                    "dummy expression used here",
                ));
            }
            ExpressionKind::Undefined => Value::Compile(CompileValue::Undefined),
            ExpressionKind::Type => Value::Compile(CompileValue::Type(Type::Type)),
            ExpressionKind::Wrapped(inner) => {
                return self.eval_expression_inner(ctx, ctx_block, scope, vars, expected_ty, inner);
            }
            ExpressionKind::Block(block_expr) => {
                let BlockExpression { statements, expression } = block_expr;

                let mut scope_inner = Scope::new_child(expr.span, scope);

                // TODO propagate return type?
                let (mut ctx_block, end) =
                    self.elaborate_block_statements(ctx, &mut scope_inner, vars, None, statements)?;
                end.unwrap_outside_function_and_loop(diags)?;

                self.eval_expression(ctx, &mut ctx_block, &scope_inner, vars, expected_ty, expression)?
                    .inner
            }
            ExpressionKind::Id(id) => {
                let result = match self.eval_id(scope, id)?.inner {
                    EvaluatedId::Value(value) => ValueWithImplications::simple(value),
                    EvaluatedId::Named(value) => match value {
                        // TODO report error when combinatorial block
                        //   reads something it has not written yet but will later write to
                        // TODO more generally, report combinatorial cycles
                        NamedValue::Variable(var) => match &vars.var_get(diags, expr.span, var)?.value_and_version {
                            Value::Compile(value) => Value::Compile(value.clone()),
                            &Value::Hardware((ref value, version)) => {
                                let versioned = ValueVersioned {
                                    value: SignalOrVariable::Variable(var),
                                    version,
                                };
                                let with_implications =
                                    apply_implications(ctx, &mut self.large, Some(versioned), value.clone());
                                Value::Hardware(with_implications)
                            }
                        },
                        NamedValue::Port(port) => {
                            ctx.check_ir_context(diags, expr.span, "port")?;
                            return self.eval_port(ctx, vars, Spanned::new(expr.span, port));
                        }
                        NamedValue::PortInterface(interface) => {
                            ctx.check_ir_context(diags, expr.span, "port")?;
                            return Ok(ValueInner::PortInterface(interface));
                        }
                        NamedValue::Wire(wire) => {
                            ctx.check_ir_context(diags, expr.span, "wire")?;
                            let wire_info = &mut self.wires[wire];

                            if let BlockDomain::Clocked(block_domain) = ctx.block_domain() {
                                wire_info
                                    .suggest_domain(Spanned::new(expr.span, ValueDomain::Sync(block_domain.inner)));
                            }
                            let wire_value = wire_info.as_hardware_value(diags, expr.span)?;

                            let versioned = vars.signal_versioned(Signal::Wire(wire));
                            let with_implications = apply_implications(ctx, &mut self.large, versioned, wire_value);
                            Value::Hardware(with_implications)
                        }
                        NamedValue::Register(reg) => {
                            ctx.check_ir_context(diags, expr.span, "register")?;
                            let reg_info = &mut self.registers[reg];

                            if let BlockDomain::Clocked(block_domain) = ctx.block_domain() {
                                reg_info.suggest_domain(Spanned::new(expr.span, block_domain.inner));
                            }
                            let reg_value = reg_info.as_hardware_value(diags, expr.span)?;

                            let versioned = vars.signal_versioned(Signal::Register(reg));
                            let with_implications = apply_implications(ctx, &mut self.large, versioned, reg_value);
                            Value::Hardware(with_implications)
                        }
                    },
                };

                return Ok(ValueInner::Value(result));
            }
            ExpressionKind::TypeFunction => Value::Compile(CompileValue::Type(Type::Function)),
            ExpressionKind::IntLiteral(ref pattern) => {
                let value = match pattern {
                    IntLiteral::Binary(s_raw) => {
                        let s_clean = s_raw[2..].replace('_', "");
                        BigUint::from_str_radix(&s_clean, 2)
                            .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?
                    }
                    IntLiteral::Decimal(s_raw) => {
                        let s_clean = s_raw.replace('_', "");
                        BigUint::from_str_radix(&s_clean, 10)
                            .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?
                    }
                    IntLiteral::Hexadecimal(s) => {
                        let s_hex = s[2..].replace('_', "");
                        BigUint::from_str_radix(&s_hex, 16)
                            .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?
                    }
                };
                Value::Compile(CompileValue::Int(BigInt::from(value)))
            }
            &ExpressionKind::BoolLiteral(literal) => Value::Compile(CompileValue::Bool(literal)),
            // TODO f-string formatting
            ExpressionKind::StringLiteral(literal) => Value::Compile(CompileValue::String(literal.clone())),
            ExpressionKind::ArrayLiteral(values) => {
                self.eval_array_literal(ctx, ctx_block, scope, vars, expected_ty, expr.span, values)?
            }
            ExpressionKind::TupleLiteral(values) => {
                self.eval_tuple_literal(ctx, ctx_block, scope, vars, expected_ty, values)?
            }
            ExpressionKind::RangeLiteral(literal) => {
                let mut eval_bound = |bound: &Expression, name: &str, op_span: Span| {
                    let bound = self.eval_expression_as_compile(
                        scope,
                        vars,
                        &Type::Int(IncRange::OPEN),
                        bound,
                        &format!("range {name}"),
                    )?;
                    let reason = TypeContainsReason::Operator(op_span);
                    check_type_is_int_compile(diags, reason, bound)
                };

                match *literal {
                    RangeLiteral::ExclusiveEnd {
                        op_span,
                        ref start,
                        ref end,
                    } => {
                        let start = start
                            .as_ref()
                            .map(|start| eval_bound(start, "start", op_span))
                            .transpose();
                        let end = end.as_ref().map(|end| eval_bound(end, "end", op_span)).transpose();

                        let range = IncRange {
                            start_inc: start?,
                            end_inc: end?.map(|end| end - 1),
                        };
                        Value::Compile(CompileValue::IntRange(range))
                    }
                    RangeLiteral::InclusiveEnd {
                        op_span,
                        ref start,
                        ref end,
                    } => {
                        let start = start
                            .as_ref()
                            .map(|start| eval_bound(start, "start", op_span))
                            .transpose();
                        let end = eval_bound(end, "end", op_span);

                        let range = IncRange {
                            start_inc: start?,
                            end_inc: Some(end?),
                        };
                        Value::Compile(CompileValue::IntRange(range))
                    }
                    RangeLiteral::Length {
                        op_span,
                        ref start,
                        ref len,
                    } => {
                        // TODO support runtime starts here too (so they can be full values),
                        //   for now those are special-cased in the array indexing evaluation.
                        //   Maybe we want real support for mixed compile/runtime compounds,
                        //     eg. arrays, tuples, ranges, ...
                        let start = eval_bound(start, "start", op_span);
                        let length = eval_bound(len, "length", op_span);

                        let start = start?;
                        let range = IncRange {
                            end_inc: Some(&start + length? - 1),
                            start_inc: Some(start),
                        };
                        Value::Compile(CompileValue::IntRange(range))
                    }
                }
            }
            ExpressionKind::ArrayComprehension(array_comprehension) => {
                let ArrayComprehension {
                    body,
                    index,
                    span_keyword,
                    iter,
                } = array_comprehension;
                let span_keyword = *span_keyword;

                let expected_ty_inner = match expected_ty {
                    Type::Array(inner, _) => inner,
                    _ => &Type::Any,
                };

                let iter = self.eval_expression_as_for_iterator(ctx, ctx_block, scope, vars, iter)?;

                let mut values = vec![];
                for index_value in iter {
                    let index_value = index_value.to_maybe_compile(&mut self.large);
                    let index_var =
                        vars.var_new_immutable_init(&mut self.variables, index.clone(), span_keyword, Ok(index_value));

                    let scope_span = body.span().join(index.span());
                    let mut scope_body = Scope::new_child(scope_span, scope);
                    scope_body.maybe_declare(
                        diags,
                        index.as_ref(),
                        Ok(ScopedEntry::Named(NamedValue::Variable(index_var))),
                    );

                    let value = body
                        .map_inner(|body_expr| {
                            self.eval_expression(ctx, ctx_block, &scope_body, vars, expected_ty_inner, body_expr)
                        })
                        .transpose()?;
                    values.push(value);
                }

                array_literal_combine_values(diags, &mut self.large, expr.span, expected_ty_inner, values)?
            }

            ExpressionKind::UnaryOp(op, operand) => match op.inner {
                UnaryOp::Plus => {
                    let operand =
                        self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, operand)?;
                    let _ = check_type_is_int(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        operand.as_ref().map_inner(ValueWithImplications::value_cloned),
                    )?;
                    return Ok(ValueInner::Value(operand.inner));
                }
                UnaryOp::Neg => {
                    let operand = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, operand)?;
                    let operand_int = check_type_is_int(diags, TypeContainsReason::Operator(op.span), operand)?;

                    match operand_int.inner {
                        Value::Compile(c) => Value::Compile(CompileValue::Int(-c)),
                        Value::Hardware(v) => {
                            let result_range = ClosedIncRange {
                                start_inc: -v.ty.end_inc,
                                end_inc: -v.ty.start_inc,
                            };
                            let result_expr = self.large.push_expr(IrExpressionLarge::IntArithmetic(
                                IrIntArithmeticOp::Sub,
                                result_range.clone(),
                                IrExpression::Int(BigInt::ZERO),
                                v.expr,
                            ));

                            let result = HardwareValue {
                                ty: HardwareType::Int(result_range),
                                domain: v.domain,
                                expr: result_expr,
                            };
                            Value::Hardware(result)
                        }
                    }
                }
                UnaryOp::Not => {
                    let operand =
                        self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, operand)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(op.span),
                        &Type::Bool,
                        operand.as_ref().map_inner(ValueWithImplications::value_cloned).as_ref(),
                        false,
                        false,
                    )?;

                    match operand.inner {
                        Value::Compile(c) => match c {
                            // TODO support boolean array
                            CompileValue::Bool(b) => Value::Compile(CompileValue::Bool(!b)),
                            _ => return Err(diags.report_internal_error(expr.span, "expected bool for unary not")),
                        },
                        Value::Hardware(v) => {
                            let result = HardwareValue {
                                ty: HardwareType::Bool,
                                domain: v.value.domain,
                                expr: self.large.push_expr(IrExpressionLarge::BoolNot(v.value.expr)),
                            };
                            let result_with_implications = HardwareValueWithImplications {
                                value: result,
                                value_versioned: None,
                                implications: v.implications.invert(),
                            };
                            return Ok(ValueInner::Value(Value::Hardware(result_with_implications)));
                        }
                    }
                }
            },
            &ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, left);
                let right = self.eval_expression_with_implications(ctx, ctx_block, scope, vars, &Type::Any, right);
                let result = eval_binary_expression(diags, &mut self.large, expr.span, op, left?, right?)?;
                return Ok(ValueInner::Value(result));
            }
            ExpressionKind::ArrayIndex(base, indices) => {
                self.eval_array_index_expression(ctx, ctx_block, scope, vars, base, indices)?
            }
            ExpressionKind::ArrayType(lens, base) => {
                let lens = lens
                    .inner
                    .iter()
                    .map(|len| {
                        let len = match len {
                            ArrayLiteralElement::Single(len) => len,
                            ArrayLiteralElement::Spread(_, _) => {
                                return Err(diags.report_todo(len.span(), "spread in array type lengths"))
                            }
                        };
                        let len_expected_ty = Type::Int(IncRange {
                            start_inc: Some(BigInt::ZERO),
                            end_inc: None,
                        });
                        let len =
                            self.eval_expression_as_compile(scope, vars, &len_expected_ty, len, "array type length")?;
                        let reason = TypeContainsReason::ArrayLen { span_len: len.span };
                        check_type_is_uint_compile(diags, reason, len)
                    })
                    .try_collect_all_vec();
                let base = self.eval_expression_as_ty(scope, vars, base);

                let lengths = lens?;
                let base = base?;

                // apply lengths inside-out
                let result = lengths
                    .into_iter()
                    .rev()
                    .fold(base.inner, |acc, len| Type::Array(Box::new(acc), len));
                Value::Compile(CompileValue::Type(result))
            }
            ExpressionKind::DotIdIndex(base, index) => {
                return self.eval_dot_id_index(ctx, ctx_block, scope, vars, expected_ty, expr.span, base, index);
            }
            ExpressionKind::DotIntIndex(base, index) => {
                let base_eval = self.eval_expression_inner(ctx, ctx_block, scope, vars, &Type::Any, base)?;

                let index_int = BigUint::from_str_radix(&index.inner, 10)
                    .map_err(|_| diags.report_internal_error(expr.span, "failed to parse int"))?;
                let err_not_tuple = |ty: &str| {
                    let diag = Diagnostic::new("indexing into non-tuple type")
                        .add_error(index.span, "attempt to index into non-tuple type here")
                        .add_info(base.span, format!("base has type `{ty}`"))
                        .finish();
                    diags.report(diag)
                };
                let err_index_out_of_bounds = |len: usize| {
                    let diag = Diagnostic::new("tuple index out of bounds")
                        .add_error(index.span, format!("index `{index_int}` is out of bounds"))
                        .add_info(base.span, format!("base is tuple with length `{len}`"))
                        .finish();
                    diags.report(diag)
                };

                match base_eval {
                    ValueInner::Value(Value::Compile(CompileValue::Tuple(inner))) => {
                        let index = index_int
                            .as_usize_if_lt(inner.len())
                            .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                        Value::Compile(inner[index].clone())
                    }
                    ValueInner::Value(Value::Compile(CompileValue::Type(Type::Tuple(inner)))) => {
                        let index = index_int
                            .as_usize_if_lt(inner.len())
                            .ok_or_else(|| err_index_out_of_bounds(inner.len()))?;
                        Value::Compile(CompileValue::Type(inner[index].clone()))
                    }
                    ValueInner::Value(Value::Hardware(value)) => {
                        let value = value.value;
                        match value.ty {
                            HardwareType::Tuple(inner_tys) => {
                                let index = index_int
                                    .as_usize_if_lt(inner_tys.len())
                                    .ok_or_else(|| err_index_out_of_bounds(inner_tys.len()))?;

                                let expr = IrExpressionLarge::TupleIndex {
                                    base: value.expr,
                                    index: index.into(),
                                };
                                Value::Hardware(HardwareValue {
                                    ty: inner_tys[index].clone(),
                                    domain: value.domain,
                                    expr: self.large.push_expr(expr),
                                })
                            }
                            _ => return Err(err_not_tuple(&value.ty.to_diagnostic_string())),
                        }
                    }
                    ValueInner::Value(v) => return Err(err_not_tuple(&v.ty().to_diagnostic_string())),
                    ValueInner::PortInterface(_) => return Err(err_not_tuple("port interface")),
                }
            }
            ExpressionKind::Call(target, args) => {
                // evaluate target and args
                let target = self.eval_expression_as_compile(scope, vars, &Type::Any, target, "call target")?;
                let args =
                    args.try_map_inner_all(|arg| self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, arg));

                // report errors for invalid target and args
                //   (only after both have been evaluated to get all diagnostics)
                let target_inner = match target.inner {
                    CompileValue::Function(f) => f,
                    _ => {
                        let e = diags.report_simple(
                            "call target must be function",
                            expr.span,
                            format!("got `{}`", target.inner.to_diagnostic_string()),
                        );
                        return Err(e);
                    }
                };
                let args = args?;

                // actually do the call
                // TODO should we do the recursion marker here or inside of the call function?
                let entry = StackEntry::FunctionCall(expr.span);
                let (result_block, result_value) = self
                    .recurse(entry, |s| {
                        s.call_function(ctx, vars, expected_ty, target.span, expr.span, &target_inner, args)
                    })
                    .flatten_err()?;

                if let Some(result_block) = result_block {
                    let result_block_spanned = Spanned {
                        span: expr.span,
                        inner: result_block,
                    };
                    ctx.push_ir_statement_block(ctx_block, result_block_spanned);
                }

                result_value
            }
            ExpressionKind::Builtin(ref args) => self.eval_builtin(ctx, ctx_block, scope, vars, expr.span, args)?,
            ExpressionKind::UnsafeValueWithDomain(value, domain) => {
                let value = self.eval_expression(ctx, ctx_block, scope, vars, expected_ty, value);
                let domain = self.eval_domain(scope, domain);

                let value = value?;
                let domain = domain?;

                match value.inner {
                    // casting compile values is useless but not harmful
                    Value::Compile(value) => Value::Compile(value),
                    Value::Hardware(value) => {
                        // TODO warn if this call is redundant, eg. if the domain would already be safely assignable
                        Value::Hardware(HardwareValue {
                            ty: value.ty,
                            domain: ValueDomain::from_domain_kind(domain.inner),
                            expr: value.expr,
                        })
                    }
                }
            }
            ExpressionKind::RegisterDelay(reg_delay) => {
                let &RegisterDelay {
                    span_keyword,
                    ref value,
                    ref init,
                } = reg_delay;

                // eval
                let value = self.eval_expression(ctx, ctx_block, scope, vars, expected_ty, value);
                let init = self.eval_expression_as_compile(scope, vars, expected_ty, init, "register init");
                let (value, init) = result_pair(value, init)?;

                // figure out type
                let value_ty = value.inner.ty();
                let init_ty = init.inner.ty();
                let ty = value_ty.union(&init_ty, true);
                let ty_hw = ty.as_hardware_type().map_err(|_| {
                    let diag = Diagnostic::new("register type must be representable in hardware")
                        .add_error(
                            span_keyword,
                            format!("got non-hardware type `{}`", ty.to_diagnostic_string()),
                        )
                        .add_info(
                            value.span,
                            format!("from combining `{}`", value_ty.to_diagnostic_string()),
                        )
                        .add_info(
                            init.span,
                            format!("from combining `{}`", init_ty.to_diagnostic_string()),
                        )
                        .finish();
                    diags.report(diag)
                })?;

                // convert values to hardware
                let value = value
                    .inner
                    .as_hardware_value(diags, &mut self.large, value.span, &ty_hw)?;
                let init = init
                    .as_ref()
                    .map_inner(|inner| inner.as_ir_expression_or_undefined(diags, &mut self.large, init.span, &ty_hw))
                    .transpose()?;

                // create a register and variable
                let dummy_id = MaybeIdentifier::Dummy(span_keyword);
                let var_info = IrVariableInfo {
                    ty: ty_hw.as_ir(),
                    debug_info_id: dummy_id.clone(),
                };
                let var = ctx.new_ir_variable(diags, span_keyword, var_info)?;
                let (reg, domain) = ctx.new_ir_register(self, diags, dummy_id, ty_hw.clone(), init)?;

                // do the right shuffle operations
                let stmt_load = IrStatement::Assign(IrAssignmentTarget::variable(var), IrExpression::Register(reg));
                ctx.push_ir_statement(diags, ctx_block, Spanned::new(span_keyword, stmt_load))?;

                let stmt_store = IrStatement::Assign(IrAssignmentTarget::register(reg), value.expr);
                ctx.push_ir_statement(diags, ctx_block, Spanned::new(span_keyword, stmt_store))?;

                // return the variable, now containing the previous value of the register
                Value::Hardware(HardwareValue {
                    ty: ty_hw,
                    domain: ValueDomain::Sync(domain),
                    expr: IrExpression::Variable(var),
                })
            }
        };

        Ok(ValueInner::Value(ValueWithImplications::simple(result_simple)))
    }

    fn eval_dot_id_index<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        expr_span: Span,
        base: &Expression,
        index: &Identifier,
    ) -> Result<ValueInner, ErrorGuaranteed> {
        // TODO make sure users don't accidentally define fields/variants/functions with the same name
        let diags = self.refs.diags;

        let base_eval = self.eval_expression_inner(ctx, ctx_block, scope, vars, &Type::Any, base)?;
        let index_str = index.string.as_str();

        // interface fields
        let base_eval = match base_eval {
            ValueInner::PortInterface(port_interface) => {
                // get the underlying port
                let port_interface_info = &self.port_interfaces[port_interface];
                let interface_info = self
                    .refs
                    .shared
                    .elaboration_arenas
                    .interface_info(port_interface_info.view.interface)?;
                let port_index = interface_info.ports.get_index_of(index_str).ok_or_else(|| {
                    let diag = Diagnostic::new(format!("port `{}` not found on interface", index_str))
                        .add_error(index.span, "attempt to access port here")
                        .add_info(interface_info.id.span(), "interface declared here")
                        .finish();
                    diags.report(diag)
                })?;
                let port = port_interface_info.ports[port_index];

                return self.eval_port(ctx, vars, Spanned::new(expr_span, port));
            }
            ValueInner::Value(base_eval) => base_eval,
        };

        // interface views
        if let &Value::Compile(CompileValue::Interface(base_interface)) = &base_eval {
            let info = self.refs.shared.elaboration_arenas.interface_info(base_interface)?;
            let _ = info.get_view(diags, index)?;

            let interface_view = ElaboratedInterfaceView {
                interface: base_interface,
                view: index_str.to_owned(),
            };
            let result = Value::Compile(CompileValue::InterfaceView(interface_view));
            return Ok(ValueInner::Value(result));
        }

        // common type attributes
        if let Value::Compile(CompileValue::Type(ty)) = &base_eval {
            match index_str {
                "size_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, Spanned::new(base.span, ty))?;
                    let width = ty_hw.as_ir().size_bits();
                    let result = Value::Compile(CompileValue::Int(width.into()));
                    return Ok(ValueInner::Value(result));
                }
                // TODO all of these should return functions with a single params,
                //   without the need for scope capturing
                "to_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, Spanned::new(base.span, ty))?;
                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::ToBits,
                    };
                    let result = Value::Compile(CompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                "from_bits" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, Spanned::new(base.span, ty))?;

                    if !ty_hw.every_bit_pattern_is_valid() {
                        let diag =
                            Diagnostic::new("from_bits is only allowed for types where every bit pattern is valid")
                                .add_error(
                                    base.span,
                                    format!("got type `{}` with invalid bit patterns", ty_hw.to_diagnostic_string()),
                                )
                                .footer(
                                    Level::Help,
                                    "consider using use target type where every bit pattern is valid",
                                )
                                .footer(
                                    Level::Help,
                                    "if you know the bits are valid for this type, use `from_bits_unsafe´ instead",
                                )
                                .finish();
                        return Err(diags.report(diag));
                    }

                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::FromBits,
                    };
                    let result = Value::Compile(CompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                "from_bits_unsafe" => {
                    let ty_hw = check_hardware_type_for_bit_operation(diags, Spanned::new(base.span, ty))?;
                    let func = FunctionBits {
                        ty_hw,
                        kind: FunctionBitsKind::FromBits,
                    };
                    let result = Value::Compile(CompileValue::Function(FunctionValue::Bits(func)));
                    return Ok(ValueInner::Value(result));
                }
                _ => {}
            }
        }

        // struct new
        if let &Value::Compile(CompileValue::Type(Type::Struct(elab, _))) = &base_eval {
            if index_str == "new" {
                let result = Value::Compile(CompileValue::Function(FunctionValue::StructNew(elab)));
                return Ok(ValueInner::Value(result));
            }
        }

        let base_item_function = match &base_eval {
            Value::Compile(CompileValue::Function(FunctionValue::User(func))) => match &func.body.inner {
                FunctionBody::ItemBody(body) => Some(body),
                _ => None,
            },
            _ => None,
        };
        if let Some(&FunctionItemBody::Struct(unique, _)) = base_item_function {
            if index_str == "new" {
                let func = FunctionValue::StructNewInfer(unique);
                let result = Value::Compile(CompileValue::Function(func));
                return Ok(ValueInner::Value(result));
            }
        }

        // enum variants
        let eval_enum = |elab, variant_tys: &Vec<Option<Type>>| {
            let info = self.refs.shared.elaboration_arenas.enum_info(elab)?;
            let variant_index = info.find_variant(diags, Spanned::new(index.span, index_str))?;
            let (_, content_ty) = &info.variants[variant_index];

            let result = match content_ty {
                None => CompileValue::Enum(elab, variant_tys.clone(), (variant_index, None)),
                Some(_) => CompileValue::Function(FunctionValue::EnumNew(elab, variant_index)),
            };

            Ok(ValueInner::Value(Value::Compile(result)))
        };

        if let &Value::Compile(CompileValue::Type(Type::Enum(elab, ref variant_tys))) = &base_eval {
            return eval_enum(elab, variant_tys);
        }
        if let Some(&FunctionItemBody::Enum(unique, _)) = base_item_function {
            if let Type::Enum(expected, variant_tys) = expected_ty {
                let expected_info = self.refs.shared.elaboration_arenas.enum_info(*expected)?;
                if expected_info.unique == unique {
                    return eval_enum(*expected, variant_tys);
                } else {
                    return Err(diags.report(error_unique_mismatch(
                        "struct",
                        base.span,
                        expected_info.unique.span_id(),
                        unique.span_id(),
                    )));
                }
            } else {
                // TODO check if there is any possible variant for this index string,
                //   otherwise we'll get confusing and delayed error messages
                let func = FunctionValue::EnumNewInfer(unique, index_str.to_owned());
                return Ok(ValueInner::Value(Value::Compile(CompileValue::Function(func))));
            }
        }

        // struct fields
        let base_ty = base_eval.ty();
        if let Type::Struct(elab, _) = base_ty {
            let info = self.refs.shared.elaboration_arenas.struct_info(elab)?;
            let field_index = info.fields.get_index_of(index_str).ok_or_else(|| {
                let diag = Diagnostic::new("field not found")
                    .add_info(base.span, format!("base has type `{}`", base_ty.to_diagnostic_string()))
                    .add_error(index.span, "attempt to access non-existing field here")
                    .add_info(info.span_body, "struct fields declared here")
                    .finish();
                diags.report(diag)
            })?;

            let result = match base_eval {
                Value::Compile(base_eval) => match base_eval {
                    CompileValue::Struct(_, _, field_values) => Value::Compile(field_values[field_index].clone()),
                    _ => return Err(diags.report_internal_error(expr_span, "expected struct compile value")),
                },
                Value::Hardware(base_eval) => {
                    let base_eval = base_eval.value;
                    match base_eval.ty {
                        HardwareType::Struct(_, field_types) => {
                            let expr = IrExpressionLarge::TupleIndex {
                                base: base_eval.expr,
                                index: field_index.into(),
                            };
                            Value::Hardware(HardwareValue {
                                ty: field_types[field_index].clone(),
                                domain: base_eval.domain,
                                expr: self.large.push_expr(expr),
                            })
                        }
                        _ => return Err(diags.report_internal_error(expr_span, "expected struct hardware value")),
                    }
                }
            };

            return Ok(ValueInner::Value(ValueWithImplications::simple(result)));
        }

        // fallthrough into error
        let diag = Diagnostic::new("invalid dot index expression")
            .add_info(base.span, format!("base has type `{}`", base_ty.to_diagnostic_string()))
            .add_error(index.span, format!("no attribute found with with name `{index_str}`"))
            .finish();
        Err(diags.report(diag))
    }

    fn eval_port<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        vars: &VariableValues,
        port: Spanned<Port>,
    ) -> Result<ValueInner, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let port_info = &self.ports[port.inner];

        match port_info.direction.inner {
            PortDirection::Input => {
                let versioned = vars.signal_versioned(Signal::Port(port.inner));
                let with_implications =
                    apply_implications(ctx, &mut self.large, versioned, port_info.as_hardware_value());
                Ok(ValueInner::Value(Value::Hardware(with_implications)))
            }
            PortDirection::Output => Err(diags.report_todo(port.span, "read back from output port")),
        }
    }

    fn eval_array_literal<C: ExpressionContext, E: Borrow<Expression>>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut <C as ExpressionContext>::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        expr_span: Span,
        values: &[ArrayLiteralElement<E>],
    ) -> Result<Value, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // intentionally ignore the length, the caller can pass "0" when they have no opinion on it
        // TODO if we stop ignoring the length at some point, then we can infer lengths in eg. `[false] * _`
        let expected_ty_inner = match expected_ty {
            Type::Array(inner, _len) => &**inner,
            _ => &Type::Any,
        };

        // evaluate
        let values = values
            .iter()
            .map(|v| {
                let expected_ty_curr = match v {
                    ArrayLiteralElement::Single(_) => expected_ty_inner,
                    ArrayLiteralElement::Spread(_, _) => {
                        &Type::Array(Box::new(expected_ty_inner.clone()), BigUint::ZERO)
                    }
                };

                v.map_inner(|value_inner| {
                    self.eval_expression(ctx, ctx_block, scope, vars, expected_ty_curr, value_inner.borrow())
                })
                .transpose()
            })
            .try_collect_all_vec()?;

        // combine into compile or non-compile value
        array_literal_combine_values(diags, &mut self.large, expr_span, expected_ty_inner, values)
    }

    fn eval_tuple_literal<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut <C as ExpressionContext>::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        values: &Vec<Expression>,
    ) -> Result<Value, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let expected_tys_inner = match expected_ty {
            Type::Tuple(tys) if tys.len() == values.len() => Some(tys),
            _ => None,
        };

        // evaluate
        let values = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let expected_ty_i = expected_tys_inner.map_or(&Type::Any, |tys| &tys[i]);
                self.eval_expression(ctx, ctx_block, scope, vars, expected_ty_i, v)
            })
            .try_collect_all_vec()?;

        // combine into compile or non-compile value
        let first_non_compile = values
            .iter()
            .find(|v| !matches!(v.inner, Value::Compile(_)))
            .map(|v| v.span);
        Ok(if let Some(first_non_compile) = first_non_compile {
            // at least one non-compile, turn everything into IR
            let mut result_ty = vec![];
            let mut result_domain = ValueDomain::CompileTime;
            let mut result_expr = vec![];

            for (i, value) in enumerate(values) {
                let expected_ty_inner = if let Some(expected_tys_inner) = expected_tys_inner {
                    expected_tys_inner[i].clone()
                } else {
                    value.inner.ty()
                };
                let expected_ty_inner_hw = expected_ty_inner.as_hardware_type().map_err(|_| {
                    let message = format!(
                        "tuple element has inferred type `{}` which is not representable in hardware",
                        expected_ty_inner.to_diagnostic_string()
                    );
                    let diag = Diagnostic::new("hardware tuple elements need to be representable in hardware")
                        .add_error(value.span, message)
                        .add_info(first_non_compile, "necessary because this other tuple element is not a compile-time value, which forces the entire tuple to be hardware")
                        .finish();
                    diags.report(diag)
                })?;

                let value_ir =
                    value
                        .inner
                        .as_hardware_value(diags, &mut self.large, value.span, &expected_ty_inner_hw)?;
                result_ty.push(value_ir.ty);
                result_domain = result_domain.join(value_ir.domain);
                result_expr.push(value_ir.expr);
            }

            Value::Hardware(HardwareValue {
                ty: HardwareType::Tuple(result_ty),
                domain: result_domain,
                expr: self.large.push_expr(IrExpressionLarge::TupleLiteral(result_expr)),
            })
        } else if values
            .iter()
            .all(|v| matches!(v.inner, Value::Compile(CompileValue::Type(_))))
        {
            // all type
            let tys = values
                .into_iter()
                .map(|v| unwrap_match!(v.inner, Value::Compile(CompileValue::Type(v)) => v))
                .collect();
            Value::Compile(CompileValue::Type(Type::Tuple(tys)))
        } else {
            // all compile
            let values = values
                .into_iter()
                .map(|v| unwrap_match!(v.inner, Value::Compile(v) => v))
                .collect();
            Value::Compile(CompileValue::Tuple(values))
        })
    }

    fn eval_array_index_expression<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        base: &Expression,
        indices: &Spanned<Vec<Expression>>,
    ) -> Result<Value, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let base = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, base);
        let steps = indices
            .inner
            .iter()
            .map(|index| self.eval_expression_as_array_step(ctx, ctx_block, scope, vars, index))
            .try_collect_all_vec();

        let base = base?;
        let steps = ArraySteps::new(steps?);

        steps.apply_to_value(diags, &mut self.large, base)
    }

    fn eval_expression_as_array_step<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        index: &Expression,
    ) -> Result<Spanned<ArrayStep>, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // special case range with length, it can have a hardware start index
        let step = if let &ExpressionKind::RangeLiteral(RangeLiteral::Length {
            op_span,
            ref start,
            ref len,
        }) = &index.inner
        {
            let reason = TypeContainsReason::Operator(op_span);
            let start = self
                .eval_expression(ctx, ctx_block, scope, vars, &Type::Any, start)
                .and_then(|start| check_type_is_int(diags, reason, start));

            let len_expected_ty = Type::Int(IncRange {
                start_inc: Some(BigInt::ZERO),
                end_inc: None,
            });
            let len = self
                .eval_expression_as_compile(scope, vars, &len_expected_ty, len, "range length")
                .and_then(|len| check_type_is_uint_compile(diags, reason, len));

            let start = start?;
            let len = len?;

            match start.inner {
                Value::Compile(start) => ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, len: Some(len) }),
                Value::Hardware(start) => ArrayStep::Hardware(ArrayStepHardware::ArraySlice { start, len }),
            }
        } else {
            let index_eval = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, index)?;

            match index_eval.transpose() {
                Value::Compile(index_or_slice) => match index_or_slice.inner {
                    CompileValue::Int(index) => ArrayStep::Compile(ArrayStepCompile::ArrayIndex(index)),
                    CompileValue::IntRange(range) => {
                        let start = range.start_inc.clone().unwrap_or(BigInt::ZERO);
                        match &range.end_inc {
                            None => ArrayStep::Compile(ArrayStepCompile::ArraySlice { start, len: None }),
                            Some(end_inc) => {
                                let slice_len = BigUint::try_from(end_inc - &start + 1).map_err(|_| {
                                    diags.report_internal_error(
                                        index.span,
                                        format!("slice range end cannot be below start, got range `{range}`"),
                                    )
                                })?;
                                ArrayStep::Compile(ArrayStepCompile::ArraySlice {
                                    start,
                                    len: Some(slice_len),
                                })
                            }
                        }
                    }
                    _ => {
                        return Err(diags.report_simple(
                            "array index needs to be an int or a range",
                            index_or_slice.span,
                            format!("got `{}`", index_or_slice.inner.to_diagnostic_string()),
                        ));
                    }
                },
                Value::Hardware(index) => {
                    // TODO make this error message better, specifically refer to non-compile-time index
                    let reason = TypeContainsReason::ArrayIndex { span_index: index.span };
                    let index = check_type_is_int_hardware(diags, reason, index)?;
                    ArrayStep::Hardware(ArrayStepHardware::ArrayIndex(index.inner))
                }
            }
        };

        Ok(Spanned {
            span: index.span,
            inner: step,
        })
    }

    // TODO replace builtin+import+prelude with keywords?
    fn eval_builtin<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expr_span: Span,
        args: &Spanned<Vec<Expression>>,
    ) -> Result<Value, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // evaluate args
        let args_eval = args
            .inner
            .iter()
            .map(|arg| {
                Ok(self
                    .eval_expression(ctx, ctx_block, scope, vars, &Type::Any, arg)?
                    .inner)
            })
            .try_collect_all_vec()?;

        if let (Some(Value::Compile(CompileValue::String(a0))), Some(Value::Compile(CompileValue::String(a1)))) =
            (args_eval.get(0), args_eval.get(1))
        {
            let rest = &args_eval[2..];
            let print_compile = |v: &Value| {
                let value_str = match v {
                    // TODO print strings without quotes
                    Value::Compile(v) => v.to_diagnostic_string(),
                    // TODO less ugly formatting for HardwareValue
                    Value::Hardware(v) => {
                        let HardwareValue { ty, domain, expr: _ } = v;
                        let ty_str = ty.to_diagnostic_string();
                        let domain_str = domain.to_diagnostic_string(self);
                        format!("HardwareValue {{ ty: {ty_str}, domain: {domain_str}, expr: _, }}")
                    }
                };
                self.refs.print_handler.println(&value_str);
            };

            match (a0.as_str(), a1.as_str(), rest) {
                ("type", "any", []) => return Ok(Value::Compile(CompileValue::Type(Type::Any))),
                ("type", "bool", []) => return Ok(Value::Compile(CompileValue::Type(Type::Bool))),
                ("type", "str", []) => return Ok(Value::Compile(CompileValue::Type(Type::String))),
                ("type", "Range", []) => return Ok(Value::Compile(CompileValue::Type(Type::Range))),
                // TODO maybe int/uint should just bit builtins
                ("type", "int_range", [Value::Compile(CompileValue::IntRange(range))]) => {
                    return Ok(Value::Compile(CompileValue::Type(Type::Int(range.clone()))));
                }
                ("fn", "typeof", [value]) => return Ok(Value::Compile(CompileValue::Type(value.ty()))),
                ("fn", "print", [value]) => {
                    if ctx.is_ir_context() {
                        if let Value::Compile(CompileValue::String(value)) = value {
                            let stmt = Spanned::new(expr_span, IrStatement::PrintLn(value.clone()));
                            ctx.push_ir_statement(diags, ctx_block, stmt)?;
                            return Ok(Value::Compile(CompileValue::UNIT));
                        }
                        // fallthough
                    } else {
                        print_compile(value);
                        return Ok(Value::Compile(CompileValue::UNIT));
                    }
                }
                (
                    "fn",
                    "assert",
                    &[Value::Compile(CompileValue::Bool(cond)), Value::Compile(CompileValue::String(ref msg))],
                ) => {
                    return if cond {
                        Ok(Value::Compile(CompileValue::UNIT))
                    } else {
                        Err(diags.report_simple(
                            format!("assertion failed with message {:?}", msg),
                            expr_span,
                            "failed here",
                        ))
                    }
                }
                ("fn", "assert", [Value::Hardware(_), Value::Compile(CompileValue::String(_))]) => {
                    return Err(diags.report_todo(expr_span, "runtime assert"));
                }
                ("fn", "unsafe_bool_to_clock", [Value::Hardware(v)]) => match v.ty {
                    HardwareType::Bool => {
                        return Ok(Value::Hardware(HardwareValue {
                            ty: HardwareType::Clock,
                            domain: ValueDomain::Clock,
                            expr: v.expr.clone(),
                        }));
                    }
                    _ => {}
                },
                ("fn", "unsafe_clock_to_bool", [Value::Hardware(v)]) => match v.ty {
                    HardwareType::Clock => {
                        // TODO what domain should this return?
                        return Ok(Value::Hardware(HardwareValue {
                            ty: HardwareType::Bool,
                            domain: ValueDomain::Async,
                            expr: v.expr.clone(),
                        }));
                    }
                    _ => {}
                },
                // fallthrough into err
                _ => {}
            }
        }

        // TODO this causes a strange error message when people call eg. int_range with non-compile args
        let diag = Diagnostic::new("invalid builtin arguments")
            .snippet(expr_span)
            .add_error(args.span, "invalid args")
            .finish()
            .finish();
        Err(diags.report(diag))
    }

    pub fn eval_expression_as_compile(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        expected_ty: &Type,
        expr: &Expression,
        reason: &str,
    ) -> Result<Spanned<CompileValue>, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let mut ctx = CompileTimeExpressionContext {
            span: expr.span,
            reason: reason.to_owned(),
        };
        let mut ctx_block = ();

        let value_eval = self
            .eval_expression(&mut ctx, &mut ctx_block, scope, vars, expected_ty, expr)?
            .inner;
        match value_eval {
            Value::Compile(c) => Ok(Spanned {
                span: expr.span,
                inner: c,
            }),
            Value::Hardware(ir_expr) => Err(diags.report_simple(
                format!("{reason} must be a compile-time value"),
                expr.span,
                format!("got value with domain `{}`", ir_expr.domain.to_diagnostic_string(self)),
            )),
        }
    }

    pub fn eval_expression_as_ty(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        expr: &Expression,
    ) -> Result<Spanned<Type>, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // TODO unify this message with the one when a normal type-check fails
        match self
            .eval_expression_as_compile(scope, vars, &Type::Type, expr, "type")?
            .inner
        {
            CompileValue::Type(ty) => Ok(Spanned {
                span: expr.span,
                inner: ty,
            }),
            value => Err(diags.report_simple(
                "expected type, got value",
                expr.span,
                format!("got value `{}`", value.to_diagnostic_string()),
            )),
        }
    }

    pub fn eval_expression_as_ty_hardware(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        expr: &Expression,
        reason: &str,
    ) -> Result<Spanned<HardwareType>, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let ty = self.eval_expression_as_ty(scope, vars, expr)?.inner;
        let ty_hw = ty.as_hardware_type().map_err(|_| {
            diags.report_simple(
                format!("{} type must be representable in hardware", reason),
                expr.span,
                format!("got type `{}`", ty.to_diagnostic_string()),
            )
        })?;
        Ok(Spanned {
            span: expr.span,
            inner: ty_hw,
        })
    }

    // TODO move typechecks here, immediately returning expected type if any
    pub fn eval_expression_as_assign_target<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        expr: &Expression,
    ) -> Result<Spanned<AssignmentTarget>, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // TODO include definition site (at least for named values)
        let build_err =
            |actual: &str| diags.report_simple("expected assignment target", expr.span, format!("got {}", actual));

        let result = match &expr.inner {
            ExpressionKind::Id(id) => match self.eval_id(scope, id)?.inner {
                EvaluatedId::Value(_) => return Err(build_err("value")),
                EvaluatedId::Named(s) => match s {
                    NamedValue::Variable(v) => {
                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Variable(v)))
                    }
                    NamedValue::Port(port) => {
                        // check direction
                        let direction = self.ports[port].direction;
                        match direction.inner {
                            PortDirection::Input => return Err(build_err("input port")),
                            PortDirection::Output => {}
                        }

                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Port(port)))
                    }
                    NamedValue::PortInterface(_) => return Err(build_err("port interface")),
                    NamedValue::Wire(w) => {
                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Wire(w)))
                    }
                    NamedValue::Register(r) => {
                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Register(r)))
                    }
                },
            },
            ExpressionKind::ArrayIndex(inner_target, indices) => {
                let inner_target = self.eval_expression_as_assign_target(ctx, ctx_block, scope, vars, inner_target);
                let array_steps = indices
                    .inner
                    .iter()
                    .map(|index| self.eval_expression_as_array_step(ctx, ctx_block, scope, vars, index))
                    .try_collect_all_vec();

                let inner_target = inner_target?;
                let array_steps = ArraySteps::new(array_steps?);

                let AssignmentTarget {
                    base: inner_base,
                    array_steps: inner_array_steps,
                } = inner_target.inner;
                if !inner_array_steps.is_empty() {
                    return Err(diags.report_todo(expr.span, "combining target expressions"));
                }

                AssignmentTarget {
                    base: inner_base,
                    array_steps,
                }
            }
            ExpressionKind::DotIdIndex(base, index) => match &base.inner {
                ExpressionKind::Id(base) => match self.eval_id(scope, base)?.inner {
                    EvaluatedId::Named(NamedValue::PortInterface(base)) => {
                        // get port
                        let port_interface_info = &self.port_interfaces[base];
                        let interface_info = self
                            .refs
                            .shared
                            .elaboration_arenas
                            .interface_info(port_interface_info.view.interface)?;
                        let (port_index, _) = interface_info.get_port(diags, index)?;
                        let port = port_interface_info.ports[port_index];

                        // check direction
                        let direction = self.ports[port].direction;
                        match direction.inner {
                            PortDirection::Input => return Err(build_err("input port")),
                            PortDirection::Output => {}
                        }

                        AssignmentTarget::simple(Spanned::new(expr.span, AssignmentTargetBase::Port(port)))
                    }
                    _ => {
                        return Err(diags.report_simple(
                            "dot index only supported on interface ports",
                            base.span,
                            "got other named value here",
                        ));
                    }
                },
                _ => {
                    return Err(diags.report_simple(
                        "dot index only supported on interface ports",
                        base.span,
                        "got other expression here",
                    ))
                }
            },
            ExpressionKind::DotIntIndex(_, _) => {
                return Err(diags.report_todo(expr.span, "assignment target dot int index"))?
            }
            _ => return Err(build_err("other expression")),
        };

        Ok(Spanned {
            span: expr.span,
            inner: result,
        })
    }

    pub fn eval_expression_as_domain_signal(
        &mut self,
        scope: &Scope,
        expr: &Expression,
    ) -> Result<Spanned<DomainSignal>, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let build_err =
            |actual: &str| diags.report_simple("expected domain signal", expr.span, format!("got `{}`", actual));
        self.try_eval_expression_as_domain_signal(scope, expr, build_err)
            .map_err(|e| e.into_inner())
    }

    pub fn try_eval_expression_as_domain_signal<E>(
        &mut self,
        scope: &Scope,
        expr: &Expression,
        build_err: impl Fn(&str) -> E,
    ) -> Result<Spanned<DomainSignal>, Either<E, ErrorGuaranteed>> {
        // TODO expand to allow general expressions again (which then probably create implicit signals)?
        let result = match &expr.inner {
            ExpressionKind::UnaryOp(
                Spanned {
                    span: _,
                    inner: UnaryOp::Not,
                },
                inner,
            ) => {
                let inner = self
                    .eval_expression_as_domain_signal(scope, inner)
                    .map_err(|e| Either::Right(e))?
                    .inner;
                Ok(inner.invert())
            }
            ExpressionKind::Id(id) => {
                let value = self.eval_id(scope, id).map_err(|e| Either::Right(e))?;
                match value.inner {
                    EvaluatedId::Value(_) => Err(build_err("value")),
                    EvaluatedId::Named(s) => match s {
                        NamedValue::Variable(_) => Err(build_err("variable")),
                        NamedValue::Port(p) => Ok(Polarized::new(Signal::Port(p))),
                        NamedValue::PortInterface(_) => Err(build_err("port interface")),
                        NamedValue::Wire(w) => Ok(Polarized::new(Signal::Wire(w))),
                        NamedValue::Register(r) => Ok(Polarized::new(Signal::Register(r))),
                    },
                }
            }
            _ => Err(build_err("expression")),
        };

        let result = result.map_err(|e| Either::Left(e))?;
        Ok(Spanned {
            span: expr.span,
            inner: result,
        })
    }

    pub fn eval_domain_sync(
        &mut self,
        scope: &Scope,
        domain: &SyncDomain<Box<Expression>>,
    ) -> Result<SyncDomain<DomainSignal>, ErrorGuaranteed> {
        let SyncDomain { clock, reset } = domain;
        let clock = self.eval_expression_as_domain_signal(scope, clock);
        let reset = reset
            .as_ref()
            .map(|reset| self.eval_expression_as_domain_signal(scope, reset));
        Ok(SyncDomain {
            clock: clock?.inner,
            reset: reset.transpose()?.map(|r| r.inner),
        })
    }

    pub fn eval_domain(
        &mut self,
        scope: &Scope,
        domain: &Spanned<DomainKind<Box<Expression>>>,
    ) -> Result<Spanned<DomainKind<DomainSignal>>, ErrorGuaranteed> {
        let result = match &domain.inner {
            DomainKind::Const => Ok(DomainKind::Const),
            DomainKind::Async => Ok(DomainKind::Async),
            DomainKind::Sync(domain) => self.eval_domain_sync(scope, domain).map(DomainKind::Sync),
        };

        Ok(Spanned {
            span: domain.span,
            inner: result?,
        })
    }

    pub fn eval_port_domain(
        &mut self,
        scope: &Scope,
        domain: &Spanned<DomainKind<Box<Expression>>>,
    ) -> Result<Spanned<DomainKind<Polarized<Port>>>, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let result = self.eval_domain(scope, domain)?;

        Ok(Spanned {
            span: result.span,
            inner: match result.inner {
                DomainKind::Const => DomainKind::Const,
                DomainKind::Async => DomainKind::Async,
                DomainKind::Sync(sync) => DomainKind::Sync(sync.try_map_signal(|signal| {
                    signal.try_map_inner(|signal| match signal {
                        Signal::Port(port) => Ok(port),
                        Signal::Wire(_) => Err(diags.report_internal_error(domain.span, "expected port, got wire")),
                        Signal::Register(_) => {
                            Err(diags.report_internal_error(domain.span, "expected port, got register"))
                        }
                    })
                })?),
            },
        })
    }

    pub fn eval_expression_as_for_iterator<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        ctx_block: &mut C::Block,
        scope: &Scope,
        vars: &mut VariableValues,
        iter: &Expression,
    ) -> Result<ForIterator, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let iter = self.eval_expression(ctx, ctx_block, scope, vars, &Type::Any, iter)?;
        let iter_span = iter.span;

        let result = match iter.inner {
            Value::Compile(CompileValue::IntRange(iter)) => {
                let IncRange { start_inc, end_inc } = iter;
                let start_inc = match start_inc {
                    Some(start_inc) => start_inc,
                    None => {
                        return Err(diags.report_simple(
                            "for loop iterator range must have start value",
                            iter_span,
                            format!(
                                "got range `{}`",
                                IncRange {
                                    start_inc: None,
                                    end_inc
                                }
                            ),
                        ))
                    }
                };

                ForIterator::Int {
                    next: start_inc,
                    end_inc,
                }
            }
            Value::Compile(CompileValue::Array(iter)) => ForIterator::CompileArray(iter.into_iter()),
            Value::Hardware(HardwareValue {
                ty: HardwareType::Array(ty_inner, len),
                domain,
                expr: array_expr,
            }) => {
                let base = HardwareValue {
                    ty: (*ty_inner, len),
                    domain,
                    expr: array_expr,
                };
                ForIterator::HardwareArray {
                    next: BigUint::ZERO,
                    base,
                }
            }
            _ => {
                throw!(diags.report_simple(
                    "invalid for loop iterator type, must be range or array",
                    iter.span,
                    format!("iterator has type `{}`", iter.inner.ty().to_diagnostic_string())
                ))
            }
        };

        Ok(result)
    }

    pub fn eval_expression_as_module(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        span_keyword: Span,
        expr: &Expression,
    ) -> Result<ElaboratedModule, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let eval = self.eval_expression_as_compile(scope, vars, &Type::Module, expr, "module")?;

        let reason = TypeContainsReason::InstanceModule(span_keyword);
        check_type_contains_compile_value(diags, reason, &Type::Module, eval.as_ref(), false)?;

        match eval.inner {
            CompileValue::Module(elab) => Ok(elab),
            _ => Err(diags.report_internal_error(eval.span, "expected module, should have already been checked")),
        }
    }
}

pub enum ForIterator {
    Int {
        next: BigInt,
        end_inc: Option<BigInt>,
    },
    CompileArray(std::vec::IntoIter<CompileValue>),
    HardwareArray {
        next: BigUint,
        base: HardwareValue<(HardwareType, BigUint)>,
    },
}

impl Iterator for ForIterator {
    type Item = Value<CompileValue, HardwareValue<HardwareType, IrExpressionLarge>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ForIterator::Int { next, end_inc } => {
                if let Some(end_inc) = end_inc {
                    if next > end_inc {
                        return None;
                    }
                }

                let curr = Value::Compile(CompileValue::Int(next.clone()));
                *next += 1;
                Some(curr)
            }
            ForIterator::CompileArray(iter) => iter.next().map(Value::Compile),
            ForIterator::HardwareArray { next, base } => {
                let HardwareValue {
                    ty: (ty_inner, ty_len),
                    domain,
                    expr: base_expr,
                } = &*base;

                if &*next >= ty_len {
                    return None;
                }
                let index_expr = IrExpression::Int(BigInt::from(next.clone()));
                *next += 1u8;

                let element_expr = IrExpressionLarge::ArrayIndex {
                    base: base_expr.clone(),
                    index: index_expr,
                };
                Some(Value::Hardware(HardwareValue {
                    ty: (*ty_inner).clone(),
                    domain: *domain,
                    expr: element_expr,
                }))
            }
        }
    }
}

fn pair_compile_int(
    left: Spanned<Value<BigInt, HardwareValue<ClosedIncRange<BigInt>>>>,
    right: Spanned<Value<BigInt, HardwareValue<ClosedIncRange<BigInt>>>>,
) -> Value<
    (Spanned<BigInt>, Spanned<BigInt>),
    (
        Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
        Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
    ),
> {
    pair_compile_general(left, right, |x| {
        Result::<_, Never>::Ok(HardwareValue {
            ty: ClosedIncRange::single(x.inner.clone()),
            domain: ValueDomain::CompileTime,
            expr: IrExpression::Int(x.inner),
        })
    })
    .remove_never()
}

fn pair_compile_general<C, T, E>(
    left: Spanned<Value<C, T>>,
    right: Spanned<Value<C, T>>,
    to_other: impl Fn(Spanned<C>) -> Result<T, E>,
) -> Result<Value<(Spanned<C>, Spanned<C>), (Spanned<T>, Spanned<T>)>, E> {
    match (left.inner, right.inner) {
        (Value::Compile(left_inner), Value::Compile(right_inner)) => Ok(Value::Compile((
            Spanned {
                span: left.span,
                inner: left_inner,
            },
            Spanned {
                span: right.span,
                inner: right_inner,
            },
        ))),
        (left_inner, right_inner) => {
            let left_inner = match left_inner {
                Value::Compile(left_inner) => to_other(Spanned {
                    span: left.span,
                    inner: left_inner,
                }),
                Value::Hardware(left_inner) => Ok(left_inner),
            };
            let right_inner = match right_inner {
                Value::Compile(right_inner) => to_other(Spanned {
                    span: right.span,
                    inner: right_inner,
                }),
                Value::Hardware(right_inner) => Ok(right_inner),
            };

            let left_inner = left_inner?;
            let right_inner = right_inner?;

            Ok(Value::Hardware((
                Spanned {
                    span: left.span,
                    inner: left_inner,
                },
                Spanned {
                    span: right.span,
                    inner: right_inner,
                },
            )))
        }
    }
}

// Proofs of the validness of the integer ranges can be found in `int_range_proofs.py`.
pub fn eval_binary_expression(
    diags: &Diagnostics,
    large: &mut IrLargeArena,
    expr_span: Span,
    op: Spanned<BinaryOp>,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
) -> Result<ValueWithImplications, ErrorGuaranteed> {
    let op_reason = TypeContainsReason::Operator(op.span);

    let check_both_int = |left, right| {
        let left = check_type_is_int(diags, op_reason, left);
        let right = check_type_is_int(diags, op_reason, right);
        Ok((left?, right?))
    };
    let eval_binary_bool = |large, left, right, op| eval_binary_bool(diags, large, op_reason, left, right, op);
    let eval_binary_int_compare =
        |large, left, right, op| eval_binary_int_compare(diags, large, op_reason, left, right, op);

    let result_simple: Value<_> = match op.inner {
        // (int, int)
        BinaryOp::Add => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value()), right.map_inner(|e| e.value()))?;
            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => Value::Compile(CompileValue::Int(left.inner + right.inner)),
                Value::Hardware((left, right)) => {
                    let range = ClosedIncRange {
                        start_inc: &left.inner.ty.start_inc + &right.inner.ty.start_inc,
                        end_inc: &left.inner.ty.end_inc + &right.inner.ty.end_inc,
                    };
                    Value::Hardware(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Add,
                        large,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Sub => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value()), right.map_inner(|e| e.value()))?;
            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => Value::Compile(CompileValue::Int(left.inner - right.inner)),
                Value::Hardware((left, right)) => {
                    let range = ClosedIncRange {
                        start_inc: &left.inner.ty.start_inc - &right.inner.ty.end_inc,
                        end_inc: &left.inner.ty.end_inc - &right.inner.ty.start_inc,
                    };
                    Value::Hardware(build_binary_int_arithmetic_op(
                        IrIntArithmeticOp::Sub,
                        large,
                        range,
                        left,
                        right,
                    ))
                }
            }
        }
        BinaryOp::Mul => {
            // TODO do we want to keep using multiplication as the "array repeat" syntax?
            //   if so, maybe allow tuples on the right side for multidimensional repeating
            let right = check_type_is_int(diags, op_reason, right.map_inner(|e| e.value()));
            match left.inner.ty() {
                Type::Array(left_ty_inner, left_len) => {
                    let right = right?;
                    let right_inner = match right.inner {
                        Value::Compile(right_inner) => right_inner,
                        Value::Hardware(_) => {
                            return Err(diags.report_simple(
                                "array repetition right hand side must be compile-time value",
                                right.span,
                                "got non-compile-time value here",
                            ));
                        }
                    };
                    let right_inner = BigUint::try_from(right_inner).map_err(|right_inner| {
                        diags.report_simple(
                            "array repetition right hand side cannot be negative",
                            right.span,
                            format!("got value `{}`", right_inner),
                        )
                    })?;
                    let right_inner = usize::try_from(right_inner).map_err(|right_inner| {
                        diags.report_simple(
                            "array repetition right hand side too large",
                            right.span,
                            format!("got value `{}`", right_inner),
                        )
                    })?;

                    match left.inner.value() {
                        Value::Compile(CompileValue::Array(left_inner)) => {
                            // do the repetition at compile-time
                            // TODO check for overflow (everywhere)
                            let mut result = Vec::with_capacity(left_inner.len() * right_inner);
                            for _ in 0..right_inner {
                                result.extend_from_slice(&left_inner);
                            }
                            Value::Compile(CompileValue::Array(result))
                        }
                        Value::Compile(_) => {
                            return Err(diags.report_internal_error(
                                left.span,
                                "compile-time value with type array is not actually an array",
                            ))
                        }
                        Value::Hardware(value) => {
                            // implement runtime repetition through spread array literal
                            let element = IrArrayLiteralElement::Spread(value.expr);
                            let elements = vec![element; right_inner];

                            let left_ty_inner_hw = left_ty_inner.as_hardware_type().unwrap();
                            let result_len = left_len * right_inner;
                            let result_expr =
                                IrExpressionLarge::ArrayLiteral(left_ty_inner_hw.as_ir(), result_len.clone(), elements);
                            Value::Hardware(HardwareValue {
                                ty: HardwareType::Array(Box::new(left_ty_inner_hw.clone()), result_len),
                                domain: value.domain,
                                expr: large.push_expr(result_expr),
                            })
                        }
                    }
                }
                Type::Int(_) => {
                    let left = check_type_is_int(diags, op_reason, left.map_inner(|e| e.value()))
                        .expect("int, already checked");
                    let right = right?;
                    match pair_compile_int(left, right) {
                        Value::Compile((left, right)) => Value::Compile(CompileValue::Int(left.inner * right.inner)),
                        Value::Hardware((left, right)) => {
                            // calculate valid range
                            let extremes = [
                                &left.inner.ty.start_inc * &right.inner.ty.start_inc,
                                &left.inner.ty.start_inc * &right.inner.ty.end_inc,
                                &left.inner.ty.end_inc * &right.inner.ty.start_inc,
                                &left.inner.ty.end_inc * &right.inner.ty.end_inc,
                            ];
                            let range = ClosedIncRange {
                                start_inc: extremes.iter().min().unwrap().clone(),
                                end_inc: extremes.iter().max().unwrap().clone(),
                            };
                            let result =
                                build_binary_int_arithmetic_op(IrIntArithmeticOp::Mul, large, range, left, right);
                            Value::Hardware(result)
                        }
                    }
                }
                _ => {
                    return Err(diags.report_simple(
                        "left hand side of multiplication must be an array or an integer",
                        left.span,
                        format!("got value with type `{}`", left.inner.ty().to_diagnostic_string()),
                    ))
                }
            }
        }
        // (int, non-zero int)
        BinaryOp::Div => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value()), right.map_inner(|e| e.value()))?;

            // check nonzero
            if right.inner.range().contains(&&BigInt::ZERO) {
                let diag = Diagnostic::new("division by zero is not allowed")
                    .add_error(
                        right.span,
                        format!(
                            "right hand side has range `{}` which contains zero",
                            right.inner.range()
                        ),
                    )
                    .add_info(op.span, "for operator here")
                    .finish();
                return Err(diags.report(diag));
            }
            let right_positive = right.inner.range().start_inc > &BigInt::ZERO;

            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => {
                    let result = left.inner.div_floor(&right.inner).unwrap();
                    Value::Compile(CompileValue::Int(result))
                }
                Value::Hardware((left, right)) => {
                    let a_min = &left.inner.ty.start_inc;
                    let a_max = &left.inner.ty.end_inc;
                    let b_min = &right.inner.ty.start_inc;
                    let b_max = &right.inner.ty.end_inc;
                    let range = if right_positive {
                        ClosedIncRange {
                            start_inc: min(a_min.div_floor(b_max).unwrap(), a_min.div_floor(b_min).unwrap()),
                            end_inc: max(a_max.div_floor(b_max).unwrap(), a_max.div_floor(b_min).unwrap()),
                        }
                    } else {
                        ClosedIncRange {
                            start_inc: min(a_max.div_floor(b_max).unwrap(), a_max.div_floor(b_min).unwrap()),
                            end_inc: max(a_min.div_floor(b_max).unwrap(), a_min.div_floor(b_min).unwrap()),
                        }
                    };

                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Div, large, range, left, right);
                    Value::Hardware(result)
                }
            }
        }
        BinaryOp::Mod => {
            let (left, right) = check_both_int(left.map_inner(|e| e.value()), right.map_inner(|e| e.value()))?;

            // check nonzero
            if right.inner.range().contains(&&BigInt::ZERO) {
                let diag = Diagnostic::new("modulo by zero is not allowed")
                    .add_error(
                        right.span,
                        format!(
                            "right hand side has range `{}` which contains zero",
                            right.inner.range()
                        ),
                    )
                    .add_info(op.span, "for operator here")
                    .finish();
                return Err(diags.report(diag));
            }
            let right_positive = right.inner.range().start_inc > &BigInt::ZERO;

            match pair_compile_int(left, right) {
                Value::Compile((left, right)) => {
                    let result = left.inner.mod_floor(&right.inner).unwrap();
                    Value::Compile(CompileValue::Int(result))
                }
                Value::Hardware((left, right)) => {
                    let range = if right_positive {
                        ClosedIncRange {
                            start_inc: BigInt::ZERO,
                            end_inc: &right.inner.ty.end_inc - 1,
                        }
                    } else {
                        ClosedIncRange {
                            start_inc: &right.inner.ty.start_inc + 1,
                            end_inc: BigInt::ZERO,
                        }
                    };

                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Mod, large, range, left, right);
                    Value::Hardware(result)
                }
            }
        }
        // (nonzero int, non-negative int) or (non-negative int, positive int)
        BinaryOp::Pow => {
            let (base, exp) = check_both_int(left.map_inner(|e| e.value()), right.map_inner(|e| e.value()))?;

            let zero = BigInt::ZERO;
            let base_range = base.inner.range();
            let exp_range = exp.inner.range();

            // check exp >= 0
            if exp_range.start_inc < &zero {
                let diag = Diagnostic::new("invalid power operation")
                    .add_error(expr_span, "exponent must be non-negative")
                    .add_info(exp.span, format!("exponent range is `{}`", exp_range))
                    .finish();
                return Err(diags.report(diag));
            }

            // check not 0 ** 0
            if base_range.contains(&&zero) && exp_range.contains(&&zero) {
                let diag = Diagnostic::new("invalid power operation `0 ** 0`")
                    .add_error(expr_span, "base and exponent can both be zero")
                    .add_info(base.span, format!("base range is `{}`", base_range))
                    .add_info(exp.span, format!("exponent range is `{}`", exp_range))
                    .finish();
                return Err(diags.report(diag));
            }

            match pair_compile_int(base, exp) {
                Value::Compile((base, exp)) => {
                    let exp = BigUint::try_from(exp.inner)
                        .map_err(|_| diags.report_internal_error(exp.span, "got negative exp"))?;

                    let result = base.inner.pow(&exp);
                    Value::Compile(CompileValue::Int(result))
                }
                Value::Hardware((base, exp)) => {
                    let exp_start_inc = BigUint::try_from(&exp.inner.ty.start_inc)
                        .map_err(|_| diags.report_internal_error(exp.span, "got negative exp start"))?;
                    let exp_end_inc = BigUint::try_from(&exp.inner.ty.end_inc)
                        .map_err(|_| diags.report_internal_error(exp.span, "got negative exp end"))?;

                    let mut result_min = min(
                        base.inner.ty.start_inc.clone().pow(&exp_start_inc),
                        base.inner.ty.start_inc.clone().pow(&exp_end_inc),
                    );
                    let mut result_max = max(
                        base.inner.ty.start_inc.clone().pow(&exp_end_inc),
                        base.inner.ty.end_inc.clone().pow(&exp_end_inc),
                    );

                    // If base is negative, even/odd powers can cause extremes.
                    // To guard this, try the next highest exponent too if it exists.
                    if exp_end_inc > BigUint::ZERO {
                        let end_exp_sub_one = BigUint::try_from(exp_end_inc.sub(&BigUint::ONE)).unwrap();
                        result_min = min(result_min, base.inner.ty.start_inc.clone().pow(&end_exp_sub_one));
                        result_max = max(result_max, base.inner.ty.start_inc.clone().pow(&end_exp_sub_one));
                    }

                    let range = ClosedIncRange {
                        start_inc: result_min,
                        end_inc: result_max,
                    };
                    let result = build_binary_int_arithmetic_op(IrIntArithmeticOp::Pow, large, range, base, exp);
                    Value::Hardware(result)
                }
            }
        }
        // (bool, bool)
        // TODO these should short-circuit, so delay evaluation of right
        BinaryOp::BoolAnd => return eval_binary_bool(large, left, right, IrBoolBinaryOp::And),
        BinaryOp::BoolOr => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Or),
        BinaryOp::BoolXor => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Xor),
        // (T, T)
        // TODO expand eq/neq to bools/tuples/structs/enums, for the latter only if the type is the same
        BinaryOp::CmpEq => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Eq),
        BinaryOp::CmpNeq => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Neq),
        BinaryOp::CmpLt => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Lt),
        BinaryOp::CmpLte => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Lte),
        BinaryOp::CmpGt => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Gt),
        BinaryOp::CmpGte => return eval_binary_int_compare(large, left, right, IrIntCompareOp::Gte),
        // (int, range)
        // TODO share code with match "in" pattern
        BinaryOp::In => return Err(diags.report_todo(expr_span, "binary op In")),
        // (bool, bool)
        // TODO support boolean arrays
        BinaryOp::BitAnd => return eval_binary_bool(large, left, right, IrBoolBinaryOp::And),
        BinaryOp::BitOr => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Or),
        BinaryOp::BitXor => return eval_binary_bool(large, left, right, IrBoolBinaryOp::Xor),
        // TODO (boolean array, non-negative int) and maybe (non-negative int, non-negative int),
        //   and maybe even negative shift amounts?
        BinaryOp::Shl => return Err(diags.report_todo(expr_span, "binary op Shl")),
        BinaryOp::Shr => return Err(diags.report_todo(expr_span, "binary op Shr")),
    };

    Ok(ValueWithImplications::simple(result_simple))
}

fn eval_binary_bool(
    diags: &Diagnostics,
    large: &mut IrLargeArena,
    op_reason: TypeContainsReason,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
    op: IrBoolBinaryOp,
) -> Result<ValueWithImplications, ErrorGuaranteed> {
    fn build_bool_gate(
        f: impl Fn(bool) -> bool,
        large: &mut IrLargeArena,
        inner_eval: HardwareValueWithImplications,
        inner_ir: HardwareValue<()>,
    ) -> ValueWithImplications {
        match (f(false), f(true)) {
            // constants
            (false, false) => ValueWithImplications::simple(Value::Compile(CompileValue::Bool(false))),
            (true, true) => ValueWithImplications::simple(Value::Compile(CompileValue::Bool(true))),
            // pass gate
            (false, true) => ValueWithImplications::Hardware(inner_eval),
            // not gate
            (true, false) => ValueWithImplications::Hardware(HardwareValueWithImplications {
                value: HardwareValue {
                    ty: HardwareType::Bool,
                    domain: inner_ir.domain,
                    expr: large.push_expr(IrExpressionLarge::BoolNot(inner_ir.expr)),
                },
                value_versioned: None,
                implications: inner_eval.implications.invert(),
            }),
        }
    }

    let left_value = check_type_is_bool(diags, op_reason, left.as_ref().map_inner(|e| e.value_cloned()));
    let right_value = check_type_is_bool(diags, op_reason, right.as_ref().map_inner(|e| e.value_cloned()));

    let left_value = left_value?;
    let right_value = right_value?;

    match (left_value.inner, right_value.inner) {
        // full compile-tim eval
        (Value::Compile(left), Value::Compile(right)) => {
            let result = CompileValue::Bool(op.eval(left, right));
            Ok(ValueWithImplications::simple(Value::Compile(result)))
        }
        // partial compile-time eval
        (Value::Compile(left_value), Value::Hardware(right_value)) => Ok(build_bool_gate(
            |b| op.eval(left_value, b),
            large,
            right.inner.unwrap_hardware(),
            right_value,
        )),
        (Value::Hardware(left_value), Value::Compile(right_value)) => Ok(build_bool_gate(
            |b| op.eval(b, right_value),
            large,
            left.inner.unwrap_hardware(),
            left_value,
        )),
        // full hardware
        (Value::Hardware(left_value), Value::Hardware(right_value)) => {
            let expr = HardwareValue {
                ty: HardwareType::Bool,
                domain: left_value.domain.join(right_value.domain),
                expr: large.push_expr(IrExpressionLarge::BoolBinary(op, left_value.expr, right_value.expr)),
            };

            let left_inner = left.inner.unwrap_hardware();
            let right_inner = right.inner.unwrap_hardware();

            let implications = match op {
                IrBoolBinaryOp::And => Implications {
                    if_true: vec_concat([left_inner.implications.if_true, right_inner.implications.if_true]),
                    if_false: vec![],
                },
                IrBoolBinaryOp::Or => Implications {
                    if_true: vec![],
                    if_false: vec_concat([left_inner.implications.if_false, right_inner.implications.if_false]),
                },
                IrBoolBinaryOp::Xor => Implications::default(),
            };

            Ok(ValueWithImplications::Hardware(HardwareValueWithImplications {
                value: expr,
                value_versioned: None,
                implications,
            }))
        }
    }
}

fn eval_binary_int_compare(
    diags: &Diagnostics,
    large: &mut IrLargeArena,
    op_reason: TypeContainsReason,
    left: Spanned<ValueWithImplications>,
    right: Spanned<ValueWithImplications>,
    op: IrIntCompareOp,
) -> Result<ValueWithImplications, ErrorGuaranteed> {
    let left_int = check_type_is_int(
        diags,
        op_reason,
        left.as_ref().map_inner(ValueWithImplications::value_cloned),
    );
    let right_int = check_type_is_int(
        diags,
        op_reason,
        right.as_ref().map_inner(ValueWithImplications::value_cloned),
    );
    let left_int = left_int?;
    let right_int = right_int?;

    // TODO spans are getting unnecessarily complicated, eg. why does this try to propagate spans?
    match pair_compile_int(left_int, right_int) {
        Value::Compile((left, right)) => {
            let result = op.eval(&left.inner, &right.inner);
            Ok(ValueWithImplications::simple(Value::Compile(CompileValue::Bool(
                result,
            ))))
        }
        Value::Hardware((left_int, right_int)) => {
            let lv = match left.inner {
                ValueWithImplications::Compile(_) => None,
                ValueWithImplications::Hardware(v) => v.value_versioned,
            };
            let rv = match right.inner {
                ValueWithImplications::Compile(_) => None,
                ValueWithImplications::Hardware(v) => v.value_versioned,
            };

            // TODO warning if the result is always true/false (depending on the ranges)
            //   or maybe just return a compile-time value again?
            // build implications
            let lr = left_int.inner.ty;
            let rr = right_int.inner.ty;
            let implications = match op {
                IrIntCompareOp::Lt => implications_lt(lv, lr, rv, rr),
                IrIntCompareOp::Lte => implications_lte(lv, lr, rv, rr),
                IrIntCompareOp::Gt => implications_lt(rv, rr, lv, lr),
                IrIntCompareOp::Gte => implications_lte(rv, rr, lv, lr),
                IrIntCompareOp::Eq => implications_eq(lv, lr, rv, rr),
                IrIntCompareOp::Neq => implications_eq(lv, lr, rv, rr).invert(),
            };

            // build the resulting expression
            let result = HardwareValue {
                ty: HardwareType::Bool,
                domain: left_int.inner.domain.join(right_int.inner.domain),
                expr: large.push_expr(IrExpressionLarge::IntCompare(
                    op,
                    left_int.inner.expr,
                    right_int.inner.expr,
                )),
            };
            Ok(Value::Hardware(HardwareValueWithImplications {
                value: result,
                value_versioned: None,
                implications,
            }))
        }
    }
}

fn build_binary_int_arithmetic_op(
    op: IrIntArithmeticOp,
    large: &mut IrLargeArena,
    range: ClosedIncRange<BigInt>,
    left: Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
    right: Spanned<HardwareValue<ClosedIncRange<BigInt>>>,
) -> HardwareValue {
    let result_expr = IrExpressionLarge::IntArithmetic(op, range.clone(), left.inner.expr, right.inner.expr);
    HardwareValue {
        ty: HardwareType::Int(range),
        domain: left.inner.domain.join(right.inner.domain),
        expr: large.push_expr(result_expr),
    }
}

fn implications_lt(
    left: Option<ValueVersioned>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersioned>,
    right_range: ClosedIncRange<BigInt>,
) -> Implications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new(left, ImplicationOp::Lt, right_range.end_inc));
        if_false.push(Implication::new(left, ImplicationOp::Gt, right_range.start_inc - 1));
    }
    if let Some(right) = right {
        if_true.push(Implication::new(right, ImplicationOp::Gt, left_range.start_inc));
        if_false.push(Implication::new(right, ImplicationOp::Lt, left_range.end_inc + 1));
    }

    Implications { if_true, if_false }
}

fn implications_lte(
    left: Option<ValueVersioned>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersioned>,
    right_range: ClosedIncRange<BigInt>,
) -> Implications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new(left, ImplicationOp::Lt, right_range.end_inc + 1));
        if_false.push(Implication::new(left, ImplicationOp::Gt, right_range.start_inc));
    }
    if let Some(right) = right {
        if_true.push(Implication::new(right, ImplicationOp::Gt, left_range.start_inc - 1));
        if_false.push(Implication::new(right, ImplicationOp::Lt, left_range.end_inc));
    }

    Implications { if_true, if_false }
}

fn implications_eq(
    left: Option<ValueVersioned>,
    left_range: ClosedIncRange<BigInt>,
    right: Option<ValueVersioned>,
    right_range: ClosedIncRange<BigInt>,
) -> Implications {
    let mut if_true = vec![];
    let mut if_false = vec![];

    if let Some(left) = left {
        if_true.push(Implication::new(left, ImplicationOp::Lt, &right_range.end_inc + 1));
        if_true.push(Implication::new(left, ImplicationOp::Gt, &right_range.start_inc - 1));

        if let Some(right) = right_range.as_single() {
            if_false.push(Implication::new(left, ImplicationOp::Neq, right.clone()));
        }
    }

    if let Some(right) = right {
        if_true.push(Implication::new(right, ImplicationOp::Lt, &left_range.end_inc + 1));
        if_true.push(Implication::new(right, ImplicationOp::Gt, &left_range.start_inc - 1));

        if let Some(left) = left_range.as_single() {
            if_false.push(Implication::new(right, ImplicationOp::Neq, left.clone()));
        }
    }

    Implications { if_true, if_false }
}

fn apply_implications<C: ExpressionContext>(
    ctx: &C,
    large: &mut IrLargeArena,
    versioned: Option<ValueVersioned>,
    expr_raw: HardwareValue,
) -> HardwareValueWithImplications {
    let versioned = match versioned {
        None => return HardwareValueWithImplications::simple(expr_raw),
        Some(versioned) => versioned,
    };

    let value_expr = if let HardwareType::Int(ty) = &expr_raw.ty {
        let mut range = ClosedIncRangeMulti::from_range(ty.clone());
        ctx.for_each_implication(versioned, |implication| {
            range.apply_implication(implication.op, &implication.right);
        });

        match range.to_range() {
            // TODO support never type or maybe specifically empty ranges
            // TODO or better, once implications discover there's a contradiction we can stop evaluating the block
            None => expr_raw,
            Some(range) => {
                if &range == ty {
                    expr_raw
                } else {
                    let expr_constr = IrExpressionLarge::ConstrainIntRange(range.clone(), expr_raw.expr);
                    HardwareValue {
                        ty: HardwareType::Int(range),
                        domain: expr_raw.domain,
                        expr: large.push_expr(expr_constr),
                    }
                }
            }
        }
    } else {
        expr_raw
    };

    HardwareValueWithImplications {
        value: value_expr,
        value_versioned: Some(versioned),
        implications: Implications::default(),
    }
}

fn array_literal_combine_values(
    diags: &Diagnostics,
    large: &mut IrLargeArena,
    expr_span: Span,
    expected_ty_inner: &Type,
    values: Vec<ArrayLiteralElement<Spanned<Value>>>,
) -> Result<Value, ErrorGuaranteed> {
    let first_non_compile_span = values
        .iter()
        .find(|v| !matches!(v.value().inner, Value::Compile(_)))
        .map(|v| v.span());
    if let Some(first_non_compile_span) = first_non_compile_span {
        // at least one non-compile, turn everything into IR
        let expected_ty_inner = match expected_ty_inner {
            Type::Any => {
                // infer type based on elements
                let mut ty_joined = Type::Undefined;
                for value in &values {
                    let value_ty = match value {
                        ArrayLiteralElement::Single(value) => value.inner.ty(),
                        ArrayLiteralElement::Spread(_, values) => match values.inner.ty() {
                            Type::Array(ty, _) => *ty,
                            _ => Type::Undefined,
                        },
                    };
                    ty_joined = ty_joined.union(&value_ty, false);
                }
                ty_joined
            }
            _ => expected_ty_inner.clone(),
        };

        let expected_ty_inner_hw = expected_ty_inner.as_hardware_type().map_err(|_| {
            // TODO clarify that inferred type comes from outside, not the expression itself
            let message = format!(
                "hardware array literal has inferred inner type `{}` which is not representable in hardware",
                expected_ty_inner.to_diagnostic_string()
            );
            let diag = Diagnostic::new("hardware array type needs to be representable in hardware")
                .add_error(expr_span, message)
                .add_info(first_non_compile_span, "necessary because this array element is not a compile-time value, which forces the entire array to be hardware")
                .finish();
            diags.report(diag)
        })?;

        let mut result_domain = ValueDomain::CompileTime;
        let mut result_exprs = vec![];
        let mut result_len = BigUint::ZERO;

        for elem in values {
            let (elem_ir, domain, elem_len) = match elem {
                ArrayLiteralElement::Single(elem_inner) => {
                    let value_ir =
                        elem_inner
                            .inner
                            .as_hardware_value(diags, large, elem_inner.span, &expected_ty_inner_hw)?;

                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(expr_span),
                        &expected_ty_inner,
                        Spanned {
                            span: elem_inner.span,
                            inner: &Value::Hardware(value_ir.clone()),
                        },
                        true,
                        true,
                    )?;

                    (
                        IrArrayLiteralElement::Single(value_ir.expr),
                        value_ir.domain,
                        BigUint::ONE,
                    )
                }
                ArrayLiteralElement::Spread(_, elem_inner) => {
                    let value_ir =
                        elem_inner
                            .inner
                            .as_hardware_value(diags, large, elem_inner.span, &expected_ty_inner_hw)?;

                    let len = match value_ir.ty() {
                        Type::Array(_, len) => len,
                        _ => BigUint::ZERO,
                    };
                    check_type_contains_value(
                        diags,
                        TypeContainsReason::Operator(expr_span),
                        &Type::Array(Box::new(expected_ty_inner.clone()), len.clone()),
                        Spanned {
                            span: elem_inner.span,
                            inner: &Value::Hardware(value_ir.clone()),
                        },
                        true,
                        true,
                    )?;

                    (IrArrayLiteralElement::Spread(value_ir.expr), value_ir.domain, len)
                }
            };

            result_domain = result_domain.join(domain);
            result_exprs.push(elem_ir);
            result_len += elem_len;
        }

        let result_expr =
            IrExpressionLarge::ArrayLiteral(expected_ty_inner_hw.as_ir(), result_len.clone(), result_exprs);
        Ok(Value::Hardware(HardwareValue {
            ty: HardwareType::Array(Box::new(expected_ty_inner_hw), result_len),
            domain: result_domain,
            expr: large.push_expr(result_expr),
        }))
    } else {
        // all compile, create compile value
        let mut result = vec![];
        for elem in values {
            match elem {
                ArrayLiteralElement::Single(elem_inner) => {
                    let elem_inner = unwrap_match!(elem_inner.inner, Value::Compile(v) => v);
                    result.push(elem_inner);
                }
                ArrayLiteralElement::Spread(span_spread, elem_inner) => {
                    let elem_inner = unwrap_match!(elem_inner.inner, Value::Compile(v) => v);
                    let elem_inner_array = match elem_inner {
                        CompileValue::Array(elem_inner) => elem_inner,
                        _ => {
                            return Err(diags.report_todo(
                                span_spread,
                                "compile-time spread only works for fully known arrays for now",
                            ))
                        }
                    };
                    result.extend(elem_inner_array)
                }
            }
        }
        Ok(Value::Compile(CompileValue::Array(result)))
    }
}
