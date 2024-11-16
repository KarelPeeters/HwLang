use crate::data::compiled::GenericParameter;
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::checking::{DomainUserControlled, TypeMismatch};
use crate::front::common::{ContextDomain, ExpressionContext, ExpressionEval, GenericContainer, GenericMap, ScopeValue, ScopedEntry, TypeOrScopedValue, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::solver::{Solver, SolverInt, UnknownOrFalse};
use crate::front::types::{GenericArguments, GenericParameters, MaybeConstructor, Type};
use crate::front::value::{ArrayAccessIndex, BoundedRangeInfo, DomainSignal, RangeInfo, Value, ValueAccess, ValueContent, ValueDomain, ValueInt};
use crate::syntax::ast;
use crate::syntax::ast::{ArrayLiteralElement, BinaryOp, DomainKind, Expression, ExpressionKind, IntPattern, RangeLiteral, Spanned, SyncDomain, UnaryOp};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::{option_pair, result_pair};
use annotate_snippets::Level;
use itertools::{zip_eq, Itertools};
use num_bigint::BigInt;
use num_traits::One;
use std::cmp::min;

impl CompileState<'_, '_> {
    pub fn eval_expression(
        &mut self,
        ctx: &ExpressionContext,
        solver: &mut Solver,
        expr: &Expression,
    ) -> ExpressionEval {
        let diags = self.diags;

        match expr.inner {
            ExpressionKind::Dummy =>
                ExpressionEval::Error(diags.report_todo(expr.span, "dummy expression")),
            ExpressionKind::Any =>
                ExpressionEval::Immediate(TypeOrValue::Type(Type::Any)),
            ExpressionKind::Wrapped(ref inner) => {
                // passthrough, except that writing is disabled for values
                let inner = self.eval_expression(ctx, solver, inner);
                match inner {
                    ExpressionEval::Immediate(v) => {
                        match v {
                            TypeOrValue::Type(t) => ExpressionEval::Immediate(TypeOrValue::Type(t)),
                            TypeOrValue::Value(v) => ExpressionEval::Immediate(TypeOrValue::Value(Value {
                                content: v.content,
                                ty_default: v.ty_default,
                                domain: v.domain,
                                access: ValueAccess::ReadOnly,
                                origin: v.origin,
                            })),
                            TypeOrValue::Error(e) => ExpressionEval::Immediate(TypeOrValue::Error(e)),
                        }
                    }
                    ExpressionEval::Constructor(c) => ExpressionEval::Constructor(c),
                    ExpressionEval::Error(e) => ExpressionEval::Error(e),
                }
            }
            ExpressionKind::Id(ref id) => {
                // lookup id in scope
                let entry = match self.compiled[ctx.scope].find(&self.compiled.scopes, diags, id, Visibility::Private) {
                    Err(e) => return ExpressionEval::Error(e),
                    Ok(entry) => entry,
                };

                // map potentially indirect to direct
                let direct = match entry.value {
                    &ScopedEntry::Item(item) => self.resolve_item_signature(item).clone(),
                    ScopedEntry::Direct(entry) => entry.clone(),
                };

                // map scoped to actual
                //   this is passthrough for most things, except for values, where we need to apply something equivalent
                //   to single-static-assignment conversion so variables interact properly with the solver
                match direct {
                    MaybeConstructor::Immediate(imm) => {
                        let imm = match imm {
                            TypeOrScopedValue::Type(ty) => TypeOrValue::Type(ty),
                            TypeOrScopedValue::Value(v) => {
                                let v = match v {
                                    // TODO for now just always return a unique variable, later do SSA-style numbering
                                    ScopeValue::Variable(_) => todo!(),
                                    ScopeValue::Value(v) => v,
                                };
                                TypeOrValue::Value(v)
                            },
                            TypeOrScopedValue::Error(e) => TypeOrValue::Error(e),
                        };
                        ExpressionEval::Immediate(imm)
                    }
                    MaybeConstructor::Constructor(c) => ExpressionEval::Constructor(c),
                    MaybeConstructor::Error(e) => ExpressionEval::Error(e)
                }
            }
            ExpressionKind::TypeFunc(_, _) =>
                ExpressionEval::Error(diags.report_todo(expr.span, "type func expression")),
            ExpressionKind::IntPattern(ref pattern) => {
                match pattern {
                    IntPattern::Hex(_) =>
                        ExpressionEval::Error(diags.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        ExpressionEval::Error(diags.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        let value_int = str_clean.parse::<BigInt>().unwrap();

                        let value_solver = solver.known_int(value_int);
                        let value = Value {
                            content: ValueContent::Int(value_solver),
                            ty_default: Type::Integer(RangeInfo {
                                start_inc: Some(value_solver),
                                end_inc: Some(value_solver),
                            }),
                            domain: ValueDomain::CompileTime,
                            access: ValueAccess::No,
                            origin: expr.span,
                        };

                        ExpressionEval::Immediate(TypeOrValue::Value(value))
                    }
                }
            }
            ExpressionKind::BoolLiteral(b) => {
                let value = Value {
                    content: ValueContent::Bool(solver.known_bool(b)),
                    ty_default: Type::Boolean,
                    domain: ValueDomain::CompileTime,
                    access: ValueAccess::No,
                    origin: expr.span,
                };
                ExpressionEval::Immediate(TypeOrValue::Value(value))
            }
            ExpressionKind::StringLiteral(_) => {
                let value = Value {
                    content: ValueContent::Opaque(solver.new_opaque()),
                    ty_default: Type::String,
                    domain: ValueDomain::CompileTime,
                    access: ValueAccess::No,
                    origin: expr.span,
                };
                ExpressionEval::Immediate(TypeOrValue::Value(value))
            }
            ExpressionKind::ArrayLiteral(ref elements) => {
                let mut inner_ty = None;
                let mut total_len = Ok(vec![]);
                let mut domain = ValueDomain::CompileTime;

                for &ArrayLiteralElement { spread, ref value } in elements {
                    let value_eval = self.eval_expression_as_value(ctx, solver, value);

                    // typecheck the value and get its length
                    let (inner_ty_next, element_len) = match spread {
                        // single value
                        None => {
                            // infer or check type, using the type checking variant that can better use the solver
                            let inner_ty_next = match inner_ty {
                                None => value_eval.ty_default,
                                Some(inner_ty) => {
                                    match self.require_type_contains_value(None, value.span, &inner_ty, &value_eval) {
                                        Ok(()) => inner_ty,
                                        Err(e) => Type::Error(e),
                                    }
                                }
                            };

                            (inner_ty_next, Ok(solver.one()))
                        }
                        // spread of array
                        Some(spread_span) => {
                            // check that the type is indeed an array
                            match value_eval.ty_default {
                                Type::Array(value_array_inner_ty, value_array_len) => {
                                    // infer or check type, using basic subtype type checking
                                    let inner_ty_next = match inner_ty {
                                        None => *value_array_inner_ty,
                                        Some(inner_ty) => {
                                            match self.require_type_match(None, &inner_ty, value.span, &value_array_inner_ty, true) {
                                                Ok(Ok(())) => inner_ty,
                                                Ok(Err(e)) => Type::Error(e),
                                                Err(TypeMismatch) => {
                                                    let actual_str = self.compiled.type_to_readable_str(self.source, self.parsed, &value_array_inner_ty);
                                                    let expected_str = self.compiled.type_to_readable_str(self.source, self.parsed, &inner_ty);

                                                    let diag = Diagnostic::new("type mismatch: elements of spread array are not subtypes of inferred array type")
                                                        .add_error(spread_span.join(value.span), format!("actual element type {}", actual_str))
                                                        .add_info(expr.span, format!("inferred type of array literal elements: `{}`", expected_str))
                                                        .finish();
                                                    Type::Error(diags.report(diag))
                                                }
                                            }
                                        }
                                    };

                                    (inner_ty_next, Ok(value_array_len))
                                }
                                Type::Error(e) => (Type::Error(e), Err(e)),
                                _ => {
                                    let diag = Diagnostic::new("spread operator requires array operand")
                                        .add_error(spread_span, "for this spread operator")
                                        .add_info(value.span, format!("expected array, actual type {}", self.compiled.type_to_readable_str(self.source, self.parsed, &value_eval.ty_default)))
                                        .finish();
                                    let e = diags.report(diag);
                                    (Type::Error(e), Err(e))
                                }
                            }
                        }
                    };

                    // set next inner type
                    inner_ty = Some(inner_ty_next);

                    // add length
                    // TODO the solver should be able to add errors itself already
                    match (&mut total_len, element_len) {
                        (Err(_), _) => {}
                        (_, Err(e)) => total_len = Err(e),
                        (Ok(total_len), Ok(element_len)) => total_len.push(element_len),
                    };

                    // combine domains
                    domain = self.merge_domains(expr.span, &domain, &value_eval.domain);
                }

                let inner_ty = match inner_ty {
                    // TODO we need at least _some_ forward type inference to implement this
                    None => Type::Error(diags.report_todo(expr.span, "empty array literal expression")),
                    Some(inner_ty) => inner_ty,
                };

                let total_len = match total_len {
                    Ok(total_len) => solver.sum(total_len),
                    Err(e) => solver.error_int(e),
                };

                let result_ty = Type::Array(Box::new(inner_ty), total_len);
                let result_value = Value {
                    content: ValueContent::Opaque(solver.new_opaque()),
                    ty_default: result_ty,
                    domain,
                    access: ValueAccess::No,
                    origin: expr.span,
                };
                ExpressionEval::Immediate(TypeOrValue::Value(result_value))
            }
            ExpressionKind::TupleLiteral(_) =>
                ExpressionEval::Error(diags.report_todo(expr.span, "tuple literal expression")),
            ExpressionKind::StructLiteral(_) =>
                ExpressionEval::Error(diags.report_todo(expr.span, "struct literal expression")),
            ExpressionKind::RangeLiteral(ref range) => {
                let &RangeLiteral { end_inclusive, start: ref start_inc, end: ref end_raw } = range;

                // double check grammar
                if end_inclusive && end_raw.is_none() {
                    diags.report_internal_error(expr.span, "range literal with inclusive end but no end value");
                }

                // evaluate bounds
                let start_inc = start_inc.as_ref()
                    .map(|start| self.eval_expression_as_value_int(ctx, solver, start, "range bound"));
                let end_raw = end_raw.as_ref()
                    .map(|end_raw| self.eval_expression_as_value_int(ctx, solver, end_raw, "range bound"));

                // turn end bound into inclusive
                let end_inc = end_raw.map(|end_raw| {
                    match end_inclusive {
                        true => end_raw,
                        false => ValueInt {
                            content: solver.add_const(end_raw.content, -1),
                            ty_default: end_raw.ty_default.map(|range| RangeInfo {
                                start_inc: range.start_inc.map(|x| solver.add_const(x, -1)),
                                end_inc: range.end_inc.map(|x| solver.add_const(x, -1)),
                            }),
                            domain: end_raw.domain,
                            access: end_raw.access,
                            origin: end_raw.origin, // TODO include "..=" in the span?
                        },
                    }
                });

                // combine domains
                let mut domain = ValueDomain::CompileTime;
                if let Some(start_inc) = &start_inc {
                    domain = self.merge_domains(expr.span, &domain, &start_inc.domain);
                }
                if let Some(end_inc) = &end_inc {
                    domain = self.merge_domains(expr.span, &domain, &end_inc.domain);
                }

                // check validness
                let range_info = match (start_inc, end_inc) {
                    (Some(start_inc), Some(end_inc)) => {
                        let cond = solver.compare_lte(start_inc.content, end_inc.content);
                        match solver.eval_bool_true(cond) {
                            Ok(Ok(())) => RangeInfo {
                                start_inc: Some(start_inc),
                                end_inc: Some(end_inc),
                            },
                            Ok(Err(UnknownOrFalse)) => {
                                let e = self.diags.report_simple("typecheck failed: could not prove that range is non-decreasing", expr.span, "for this range");
                                RangeInfo {
                                    start_inc: Some(ValueInt::error_int(solver, e, start_inc.origin)),
                                    end_inc: Some(ValueInt::error_int(solver, e, end_inc.origin)),
                                }
                            }
                            Err(e) => {
                                RangeInfo {
                                    start_inc: Some(ValueInt::error_int(solver, e, start_inc.origin)),
                                    end_inc: Some(ValueInt::error_int(solver, e, end_inc.origin)),
                                }
                            }
                        }
                    }
                    (start_inc, end_inc) => {
                        RangeInfo {
                            start_inc,
                            end_inc,
                        }
                    }
                };

                // construct final range
                let value = Value {
                    content: ValueContent::Range(range_info),
                    ty_default: Type::Range,
                    domain,
                    access: ValueAccess::No,
                    origin: expr.span,
                };
                ExpressionEval::Immediate(TypeOrValue::Value(value))
            }
            ExpressionKind::UnaryOp(op, ref inner) => {
                let result = match op {
                    UnaryOp::Negate => {
                        let inner = self.eval_expression_as_value_int(ctx, solver, inner, "unary minus operand");

                        Value {
                            // negate content
                            content: ValueContent::Int(solver.negate(inner.content)),
                            // negate and swap start and end
                            ty_default: inner.ty_default.map_or_else(
                                Type::Error,
                                |range| Type::Integer(RangeInfo {
                                    start_inc: range.end_inc.map(|v| solver.negate(v)),
                                    end_inc: range.start_inc.map(|v| solver.negate(v)),
                                }),
                            ),
                            domain: inner.domain,
                            access: ValueAccess::No,
                            origin: expr.span,
                        }
                    }
                    UnaryOp::Not => {
                        let inner = self.eval_expression_as_value(ctx, solver, inner);

                        let generate_type_error = || {
                            let inner_ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &inner.ty_default);
                            let title = format!("unary not only works for boolean or boolean array types, got `{}`", inner_ty_str);
                            Value::error(diags.report_simple(title, expr.span, "for this expression"), expr.span)
                        };

                        match &inner.ty_default {
                            Type::Boolean => {
                                let inner_bool = inner.content.unwrap_bool(expr.span, diags, solver);
                                Value {
                                    content: ValueContent::Bool(solver.not(inner_bool)),
                                    ty_default: Type::Boolean,
                                    domain: inner.domain,
                                    access: ValueAccess::No,
                                    origin: expr.span,
                                }
                            }
                            Type::Array(element_ty, len) => {
                                // TODO support multidimensional boolean arrays?
                                match &**element_ty {
                                    Type::Boolean => {
                                        Value {
                                            content: ValueContent::Opaque(solver.new_opaque()),
                                            ty_default: Type::Array(Box::new(Type::Boolean), len.clone()),
                                            domain: inner.domain,
                                            access: ValueAccess::No,
                                            origin: expr.span,
                                        }
                                    }
                                    _ => generate_type_error(),
                                }
                            }
                            _ => generate_type_error(),
                        }
                    }
                };

                ExpressionEval::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let result = match op {
                    BinaryOp::Add => {
                        let left = self.eval_expression_as_value_int(ctx, solver, left, "addition operand");
                        let right = self.eval_expression_as_value_int(ctx, solver, right, "addition operand");

                        let result = solver.add(left.content, right.content);

                        let ty = result_pair(left.ty_default, right.ty_default)
                            .map_or_else(Type::Error, |(left, right)| {
                                Type::Integer(RangeInfo {
                                    start_inc: option_pair(left.start_inc, right.start_inc)
                                        .map(|(left, right)| solver.add(left, right)),
                                    end_inc: option_pair(left.end_inc, right.end_inc)
                                        .map(|(left, right)| solver.add(left, right)),
                                })
                            });

                        Value {
                            content: ValueContent::Int(result),
                            ty_default: ty,
                            domain: self.merge_domains(expr.span, &left.domain, &right.domain),
                            access: ValueAccess::No,
                            origin: expr.span,
                        }
                    }
                    BinaryOp::Sub => todo!(),
                    BinaryOp::Mul => todo!(),
                    BinaryOp::Div => todo!(),
                    BinaryOp::Mod => todo!(),
                    BinaryOp::Pow => todo!(),
                    BinaryOp::CmpLt => todo!(),
                    BinaryOp::CmpLte => todo!(),
                    BinaryOp::CmpGt => todo!(),
                    BinaryOp::CmpGte => todo!(),

                    // needs arrays of bools with matching len
                    BinaryOp::BitAnd => todo!(),
                    BinaryOp::BitOr => todo!(),
                    BinaryOp::BitXor => todo!(),

                    // needs bools
                    BinaryOp::BoolAnd => todo!(),
                    BinaryOp::BoolOr => todo!(),
                    BinaryOp::BoolXor => todo!(),

                    // TODO
                    BinaryOp::Shl => todo!(),
                    BinaryOp::Shr => todo!(),

                    // needs matching types
                    BinaryOp::CmpEq => todo!(),
                    BinaryOp::CmpNeq => todo!(),

                    // needs (int, range)
                    BinaryOp::In => todo!(),
                };

                ExpressionEval::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::TernarySelect(_, _, _) =>
                ExpressionEval::Error(diags.report_todo(expr.span, "ternary select expression")),
            ExpressionKind::ArrayIndex(ref base, ref args) => {
                let base_span = base.span;
                let base = self.eval_expression_as_ty_or_value(ctx, solver, base);

                let args = args.map_inner(|arg| {
                    Spanned { span: arg.span, inner: self.eval_expression_as_value(ctx, solver, arg) }
                });
                let args_len = args.inner.len();

                match base {
                    TypeOrValue::Error(e) => ExpressionEval::Error(e),
                    TypeOrValue::Type(base_ty) => {
                        if args.inner.is_empty() {
                            ExpressionEval::Error(diags.report_simple(
                                "array type definition requires at least one dimension",
                                args.span,
                                "array index",
                            ))
                        } else {
                            // array dimensions are reversed, the innermost type is the final dimension
                            let result_ty = args.inner.into_iter().rev().fold(base_ty, |ty_acc, arg| {
                                let ast::Arg { span: _, name, value } = arg;
                                let value = self.expect_value_int(solver, value.inner, "array type dimension length");

                                let zero = solver.zero();
                                let cond = solver.compare_lte(zero, value.content);
                                let value_checked = match solver.eval_bool_true(cond) {
                                    Ok(Ok(())) => value.content,
                                    Err(e) => solver.error_int(e),
                                    Ok(Err(UnknownOrFalse)) => {
                                        let e = diags.report_simple(
                                            "array type dimension length must be non-negative",
                                            value.origin, "failed to prove non-negativity for this value",
                                        );
                                        solver.error_int(e)
                                    }
                                };

                                if let Some(name) = name {
                                    Type::Error(diags.report_todo(name.span, "named array dimensions"))
                                } else {
                                    Type::Array(Box::new(ty_acc), value_checked)
                                }
                            });

                            ExpressionEval::Immediate(TypeOrValue::Type(result_ty))
                        }
                    }
                    TypeOrValue::Value(base) => {
                        // check indices and get inner type
                        let mut indices = vec![];
                        let mut curr_ty = base.ty_default.clone();

                        for (arg_i, arg) in args.inner.into_iter().enumerate() {
                            let (next_ty, index) = self.eval_array_index(solver, expr.span, base_span, curr_ty, args.span, args_len, arg_i, arg);
                            curr_ty = next_ty;
                            indices.push(index);
                        };
                        let inner_ty = curr_ty;

                        // build result type
                        let result_ty = indices.iter().rev().fold(inner_ty, |acc, index| {
                            match index {
                                &ArrayAccessIndex::Error(e) => Type::Error(e),
                                ArrayAccessIndex::Single(_) => acc,
                                &ArrayAccessIndex::Range(BoundedRangeInfo { start_inc, end_inc }) => {
                                    let length_1 = solver.sub(end_inc, start_inc);
                                    let length = solver.add_const(length_1, 1);
                                    Type::Array(Box::new(acc), length)
                                }
                            }
                        });

                        // result
                        ExpressionEval::Immediate(TypeOrValue::Value(Value {
                            content: ValueContent::opaque_of_type(&result_ty, solver),
                            ty_default: result_ty,
                            domain: base.domain,
                            access: base.access,
                            origin: expr.span,
                        }))
                    }
                }
            }
            ExpressionKind::DotIdIndex(_, _) =>
                ExpressionEval::Error(diags.report_todo(expr.span, "dot id index expression")),
            ExpressionKind::DotIntIndex(_, _) =>
                ExpressionEval::Error(diags.report_todo(expr.span, "dot int index expression")),
            ExpressionKind::Call(ref target, ref args) => {
                let target_entry = self.eval_expression(ctx, solver, target);
                let args_entry = args.map_inner(|e| {
                    self.eval_expression_as_ty_or_value(ctx, solver, e)
                });

                match target_entry {
                    ExpressionEval::Constructor(constr) => {
                        match self.eval_constructor_call(&constr.parameters, &constr.inner, args_entry, true) {
                            Ok((v, _)) => MaybeConstructor::Immediate(v),
                            Err(e) => MaybeConstructor::Error(e),
                        }
                    }
                    ExpressionEval::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => {
                                let err = Diagnostic::new_simple("invalid call target", target.span, "invalid call target kind 'type'");
                                ExpressionEval::Error(diags.report(err))
                            }
                            TypeOrValue::Value(_) =>
                                ExpressionEval::Error(diags.report_todo(target.span, "value call target")),
                            TypeOrValue::Error(e)
                            => ExpressionEval::Error(e),
                        }
                    }
                    ExpressionEval::Error(e) => ExpressionEval::Error(e),
                }
            }
            ExpressionKind::Builtin(ref args) => {
                match self.eval_builtin_call(ctx, solver, expr.span, args) {
                    Ok(result) => MaybeConstructor::Immediate(result),
                    Err(e) => MaybeConstructor::Error(e),
                }
            }
        }
    }

    fn eval_array_index(
        &mut self,
        solver: &mut Solver,
        expr_span: Span,
        base_span: Span,
        base_ty: Type,
        args_span: Span,
        args_len: usize,
        arg_i: usize,
        arg: ast::Arg<Spanned<Value>>,
    ) -> (Type, ArrayAccessIndex<SolverInt>) {
        let diags = self.diags;

        let ast::Arg { span: _, name, value: index } = arg;

        if let Some(name) = name {
            let e = diags.report_todo(name.span, "named array dimensions");
            return (Type::Error(e), ArrayAccessIndex::Error(e));
        }

        match base_ty {
            Type::Error(e) => {
                (Type::Error(e), ArrayAccessIndex::Error(e))
            }
            Type::Array(inner, len) => {
                match index.inner.content {
                    ValueContent::Error(e) => (Type::Error(e), ArrayAccessIndex::Error(e)),
                    ValueContent::Int(index_raw) => {
                        let mut any_err = Ok(());

                        let zero = solver.zero();
                        let cond_non_neg = solver.compare_gte(index_raw, zero);
                        match solver.eval_bool_true(cond_non_neg) {
                            Ok(Ok(())) => {}
                            Ok(Err(UnknownOrFalse)) => {
                                any_err = Err(diags.report_simple("array index must be non-negative", index.span, "for this array index"))
                            }
                            Err(e) => {
                                any_err = Err(e)
                            }
                        }

                        let cond_lt_len = solver.compare_lt(index_raw, len);
                        match solver.eval_bool_true(cond_lt_len) {
                            Ok(Ok(())) => {}
                            Ok(Err(UnknownOrFalse)) => {
                                any_err = Err(diags.report_simple("array index out of bounds", index.span, "for this array index"))
                            }
                            Err(e) => {
                                any_err = Err(e)
                            }
                        }

                        let index_checked = match any_err {
                            Ok(()) => index_raw,
                            Err(e) => solver.error_int(e),
                        };

                        (*inner, ArrayAccessIndex::Single(index_checked))
                    }
                    ValueContent::Range(RangeInfo { start_inc, end_inc }) => {
                        // Notes:
                        // * the validness of the range was already checked when it was constructed
                        // * range edges can be equal to the length, contrary to single indices
                        let mut check_edge = |solver: &mut Solver, v: ValueInt, edge: &str| {
                            let mut result = Ok(v.content);

                            let zero = solver.zero();
                            let cond_gte_zero = solver.compare_lte(zero, v.content);
                            let cond_lte_len = solver.compare_lte(v.content, len);

                            match solver.eval_bool_true(cond_gte_zero) {
                                Ok(Ok(())) => {}
                                Ok(Err(UnknownOrFalse)) => {
                                    result = Err(diags.report_simple(
                                        format!("array slice range {edge} should be >= 0"),
                                        index.span,
                                        "failed to prove that this value is non-negative",
                                    ))
                                }
                                Err(e) => result = Err(e),
                            }
                            match solver.eval_bool_true(cond_lte_len) {
                                Ok(Ok(())) => {}
                                Ok(Err(UnknownOrFalse)) => {
                                    result = Err(diags.report_simple(
                                        format!("array slice range {edge} should be <= array length"),
                                        index.span,
                                        "failed to prove that this value is not larger than the length",
                                    ))
                                }
                                Err(e) => result = Err(e),
                            }

                            result.unwrap_or_else(|e| solver.error_int(e))
                        };

                        let start_inc_checked = match start_inc {
                            Some(start_inc) => check_edge(solver, start_inc, "start"),
                            None => solver.zero(),
                        };
                        let end_inc_checked = match end_inc {
                            Some(end_inc) => check_edge(solver, end_inc, "end"),
                            None => len,
                        };

                        (*inner, ArrayAccessIndex::Range(BoundedRangeInfo {
                            start_inc: start_inc_checked,
                            end_inc: end_inc_checked,
                        }))
                    }
                    _content => {
                        let diag = Diagnostic::new("type mismatch: expected integer or range type for array index")
                            .add_error(expr_span, "for this array indexing operation")
                            .add_info(index.span, format!("actual type {}", self.compiled.type_to_readable_str(self.source, self.parsed, &index.inner.ty_default)))
                            .finish();
                        let e = diags.report(diag);
                        (Type::Error(e), ArrayAccessIndex::Error(e))
                    }
                }
            }
            _ => {
                let diag = if arg_i == 0 {
                    Diagnostic::new("type mismatch: expected array type for array indexing operator")
                        .add_error(expr_span, "for this array indexing operation")
                        .add_info(base_span, format!("actual type {}", self.compiled.type_to_readable_str(self.source, self.parsed, &base_ty)))
                        .finish()
                } else {
                    Diagnostic::new(format!("type mismatch: expected array type with at least {} dimensions", args_len))
                        .add_error(expr_span, "for this array indexing operation")
                        .add_info(base_span, format!("actual type {}", self.compiled.type_to_readable_str(self.source, self.parsed, &base_ty)))
                        .add_info(args_span, format!("got {} indices", args_len))
                        .finish()
                };

                let e = diags.report(diag);
                (Type::Error(e), ArrayAccessIndex::Error(e))
            }
        }
    }

    pub fn eval_constructor_call<T: GenericContainer>(
        &mut self,
        parameters: &GenericParameters,
        inner: &T,
        args: ast::Args<TypeOrValue>,
        allow_positional: bool,
    ) -> Result<(T::Result, GenericArguments), ErrorGuaranteed> {
        let diags = self.diags;
        let mut any_err = None;

        // TODO allow different declaration and use orderings, be careful about interactions
        // TODO add span where the parameters are defined
        // check count match
        if parameters.vec.len() != args.inner.len() {
            let err = Diagnostic::new_simple(
                format!("constructor argument count mismatch, expected {}, got {}", parameters.vec.len(), args.inner.len()),
                args.span,
                "arguments here",
            );
            any_err = Some(diags.report(err));
        }
        let min_len = min(parameters.vec.len(), args.inner.len());

        // check kind and type match, and collect in replacement map
        let mut map = GenericMap::empty();
        let mut any_named = false;

        let mut ordered_args = vec![];

        for (&param, arg) in zip_eq(&parameters.vec[..min_len], args.inner.into_iter().take(min_len)) {
            let ast::Arg { span: arg_span, name: arg_name, value: arg_value, } = arg;

            // check positional allowed or name match
            match arg_name {
                None => {
                    if !allow_positional {
                        let err = Diagnostic::new_simple("positional arguments are not allowed here", arg_span, "positional argument");
                        any_err = Some(diags.report(err));
                    }
                    if any_named {
                        let err = Diagnostic::new_simple("positional argument is not allowed after named argument", arg_span, "positional argument");
                        any_err = Some(diags.report(err));
                    }
                }
                Some(arg_name) => {
                    any_named = true;

                    let param_id = match param {
                        GenericParameter::Type(param) => &self.compiled[param].defining_id,
                        GenericParameter::Value(param) => &self.compiled[param].defining_id,
                    };

                    if arg_name.string != param_id.string {
                        let err = Diagnostic::new("argument name mismatch")
                            .add_info(param_id.span, format!("expected `{}`, defined here", param_id.string))
                            .add_error(arg_span, format!("got `{}`", arg_name.string))
                            .footer(Level::Note, "different parameter and argument orderings are not yet supported")
                            .finish();
                        any_err = Some(diags.report(err));

                        // from now on generic replacement is broken, so we have to stop the loop
                        break;
                    }
                }
            }

            // immediately use the existing generic params to replace the current one
            // TODO call replace_generics instead?
            match param {
                GenericParameter::Type(param) => {
                    let arg_ty = arg_value.unwrap_ty(diags, arg_span);

                    // TODO use for bound-check (once we add type bounds)
                    // TODO apply generic map to info, certainly for the bounds
                    let _param_info = &self.compiled[param];
                    map.generic_ty.insert_first(param, arg_ty.clone());
                    ordered_args.push(TypeOrValue::Type(arg_ty));
                }
                GenericParameter::Value(param) => {
                    let arg_value = arg_value.unwrap_value(diags, arg_span);

                    let param_info = &self.compiled[param];
                    let ty_span = param_info.ty_span;
                    let ty_replaced = param_info.ty.clone()
                        .replace_generics(&mut self.compiled, &map);

                    match self.require_type_contains_value(Some(ty_span), arg_span, &ty_replaced, &arg_value) {
                        Ok(()) => {}
                        Err(e) => any_err = Some(e),
                    }
                    map.generic_value.insert_first(param, arg_value.clone());
                    ordered_args.push(TypeOrValue::Value(arg_value));
                }
            }
        }

        // only bail once all parameters have been checked
        if let Some(e) = any_err {
            return Err(e);
        }

        // at this point we know the generic arg count is right, and the order is correct
        assert_eq!(parameters.vec.len(), ordered_args.len());
        let generic_args = GenericArguments {
            vec: ordered_args,
        };

        // do the actual replacement
        let result = inner.replace_generics(&mut self.compiled, &map);

        Ok((result, generic_args))
    }

    pub fn eval_expression_as_ty_or_value(&mut self, ctx: &ExpressionContext, solver: &mut Solver, expr: &Expression) -> TypeOrValue {
        let entry = self.eval_expression(ctx, solver, expr);

        match entry {
            ExpressionEval::Immediate(entry) => entry,
            ExpressionEval::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type or value, got constructor", expr.span, "constructor");
                TypeOrValue::Error(self.diags.report(diag))
            }
            ExpressionEval::Error(e) => TypeOrValue::Error(e),
        }
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, solver: &mut Solver, expr: &Expression) -> Type {
        let ctx = ExpressionContext::constant(expr.span, scope);
        let entry = self.eval_expression(&ctx, solver, expr);

        match entry {
            // TODO unify these error strings somewhere
            // TODO maybe move back to central error collection place for easier unit testing?
            // TODO report span for the _reason_ why we expect one or the other
            ExpressionEval::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type, got constructor", expr.span, "constructor");
                Type::Error(self.diags.report(diag))
            }
            ExpressionEval::Immediate(entry) => entry.unwrap_ty(self.diags, expr.span),
            ExpressionEval::Error(e) => Type::Error(e),
        }
    }

    pub fn eval_expression_as_value_read(&mut self, ctx: &ExpressionContext, solver: &mut Solver, expr: &Expression) -> Value {
        let entry = self.eval_expression(ctx, solver, expr);

        // check expression is a value
        let value = match entry {
            ExpressionEval::Immediate(entry) => {
                entry.unwrap_value(self.diags, expr.span)
            }
            ExpressionEval::Constructor(_) => {
                let err = Diagnostic::new_simple("expected value, got constructor", expr.span, "constructor");
                Value::error(self.diags.report(err), expr.span)
            }
            ExpressionEval::Error(e) => Value::error(e, expr.span),
        };

        // check readable
        match value.access {
            ValueAccess::ReadOnly => {}
            // TODO make everything readable, then we can just remove this
            ValueAccess::WriteOnlyPort(_) => {
                todo!()
            }
            ValueAccess::WriteReadWire(_) => {}
            ValueAccess::WriteReadRegister(_) => {}
            ValueAccess::WriteReadVariable(_) => {}
            ValueAccess::Error(_) => {}
        }

        // check domain
        match &ctx.domain {
            ContextDomain::Specific { domain, error_hint } => {
                let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
                    domain.span,
                    domain.inner,
                    value.origin,
                    &value.domain,
                    DomainUserControlled::Source,
                    error_hint,
                );
            }
            ContextDomain::Passthrough => {}
        }

        value
    }

    pub fn eval_expression_as_value_int(
        &mut self,
        ctx: &ExpressionContext,
        solver: &mut Solver,
        expr: &Expression,
        reason: &str,
    ) -> ValueInt {
        let value = self.eval_expression_as_value(ctx, solver, expr);
        self.expect_value_int(solver, value, reason)
    }

    pub fn expect_value_int(&mut self, solver: &mut Solver, value: Value, reason: &str) -> ValueInt {
        let diags = self.diags;

        match value.ty_default {
            Type::Error(e) => ValueInt::error_int(solver, e, value.origin),
            Type::Integer(range) => {
                ValueInt {
                    content: value.content.unwrap_int(value.origin, diags, solver),
                    ty_default: Ok(range),
                    domain: value.domain,
                    access: value.access,
                    origin: value.origin,
                }
            }
            _ => {
                let ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &value.ty_default);
                let title = format!("expected integer value for {reason}");
                let diag = Diagnostic::new(title)
                    .add_error(value.origin, format!("actual type {}", ty_str))
                    .finish();
                let e = diags.report(diag);
                ValueInt::error_int(solver, e, value.origin)
            }
        }
    }

    pub fn eval_domain(&mut self, scope: Scope, solver: &mut Solver, domain: Spanned<&DomainKind<Box<Expression>>>) -> DomainKind<DomainSignal> {
        match domain.inner {
            DomainKind::Async =>
                DomainKind::Async,
            DomainKind::Sync(sync_domain) => {
                let sync = SyncDomain { clock: &*sync_domain.clock, reset: &*sync_domain.reset };
                let sync = Spanned { span: domain.span, inner: sync };
                DomainKind::Sync(self.eval_sync_domain(scope, solver, sync))
            }
        }
    }

    pub fn eval_expression_as_domain_signal(&mut self, ctx: &ExpressionContext, solver: &mut Solver, expr: &Expression) -> Value<DomainSignal, Type> {
        todo!()
    }

    pub fn eval_sync_domain(&mut self, scope: Scope, solver: &mut Solver, sync: Spanned<SyncDomain<&Expression>>) -> SyncDomain<DomainSignal> {
        let Spanned { span: _, inner: sync } = sync;
        let SyncDomain { clock, reset } = sync;

        let ctx = ExpressionContext::passthrough(scope);
        let clock_value_unchecked = self.eval_expression_as_domain_signal(&ctx, solver, clock);
        let reset_value_unchecked = self.eval_expression_as_domain_signal(&ctx, solver, reset);

        // check that clock is a clock
        // TODO also/instead check type?
        let clock_value = match &clock_value_unchecked.domain {
            ValueDomain::Clock => clock_value_unchecked,
            &ValueDomain::Error(e) => Value::error(e, clock.span),
            _ => {
                let sync_str = self.compiled.sync_kind_to_readable_string(self.source, self.parsed, &clock_value_unchecked.domain);
                let title = format!("clock must be a clock, has domain {}", sync_str);
                let e = self.diags.report_simple(title, clock.span, "clock value");
                Value::error(e, clock.span)
            }
        };

        // check that reset is an async bool
        // TODO allow sync reset
        // TODO require that async reset still comes out of sync in phase with the clock
        let reset_value_bool = match reset_value_unchecked.ty_default {
            Type::Error(e) => Value::error(e, reset.span),
            Type::Boolean => reset_value_unchecked,
            _ => {
                let ty_str = self.compiled.type_to_readable_str(self.source, self.parsed, &reset_value_unchecked.ty_default);
                let title = "type mismatch: reset must be a boolean".to_string();
                let label = format!("expected boolean, actual type `{}`", ty_str);
                let e = self.diags.report_simple(title, reset.span, label);
                Value::error(e, reset.span)
            }
        };

        let reset_value = match &reset_value_bool.domain {
            ValueDomain::Async => reset_value_bool,
            &ValueDomain::Error(e) => Value::error(e, reset.span),
            _ => {
                let title = format!("reset must be an async boolean, has domain {}", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, &reset_value_bool.domain));
                let e = self.diags.report_simple(title, reset.span, "reset value");
                Value::error(e, reset.span)
            }
        };

        SyncDomain {
            clock: clock_value.content,
            reset: reset_value.content,
        }
    }

    fn eval_builtin_call(
        &mut self,
        ctx: &ExpressionContext,
        solver: &mut Solver,
        expr_span: Span,
        args: &ast::Args,
    ) -> Result<TypeOrValue, ErrorGuaranteed> {
        let diags = self.diags;

        let args_span = args.span;

        for arg in &args.inner {
            if let Some(name) = &arg.name {
                return Err(diags.report_simple(
                    "named arguments are not allowed for __builtin calls",
                    name.span,
                    "named argument",
                ));
            }
        }

        if let (Some(first), Some(second)) = (args.inner.get(0), args.inner.get(1)) {
            if let (ExpressionKind::StringLiteral(first), ExpressionKind::StringLiteral(second)) = (&first.value.inner, &second.value.inner) {
                // TODO should we do type-checking of the "rest" arguments here,
                //   or can we trust the (stdlib) caller to do this right?
                let rest = args.inner[2..].iter()
                    .map(|a| self.eval_expression_as_ty_or_value(ctx, solver, &a.value))
                    .collect_vec();

                match (first.as_str(), second.as_str(), rest.as_slice()) {
                    ("type", "unchecked", &[]) =>
                        return Ok(TypeOrValue::Type(Type::Unchecked)),
                    ("value", "undefined", &[]) =>
                        return Ok(TypeOrValue::Value(Value {
                            content: ValueContent::Undefined,
                            ty_default: Type::Unchecked,
                            domain: ValueDomain::CompileTime,
                            access: ValueAccess::No,
                            origin: expr_span,
                        })),
                    ("type", "bool", &[]) =>
                        return Ok(TypeOrValue::Type(Type::Boolean)),
                    ("type", "int", &[]) => {
                        return Ok(TypeOrValue::Type(Type::Integer(RangeInfo {
                            start_inc: None,
                            end_inc: None,
                        })));
                    }
                    ("type", "int_range", [TypeOrValue::Value(range)]) => {
                        if let ValueContent::Range(range) = &range.content {
                            return Ok(TypeOrValue::Type(Type::Integer(range.as_ref().map_inner(|v| v.content))));
                        }
                    }
                    ("type", "Range", &[]) =>
                        return Ok(TypeOrValue::Type(Type::Range)),
                    ("type", "bits_inf", &[]) =>
                        return Ok(TypeOrValue::Type(Type::Bits(None))),
                    ("function", "print", [TypeOrValue::Value(value)]) => {
                        return Ok(TypeOrValue::Value(value.clone()));
                    }
                    // fallthrough into error
                    _ => {}
                }
            }
        }

        let err = Diagnostic::new("invalid arguments for __builtin call")
            .snippet(expr_span)
            .add_error(args_span, "invalid arguments")
            .finish()
            .finish()
            .into();
        Err(self.diags.report(err))
    }
}

impl TypeOrValue {
    pub fn unwrap_ty(self, diags: &Diagnostics, origin: Span) -> Type {
        match self {
            TypeOrValue::Type(ty) => ty,
            TypeOrValue::Value(_) => {
                let diag = Diagnostic::new_simple("expected type, got value", origin, "value");
                Type::Error(diags.report(diag))
            }
            TypeOrValue::Error(e) => Type::Error(e),
        }
    }

    pub fn unwrap_value(self, diags: &Diagnostics, origin: Span) -> Value {
        match self {
            TypeOrValue::Type(_) => {
                let diag = Diagnostic::new_simple("expected value, got type", origin, "type");
                Value::error(diags.report(diag), origin)
            }
            TypeOrValue::Value(value) => value,
            TypeOrValue::Error(e) => Value::error(e, origin),
        }
    }
}
