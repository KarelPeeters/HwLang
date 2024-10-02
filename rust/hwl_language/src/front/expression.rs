use crate::data::compiled::{GenericParameter, GenericTypeParameter, GenericValueParameter, VariableInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::common::{ExpressionContext, GenericContainer, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{Constructor, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::{Args, BinaryOp, Expression, ExpressionKind, ForExpression, IntPattern, RangeLiteral, SyncDomain, UnaryOp};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use itertools::zip_eq;
use num_bigint::BigInt;

impl CompileState<'_, '_> {
    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&mut self, ctx: &ExpressionContext, scope: Scope, expr: &Expression) -> ScopedEntryDirect {
        let diags = self.diags;

        match expr.inner {
            ExpressionKind::Dummy =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "dummy expression")),
            ExpressionKind::Any =>
                ScopedEntryDirect::Immediate(TypeOrValue::Type(Type::Any)),
            ExpressionKind::Wrapped(ref inner) =>
                self.eval_expression(ctx, scope, inner),
            ExpressionKind::Id(ref id) => {
                let entry = match self.compiled[scope].find(&self.compiled.scopes, diags, id, Visibility::Private) {
                    Err(e) => return ScopedEntryDirect::Error(e),
                    Ok(entry) => entry,
                };
                match entry.value {
                    &ScopedEntry::Item(item) => self.resolve_item_signature(item).clone(),
                    ScopedEntry::Direct(entry) => entry.clone(),
                }
            }
            ExpressionKind::TypeFunc(_, _) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "type func expression")),
            ExpressionKind::Block(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "block expression")),
            ExpressionKind::If(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "if expression")),
            ExpressionKind::Loop(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "loop expression")),
            ExpressionKind::While(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "while expression")),
            ExpressionKind::For(ref expr_for) => {
                let ForExpression { index, index_ty, iter, body } = expr_for;

                // define index variable
                let iter_span = iter.span;
                if let Some(index_ty) = index_ty {
                    diags.report_todo(index_ty.span, "for loop index type");
                }
                let iter = self.eval_expression_as_value(ctx, scope, iter);

                let _: Result<(), ErrorGuaranteed> = self.check_type_contains(expr.span, iter_span, &Type::Range, &iter);
                let (start, end, end_inclusive) = match &iter {
                    &Value::Error(e) => (Value::Error(e), Value::Error(e), false),
                    Value::Range(info) => {
                        // avoid duplicate error if both ends are missing
                        let report_unbounded = || diags.report_simple("for loop over unbounded range", iter_span, "range");

                        let &RangeInfo { ref start, ref end, end_inclusive } = info;
                        let (start, end) = match (start, end) {
                            (Some(start), Some(end)) => (start.as_ref().clone(), end.as_ref().clone()),
                            (Some(start), None) => (start.as_ref().clone(), Value::Error(report_unbounded())),
                            (None, Some(end)) => (Value::Error(report_unbounded()), end.as_ref().clone()),
                            (None, None) => {
                                let e = report_unbounded();
                                (Value::Error(e), Value::Error(e))
                            }
                        };
                        (start, end, end_inclusive)
                    }
                    _ => {
                        let e = diags.report_todo(iter_span, "for loop over non-range literal");
                        (Value::Error(e), Value::Error(e), false)
                    }
                };

                let index_range = Value::Range(RangeInfo {
                    start: Some(Box::new(start)),
                    end: Some(Box::new(end)),
                    end_inclusive,
                });
                let index_var = self.compiled.variables.push(VariableInfo {
                    defining_id: index.clone(),
                    ty: Type::Integer(IntegerTypeInfo { range: Box::new(index_range) }),
                    mutable: false,
                });

                let scope_index = self.compiled.scopes.new_child(scope, body.span, Visibility::Private);
                let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(index_var))));
                self.compiled[scope_index].maybe_declare(diags, index, entry, Visibility::Private);

                // typecheck body
                self.visit_block(ctx, scope_index, &body);

                // for loops always return unit
                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Unit))
            }
            ExpressionKind::Return(ref ret_value) => {
                let ret_value = ret_value.as_ref()
                    .map(|v| (v.span, self.eval_expression_as_value(ctx, scope, v)));

                match ctx {
                    &ExpressionContext::FunctionBody { ret_ty_span, ref ret_ty } => {
                        match (ret_value, ret_ty) {
                            (None, Type::Unit | Type::Error(_)) => {
                                // accept
                            }
                            (None, _) => {
                                diags.report_simple("missing return value", expr.span, "return");
                            },
                            (Some((ret_value_span, ret_value)), expected_ret_ty) => {
                                let _: Result<(), ErrorGuaranteed> = self.check_type_contains(ret_ty_span, ret_value_span, expected_ret_ty, &ret_value);
                            }
                        }
                    },
                    _ => {
                        diags.report_simple("return outside function body", expr.span, "return");
                    }
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Never))
            }
            ExpressionKind::Break(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "break expression")),
            ExpressionKind::Continue =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "continue expression")),
            ExpressionKind::IntPattern(ref pattern) => {
                match pattern {
                    IntPattern::Hex(_) =>
                        ScopedEntryDirect::Error(diags.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        ScopedEntryDirect::Error(diags.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        let value = str_clean.parse::<BigInt>().unwrap();
                        ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::InstConstant(value)))
                    }
                }
            }
            ExpressionKind::BoolLiteral(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "bool literal expression")),
            ExpressionKind::StringLiteral(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "string literal expression")),
            ExpressionKind::ArrayLiteral(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "array literal expression")),
            ExpressionKind::TupleLiteral(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "tuple literal expression")),
            ExpressionKind::StructLiteral(_) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "struct literal expression")),
            ExpressionKind::RangeLiteral(ref range) => {
                let &RangeLiteral { end_inclusive, ref start, ref end } = range;

                let mut map_point = |point: &Option<Box<Expression>>| {
                    point.as_ref()
                        .map(|p| Box::new(self.eval_expression_as_value(ctx, scope, p)))
                };

                let start = map_point(start);
                let end = map_point(end);

                if let (Some(start), Some(end)) = (&start, &end) {
                    let op = if end_inclusive { BinaryOp::CmpLt } else { BinaryOp::CmpLte };
                    match self.require_value_true_for_range(expr.span, &Value::Binary(op, start.clone(), end.clone())) {
                        Ok(()) => {}
                        Err(e) => return ScopedEntryDirect::Error(e),
                    }
                }

                let value = Value::Range(RangeInfo::new(start, end, end_inclusive));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(value))
            }
            ExpressionKind::UnaryOp(op, ref inner) => {
                let result = match op {
                    UnaryOp::Neg => {
                        Value::Binary(
                            BinaryOp::Sub,
                            Box::new(Value::InstConstant(BigInt::ZERO)),
                            Box::new(self.eval_expression_as_value(ctx, scope, inner)),
                        )
                    }
                    UnaryOp::Not =>
                        Value::UnaryNot(Box::new(self.eval_expression_as_value(ctx, scope, inner))),
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_as_value(ctx, scope, left);
                let right = self.eval_expression_as_value(ctx, scope, right);

                let result = Value::Binary(op, Box::new(left), Box::new(right));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::TernarySelect(_, _, _) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "ternary select expression")),
            ExpressionKind::ArrayIndex(_, _) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "array index expression")),
            ExpressionKind::DotIdIndex(_, _) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "dot id index expression")),
            ExpressionKind::DotIntIndex(_, _) =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "dot int index expression")),
            ExpressionKind::Call(ref target, ref args) => {
                let target_entry = self.eval_expression(ctx, scope, target);

                match target_entry {
                    ScopedEntryDirect::Constructor(constr) => {
                        // goal: replace parameters with the arguments of this call
                        let Constructor { inner, parameters } = constr;

                        // check count match
                        if parameters.vec.len() != args.inner.len() {
                            let err = Diagnostic::new_simple(
                                format!("constructor argument count mismatch, expected {}, got {}", parameters.vec.len(), args.inner.len()),
                                args.span,
                                format!("expected {} arguments, got {}", parameters.vec.len(), args.inner.len()),
                            );
                            return ScopedEntryDirect::Error(diags.report(err));
                        }

                        // check kind and type match, and collect in replacement map
                        let mut map_ty: IndexMap<GenericTypeParameter, Type> = IndexMap::new();
                        let mut map_value: IndexMap<GenericValueParameter, Value> = IndexMap::new();
                        let mut last_err = None;

                        for (&param, arg) in zip_eq(&parameters.vec, &args.inner) {
                            match param {
                                GenericParameter::Type(param) => {
                                    let arg_ty = self.eval_expression_as_ty(scope, arg);
                                    // TODO use for bound-check (once we add type bounds)
                                    let _param_info = &self.compiled[param];
                                    map_ty.insert_first(param, arg_ty)
                                }
                                GenericParameter::Value(param) => {
                                    let arg_value = self.eval_expression_as_value(ctx, scope, arg);

                                    // immediately use the existing generic params to replace the current one
                                    let param_info = &self.compiled[param];
                                    let ty_span = param_info.ty_span;
                                    let ty_replaced = param_info.ty.clone()
                                        .replace_generic_params(&mut self.compiled, &map_ty, &map_value);

                                    match self.check_type_contains(ty_span, arg.span, &ty_replaced, &arg_value) {
                                        Ok(()) => {}
                                        Err(e) => last_err = Some(e),
                                    }
                                    map_value.insert_first(param, arg_value)
                                }
                            }
                        }

                        // only bail once all parameters have been checked
                        if let Some(e) = last_err {
                            return ScopedEntryDirect::Error(e);
                        }

                        // do the actual replacement
                        let result = inner.replace_generic_params(&mut self.compiled, &map_ty, &map_value);
                        MaybeConstructor::Immediate(result)
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => {
                                let err = Diagnostic::new_simple("invalid call target", target.span, "invalid call target kind 'type'");
                                ScopedEntryDirect::Error(diags.report(err))
                            }
                            TypeOrValue::Value(_) =>
                                ScopedEntryDirect::Error(diags.report_todo(target.span, "value call target")),
                        }
                    }
                    ScopedEntryDirect::Error(e) => ScopedEntryDirect::Error(e),
                }
            }
            ExpressionKind::Builtin(ref args) => {
                match self.eval_builtin_call(scope, expr.span, args) {
                    Ok(result) => MaybeConstructor::Immediate(result),
                    Err(e) => MaybeConstructor::Error(e),
                }
            }
        }
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> Type {
        let ctx = &ExpressionContext::Type;
        let entry = self.eval_expression(ctx, scope, expr);

        match entry {
            // TODO unify these error strings somewhere
            // TODO maybe move back to central error collection place for easier unit testing?
            ScopedEntryDirect::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type, got constructor", expr.span, "constructor");
                Type::Error(self.diags.report(diag))
            }
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(ty) => ty,
                TypeOrValue::Value(_) => {
                    let diag = Diagnostic::new_simple("expected type, got value", expr.span, "value");
                    Type::Error(self.diags.report(diag))
                }
            }
            ScopedEntryDirect::Error(e) => Type::Error(e)
        }
    }

    pub fn eval_expression_as_value(&mut self, ctx: &ExpressionContext, scope: Scope, expr: &Expression) -> Value {
        let entry = self.eval_expression(ctx, scope, expr);
        match entry {
            ScopedEntryDirect::Constructor(_) => {
                let err = Diagnostic::new_simple("expected value, got constructor", expr.span, "constructor");
                Value::Error(self.diags.report(err))
            }
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(_) => {
                    let err = Diagnostic::new_simple("expected value, got type", expr.span, "type");
                    Value::Error(self.diags.report(err))
                }
                TypeOrValue::Value(value) => value,
            }
            ScopedEntryDirect::Error(e) => Value::Error(e),
        }
    }

    pub fn eval_sync_domain(&mut self, scope: Scope, domain: &SyncDomain<Box<Expression>>) -> SyncDomain<Value> {
        let ctx = &ExpressionContext::Type;
        let clock = self.eval_expression_as_value(ctx, scope, &domain.clock);
        let reset = self.eval_expression_as_value(ctx, scope, &domain.reset);

        // TODO check that clock is a clock
        // TODO check that reset is a boolean
        // TODO check that reset is either async or sync to the same clock

        SyncDomain {
            clock,
            reset,
        }
    }

    fn eval_builtin_call(
        &mut self,
        scope: Scope,
        expr_span: Span,
        args: &Args,
    ) -> Result<TypeOrValue, ErrorGuaranteed> {
        let get_arg_str = |i: usize| -> Option<&str> {
            args.inner.get(i).and_then(|e| match &e.inner {
                ExpressionKind::StringLiteral(s) => Some(s.as_str()),
                _ => None,
            })
        };

        if let (Some(first), Some(second)) = (get_arg_str(0), get_arg_str(1)) {
            let rest = &args.inner[2..];
            let ctx = &ExpressionContext::Type;

            match (first, second, rest) {
                ("type", "bool", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Boolean)),
                ("type", "int", &[]) => {
                    let range = Box::new(Value::Range(RangeInfo::unbounded()));
                    return Ok(TypeOrValue::Type(Type::Integer(IntegerTypeInfo { range })));
                }
                ("type", "int_range", [range]) => {
                    // TODO typecheck (range must be integer)
                    let range = Box::new(self.eval_expression_as_value(ctx, scope, range));
                    let ty_info = IntegerTypeInfo { range };
                    return Ok(TypeOrValue::Type(Type::Integer(ty_info)));
                }
                ("type", "Range", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Range)),
                ("type", "bits_inf", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Bits(None))),
                ("type", "bits", [bits]) => {
                    // TODO typecheck (bits must be non-negative integer)
                    let bits = self.eval_expression_as_value(ctx, scope, bits);
                    return Ok(TypeOrValue::Type(Type::Bits(Some(Box::new(bits)))));
                }
                ("type", "Array", [ty, len]) => {
                    // TODO typecheck: len must be uint
                    let ty = self.eval_expression_as_ty(scope, ty);
                    let len = self.eval_expression_as_value(ctx, scope, len);
                    return Ok(TypeOrValue::Type(Type::Array(Box::new(ty), Box::new(len))));
                }
                ("function", "print", [value]) => {
                    let _: Value = self.eval_expression_as_value(ctx, scope, value);
                    return Ok(TypeOrValue::Value(Value::Unit));
                }
                // fallthrough into error
                _ => {}
            }
        }

        let err = Diagnostic::new("invalid arguments for __builtin call")
            .snippet(expr_span)
            .add_error(args.span, "invalid arguments")
            .finish()
            .finish()
            .into();
        Err(self.diags.report(err))
    }
}

