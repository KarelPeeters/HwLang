use crate::data::compiled::{GenericParameter, GenericTypeParameter, GenericValueParameter};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::common::{GenericContainer, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::{CompileState, ResolveResult};
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{Constructor, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast::{Args, BinaryOp, Expression, ExpressionKind, IntPattern, RangeLiteral, SyncDomain, UnaryOp};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use indexmap::IndexMap;
use itertools::zip_eq;
use num_bigint::BigInt;

impl CompileState<'_, '_> {
    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    fn eval_expression(&mut self, scope: Scope, expr: &Expression) -> ResolveResult<ScopedEntryDirect> {
        let result = match expr.inner {
            ExpressionKind::Dummy =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "dummy expression")),
            ExpressionKind::Any =>
                ScopedEntryDirect::Immediate(TypeOrValue::Type(Type::Any)),
            ExpressionKind::Wrapped(ref inner) =>
                self.eval_expression(scope, inner)?,
            ExpressionKind::Id(ref id) => {
                let entry = match self.compiled[scope].find(&self.compiled.scopes, self.diag, id, Visibility::Private) {
                    Err(e) => return Ok(ScopedEntryDirect::Error(e)),
                    Ok(entry) => entry,
                };
                match entry.value {
                    &ScopedEntry::Item(item) => self.resolve_item_signature(item)?.clone(),
                    ScopedEntry::Direct(entry) => entry.clone(),
                }
            }
            ExpressionKind::TypeFunc(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "type func expression")),
            ExpressionKind::Block(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "block expression")),
            ExpressionKind::If(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "if expression")),
            ExpressionKind::Loop(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "loop expression")),
            ExpressionKind::While(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "while expression")),
            ExpressionKind::For(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "for expression")),
            ExpressionKind::Return(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "return expression")),
            ExpressionKind::Break(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "break expression")),
            ExpressionKind::Continue =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "continue expression")),
            ExpressionKind::IntPattern(ref pattern) => {
                match pattern {
                    IntPattern::Hex(_) =>
                        ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        let value = str_clean.parse::<BigInt>().unwrap();
                        ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Int(value)))
                    }
                }
            }
            ExpressionKind::BoolLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "bool literal expression")),
            ExpressionKind::StringLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "string literal expression")),
            ExpressionKind::ArrayLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "array literal expression")),
            ExpressionKind::TupleLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "tuple literal expression")),
            ExpressionKind::StructLiteral(_) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "struct literal expression")),
            ExpressionKind::RangeLiteral(ref range) => {
                let &RangeLiteral { end_inclusive, ref start, ref end } = range;

                let mut map_point = |point: &Option<Box<Expression>>| -> ResolveResult<_> {
                    match point {
                        None => Ok(None),
                        Some(point) => Ok(Some(Box::new(self.eval_expression_as_value(scope, point)?))),
                    }
                };

                let start = map_point(start)?;
                let end = map_point(end)?;

                if let (Some(start), Some(end)) = (&start, &end) {
                    let op = if end_inclusive { BinaryOp::CmpLt } else { BinaryOp::CmpLte };
                    match self.require_value_true_for_range(expr.span, &Value::Binary(op, start.clone(), end.clone())) {
                        Ok(()) => {}
                        Err(e) => return Ok(ScopedEntryDirect::Error(e)),
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
                            Box::new(Value::Int(BigInt::ZERO)),
                            Box::new(self.eval_expression_as_value(scope, inner)?),
                        )
                    }
                    UnaryOp::Not =>
                        Value::UnaryNot(Box::new(self.eval_expression_as_value(scope, inner)?)),
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_as_value(scope, left)?;
                let right = self.eval_expression_as_value(scope, right)?;

                let result = Value::Binary(op, Box::new(left), Box::new(right));
                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::TernarySelect(_, _, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "ternary select expression")),
            ExpressionKind::ArrayIndex(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "array index expression")),
            ExpressionKind::DotIdIndex(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "dot id index expression")),
            ExpressionKind::DotIntIndex(_, _) =>
                ScopedEntryDirect::Error(self.diag.report_todo(expr.span, "dot int index expression")),
            ExpressionKind::Call(ref target, ref args) => {
                let target_entry = self.eval_expression(scope, target)?;

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
                            return Ok(ScopedEntryDirect::Error(self.diag.report(err)));
                        }

                        // check kind and type match, and collect in replacement map
                        let mut map_ty: IndexMap<GenericTypeParameter, Type> = IndexMap::new();
                        let mut map_value: IndexMap<GenericValueParameter, Value> = IndexMap::new();
                        let mut last_err = None;

                        for (&param, arg) in zip_eq(&parameters.vec, &args.inner) {
                            match param {
                                GenericParameter::Type(param) => {
                                    let arg_ty = self.eval_expression_as_ty(scope, arg)?;
                                    // TODO use for bound-check (once we add type bounds)
                                    let _param_info = &self.compiled[param];
                                    map_ty.insert_first(param, arg_ty)
                                }
                                GenericParameter::Value(param) => {
                                    let arg_value = self.eval_expression_as_value(scope, arg)?;

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
                            return Ok(ScopedEntryDirect::Error(e));
                        }

                        // do the actual replacement
                        let result = inner.replace_generic_params(&mut self.compiled, &map_ty, &map_value);
                        MaybeConstructor::Immediate(result)
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => {
                                let diag = Diagnostic::new_simple("invalid call target", target.span, "invalid call target kind 'type'");
                                ScopedEntryDirect::Error(self.diag.report(diag))
                            }
                            TypeOrValue::Value(_) =>
                                ScopedEntryDirect::Error(self.diag.report_todo(target.span, "value call target")),
                        }
                    }
                    ScopedEntryDirect::Error(e) => ScopedEntryDirect::Error(e),
                }
            }
            ExpressionKind::Builtin(ref args) => {
                return match self.eval_builtin_call(scope, expr.span, args)? {
                    Ok(result) => Ok(MaybeConstructor::Immediate(result)),
                    Err(e) => Ok(MaybeConstructor::Error(e)),
                };
            }
        };
        Ok(result)
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> ResolveResult<Type> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            // TODO unify these error strings somewhere
            // TODO maybe move back to central error collection place for easier unit testing?
            ScopedEntryDirect::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type, got constructor", expr.span, "constructor");
                Ok(Type::Error(self.diag.report(diag)))
            }
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(ty) => Ok(ty),
                TypeOrValue::Value(_) => {
                    let diag = Diagnostic::new_simple("expected type, got value", expr.span, "value");
                    Ok(Type::Error(self.diag.report(diag)))
                }
            }
            ScopedEntryDirect::Error(e) => Ok(Type::Error(e))
        }
    }

    pub fn eval_expression_as_value(&mut self, scope: Scope, expr: &Expression) -> ResolveResult<Value> {
        let entry = self.eval_expression(scope, expr)?;
        match entry {
            ScopedEntryDirect::Constructor(_) => {
                let err = Diagnostic::new_simple("expected value, got constructor", expr.span, "constructor");
                Ok(Value::Error(self.diag.report(err)))
            }
            ScopedEntryDirect::Immediate(entry) => match entry {
                TypeOrValue::Type(_) => {
                    let err = Diagnostic::new_simple("expected value, got type", expr.span, "type");
                    Ok(Value::Error(self.diag.report(err)))
                }
                TypeOrValue::Value(value) => Ok(value),
            }
            ScopedEntryDirect::Error(e) => Ok(Value::Error(e)),
        }
    }

    pub fn eval_sync_domain(&mut self, scope: Scope, domain: &SyncDomain<Box<Expression>>) -> ResolveResult<SyncDomain<Value>> {
        let clock = self.eval_expression_as_value(scope, &domain.clock)?;
        let reset = self.eval_expression_as_value(scope, &domain.reset)?;

        // TODO check that clock is a clock
        // TODO check that reset is a boolean
        // TODO check that reset is either async or sync to the same clock

        Ok(SyncDomain {
            clock,
            reset,
        })
    }

    fn eval_builtin_call(
        &mut self,
        scope: Scope,
        expr_span: Span,
        args: &Args,
    ) -> ResolveResult<Result<TypeOrValue, ErrorGuaranteed>> {
        let get_arg_str = |i: usize| -> Option<&str> {
            args.inner.get(i).and_then(|e| match &e.inner {
                ExpressionKind::StringLiteral(s) => Some(s.as_str()),
                _ => None,
            })
        };

        if let (Some(first), Some(second)) = (get_arg_str(0), get_arg_str(1)) {
            let rest = &args.inner[2..];

            match (first, second, rest) {
                ("type", "bool", &[]) =>
                    return Ok(Ok(TypeOrValue::Type(Type::Boolean))),
                ("type", "int", &[]) => {
                    let range = Box::new(Value::Range(RangeInfo::unbounded()));
                    return Ok(Ok(TypeOrValue::Type(Type::Integer(IntegerTypeInfo { range }))));
                },
                ("type", "int_range", [range]) => {
                    // TODO typecheck (range must be integer)
                    let range = Box::new(self.eval_expression_as_value(scope, range)?);
                    let ty_info = IntegerTypeInfo { range };
                    return Ok(Ok(TypeOrValue::Type(Type::Integer(ty_info))));
                },
                ("type", "Range", &[]) =>
                    return Ok(Ok(TypeOrValue::Type(Type::Range))),
                ("type", "bits_inf", &[]) =>
                    return Ok(Ok(TypeOrValue::Type(Type::Bits(None)))),
                ("type", "bits", [bits]) => {
                    // TODO typecheck (bits must be non-negative integer)
                    let bits = self.eval_expression_as_value(scope, bits)?;
                    return Ok(Ok(TypeOrValue::Type(Type::Bits(Some(Box::new(bits))))));
                },
                ("type", "Array", [ty, len]) => {
                    // TODO typecheck: len must be uint
                    let ty = self.eval_expression_as_ty(scope, ty)?;
                    let len = self.eval_expression_as_value(scope, len)?;
                    return Ok(Ok(TypeOrValue::Type(Type::Array(Box::new(ty), Box::new(len)))));
                },
                ("function", "print", [value]) => {
                    let _: Value = self.eval_expression_as_value(scope, value)?;
                    return Ok(Ok(TypeOrValue::Value(Value::Unit)))
                }
                // fallthrough into error
                _ => {},
            }
        }

        let err = Diagnostic::new("invalid arguments for __builtin call")
            .snippet(expr_span)
            .add_error(args.span, "invalid arguments")
            .finish()
            .finish()
            .into();
        Ok(Err(self.diag.report(err)))
    }
}

