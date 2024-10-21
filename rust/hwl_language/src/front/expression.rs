use crate::data::compiled::GenericParameter;
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::common::{ExpressionContext, GenericContainer, GenericMap, ScopedEntry, ScopedEntryDirect, TypeOrValue, ValueDomain};
use crate::front::driver::CompileState;
use crate::front::module::MaybeDriverCollector;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{GenericArguments, GenericParameters, IntegerTypeInfo, MaybeConstructor, Type};
use crate::front::values::{RangeInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{BinaryOp, DomainKind, Expression, ExpressionKind, IntPattern, RangeLiteral, Spanned, SyncDomain, UnaryOp};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use annotate_snippets::Level;
use itertools::{zip_eq, Itertools};
use num_bigint::BigInt;
use num_traits::One;
use std::cmp::min;

impl CompileState<'_, '_> {
    // TODO this should support separate signature and value queries too
    //    eg. if we want to implement a "typeof" operator that doesn't run code we need it
    //    careful, think about how this interacts with the future type inference system
    // TODO return a spanned value here
    pub fn eval_expression(&mut self, ctx: &ExpressionContext, collector: &mut MaybeDriverCollector, expr: &Expression) -> ScopedEntryDirect {
        let diags = self.diags;

        match expr.inner {
            ExpressionKind::Dummy =>
                ScopedEntryDirect::Error(diags.report_todo(expr.span, "dummy expression")),
            ExpressionKind::Any =>
                ScopedEntryDirect::Immediate(TypeOrValue::Type(Type::Any)),
            ExpressionKind::Wrapped(ref inner) =>
                self.eval_expression(ctx, collector, inner),
            ExpressionKind::Id(ref id) => {
                let entry = match self.compiled[ctx.scope].find(&self.compiled.scopes, diags, id, Visibility::Private) {
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
            ExpressionKind::IntPattern(ref pattern) => {
                match pattern {
                    IntPattern::Hex(_) =>
                        ScopedEntryDirect::Error(diags.report_todo(expr.span, "hex int-pattern expression")),
                    IntPattern::Bin(_) =>
                        ScopedEntryDirect::Error(diags.report_todo(expr.span, "bin int-pattern expression")),
                    IntPattern::Dec(str_raw) => {
                        let str_clean = str_raw.replace("_", "");
                        let value = str_clean.parse::<BigInt>().unwrap();
                        ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::IntConstant(value)))
                    }
                }
            }
            ExpressionKind::BoolLiteral(b) =>
                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::BoolConstant(b))),
            ExpressionKind::StringLiteral(ref s) =>
                ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::StringConstant(s.clone()))),
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
                        .map(|p| Box::new(self.eval_expression_as_value(ctx, collector, p)))
                };

                let start = map_point(start);
                let end_raw = map_point(end);

                let end = if end_inclusive {
                    end_raw.map(|e| Box::new(Value::Binary(BinaryOp::Add, e, Box::new(Value::IntConstant(BigInt::one())))))
                } else {
                    end_raw
                };

                if let (Some(start), Some(end)) = (&start, &end) {
                    match self.require_value_true_for_range(expr.span, &Value::Binary(BinaryOp::CmpLte, start.clone(), end.clone())) {
                        Ok(()) => {}
                        Err(e) => return ScopedEntryDirect::Error(e),
                    }
                }

                let value = Value::Range(RangeInfo { start, end });
                ScopedEntryDirect::Immediate(TypeOrValue::Value(value))
            }
            ExpressionKind::UnaryOp(op, ref inner) => {
                let result = match op {
                    UnaryOp::Neg => {
                        Value::Binary(
                            BinaryOp::Sub,
                            Box::new(Value::IntConstant(BigInt::ZERO)),
                            Box::new(self.eval_expression_as_value(ctx, collector, inner)),
                        )
                    }
                    UnaryOp::Not =>
                        Value::UnaryNot(Box::new(self.eval_expression_as_value(ctx, collector, inner))),
                };

                ScopedEntryDirect::Immediate(TypeOrValue::Value(result))
            }
            ExpressionKind::BinaryOp(op, ref left, ref right) => {
                let left = self.eval_expression_as_value(ctx, collector, left);
                let right = self.eval_expression_as_value(ctx, collector, right);

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
                let target_entry = self.eval_expression(ctx, collector, target);
                let args_entry = args.map_inner(|e| self.eval_expression_as_ty_or_value(ctx, collector, e));

                match target_entry {
                    ScopedEntryDirect::Constructor(constr) => {
                        match self.eval_constructor_call(&constr.parameters, &constr.inner, args_entry, true) {
                            Ok((v, _)) => MaybeConstructor::Immediate(v),
                            Err(e) => MaybeConstructor::Error(e),
                        }
                    }
                    ScopedEntryDirect::Immediate(entry) => {
                        match entry {
                            TypeOrValue::Type(_) => {
                                let err = Diagnostic::new_simple("invalid call target", target.span, "invalid call target kind 'type'");
                                ScopedEntryDirect::Error(diags.report(err))
                            }
                            TypeOrValue::Value(_) =>
                                ScopedEntryDirect::Error(diags.report_todo(target.span, "value call target")),
                            TypeOrValue::Error(e)
                            => ScopedEntryDirect::Error(e),
                        }
                    }
                    ScopedEntryDirect::Error(e) => ScopedEntryDirect::Error(e),
                }
            }
            ExpressionKind::Builtin(ref args) => {
                match self.eval_builtin_call(ctx, collector, expr.span, args) {
                    Ok(result) => MaybeConstructor::Immediate(result),
                    Err(e) => MaybeConstructor::Error(e),
                }
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
            let ast::Arg { span: arg_span, name: arg_name, expr: arg_value, } = arg;

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

                    match self.check_type_contains(Some(ty_span), arg_span, &ty_replaced, &arg_value) {
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

    pub fn eval_expression_as_ty_or_value(&mut self, ctx: &ExpressionContext, collector: &mut MaybeDriverCollector, expr: &Expression) -> TypeOrValue {
        let entry = self.eval_expression(ctx, collector, expr);

        match entry {
            ScopedEntryDirect::Immediate(entry) => entry,
            ScopedEntryDirect::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type or value, got constructor", expr.span, "constructor");
                TypeOrValue::Error(self.diags.report(diag))
            }
            ScopedEntryDirect::Error(e) => TypeOrValue::Error(e),
        }
    }

    pub fn eval_expression_as_ty(&mut self, scope: Scope, expr: &Expression) -> Type {
        let ctx = ExpressionContext::constant(expr.span, scope);
        let entry = self.eval_expression(&ctx, &mut MaybeDriverCollector::None, expr);

        match entry {
            // TODO unify these error strings somewhere
            // TODO maybe move back to central error collection place for easier unit testing?
            // TODO report span for the _reason_ why we expect one or the other
            ScopedEntryDirect::Constructor(_) => {
                let diag = Diagnostic::new_simple("expected type, got constructor", expr.span, "constructor");
                Type::Error(self.diags.report(diag))
            }
            ScopedEntryDirect::Immediate(entry) => entry.unwrap_ty(self.diags, expr.span),
            ScopedEntryDirect::Error(e) => Type::Error(e),
        }
    }

    pub fn eval_expression_as_value(
        &mut self,
        ctx: &ExpressionContext, collector: &mut MaybeDriverCollector,
        expr: &Expression,
    ) -> Value {
        let entry = self.eval_expression(ctx, collector, expr);

        match entry {
            ScopedEntryDirect::Constructor(_) => {
                let err = Diagnostic::new_simple("expected value, got constructor", expr.span, "constructor");
                Value::Error(self.diags.report(err))
            }
            ScopedEntryDirect::Immediate(entry) => entry.unwrap_value(self.diags, expr.span),
            ScopedEntryDirect::Error(e) => Value::Error(e),
        }
    }

    pub fn eval_domain(&mut self, scope: Scope, domain: Spanned<&DomainKind<Box<Expression>>>) -> DomainKind<Value> {
        match domain.inner {
            DomainKind::Async =>
                DomainKind::Async,
            DomainKind::Sync(sync_domain) => {
                let sync = SyncDomain { clock: &*sync_domain.clock, reset: &*sync_domain.reset };
                let sync = Spanned { span: domain.span, inner: sync };
                DomainKind::Sync(self.eval_sync_domain(scope, sync))
            }
        }
    }

    pub fn eval_sync_domain(&mut self, scope: Scope, sync: Spanned<SyncDomain<&Expression>>) -> SyncDomain<Value> {
        let Spanned { span: _, inner: sync } = sync;
        let SyncDomain { clock, reset } = sync;

        let ctx = ExpressionContext::passthrough(scope);
        let mut collector = MaybeDriverCollector::None;
        let clock_value_unchecked = self.eval_expression_as_value(&ctx, &mut collector, clock);
        let reset_value_unchecked = self.eval_expression_as_value(&ctx, &mut collector, reset);

        // check that clock is a clock
        let clock_domain = self.domain_of_value(clock.span, &clock_value_unchecked);
        let clock_value = match &clock_domain {
            ValueDomain::Clock => clock_value_unchecked,
            &ValueDomain::Error(e) => Value::Error(e),
            _ => {
                let title = format!("clock must be a clock, has domain {}", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, &clock_domain));
                let e = self.diags.report_simple(title, clock.span, "clock value");
                Value::Error(e)
            }
        };

        // check that reset is an async bool
        // TODO allow sync reset
        // TODO require that async reset still comes out of sync in phase with the clock
        let reset_value_bool = match self.check_type_contains(None, reset.span, &Type::Boolean, &reset_value_unchecked) {
            Ok(()) => reset_value_unchecked,
            Err(e) => Value::Error(e),
        };
        let reset_domain = self.domain_of_value(reset.span, &reset_value_bool);
        let reset_value = match &reset_domain {
            ValueDomain::Async => reset_value_bool,
            &ValueDomain::Error(e) => Value::Error(e),
            _ => {
                let title = format!("reset must be an async boolean, has domain {}", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, &reset_domain));
                let e = self.diags.report_simple(title, reset.span, "reset value");
                Value::Error(e)
            }
        };

        SyncDomain {
            clock: clock_value,
            reset: reset_value,
        }
    }

    fn eval_builtin_call(
        &mut self,
        ctx: &ExpressionContext, collector: &mut MaybeDriverCollector,
        expr_span: Span,
        args: &ast::Args,
    ) -> Result<TypeOrValue, ErrorGuaranteed> {
        let diags = self.diags;

        let args_span = args.span;
        let args = args.inner.iter()
            .map(|arg| {
                let ast::Arg { span: _, name, expr: arg_expr } = arg;
                if let Some(name) = name {
                    diags.report_simple("named arguments are not allowed for __builtin calls", name.span, "named argument");
                }
                let inner = self.eval_expression_as_ty_or_value(ctx, collector, arg_expr);
                Spanned { span: arg.span, inner }
            })
            .collect_vec();

        let get_arg_str = |i: usize| -> Option<&str> {
            args.get(i).and_then(|e| match &e.inner {
                TypeOrValue::Value(Value::StringConstant(s)) => Some(s.as_str()),
                _ => None,
            })
        };

        if let (Some(first), Some(second)) = (get_arg_str(0), get_arg_str(1)) {
            let rest = &args[2..];

            match (first, second, rest) {
                ("type", "unchecked", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Unchecked)),
                ("value", "undefined", &[]) =>
                    return Ok(TypeOrValue::Value(Value::Undefined)),
                ("type", "bool", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Boolean)),
                ("type", "int", &[]) => {
                    let range = Box::new(Value::Range(RangeInfo::UNBOUNDED));
                    return Ok(TypeOrValue::Type(Type::Integer(IntegerTypeInfo { range })));
                }
                ("type", "int_range", [range]) => {
                    // TODO typecheck (range must be integer)
                    let range = Box::new(range.inner.clone().unwrap_value(diags, range.span));
                    let ty_info = IntegerTypeInfo { range };
                    return Ok(TypeOrValue::Type(Type::Integer(ty_info)));
                }
                ("type", "Range", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Range)),
                ("type", "bits_inf", &[]) =>
                    return Ok(TypeOrValue::Type(Type::Bits(None))),
                ("type", "bits", [bits]) => {
                    // TODO typecheck (bits must be non-negative integer)
                    let bits = bits.inner.clone().unwrap_value(diags, bits.span);
                    return Ok(TypeOrValue::Type(Type::Bits(Some(Box::new(bits)))));
                }
                ("type", "Array", [ty, len]) => {
                    // TODO typecheck: len must be uint
                    let ty = ty.inner.clone().unwrap_ty(diags, ty.span);
                    let len = len.inner.clone().unwrap_value(diags, len.span);
                    return Ok(TypeOrValue::Type(Type::Array(Box::new(ty), Box::new(len))));
                }
                ("function", "print", [value]) => {
                    let _: Value = value.inner.clone().unwrap_value(diags, value.span);
                    return Ok(TypeOrValue::Value(Value::Unit));
                }
                // fallthrough into error
                _ => {}
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

impl TypeOrValue<Type, Value> {
    pub fn unwrap_ty(self, diags: &Diagnostics, span: Span) -> Type {
        match self {
            TypeOrValue::Type(ty) => ty,
            TypeOrValue::Value(_) => {
                let diag = Diagnostic::new_simple("expected type, got value", span, "value");
                Type::Error(diags.report(diag))
            }
            TypeOrValue::Error(e) => Type::Error(e),
        }
    }

    pub fn unwrap_value(self, diags: &Diagnostics, span: Span) -> Value {
        match self {
            TypeOrValue::Type(_) => {
                let diag = Diagnostic::new_simple("expected value, got type", span, "type");
                Value::Error(diags.report(diag))
            }
            TypeOrValue::Value(value) => value,
            TypeOrValue::Error(e) => Value::Error(e),
        }
    }
}
