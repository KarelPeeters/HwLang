use crate::front::check::{
    TypeContainsReason, check_type_is_bool, check_type_is_bool_compile, check_type_is_string,
    check_type_is_string_compile,
};
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::DiagResult;
use crate::front::domain::ValueDomain;
use crate::front::expression::NamedOrValue;
use crate::front::flow::{Flow, FlowKind};
use crate::front::implication::ValueWithImplications;
use crate::front::scope::{NamedValue, Scope};
use crate::front::string::hardware_print_string;
use crate::front::types::{HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, MaybeCompile, NotCompile, Value};
use crate::mid::ir::IrExpression;
use crate::syntax::ast::{Arg, Args, ExpressionKind};
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::token::TOKEN_STR_BUILTIN;
use crate::util::data::VecExt;
use crate::util::store::ArcOrRef;

impl CompileItemContext<'_, '_> {
    pub fn eval_type_of(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr_span: Span,
        args: &Args,
    ) -> DiagResult<Type> {
        let diags = self.refs.diags;

        // check single unnamed arg
        // TODO extract common code, there are probably other users of this pattern
        let Args {
            span: args_span,
            inner: args_inner,
        } = args;
        let arg = match args_inner.single_ref() {
            Ok(&Arg { span: _, name, value }) => {
                if let Some(name) = name {
                    return Err(diags.report_simple(
                        "typeof only takes a single unnamed argument",
                        name.span,
                        "tried to pass named argument here",
                    ));
                } else {
                    value
                }
            }
            Err(()) => {
                return Err(diags.report_simple(
                    "typeof only takes a single unnamed argument",
                    *args_span,
                    "incorrect number of arguments here",
                ));
            }
        };

        // eval id
        let &ExpressionKind::Id(id) = self.refs.get_expr(arg) else {
            return Err(diags.report_simple(
                "typeof only works on identifiers, not general expressions",
                arg.span,
                "tried to pass non-identifier here",
            ));
        };
        let id = self.eval_general_id(scope, flow, id)?;
        let value = self
            .eval_named_or_value(scope, id.as_ref().map_inner(ArcOrRef::as_ref))?
            .inner;

        // get type
        let ty = match value {
            NamedOrValue::ItemValue(value) => value.ty(),
            NamedOrValue::Named(value) => match value {
                NamedValue::Variable(var) => {
                    let info = flow.var_eval(diags, &mut self.large, Spanned::new(arg.span, var))?;
                    info.into_value().ty()
                }
                NamedValue::Port(port) => self.ports[port].ty.inner.as_type(),
                NamedValue::Wire(wire) => {
                    let typed = self.wires[wire].typed(self.refs, &self.wire_interfaces, arg.span)?;
                    typed.ty.inner.as_type()
                }
                NamedValue::Register(reg) => self.registers[reg].ty.inner.as_type(),
                NamedValue::PortInterface(_) | NamedValue::WireInterface(_) => {
                    return Err(diags.report_todo(expr_span, "typeof for interfaces"));
                }
            },
        };
        Ok(ty)
    }

    pub fn eval_builtin(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        expr_span: Span,
        target_span: Span,
        args: &Args,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        // check that there are no named arguments
        let Args {
            span: _,
            inner: args_inner,
        } = args;
        for arg in args_inner {
            let Arg {
                span: _,
                name,
                value: _,
            } = arg;
            if let Some(name) = name {
                let msg = format!("{TOKEN_STR_BUILTIN} does not support named arguments");
                return Err(diags.report_internal_error(name.span, msg));
            }
        }

        // evaluate the first two arguments as string literals
        if args_inner.len() < 2 {
            return Err(diags.report_internal_error(
                args.span,
                format!("{TOKEN_STR_BUILTIN} requires at least two arguments"),
            ));
        }
        let arg_0 = check_type_is_string_compile(
            diags,
            TypeContainsReason::Operator(target_span),
            self.eval_expression_as_compile(scope, flow, &Type::String, args_inner[0].value, "builtin arg 0")?,
        )?;
        let arg_1 = check_type_is_string_compile(
            diags,
            TypeContainsReason::Operator(target_span),
            self.eval_expression_as_compile(scope, flow, &Type::String, args_inner[1].value, "builtin arg 1")?,
        )?;
        let args_rest = &args_inner[2..];

        // handle the different builtins
        match (arg_0.as_str(), arg_1.as_str(), args_rest) {
            // basic types
            ("type", "any", &[]) => Ok(Value::new_ty(Type::Any)),
            ("type", "bool", &[]) => Ok(Value::new_ty(Type::Bool)),
            ("type", "str", &[]) => Ok(Value::new_ty(Type::String)),
            ("type", "Range", &[]) => Ok(Value::new_ty(Type::Range)),
            ("type", "int", &[]) => Ok(Value::new_ty(Type::Int(IncRange::OPEN))),
            // print
            ("fn", "print", &[msg]) => {
                let msg = self.eval_expression(scope, flow, &Type::String, msg.value)?;
                let reason_str = TypeContainsReason::Internal(expr_span);

                match flow.kind_mut() {
                    FlowKind::Compile(_) => {
                        let msg_inner = CompileValue::try_from(&msg.inner).map_err(|_: NotCompile| {
                            diags.report_internal_error(expr_span, "non-compile expression in compile flow")
                        })?;
                        let msg = check_type_is_string_compile(diags, reason_str, Spanned::new(msg.span, msg_inner))?;
                        self.refs.print_handler.println(&msg);
                    }
                    FlowKind::Hardware(flow) => {
                        let msg = check_type_is_string(diags, TypeContainsReason::Internal(expr_span), msg)?;
                        hardware_print_string(
                            &self.refs.shared.elaboration_arenas,
                            flow,
                            &mut self.large,
                            expr_span,
                            &msg,
                        );
                    }
                }

                Ok(Value::unit())
            }
            // assert
            ("fn", "assert", &[cond, msg]) => {
                // TODO support hardware cond and msg
                let cond =
                    self.eval_expression_as_compile(scope, flow, &Type::Bool, cond.value, "assertion condition")?;
                let cond = check_type_is_bool_compile(diags, TypeContainsReason::Internal(expr_span), cond)?;

                let msg = self.eval_expression_as_compile(scope, flow, &Type::String, msg.value, "message")?;
                let msg = check_type_is_string_compile(diags, TypeContainsReason::Internal(expr_span), msg)?;

                if cond {
                    Ok(Value::unit())
                } else {
                    // TODO include stack trace
                    //   (and ensure it's deterministic even in multithreaded builds)
                    Err(diags.report_simple(
                        format!("assertion failed with message {msg:?}"),
                        expr_span,
                        "failed here",
                    ))
                }
            }
            // casting
            ("fn", "unsafe_bool_to_clock", &[value]) => {
                let value = self.eval_expression(scope, flow, &Type::Bool, value.value)?;
                let value = check_type_is_bool(
                    diags,
                    TypeContainsReason::Internal(expr_span),
                    value.map_inner(ValueWithImplications::simple),
                )?;

                let expr = match value {
                    MaybeCompile::Compile(v) => IrExpression::Bool(v),
                    MaybeCompile::Hardware(v) => v.value.expr,
                };
                Ok(Value::Hardware(HardwareValue {
                    ty: HardwareType::Bool,
                    domain: ValueDomain::Clock,
                    expr,
                }))
            }
            _ => Err(diags.report_internal_error(expr_span, "invalid builtin arguments")),
        }
    }
}
