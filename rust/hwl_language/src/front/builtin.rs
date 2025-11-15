use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable};
use crate::front::domain::ValueDomain;
use crate::front::flow::{Flow, FlowKind};
use crate::front::types::{HardwareType, IncRange, Type, Typed};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::IrStatement;
use crate::syntax::ast::{Arg, Args};
use crate::syntax::pos::{Span, Spanned};
use crate::syntax::token::TOKEN_STR_BUILTIN;
use crate::util::iter::IterExt;

impl CompileItemContext<'_, '_> {
    // TODO replace builtin+import+prelude with keywords?
    pub fn eval_builtin(
        &mut self,
        flow: &mut impl Flow,
        expr_span: Span,
        target_span: Span,
        args: Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        // evaluate args
        // TODO delay arg evaluation so we can do weird things like use certain args as a domain?
        let Args {
            span: _,
            inner: args_inner,
        } = args;
        let args_eval = args_inner
            .into_iter()
            .map(|arg| {
                let Arg { span: _, name, value } = arg;
                if let Some(name) = name {
                    let diag = Diagnostic::new(format!("{TOKEN_STR_BUILTIN} does not support named arguments"))
                        .snippet(expr_span)
                        .add_error(name.span, "tried to pass named argument here")
                        .add_info(target_span, format!("calling {TOKEN_STR_BUILTIN} here"))
                        .finish()
                        .finish();
                    return Err(diags.report(diag));
                }
                Ok(value.inner)
            })
            .try_collect_all_vec()?;

        if let (Some(Value::Compile(CompileValue::String(a0))), Some(Value::Compile(CompileValue::String(a1)))) =
            (args_eval.get(0), args_eval.get(1))
        {
            let rest = &args_eval[2..];
            let print_compile = |v: &Value| {
                let value_str = match v {
                    // TODO print strings without quotes
                    Value::Compile(v) => v.diagnostic_string(),
                    // TODO less ugly formatting for HardwareValue
                    Value::Hardware(v) => {
                        let HardwareValue { ty, domain, expr: _ } = v;
                        let ty_str = ty.diagnostic_string();
                        let domain_str = domain.diagnostic_string(self);
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
                ("type", "int", []) => {
                    return Ok(Value::Compile(CompileValue::Type(Type::Int(IncRange::OPEN))));
                }
                ("fn", "typeof", [value]) => return Ok(Value::Compile(CompileValue::Type(value.ty()))),
                ("fn", "print", [value]) => {
                    match flow.kind_mut() {
                        FlowKind::Compile(_) => {
                            // TODO record similarly to diagnostics, where they can be deterministically printed later
                            print_compile(value);
                            return Ok(Value::Compile(CompileValue::unit()));
                        }
                        FlowKind::Hardware(flow) => {
                            if let Value::Compile(CompileValue::String(value)) = value {
                                let stmt = Spanned::new(expr_span, IrStatement::PrintLn((**value).clone()));
                                flow.push_ir_statement(stmt);
                                return Ok(Value::Compile(CompileValue::unit()));
                            }
                            // fallthough
                        }
                    }
                }
                (
                    "fn",
                    "assert",
                    &[
                        Value::Compile(CompileValue::Bool(cond)),
                        Value::Compile(CompileValue::String(ref msg)),
                    ],
                ) => {
                    return if cond {
                        Ok(Value::Compile(CompileValue::unit()))
                    } else {
                        Err(diags.report_simple(
                            format!("assertion failed with message {msg:?}"),
                            expr_span,
                            "failed here",
                        ))
                    };
                }
                ("fn", "assert", [Value::Hardware(_), Value::Compile(CompileValue::String(_))]) => {
                    return Err(diags.report_todo(expr_span, "runtime assert"));
                }
                ("fn", "unsafe_bool_to_clock", [v]) => match v.ty() {
                    Type::Bool => {
                        let expr = v.as_hardware_value(self.refs, &mut self.large, expr_span, &HardwareType::Bool)?;
                        return Ok(Value::Hardware(HardwareValue {
                            ty: HardwareType::Bool,
                            domain: ValueDomain::Clock,
                            expr: expr.expr.clone(),
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
}
