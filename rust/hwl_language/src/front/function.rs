use crate::front::block::{BlockEnd, TypedIrExpression, VariableValues};
use crate::front::check::{check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileState, ElaborationStackEntry, ParameterInfo};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::misc::ScopedEntry;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::Type;
use crate::front::value::{CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{
    Arg, Args, Block, BlockStatement, Expression, GenericParameter, Identifier, MaybeIdentifier, Spanned,
};
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::ResultDoubleExt;
use indexmap::map::Entry;
use indexmap::IndexMap;
use itertools::Itertools;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct FunctionValue {
    // TODO only this value is used for eq/hash, is that okay?
    //   this will certainly need to be expanded once lambdas are supported
    pub item: AstRefItem,

    pub outer_scope: Scope,

    // TODO avoid ast cloning
    // TODO Eq+Hash are a bit weird for types containing ast nodes
    pub params: Spanned<Vec<GenericParameter>>,

    pub body_span: Span,
    pub body: FunctionBody,
}

impl Eq for FunctionValue {}

impl PartialEq for FunctionValue {
    fn eq(&self, other: &Self) -> bool {
        self.item == other.item
    }
}

impl Hash for FunctionValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.item.hash(state);
    }
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    TypeAliasExpr(Box<Expression>),
    FunctionBodyBlock {
        body: Block<BlockStatement>,
        ret_ty: Option<Box<Expression>>,
    }, // Enum(/*TODO*/),
       // Struct(/*TODO*/),
}

impl CompileState<'_> {
    pub fn match_args_to_params_and_typecheck<I, V: Clone + Into<MaybeCompile<TypedIrExpression>>>(
        &mut self,
        params: &Spanned<Vec<GenericParameter>>,
        args: &Args<I, Spanned<V>>,
        scope_outer: Scope,
        span_scope_inner: Span,
    ) -> Result<(Scope, Vec<(Identifier, V)>), ErrorGuaranteed>
    where
        for<'i> &'i I: Into<Option<&'i Identifier>>,
    {
        let diags = self.diags;

        // check params unique
        // TODO we could do this earlier, but then parameters that only exist conditionally can't get checked yet
        //   eventually we will do partial checking of generic items, then this check will trigger early enough
        let mut param_ids: IndexMap<&str, &Identifier> = IndexMap::new();
        let mut e = Ok(());
        for param in &params.inner {
            match param_ids.entry(&param.id.string) {
                Entry::Occupied(entry) => {
                    let diag = Diagnostic::new("parameter declared twice")
                        .add_info(entry.get().span, "first declared here")
                        .add_error(param.span, "redeclared here".to_string())
                        .finish();
                    e = Err(diags.report(diag));
                }
                Entry::Vacant(entry) => {
                    entry.insert(&param.id);
                }
            }
        }
        let () = e?;

        // match args to params
        let mut first_named_span = None;
        let mut args_passed = IndexMap::new();

        for arg in &args.inner {
            let &Arg {
                span: arg_span,
                name: ref arg_name,
                value: ref arg_value,
            } = arg;
            let arg_name = arg_name.into();

            match (first_named_span, arg_name) {
                (None, None) => {
                    // positional arg
                    match param_ids.get_index(args_passed.len()) {
                        Some((_, &param_id)) => {
                            args_passed.insert_first(param_id.string.clone(), (param_id, arg_span, arg_value));
                        }
                        None => {
                            let diag = Diagnostic::new("too many arguments")
                                .add_info(params.span, format!("expected {} parameter(s)", param_ids.len()))
                                .add_error(arg.span, format!("trying to pass {} argument(s)", args.inner.len()))
                                .finish();
                            return Err(diags.report(diag));
                        }
                    }
                }
                (_, Some(name)) => {
                    // named arg
                    match args_passed.get(&name.string) {
                        None => match param_ids.get(name.string.as_str()) {
                            Some(&param_id) => {
                                args_passed.insert(name.string.clone(), (param_id, arg_span, arg_value));
                                first_named_span = first_named_span.or(Some(arg_span));
                            }
                            None => {
                                let diag = Diagnostic::new(format!("unexpected argument `{}`", name.string))
                                    .add_info(params.span, "parameters declared here")
                                    .add_error(name.span, "unexpected argument")
                                    .finish();
                                return Err(diags.report(diag));
                            }
                        },
                        Some(&(_, prev_span, _)) => {
                            let diag = Diagnostic::new(format!("argument `{}` passed twice", name.string))
                                .add_info(prev_span, "first passed here")
                                .add_error(arg.span, "passed again here")
                                .finish();
                            return Err(diags.report(diag));
                        }
                    }
                }
                (Some(first_named_span), None) => {
                    let diag = Diagnostic::new("positional argument after named argument")
                        .add_info(first_named_span, "first named argument here")
                        .add_error(arg.span, "positional argument here".to_string())
                        .finish();
                    return Err(diags.report(diag));
                }
            }
        }

        // report missing args
        for (_, param_id) in param_ids {
            if !args_passed.contains_key(param_id.string.as_str()) {
                let diag = Diagnostic::new("missing argument")
                    .add_error(
                        args.span,
                        format!("missing argument for parameter `{}`", param_id.string),
                    )
                    .add_info(param_id.span, "parameter declared here")
                    .finish();
                return Err(diags.report(diag));
            }
        }

        // typecheck and final result building
        if params.inner.len() != args_passed.len() {
            return Err(diags.report_internal_error(args.span, "finished matching args, but got wrong final length"));
        }

        let scope_params = self
            .scopes
            .new_child(scope_outer, span_scope_inner, Visibility::Private);
        let mut param_values_vec = vec![];
        let no_vars = VariableValues::new_no_vars();

        for param_info in &params.inner {
            let param_id = &param_info.id;

            // check param type
            let param_ty = self.eval_expression_as_ty(scope_params, &no_vars, &param_info.ty)?;
            let (_, _, arg_value) = args_passed.get(&param_id.string).ok_or_else(|| {
                diags.report_internal_error(params.span, "finished matching args, but got missing param name")
            })?;

            // TODO this is abusing the assignment reason, and the span is probably not even right
            let reason = TypeContainsReason::Assignment {
                span_target: param_id.span,
                span_target_ty: param_ty.span,
            };
            let arg_value_maybe = arg_value.as_ref().map_inner(|arg_value| arg_value.clone().into());
            check_type_contains_value(diags, reason, &param_ty.inner, arg_value_maybe.as_ref(), true)?;

            // record value into vec
            param_values_vec.push((param_id.clone(), arg_value.inner.clone()));

            // declare param in scope
            let param = self.parameters.push(ParameterInfo {
                id: MaybeIdentifier::Identifier(param_id.clone()),
                value: arg_value_maybe.inner,
            });
            let entry = ScopedEntry::Direct(NamedValue::Parameter(param));
            self.scopes[scope_params].declare_already_checked(
                diags,
                param_id.string.clone(),
                param_id.span,
                Ok(entry),
            )?;
        }

        Ok((scope_params, param_values_vec))
    }
}

impl FunctionValue {
    pub fn call<C: ExpressionContext>(
        &self,
        state: &mut CompileState,
        ctx: &mut C,
        args: Args<Option<Identifier>, Spanned<MaybeCompile<TypedIrExpression>>>,
    ) -> Result<(C::Block, MaybeCompile<TypedIrExpression>), ErrorGuaranteed> {
        let diags = state.diags;

        // map params
        // TODO discard scope after use?
        let span_scope_inner = self.params.span.join(self.body_span);
        let (param_scope, param_values) =
            state.match_args_to_params_and_typecheck(&self.params, &args, self.outer_scope, span_scope_inner)?;

        // TODO cache function calls?
        // TODO we already do this for module elaborations, which are similar
        let param_key = param_values.into_iter().map(|(_, v)| v).collect_vec();
        let stack_entry = ElaborationStackEntry::FunctionRun(self.item, param_key);
        state
            .check_compile_loop(stack_entry, |state| {
                // run the body
                // TODO add execution limits?
                match &self.body {
                    FunctionBody::TypeAliasExpr(expr) => {
                        let result = state
                            .eval_expression_as_ty(param_scope, &VariableValues::new_no_vars(), expr)?
                            .inner;
                        let result = MaybeCompile::Compile(CompileValue::Type(result));

                        let empty_block = ctx.new_ir_block();
                        Ok((empty_block, result))
                    }
                    FunctionBody::FunctionBodyBlock { body, ret_ty } => {
                        // evaluate return type
                        let ret_ty = ret_ty
                            .as_ref()
                            .map(|ret_ty| {
                                state.eval_expression_as_ty(param_scope, &VariableValues::new_no_vars(), ret_ty)
                            })
                            .transpose();

                        // evaluate block
                        let (ir_block, end) = state.elaborate_block(ctx, VariableValues::new(), param_scope, body)?;

                        // check return type
                        let ret_ty = ret_ty?;
                        let ret_value = check_function_return_value(diags, body.span, &ret_ty, end)?;

                        Ok((ir_block, ret_value))
                    }
                }
            })
            .flatten_err()
    }
}

fn check_function_return_value(
    diags: &Diagnostics,
    body_span: Span,
    ret_ty: &Option<Spanned<Type>>,
    end: BlockEnd,
) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
    match end {
        BlockEnd::Normal(_next_vars) => {
            // no return, only allowed for unit-returning functions
            match ret_ty {
                None => Ok(MaybeCompile::Compile(CompileValue::UNIT)),
                Some(ret_ty) => {
                    if ret_ty.inner == Type::UNIT {
                        Ok(MaybeCompile::Compile(CompileValue::UNIT))
                    } else {
                        let diag = Diagnostic::new("control flow reaches end of function with return type")
                            .add_error(Span::single_at(body_span.end), "end of function is reached here")
                            .add_info(
                                ret_ty.span,
                                format!("return type `{}` declared here", ret_ty.inner.to_diagnostic_string()),
                            )
                            .finish();
                        Err(diags.report(diag))
                    }
                }
            }
        }
        BlockEnd::Return { span_keyword, value } => {
            // return, check type
            match (ret_ty, value) {
                (None, None) => Ok(MaybeCompile::Compile(CompileValue::UNIT)),
                (Some(ret_ty), None) => {
                    if ret_ty.inner == Type::UNIT {
                        Ok(MaybeCompile::Compile(CompileValue::UNIT))
                    } else {
                        let diag = Diagnostic::new("missing return value in function with return type")
                            .add_error(span_keyword, "return here without value")
                            .add_info(
                                ret_ty.span,
                                format!(
                                    "function return type `{}` declared here",
                                    ret_ty.inner.to_diagnostic_string()
                                ),
                            )
                            .finish();
                        Err(diags.report(diag))
                    }
                }
                (None, Some(ret_value)) => {
                    if ret_value.inner == MaybeCompile::Compile(CompileValue::UNIT) {
                        Ok(MaybeCompile::Compile(CompileValue::UNIT))
                    } else {
                        let diag = Diagnostic::new("return value in function without return type")
                            .add_error(ret_value.span, "return value here")
                            .finish();
                        Err(diags.report(diag))
                    }
                }
                (Some(ret_ty), Some(value)) => {
                    let reason = TypeContainsReason::Return {
                        span_keyword,
                        span_return_ty: ret_ty.span,
                    };
                    check_type_contains_value(diags, reason, &ret_ty.inner, value.as_ref(), true)?;
                    Ok(value.inner)
                }
            }
        }
    }
}
