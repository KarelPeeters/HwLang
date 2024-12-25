use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::parsed::AstRefItem;
use crate::front::scope::{Scope, Visibility};
use crate::new::block::VariableValues;
use crate::new::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::new::compile::{CompileState, ConstantInfo, ElaborationStackEntry};
use crate::new::misc::ScopedEntry;
use crate::new::types::Type;
use crate::new::value::{CompileValue, NamedValue};
use crate::syntax::ast::{
    Arg, Args, Block, BlockStatement, Expression, GenericParameter, Identifier, MaybeIdentifier, Spanned,
};
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
    pub ret_ty: ReturnType,

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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ReturnType {
    Evaluated(Type),
    Expression(Box<Expression>),
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    Type(Box<Expression>),
    Block(Block<BlockStatement>),
    // Enum(/*TODO*/),
    // Struct(/*TODO*/),
}

impl CompileState<'_> {
    pub fn match_args_to_params<I>(
        &mut self,
        params: &Spanned<Vec<GenericParameter>>,
        args: &Args<I, Spanned<CompileValue>>,
        scope_outer: Scope,
        span_scope_inner: Span,
    ) -> Result<(Scope, Vec<(Identifier, CompileValue)>), ErrorGuaranteed>
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
            check_type_contains_compile_value(diags, reason, &param_ty.inner, arg_value.as_ref(), true)?;

            // record value into vec
            param_values_vec.push((param_id.clone(), arg_value.inner.clone()));

            // declare param in scope
            let param = self.parameters.push(ConstantInfo {
                id: MaybeIdentifier::Identifier(param_id.clone()),
                value: arg_value.inner.clone(),
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

// TODO implement call_runtime which generates ir code
impl FunctionValue {
    pub fn call_compile_time(
        &self,
        state: &mut CompileState,
        args: Args<Option<Identifier>, Spanned<CompileValue>>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        // map params
        // TODO discard scope after use?
        let span_scope_inner = self.params.span.join(self.body_span);
        let (param_scope, param_values) =
            state.match_args_to_params(&self.params, &args, self.outer_scope, span_scope_inner)?;

        // TODO cache function calls?
        // TODO we already do this for module elaborations, which are similar
        let param_key = param_values.into_iter().map(|(_, v)| v).collect_vec();
        let stack_entry = ElaborationStackEntry::FunctionRun(self.item, param_key);
        state
            .check_compile_loop(stack_entry, |state| {
                // run the body
                // TODO add execution limits?
                match &self.body {
                    FunctionBody::Type(expr) => {
                        let result = state
                            .eval_expression_as_compile(param_scope, &VariableValues::new_no_vars(), expr, "type body")?
                            .inner;
                        Ok(result)
                    }
                    FunctionBody::Block(block) => Err(state.diags.report_todo(block.span, "run function body")),
                }
            })
            .flatten_err()
    }
}
