use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::parsed::AstRefItem;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::{CompileState, ElaborationStackEntry};
use crate::new::misc::ScopedEntry;
use crate::new::value::{CompileValue, ScopedValue};
use crate::syntax::ast::{Arg, Args, Expression, GenericParameter, Identifier, Spanned};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use crate::util::ResultDoubleExt;
use indexmap::map::Entry;
use indexmap::IndexMap;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionValue {
    pub outer_scope: Scope,
    pub item: AstRefItem,

    // TODO avoid ast cloning
    // TODO Eq+Hash are a bit weird for types containing ast nodes
    pub params: Spanned<Vec<GenericParameter>>,
    pub ret_ty: ReturnType,

    pub body_span: Span,
    pub body: FunctionBody,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ReturnType {
    Type,
    Expression(Box<Expression>),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum FunctionBody {
    Type(Box<Expression>),
    // Enum(/*TODO*/),
    // Struct(/*TODO*/),
    // TODO add normal functions
}

// TODO implement call_runtime which generates ir code
impl FunctionValue {
    pub fn call_compile_time(&self, state: &mut CompileState, args: Args<CompileValue>) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = state.diags;

        // check params unique
        // TODO we could do this earlier, but then parameters that only exist conditionally can't get checked yet
        //   eventually we will do partial checking of generic items, then this check will trigger early enough
        let mut param_ids: IndexMap<&str, &Identifier> = IndexMap::new();
        let mut e = Ok(());
        for param in &self.params.inner {
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

        // check args valid
        let mut first_named_span = None;
        let mut args_passed = IndexMap::new();

        for arg in &args.inner {
            let &Arg { span: arg_span, name: ref arg_name, value: ref arg_value } = arg;

            match (first_named_span, arg_name) {
                (None, None) => {
                    // positional arg
                    match param_ids.get_index(args_passed.len()) {
                        Some((_, param_id)) => {
                            args_passed.insert_first(param_id.string.clone(), (arg_span, arg_value));
                        }
                        None => {
                            let diag = Diagnostic::new("too many arguments")
                                .add_info(self.params.span, format!("expected {} parameter(s)", param_ids.len()))
                                .add_error(arg.span, format!("trying to pass {} argument(s)", args.inner.len()))
                                .finish();
                            return Err(diags.report(diag));
                        }
                    }
                }
                (_, Some(name)) => {
                    // named arg
                    match args_passed.get(&name.string) {
                        None => {
                            match param_ids.get(name.string.as_str()) {
                                Some(_) => {
                                    args_passed.insert(name.string.clone(), (arg_span, arg_value));
                                    first_named_span = first_named_span.or(Some(arg_span));
                                }
                                None => {
                                    let diag = Diagnostic::new(format!("unexpected argument `{}`", name.string))
                                        .add_info(self.params.span, "parameters declared here")
                                        .add_error(name.span, "unexpected argument")
                                        .finish();
                                    return Err(diags.report(diag));
                                }
                            }
                        }
                        Some(&(prev_span, _)) => {
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

        // TODO cache function calls?
        let stack_entry = ElaborationStackEntry::FunctionRun(self.item, args.clone());
        state.check_compile_loop(stack_entry, |state| {
            // create temporary scope
            // TODO discard scope after use?
            let scope_span = self.params.span.join(self.body_span);
            let scope = state.scopes.new_child(self.outer_scope, scope_span, Visibility::Private);

            // populate scope with args
            for (id, (span, value)) in args_passed {
                let entry = ScopedEntry::Direct(ScopedValue::Compile(value.clone()));
                state.scopes[scope].declare_already_checked(diags, id, span, Ok(entry))?;
            }

            // run the body
            // TODO add execution limits?
            match &self.body {
                FunctionBody::Type(expr) => {
                    state.eval_expression_as_compile(scope, expr, "type body")
                }
            }
        }).flatten_err()
    }
}