use crate::front::assignment::{MaybeAssignedValue, VariableValues};
use crate::front::block::{BlockEnd, BlockEndReturn, TypedIrExpression};
use crate::front::check::{check_type_contains_value, TypeContainsReason};
use crate::front::compile::{CompileItemContext, CompileRefs, StackEntry};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::misc::ScopedEntry;
use crate::front::scope::{DeclaredValueSingle, Scope, ScopeParent};
use crate::front::types::Type;
use crate::front::value::{CompileValue, MaybeCompile, NamedValue};
use crate::syntax::ast::{
    Arg, Args, Block, BlockStatement, Expression, Identifier, Parameter as AstParameter, Spanned,
};
use crate::syntax::parsed::AstRefItem;
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::util::data::IndexMapExt;
use crate::util::{ResultDoubleExt, ResultExt};
use indexmap::map::Entry as IndexMapEntry;
use indexmap::IndexMap;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct FunctionValue {
    // only used for uniqueness
    pub decl_span: Span,
    pub scope_captured: CapturedScope,

    pub params: Spanned<Vec<AstParameter>>,
    pub body: Spanned<FunctionBody>,
}

impl FunctionValue {
    pub fn equality_key(&self) -> impl Eq + Hash + '_ {
        // TODO get this to implement hash
        (self.decl_span, &self.scope_captured)
    }
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    TypeAliasExpr(Box<Expression>),
    FunctionBodyBlock {
        // TODO avoid ast clones?
        body: Block<BlockStatement>,
        ret_ty: Option<Box<Expression>>,
    },
    // TODO Enum, Struct
    // TODO should generic modules are be implemented here?
}

// TODO avoid repeated hashing of this potentially large type
/// The parent scope is kept separate to avoid a hard dependency on all items that are in scope,
///   now capturing functions still allow graph-based item evaluation.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct CapturedScope {
    parent_file: FileId,
    child_values: BTreeMap<String, Result<Spanned<CapturedValue>, ErrorGuaranteed>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum CapturedValue {
    Item(AstRefItem),
    Value(CompileValue),
    FailedCapture(FailedCaptureReason),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum FailedCaptureReason {
    NotCompile,
    NotFullyInitialized,
}

impl CompileItemContext<'_, '_> {
    pub fn match_args_to_params_and_typecheck<I, V: Clone + Into<MaybeCompile<TypedIrExpression>>>(
        &mut self,
        scope: &mut Scope,
        params: &Spanned<Vec<AstParameter>>,
        args: &Args<I, Spanned<V>>,
    ) -> Result<Vec<(Identifier, V)>, ErrorGuaranteed>
    where
        for<'i> &'i I: Into<Option<&'i Identifier>>,
    {
        let diags = self.refs.diags;

        // check params unique
        // TODO we could do this earlier, but then parameters that only exist conditionally can't get checked yet
        //   eventually we will do partial checking of generic items, then this check will trigger early enough
        let mut param_ids: IndexMap<&str, &Identifier> = IndexMap::new();
        let mut e = Ok(());
        for param in &params.inner {
            match param_ids.entry(&param.id.string) {
                IndexMapEntry::Occupied(entry) => {
                    let diag = Diagnostic::new("parameter declared twice")
                        .add_info(entry.get().span, "first declared here")
                        .add_error(param.span, "redeclared here".to_string())
                        .finish();
                    e = Err(diags.report(diag));
                }
                IndexMapEntry::Vacant(entry) => {
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

        let mut param_values_vec = vec![];
        let no_vars = VariableValues::NO_VARS;

        for param_info in &params.inner {
            let param_id = &param_info.id;

            // eval and check param type
            let param_ty = self.eval_expression_as_ty(scope, &no_vars, &param_info.ty)?;
            let (_, _, arg_value) = args_passed.get(&param_id.string).ok_or_else(|| {
                diags.report_internal_error(params.span, "finished matching args, but got missing param name")
            })?;

            // TODO this is abusing the assignment reason, and the span is probably not even right
            let reason = TypeContainsReason::Assignment {
                span_target: param_id.span,
                span_target_ty: param_ty.span,
            };
            let arg_value_maybe = arg_value.as_ref().map_inner(|arg_value| arg_value.clone().into());
            check_type_contains_value(diags, reason, &param_ty.inner, arg_value_maybe.as_ref(), true, true)?;

            // record value into vec
            param_values_vec.push((param_id.clone(), arg_value.inner.clone()));

            // declare param in scope
            let entry = DeclaredValueSingle::Value {
                span: param_id.span,
                value: ScopedEntry::Value(arg_value_maybe.inner),
            };
            scope.declare_already_checked(param_id.string.clone(), entry);
        }

        Ok(param_values_vec)
    }
}

impl CompileItemContext<'_, '_> {
    pub fn call_function<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        function: &FunctionValue,
        args: Args<Option<Identifier>, Spanned<MaybeCompile<TypedIrExpression>>>,
    ) -> Result<(C::Block, MaybeCompile<TypedIrExpression>), ErrorGuaranteed> {
        let diags = self.refs.diags;

        // recreate captured scope
        let span_scope = function.params.span.join(function.body.span);
        let scope_captured = function.scope_captured.to_scope(self.refs, span_scope)?;

        // map params into scope
        let mut scope = Scope::new_child(span_scope, &scope_captured);
        self.match_args_to_params_and_typecheck(&mut scope, &function.params, &args)?;
        let scope = &scope;

        // run the body
        let entry = StackEntry::FunctionRun(function.decl_span);
        self.recurse(entry, |s| {
            match &function.body.inner {
                FunctionBody::TypeAliasExpr(expr) => {
                    let result = s.eval_expression_as_ty(scope, &VariableValues::NO_VARS, expr)?.inner;
                    let result = MaybeCompile::Compile(CompileValue::Type(result));

                    let empty_block = ctx.new_ir_block();
                    Ok((empty_block, result))
                }
                FunctionBody::FunctionBodyBlock { body, ret_ty } => {
                    // evaluate return type
                    let ret_ty = ret_ty
                        .as_ref()
                        .map(|ret_ty| s.eval_expression_as_ty(scope, &VariableValues::NO_VARS, ret_ty))
                        .transpose();

                    // evaluate block
                    let (ir_block, end) = s.elaborate_block(ctx, VariableValues::new(), scope, body)?;

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
    match end.unwrap_normal_or_return_in_function(diags)? {
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
        BlockEnd::Stopping(BlockEndReturn { span_keyword, value }) => {
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
                    check_type_contains_value(diags, reason, &ret_ty.inner, value.as_ref(), true, true)?;
                    Ok(value.inner)
                }
            }
        }
    }
}

impl CapturedScope {
    pub fn from_scope(
        diags: &Diagnostics,
        scope: &Scope,
        vars: Option<&VariableValues>,
    ) -> Result<CapturedScope, ErrorGuaranteed> {
        // TODO should we build this incrementally, or build a normal hashmap once and then sort it at the end?
        let mut child_values = HashMap::new();

        let mut curr = scope;
        let parent_file = loop {
            match curr.parent() {
                ScopeParent::Some(parent) => {
                    // this is a non-root scope, capture it
                    for (id, value) in curr.immediate_entries() {
                        let child_values_entry = match child_values.entry(id.to_owned()) {
                            HashMapEntry::Occupied(_) => {
                                // shadowed by child scope
                                continue;
                            }
                            HashMapEntry::Vacant(child_values_entry) => child_values_entry,
                        };

                        let captured = match value {
                            DeclaredValueSingle::Value { span, value } => {
                                let captured = match value {
                                    &ScopedEntry::Item(value) => Ok(CapturedValue::Item(value)),
                                    ScopedEntry::Named(named) => match named {
                                        &NamedValue::Variable(var) => {
                                            let vars = vars.unwrap_or(&VariableValues::NO_VARS);

                                            // TODO these spans are probably wrong
                                            match vars.get_maybe(diags, span, var)? {
                                                MaybeAssignedValue::Assigned(assigned) => match &assigned.value {
                                                    MaybeCompile::Compile(value) => {
                                                        Ok(CapturedValue::Value(value.clone()))
                                                    }
                                                    MaybeCompile::Other(_) => Ok(CapturedValue::FailedCapture(
                                                        FailedCaptureReason::NotCompile,
                                                    )),
                                                },
                                                MaybeAssignedValue::NotYetAssigned
                                                | MaybeAssignedValue::PartiallyAssigned
                                                | MaybeAssignedValue::FailedIfMerge(_) => {
                                                    Ok(CapturedValue::FailedCapture(
                                                        FailedCaptureReason::NotFullyInitialized,
                                                    ))
                                                }
                                            }
                                        }
                                        NamedValue::Port(_) | NamedValue::Wire(_) | NamedValue::Register(_) => {
                                            Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotCompile))
                                        }
                                    },
                                    ScopedEntry::Value(MaybeCompile::Compile(value)) => {
                                        Ok(CapturedValue::Value(value.clone()))
                                    }
                                    ScopedEntry::Value(MaybeCompile::Other(_)) => {
                                        Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotCompile))
                                    }
                                };
                                captured.map(|c| Spanned::new(span, c))
                            }
                            DeclaredValueSingle::FailedCapture(span, reason) => {
                                Ok(Spanned::new(span, CapturedValue::FailedCapture(reason)))
                            }
                            DeclaredValueSingle::Error(e) => Err(e),
                        };

                        child_values_entry.insert(captured);
                    }

                    curr = parent;
                }
                ScopeParent::None(file) => {
                    // this is the top file scope, no need to capture this
                    break file;
                }
            }
        };

        Ok(CapturedScope {
            parent_file,
            child_values: child_values.into_iter().collect(),
        })
    }

    pub fn to_scope<'s>(&self, refs: CompileRefs<'_, 's>, scope_span: Span) -> Result<Scope<'s>, ErrorGuaranteed> {
        let CapturedScope {
            parent_file,
            child_values,
        } = self;

        let parent_file = refs.shared.file_scopes.get(parent_file).unwrap().as_ref_ok()?;
        let mut scope = Scope::new_child(scope_span, parent_file);

        // TODO we need a span, even for errors
        for (id, value) in child_values {
            let declared = match value {
                Ok(value) => {
                    let span = value.span;
                    match &value.inner {
                        &CapturedValue::Item(item) => DeclaredValueSingle::Value {
                            span,
                            value: ScopedEntry::Item(item),
                        },
                        CapturedValue::Value(value) => DeclaredValueSingle::Value {
                            span,
                            value: ScopedEntry::Value(MaybeCompile::Compile(value.clone())),
                        },
                        &CapturedValue::FailedCapture(reason) => DeclaredValueSingle::FailedCapture(span, reason),
                    }
                }
                &Err(e) => DeclaredValueSingle::Error(e),
            };
            scope.declare_already_checked(id.clone(), declared);
        }

        Ok(scope)
    }
}

impl Eq for FunctionValue {}

impl PartialEq for FunctionValue {
    fn eq(&self, other: &Self) -> bool {
        self.equality_key() == other.equality_key()
    }
}

impl Hash for FunctionValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.equality_key().hash(state);
    }
}
