use crate::front::block::{BlockEnd, BlockEndReturn};
use crate::front::check::{check_type_contains_value, check_type_is_bool_array, TypeContainsReason};
use crate::front::compile::{ArenaVariables, CompileItemContext, CompileRefs, StackEntry};
use crate::front::context::ExpressionContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::item::ElaboratedItemParams;
use crate::front::scope::{DeclaredValueSingle, Scope, ScopeParent};
use crate::front::scope::{NamedValue, ScopedEntry};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::front::variables::{MaybeAssignedValue, VariableValues};
use crate::mid::ir::IrExpressionLarge;
use crate::syntax::ast::{
    Arg, Args, Block, BlockStatement, Expression, Identifier, MaybeIdentifier, Parameter as AstParameter, Spanned,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModule};
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::util::data::{IndexMapExt, VecExt};
use crate::util::{ResultDoubleExt, ResultExt};
use indexmap::map::Entry as IndexMapEntry;
use indexmap::IndexMap;
use itertools::Itertools;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

#[derive(Debug, Clone)]
pub enum FunctionValue {
    User(UserFunctionValue),
    Bits(FunctionBits),
}

#[derive(Debug, Clone)]
pub struct FunctionBits {
    pub ty_hw: HardwareType,
    pub kind: FunctionBitsKind,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum FunctionBitsKind {
    ToBits,
    FromBits,
}

// TODO find a better name for this
#[derive(Debug, Clone)]
pub struct UserFunctionValue {
    // only used for uniqueness
    pub decl_span: Span,
    pub scope_captured: CapturedScope,

    // TODO point into ast instead of storing a clone here
    pub params: Spanned<Vec<AstParameter>>,
    pub body: Spanned<FunctionBody>,
}

#[derive(Debug, Clone)]
pub enum FunctionBody {
    FunctionBodyBlock {
        // TODO avoid ast clones, just refer to the ast item here
        body: Block<BlockStatement>,
        ret_ty: Option<Box<Expression>>,
    },

    TypeAliasExpr(Box<Expression>),
    ModulePortsAndBody(AstRefModule),
    Interface(AstRefInterface),
    // TODO add struct, enum
}

impl FunctionBody {
    pub fn params_must_be_compile(&self) -> bool {
        match self {
            // TODO add optional marker to functions that can only be used with compare args
            FunctionBody::FunctionBodyBlock { .. } => false,
            FunctionBody::TypeAliasExpr(_) | FunctionBody::ModulePortsAndBody(_) | FunctionBody::Interface(_) => true,
        }
    }
}

// TODO maybe move this into the variables module
// TODO avoid repeated hashing of this potentially large type
// TODO this Eq is too comprehensive, this can cause duplicate module backend generation.
//   We only need to check for captures values that could actually be used
//   this is really hard to known in advance,
//   but maybe we can a an approximation pre-pass that checks all usages that _could_ happen?
//   For now users can do this themselves already with a file-level trampoline function
//   that returns a new function that can only capture the outer params, not a full scope.
//   As another solution, we could de-duplicate modules after IR generation again.
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
    pub fn match_args_to_params_and_typecheck<'p, I, V: Clone + Into<Value>>(
        &mut self,
        vars: &mut VariableValues,
        scope_outer: &'p Scope,
        params: &Spanned<Vec<AstParameter>>,
        args: &Args<I, Spanned<V>>,
        args_must_be_compile: bool,
    ) -> Result<(Scope<'p>, Vec<(Identifier, V)>), ErrorGuaranteed>
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

        // TODO wrong span, this should include the body
        let mut param_values_vec = vec![];
        let mut scope = Scope::new_child(params.span, scope_outer);

        for param_info in &params.inner {
            let param_id = &param_info.id;

            // eval and check param type
            let param_ty = self.eval_expression_as_ty(&scope, vars, &param_info.ty)?;
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

            // check compile-time
            if args_must_be_compile {
                match arg_value_maybe.inner {
                    Value::Compile(_) => {}
                    Value::Hardware(_) => {
                        let diag = Diagnostic::new("arguments must be compile-time constants")
                            .add_info(param_id.span, "param declared here needs to be compile-time constant")
                            .add_error(arg_value.span, "hardware value passed here")
                            .finish();
                        return Err(diags.report(diag));
                    }
                }
            }

            // record value into vec
            param_values_vec.push((param_id.clone(), arg_value.inner.clone()));

            // declare param in scope
            let param_var = vars.var_new_immutable_init(
                &mut self.variables,
                MaybeIdentifier::Identifier(param_id.clone()),
                param_id.span,
                arg_value_maybe.inner,
            );
            let entry = DeclaredValueSingle::Value {
                span: param_id.span,
                value: ScopedEntry::Named(NamedValue::Variable(param_var)),
            };
            scope.declare_already_checked(param_id.string.clone(), entry);
        }

        Ok((scope, param_values_vec))
    }
}

impl CompileItemContext<'_, '_> {
    pub fn call_function<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        vars: &mut VariableValues,
        span_call: Span,
        function: &FunctionValue,
        args: Args<Option<Identifier>, Spanned<Value>>,
    ) -> Result<(Option<C::Block>, Value), ErrorGuaranteed> {
        match function {
            FunctionValue::User(function) => self.call_normal_function(ctx, vars, function, args),
            FunctionValue::Bits(function) => {
                let result = self.call_bits_function(span_call, function, args)?;
                Ok((None, result))
            }
        }
    }

    fn call_normal_function<C: ExpressionContext>(
        &mut self,
        ctx: &mut C,
        vars: &mut VariableValues,
        function: &UserFunctionValue,
        args: Args<Option<Identifier>, Spanned<Value>>,
    ) -> Result<(Option<C::Block>, Value), ErrorGuaranteed> {
        let diags = self.refs.diags;
        self.refs.check_should_stop(function.decl_span)?;

        // recreate captured scope
        let span_scope = function.params.span.join(function.body.span);
        let scope_captured = function
            .scope_captured
            .to_scope(&mut self.variables, vars, self.refs, span_scope)?;

        // map params into scope
        let (scope, param_values) = self.match_args_to_params_and_typecheck(
            vars,
            &scope_captured,
            &function.params,
            &args,
            function.body.inner.params_must_be_compile(),
        )?;

        // run the body
        let entry = StackEntry::FunctionRun(function.decl_span);
        self.recurse(entry, |s| {
            match &function.body.inner {
                FunctionBody::FunctionBodyBlock { body, ret_ty } => {
                    // evaluate return type
                    let ret_ty = ret_ty
                        .as_ref()
                        .map(|ret_ty| s.eval_expression_as_ty(&scope, vars, ret_ty))
                        .transpose();

                    // evaluate block
                    let (ir_block, end) = s.elaborate_block(ctx, &scope, vars, body)?;

                    // check return type
                    let ret_ty = ret_ty?;
                    let ret_value = check_function_return_value(diags, body.span, &ret_ty, end)?;

                    Ok((Some(ir_block), ret_value))
                }
                FunctionBody::TypeAliasExpr(expr) => {
                    let result_ty = s.eval_expression_as_ty(&scope, vars, expr)?.inner;
                    let result_value = Value::Compile(CompileValue::Type(result_ty));
                    Ok((None, result_value))
                }
                &FunctionBody::ModulePortsAndBody(item) => {
                    let param_values = param_values
                        .into_iter()
                        .map(|(id, v)| (id, v.unwrap_compile()))
                        .collect_vec();

                    let item_params = ElaboratedItemParams {
                        item,
                        params: Some(param_values),
                    };
                    let (result_id, _) = s.refs.elaborate_module(item_params)?;
                    let result_value = Value::Compile(CompileValue::Module(result_id));
                    Ok((None, result_value))
                }
                &FunctionBody::Interface(item) => {
                    let param_values = param_values
                        .into_iter()
                        .map(|(id, v)| (id, v.unwrap_compile()))
                        .collect_vec();

                    let item_params = ElaboratedItemParams {
                        item,
                        params: Some(param_values),
                    };
                    let (result_id, _) = s.refs.elaborate_interface(item_params)?;
                    let result_value = Value::Compile(CompileValue::Interface(result_id));
                    Ok((None, result_value))
                }
            }
        })
        .flatten_err()
    }

    fn call_bits_function(
        &mut self,
        span_call: Span,
        function: &FunctionBits,
        args: Args<Option<Identifier>, Spanned<Value>>,
    ) -> Result<Value, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // check arg is single non-named value
        let value = match args.inner.single() {
            Some(Arg { span: _, name, value }) => {
                if let Some(name) = name {
                    return Err(diags.report_simple(
                        "this function expects a single unnamed parameter",
                        name.span,
                        "got named argument here",
                    ));
                }
                value
            }
            None => {
                return Err(diags.report_simple(
                    "this function expects a single parameter",
                    args.span,
                    "incorrect arguments here",
                ));
            }
        };

        // actual implementation
        let FunctionBits { ty_hw, kind } = function;
        match kind {
            FunctionBitsKind::ToBits => {
                check_type_contains_value(
                    diags,
                    TypeContainsReason::Operator(span_call),
                    &ty_hw.as_type(),
                    value.as_ref(),
                    false,
                    false,
                )?;

                let ty_ir = ty_hw.as_ir();
                let width = ty_ir.size_bits();

                let result = match &value.inner {
                    Value::Compile(value) => {
                        // TODO dedicated compile-time bits value that's faster than a boxed array of bools
                        let bits = ty_hw
                            .value_to_bits(value)
                            .map_err(|_| diags.report_internal_error(span_call, "to_bits failed"))?;
                        Value::Compile(CompileValue::Array(
                            bits.into_iter().map(CompileValue::Bool).collect_vec(),
                        ))
                    }
                    Value::Hardware(value_raw) => {
                        let value = value_raw.clone().soft_expand_to_type(&mut self.large, ty_hw);

                        let expr = self
                            .large
                            .push_expr(IrExpressionLarge::ToBits(ty_ir, value.expr.clone()));
                        let ty_bits = HardwareType::Array(Box::new(HardwareType::Bool), width);

                        Value::Hardware(HardwareValue {
                            ty: ty_bits,
                            domain: value.domain,
                            expr,
                        })
                    }
                };
                Ok(result)
            }
            FunctionBitsKind::FromBits => {
                let ty_ir = ty_hw.as_ir();
                let width = ty_ir.size_bits();

                let value =
                    check_type_is_bool_array(diags, TypeContainsReason::Operator(span_call), value, Some(&width))?;

                let result = match value {
                    Value::Compile(v) => Value::Compile(ty_hw.value_from_bits(&v).map_err(|_| {
                        let msg = format!(
                            "while converting value `{:?}` into type `{}`",
                            v,
                            ty_hw.to_diagnostic_string()
                        );
                        diags.report_simple("`from_bits` failed", span_call, msg)
                    })?),
                    Value::Hardware(v) => {
                        let expr = self.large.push_expr(IrExpressionLarge::FromBits(ty_ir, v.expr.clone()));
                        Value::Hardware(HardwareValue {
                            ty: ty_hw.clone(),
                            domain: v.domain,
                            expr,
                        })
                    }
                };
                Ok(result)
            }
        }
    }
}

fn check_function_return_value(
    diags: &Diagnostics,
    body_span: Span,
    ret_ty: &Option<Spanned<Type>>,
    end: BlockEnd,
) -> Result<Value, ErrorGuaranteed> {
    match end.unwrap_normal_or_return_in_function(diags)? {
        BlockEnd::Normal => {
            // no return, only allowed for unit-returning functions
            match ret_ty {
                None => Ok(Value::Compile(CompileValue::UNIT)),
                Some(ret_ty) => {
                    if ret_ty.inner == Type::UNIT {
                        Ok(Value::Compile(CompileValue::UNIT))
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
                (None, None) => Ok(Value::Compile(CompileValue::UNIT)),
                (Some(ret_ty), None) => {
                    if ret_ty.inner == Type::UNIT {
                        Ok(Value::Compile(CompileValue::UNIT))
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
                    let is_unit =
                        matches!(&ret_value.inner, Value::Compile(CompileValue::Tuple(tuple)) if tuple.is_empty());
                    if is_unit {
                        Ok(Value::Compile(CompileValue::UNIT))
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
    pub fn from_file_scope(scope: FileId) -> CapturedScope {
        CapturedScope {
            parent_file: scope,
            child_values: BTreeMap::new(),
        }
    }

    pub fn from_scope(
        diags: &Diagnostics,
        scope: &Scope,
        vars: &VariableValues,
    ) -> Result<CapturedScope, ErrorGuaranteed> {
        // TODO should we build this incrementally, or build a normal hashmap once and then sort it at the end?
        // it's fine to use a hashmap here, this will be sorted into a BTreeMap later
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
                                            // TODO these spans are probably wrong
                                            let maybe = vars.var_get_maybe(diags, span, var)?;
                                            maybe_assigned_to_captured(maybe)
                                        }
                                        NamedValue::Port(_)
                                        | NamedValue::Wire(_)
                                        | NamedValue::Register(_)
                                        | NamedValue::PortInterface(_) => {
                                            Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotCompile))
                                        }
                                    },
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

    pub fn to_scope<'s>(
        &self,
        variables: &mut ArenaVariables,
        vars: &mut VariableValues,
        refs: CompileRefs<'_, 's>,
        scope_span: Span,
    ) -> Result<Scope<'s>, ErrorGuaranteed> {
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
                        CapturedValue::Value(ref value) => {
                            let id_recreated = MaybeIdentifier::Identifier(Identifier {
                                span,
                                string: id.clone(),
                            });
                            let var = vars.var_new_immutable_init(
                                variables,
                                id_recreated,
                                span,
                                Value::Compile(value.clone()),
                            );

                            DeclaredValueSingle::Value {
                                span,
                                value: ScopedEntry::Named(NamedValue::Variable(var)),
                            }
                        }
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

fn maybe_assigned_to_captured(maybe: &MaybeAssignedValue) -> Result<CapturedValue, ErrorGuaranteed> {
    match maybe {
        MaybeAssignedValue::Assigned(assigned) => match &assigned.value_and_version {
            Value::Compile(value) => Ok(CapturedValue::Value(value.clone())),
            Value::Hardware(_) => Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotCompile)),
        },
        MaybeAssignedValue::NotYetAssigned | MaybeAssignedValue::PartiallyAssigned => {
            Ok(CapturedValue::FailedCapture(FailedCaptureReason::NotFullyInitialized))
        }
        &MaybeAssignedValue::Error(e) => Err(e),
    }
}

impl Eq for FunctionValue {}

impl FunctionValue {
    pub fn equality_key(&self) -> impl Eq + Hash + '_ {
        match self {
            FunctionValue::User(UserFunctionValue {
                decl_span,
                scope_captured,
                params,
                body,
            }) => {
                let _ = (params, body);
                (Some((*decl_span, scope_captured)), None)
            }
            FunctionValue::Bits(FunctionBits { ty_hw: ty, kind }) => (None, Some((ty, kind))),
        }
    }
}

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
