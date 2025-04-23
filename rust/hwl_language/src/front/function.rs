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
    Arg, Args, Block, BlockStatement, Expression, Identifier, MaybeIdentifier, Parameter, ParameterItem, Parameters,
    Spanned,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModule};
use crate::syntax::pos::Span;
use crate::syntax::source::FileId;
use crate::util::data::VecExt;
use crate::util::{ResultDoubleExt, ResultExt};
use indexmap::map::Entry;
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};
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
    pub params: Parameters,
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
    pub fn match_args_to_params_and_typecheck<'a, 'p>(
        &mut self,
        vars: &mut VariableValues,
        scope_outer: &'p Scope,
        params: &'a Parameters,
        args: &Args<Option<Identifier>, Spanned<Value>>,
        args_must_be_compile: bool,
    ) -> Result<(Scope<'p>, Vec<(Identifier, Value)>), ErrorGuaranteed> {
        let diags = self.refs.diags;

        // check for duplicate arg names and check that positional args are before named args
        let mut arg_name_to_index: IndexMap<&str, usize> = IndexMap::new();
        let mut first_named_span = None;
        let mut positional_count: usize = 0;
        let mut any_err_args = Ok(());
        for (arg_index, arg) in enumerate(&args.inner) {
            if args_must_be_compile && !matches!(arg.value.inner, Value::Compile(_)) {
                let diag = Diagnostic::new("call target only supports compile-time arguments")
                    .add_info(params.span, "parameters defined here")
                    .add_error(arg.value.span, "hardware value passed here")
                    .finish();
                any_err_args = Err(diags.report(diag));
            }

            match &arg.name {
                Some(id) => {
                    match arg_name_to_index.entry(id.string.as_str()) {
                        Entry::Occupied(entry) => {
                            let prev_index = *entry.get();
                            let diag = Diagnostic::new("duplicate named parameter")
                                .add_info(args.inner[prev_index].span, "previously passed here")
                                .add_error(id.span, "passed again here")
                                .finish();
                            any_err_args = Err(diags.report(diag));
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(arg_index);
                        }
                    }

                    if first_named_span.is_none() {
                        first_named_span = Some(id.span);
                    }
                }
                None => {
                    if let Some(first_named_span) = first_named_span {
                        let diag = Diagnostic::new("positional arguments after named arguments are not allowed")
                            .add_info(first_named_span, "first named argument here")
                            .add_error(arg.span, "later positional argument here".to_string())
                            .finish();
                        any_err_args = Err(diags.report(diag));
                    }
                    positional_count += 1;
                }
            }
        }
        any_err_args?;

        // match args to params
        let mut scope = Scope::new_child(params.span, scope_outer);
        let mut next_param_index: usize = 0;
        let mut arg_used = vec![false; args.inner.len()];
        let mut param_values_vec = vec![];

        let mut param_names: IndexMap<&str, Span> = IndexMap::new();

        let mut any_err_params = Ok(());
        let mut visit_param =
            |ctx: &mut CompileItemContext, vars: &mut VariableValues, scope: &mut Scope, param: &'a Parameter| {
                let param_index = next_param_index;
                next_param_index += 1;

                if let Some(prev_span) = param_names.insert(&param.id.string, param.id.span) {
                    let diag = Diagnostic::new("duplicate parameter name")
                        .add_info(prev_span, "previously defined here")
                        .add_error(param.id.span, "defined again here")
                        .finish();
                    any_err_params = Err(diags.report(diag));
                    return;
                }

                let param_value = if param_index < positional_count {
                    // positional match
                    let arg_index = param_index;
                    let arg = &args.inner[arg_index];
                    assert!(arg.name.is_none());
                    assert!(!arg_used[arg_index]);

                    // check if there's also a named match to get better error messages
                    if let Some(&other_arg_index) = arg_name_to_index.get(param.id.string.as_str()) {
                        let diag = Diagnostic::new("argument matches positionally but is also passed as named")
                            .add_info(param.id.span, "parameter defined here")
                            .add_info(arg.span, "positional match here")
                            .add_error(args.inner[other_arg_index].span, "named match here")
                            .finish();
                        any_err_params = Err(diags.report(diag));
                        arg_used[other_arg_index] = true;
                    }

                    arg_used[arg_index] = true;
                    Ok(&arg.value)
                } else {
                    // named match
                    match arg_name_to_index.get(param.id.string.as_str()) {
                        Some(&arg_index) => {
                            let arg = &args.inner[arg_index];
                            assert!(arg.name.is_some());
                            assert!(!arg_used[param_index]);
                            arg_used[arg_index] = true;
                            Ok(&arg.value)
                        }
                        None => {
                            let diag = Diagnostic::new(format!("missing argument for parameter `{}`", param.id.string))
                                .add_info(param.id.span, "parameter defined here")
                                .add_error(args.span, "missing argument here")
                                .finish();
                            let e = diags.report(diag);
                            any_err_params = Err(e);
                            Err(e)
                        }
                    }
                };

                if let Ok(param_value) = param_value {
                    // record value into vec
                    param_values_vec.push((param.id.clone(), param_value.inner.clone()));
                }

                // declare param in scope
                let param_var = vars.var_new_immutable_init(
                    &mut ctx.variables,
                    MaybeIdentifier::Identifier(param.id.clone()),
                    param.id.span,
                    param_value.map(|v| v.inner.clone()),
                );
                let entry = DeclaredValueSingle::Value {
                    span: param.id.span,
                    value: ScopedEntry::Named(NamedValue::Variable(param_var)),
                };
                scope.declare_already_checked(param.id.string.clone(), entry);
            };

        fn visit_param_item<'a>(
            visit_param: &mut impl FnMut(&mut CompileItemContext, &mut VariableValues, &mut Scope, &'a Parameter),
            ctx: &mut CompileItemContext,
            vars: &mut VariableValues,
            scope: &mut Scope,
            item: &'a ParameterItem,
        ) -> Result<(), ErrorGuaranteed> {
            match item {
                ParameterItem::Parameter(param) => visit_param(ctx, vars, scope, param),
                ParameterItem::If(if_stmt) => {
                    if let Some(block) = ctx.compile_if_statement_choose_block(&scope, vars, if_stmt)? {
                        for inner_item in &block.items {
                            visit_param_item(visit_param, ctx, vars, scope, inner_item)?;
                        }
                    }
                }
            }
            Ok(())
        }
        for item in &params.items {
            visit_param_item(&mut visit_param, self, vars, &mut scope, item)?;
        }
        any_err_params?;

        let mut any_err_used = Ok(());
        for (i, arg_used) in enumerate(arg_used) {
            if !arg_used {
                let diag = Diagnostic::new("argument did not match any param")
                    .add_info(params.span, "parameters defined here")
                    .add_error(args.inner[i].span, "argument passed here")
                    .finish();
                any_err_used = Err(diags.report(diag));
            }
        }
        any_err_used?;

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
                                Ok(Value::Compile(value.clone())),
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
