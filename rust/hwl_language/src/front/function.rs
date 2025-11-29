use crate::front::block::{BlockEnd, EarlyExitKind};
use crate::front::check::{TypeContainsReason, check_type_contains_value, check_type_is_bool_array};
use crate::front::compile::{CompileItemContext, CompileRefs, StackEntry};
use crate::front::diagnostic::{DiagError, DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::exit::{ExitFlag, ExitStack, ReturnEntry, ReturnEntryHardware, ReturnEntryKind};
use crate::front::flow::{CapturedValue, FailedCaptureReason, FlowKind, VariableId, VariableInfo};
use crate::front::flow::{Flow, FlowCompile};
use crate::front::item::{ElaboratedEnum, ElaboratedStruct, ElaboratedStructInfo, FunctionItemBody, UniqueDeclaration};
use crate::front::scope::{DeclaredValueSingle, Scope, ScopeParent};
use crate::front::scope::{NamedValue, ScopedEntry};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{
    CompileValue, EnumValue, HardwareValue, MaybeCompile, MixedCompoundValue, NotCompile, SimpleCompileValue,
    StructValue, Value, ValueCommon,
};
use crate::mid::ir::{IrExpressionLarge, IrLargeArena};
use crate::syntax::ast::{
    Arg, Args, Block, BlockStatement, Expression, Identifier, MaybeIdentifier, Parameter, Parameters,
};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::syntax::source::{FileId, SourceDatabase};
use crate::util::data::VecExt;
use crate::util::{ResultDoubleExt, ResultExt};
use annotate_snippets::Level;
use indexmap::IndexMap;
use indexmap::map::Entry;
use itertools::{Itertools, enumerate};
use std::collections::HashMap;
use std::collections::hash_map::Entry as HashMapEntry;
use std::hash::Hash;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum FunctionValue {
    User(Arc<UserFunctionValue>),
    Bits(FunctionBits),
    StructNew(ElaboratedStruct),
    StructNewInfer(UniqueDeclaration),
    EnumNew(ElaboratedEnum, usize),
    EnumNewInfer(UniqueDeclaration, Arc<String>),
}

#[derive(Debug, Clone)]
pub struct FunctionBits {
    pub ty_hw: HardwareType,
    pub kind: FunctionBitsKind,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum FunctionBitsKind {
    ToBits,
    FromBits,
}

// TODO find a better name for this
#[derive(Debug, Clone)]
pub struct UserFunctionValue {
    // only used for uniqueness
    // TODO switch to newer UniqueDeclaration?
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
        ret_ty: Option<Expression>,
    },
    ItemBody(FunctionItemBody),
}

impl FunctionBody {
    pub fn params_must_be_compile(&self) -> bool {
        match self {
            // TODO add optional marker to functions that can only be used with compare args
            FunctionBody::FunctionBodyBlock { .. } => false,
            FunctionBody::ItemBody(_) => true,
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
// TODO allow capturing hardware values, eg. for functions defined in module bodies or in hardware blocks
/// The parent scope is kept separate to avoid a hard dependency on all items that are in scope,
///   now capturing functions still allow graph-based item evaluation.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct CapturedScope {
    root_file: FileId,

    /// Sorted by name, to get some extra determinism and cache key hits.
    captured_values: Vec<(String, DiagResult<Spanned<CapturedValue>>)>,
}

#[must_use]
pub struct ParamArgMacher<'a> {
    // constant initial values
    diags: &'a Diagnostics,
    source: &'a SourceDatabase,
    args: &'a Args<Option<Spanned<&'a str>>, Spanned<Value>>,
    arg_name_to_index: IndexMap<&'a str, usize>,
    positional_count: usize,
    params_span: Span,

    // mutable state
    next_param_index: usize,
    arg_used: Vec<bool>,
    param_names: IndexMap<&'a str, Span>,

    any_err: DiagResult,
}

#[derive(Debug, Copy, Clone)]
enum NamedRule {
    OnlyNamed,
    OnlyPositional,
    PositionalAndNamed,
}

// TODO make generic over the "value", this should be reused for instance connections in the future
impl<'a> ParamArgMacher<'a> {
    fn new(
        diags: &'a Diagnostics,
        source: &'a SourceDatabase,
        params_span: Span,
        args: &'a Args<Option<Spanned<&'a str>>, Spanned<Value>>,
        args_must_be_compile: bool,
        args_must_be_named: NamedRule,
    ) -> DiagResult<Self> {
        // check for duplicate arg names and check that positional args are before named args
        let mut arg_name_to_index: IndexMap<&str, usize> = IndexMap::new();
        let mut first_named_span = None;
        let mut positional_count: usize = 0;
        let mut any_err_args = Ok(());
        for (arg_index, arg) in enumerate(&args.inner) {
            if args_must_be_compile && CompileValue::try_from(&arg.value.inner).is_err() {
                let diag = Diagnostic::new("call target only supports compile-time arguments")
                    .add_info(params_span, "parameters defined here")
                    .add_error(arg.value.span, "hardware value passed here")
                    .finish();
                any_err_args = Err(diags.report(diag));
            }

            match (&arg.name, args_must_be_named) {
                (None, NamedRule::OnlyNamed) => {
                    let diag = Diagnostic::new("positional arguments are not allowed for this call target")
                        .add_info(params_span, "parameters defined here")
                        .add_error(arg.span, "positional argument passed here")
                        .finish();
                    any_err_args = Err(diags.report(diag));
                }
                (Some(_), NamedRule::OnlyPositional) => {
                    let diag = Diagnostic::new("named arguments are not allowed for this call target")
                        .add_info(params_span, "parameters defined here")
                        .add_error(arg.span, "named argument passed here")
                        .finish();
                    any_err_args = Err(diags.report(diag));
                }

                (None, NamedRule::OnlyPositional) => {}
                (Some(_), NamedRule::OnlyNamed) => {}
                (_, NamedRule::PositionalAndNamed) => {}
            }

            match &arg.name {
                Some(id) => {
                    match arg_name_to_index.entry(id.inner) {
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

        Ok(Self {
            diags,
            source,
            args,
            positional_count,
            arg_name_to_index,
            params_span,
            next_param_index: 0,
            arg_used: vec![false; args.inner.len()],
            param_names: IndexMap::new(),
            any_err: Ok(()),
        })
    }

    pub fn resolve_param(
        &mut self,
        id: Identifier,
        ty: Spanned<&Type>,
        default: Option<Spanned<Value>>,
    ) -> DiagResult<Spanned<Value>> {
        let diags = self.diags;
        let id_str = id.str(self.source);

        let param_index = self.next_param_index;
        self.next_param_index += 1;

        if let Some(prev_span) = self.param_names.insert(id_str, id.span) {
            let diag = Diagnostic::new("duplicate parameter name")
                .add_info(prev_span, "previously defined here")
                .add_error(id.span, "defined again here")
                .finish();
            let e = diags.report(diag);
            self.any_err = Err(e);
            return Err(e);
        }

        let value = if param_index < self.positional_count {
            // positional match
            let arg_index = param_index;
            let arg = &self.args.inner[arg_index];
            assert!(arg.name.is_none());
            assert!(!self.arg_used[arg_index]);

            // check if there's also a named match to get better error messages
            if let Some(&other_arg_index) = self.arg_name_to_index.get(id_str) {
                let diag = Diagnostic::new("argument matches positionally but is also passed as named")
                    .add_info(id.span, "parameter defined here")
                    .add_info(arg.span, "positional match here")
                    .add_error(self.args.inner[other_arg_index].span, "named match here")
                    .finish();
                let err = diags.report(diag);
                self.any_err = Err(err);
                self.arg_used[other_arg_index] = true;
            }

            self.arg_used[arg_index] = true;
            Ok(arg.value.clone())
        } else {
            match self.arg_name_to_index.get(id_str) {
                Some(&arg_index) => {
                    // named match
                    let arg = &self.args.inner[arg_index];
                    assert!(arg.name.is_some());
                    assert!(!self.arg_used[arg_index]);
                    self.arg_used[arg_index] = true;
                    Ok(arg.value.clone())
                }
                None => {
                    if let Some(default) = default {
                        // default value
                        Ok(default)
                    } else {
                        // nothing matched, report error
                        let diag = Diagnostic::new(format!("missing argument for parameter `{id_str}`"))
                            .add_info(id.span, "parameter defined without default value here")
                            .add_error(self.args.span, "missing argument here")
                            .finish();
                        let e = diags.report(diag);
                        self.any_err = Err(e);
                        Err(e)
                    }
                }
            }
        };

        // check type match
        let value = value.and_then(|value| {
            let reason = TypeContainsReason::Parameter { param_ty: ty.span };
            check_type_contains_value(diags, reason, ty.inner, value.as_ref())?;
            Ok(value)
        });

        if let Err(e) = value {
            self.any_err = Err(e);
        }

        value
    }

    pub fn finish(self) -> DiagResult {
        let diags = self.diags;
        self.any_err?;

        let mut any_err_used = Ok(());
        for (i, arg_used) in enumerate(self.arg_used) {
            if !arg_used {
                let diag = Diagnostic::new("argument did not match any param")
                    .add_info(self.params_span, "parameters defined here")
                    .add_error(self.args.inner[i].span, "argument passed here")
                    .finish();
                any_err_used = Err(diags.report(diag));
            }
        }

        any_err_used?;
        Ok(())
    }
}

impl CompileItemContext<'_, '_> {
    pub fn call_function(
        &mut self,
        flow: &mut impl Flow,
        expected_ty: &Type,
        span_target: Span,
        span_call: Span,
        function: &FunctionValue,
        args: Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        let err_infer_any = |kind: &str| {
            let diag = Diagnostic::new(format!("cannot infer {kind} params"))
                .add_info(span_call, "no expected type")
                .add_error(span_target, format!("this {kind} has unbound generic parameters"))
                .footer(
                    Level::Help,
                    "either set an expected type or use use the full type before calling new",
                )
                .finish();
            self.refs.diags.report(diag)
        };
        let err_infer_mismatch = |kind: &str, actual_span: Span| {
            let diag = Diagnostic::new("mismatching expected type")
                .add_info(
                    span_call,
                    format!("non-{kind} expected type {:?}", expected_ty.diagnostic_string()),
                )
                .add_error(actual_span, format!("{kind} type set here"))
                .finish();
            self.refs.diags.report(diag)
        };

        match function {
            FunctionValue::User(function) => self.call_user_function(flow, function, args),
            FunctionValue::Bits(function) => self.call_bits_function(span_call, function, args),
            &FunctionValue::StructNew(struct_elab) => self.call_struct_new(span_call, struct_elab, args),
            &FunctionValue::StructNewInfer(func_unique) => match *expected_ty {
                Type::Struct(expected_elab) => {
                    let expected_info = self.refs.shared.elaboration_arenas.struct_info(expected_elab);
                    if expected_info.unique == func_unique {
                        self.call_struct_new(span_call, expected_elab, args)
                    } else {
                        Err(diags.report(error_unique_mismatch(
                            "struct",
                            span_target,
                            expected_info.unique.id().span(),
                            func_unique.id().span(),
                        )))
                    }
                }
                Type::Any => Err(err_infer_any("struct")),
                _ => Err(err_infer_mismatch("struct", func_unique.id().span())),
            },
            &FunctionValue::EnumNew(enum_elab, variant_index) => {
                self.call_enum_new(span_call, enum_elab, variant_index, &args)
            }
            &FunctionValue::EnumNewInfer(unique, ref variant_str) => match expected_ty {
                &Type::Enum(elab) => {
                    let info = self.refs.shared.elaboration_arenas.enum_info(elab);
                    let variant_index = info.find_variant(diags, Spanned::new(span_target, variant_str))?;
                    self.call_enum_new(span_call, elab, variant_index, &args)
                }
                Type::Any => Err(err_infer_any("enum")),
                _ => Err(err_infer_mismatch("enum", unique.id().span())),
            },
        }
    }

    pub fn call_function_compile(
        &mut self,
        flow: &mut FlowCompile,
        expected_ty: &Type,
        span_target: Span,
        span_call: Span,
        function: &FunctionValue,
        args: Args<Option<Spanned<&str>>, Spanned<CompileValue>>,
    ) -> DiagResult<CompileValue> {
        let args = Args {
            span: args.span,
            inner: args
                .inner
                .into_iter()
                .map(|arg| Arg {
                    span: arg.span,
                    name: arg.name,
                    value: arg.value.map_inner(Value::from),
                })
                .collect_vec(),
        };

        let result = self.call_function(flow, expected_ty, span_target, span_call, function, args)?;

        CompileValue::try_from(&result).map_err(|_: NotCompile| {
            self.refs.diags.report_internal_error(
                span_call,
                "calling a function with compile-time args should return a compile-time value, got hardware value",
            )
        })
    }

    // TODO ensure the expected type for fields is correctly propagated to the args
    //   (this might need a major re-think, currently args are always evaluated in advance)
    fn call_struct_new(
        &mut self,
        span_call: Span,
        elab: ElaboratedStruct,
        args: Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let _ = span_call;
        let &ElaboratedStructInfo {
            unique: _,
            name: _,
            span_body,
            ref fields,
            fields_hw: _,
        } = self.refs.shared.elaboration_arenas.struct_info(elab);

        let mut matcher = ParamArgMacher::new(
            self.refs.diags,
            self.refs.fixed.source,
            span_body,
            &args,
            false,
            NamedRule::OnlyNamed,
        )?;

        let mut field_values = vec![];
        for &(field_id, ref field_ty) in fields.values() {
            if let Ok(v) = matcher.resolve_param(field_id, field_ty.as_ref(), None) {
                field_values.push(v.inner);
            }
        }
        matcher.finish()?;

        let result = StructValue {
            ty: elab,
            fields: field_values,
        };
        Ok(Value::Compound(MixedCompoundValue::Struct(result)))
    }

    fn call_enum_new(
        &mut self,
        span_call: Span,
        elab: ElaboratedEnum,
        variant_index: usize,
        args: &Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        let enum_info = self.refs.shared.elaboration_arenas.enum_info(elab);
        let &(variant_id, ref variant_content) = &enum_info.variants[variant_index];
        let variant_payload_ty = variant_content.as_ref().unwrap();

        let mut matcher = ParamArgMacher::new(
            diags,
            self.refs.fixed.source,
            span_call,
            args,
            false,
            NamedRule::OnlyPositional,
        )?;
        let payload = matcher.resolve_param(variant_id, variant_payload_ty.as_ref(), None)?;
        matcher.finish()?;

        let result = EnumValue {
            ty: elab,
            variant: variant_index,
            payload: Some(Box::new(payload.inner)),
        };
        Ok(Value::Compound(MixedCompoundValue::Enum(result)))
    }

    fn call_user_function(
        &mut self,
        flow: &mut impl Flow,
        function: &UserFunctionValue,
        args: Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        let UserFunctionValue {
            decl_span,
            scope_captured,
            params,
            body,
        } = function;
        let decl_span = *decl_span;

        self.refs.check_should_stop(decl_span)?;

        // recreate captured scope
        let span_scope = params.span.join(body.span);
        let scope_captured = scope_captured.to_scope(self.refs, flow, span_scope);

        // map params into scope
        let mut scope = Scope::new_child(span_scope, &scope_captured);
        let mut param_values = vec![];

        let compile = body.inner.params_must_be_compile();
        let mut matcher = ParamArgMacher::new(
            diags,
            source,
            params.span,
            &args,
            compile,
            NamedRule::PositionalAndNamed,
        )?;

        self.compile_elaborate_extra_list(&mut scope, flow, &params.items, &mut |ctx, scope, flow, param| {
            let &Parameter {
                span: _,
                id,
                ty,
                default,
            } = param;

            let ty = ctx.eval_expression_as_ty(scope, flow, ty)?;
            let default = default
                .as_ref()
                .map(|&default| {
                    let value =
                        ctx.eval_expression_as_compile(scope, flow, &ty.inner, default, "parameter default value")?;
                    Ok(value.map_inner(Value::from))
                })
                .transpose()?;

            let value = matcher.resolve_param(id, ty.as_ref(), default);

            // record value into vec
            if let Ok(value) = &value {
                param_values.push((param.id, value.inner.clone()));
            }

            // declare param in scope
            let param_var = flow.var_new_immutable_init(
                param.id.span,
                VariableId::Id(MaybeIdentifier::Identifier(param.id)),
                param.id.span,
                value.map(|v| v.inner),
            );
            let entry = DeclaredValueSingle::Value {
                span: param.id.span,
                value: ScopedEntry::Named(NamedValue::Variable(param_var)),
            };
            scope.declare_already_checked(param.id.str(source).to_owned(), entry);

            Ok(())
        })?;
        matcher.finish()?;

        // run the body
        let entry = StackEntry::FunctionRun(decl_span);
        self.recurse(entry, |s| {
            match &body.inner {
                FunctionBody::FunctionBodyBlock { body, ret_ty } => {
                    // evaluate return type
                    let return_type = ret_ty
                        .map(|ret_ty| s.eval_expression_as_ty(&scope, flow, ret_ty))
                        .transpose()?;

                    // set up the stack
                    let return_entry_kind = match flow.kind_mut() {
                        FlowKind::Compile(_) => ReturnEntryKind::Compile,
                        FlowKind::Hardware(flow) => {
                            let return_flag = ExitFlag::new(flow, decl_span, EarlyExitKind::Return);
                            ReturnEntryKind::Hardware(ReturnEntryHardware { return_flag })
                        }
                    };
                    let return_var = if let Some(return_type) = &return_type
                        && !return_type.inner.is_unit()
                    {
                        let return_var_info = VariableInfo {
                            span_decl: decl_span,
                            id: VariableId::Custom("return_value"),
                            mutable: false,
                            ty: None,
                            use_ir_variable: None,
                        };
                        let return_var = flow.var_new(return_var_info);

                        // As far as the flow is concerned,
                        //   it might look like not all branches are guaranteed to initialize the return value.
                        // To avoid wrong error messages and skipped merging, we always start with an initial value.
                        flow.var_set_undefined(return_var, decl_span);

                        Some(return_var)
                    } else {
                        None
                    };
                    let return_entry = ReturnEntry {
                        span_function_decl: decl_span,
                        return_type: return_type.as_ref().map(Spanned::as_ref),
                        return_var,
                        kind: return_entry_kind,
                    };
                    let mut stack = ExitStack::new_in_function(return_entry);

                    // evaluate block
                    let end = s.elaborate_block(&scope, flow, &mut stack, body)?;

                    // check end and extract return value
                    let return_entry = stack.return_info_option().unwrap();
                    check_function_end(diags, flow, &mut s.large, body.span, return_entry, end)
                }
                FunctionBody::ItemBody(item_body) => {
                    // unwrap compile, we checked that these values are compile-time during argument matching
                    let param_values = param_values
                        .into_iter()
                        .map(|(id, v)| (id, CompileValue::try_from(&v).unwrap()))
                        .collect_vec();

                    let mut flow_inner = flow.new_child_compile(body.span, "item body");
                    let value = s.eval_item_function_body(
                        &scope,
                        &mut flow_inner,
                        Some(param_values),
                        Spanned::new(body.span, item_body),
                    )?;
                    Ok(Value::from(value))
                }
            }
        })
        .flatten_err()
    }

    fn call_bits_function(
        &mut self,
        span_call: Span,
        function: &FunctionBits,
        args: Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;

        // check arg is single non-named value
        // TODO use new common arg-matching machinery
        let value = match args.inner.single() {
            Ok(Arg { span: _, name, value }) => {
                if let Some(name) = name {
                    return Err(diags.report_simple(
                        "this function expects a single unnamed parameter",
                        name.span,
                        "got named argument here",
                    ));
                }
                value
            }
            Err(_) => {
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
                )?;

                let ty_ir = ty_hw.as_ir(self.refs);
                let width = ty_ir.size_bits();

                // try as compile-time first so we get compile-time bits back
                match CompileValue::try_from(&value.inner) {
                    Ok(value) => {
                        let bits = ty_hw.value_to_bits(self.refs, span_call, &value)?;
                        let bits_wrapped = bits.into_iter().map(CompileValue::new_bool).collect_vec();
                        Ok(Value::Simple(SimpleCompileValue::Array(Arc::new(bits_wrapped))))
                    }
                    Err(NotCompile) => {
                        let value = value.inner.as_hardware_value_unchecked(
                            self.refs,
                            &mut self.large,
                            span_call,
                            ty_hw.clone(),
                        )?;
                        let ty_bits = HardwareType::Array(Arc::new(HardwareType::Bool), width);
                        let bits_hw = HardwareValue {
                            ty: ty_bits,
                            domain: value.domain,
                            expr: self.large.push_expr(IrExpressionLarge::ToBits(ty_ir, value.expr)),
                        };
                        Ok(Value::Hardware(bits_hw))
                    }
                }
            }
            FunctionBitsKind::FromBits => {
                let ty_ir = ty_hw.as_ir(self.refs);
                let width = ty_ir.size_bits();

                let value =
                    check_type_is_bool_array(diags, TypeContainsReason::Operator(span_call), value, Some(&width))?;

                let result = match value {
                    MaybeCompile::Compile(v) => {
                        let result = ty_hw.value_from_bits(self.refs, span_call, &v).map_err(|_| {
                            let msg = format!(
                                "while converting value `{:?}` into type `{}`",
                                v,
                                ty_hw.diagnostic_string()
                            );
                            diags.report_simple("`from_bits` failed", span_call, msg)
                        })?;
                        Value::from(result)
                    }
                    MaybeCompile::Hardware(v) => {
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

pub fn check_function_return_type_and_set_value(
    diags: &Diagnostics,
    flow: &mut impl Flow,
    entry: &ReturnEntry,
    span_stmt: Span,
    span_keyword: Span,
    value: Option<Spanned<Value>>,
) -> DiagResult {
    let ty = entry.return_type;

    match (ty, value) {
        (None, None) => {}
        (Some(ty), Some(value)) => {
            let reason = TypeContainsReason::Return {
                span_keyword,
                span_return_ty: ty.span,
            };
            let result_ty = check_type_contains_value(diags, reason, ty.inner, value.as_ref());

            if let Some(return_var) = entry.return_var {
                flow.var_set(return_var, span_stmt, result_ty.map(|()| value.inner));
            }

            result_ty?;
        }
        (Some(ty), None) => {
            let diag = Diagnostic::new("missing return value in function with return type")
                .add_error(span_keyword, "return without value here")
                .add_info(
                    ty.span,
                    format!("function return type `{}` declared here", ty.inner.diagnostic_string()),
                )
                .finish();
            return Err(diags.report(diag));
        }
        (None, Some(value)) => {
            let diag = Diagnostic::new("return value in function without return type")
                .add_error(value.span, "return value here")
                .add_info(entry.span_function_decl, "function declared without return type here")
                .finish();
            return Err(diags.report(diag));
        }
    }

    Ok(())
}

fn check_function_end(
    diags: &Diagnostics,
    flow: &mut impl Flow,
    large: &mut IrLargeArena,
    body_span: Span,
    return_entry: &ReturnEntry,
    end: BlockEnd,
) -> DiagResult<Value> {
    // some of these should be impossible, but checking again here is redundant
    let is_certain_return = match end {
        BlockEnd::CompileExit(end) => match end {
            EarlyExitKind::Return => true,
            EarlyExitKind::Break => false,
            EarlyExitKind::Continue => false,
        },
        BlockEnd::Normal => false,
        BlockEnd::HardwareExit(_) => false,
        BlockEnd::HardwareMaybeExit(_) => false,
    };

    let value = if is_certain_return {
        if let Some(var) = return_entry.return_var {
            // normal return, get the value
            flow.var_eval(diags, large, Spanned::new(body_span, var))
                .map_err(|_: DiagError| diags.report_internal_error(body_span, "failed to evaluate return value"))?
        } else {
            // normal return with unit return type, return unit
            Value::unit()
        }
    } else {
        // the end of the body might be reachable, this is only okay for functions without a return type
        match return_entry.return_type {
            None => Value::unit(),
            Some(return_type) => {
                let diag = Diagnostic::new("missing return in function")
                    .add_error(Span::empty_at(body_span.end()), "end of function is reached here")
                    .add_info(
                        return_type.span,
                        format!("return type `{}` declared here", return_type.inner.diagnostic_string()),
                    )
                    .finish();
                return Err(diags.report(diag));
            }
        }
    };
    Ok(value.into_value())
}

impl CapturedScope {
    pub fn from_file_scope(file: FileId) -> CapturedScope {
        CapturedScope {
            root_file: file,
            captured_values: vec![],
        }
    }

    pub fn from_scope(scope: &Scope, flow: &impl Flow) -> CapturedScope {
        // it's fine to use a hashmap here, this will be sorted into a BTreeMap later
        let mut captured_values = HashMap::new();

        // walk up scopes, starting from the current scope up to the root
        //   try to capture all values that have not yet been shadowed by a child scope
        let mut curr = scope;
        let root_file = loop {
            match curr.parent() {
                ScopeParent::Some(parent) => {
                    // this is a non-root scope, capture it
                    for (id, value) in curr.immediate_entries() {
                        let child_values_entry = match captured_values.entry(id.to_owned()) {
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
                                        &NamedValue::Variable(var) => flow.var_capture(Spanned::new(span, var)),
                                        NamedValue::Port(_)
                                        | NamedValue::Wire(_)
                                        | NamedValue::Register(_)
                                        | NamedValue::PortInterface(_)
                                        | NamedValue::WireInterface(_) => {
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

        // sort captured values
        // TODO maybe we don't need to sort, they already have a deterministic order anyway
        let mut captured_values = captured_values.into_iter().collect_vec();
        captured_values.sort_by(|a, b| a.0.cmp(&b.0));

        CapturedScope {
            root_file,
            captured_values,
        }
    }

    pub fn to_scope<'s>(&self, refs: CompileRefs<'_, 's>, flow: &mut impl Flow, scope_span: Span) -> Scope<'s> {
        let CapturedScope {
            root_file,
            captured_values,
        } = self;

        let parent_file = refs
            .shared
            .file_scopes
            .get(root_file)
            .unwrap()
            .as_ref_ok()
            .expect("file scope should be valid since the capturing succeeded");
        let mut scope = Scope::new_child(scope_span, parent_file);

        // TODO we need a span, even for errors
        for (id, value) in captured_values {
            let declared = match value {
                Ok(value) => {
                    let span = value.span;
                    match &value.inner {
                        &CapturedValue::Item(item) => DeclaredValueSingle::Value {
                            span,
                            value: ScopedEntry::Item(item),
                        },
                        CapturedValue::Value(value) => {
                            // TODO this can be simplified, identifiers can be stored by value now
                            let id_recreated = MaybeIdentifier::Identifier(Identifier { span });
                            let var = flow.var_new_immutable_init(
                                id_recreated.span(),
                                VariableId::Id(id_recreated),
                                span,
                                Ok(Value::from(value.clone())),
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

        scope
    }
}

impl Eq for FunctionValue {}

impl FunctionValue {
    pub fn equality_key(&self) -> impl Eq + Hash + '_ {
        #[derive(Eq, PartialEq, Hash)]
        enum Key<'a> {
            User(Span, &'a CapturedScope),
            Bits(&'a HardwareType, FunctionBitsKind),
            StructNew(ElaboratedStruct),
            StructNewInfer(UniqueDeclaration),
            EnumNew(ElaboratedEnum, usize),
            EnumNewInfer(UniqueDeclaration, &'a str),
        }

        match self {
            FunctionValue::User(func) => {
                let UserFunctionValue {
                    decl_span,
                    scope_captured,
                    params,
                    body,
                } = &**func;

                // these are both derivable from the decl_span, so redundant
                // TODO actually remove them from the struct
                let _ = (params, body);
                Key::User(*decl_span, scope_captured)
            }
            FunctionValue::Bits(FunctionBits { ty_hw: ty, kind }) => Key::Bits(ty, *kind),
            FunctionValue::StructNew(elab) => Key::StructNew(*elab),
            FunctionValue::StructNewInfer(ref_struct) => Key::StructNewInfer(*ref_struct),
            FunctionValue::EnumNew(elab, index) => Key::EnumNew(*elab, *index),
            FunctionValue::EnumNewInfer(ref_struct, variant) => Key::EnumNewInfer(*ref_struct, variant),
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

pub fn error_unique_mismatch(kind: &str, target_span: Span, expected_span: Span, actual_span: Span) -> Diagnostic {
    // TODO include struct/enum name
    Diagnostic::new(format!("{kind} expected type mismatch"))
        .add_error(target_span, format!("actual {kind} type is set here"))
        .add_info(expected_span, format!("expected {kind} type declared here"))
        .add_info(actual_span, format!("actual {kind} type defined here"))
        .finish()
}
