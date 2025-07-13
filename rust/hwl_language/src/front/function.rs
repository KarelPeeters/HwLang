use crate::front::block::{BlockEnd, BlockEndReturn};
use crate::front::check::{check_type_contains_value, check_type_is_bool_array, TypeContainsReason};
use crate::front::compile::{CompileItemContext, CompileRefs, StackEntry};
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::domain::ValueDomain;
use crate::front::flow::Flow;
use crate::front::flow::{CapturedValue, FailedCaptureReason};
use crate::front::item::{
    ElaboratedEnum, ElaboratedStruct, ElaboratedStructInfo, FunctionItemBody, HardwareChecked, NonHardwareEnum,
    NonHardwareStruct, UniqueDeclaration,
};
use crate::front::scope::{DeclaredValueSingle, Scope, ScopeParent};
use crate::front::scope::{NamedValue, ScopedEntry};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, HardwareValue, Value};
use crate::mid::ir::IrExpressionLarge;
use crate::syntax::ast::{
    Arg, Args, Block, BlockStatement, Expression, Identifier, MaybeIdentifier, Parameter, Parameters, Spanned,
};
use crate::syntax::pos::Span;
use crate::syntax::source::{FileId, SourceDatabase};
use crate::util::data::VecExt;
use crate::util::{ResultDoubleExt, ResultExt};
use annotate_snippets::Level;
use indexmap::map::Entry;
use indexmap::IndexMap;
use itertools::{enumerate, Itertools};
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use unwrap_match::unwrap_match;

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

    any_err: DiagResult<()>,
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
            if args_must_be_compile && !matches!(arg.value.inner, Value::Compile(_)) {
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
                        let diag = Diagnostic::new(format!("missing argument for parameter `{}`", id_str))
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
            check_type_contains_value(diags, reason, ty.inner, value.as_ref(), false, false)?;
            Ok(value)
        });

        if let Err(e) = value {
            self.any_err = Err(e);
        }

        value
    }

    pub fn finish(self) -> DiagResult<()> {
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
                            expected_info.unique.span_id(),
                            func_unique.span_id(),
                        )))
                    }
                }
                Type::Any => Err(err_infer_any("struct")),
                _ => Err(err_infer_mismatch("struct", func_unique.span_id())),
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
                _ => Err(err_infer_mismatch("enum", unique.span_id())),
            },
        }
    }

    // TODO ensure the expected type for fields is correctly propagated to the args
    //   (this might need a major re-think, currently args are always evaluated in advance)
    fn call_struct_new(
        &mut self,
        span_call: Span,
        elab: ElaboratedStruct,
        args: Args<Option<Spanned<&str>>, Spanned<Value>>,
    ) -> DiagResult<Value> {
        let diags = self.refs.diags;
        let &ElaboratedStructInfo {
            span_body,
            unique: _,
            ref fields,
            ref fields_hw,
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
                field_values.push(v);
            }
        }
        matcher.finish()?;

        // decide between hardware/compile
        // combine into compile or non-compile value
        // TODO share this code with tuple and array literals
        let first_non_compile = field_values
            .iter()
            .find(|v| !matches!(v.inner, Value::Compile(_)))
            .map(|v| v.span);

        let result = if let Some(first_non_compile) = first_non_compile {
            // at least one non-compile, turn everything into IR
            let fields_hw = match fields_hw {
                Ok(fields_hw) => fields_hw,
                &Err(NonHardwareStruct { first_failing_field }) => {
                    let (field_id, field_ty) = &fields[first_failing_field];
                    let diag = Diagnostic::new("cannot construct hardware value of struct")
                        .add_error(span_call, "during construction of struct here")
                        .add_info(first_non_compile, "necessary because this field value is hardware")
                        .add_info(field_id.span, "field declared here")
                        .add_info(
                            field_ty.span,
                            format!("with non-hardware type `{}`", field_ty.inner.diagnostic_string()),
                        )
                        .finish();
                    return Err(diags.report(diag));
                }
            };
            let elab_hw = HardwareChecked::new_unchecked(elab);

            let mut result_ty = vec![];
            let mut result_domain = ValueDomain::CompileTime;
            let mut result_expr = vec![];

            for (i, value) in enumerate(field_values) {
                let expected_ty_inner_hw = &fields_hw[i];

                let value_ir =
                    value
                        .inner
                        .as_hardware_value(self.refs, &mut self.large, value.span, expected_ty_inner_hw)?;

                result_ty.push(value_ir.ty);
                result_domain = result_domain.join(value_ir.domain);
                result_expr.push(value_ir.expr);
            }

            Value::Hardware(HardwareValue {
                ty: HardwareType::Struct(elab_hw),
                domain: result_domain,
                expr: self.large.push_expr(IrExpressionLarge::TupleLiteral(result_expr)),
            })
        } else {
            // all compile
            let values = field_values
                .into_iter()
                .map(|v| unwrap_match!(&v.inner, Value::Compile(v) => v.clone()))
                .collect();
            Value::Compile(CompileValue::Struct(elab, Arc::new(values)))
        };
        Ok(result)
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
        let variant_content = variant_content.as_ref().unwrap();

        let mut matcher = ParamArgMacher::new(
            diags,
            self.refs.fixed.source,
            span_call,
            args,
            false,
            NamedRule::OnlyPositional,
        )?;
        let content = matcher.resolve_param(variant_id, variant_content.as_ref(), None)?;
        matcher.finish()?;

        let result = match content.inner {
            Value::Compile(content) => Value::Compile(CompileValue::Enum(
                elab,
                (variant_index, Some(Box::new(content.clone()))),
            )),
            Value::Hardware(content_inner) => {
                let enum_info_hw = match &enum_info.hw {
                    Ok(hw_info) => hw_info,
                    &Err(NonHardwareEnum { first_failing_variant }) => {
                        let (variant_id, variant_content) = &enum_info.variants[first_failing_variant];
                        let variant_content = variant_content.as_ref().unwrap();

                        let diag = Diagnostic::new("cannot construct hardware value of struct")
                            .add_error(span_call, "during construction of struct here")
                            .add_info(content.span, "necessary because this content value is hardware")
                            .add_info(variant_id.span, "variant declared here")
                            .add_info(
                                variant_content.span,
                                format!("with non-hardware type `{}`", variant_content.inner.diagnostic_string()),
                            )
                            .finish();
                        return Err(diags.report(diag));
                    }
                };
                let ty_hw = HardwareChecked::new_unchecked(elab);

                // build new expression
                let content_bits = self.large.push_expr(IrExpressionLarge::ToBits(
                    content_inner.ty.as_ir(self.refs),
                    content_inner.expr.clone(),
                ));
                let expr =
                    enum_info_hw.build_ir_expression(self.refs, &mut self.large, variant_index, Some(content_bits))?;

                Value::Hardware(HardwareValue {
                    ty: HardwareType::Enum(ty_hw),
                    domain: content_inner.domain,
                    expr,
                })
            }
        };

        Ok(result)
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
            let &Parameter { id, ty, default } = param;

            let ty = ctx.eval_expression_as_ty(scope, flow, ty)?;
            let default = default
                .as_ref()
                .map(|&default| {
                    let value =
                        ctx.eval_expression_as_compile(scope, flow, &ty.inner, default, "parameter default value")?;
                    Ok(value.map_inner(Value::Compile))
                })
                .transpose()?;

            let value = matcher.resolve_param(id, ty.as_ref(), default);

            // record value into vec
            if let Ok(value) = &value {
                param_values.push((param.id, value.inner.clone()));
            }

            // declare param in scope
            let param_var = flow.var_new_immutable_init(
                MaybeIdentifier::Identifier(param.id),
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
                    let ret_ty = ret_ty
                        .map(|ret_ty| s.eval_expression_as_ty(&scope, flow, ret_ty))
                        .transpose()?;

                    // evaluate block
                    let ty_unit = Type::unit();
                    let expected_ret_ty = ret_ty.as_ref().map_or(&ty_unit, |ty| &ty.inner);
                    let end = s.elaborate_block(&scope, flow, Some(expected_ret_ty), body)?;

                    // check return type and extract value
                    check_function_return_value(diags, body.span, &ret_ty, end)
                }
                FunctionBody::ItemBody(item_body) => {
                    // unwrap compile, we checked that these values are compile-time during argument matching
                    let param_values = param_values
                        .into_iter()
                        .map(|(id, v)| (id, v.unwrap_compile()))
                        .collect_vec();

                    let mut flow_inner = flow.new_child_compile(body.span, "item body");
                    let value = s.eval_item_function_body(
                        &scope,
                        &mut flow_inner,
                        Some(param_values),
                        Spanned::new(body.span, item_body),
                    )?;
                    Ok(Value::Compile(value))
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

                let ty_ir = ty_hw.as_ir(self.refs);
                let width = ty_ir.size_bits();

                let result = match &value.inner {
                    Value::Compile(value) => {
                        // TODO dedicated compile-time bits value that's faster than a boxed array of bools
                        let bits = ty_hw.value_to_bits(self.refs, span_call, value)?;
                        Value::Compile(CompileValue::Array(Arc::new(
                            bits.into_iter().map(CompileValue::Bool).collect_vec(),
                        )))
                    }
                    Value::Hardware(value_raw) => {
                        let value = value_raw.clone().soft_expand_to_type(&mut self.large, ty_hw);

                        let expr = self
                            .large
                            .push_expr(IrExpressionLarge::ToBits(ty_ir, value.expr.clone()));
                        let ty_bits = HardwareType::Array(Arc::new(HardwareType::Bool), width);

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
                let ty_ir = ty_hw.as_ir(self.refs);
                let width = ty_ir.size_bits();

                let value =
                    check_type_is_bool_array(diags, TypeContainsReason::Operator(span_call), value, Some(&width))?;

                let result = match value {
                    Value::Compile(v) => {
                        Value::Compile(ty_hw.value_from_bits(self.refs, span_call, &v).map_err(|_| {
                            let msg = format!(
                                "while converting value `{:?}` into type `{}`",
                                v,
                                ty_hw.diagnostic_string()
                            );
                            diags.report_simple("`from_bits` failed", span_call, msg)
                        })?)
                    }
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
) -> DiagResult<Value> {
    match end.unwrap_normal_or_return_in_function(diags)? {
        BlockEnd::Normal => {
            // no return, only allowed for unit-returning functions
            match ret_ty {
                None => Ok(Value::Compile(CompileValue::unit())),
                Some(ret_ty) => {
                    if ret_ty.inner.is_unit() {
                        Ok(Value::Compile(CompileValue::unit()))
                    } else {
                        let diag = Diagnostic::new("control flow reaches end of function with return type")
                            .add_error(Span::empty_at(body_span.end()), "end of function is reached here")
                            .add_info(
                                ret_ty.span,
                                format!("return type `{}` declared here", ret_ty.inner.diagnostic_string()),
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
                (None, None) => Ok(Value::Compile(CompileValue::unit())),
                (Some(ret_ty), None) => {
                    if ret_ty.inner.is_unit() {
                        Ok(Value::Compile(CompileValue::unit()))
                    } else {
                        let diag = Diagnostic::new("missing return value in function with return type")
                            .add_error(span_keyword, "return here without value")
                            .add_info(
                                ret_ty.span,
                                format!(
                                    "function return type `{}` declared here",
                                    ret_ty.inner.diagnostic_string()
                                ),
                            )
                            .finish();
                        Err(diags.report(diag))
                    }
                }
                (None, Some(ret_value)) => {
                    let as_unit = match ret_value.inner {
                        Value::Compile(v) => {
                            if v.is_unit() {
                                Some(v)
                            } else {
                                None
                            }
                        }
                        Value::Hardware(_) => None,
                    };
                    if let Some(unit) = as_unit {
                        Ok(Value::Compile(unit))
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
                                            Ok(CapturedValue::FailedCapture(FailedCaptureReason::Hardware))
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

    pub fn to_scope<'s, 'f>(&self, refs: CompileRefs<'_, 's>, flow: &mut impl Flow, scope_span: Span) -> Scope<'s> {
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
                        CapturedValue::Value(ref value) => {
                            // TODO this can be simplified, identifiers can be stored by value now
                            let id_recreated = MaybeIdentifier::Identifier(Identifier { span });
                            let var =
                                flow.var_new_immutable_init(id_recreated, span, Ok(Value::Compile(value.clone())));
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
    Diagnostic::new(format!("{kind} expected type mismatch"))
        .add_error(target_span, format!("actual {kind} type is set here"))
        .add_info(expected_span, format!("expected {kind} type declared here"))
        .add_info(actual_span, format!("actual {kind} type defined here"))
        .finish()
}
