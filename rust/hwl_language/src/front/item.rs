use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::{AstRefEnum, AstRefStruct, CompileItemContext, ElaboratedModule};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::front::function::{CapturedScope, FunctionBody, FunctionValue, UserFunctionValue};
use crate::front::scope::{DeclaredValueSingle, ScopedEntry};
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::Type;
use crate::front::value::{CompileValue, Value};
use crate::front::variables::VariableValues;
use crate::syntax::ast::{
    CommonDeclaration, ConditionalItem, ConstDeclaration, EnumDeclaration, EnumVariant, Expression,
    FunctionDeclaration, Identifier, Item, ItemDeclaration, ItemDefInterface, ItemDefModule, MaybeIdentifier,
    ModuleInstanceItem, Parameters, Spanned, StructDeclaration, StructField, TypeDeclaration,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModule};
use crate::syntax::pos::Span;
use indexmap::map::Entry;
use indexmap::IndexMap;
use itertools::Itertools;

pub struct ElaboratedItemParams<I> {
    pub item: I,
    pub params: Option<Vec<(Identifier, CompileValue)>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedItemKey<I> {
    item: I,
    params: Option<Vec<CompileValue>>,
}

impl<I: Copy> ElaboratedItemParams<I> {
    pub fn cache_key(&self) -> ElaboratedItemKey<I> {
        let &ElaboratedItemParams { item, ref params } = self;
        let param_values = params
            .as_ref()
            .map(|params| params.iter().map(|(_, v)| v.clone()).collect());
        ElaboratedItemKey {
            item,
            params: param_values,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FunctionItemBody {
    // TODO change this to be a type alias, or maybe change the others to be ast references too?
    TypeAliasExpr(Box<Expression>),
    Module(AstRefModule),
    Interface(AstRefInterface),
    Struct(AstRefStruct, Vec<ConditionalItem<StructField>>),
    Enum(AstRefEnum, Vec<ConditionalItem<EnumVariant>>),
}

#[derive(Debug)]
pub struct ElaboratedStructInfo {
    pub span_body: Span,
    pub fields: IndexMap<String, (Identifier, Spanned<Type>)>,
}

#[derive(Debug)]
pub struct ElaboratedEnumInfo {
    pub span_body: Span,
    pub variants: IndexMap<String, (Identifier, Option<Spanned<Type>>)>,
}

impl<'s> CompileItemContext<'_, 's> {
    pub fn eval_item_new(&mut self, item: AstRefItem) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let file_scope = self.refs.shared.file_scope(item.file())?;

        let item_ast = &self.refs.fixed.parsed[item];
        let span_short = item_ast.common_info().span_short;
        self.refs.check_should_stop(span_short)?;

        match item_ast {
            Item::Import(item_inner) => {
                let reason = "import items should have been resolved in a separate pass already";
                Err(diags.report_internal_error(item_inner.span, reason))
            }
            Item::Instance(instance) => {
                let &ModuleInstanceItem {
                    span: _,
                    span_keyword,
                    ref module,
                } = instance;
                let mut vars = VariableValues::new_root(&self.variables);
                let _: ElaboratedModule =
                    self.eval_expression_as_module(file_scope, &mut vars, span_keyword, module)?;
                Ok(CompileValue::UNIT)
            }
            Item::CommonDeclaration(decl) => {
                let ItemDeclaration { vis: _, decl } = decl;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_declaration(file_scope, &mut vars, decl)
            }
            Item::Module(module) => {
                let ItemDefModule {
                    span: _,
                    vis: _,
                    id,
                    params,
                    ports: _,
                    body,
                } = module;
                let item = AstRefModule::new_unchecked(item, module);
                let body_span = body.span;

                let body = FunctionItemBody::Module(item);

                let scope = self.refs.shared.file_scope(item.file())?;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_maybe_generic_item(id.span(), body_span, scope, &mut vars, params, body)
            }
            Item::Interface(interface) => {
                let ItemDefInterface {
                    span: _,
                    vis: _,
                    id,
                    params,
                    span_body,
                    port_types: _,
                    views: _,
                } = interface;
                let item = AstRefInterface::new_unchecked(item, interface);

                let body = FunctionItemBody::Interface(item);

                let scope = self.refs.shared.file_scope(item.file())?;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_maybe_generic_item(id.span(), *span_body, scope, &mut vars, params, body)
            }
        }
    }

    pub fn rebuild_params_scope(
        &mut self,
        item: AstRefItem,
        vars: &mut VariableValues,
        params: &Option<Vec<(Identifier, CompileValue)>>,
    ) -> Result<Scope<'s>, ErrorGuaranteed> {
        let file_scope = self.refs.shared.file_scope(item.file())?;
        let full_span = self.refs.fixed.parsed[item].common_info().span_full;

        let mut scope_params = Scope::new_child(full_span, file_scope);
        if let Some(params) = &params {
            for (id, value) in params {
                let var = vars.var_new_immutable_init(
                    &mut self.variables,
                    MaybeIdentifier::Identifier(id.clone()),
                    id.span,
                    Ok(Value::Compile(value.clone())),
                );
                let declared = DeclaredValueSingle::Value {
                    span: id.span,
                    value: ScopedEntry::Named(NamedValue::Variable(var)),
                };
                scope_params.declare_already_checked(id.string.clone(), declared);
            }
        }
        Ok(scope_params)
    }

    pub fn eval_declaration(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclaration,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;

        match decl {
            CommonDeclaration::Type(decl) => {
                let TypeDeclaration {
                    span: _,
                    id,
                    params,
                    body,
                } = decl;
                let body_span = body.span;

                let body = FunctionItemBody::TypeAliasExpr(body.clone());
                self.eval_maybe_generic_item(id.span(), body_span, scope, vars, params, body)
            }
            CommonDeclaration::Const(decl) => {
                let ConstDeclaration { span: _, id, ty, value } = decl;

                let ty = ty
                    .as_ref()
                    .map(|ty| self.eval_expression_as_ty(scope, vars, ty))
                    .transpose();
                let value = self.eval_expression_as_compile(scope, vars, value, "const value");

                let ty = ty?;
                let value = value?;

                // check type
                if let Some(ty) = ty {
                    let reason = TypeContainsReason::Assignment {
                        span_target: id.span(),
                        span_target_ty: ty.span,
                    };
                    check_type_contains_compile_value(diags, reason, &ty.inner, value.as_ref(), true)?;
                };

                Ok(value.inner)
            }
            CommonDeclaration::Struct(decl) => {
                let StructDeclaration {
                    span: _,
                    span_body,
                    id,
                    params,
                    fields,
                } = decl;

                let ast_ref = AstRefStruct(id.span());
                let body = FunctionItemBody::Struct(ast_ref, fields.clone());
                self.eval_maybe_generic_item(id.span(), *span_body, scope, vars, params, body)
            }
            CommonDeclaration::Enum(decl) => {
                let EnumDeclaration {
                    span,
                    id,
                    params,
                    variants,
                } = decl;

                let ast_ref = AstRefEnum(id.span());
                let body = FunctionItemBody::Enum(ast_ref, variants.clone());
                self.eval_maybe_generic_item(id.span(), *span, scope, vars, params, body)
            }
            CommonDeclaration::Function(decl) => {
                let FunctionDeclaration {
                    span: _,
                    id,
                    params,
                    ret_ty,
                    body,
                } = decl;

                let body_inner = FunctionBody::FunctionBodyBlock {
                    body: body.clone(),
                    ret_ty: ret_ty.as_ref().map(|ret_ty| Box::new(ret_ty.clone())),
                };
                let function = UserFunctionValue {
                    decl_span: id.span(),
                    scope_captured: CapturedScope::from_scope(diags, scope, vars)?,
                    params: params.clone(),
                    body: Spanned {
                        span: body.span,
                        inner: body_inner,
                    },
                };
                Ok(CompileValue::Function(FunctionValue::User(function)))
            }
        }
    }

    pub fn eval_and_declare_declaration(
        &mut self,
        scope: &mut Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclaration,
    ) {
        let diags = self.refs.diags;
        let (decl_span, id) = decl.info();
        let entry = self.eval_declaration(scope, vars, decl).map(|v| {
            let var = vars.var_new_immutable_init(&mut self.variables, id.clone(), decl_span, Ok(Value::Compile(v)));
            ScopedEntry::Named(NamedValue::Variable(var))
        });
        scope.maybe_declare(diags, id.as_ref(), entry);
    }

    fn eval_maybe_generic_item(
        &mut self,
        span_decl: Span,
        span_body: Span,
        scope: &Scope,
        vars: &mut VariableValues,
        params: &Option<Parameters>,
        body: FunctionItemBody,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;

        match params {
            None => {
                // eval immediately
                let body = Spanned::new(span_body, &body);
                self.eval_item_function_body(scope, vars, None, body)
            }
            Some(params) => {
                // build function
                let func = UserFunctionValue {
                    decl_span: span_decl,
                    scope_captured: CapturedScope::from_scope(diags, scope, vars)?,
                    params: params.clone(),
                    body: Spanned {
                        span: span_body,
                        inner: FunctionBody::ItemBody(body),
                    },
                };
                Ok(CompileValue::Function(FunctionValue::User(func)))
            }
        }
    }

    pub fn eval_item_function_body(
        &mut self,
        scope_params: &Scope,
        vars: &mut VariableValues,
        params: Option<Vec<(Identifier, CompileValue)>>,
        body: Spanned<&FunctionItemBody>,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        match body.inner {
            FunctionItemBody::TypeAliasExpr(expr) => {
                let result_ty = self.eval_expression_as_ty(scope_params, vars, expr)?.inner;
                Ok(CompileValue::Type(result_ty))
            }
            &FunctionItemBody::Module(item) => {
                let item_params = ElaboratedItemParams { item, params };
                let (result_id, _) = self.refs.elaborate_module(item_params)?;
                Ok(CompileValue::Module(result_id))
            }
            &FunctionItemBody::Interface(item) => {
                let item_params = ElaboratedItemParams { item, params };
                let (result_id, _) = self.refs.elaborate_interface(item_params)?;
                Ok(CompileValue::Interface(result_id))
            }
            &FunctionItemBody::Struct(ast_ref, ref fields) => {
                let item_params = ElaboratedItemParams { item: ast_ref, params };
                let (result_id, result_info) = self.refs.shared.elaborated_structs.elaborate(item_params, |_| {
                    self.elaborate_struct_new(scope_params, vars, body.span, fields)
                })?;

                let fields = result_info
                    .fields
                    .values()
                    .map(|(_, ty)| ty.inner.clone())
                    .collect_vec();
                Ok(CompileValue::Type(Type::Struct(result_id, fields)))
            }
            &FunctionItemBody::Enum(ast_ref, ref variants) => {
                let item_params = ElaboratedItemParams { item: ast_ref, params };
                let (result_id, result_info) = self.refs.shared.elaborated_enums.elaborate(item_params, |_| {
                    self.elaborate_enum_new(scope_params, vars, body.span, variants)
                })?;
                let variants = result_info
                    .variants
                    .values()
                    .map(|(_, ty)| ty.as_ref().map(|ty| ty.inner.clone()))
                    .collect_vec();
                Ok(CompileValue::Type(Type::Enum(result_id, variants)))
            }
        }
    }

    fn elaborate_struct_new(
        &mut self,
        scope_params: &Scope,
        vars: &mut VariableValues,
        span_body: Span,
        fields: &[ConditionalItem<StructField>],
    ) -> Result<ElaboratedStructInfo, ErrorGuaranteed> {
        let diags = self.refs.diags;

        // TODO generalize this indexmap "already defined" structure
        let mut fields_eval = IndexMap::new();

        let mut any_field_err = Ok(());
        let mut visit_field = |s: &mut Self, scope: &mut Scope, vars: &mut VariableValues, field: &StructField| {
            let StructField { span: _, id, ty } = field;

            let ty = s.eval_expression_as_ty(scope, vars, ty)?;

            match fields_eval.entry(id.string.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert((id.clone(), ty));
                }
                Entry::Occupied(entry) => {
                    let diag = Diagnostic::new("duplicate struct field name")
                        .add_info(entry.get().0.span, "previously declared here")
                        .add_error(id.span, "declared again here")
                        .finish();
                    any_field_err = Err(diags.report(diag));
                }
            }

            Ok(())
        };

        let mut scope = Scope::new_child(span_body, scope_params);
        for field in fields {
            self.compile_visit_conditional_items(&mut scope, vars, field, &mut visit_field)?;
        }
        any_field_err?;

        Ok(ElaboratedStructInfo {
            span_body,
            fields: fields_eval,
        })
    }

    fn elaborate_enum_new(
        &mut self,
        scope_params: &Scope,
        vars: &mut VariableValues,
        span_body: Span,
        variants: &[ConditionalItem<EnumVariant>],
    ) -> Result<ElaboratedEnumInfo, ErrorGuaranteed> {
        let diags = self.refs.diags;

        let mut variants_eval = IndexMap::new();
        let mut any_variant_err = Ok(());

        let mut visit_variant = |s: &mut Self, scope: &mut Scope, vars: &mut VariableValues, variant: &EnumVariant| {
            let EnumVariant { span: _, id, content } = variant;

            let content = content
                .as_ref()
                .map(|content| s.eval_expression_as_ty(scope, vars, content))
                .transpose()?;

            match variants_eval.entry(id.string.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert((id.clone(), content));
                }
                Entry::Occupied(entry) => {
                    let diag = Diagnostic::new("duplicate enum variant name")
                        .add_info(entry.get().0.span, "previously declared here")
                        .add_error(id.span, "declared again here")
                        .finish();
                    any_variant_err = Err(diags.report(diag));
                }
            }

            Ok(())
        };

        let mut scope = Scope::new_child(span_body, scope_params);
        for variant in variants {
            self.compile_visit_conditional_items(&mut scope, vars, variant, &mut visit_variant)?;
        }
        any_variant_err?;

        Ok(ElaboratedEnumInfo {
            span_body,
            variants: variants_eval,
        })
    }
}
