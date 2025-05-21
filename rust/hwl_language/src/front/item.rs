use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::{CompileItemContext, WorkItem};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::function::{CapturedScope, FunctionBody, FunctionValue, UserFunctionValue};
use crate::front::interface::ElaboratedInterfaceInfo;
use crate::front::module::{ElaboratedModuleExternalInfo, ElaboratedModuleInternalInfo};
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::Type;
use crate::front::value::{CompileValue, Value};
use crate::front::variables::VariableValues;
use crate::syntax::ast::{
    CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration, EnumDeclaration,
    EnumVariant, Expression, ExtraList, FunctionDeclaration, Identifier, Item, ItemDefInterface, ItemDefModuleExternal,
    ItemDefModuleInternal, Parameters, Spanned, StructDeclaration, StructField, TypeDeclaration,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModuleExternal, AstRefModuleInternal};
use crate::syntax::pos::Span;
use crate::util::big_int::BigInt;
use crate::util::iter::IterExt;
use crate::util::sync::ComputeOnceMap;
use crate::util::ResultExt;
use indexmap::map::Entry;
use indexmap::IndexMap;
use itertools::Itertools;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ElaboratedModule<I = ElaboratedModuleInternal, E = ElaboratedModuleExternal> {
    Internal(I),
    External(E),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedModuleInternal(usize);
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedModuleExternal(usize);
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedInterface(usize);
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedStruct(usize);
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedEnum(usize);

pub struct ElaborationArenas {
    elaborated_modules_internal: ElaborateItemArena<ElaboratedModuleInternal, ElaboratedModuleInternalInfo>,
    elaborated_modules_external: ElaborateItemArena<ElaboratedModuleExternal, ElaboratedModuleExternalInfo>,
    elaborated_interfaces: ElaborateItemArena<ElaboratedInterface, ElaboratedInterfaceInfo>,
    elaborated_structs: ElaborateItemArena<ElaboratedStruct, ElaboratedStructInfo>,
    elaborated_enums: ElaborateItemArena<ElaboratedEnum, ElaboratedEnumInfo>,
    next_unique_declaration: AtomicUsize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct UniqueDeclaration(usize, Span);

impl UniqueDeclaration {
    pub fn span_id(&self) -> Span {
        self.1
    }
}

impl ElaborationArenas {
    pub fn new() -> Self {
        ElaborationArenas {
            elaborated_modules_internal: ElaborateItemArena::new(),
            elaborated_modules_external: ElaborateItemArena::new(),
            elaborated_interfaces: ElaborateItemArena::new(),
            elaborated_structs: ElaborateItemArena::new(),
            elaborated_enums: ElaborateItemArena::new(),
            next_unique_declaration: AtomicUsize::new(0),
        }
    }

    pub fn module_internal_info(
        &self,
        elab: ElaboratedModuleInternal,
    ) -> Result<&ElaboratedModuleInternalInfo, ErrorGuaranteed> {
        self.elaborated_modules_internal.get(elab)
    }

    pub fn module_external_info(
        &self,
        elab: ElaboratedModuleExternal,
    ) -> Result<&ElaboratedModuleExternalInfo, ErrorGuaranteed> {
        self.elaborated_modules_external.get(elab)
    }

    pub fn interface_info(&self, elab: ElaboratedInterface) -> Result<&ElaboratedInterfaceInfo, ErrorGuaranteed> {
        self.elaborated_interfaces.get(elab)
    }

    pub fn struct_info(&self, elab: ElaboratedStruct) -> Result<&ElaboratedStructInfo, ErrorGuaranteed> {
        self.elaborated_structs.get(elab)
    }

    pub fn enum_info(&self, elab: ElaboratedEnum) -> Result<&ElaboratedEnumInfo, ErrorGuaranteed> {
        self.elaborated_enums.get(elab)
    }

    fn next_unique_declaration(&self, span_id: Span) -> UniqueDeclaration {
        let id = self.next_unique_declaration.fetch_add(1, Ordering::Relaxed);
        assert!(id < usize::MAX / 2, "(close to) overflowing");
        UniqueDeclaration(id, span_id)
    }
}

pub struct ElaborateItemArena<E, F> {
    next_id: AtomicUsize,
    key_to_id: ComputeOnceMap<ElaboratedItemKey, E>,
    id_to_info: ComputeOnceMap<E, Result<F, ErrorGuaranteed>>,
}

impl<E: Copy + Eq + Hash, F> ElaborateItemArena<E, F> {
    pub fn new() -> Self {
        ElaborateItemArena {
            next_id: AtomicUsize::new(0),
            key_to_id: ComputeOnceMap::new(),
            id_to_info: ComputeOnceMap::new(),
        }
    }

    pub fn get(&self, id: E) -> Result<&F, ErrorGuaranteed> {
        self.id_to_info.get(&id).unwrap().as_ref_ok()
    }

    pub fn elaborate(
        &self,
        params: ElaboratedItemParams,
        e: impl FnOnce(usize) -> E,
        f: impl FnOnce(ElaboratedItemParams) -> Result<F, ErrorGuaranteed>,
    ) -> Result<(E, &F), ErrorGuaranteed> {
        let key = params.cache_key();

        let &id = self.key_to_id.get_or_compute(key, |_| {
            let index = self.next_id.fetch_add(1, Ordering::Relaxed);
            let id = e(index);

            let info = f(params);

            self.id_to_info.set(id, info).unwrap();
            id
        });

        let info = self.get(id)?;
        Ok((id, info))
    }
}

pub struct ElaboratedItemParams {
    // TODO maybe remove this field?
    pub unique: UniqueDeclaration,
    pub params: Option<Vec<(Identifier, CompileValue)>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ElaboratedItemKey {
    unique: UniqueDeclaration,
    params: Option<Vec<CompileValue>>,
}

impl ElaboratedItemParams {
    pub fn cache_key(&self) -> ElaboratedItemKey {
        let &ElaboratedItemParams { unique, ref params } = self;
        let param_values = params
            .as_ref()
            .map(|params| params.iter().map(|(_, v)| v.clone()).collect());
        ElaboratedItemKey {
            unique,
            params: param_values,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FunctionItemBody {
    // TODO change this to be a type alias, or maybe change the others to be ast references too?
    TypeAliasExpr(Expression),
    ModuleInternal(UniqueDeclaration, AstRefModuleInternal),
    ModuleExternal(UniqueDeclaration, AstRefModuleExternal),
    Interface(UniqueDeclaration, AstRefInterface),
    Struct(UniqueDeclaration, ExtraList<StructField>),
    Enum(UniqueDeclaration, ExtraList<EnumVariant>),
}

#[derive(Debug)]
pub struct ElaboratedStructInfo {
    pub unique: UniqueDeclaration,
    pub span_body: Span,
    pub fields: IndexMap<String, (Identifier, Spanned<Type>)>,
}

#[derive(Debug)]
pub struct ElaboratedEnumInfo {
    pub unique: UniqueDeclaration,
    pub span_body: Span,
    pub variants: IndexMap<String, (Identifier, Option<Spanned<Type>>)>,
}

impl ElaboratedEnumInfo {
    pub fn find_variant(&self, diags: &Diagnostics, variant: Spanned<&str>) -> Result<usize, ErrorGuaranteed> {
        self.variants.get_index_of(variant.inner).ok_or_else(|| {
            let diag = Diagnostic::new(format!("variant `{}` not found on enum", variant.inner))
                .add_error(variant.span, "attempt to access variant here")
                .add_info(self.span_body, "enum variants declared here")
                .finish();
            diags.report(diag)
        })
    }
}

impl CompileItemContext<'_, '_> {
    pub fn eval_item_new(&mut self, item: AstRefItem) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let file_scope = self.refs.shared.file_scope(item.file())?;

        let item_ast = &self.refs.fixed.parsed[item];
        self.refs.check_should_stop(item_ast.info().span_short)?;

        match item_ast {
            Item::Import(item_inner) => {
                let reason = "import items should have been resolved in a separate pass already";
                Err(diags.report_internal_error(item_inner.span, reason))
            }
            Item::CommonDeclaration(decl) => {
                let mut vars = VariableValues::new_root(&self.variables);
                let value = self.eval_declaration(file_scope, &mut vars, &decl.inner)?;
                Ok(value.unwrap_or_else(CompileValue::unit))
            }
            Item::ModuleInternal(module) => {
                let ItemDefModuleInternal {
                    span: _,
                    vis: _,
                    id,
                    params,
                    ports,
                    body,
                } = module;
                let item = AstRefModuleInternal::new_unchecked(item, module);
                let body_span = ports.span.join(body.span);

                let unique = self.refs.shared.elaboration_arenas.next_unique_declaration(id.span());
                let body = FunctionItemBody::ModuleInternal(unique, item);

                let scope = self.refs.shared.file_scope(item.file())?;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_maybe_generic_item(id.span(), body_span, scope, &mut vars, params, body)
            }
            Item::ModuleExternal(module) => {
                let ItemDefModuleExternal {
                    span: _,
                    span_ext: _,
                    vis: _,
                    id,
                    params,
                    ports,
                } = module;
                let item = AstRefModuleExternal::new_unchecked(item, module);
                let body_span = ports.span;

                let unique = self.refs.shared.elaboration_arenas.next_unique_declaration(id.span);
                let body = FunctionItemBody::ModuleExternal(unique, item);

                let scope = self.refs.shared.file_scope(item.file())?;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_maybe_generic_item(id.span, body_span, scope, &mut vars, params, body)
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

                let unique = self.refs.shared.elaboration_arenas.next_unique_declaration(id.span());
                let body = FunctionItemBody::Interface(unique, item);

                let scope = self.refs.shared.file_scope(item.file())?;
                let mut vars = VariableValues::new_root(&self.variables);
                self.eval_maybe_generic_item(id.span(), *span_body, scope, &mut vars, params, body)
            }
        }
    }

    pub fn eval_declaration<V>(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclaration<V>,
    ) -> Result<Option<CompileValue>, ErrorGuaranteed> {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis: _, kind } = decl;
                self.eval_declaration_named(scope, vars, kind).map(Some)
            }
            CommonDeclaration::ConstBlock(decl) => {
                self.elaborate_const_block(scope, vars, decl)?;
                Ok(None)
            }
        }
    }

    pub fn eval_declaration_named(
        &mut self,
        scope: &Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclarationNamedKind,
    ) -> Result<CompileValue, ErrorGuaranteed> {
        let diags = self.refs.diags;

        match decl {
            CommonDeclarationNamedKind::Type(decl) => {
                let &TypeDeclaration {
                    span: _,
                    id,
                    ref params,
                    body,
                } = decl;
                let body_span = body.span;

                let body = FunctionItemBody::TypeAliasExpr(body);
                self.eval_maybe_generic_item(id.span(), body_span, scope, vars, params, body)
            }
            CommonDeclarationNamedKind::Const(decl) => {
                let &ConstDeclaration { span: _, id, ty, value } = decl;

                let ty = ty.map(|ty| self.eval_expression_as_ty(scope, vars, ty)).transpose()?;

                let expected_ty = ty.as_ref().map_or(&Type::Any, |ty| &ty.inner);
                let value = self.eval_expression_as_compile(scope, vars, expected_ty, value, "const value")?;

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
            CommonDeclarationNamedKind::Struct(decl) => {
                let StructDeclaration {
                    span: _,
                    span_body,
                    id,
                    params,
                    fields,
                } = decl;

                let unique = self.refs.shared.elaboration_arenas.next_unique_declaration(id.span());
                let body = FunctionItemBody::Struct(unique, fields.clone());
                self.eval_maybe_generic_item(id.span(), *span_body, scope, vars, params, body)
            }
            CommonDeclarationNamedKind::Enum(decl) => {
                let EnumDeclaration {
                    span,
                    id,
                    params,
                    variants,
                } = decl;

                let unique = self.refs.shared.elaboration_arenas.next_unique_declaration(id.span());
                let body = FunctionItemBody::Enum(unique, variants.clone());
                self.eval_maybe_generic_item(id.span(), *span, scope, vars, params, body)
            }
            CommonDeclarationNamedKind::Function(decl) => {
                let &FunctionDeclaration {
                    span: _,
                    id,
                    ref params,
                    ret_ty,
                    ref body,
                } = decl;

                let body_inner = FunctionBody::FunctionBodyBlock {
                    body: body.clone(),
                    ret_ty,
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
                Ok(CompileValue::Function(FunctionValue::User(Arc::new(function))))
            }
        }
    }

    pub fn eval_and_declare_declaration(
        &mut self,
        scope: &mut Scope,
        vars: &mut VariableValues,
        decl: &CommonDeclaration<()>,
    ) {
        let diags = self.refs.diags;

        match decl {
            CommonDeclaration::Named(decl) => {
                // eval and declare
                let CommonDeclarationNamed { vis: _, kind } = decl;
                let decl_id = kind.id();
                let decl_span = kind.span();

                let entry = self.eval_declaration_named(scope, vars, kind).map(|v| {
                    let var =
                        vars.var_new_immutable_init(&mut self.variables, decl_id, decl_span, Ok(Value::Compile(v)));
                    ScopedEntry::Named(NamedValue::Variable(var))
                });
                scope.maybe_declare(diags, Ok(decl_id.spanned_str(self.refs.fixed.source)), entry);
            }
            CommonDeclaration::ConstBlock(decl) => {
                // elaborate, don't declare anything
                let _ = self.elaborate_const_block(scope, vars, decl);
            }
        }
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
                Ok(CompileValue::Function(FunctionValue::User(Arc::new(func))))
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
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        match *body.inner {
            FunctionItemBody::TypeAliasExpr(expr) => {
                let result_ty = self.eval_expression_as_ty(scope_params, vars, expr)?.inner;
                Ok(CompileValue::Type(result_ty))
            }
            FunctionItemBody::ModuleInternal(unique, ast_ref) => {
                let item_params = ElaboratedItemParams { unique, params };
                let refs = self.refs;

                let (result_id, _) = refs.shared.elaboration_arenas.elaborated_modules_internal.elaborate(
                    item_params,
                    ElaboratedModuleInternal,
                    |item_params| {
                        // elaborate ports
                        let scope_captured = CapturedScope::from_scope(diags, scope_params, vars)?;

                        let ast = &refs.fixed.parsed[ast_ref];

                        let (connectors, header) = refs.elaborate_module_ports_new(
                            ast_ref,
                            ast.span,
                            item_params,
                            scope_captured,
                            &ast.ports,
                        )?;

                        // reserve ir module key, will be filled in later during body elaboration
                        let ir_module = { refs.shared.ir_database.lock().unwrap().ir_modules.push(None) };

                        // queue body elaboration for later
                        refs.shared
                            .work_queue
                            .push(WorkItem::ElaborateModuleBody(header, ir_module));

                        Ok(ElaboratedModuleInternalInfo {
                            unique,
                            ast_ref,
                            module_ir: ir_module,
                            connectors,
                        })
                    },
                )?;

                Ok(CompileValue::Module(ElaboratedModule::Internal(result_id)))
            }
            FunctionItemBody::ModuleExternal(unique, ast_ref) => {
                let item_params = ElaboratedItemParams { unique, params };
                let refs = self.refs;
                let ast = &refs.fixed.parsed[ast_ref];

                let (result_id, _) = refs.shared.elaboration_arenas.elaborated_modules_external.elaborate(
                    item_params,
                    ElaboratedModuleExternal,
                    |item_params| {
                        // save generic args for later
                        let generic_args = item_params
                            .params
                            .as_ref()
                            .map(|item_params| {
                                item_params
                                    .iter()
                                    .map(|(id, value)| {
                                        let value = match value {
                                            &CompileValue::Bool(value) => {
                                                if value {
                                                    BigInt::ONE
                                                } else {
                                                    BigInt::ZERO
                                                }
                                            }
                                            CompileValue::Int(value) => value.clone(),
                                            _ => {
                                                return Err(diags.report_todo(
                                                    ast.params.as_ref().map_or(ast.span, |p| p.span),
                                                    "external module generic parameters that are not bool or int",
                                                ))
                                            }
                                        };

                                        Ok((id.str(source).to_owned(), value))
                                    })
                                    .try_collect_all_vec()
                            })
                            .transpose()?;

                        // elaborate ports
                        let scope_captured = CapturedScope::from_scope(diags, scope_params, vars)?;
                        let (connectors, header) = refs.elaborate_module_ports_new(
                            ast_ref,
                            ast.span,
                            item_params,
                            scope_captured,
                            &ast.ports,
                        )?;
                        let port_names = header
                            .ports_ir
                            .values()
                            .map(|port_info| port_info.name.clone())
                            .collect_vec();

                        Ok(ElaboratedModuleExternalInfo {
                            ast_ref,
                            module_name: ast.id.str(source).to_owned(),
                            generic_args,
                            port_names,
                            connectors,
                        })
                    },
                )?;

                Ok(CompileValue::Module(ElaboratedModule::External(result_id)))
            }
            FunctionItemBody::Interface(unique, ast_ref) => {
                let item_params = ElaboratedItemParams { unique, params };
                let scope_captured = CapturedScope::from_scope(diags, scope_params, vars)?;

                let refs = self.refs;
                let (result_id, _) = refs.shared.elaboration_arenas.elaborated_interfaces.elaborate(
                    item_params,
                    ElaboratedInterface,
                    |_| refs.elaborate_interface_new(ast_ref, scope_captured),
                )?;

                Ok(CompileValue::Interface(result_id))
            }
            FunctionItemBody::Struct(unique, ref fields) => {
                let item_params = ElaboratedItemParams { unique, params };

                let (result_id, result_info) = self.refs.shared.elaboration_arenas.elaborated_structs.elaborate(
                    item_params,
                    ElaboratedStruct,
                    |_| self.elaborate_struct_new(scope_params, vars, unique, body.span, fields),
                )?;

                let fields = result_info
                    .fields
                    .values()
                    .map(|(_, ty)| ty.inner.clone())
                    .collect_vec();
                Ok(CompileValue::Type(Type::Struct(result_id, fields)))
            }
            FunctionItemBody::Enum(unique, ref variants) => {
                let item_params = ElaboratedItemParams { unique, params };

                let (result_id, result_info) = self.refs.shared.elaboration_arenas.elaborated_enums.elaborate(
                    item_params,
                    ElaboratedEnum,
                    |_| self.elaborate_enum_new(scope_params, vars, unique, body.span, variants),
                )?;

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
        unique: UniqueDeclaration,
        span_body: Span,
        fields: &ExtraList<StructField>,
    ) -> Result<ElaboratedStructInfo, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        // TODO generalize this indexmap "already defined" structure
        let mut fields_eval = IndexMap::new();

        let mut any_field_err = Ok(());
        let mut visit_field = |s: &mut Self, scope: &mut Scope, vars: &mut VariableValues, field: &StructField| {
            let &StructField { span: _, id, ty } = field;

            let ty = s.eval_expression_as_ty(scope, vars, ty)?;

            match fields_eval.entry(id.str(source).to_owned()) {
                Entry::Vacant(entry) => {
                    entry.insert((id, ty));
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
        self.compile_elaborate_extra_list(&mut scope, vars, fields, &mut visit_field)?;
        any_field_err?;

        Ok(ElaboratedStructInfo {
            unique,
            span_body,
            fields: fields_eval,
        })
    }

    fn elaborate_enum_new(
        &mut self,
        scope_params: &Scope,
        vars: &mut VariableValues,
        unique: UniqueDeclaration,
        span_body: Span,
        variants: &ExtraList<EnumVariant>,
    ) -> Result<ElaboratedEnumInfo, ErrorGuaranteed> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        let mut variants_eval = IndexMap::new();
        let mut any_variant_err = Ok(());

        let mut visit_variant = |s: &mut Self, scope: &mut Scope, vars: &mut VariableValues, variant: &EnumVariant| {
            let &EnumVariant { span: _, id, content } = variant;

            let content = content
                .map(|content| s.eval_expression_as_ty(scope, vars, content))
                .transpose()?;

            match variants_eval.entry(id.str(source).to_owned()) {
                Entry::Vacant(entry) => {
                    entry.insert((id, content));
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
        self.compile_elaborate_extra_list(&mut scope, vars, variants, &mut visit_variant)?;
        any_variant_err?;

        Ok(ElaboratedEnumInfo {
            unique,
            span_body,
            variants: variants_eval,
        })
    }
}
