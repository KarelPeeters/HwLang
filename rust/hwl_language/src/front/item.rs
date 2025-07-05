use crate::front::check::{check_type_contains_compile_value, TypeContainsReason};
use crate::front::compile::{CompileItemContext, WorkItem};
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::flow::{Flow, FlowCompile, FlowRoot};
use crate::front::function::{CapturedScope, FunctionBody, FunctionValue, UserFunctionValue};
use crate::front::interface::ElaboratedInterfaceInfo;
use crate::front::module::{ElaboratedModuleExternalInfo, ElaboratedModuleInternalInfo};
use crate::front::scope::ScopedEntry;
use crate::front::scope::{NamedValue, Scope};
use crate::front::types::{HardwareType, Type};
use crate::front::value::{CompileValue, Value};
use crate::syntax::ast::{
    CommonDeclaration, CommonDeclarationNamed, CommonDeclarationNamedKind, ConstDeclaration, EnumDeclaration,
    EnumVariant, Expression, ExtraList, FunctionDeclaration, Identifier, Item, ItemDefInterface, ItemDefModuleExternal,
    ItemDefModuleInternal, Parameters, Spanned, StructDeclaration, StructField, TypeDeclaration,
};
use crate::syntax::parsed::{AstRefInterface, AstRefItem, AstRefModuleExternal, AstRefModuleInternal};
use crate::syntax::pos::Span;
use crate::util::big_int::{BigInt, BigUint};
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

    pub fn module_internal_info(&self, elab: ElaboratedModuleInternal) -> &ElaboratedModuleInternalInfo {
        self.elaborated_modules_internal.get(elab)
    }

    pub fn module_external_info(&self, elab: ElaboratedModuleExternal) -> &ElaboratedModuleExternalInfo {
        self.elaborated_modules_external.get(elab)
    }

    pub fn interface_info(&self, elab: ElaboratedInterface) -> &ElaboratedInterfaceInfo {
        self.elaborated_interfaces.get(elab)
    }

    pub fn struct_info(&self, elab: ElaboratedStruct) -> &ElaboratedStructInfo {
        self.elaborated_structs.get(elab)
    }

    pub fn enum_info(&self, elab: ElaboratedEnum) -> &ElaboratedEnumInfo {
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
    id_to_info: ComputeOnceMap<E, DiagResult<F>>,
}

impl<E: Copy + Eq + Hash, F> ElaborateItemArena<E, F> {
    pub fn new() -> Self {
        ElaborateItemArena {
            next_id: AtomicUsize::new(0),
            key_to_id: ComputeOnceMap::new(),
            id_to_info: ComputeOnceMap::new(),
        }
    }

    pub fn get(&self, id: E) -> &F {
        // The key only gets out if the computation was successful,
        //   so we can safely unwrap here (twice).
        self.id_to_info.get(&id).unwrap().as_ref_ok().unwrap()
    }

    pub fn elaborate(
        &self,
        params: ElaboratedItemParams,
        e: impl FnOnce(usize) -> E,
        f: impl FnOnce(ElaboratedItemParams) -> DiagResult<F>,
    ) -> DiagResult<(E, &F)> {
        let key = params.cache_key();

        let &id = self.key_to_id.get_or_compute(key, |_| {
            let index = self.next_id.fetch_add(1, Ordering::Relaxed);
            let id = e(index);

            let info = f(params);

            self.id_to_info.set(id, info).unwrap();
            id
        });

        let info = self.id_to_info.get(&id).unwrap().as_ref_ok()?;
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

/// Newtype wrapper that promises that the fields are representable in hardware.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct HardwareChecked<T> {
    inner: T,
}

impl<T: Copy> HardwareChecked<T> {
    pub fn new_unchecked(inner: T) -> Self {
        HardwareChecked { inner }
    }

    pub fn inner(self) -> T {
        self.inner
    }
}

#[derive(Debug)]
pub struct ElaboratedStructInfo {
    pub unique: UniqueDeclaration,
    pub span_body: Span,
    pub fields: IndexMap<String, (Identifier, Spanned<Type>)>,
    pub fields_hw: Result<Vec<HardwareType>, NonHardwareStruct>,
}

#[derive(Debug, Copy, Clone)]
pub struct NonHardwareStruct {
    pub first_failing_field: usize,
}

#[derive(Debug)]
pub struct ElaboratedEnumInfo {
    pub unique: UniqueDeclaration,
    pub span_body: Span,
    pub variants: IndexMap<String, (Identifier, Option<Spanned<Type>>)>,
    pub hw: Result<HardwareEnumInfo, NonHardwareEnum>,
}

#[derive(Debug, Copy, Clone)]
pub struct NonHardwareEnum {
    pub first_failing_variant: usize,
}

#[derive(Debug)]
pub struct HardwareEnumInfo {
    pub content_types: Vec<Option<HardwareType>>,
    // TODO remove once this (or something similar enough) is cached in HardwareType
    pub max_content_size: usize,
}

impl ElaboratedEnumInfo {
    pub fn find_variant(&self, diags: &Diagnostics, variant: Spanned<&str>) -> DiagResult<usize> {
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
    pub fn eval_item_new(&mut self, item: AstRefItem) -> DiagResult<CompileValue> {
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
                let flow_root = FlowRoot::new(diags);
                let mut flow = FlowCompile::new_root(&flow_root, decl.span, "item declaration");
                let value = self.eval_declaration(file_scope, &mut flow, &decl.inner)?;
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
                let flow_root = FlowRoot::new(diags);
                let mut flow = FlowCompile::new_root(&flow_root, module.span, "item declaration");
                self.eval_maybe_generic_item(id.span(), body_span, scope, &mut flow, params, body)
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
                let flow_root = FlowRoot::new(diags);
                let mut flow = FlowCompile::new_root(&flow_root, module.span, "item declaration");
                self.eval_maybe_generic_item(id.span, body_span, scope, &mut flow, params, body)
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
                let flow_root = FlowRoot::new(diags);
                let mut flow = FlowCompile::new_root(&flow_root, interface.span, "item declaration");
                self.eval_maybe_generic_item(id.span(), *span_body, scope, &mut flow, params, body)
            }
        }
    }

    pub fn eval_declaration<'f, V>(
        &mut self,
        scope: &Scope,
        flow: &mut FlowCompile,
        decl: &CommonDeclaration<V>,
    ) -> DiagResult<Option<CompileValue>> {
        match decl {
            CommonDeclaration::Named(decl) => {
                let CommonDeclarationNamed { vis: _, kind } = decl;
                self.eval_declaration_named(scope, flow, kind).map(Some)
            }
            CommonDeclaration::ConstBlock(decl) => {
                self.elaborate_const_block(scope, flow, decl)?;
                Ok(None)
            }
        }
    }

    pub fn eval_declaration_named(
        &mut self,
        scope: &Scope,
        flow: &mut impl Flow,
        decl: &CommonDeclarationNamedKind,
    ) -> DiagResult<CompileValue> {
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
                self.eval_maybe_generic_item(id.span(), body_span, scope, flow, params, body)
            }
            CommonDeclarationNamedKind::Const(decl) => {
                let &ConstDeclaration { span: _, id, ty, value } = decl;

                let ty = ty.map(|ty| self.eval_expression_as_ty(scope, flow, ty)).transpose()?;

                let expected_ty = ty.as_ref().map_or(&Type::Any, |ty| &ty.inner);
                let value = self.eval_expression_as_compile(scope, flow, expected_ty, value, "const value")?;

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
                self.eval_maybe_generic_item(id.span(), *span_body, scope, flow, params, body)
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
                self.eval_maybe_generic_item(id.span(), *span, scope, flow, params, body)
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
                    scope_captured: CapturedScope::from_scope(scope, flow),
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
        flow: &mut impl Flow,
        decl: &CommonDeclaration<()>,
    ) {
        let diags = self.refs.diags;

        match decl {
            CommonDeclaration::Named(decl) => {
                // eval and declare
                let CommonDeclarationNamed { vis: _, kind } = decl;
                let decl_id = kind.id();
                let decl_span = kind.span();

                let entry = self.eval_declaration_named(scope, flow, kind).map(|v| {
                    let var = flow.var_new_immutable_init(decl_id, decl_span, Ok(Value::Compile(v)));
                    ScopedEntry::Named(NamedValue::Variable(var))
                });
                scope.maybe_declare(diags, Ok(decl_id.spanned_str(self.refs.fixed.source)), entry);
            }
            CommonDeclaration::ConstBlock(decl) => {
                // elaborate, don't declare anything
                let _ = self.elaborate_const_block(scope, flow, decl);
            }
        }
    }

    fn eval_maybe_generic_item(
        &mut self,
        span_decl: Span,
        span_body: Span,
        scope: &Scope,
        flow: &mut impl Flow,
        params: &Option<Parameters>,
        body: FunctionItemBody,
    ) -> DiagResult<CompileValue> {
        let diags = self.refs.diags;

        match params {
            None => {
                // eval immediately
                let body = Spanned::new(span_body, &body);
                let mut flow = flow.new_child_compile(span_decl, "item body");
                self.eval_item_function_body(scope, &mut flow, None, body)
            }
            Some(params) => {
                // build function
                let func = UserFunctionValue {
                    decl_span: span_decl,
                    scope_captured: CapturedScope::from_scope(scope, flow),
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
        flow: &mut FlowCompile,
        params: Option<Vec<(Identifier, CompileValue)>>,
        body: Spanned<&FunctionItemBody>,
    ) -> DiagResult<CompileValue> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        match *body.inner {
            FunctionItemBody::TypeAliasExpr(expr) => {
                let result_ty = self.eval_expression_as_ty(scope_params, flow, expr)?.inner;
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
                        let scope_captured = CapturedScope::from_scope(scope_params, flow);

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
                        let scope_captured = CapturedScope::from_scope(scope_params, flow);
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
                let scope_captured = CapturedScope::from_scope(scope_params, flow);

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

                let (result_id, _) = self.refs.shared.elaboration_arenas.elaborated_structs.elaborate(
                    item_params,
                    ElaboratedStruct,
                    |_| self.elaborate_struct_new(scope_params, flow, unique, body.span, fields),
                )?;
                Ok(CompileValue::Type(Type::Struct(result_id)))
            }
            FunctionItemBody::Enum(unique, ref variants) => {
                let item_params = ElaboratedItemParams { unique, params };

                let (result_id, _) = self.refs.shared.elaboration_arenas.elaborated_enums.elaborate(
                    item_params,
                    ElaboratedEnum,
                    |_| self.elaborate_enum_new(scope_params, flow, unique, body.span, variants),
                )?;

                Ok(CompileValue::Type(Type::Enum(result_id)))
            }
        }
    }

    fn elaborate_struct_new(
        &mut self,
        scope_params: &Scope,
        flow: &mut FlowCompile,
        unique: UniqueDeclaration,
        span_body: Span,
        fields: &ExtraList<StructField>,
    ) -> DiagResult<ElaboratedStructInfo> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        // TODO generalize this indexmap "already defined" structure
        let mut fields_eval = IndexMap::new();

        let mut any_field_err = Ok(());
        let mut visit_field = |s: &mut Self, scope: &mut Scope, flow: &mut FlowCompile, field: &StructField| {
            let &StructField { span: _, id, ty } = field;

            let ty = s.eval_expression_as_ty(scope, flow, ty)?;

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
        self.compile_elaborate_extra_list(&mut scope, flow, fields, &mut visit_field)?;
        any_field_err?;

        // check if this struct can be represented in hardware
        //   we do this once now instead of each time we need to know this
        let fields_hw = fields_eval
            .iter()
            .enumerate()
            .map(|(i, (_, (_, ty)))| {
                ty.inner
                    .as_hardware_type(self.refs)
                    .map_err(|_| NonHardwareStruct { first_failing_field: i })
            })
            .try_collect_vec();

        Ok(ElaboratedStructInfo {
            unique,
            span_body,
            fields: fields_eval,
            fields_hw,
        })
    }

    fn elaborate_enum_new(
        &mut self,
        scope_params: &Scope,
        flow: &mut FlowCompile,
        unique: UniqueDeclaration,
        span_body: Span,
        variants: &ExtraList<EnumVariant>,
    ) -> DiagResult<ElaboratedEnumInfo> {
        let diags = self.refs.diags;
        let source = self.refs.fixed.source;

        let mut variants_eval = IndexMap::new();
        let mut any_variant_err = Ok(());

        let mut visit_variant = |s: &mut Self, scope: &mut Scope, flow: &mut FlowCompile, variant: &EnumVariant| {
            let &EnumVariant { span: _, id, content } = variant;

            let content = content
                .map(|content| s.eval_expression_as_ty(scope, flow, content))
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
        self.compile_elaborate_extra_list(&mut scope, flow, variants, &mut visit_variant)?;
        any_variant_err?;

        // check if this enum can be represented in hardware
        //   we do this once now instead of each time we need to know this
        let mut size_err = Ok(());
        let hw = variants_eval
            .iter()
            .enumerate()
            .map(|(i, (_, (_, ty)))| {
                ty.as_ref()
                    .map(|ty| ty.inner.as_hardware_type(self.refs))
                    .transpose()
                    .map_err(|_| NonHardwareEnum {
                        first_failing_variant: i,
                    })
            })
            .try_collect_vec()
            .and_then(|content_types| {
                let max_content_size = content_types
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(|ty| ty.size_bits(self.refs))
                    .max()
                    .unwrap_or(BigUint::ZERO);
                let max_content_size = usize::try_from(max_content_size).map_err(|size| {
                    size_err = Err(diags.report_simple("enum size too large", span_body, format!("got size {size}")));
                    NonHardwareEnum {
                        first_failing_variant: 0,
                    }
                })?;
                Ok(HardwareEnumInfo {
                    content_types,
                    max_content_size,
                })
            });

        size_err?;

        Ok(ElaboratedEnumInfo {
            unique,
            span_body,
            variants: variants_eval,
            hw,
        })
    }
}
