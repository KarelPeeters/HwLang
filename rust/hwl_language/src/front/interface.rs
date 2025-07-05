use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, Diagnostic, DiagnosticAddable, Diagnostics};
use crate::front::flow::{FlowCompile, FlowRoot};
use crate::front::function::CapturedScope;
use crate::front::scope::Scope;
use crate::front::types::HardwareType;
use crate::syntax::ast::{Identifier, InterfaceView, ItemDefInterface, MaybeIdentifier, PortDirection, Spanned};
use crate::syntax::parsed::AstRefInterface;
use crate::syntax::source::SourceDatabase;
use crate::util::iter::IterExt;
use crate::util::ResultDoubleExt;
use indexmap::map::Entry;
use indexmap::IndexMap;

#[derive(Debug)]
pub struct ElaboratedInterfaceInfo {
    pub id: MaybeIdentifier,
    pub ports: IndexMap<String, ElaboratedInterfacePortInfo>,
    pub views: IndexMap<String, ElaboratedInterfaceViewInfo>,
}

impl ElaboratedInterfaceInfo {
    pub fn get_port(
        &self,
        diags: &Diagnostics,
        source: &SourceDatabase,
        index: Identifier,
    ) -> DiagResult<(usize, &ElaboratedInterfacePortInfo)> {
        match self.ports.get_index_of(index.str(source)) {
            None => {
                let diag = Diagnostic::new("dot index does not match any interface port")
                    .add_error(index.span, "this identifier should match a port")
                    .add_info(self.id.span(), "interface declared here")
                    .finish();
                Err(diags.report(diag))
            }
            Some(index) => Ok((index, &self.ports[index])),
        }
    }

    pub fn get_view(
        &self,
        diags: &Diagnostics,
        source: &SourceDatabase,
        index: Identifier,
    ) -> DiagResult<(usize, &ElaboratedInterfaceViewInfo)> {
        match self.views.get_index_of(index.str(source)) {
            None => {
                let diag = Diagnostic::new("dot index does not match any interface view")
                    .add_error(index.span, "this identifier should match a view")
                    .add_info(self.id.span(), "interface declared here")
                    .finish();
                Err(diags.report(diag))
            }
            Some(index) => Ok((index, &self.views[index])),
        }
    }
}

#[derive(Debug)]
pub struct ElaboratedInterfacePortInfo {
    pub id: Identifier,
    pub ty: DiagResult<Spanned<HardwareType>>,
}

#[derive(Debug)]
pub struct ElaboratedInterfaceViewInfo {
    pub id: MaybeIdentifier,
    pub port_dirs: DiagResult<Vec<(Identifier, Spanned<PortDirection>)>>,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_interface_new(
        self,
        ast_ref: AstRefInterface,
        scope_params: CapturedScope,
    ) -> DiagResult<ElaboratedInterfaceInfo> {
        let diags = self.diags;
        let source = self.fixed.source;
        let &ItemDefInterface {
            span,
            vis: _,
            id: ref interface_id,
            params: _,
            span_body,
            ref port_types,
            ref views,
        } = &self.fixed.parsed[ast_ref];

        // rebuild params scope
        let mut ctx = CompileItemContext::new_empty(self, None);
        let flow_root = FlowRoot::new(diags);
        let mut flow = FlowCompile::new_root(&flow_root, span_body, "item body");
        let scope_params = scope_params.to_scope(self, &mut flow, span);

        // elaborate port types
        let mut scope_ports = Scope::new_child(span_body, &scope_params);
        let mut port_map = IndexMap::new();

        ctx.compile_elaborate_extra_list(
            &mut scope_ports,
            &mut flow,
            port_types,
            &mut |ctx, scope_params, flow, &(port_id, ty)| {
                let ty_eval = ctx.eval_expression_as_ty(scope_params, flow, ty).and_then(|ty| {
                    match ty.inner.as_hardware_type(self) {
                        Ok(ty_hw) => Ok(Spanned::new(ty.span, ty_hw)),
                        Err(_) => Err(diags.report_simple(
                            "interface ports must have hardware types",
                            ty.span,
                            format!("got non-hardware type `{}`", ty.inner.diagnostic_string()),
                        )),
                    }
                });

                match port_map.entry(port_id.str(source).to_owned()) {
                    Entry::Occupied(mut entry) => {
                        let prev: &mut ElaboratedInterfacePortInfo = entry.get_mut();
                        let diag = Diagnostic::new("port declared twice")
                            .add_info(prev.id.span, "previously declared here")
                            .add_error(port_id.span, "redeclared here")
                            .finish();
                        prev.ty = Err(diags.report(diag));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(ElaboratedInterfacePortInfo {
                            id: port_id,
                            ty: ty_eval,
                        });
                    }
                }

                Ok(())
            },
        )?;

        // elaborate views
        let mut view_map: IndexMap<String, ElaboratedInterfaceViewInfo> = IndexMap::new();
        for view in views {
            let &InterfaceView {
                id: view_id,
                ref port_dirs,
            } = view;

            let mut port_dir_vec: Vec<Option<DiagResult<(Identifier, Spanned<PortDirection>)>>> =
                vec![None; port_map.len()];
            let mut any_view_err = Ok(());

            ctx.compile_elaborate_extra_list(
                &mut scope_ports,
                &mut flow,
                port_dirs,
                &mut (|_, _, _, (port_id, dir)| {
                    if let Some(port_index) = port_map.get_index_of(port_id.str(source)) {
                        let slot = &mut port_dir_vec[port_index];
                        if let Some(prev) = &*slot {
                            if let Ok((prev_id, _)) = prev {
                                let diag = Diagnostic::new("port direction set twice")
                                    .add_info(prev_id.span, "previously set here")
                                    .add_error(port_id.span, "set again here")
                                    .finish();
                                *slot = Some(Err(diags.report(diag)));
                            }
                        } else {
                            *slot = Some(Ok((*port_id, *dir)));
                        }
                    } else {
                        any_view_err = Err(diags.report_simple(
                            "port not found in this interface",
                            port_id.span,
                            "attempt to set direction here",
                        ));
                    }

                    Ok(())
                }),
            )?;

            let port_dir_vec = any_view_err.and_then(|()| {
                port_dir_vec
                    .into_iter()
                    .enumerate()
                    .map(|(port_index, dir)| {
                        dir.ok_or_else(|| {
                            let port_id = &port_map.get_index(port_index).unwrap().1.id;
                            let diag = Diagnostic::new(format!("missing direction for port `{}`", port_id.str(source)))
                                .add_info(port_id.span, "port declared here")
                                .add_error(
                                    interface_id.span(),
                                    "this interface does not set a direction for this port",
                                )
                                .finish();
                            diags.report(diag)
                        })
                        .flatten_err()
                    })
                    .try_collect_all_vec()
            });

            let view_eval = ElaboratedInterfaceViewInfo {
                id: view_id,
                port_dirs: port_dir_vec,
            };
            if let MaybeIdentifier::Identifier(view_id) = view_id {
                if let Some(prev) = port_map.get(view_id.str(source)) {
                    let diag = Diagnostic::new("view name conflicts with port name")
                        .add_error(view_id.span, "view declared here")
                        .add_info(prev.id.span, "port declared here")
                        .finish();
                    diags.report(diag);
                }

                match view_map.entry(view_id.str(source).to_owned()) {
                    Entry::Occupied(mut entry) => {
                        let diag = Diagnostic::new("view name already declared")
                            .add_info(entry.get().id.span(), "previously declared here")
                            .add_error(view_id.span, "redeclared here")
                            .finish();
                        entry.get_mut().port_dirs = Err(diags.report(diag));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(view_eval);
                    }
                }
            }
        }

        Ok(ElaboratedInterfaceInfo {
            id: *interface_id,
            ports: port_map,
            views: view_map,
        })
    }
}
