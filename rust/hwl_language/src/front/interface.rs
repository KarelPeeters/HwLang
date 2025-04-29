use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::item::ElaboratedItemParams;
use crate::front::types::HardwareType;
use crate::front::variables::VariableValues;
use crate::syntax::ast::{Identifier, InterfaceView, ItemDefInterface, MaybeIdentifier, PortDirection, Spanned};
use crate::syntax::parsed::AstRefInterface;
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
        index: &Identifier,
    ) -> Result<(usize, &ElaboratedInterfacePortInfo), ErrorGuaranteed> {
        match self.ports.get_index_of(&index.string) {
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
        index: &Identifier,
    ) -> Result<&ElaboratedInterfaceViewInfo, ErrorGuaranteed> {
        match self.views.get(&index.string) {
            None => {
                let diag = Diagnostic::new("dot index does not match any interface view")
                    .add_error(index.span, "this identifier should match a view")
                    .add_info(self.id.span(), "interface declared here")
                    .finish();
                Err(diags.report(diag))
            }
            Some(view) => Ok(view),
        }
    }
}

#[derive(Debug)]
pub struct ElaboratedInterfacePortInfo {
    pub id: Identifier,
    pub ty: Result<Spanned<HardwareType>, ErrorGuaranteed>,
}

#[derive(Debug)]
pub struct ElaboratedInterfaceViewInfo {
    pub id: MaybeIdentifier,
    pub port_dirs: Result<Vec<(Identifier, Spanned<PortDirection>)>, ErrorGuaranteed>,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_interface_new(
        self,
        params: ElaboratedItemParams<AstRefInterface>,
    ) -> Result<ElaboratedInterfaceInfo, ErrorGuaranteed> {
        let ElaboratedItemParams { item, params } = params;
        let ItemDefInterface {
            span: _,
            vis: _,
            id: interface_id,
            params: _,
            span_body: _,
            port_types,
            views,
        } = &self.fixed.parsed[item];
        let diags = self.diags;

        // rebuild params scope
        let mut ctx = CompileItemContext::new_empty(self, None);
        let mut vars = VariableValues::new_root(&ctx.variables);
        let scope_params = ctx.rebuild_params_scope(item.into(), &mut vars, &params)?;

        // elaborate port types
        let mut port_map = IndexMap::new();

        for (port_id, ty) in port_types {
            let ty_eval = ctx.eval_expression_as_ty(&scope_params, &mut vars, ty).and_then(|ty| {
                match ty.inner.as_hardware_type() {
                    Ok(ty_hw) => Ok(Spanned::new(ty.span, ty_hw)),
                    Err(_) => Err(diags.report_simple(
                        "interface ports must have hardware types",
                        ty.span,
                        format!("got non-hardware type `{}`", ty.inner.to_diagnostic_string()),
                    )),
                }
            });

            match port_map.entry(port_id.string.clone()) {
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
                        id: port_id.clone(),
                        ty: ty_eval,
                    });
                }
            }
        }

        // elaborate views
        let mut view_map: IndexMap<String, ElaboratedInterfaceViewInfo> = IndexMap::new();
        for view in views {
            let InterfaceView { id: view_id, port_dirs } = view;

            let mut port_dir_vec: Vec<Option<Result<(Identifier, Spanned<PortDirection>), ErrorGuaranteed>>> =
                vec![None; port_map.len()];
            let mut any_view_err = Ok(());

            for (port_id, dir) in port_dirs {
                if let Some(port_index) = port_map.get_index_of(&port_id.string) {
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
                        *slot = Some(Ok((port_id.clone(), *dir)));
                    }
                } else {
                    any_view_err = Err(diags.report_simple(
                        "port not found in this interface",
                        port_id.span,
                        "attempt to set direction here",
                    ));
                }
            }

            let port_dir_vec = any_view_err.and_then(|()| {
                port_dir_vec
                    .into_iter()
                    .enumerate()
                    .map(|(port_index, dir)| {
                        dir.ok_or_else(|| {
                            let port_id = &port_map.get_index(port_index).unwrap().1.id;
                            let diag = Diagnostic::new(format!("missing direction for port `{}`", port_id.string))
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
                id: view_id.clone(),
                port_dirs: port_dir_vec,
            };
            if let MaybeIdentifier::Identifier(view_id) = view_id {
                if let Some(prev) = port_map.get(&view_id.string) {
                    let diag = Diagnostic::new("view name conflicts with port name")
                        .add_error(view_id.span, "view declared here")
                        .add_info(prev.id.span, "port declared here")
                        .finish();
                    diags.report(diag);
                }

                match view_map.entry(view_id.string.clone()) {
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
            id: interface_id.clone(),
            ports: port_map,
            views: view_map,
        })
    }
}
