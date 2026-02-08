use crate::front::compile::{CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::front::flow::{FlowCompile, FlowRoot};
use crate::front::function::CapturedScope;
use crate::front::item::{UniqueDeclaration, debug_info_name_including_params};
use crate::front::scope::Scope;
use crate::front::types::HardwareType;
use crate::front::value::CompileValue;
use crate::syntax::ast::{
    Identifier, InterfaceListItem, InterfaceView, ItemDefInterface, MaybeIdentifier, PortDirection,
};
use crate::syntax::parsed::AstRefInterface;
use crate::syntax::pos::{HasSpan, Spanned};
use crate::syntax::source::SourceDatabase;
use crate::util::ResultDoubleExt;
use crate::util::iter::IterExt;
use indexmap::IndexMap;
use indexmap::map::Entry;

#[derive(Debug)]
pub struct ElaboratedInterfaceInfo {
    pub id: MaybeIdentifier,
    pub debug_info_name: String,
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
            None => Err(DiagnosticError::new(
                "dot index does not match any interface port",
                index.span,
                "this identifier should match a port",
            )
            .add_info(self.id.span(), "interface declared here")
            .report(diags)),
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
            None => Err(DiagnosticError::new(
                "dot index does not match any interface view",
                index.span,
                "this identifier should match a view",
            )
            .add_info(self.id.span(), "interface declared here")
            .report(diags)),
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
    pub debug_info_name: String,
    pub port_dirs: DiagResult<Vec<(Identifier, Spanned<PortDirection>)>>,
}

struct InterfaceViewPartialElab {
    pub id: MaybeIdentifier,
    pub ports_dirs: Vec<(Identifier, Spanned<PortDirection>)>,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_interface_new(
        self,
        ast_ref: AstRefInterface,
        scope_params: CapturedScope,
        unique: UniqueDeclaration,
        params: &Option<Vec<(Identifier, CompileValue)>>,
    ) -> DiagResult<ElaboratedInterfaceInfo> {
        let diags = self.diags;
        let source = self.fixed.source;
        let elab = &self.shared.elaboration_arenas;

        let &ItemDefInterface {
            span,
            vis: _,
            id: ref interface_id,
            params: _,
            span_body,
            ref body,
        } = &self.fixed.parsed[ast_ref];

        // rebuild params scope
        let mut ctx = CompileItemContext::new_empty(self, None);
        let flow_root = FlowRoot::new(diags);
        let mut flow = FlowCompile::new_root(&flow_root, span_body, "item body");
        let scope_params = scope_params.to_scope(self, &mut flow, span)?;

        // elaborate extra list, collect port types and view directions immediately,
        //   then later actually whether the views ports match the actual ports
        let mut scope_body = Scope::new_child(span_body, &scope_params);
        let mut port_map = IndexMap::new();
        let mut views_partial: Vec<InterfaceViewPartialElab> = vec![];

        ctx.elaborate_extra_list(&mut scope_body, &mut flow, body, &mut |ctx, scope, flow, item| {
            match item {
                &InterfaceListItem::PortType { port_id, port_ty } => {
                    let ty_eval = ctx
                        .eval_expression_as_ty(scope.as_scope(), flow, port_ty)
                        .and_then(|ty| match ty.inner.as_hardware_type(elab) {
                            Ok(ty_hw) => Ok(Spanned::new(ty.span, ty_hw)),
                            Err(_) => Err(diags.report_error_simple(
                                "interface ports must have hardware types",
                                ty.span,
                                format!("got non-hardware type `{}`", ty.inner.value_string(elab)),
                            )),
                        });

                    match port_map.entry(port_id.str(source).to_owned()) {
                        Entry::Occupied(mut entry) => {
                            let prev: &mut ElaboratedInterfacePortInfo = entry.get_mut();
                            let e = DiagnosticError::new("port declared twice", port_id.span, "redeclared here")
                                .add_info(prev.id.span, "previously declared here")
                                .report(diags);
                            prev.ty = Err(e);
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(ElaboratedInterfacePortInfo {
                                id: port_id,
                                ty: ty_eval,
                            });
                        }
                    }
                }
                InterfaceListItem::View(view) => {
                    let &InterfaceView {
                        span: _,
                        id: view_id,
                        ref port_dirs,
                    } = view;

                    let mut port_dirs_partial = vec![];
                    ctx.elaborate_extra_list(scope.as_scope(), flow, port_dirs, &mut |_, _, _, &port_dir| {
                        port_dirs_partial.push(port_dir);
                        Ok(())
                    })?;

                    views_partial.push(InterfaceViewPartialElab {
                        id: view_id,
                        ports_dirs: port_dirs_partial,
                    });
                }
            }

            Ok(())
        })?;

        // check views
        let mut view_map: IndexMap<String, ElaboratedInterfaceViewInfo> = IndexMap::new();
        for view in views_partial {
            let InterfaceViewPartialElab {
                id: view_id,
                ports_dirs,
            } = view;

            let mut port_dir_vec: Vec<Option<DiagResult<(Identifier, Spanned<PortDirection>)>>> =
                vec![None; port_map.len()];
            let mut any_view_err = Ok(());

            for (port_id, port_dir) in ports_dirs {
                if let Some(port_index) = port_map.get_index_of(port_id.str(source)) {
                    let slot = &mut port_dir_vec[port_index];
                    if let Some(prev) = &*slot {
                        if let Ok((prev_id, _)) = prev {
                            let e = DiagnosticError::new("port direction set twice", port_id.span, "set again here")
                                .add_info(prev_id.span, "previously set here")
                                .report(diags);
                            *slot = Some(Err(e));
                        }
                    } else {
                        *slot = Some(Ok((port_id, port_dir)));
                    }
                } else {
                    any_view_err = Err(diags.report_error_simple(
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
                            DiagnosticError::new(
                                format!("missing direction for port `{}`", port_id.str(source)),
                                interface_id.span(),
                                "this interface does not set a direction for this port",
                            )
                            .add_info(port_id.span, "port declared here")
                            .report(diags)
                        })
                        .flatten_err()
                    })
                    .try_collect_all_vec()
            });

            let debug_info_name = view_id.spanned_string(source).inner.unwrap_or_else(|| "_".to_owned());
            let view_eval = ElaboratedInterfaceViewInfo {
                id: view_id,
                debug_info_name,
                port_dirs: port_dir_vec,
            };
            if let MaybeIdentifier::Identifier(view_id) = view_id {
                if let Some(prev) = port_map.get(view_id.str(source)) {
                    let _ =
                        DiagnosticError::new("view name conflicts with port name", view_id.span, "view declared here")
                            .add_info(prev.id.span, "port declared here")
                            .report(diags);
                }

                match view_map.entry(view_id.str(source).to_owned()) {
                    Entry::Occupied(mut entry) => {
                        let e = DiagnosticError::new("view name already declared", view_id.span, "redeclared here")
                            .add_info(entry.get().id.span(), "previously declared here")
                            .report(diags);
                        entry.get_mut().port_dirs = Err(e);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(view_eval);
                    }
                }
            }
        }

        let debug_info_name = debug_info_name_including_params(source, elab, unique, params);

        Ok(ElaboratedInterfaceInfo {
            id: *interface_id,
            debug_info_name,
            ports: port_map,
            views: view_map,
        })
    }
}
