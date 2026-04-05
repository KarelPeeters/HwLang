use crate::front::compile::CompileItemContext;
use crate::front::flow::FlowCompile;
use crate::front::item::{UniqueDeclaration, debug_info_name_including_params};
use crate::front::scope::Scope;
use crate::front::types::HardwareType;
use crate::front::value::CompileValue;
use crate::syntax::ast::{
    Identifier, InterfaceListItem, InterfaceSignal, InterfaceView, ItemDefInterface, MaybeIdentifier,
};
use crate::syntax::parsed::AstRefInterface;
use hwl_common::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use hwl_common::mid::ir::PortDirection;
use hwl_common::pos::{HasSpan, Spanned};
use hwl_common::source::SourceDatabase;
use hwl_common::util::ResultDoubleExt;
use hwl_common::util::iter::IterExt;
use indexmap::IndexMap;
use indexmap::map::Entry;

#[derive(Debug)]
pub struct ElaboratedInterfaceInfo {
    pub id: MaybeIdentifier,
    pub debug_info_name: String,
    pub signals: IndexMap<String, ElaboratedInterfaceSignalInfo>,
    pub views: IndexMap<String, ElaboratedInterfaceViewInfo>,
}

impl ElaboratedInterfaceInfo {
    pub fn get_port(
        &self,
        diags: &Diagnostics,
        source: &SourceDatabase,
        index: Identifier,
    ) -> DiagResult<(usize, &ElaboratedInterfaceSignalInfo)> {
        match self.signals.get_index_of(index.str(source)) {
            None => Err(DiagnosticError::new(
                "dot index does not match any interface port",
                index.span,
                "this identifier should match a port",
            )
            .add_info(self.id.span(), "interface declared here")
            .report(diags)),
            Some(index) => Ok((index, &self.signals[index])),
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
pub struct ElaboratedInterfaceSignalInfo {
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

impl CompileItemContext<'_, '_> {
    pub fn elaborate_interface_new(
        &mut self,
        scope_params: &Scope,
        flow: &mut FlowCompile,
        unique: UniqueDeclaration,
        params: &Option<Vec<(Identifier, CompileValue)>>,
        ast_ref: AstRefInterface,
    ) -> DiagResult<ElaboratedInterfaceInfo> {
        let refs = self.refs;
        let diags = refs.diags;
        let source = refs.fixed.source;
        let elab = &refs.shared.elaboration_arenas;

        let &ItemDefInterface {
            span: _,
            vis: _,
            id: ref interface_id,
            params: _,
            span_body,
            ref body,
        } = &refs.fixed.parsed[ast_ref];

        // elaborate extra list, collect signal types and view directions immediately,
        //   then later actually whether the views signals match the actual signals
        let mut scope_body = scope_params.new_child(span_body);
        let mut signal_map = IndexMap::new();
        let mut views_partial: Vec<InterfaceViewPartialElab> = vec![];

        self.elaborate_extra_list(&mut scope_body, flow, body, true, &mut |slf, scope, flow, item| {
            match item {
                &InterfaceListItem::Signal(signal) => {
                    let InterfaceSignal {
                        id: signal_id,
                        ty: signal_ty,
                    } = signal;
                    let ty_eval = slf
                        .eval_expression_as_ty(scope.as_scope(), flow, signal_ty)
                        .and_then(|ty| match ty.inner.as_hardware_type(elab) {
                            Ok(ty_hw) => Ok(Spanned::new(ty.span, ty_hw)),
                            Err(_) => Err(diags.report_error_simple(
                                "interface signals must have hardware types",
                                ty.span,
                                format!("got non-hardware type `{}`", ty.inner.value_string(elab)),
                            )),
                        });

                    match signal_map.entry(signal_id.str(source).to_owned()) {
                        Entry::Occupied(mut entry) => {
                            let prev: &mut ElaboratedInterfaceSignalInfo = entry.get_mut();
                            let e = DiagnosticError::new("signal declared twice", signal_id.span, "redeclared here")
                                .add_info(prev.id.span, "previously declared here")
                                .report(diags);
                            prev.ty = Err(e);
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(ElaboratedInterfaceSignalInfo {
                                id: signal_id,
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
                    slf.elaborate_extra_list(scope.as_scope(), flow, port_dirs, true, &mut |_, _, _, &port_dir| {
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
                vec![None; signal_map.len()];
            let mut any_view_err = Ok(());

            for (port_id, port_dir) in ports_dirs {
                if let Some(port_index) = signal_map.get_index_of(port_id.str(source)) {
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
                            let port_id = &signal_map.get_index(port_index).unwrap().1.id;
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
                if let Some(prev) = signal_map.get(view_id.str(source)) {
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
            signals: signal_map,
            views: view_map,
        })
    }
}
