use crate::front::check::{TypeContainsReason, check_type_contains_value};
use crate::front::compile::{ArenaPortInterfaces, ArenaPorts, CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::PortDomain;
use crate::front::extra::ExtraScope;
use crate::front::flow::{Flow, FlowCompile, FlowCompileContent, FlowRoot, FlowRootContent};
use crate::front::item::{ElaboratedInterfaceView, ElaboratedItemParams, ElaboratedModule, UniqueDeclaration};
use crate::front::scope::{FrozenScope, NamedValue, Scope, ScopeContent, ScopedEntry};
use crate::front::signal::{Interface, Polarized, Port, PortInfo, PortInterfaceInfo, Signal};
use crate::front::types::{HardwareType, Type, Typed};
use crate::front::value::{CompileValue, SimpleCompileValue};
use crate::mid::ir::{IrModule, IrPort, IrPortInfo, IrPorts};
use crate::new_index_type;
use crate::syntax::ast::{
    DomainKind, ExtraList, Identifier, ModulePortDomainBlock, ModulePortInBlock, ModulePortInBlockKind, ModulePortItem,
    ModulePortSingle, ModulePortSingleKind, PortDirection, PortSingleKindInner,
};
use crate::syntax::parsed::{AstRefItemKind, AstRefModuleExternal, AstRefModuleInternal};
use crate::syntax::pos::{Span, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::BigInt;
use crate::util::data::IndexMapExt;
use crate::util::{ResultExt, result_pair, result_pair_split};
use indexmap::IndexMap;
use indexmap::map::Entry;
use itertools::{Itertools, enumerate};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

new_index_type!(pub Connector);

pub type ArenaConnectors = Arena<Connector, ConnectorInfo>;

pub struct ConnectorInfo {
    pub id: Identifier,
    pub kind: DiagResult<ConnectorKind>,
}

pub enum ConnectorKind {
    Port {
        direction: Spanned<PortDirection>,
        domain: Spanned<PortDomain<ConnectorSingle>>,
        ty: Spanned<HardwareType>,
        single: ConnectorSingle,
    },
    Interface {
        domain: Spanned<DomainKind<Polarized<ConnectorSingle>>>,
        view: Spanned<ElaboratedInterfaceView>,
        singles: Vec<ConnectorSingle>,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ConnectorSingle(usize);

#[derive(Debug)]
pub struct ElaboratedModuleHeader<A> {
    pub ast_ref: A,
    pub elab_module: ElaboratedModule,
    pub debug_info_params: Option<Vec<(String, String)>>,

    pub ports: ArenaPorts,
    pub port_interfaces: ArenaPortInterfaces,
    pub ports_ir: Arena<IrPort, IrPortInfo>,

    pub scope_params: Arc<FrozenScope>,
    pub scope_ports: ScopeContent,
    pub flow_root: FlowRootContent,
    pub flow: FlowCompileContent,
}

pub struct ElaboratedModuleInternalInfo {
    pub ast_ref: AstRefModuleInternal,
    pub unique: UniqueDeclaration,

    pub module_ir: IrModule,
    pub connectors: ArenaConnectors,
}

pub struct ElaboratedModuleExternalInfo {
    pub ast_ref: AstRefModuleExternal,
    pub module_name: String,
    pub generic_args: Option<Vec<(String, BigInt)>>,
    pub ports: Vec<(String, HardwareType)>,
    pub connectors: ArenaConnectors,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_module_ports_new<A: AstRefItemKind>(
        self,
        ast_ref: A,
        def_span: Span,
        elab_module: ElaboratedModule,
        params: ElaboratedItemParams,
        scope_params: Arc<FrozenScope>,
        ports: &Spanned<ExtraList<ModulePortItem>>,
    ) -> DiagResult<(ArenaConnectors, ElaboratedModuleHeader<A>)> {
        let ElaboratedItemParams {
            unique: _,
            params: debug_info_params,
        } = params;

        // reconstruct header scope
        let mut ctx = CompileItemContext::new_empty(self, None, Some(elab_module));
        let flow_root = FlowRoot::new(self.diags, &self.shared.next_flow_root_id);
        let mut flow = FlowCompile::new_root(&flow_root, def_span, "item declaration");

        // elaborate ports
        let scope_params_tmp = &Arc::clone(&scope_params).as_scope();
        let (connectors, scope_ports, ports_ir) =
            self.elaborate_module_ports_impl(&mut ctx, scope_params_tmp, &mut flow, ports, def_span)?;

        // save scope and flow
        let scope_ports = scope_ports.into_content();
        let flow = flow.into_content();

        // create params debug info string
        let source = self.fixed.source;
        let debug_info_params = debug_info_params.map(|p| {
            p.into_iter()
                .map(|(k, v)| {
                    (
                        k.str(source).to_owned(),
                        v.value_string(&self.shared.elaboration_arenas),
                    )
                })
                .collect_vec()
        });

        let header = ElaboratedModuleHeader {
            elab_module,
            ast_ref,
            debug_info_params,
            ports: ctx.ports,
            port_interfaces: ctx.port_interfaces,
            ports_ir,

            scope_params,
            scope_ports,
            flow_root: flow_root.into_content(),
            flow,
        };
        Ok((connectors, header))
    }

    fn elaborate_module_ports_impl<'p, F: Flow>(
        &self,
        ctx: &mut CompileItemContext,
        scope_params: &'p Scope<'p>,
        flow: &mut F,
        ports: &Spanned<ExtraList<ModulePortItem>>,
        module_def_span: Span,
    ) -> DiagResult<(ArenaConnectors, Scope<'p>, Arena<IrPort, IrPortInfo>)> {
        let diags = self.diags;
        let source = self.fixed.source;
        let elab = &self.shared.elaboration_arenas;

        let mut scope_ports_root = scope_params.new_child(ports.span.join(Span::empty_at(module_def_span.end())));

        let mut connectors: ArenaConnectors = Arena::new();
        let mut port_to_single: IndexMap<Port, ConnectorSingle> = IndexMap::new();
        let mut ports_ir = Arena::new();
        let mut used_ir_names = IndexMap::new();

        let mut next_single = 0;
        let mut next_single = move || {
            let scalar = ConnectorSingle(next_single);
            next_single += 1;
            scalar
        };

        let mut visit_port_item = |ctx: &mut CompileItemContext,
                                   scope_ports: &mut ExtraScope,
                                   flow: &mut F,
                                   port_item: &ModulePortItem| {
            match port_item {
                ModulePortItem::Single(port_item) => {
                    let &ModulePortSingle { span: _, id, ref kind } = port_item;
                    match *kind {
                        ModulePortSingleKind::Port { direction, ref kind } => {
                            let (domain, ty) = match *kind {
                                PortSingleKindInner::Clock { span_clock } => (
                                    Ok(Spanned::new(span_clock, PortDomain::Clock)),
                                    Ok(Spanned::new(span_clock, HardwareType::Bool)),
                                ),
                                PortSingleKindInner::Normal { domain, ty } => (
                                    ctx.eval_port_domain(scope_ports.as_scope(), flow, domain)
                                        .map(|d| d.map_inner(PortDomain::Kind)),
                                    ctx.eval_expression_as_ty_hardware(scope_ports.as_scope(), flow, ty, "port"),
                                ),
                            };

                            let entry = push_connector_single(
                                ctx,
                                &mut used_ir_names,
                                &mut ports_ir,
                                &mut connectors,
                                &mut port_to_single,
                                &mut next_single,
                                id,
                                direction,
                                domain,
                                ty,
                            );
                            scope_ports.declare_root(diags, id.spanned_str(source), entry);
                        }
                        ModulePortSingleKind::Interface {
                            span_keyword,
                            domain,
                            interface,
                        } => {
                            let domain = ctx.eval_port_domain(scope_ports.as_scope(), flow, domain);
                            let interface_view = ctx
                                .eval_expression_as_compile(
                                    scope_ports.as_scope(),
                                    flow,
                                    &Type::InterfaceView,
                                    interface,
                                    Spanned::new(interface.span, "interface view"),
                                )
                                .and_then(|view| {
                                    let reason = TypeContainsReason::InterfacePortView(span_keyword);
                                    check_type_contains_value(
                                        diags,
                                        elab,
                                        reason,
                                        &Type::InterfaceView,
                                        view.as_ref(),
                                    )?;

                                    match view.inner {
                                        CompileValue::Simple(SimpleCompileValue::InterfaceView(inner)) => {
                                            Ok(Spanned::new(view.span, inner))
                                        }
                                        _ => Err(diags.report_error_internal(view.span, "expected interface view")),
                                    }
                                });

                            let entry = push_connector_interface(
                                ctx,
                                &mut used_ir_names,
                                &mut ports_ir,
                                &mut connectors,
                                &mut port_to_single,
                                &mut next_single,
                                id,
                                domain,
                                interface_view,
                            );
                            scope_ports.declare_root(diags, id.spanned_str(source), entry);
                        }
                    }
                }
                ModulePortItem::DomainBlock(port_item) => {
                    let &ModulePortDomainBlock {
                        span: _,
                        domain,
                        ref ports,
                    } = port_item;

                    let domain = ctx.eval_port_domain(scope_ports.as_scope(), flow, domain);

                    let mut visit_port_item_in_block =
                        |ctx: &mut CompileItemContext,
                         scope_ports: &mut ExtraScope,
                         flow: &mut F,
                         port_item_in_block: &ModulePortInBlock| {
                            let &ModulePortInBlock { span: _, id, ref kind } = port_item_in_block;

                            match *kind {
                                ModulePortInBlockKind::Port { direction, ty } => {
                                    let domain = domain.map(|d| d.map_inner(PortDomain::Kind));
                                    let ty =
                                        ctx.eval_expression_as_ty_hardware(scope_ports.as_scope(), flow, ty, "port");

                                    let entry = push_connector_single(
                                        ctx,
                                        &mut used_ir_names,
                                        &mut ports_ir,
                                        &mut connectors,
                                        &mut port_to_single,
                                        &mut next_single,
                                        id,
                                        direction,
                                        domain,
                                        ty,
                                    );
                                    scope_ports.declare_root(diags, id.spanned_str(source), entry);
                                }
                                ModulePortInBlockKind::Interface {
                                    span_keyword: _,
                                    interface,
                                } => {
                                    let interface_view = ctx
                                        .eval_expression_as_compile(
                                            scope_ports.as_scope(),
                                            flow,
                                            &Type::InterfaceView,
                                            interface,
                                            Spanned::new(interface.span, "interface"),
                                        )
                                        .and_then(|view| match view.inner {
                                            CompileValue::Simple(SimpleCompileValue::InterfaceView(inner)) => {
                                                Ok(Spanned::new(view.span, inner))
                                            }
                                            value => Err(diags.report_error_simple(
                                                "expected interface view",
                                                view.span,
                                                format!(
                                                    "got other value with type `{}`",
                                                    value.ty().value_string(elab)
                                                ),
                                            )),
                                        });

                                    let entry = push_connector_interface(
                                        ctx,
                                        &mut used_ir_names,
                                        &mut ports_ir,
                                        &mut connectors,
                                        &mut port_to_single,
                                        &mut next_single,
                                        id,
                                        domain,
                                        interface_view,
                                    );
                                    scope_ports.declare_root(diags, id.spanned_str(source), entry);
                                }
                            }
                            Ok(())
                        };

                    ctx.elaborate_extra_list_block(scope_ports, flow, ports, true, &mut visit_port_item_in_block)?;
                }
            }
            Ok(())
        };

        ctx.elaborate_extra_list(&mut scope_ports_root, flow, &ports.inner, true, &mut visit_port_item)?;

        Ok((connectors, scope_ports_root, ports_ir))
    }
}

fn push_connector_single(
    ctx: &mut CompileItemContext,
    used_ir_names: &mut IndexMap<String, Span>,
    ports_ir: &mut IrPorts,
    connectors: &mut ArenaConnectors,
    port_to_single: &mut IndexMap<Port, ConnectorSingle>,
    next_single: &mut impl FnMut() -> ConnectorSingle,
    id: Identifier,
    direction: Spanned<PortDirection>,
    domain: DiagResult<Spanned<PortDomain<Port>>>,
    ty: DiagResult<Spanned<HardwareType>>,
) -> DiagResult<ScopedEntry> {
    let diags = ctx.refs.diags;
    let source = ctx.refs.fixed.source;
    let elab = &ctx.refs.shared.elaboration_arenas;

    claim_ir_name(diags, used_ir_names, id.str(source), id.span)?;

    let kind_and_entry = result_pair(domain, ty).map(|(domain, ty)| {
        let ir_port = ports_ir.push(IrPortInfo {
            name: id.str(source).to_owned(),
            direction: direction.inner,
            ty: ty.inner.as_ir(ctx.refs),
            debug_span: id.span,
            debug_info_ty: ty.as_ref().map_inner(|inner| inner.value_string(elab)),
            debug_info_domain: domain.inner.diagnostic_string(ctx),
        });

        let port = ctx.ports.push(PortInfo {
            span: id.span,
            name: id.str(source).to_owned(),
            direction,
            domain,
            ty: ty.clone(),
            ir: ir_port,
        });

        let single = next_single();
        port_to_single.insert_first(port, single);

        let connector_domain = domain.map_inner(|d| d.map_inner(|p| *port_to_single.get(&p).unwrap()));
        let kind = ConnectorKind::Port {
            direction,
            domain: connector_domain,
            ty,
            single,
        };

        let entry = ScopedEntry::Named(NamedValue::Signal(Signal::Port(port)));
        (kind, entry)
    });

    let (kind, entry) = result_pair_split(kind_and_entry);
    connectors.push(ConnectorInfo { id, kind });
    entry
}

fn push_connector_interface(
    ctx: &mut CompileItemContext,
    used_ir_names: &mut IndexMap<String, Span>,
    ports_ir: &mut IrPorts,
    connectors: &mut ArenaConnectors,
    port_to_single: &mut IndexMap<Port, ConnectorSingle>,
    next_single: &mut impl FnMut() -> ConnectorSingle,
    id: Identifier,
    domain: DiagResult<Spanned<DomainKind<Polarized<Port>>>>,
    view: DiagResult<Spanned<ElaboratedInterfaceView>>,
) -> DiagResult<ScopedEntry> {
    let diags = ctx.refs.diags;
    let source = ctx.refs.fixed.source;
    let elab = &ctx.refs.shared.elaboration_arenas;

    let kind_and_entry = result_pair(domain, view).and_then(|(domain, view)| {
        let ElaboratedInterfaceView { interface, view_index } = view.inner;
        let interface = ctx.refs.shared.elaboration_arenas.interface_info(interface);
        let view_info = &interface.views[view_index];

        let port_dirs = view_info.port_dirs.as_ref_ok()?;

        let mut declared_ports = vec![];
        let mut singles = vec![];

        // TODO rework interface port names, to ensure unique but still readable names
        for (port_index, (_, port)) in enumerate(&interface.signals) {
            let id_str = id.str(source);
            let port_id_str = port.id.str(source);
            let name = format!("{id_str}.{port_id_str}");
            let ir_name = format!("{id_str}_{port_id_str}");
            claim_ir_name(diags, used_ir_names, &ir_name, id.span)?;

            let direction = port_dirs[port_index].1;
            let ty = port.ty.as_ref_ok()?;

            let ir_port = ports_ir.push(IrPortInfo {
                name: ir_name,
                direction: direction.inner,
                ty: ty.inner.as_ir(ctx.refs),
                debug_span: id.span,
                debug_info_ty: ty.as_ref().map_inner(|ty| ty.value_string(elab)),
                debug_info_domain: domain.inner.diagnostic_string(ctx),
            });
            let port = ctx.ports.push(PortInfo {
                span: id.span,
                name,
                direction,
                domain: domain.map_inner(PortDomain::Kind),
                ty: ty.clone(),
                ir: ir_port,
            });

            declared_ports.push(port);

            let single = next_single();
            singles.push(single);
            port_to_single.insert_first(port, single);
        }

        let port_interface = ctx.port_interfaces.push(PortInterfaceInfo {
            id,
            view,
            domain,
            ports: declared_ports,
        });
        let connector_domain =
            domain.map_inner(|d| d.map_signal(|p| p.map_inner(|p| *port_to_single.get(&p).unwrap())));
        let kind = ConnectorKind::Interface {
            domain: connector_domain,
            view,
            singles,
        };

        let entry = ScopedEntry::Named(NamedValue::Interface(Interface::Port(port_interface)));
        Ok((kind, entry))
    });

    let (kind, entry) = result_pair_split(kind_and_entry);
    connectors.push(ConnectorInfo { id, kind });
    entry
}

// TODO think about this, and maybe expand to include more things (eg. signals, child modules, ...)
fn claim_ir_name(
    diags: &Diagnostics,
    used_ir_names: &mut IndexMap<String, Span>,
    name: &str,
    span: Span,
) -> DiagResult {
    match used_ir_names.entry(name.to_owned()) {
        Entry::Vacant(entry) => {
            entry.insert(span);
            Ok(())
        }
        Entry::Occupied(entry) => {
            let diag = DiagnosticError::new(
                format!("port with name `{name}` conflicts with earlier port with the same name"),
                span,
                "new port defined here",
            )
            .add_info(*entry.get(), "previous port defined here")
            .report(diags);
            Err(diag)
        }
    }
}
