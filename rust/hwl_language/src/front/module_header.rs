use crate::front::check::{TypeContainsReason, check_type_contains_value};
use crate::front::compile::{ArenaPortInterfaces, ArenaPorts, CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, DiagnosticError, Diagnostics};
use crate::front::domain::PortDomain;
use crate::front::extra::ExtraScope;
use crate::front::flow::{FlowCompile, FlowCompileContent, FlowRoot, FlowRootContent};
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

        // collect results
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

    fn elaborate_module_ports_impl<'p>(
        &self,
        ctx: &mut CompileItemContext,
        scope_params: &'p Scope<'p>,
        flow: &mut FlowCompile,
        ports: &Spanned<ExtraList<ModulePortItem>>,
        module_def_span: Span,
    ) -> DiagResult<(ArenaConnectors, Scope<'p>, Arena<IrPort, IrPortInfo>)> {
        let mut scope_ports = scope_params.new_child(ports.span.join(Span::empty_at(module_def_span.end())));

        // build context
        let mut connectors: ArenaConnectors = Arena::new();
        let mut ports_ir = Arena::new();
        let mut ctx_ports = ModulePortsContext {
            connectors: &mut connectors,
            ports_ir: &mut ports_ir,
            port_to_single: IndexMap::new(),
            used_ir_names: IndexMap::new(),
            next_single_index: 0,
        };

        // visit port extra list
        ctx.elaborate_extra_list(
            &mut scope_ports,
            flow,
            &ports.inner,
            true,
            &mut |ctx, scope, flow, port_item| ctx_ports.visit_port_item(ctx, scope, flow, port_item),
        )?;

        Ok((connectors, scope_ports, ports_ir))
    }
}

pub struct ModulePortsContext<'a> {
    // results
    connectors: &'a mut ArenaConnectors,
    ports_ir: &'a mut IrPorts,

    // intermediate state
    port_to_single: IndexMap<Port, ConnectorSingle>,
    used_ir_names: IndexMap<String, Span>,
    next_single_index: usize,
}

impl ModulePortsContext<'_> {
    fn visit_port_item(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &mut ExtraScope,
        flow: &mut FlowCompile,
        port_item: &ModulePortItem,
    ) -> DiagResult<()> {
        let refs = ctx.refs;
        let diags = refs.diags;
        let source = refs.fixed.source;
        let elab = &refs.shared.elaboration_arenas;

        match port_item {
            ModulePortItem::Single(port_item) => {
                let &ModulePortSingle { span: _, id, ref kind } = port_item;
                match *kind {
                    ModulePortSingleKind::Port { direction, ref kind } => {
                        // eval domain and type
                        let (domain, ty) = match *kind {
                            PortSingleKindInner::Clock { span_clock } => (
                                Ok(Spanned::new(span_clock, PortDomain::Clock)),
                                Ok(Spanned::new(span_clock, HardwareType::Bool)),
                            ),
                            PortSingleKindInner::Normal { domain, ty } => (
                                ctx.eval_port_domain(scope.as_scope(), flow, domain)
                                    .map(|d| d.map_inner(PortDomain::Kind)),
                                ctx.eval_expression_as_ty_hardware(scope.as_scope(), flow, ty, "port"),
                            ),
                        };

                        // record
                        let entry = self.push_connector_single(ctx, id, direction, domain, ty);
                        scope.declare_root(diags, id.spanned_str(source), entry);
                    }
                    ModulePortSingleKind::Interface {
                        span_keyword,
                        domain,
                        interface,
                    } => {
                        // eval domain
                        let domain = ctx.eval_port_domain(scope.as_scope(), flow, domain);

                        // eval interface view
                        let interface_view = ctx
                            .eval_expression_as_compile(
                                scope.as_scope(),
                                flow,
                                &Type::InterfaceView,
                                interface,
                                Spanned::new(interface.span, "interface view"),
                            )
                            .and_then(|view| {
                                let reason = TypeContainsReason::InterfacePortView(span_keyword);
                                check_type_contains_value(diags, elab, reason, &Type::InterfaceView, view.as_ref())?;

                                match view.inner {
                                    CompileValue::Simple(SimpleCompileValue::InterfaceView(inner)) => {
                                        Ok(Spanned::new(view.span, inner))
                                    }
                                    _ => Err(diags.report_error_internal(view.span, "expected interface view")),
                                }
                            });

                        // record
                        let entry = self.push_connector_interface(ctx, id, domain, interface_view);
                        scope.declare_root(diags, id.spanned_str(source), entry);
                    }
                }
            }
            ModulePortItem::DomainBlock(port_item) => {
                let &ModulePortDomainBlock {
                    span: _,
                    domain,
                    ref ports,
                } = port_item;

                // eval domain (shared between all inner ports)
                let domain = ctx.eval_port_domain(scope.as_scope(), flow, domain);

                // visit inner ports
                ctx.elaborate_extra_list_block(
                    scope,
                    flow,
                    ports,
                    true,
                    &mut |ctx, scope, flow, port_item_in_block| {
                        self.visit_port_item_in_block(ctx, scope, flow, domain, port_item_in_block);
                        Ok(())
                    },
                )?;
            }
        }

        Ok(())
    }

    fn visit_port_item_in_block(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &mut ExtraScope,
        flow: &mut FlowCompile,
        domain: DiagResult<Spanned<DomainKind<Polarized<Port>>>>,
        port_item_in_block: &ModulePortInBlock,
    ) {
        let refs = ctx.refs;
        let diags = refs.diags;
        let source = refs.fixed.source;
        let elab = &refs.shared.elaboration_arenas;

        let &ModulePortInBlock { span: _, id, ref kind } = port_item_in_block;

        match *kind {
            ModulePortInBlockKind::Port { direction, ty } => {
                // map domain
                let domain = domain.map(|d| d.map_inner(PortDomain::Kind));

                // eval type
                let ty = ctx.eval_expression_as_ty_hardware(scope.as_scope(), flow, ty, "port");

                // record
                let entry = self.push_connector_single(ctx, id, direction, domain, ty);
                scope.declare_root(diags, id.spanned_str(source), entry);
            }
            ModulePortInBlockKind::Interface {
                span_keyword: _,
                interface,
            } => {
                // eval interface view
                let interface_view = ctx
                    .eval_expression_as_compile(
                        scope.as_scope(),
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
                            format!("got other value with type `{}`", value.ty().value_string(elab)),
                        )),
                    });

                // record
                let entry = self.push_connector_interface(ctx, id, domain, interface_view);
                scope.declare_root(diags, id.spanned_str(source), entry);
            }
        }
    }

    fn push_connector_single(
        &mut self,
        ctx: &mut CompileItemContext,
        id: Identifier,
        direction: Spanned<PortDirection>,
        domain: DiagResult<Spanned<PortDomain<Port>>>,
        ty: DiagResult<Spanned<HardwareType>>,
    ) -> DiagResult<ScopedEntry> {
        let diags = ctx.refs.diags;
        let source = ctx.refs.fixed.source;
        let elab = &ctx.refs.shared.elaboration_arenas;

        claim_ir_name(diags, &mut self.used_ir_names, id.str(source), id.span)?;

        let kind_and_entry = result_pair(domain, ty).map(|(domain, ty)| {
            let ir_port = self.ports_ir.push(IrPortInfo {
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

            let single = self.create_port_single(port);

            let connector_domain = domain.map_inner(|d| d.map_inner(|p| self.get_port_single(p)));
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
        self.connectors.push(ConnectorInfo { id, kind });
        entry
    }

    fn push_connector_interface(
        &mut self,
        ctx: &mut CompileItemContext,
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
                claim_ir_name(diags, &mut self.used_ir_names, &ir_name, id.span)?;

                let direction = port_dirs[port_index].1;
                let ty = port.ty.as_ref_ok()?;

                let ir_port = self.ports_ir.push(IrPortInfo {
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

                let single = self.create_port_single(port);
                singles.push(single);
            }

            let port_interface = ctx.port_interfaces.push(PortInterfaceInfo {
                id,
                view,
                domain,
                ports: declared_ports,
            });
            let connector_domain = domain.map_inner(|d| d.map_signal(|p| p.map_inner(|p| self.get_port_single(p))));
            let kind = ConnectorKind::Interface {
                domain: connector_domain,
                view,
                singles,
            };

            let entry = ScopedEntry::Named(NamedValue::Interface(Interface::Port(port_interface)));
            Ok((kind, entry))
        });

        let (kind, entry) = result_pair_split(kind_and_entry);
        self.connectors.push(ConnectorInfo { id, kind });
        entry
    }

    fn create_port_single(&mut self, port: Port) -> ConnectorSingle {
        let index = self.next_single_index;
        self.next_single_index += 1;

        let single = ConnectorSingle(index);
        self.port_to_single.insert(port, single);
        single
    }

    fn get_port_single(&self, port: Port) -> ConnectorSingle {
        self.port_to_single.get(&port).copied().unwrap()
    }
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
