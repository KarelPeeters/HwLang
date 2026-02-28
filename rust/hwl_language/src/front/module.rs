use crate::front::check::{TypeContainsReason, check_type_contains_type, check_type_contains_value};
use crate::front::compile::{ArenaPortInterfaces, ArenaPorts, CompileItemContext, CompileRefs};
use crate::front::diagnostic::{DiagResult, DiagnosticError, DiagnosticWarning, Diagnostics};
use crate::front::domain::{DomainSignal, PortDomain, ValueDomain};
use crate::front::exit::ExitStack;
use crate::front::expression::{NamedOrValue, ValueInner};
use crate::front::extra::ExtraScope;
use crate::front::flow::{
    Flow, FlowCompile, FlowCompileContent, FlowHardwareRoot, FlowRoot, FlowRootContent, HardwareProcessKind,
    RegisterInfo,
};
use crate::front::function::CapturedScope;
use crate::front::interface::ElaboratedInterfacePortInfo;
use crate::front::item::{ElaboratedInterfaceView, ElaboratedItemParams, ElaboratedModule, UniqueDeclaration};
use crate::front::scope::{NamedValue, Scope, ScopeContent, ScopeParent, ScopedEntry};
use crate::front::signal::{
    Polarized, Port, PortInfo, PortInterfaceInfo, Signal, WireInfo, WireInfoInInterface, WireInfoSingle,
    WireInterfaceInfo,
};
use crate::front::types::{HardwareType, NonHardwareType, Type, Typed};
use crate::front::value::{CompileValue, MaybeUndefined, SimpleCompileValue, ValueCommon};
use crate::mid::ir::{
    IrAssignmentTarget, IrAsyncResetInfo, IrBlock, IrClockedProcess, IrCombinatorialProcess, IrExpression,
    IrIfStatement, IrModule, IrModuleChild, IrModuleExternalInstance, IrModuleInfo, IrModuleInternalInstance, IrPort,
    IrPortConnection, IrPortInfo, IrPorts, IrSignal, IrStatement, IrWire, IrWireInfo,
};
use crate::syntax::ast::{
    self, ClockedProcessReset, ExpressionKind, ExtraList, ModuleInstance, ModulePortDomainBlock, ModulePortInBlock,
    ModulePortInBlockKind, ModulePortSingleKind, ModuleStatement, ModuleStatementKind, PortDirection,
    PortSingleKindInner, ResetKind, Visibility, WireDeclarationDomainTyKind, WireDeclarationKind,
};
use crate::syntax::ast::{
    ClockedProcess, CombinatorialProcess, DomainKind, Identifier, ModulePortItem, ModulePortSingle, PortConnection,
    SyncDomain, WireDeclaration,
};
use crate::syntax::parsed::{AstRefModuleExternal, AstRefModuleInternal};
use crate::syntax::pos::{HasSpan, Span, Spanned};
use crate::util::arena::Arena;
use crate::util::big_int::BigInt;
use crate::util::data::IndexMapExt;
use crate::util::store::ArcOrRef;
use crate::util::{ResultExt, result_pair, result_pair_split};
use crate::{new_index_type, throw};
use indexmap::IndexMap;
use indexmap::map::Entry;
use itertools::{Either, Itertools, chain, enumerate};
use std::fmt::Debug;
use std::hash::Hash;

new_index_type!(pub Connector);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct ConnectorSingle(usize);

type ArenaConnectors = Arena<Connector, ConnectorInfo>;

pub struct ConnectorInfo {
    id: Identifier,
    kind: DiagResult<ConnectorKind>,
}

enum ConnectorKind {
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

#[derive(Debug)]
pub struct ElaboratedModuleHeader<A> {
    pub ast_ref: A,
    debug_info_params: Option<Vec<(String, String)>>,
    ports: ArenaPorts,
    port_interfaces: ArenaPortInterfaces,
    pub ports_ir: Arena<IrPort, IrPortInfo>,

    captured_scope_params: CapturedScope,
    scope_ports: ScopeContent,
    flow_root: FlowRootContent,
    flow: FlowCompileContent,
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
    pub port_names: Vec<String>,
    pub connectors: ArenaConnectors,
}

impl CompileRefs<'_, '_> {
    pub fn elaborate_module_ports_new<A>(
        self,
        ast_ref: A,
        def_span: Span,
        params: ElaboratedItemParams,
        captured_scope_params: CapturedScope,
        ports: &Spanned<ExtraList<ModulePortItem>>,
    ) -> DiagResult<(ArenaConnectors, ElaboratedModuleHeader<A>)> {
        let ElaboratedItemParams {
            unique: _,
            params: debug_info_params,
        } = params;

        // reconstruct header scope
        let mut ctx = CompileItemContext::new_empty(self, None);
        let flow_root = FlowRoot::new(self.diags);
        let mut flow = FlowCompile::new_root(&flow_root, def_span, "item declaration");
        let scope_params = captured_scope_params.to_scope(self, &mut flow, def_span)?;

        // elaborate ports
        // TODO we actually need a full context/flow here?
        let (connectors, scope_ports, ports_ir) =
            self.elaborate_module_ports_impl(&mut ctx, &scope_params, &mut flow, ports, def_span)?;
        let scope_ports = scope_ports.into_content();

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

        let flow = flow.into_content();
        let header = ElaboratedModuleHeader {
            ast_ref,
            debug_info_params,
            ports: ctx.ports,
            port_interfaces: ctx.port_interfaces,
            ports_ir,

            captured_scope_params,
            scope_ports,
            flow_root: flow_root.into_content(),
            flow,
        };
        Ok((connectors, header))
    }

    pub fn elaborate_module_body_new(
        self,
        ports: ElaboratedModuleHeader<AstRefModuleInternal>,
    ) -> DiagResult<IrModuleInfo> {
        let ElaboratedModuleHeader {
            ast_ref,
            debug_info_params,
            ports,
            port_interfaces,
            ports_ir,
            captured_scope_params,
            scope_ports,
            flow_root,
            flow,
        } = ports;
        let &ast::ItemDefModuleInternal {
            span: def_span,
            vis: _,
            id: def_id,
            params: _,
            ports: _,
            ref body,
        } = &self.fixed.parsed[ast_ref];

        self.check_should_stop(def_id.span())?;

        // rebuild scopes
        let mut ctx = CompileItemContext::new_restore(self, None, ports, port_interfaces);
        let flow_root = FlowRoot::restore(self.diags, flow_root);
        let mut flow = FlowCompile::restore_root(&flow_root, flow);

        let scope_params = captured_scope_params.to_scope(self, &mut flow, def_span)?;
        let scope_ports = Scope::restore_from_content(ScopeParent::Normal(&scope_params), scope_ports);

        // elaborate the body
        let mut ctx_body = BodyContext {
            ir_ports: ports_ir,
            ir_wires: Arena::new(),
            drivers: IndexMap::new(),
            children: vec![],
            delayed_err: Ok(()),
        };
        let mut scope_body = scope_ports.new_child(body.span);
        ctx.elaborate_extra_list(
            &mut scope_body,
            &mut flow,
            &body.inner,
            &mut |ctx, scope, flow, stmt| ctx_body.elaborate_module_statement(ctx, scope, flow, stmt),
        )?;
        ctx_body.delayed_err?;

        // check for driver issues
        // (wires without inferred types are fine if they're not actually used or driven)
        // TODO add warning for unused signals / input ports
        let signals_that_need_drivers = chain(
            ctx.wires.keys().map(Signal::Wire),
            ctx.ports
                .iter()
                .filter_map(|(p, info)| match info.direction.inner {
                    PortDirection::Input => None,
                    PortDirection::Output => Some(p),
                })
                .map(Signal::Port),
        );

        let mut drivers = ctx_body.drivers;
        let mut any_driver_err = Ok(());
        for signal in signals_that_need_drivers {
            let drivers = drivers.swap_remove(&signal);
            if drivers.as_ref().is_some_and(|d| d.len() == 1) {
                // okay, exactly one driver
                continue;
            }

            let (decl_span, diag_str, kind_str) = match signal {
                Signal::Port(signal) => {
                    let info = &ctx.ports[signal];
                    (info.span, info.name.as_str(), "port")
                }
                Signal::Wire(signal) => {
                    let info = &ctx.wires[signal];
                    (info.decl_span(), info.diagnostic_str(), "wire")
                }
            };

            if let Some(drivers) = drivers {
                // error: multiple drivers
                let mut diag = DiagnosticError::new(
                    format!("{kind_str} `{diag_str}` has multiple drivers"),
                    decl_span,
                    format!("{kind_str} declared here"),
                );

                for (kind, span) in drivers {
                    let kind_str = match kind {
                        DriverKind::WireDeclaration => "wire declaration expression",
                        DriverKind::CombinatorialProcess => "combinatorial process",
                        DriverKind::ClockedProcessRegister => "clocked process register",
                        DriverKind::InstanceOutputPort => "instance output port",
                    };
                    diag = diag.add_info(span, format!("driven by {kind_str} here"));
                }

                any_driver_err = Err(diag.report(self.diags));
            } else {
                // warning: no drivers
                DiagnosticWarning::new(
                    format!("{kind_str} `{diag_str}` has no driver"),
                    decl_span,
                    "declared here",
                )
                .report(self.diags);
            }
        }
        if !drivers.is_empty() {
            return Err(self
                .diags
                .report_error_internal(def_span, "leftover drivers after checking all signals"));
        }

        any_driver_err?;

        // finish building the ir module
        let debug_info_location = match self.fixed.hierarchy.file_steps(def_id.span().file) {
            None => "unknown".to_string(),
            Some(steps) => steps.join(","),
        };
        Ok(IrModuleInfo {
            ports: ctx_body.ir_ports,
            wires: ctx_body.ir_wires,
            large: ctx.large,
            children: ctx_body.children,
            debug_info_location,
            debug_info_id: def_id.spanned_string(self.fixed.source),
            debug_info_generic_args: debug_info_params,
        })
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
                            scope_ports.declare_root(diags, Ok(id.spanned_str(source)), entry);
                        }
                        ModulePortSingleKind::Interface {
                            span_keyword: _,
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
                            scope_ports.declare_root(diags, Ok(id.spanned_str(source)), entry);
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
                                    scope_ports.declare_root(diags, Ok(id.spanned_str(source)), entry);
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
                                    scope_ports.declare_root(diags, Ok(id.spanned_str(source)), entry);
                                }
                            }
                            Ok(())
                        };

                    ctx.elaborate_extra_list_block(scope_ports, flow, ports, &mut visit_port_item_in_block)?;
                }
            }
            Ok(())
        };

        ctx.elaborate_extra_list(&mut scope_ports_root, flow, &ports.inner, &mut visit_port_item)?;

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

        let entry = ScopedEntry::Named(NamedValue::Port(port));
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
        for (port_index, (_, port)) in enumerate(&interface.ports) {
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

        let entry = ScopedEntry::Named(NamedValue::PortInterface(port_interface));
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

struct BodyContext {
    ir_ports: Arena<IrPort, IrPortInfo>,
    ir_wires: Arena<IrWire, IrWireInfo>,

    drivers: IndexMap<Signal, Vec<(DriverKind, Span)>>,
    children: Vec<Spanned<IrModuleChild>>,

    delayed_err: DiagResult,
}

#[derive(Debug, Copy, Clone)]
pub enum DriverKind {
    WireDeclaration,
    CombinatorialProcess,
    ClockedProcessRegister,
    InstanceOutputPort,
}

enum ConnectionSignal {
    Signal(DomainSignal),
    Dummy(Span),
    Expression(Span),
}

impl BodyContext {
    fn elaborate_module_statement(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &mut ExtraScope,
        flow: &mut FlowCompile,
        stmt: &ModuleStatement,
    ) -> DiagResult {
        match &stmt.inner {
            ModuleStatementKind::WireDeclaration(decl) => {
                self.elaborate_wire_declaration(ctx, scope, flow, Spanned::new(stmt.span, decl))
            }
            ModuleStatementKind::CombinatorialProcess(block) => {
                self.elaborate_combinatorial_block(ctx, scope.as_scope(), flow, Spanned::new(stmt.span, block))
            }
            ModuleStatementKind::ClockedProcess(block) => {
                self.elaborate_clocked_block(ctx, scope.as_scope(), flow, Spanned::new(stmt.span, block))
            }
            ModuleStatementKind::Instance(inst) => {
                self.elaborate_instance(ctx, scope.as_scope(), flow, Spanned::new(stmt.span, inst))
            }
        }
    }

    fn elaborate_wire_declaration(
        &mut self,
        ctx: &mut CompileItemContext,
        scope_extra: &mut ExtraScope,
        flow_parent: &mut FlowCompile,
        stmt: Spanned<&WireDeclaration>,
    ) -> DiagResult {
        let &WireDeclaration {
            vis,
            span_keyword,
            id,
            kind,
        } = stmt.inner;

        let refs = ctx.refs;
        let diags = refs.diags;
        let elab = &refs.shared.elaboration_arenas;

        let scope = scope_extra.as_scope();

        // evaluate id
        let id = ctx.eval_maybe_general_id(scope, flow_parent, id)?;
        let id_owned = id
            .as_ref()
            .map_id(|id| id.as_ref().map_inner(|s| s.as_ref().to_owned()));

        // evaluate kind: ty/value
        let named_value = match kind {
            WireDeclarationKind::Normal {
                domain_ty,
                assign_span_and_value,
            } => {
                // create wire immediately, we'll fill in the domain and type later
                let wire = ctx.wires.push(WireInfo::Single(WireInfoSingle {
                    id: id_owned,
                    domain: Ok(None),
                    typed: Ok(None),
                }));

                let (domain, ty) = match domain_ty {
                    WireDeclarationDomainTyKind::Clock { span_clock } => (
                        Some(Spanned::new(span_clock, ValueDomain::Clock)),
                        Some(Spanned::new(span_clock, HardwareType::Bool)),
                    ),
                    WireDeclarationDomainTyKind::Normal { domain, ty } => {
                        let domain = domain
                            .map(|domain| {
                                let domain = ctx.eval_domain(scope, flow_parent, domain)?;
                                Ok(domain.map_inner(ValueDomain::from_domain_kind))
                            })
                            .transpose();
                        let ty = ty
                            .map(|ty| ctx.eval_expression_as_ty_hardware(scope, flow_parent, ty, "wire"))
                            .transpose();

                        let domain = domain?;
                        let ty = ty?;

                        (domain, ty)
                    }
                };

                match assign_span_and_value {
                    None => {
                        // just set the domain and type
                        if let Some(domain) = domain {
                            ctx.wires[wire].suggest_domain(&mut ctx.wire_interfaces, domain)?;
                        }
                        if let Some(ty) = ty.as_ref() {
                            ctx.wires[wire].suggest_ty(refs, &ctx.wire_interfaces, &mut self.ir_wires, ty.as_ref())?;
                        }
                    }
                    Some((assign_span, value)) => {
                        // eval value
                        let expected_ty = ty.as_ref().map_or(Type::Any, |ty| ty.inner.as_type());

                        let flow_kind = HardwareProcessKind::WireExpression {
                            span_keyword,
                            span_init: value.span,
                        };
                        let mut flow_value =
                            FlowHardwareRoot::new(flow_parent, value.span, flow_kind, &mut self.ir_wires);
                        let value = ctx.eval_expression(scope, &mut flow_value.as_flow(), &expected_ty, value)?;
                        let (ir_vars, mut ir_block) = flow_value.finish();

                        // infer or check domain
                        let value_domain = value.as_ref().map_inner(|v| v.domain());
                        let domain = match domain {
                            None => Ok(value_domain),
                            Some(domain) => ctx
                                .check_valid_domain_crossing(
                                    assign_span,
                                    domain,
                                    value_domain,
                                    "wire declaration value",
                                )
                                .map(|()| domain),
                        };

                        // infer or check type
                        let ty = match ty {
                            None => match value.inner.ty().as_hardware_type(elab) {
                                Ok(ty) => Ok(Spanned::new(value.span, ty)),
                                Err(e) => {
                                    let _: NonHardwareType = e;
                                    let err_msg = format!(
                                        "value with type `{}` cannot be represented in hardware",
                                        value.inner.ty().value_string(elab)
                                    );
                                    let diag = DiagnosticError::new(
                                        "cannot assign non-hardware value to wire",
                                        value.span,
                                        err_msg,
                                    )
                                    .add_info(assign_span, "assignment to wire here")
                                    .report(diags);
                                    Err(diag)
                                }
                            },
                            Some(ty) => {
                                let reason = TypeContainsReason::Assignment {
                                    span_target: id.span(),
                                    span_target_ty: ty.span,
                                };
                                check_type_contains_value(diags, elab, reason, &ty.inner.as_type(), value.as_ref())
                                    .map(|()| ty)
                            }
                        };

                        let domain = domain?;
                        let ty = ty?;

                        // create the wire by suggesting the domain and ty
                        let wire_info = &mut ctx.wires[wire];
                        wire_info.suggest_domain(&mut ctx.wire_interfaces, domain)?;
                        let wire_info_typed =
                            wire_info.suggest_ty(refs, &ctx.wire_interfaces, &mut self.ir_wires, ty.as_ref())?;

                        // append final assignment to process
                        let expr_hw = value.inner.as_ir_expression_unchecked(
                            refs,
                            &mut ctx.large,
                            value.span,
                            wire_info_typed.ty.inner,
                        )?;
                        let target = IrAssignmentTarget::simple(wire_info_typed.ir.into());

                        ir_block
                            .statements
                            .push(Spanned::new(stmt.span, IrStatement::Assign(target, expr_hw)));

                        // record process
                        let process = IrCombinatorialProcess {
                            locals: ir_vars,
                            block: ir_block,
                        };
                        self.children
                            .push(Spanned::new(stmt.span, IrModuleChild::CombinatorialProcess(process)));

                        self.report_driver(wire.into(), DriverKind::WireDeclaration, assign_span);
                    }
                }

                NamedValue::Wire(wire)
            }
            WireDeclarationKind::Interface {
                domain,
                span_keyword,
                interface,
            } => {
                // eval domain and interface
                let domain = domain
                    .map(|domain| {
                        let domain = ctx.eval_domain(scope, flow_parent, domain)?;
                        Ok(domain.map_inner(ValueDomain::from_domain_kind))
                    })
                    .transpose();

                let interface = ctx
                    .eval_expression_as_compile(
                        scope,
                        flow_parent,
                        &Type::Interface,
                        interface,
                        Spanned::new(interface.span, "wire interface"),
                    )
                    .and_then(|interface| match interface.inner {
                        CompileValue::Simple(SimpleCompileValue::Interface(interface_inner)) => {
                            Ok(Spanned::new(interface.span, interface_inner))
                        }
                        _ => {
                            let diag = DiagnosticError::new(
                                "expected interface value",
                                interface.span,
                                "got non-interface expression",
                            )
                            .add_info(
                                span_keyword,
                                "expected an interface because of this wire interface declaration",
                            )
                            .report(diags);
                            Err(diag)
                        }
                    });

                let domain = domain?;
                let interface = interface?;

                // create interface wire
                let wire_interface = ctx.wire_interfaces.push(WireInterfaceInfo {
                    id: id_owned.clone(),
                    domain: Ok(domain),
                    interface,
                    // these will be filled in immediately after this
                    wires: vec![],
                    ir_wires: vec![],
                });

                // create inner wires
                let interface_info = elab.interface_info(interface.inner);
                let mut wires = vec![];
                let mut ir_wires = vec![];
                for (port_index, (port_name, port_info)) in enumerate(&interface_info.ports) {
                    let ElaboratedInterfacePortInfo { id, ty } = port_info;
                    let ty = ty.as_ref_ok()?;

                    let diagnostic_str = format!("{}.{}", id_owned.diagnostic_str(), port_name);
                    let ir_name = format!("{}_{}", id_owned.diagnostic_str(), port_name);

                    let wire_ir_info = IrWireInfo {
                        ty: ty.inner.as_ir(refs),
                        debug_info_id: Spanned::new(id.span, Some(ir_name)),
                        debug_info_ty: ty.inner.value_string(elab),
                        // will be filled in later during the inference checking pass
                        debug_info_domain: String::new(),
                    };
                    let wire_ir = self.ir_wires.push(wire_ir_info);

                    let wire_info = WireInfoInInterface {
                        decl_span: stmt.span,
                        interface: Spanned::new(interface.span, wire_interface),
                        index: port_index,
                        diagnostic_string: diagnostic_str,
                        ir: wire_ir,
                    };
                    let wire = ctx.wires.push(WireInfo::Interface(wire_info));

                    wires.push(wire);
                    ir_wires.push(wire_ir);
                }

                let wire_interface_info = &mut ctx.wire_interfaces[wire_interface];
                wire_interface_info.wires = wires;
                wire_interface_info.ir_wires = ir_wires;

                NamedValue::WireInterface(wire_interface)
            }
        };

        // declare wire in the right scope
        let id_ref = id.as_ref().map_id(|id| id.as_ref().map_inner(|s| s.as_ref()));
        let entry = ScopedEntry::Named(named_value);
        match vis {
            Visibility::Public { span: _ } => scope_extra.maybe_declare_root(diags, Ok(id_ref), Ok(entry)),
            Visibility::Private => scope_extra.as_scope().maybe_declare(diags, Ok(id_ref), Ok(entry)),
        }

        Ok(())
    }

    fn elaborate_combinatorial_block(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        stmt: Spanned<&CombinatorialProcess>,
    ) -> DiagResult {
        let &CombinatorialProcess {
            span_keyword,
            ref block,
        } = stmt.inner;

        let diags = ctx.refs.diags;

        // elaborate block
        let mut signals_driven = IndexMap::new();
        let flow_kind = HardwareProcessKind::CombinatorialProcessBody {
            span_keyword,
            signals_driven: &mut signals_driven,
        };
        let mut flow = FlowHardwareRoot::new(flow_parent, block.span, flow_kind, &mut self.ir_wires);

        let mut stack = ExitStack::new_root();

        let end = ctx.elaborate_block(scope, &mut flow.as_flow(), &mut stack, block)?;
        end.unwrap_normal(diags, block.span)?;
        let (ir_vars, ir_block) = flow.finish();

        // report drivers
        for (signal, span) in signals_driven {
            self.report_driver(signal, DriverKind::CombinatorialProcess, span);
        }

        // record process
        let process = IrCombinatorialProcess {
            locals: ir_vars,
            block: ir_block,
        };
        self.children
            .push(Spanned::new(stmt.span, IrModuleChild::CombinatorialProcess(process)));

        Ok(())
    }

    fn elaborate_clocked_block(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        stmt: Spanned<&ClockedProcess>,
    ) -> DiagResult {
        let &ClockedProcess {
            span_keyword,
            span_domain,
            clock,
            reset,
            ref block,
        } = stmt.inner;

        let diags = ctx.refs.diags;

        // eval domain
        let clock = ctx.eval_expression_as_domain_signal(scope, flow_parent, clock);
        let reset = reset
            .as_ref()
            .map(|reset| {
                reset
                    .as_ref()
                    .map_inner(|reset| {
                        let &ClockedProcessReset { kind, signal } = reset;
                        let signal = ctx.eval_expression_as_domain_signal(scope, flow_parent, signal)?;
                        Ok(ClockedProcessReset { kind, signal })
                    })
                    .transpose()
            })
            .transpose();
        let (clock, reset) = result_pair(clock, reset)?;

        // check reset domain
        if let Some(reset) = &reset {
            let ClockedProcessReset { kind, signal } = reset.inner;
            match kind.inner {
                // nothing to check for async resets
                // TODO check that the posedge is sync to the clock
                ResetKind::Async => {}
                // check that the reset is sync to the clock
                ResetKind::Sync => {
                    let target = clock.map_inner(|s| ValueDomain::Sync(SyncDomain { clock: s, reset: None }));

                    let source_domain = signal.inner.signal.domain(ctx, signal.span)?;
                    let source = Spanned::new(reset.span, source_domain.inner);
                    ctx.check_valid_domain_crossing(span_domain, target, source, "sync reset")?;
                }
            }
        };

        // map to domain: async reset stays, a sync reset disappears
        let domain = SyncDomain {
            clock: clock.inner,
            reset: reset.as_ref().and_then(|reset| match reset.inner.kind.inner {
                ResetKind::Async => Some(reset.inner.signal.inner),
                ResetKind::Sync => None,
            }),
        };
        let domain = Spanned::new(span_domain, domain);

        // elaborate block
        let mut registers = IndexMap::new();
        let flow_kind = HardwareProcessKind::ClockedProcessBody {
            span_keyword,
            domain,
            registers: &mut registers,
        };
        let mut flow = FlowHardwareRoot::new(flow_parent, block.span, flow_kind, &mut self.ir_wires);

        let mut stack = ExitStack::new_root();
        let end = ctx.elaborate_block(scope, &mut flow.as_flow(), &mut stack, block)?;
        end.unwrap_normal(diags, block.span)?;

        let (ir_vars, ir_block) = flow.finish();

        // report drivers
        for (&signal, info) in &registers {
            self.report_driver(signal, DriverKind::ClockedProcessRegister, info.span);
        }

        // build reset structure
        let (clock_block, async_reset) = match reset {
            None => {
                // check that registers don't need reset values
                for info in registers.into_values() {
                    let RegisterInfo {
                        span: reg_span,
                        ir: _,
                        reset: reg_reset,
                    } = info;
                    match reg_reset.inner {
                        MaybeUndefined::Undefined => {
                            // no reset value is allowed
                        }
                        MaybeUndefined::Defined(_) => {
                            let diag = DiagnosticError::new(
                                "clocked block without reset cannot drive register with reset value",
                                reg_span,
                                "register with reset value declared here",
                            )
                            .add_info(clock.span, "clocked block declared without reset here")
                            .add_info(reg_reset.span, "register reset value defined here")
                            .add_footer_hint("either add a reset to the block or use `undef` as the reset value")
                            .report(diags);
                            self.delayed_err = Err(diag);
                        }
                    }
                }

                (ir_block, None)
            }
            Some(reset) => {
                let reset_ir = Spanned::new(reset.inner.signal.span, ctx.domain_signal_to_ir(reset.inner.signal)?);

                // collect resets
                let mut resets = vec![];
                for info in registers.into_values() {
                    let RegisterInfo {
                        span: reg_span,
                        ir: reg_ir,
                        reset: reg_reset,
                    } = info;
                    match reg_reset.inner {
                        MaybeUndefined::Undefined => {
                            // no reset value, do nothing
                        }
                        MaybeUndefined::Defined(reg_reset) => {
                            resets.push(Spanned::new(reg_span, (reg_ir, reg_reset)));
                        }
                    }
                }

                // build proper reset structure
                match reset.inner.kind.inner {
                    ResetKind::Async => {
                        // async reset has a dedicated structure
                        let reset_info = IrAsyncResetInfo {
                            signal: reset_ir,
                            resets,
                        };
                        (ir_block, Some(reset_info))
                    }
                    ResetKind::Sync => {
                        // sync reset just becomes assignments inside an if branch
                        let reset_statements = resets
                            .into_iter()
                            .map(|reset| {
                                let (signal, reset_value) = reset.inner;
                                let stmt = IrStatement::Assign(IrAssignmentTarget::simple(signal.into()), reset_value);
                                Spanned::new(reset.span, stmt)
                            })
                            .collect_vec();
                        let reset_block = IrBlock {
                            statements: reset_statements,
                        };

                        let if_stmt = IrIfStatement {
                            condition: reset_ir.inner.as_expression(&mut ctx.large),
                            then_block: reset_block,
                            else_block: Some(ir_block),
                        };
                        let root_block = IrBlock {
                            statements: vec![Spanned::new(reset_ir.span, IrStatement::If(if_stmt))],
                        };
                        (root_block, None)
                    }
                }
            }
        };

        // record process
        let clock_ir = Spanned::new(clock.span, ctx.domain_signal_to_ir(clock)?);
        let process = IrClockedProcess {
            locals: ir_vars,
            async_reset,
            clock_signal: clock_ir,
            clock_block,
        };
        self.children
            .push(Spanned::new(stmt.span, IrModuleChild::ClockedProcess(process)));

        Ok(())
    }

    fn elaborate_instance(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        stmt: Spanned<&ModuleInstance>,
    ) -> DiagResult {
        let refs = ctx.refs;
        let diags = refs.diags;
        let source = refs.fixed.source;

        let &ModuleInstance {
            ref name,
            span_keyword,
            module,
            ref port_connections,
        } = stmt.inner;

        // eval module
        let elaborated_module = ctx.eval_expression_as_module(scope, flow_parent, span_keyword, module)?;

        let (instance_info, connectors, def_ports_span) = match elaborated_module {
            ElaboratedModule::Internal(module) => {
                let ElaboratedModuleInternalInfo {
                    ast_ref,
                    unique: _,
                    module_ir,
                    connectors,
                } = ctx.refs.shared.elaboration_arenas.module_internal_info(module);
                (
                    ElaboratedModule::Internal(*module_ir),
                    connectors,
                    refs.fixed.parsed[*ast_ref].ports.span,
                )
            }
            ElaboratedModule::External(module) => {
                let ElaboratedModuleExternalInfo {
                    ast_ref,
                    module_name,
                    generic_args,
                    port_names,
                    connectors,
                } = ctx.refs.shared.elaboration_arenas.module_external_info(module);
                (
                    ElaboratedModule::External((module_name, generic_args, port_names)),
                    connectors,
                    refs.fixed.parsed[*ast_ref].ports.span,
                )
            }
        };

        // eval port connections
        let mut port_connections_eval = vec![];
        {
            // declarations are not allowed here, so no need to worry about expression scopes
            let mut scope_connections = scope.new_child(port_connections.span);
            ctx.elaborate_extra_list(
                &mut scope_connections,
                flow_parent,
                &port_connections.inner,
                &mut |_, _, _, connection| {
                    port_connections_eval.push(connection);
                    Ok(())
                },
            )?;
        }

        // check that connections are unique
        let mut id_to_connection_and_used: IndexMap<&str, (&PortConnection, bool)> = IndexMap::new();
        for connection in port_connections_eval {
            match id_to_connection_and_used.entry(connection.id.str(source)) {
                Entry::Vacant(entry) => {
                    entry.insert((connection, false));
                }
                Entry::Occupied(entry) => {
                    let (prev_connection, _) = entry.get();
                    let diag = DiagnosticError::new("duplicate connection", connection.span(), "connected again here")
                        .add_info(prev_connection.span(), "previous connection here")
                        .report(diags);
                    return Err(diag);
                }
            }
        }

        // match connectors to connections
        // TODO it's a bit weird that these are evaluated in declaration instead of connection order,
        //   but that's much more convenient for domain resolution
        let mut single_to_signal = IndexMap::new();
        let mut ir_connections = vec![];

        for (connector, connector_info) in connectors {
            let connector_id_str = connector_info.id.str(source);
            match id_to_connection_and_used.get_mut(connector_id_str) {
                Some((connection, connection_used)) => {
                    if *connection_used {
                        // this should have already been caught during module header elaboration
                        return Err(diags.report_error_internal(connection.span(), "connection used twice"));
                    }
                    *connection_used = true;

                    let connections = self.elaborate_instance_port_connection(
                        ctx,
                        scope,
                        flow_parent,
                        connectors,
                        &single_to_signal,
                        connector,
                        connection,
                    )?;

                    for (single, signal, ir_connection) in connections {
                        single_to_signal.insert_first(single, signal);
                        ir_connections.push(ir_connection);
                    }
                }
                None => {
                    let diag = DiagnosticError::new(
                        format!("missing connection for port {connector_id_str}"),
                        Span::empty_at(port_connections.span.end()),
                        "connections here",
                    )
                    .add_info(connector_info.id.span, "port declared here")
                    .report(diags);
                    return Err(diag);
                }
            }
        }

        let mut any_unused_err = Ok(());
        for (_, &(connection, used)) in id_to_connection_and_used.iter() {
            if !used {
                let diag = DiagnosticError::new(
                    "connection does not match any port",
                    connection.span(),
                    "invalid connection here",
                )
                .add_info(def_ports_span, "ports declared here")
                .report(diags);
                any_unused_err = Err(diag);
            }
        }
        any_unused_err?;

        // build instance
        let name = name.as_ref().map(|name| name.str(source).to_owned());
        let ir_instance = match instance_info {
            ElaboratedModule::Internal(module_ir) => IrModuleChild::ModuleInternalInstance(IrModuleInternalInstance {
                name,
                module: module_ir,
                port_connections: ir_connections,
            }),
            ElaboratedModule::External((module_name, generic_args, port_names)) => {
                IrModuleChild::ModuleExternalInstance(IrModuleExternalInstance {
                    name,
                    module_name: module_name.clone(),
                    generic_args: generic_args.clone(),
                    port_names: port_names.clone(),
                    port_connections: ir_connections,
                })
            }
        };

        // record instance
        self.children.push(Spanned::new(stmt.span, ir_instance));
        Ok(())
    }

    fn elaborate_instance_port_connection(
        &mut self,
        ctx: &mut CompileItemContext,
        scope: &Scope,
        flow_parent: &mut FlowCompile,
        connectors: &ArenaConnectors,
        prev_single_to_signal: &IndexMap<ConnectorSingle, ConnectionSignal>,
        connector: Connector,
        connection: &PortConnection,
    ) -> DiagResult<Vec<(ConnectorSingle, ConnectionSignal, Spanned<IrPortConnection>)>> {
        let refs = ctx.refs;
        let diags = refs.diags;
        let source = refs.fixed.source;
        let elab = &refs.shared.elaboration_arenas;

        let connection_span = connection.span();
        let &PortConnection {
            id: connection_id,
            expr: value_expr,
        } = &connection;

        let value_expr = value_expr.expr();

        let ConnectorInfo { id: connector_id, kind } = &connectors[connector];

        // double-check id match
        if connector_id.str(source) != connection_id.str(source) {
            return Err(diags.report_error_internal(connection_span, "connection name mismatch"));
        }

        // replace signals that are earlier ports with their connected value
        let map_domain_kind = |domain_span: Span, domain: DomainKind<Polarized<ConnectorSingle>>| {
            Ok(match domain {
                DomainKind::Const => DomainKind::Const,
                DomainKind::Async => DomainKind::Async,
                DomainKind::Sync(sync) => DomainKind::Sync(sync.try_map_signal(|raw_port| {
                    let mapped_port = match prev_single_to_signal.get(&raw_port.signal) {
                        None => throw!(
                            diags.report_error_internal(connection_span, "failed to get signal for previous port")
                        ),
                        Some(&ConnectionSignal::Dummy(dummy_span)) => {
                            let diag = DiagnosticError::new_todo(
                                "dummy port connections that are used in the domain of other ports",
                                dummy_span,
                            )
                            .add_info(domain_span, "port used in a domain here")
                            .report(diags);
                            return Err(diag);
                        }
                        Some(&ConnectionSignal::Expression(expr_span)) => {
                            let diag = DiagnosticError::new_todo(
                                "expression port connections that are used in the domain of other ports",
                                expr_span,
                            )
                            .add_info(domain_span, "port used in a domain here")
                            .report(diags);
                            return Err(diag);
                        }
                        Some(&ConnectionSignal::Signal(signal)) => Ok(signal),
                    }?;
                    Ok(Polarized {
                        signal: mapped_port.signal,
                        inverted: mapped_port.inverted ^ raw_port.inverted,
                    })
                })?),
            })
        };

        // always try to evaluate as signal for domain replacing purposes
        let signal = match &refs.get_expr(value_expr) {
            ExpressionKind::Dummy => ConnectionSignal::Dummy(value_expr.span),
            _ => {
                let mut flow_domain = flow_parent.new_child_isolated();
                match ctx.try_eval_expression_as_domain_signal(scope, &mut flow_domain, value_expr, |_| ()) {
                    Ok(signal) => ConnectionSignal::Signal(signal.inner),
                    Err(Either::Left(())) => ConnectionSignal::Expression(value_expr.span),
                    Err(Either::Right(e)) => throw!(e),
                }
            }
        };

        // evaluate the connection differently depending on the port direction
        match kind.as_ref_ok()? {
            &ConnectorKind::Port {
                direction,
                domain,
                ref ty,
                single,
            } => {
                let connector_domain = domain
                    .map_inner(|d| match d {
                        PortDomain::Clock => Ok(ValueDomain::Clock),
                        PortDomain::Kind(kind) => {
                            Ok(ValueDomain::from_domain_kind(map_domain_kind(domain.span, kind)?))
                        }
                    })
                    .transpose()?;

                let ir_connection = match direction.inner {
                    PortDirection::Input => {
                        // better dummy port error message
                        if let ExpressionKind::Dummy = refs.get_expr(value_expr) {
                            let diag = DiagnosticError::new(
                                "dummy connections are only allowed for output ports",
                                value_expr.span,
                                "dummy connection used here",
                            )
                            .add_info(direction.span, "port declared as input here")
                            .report(diags);
                            return Err(diag);
                        }

                        // eval expr
                        let flow_kind = HardwareProcessKind::InstancePortConnection {
                            span_connection: connection_span,
                        };
                        let mut flow =
                            FlowHardwareRoot::new(flow_parent, value_expr.span, flow_kind, &mut self.ir_wires);
                        let connection_value =
                            ctx.eval_expression(scope, &mut flow.as_flow(), &ty.inner.as_type(), value_expr)?;
                        let (ir_vars, mut ir_block) = flow.finish();

                        // check type
                        let reason = TypeContainsReason::InstancePortInput {
                            span_connection_port_id: connection_id.span,
                            span_port_ty: ty.span,
                        };
                        let check_ty = check_type_contains_value(
                            diags,
                            elab,
                            reason,
                            &ty.inner.as_type(),
                            connection_value.as_ref(),
                        );

                        // check domain
                        let target_domain = Spanned {
                            span: connection_id.span,
                            inner: connector_domain.inner,
                        };
                        let source_domain = connection_value.as_ref().map_inner(|v| v.domain());
                        let check_domain = ctx.check_valid_domain_crossing(
                            connection_span,
                            target_domain,
                            source_domain,
                            "input port connection",
                        );

                        check_ty?;
                        check_domain?;

                        // convert value to ir
                        let connection_value_ir_raw = connection_value
                            .as_ref()
                            .map_inner(|v| {
                                Ok(v.as_hardware_value_unchecked(
                                    refs,
                                    &mut ctx.large,
                                    value_expr.span,
                                    ty.inner.clone(),
                                )?
                                .expr)
                            })
                            .transpose()?;

                        // build extra wire and process if necessary
                        // TODO rework this, because of flow reworks this never triggers any more
                        let connection_signal_ir = if ir_block.statements.is_empty()
                            && let IrExpression::Signal(connection_signal) = connection_value_ir_raw.inner
                        {
                            connection_signal
                        } else {
                            let extra_ir_wire = self.ir_wires.push(IrWireInfo {
                                ty: ty.inner.as_ir(refs),
                                debug_info_id: connector_id.spanned_string(source).map_inner(Some),
                                debug_info_ty: ty.inner.clone().value_string(elab),
                                debug_info_domain: connection_value.inner.domain().diagnostic_string(ctx),
                            });

                            ir_block.statements.push(Spanned {
                                span: connection_span,
                                inner: IrStatement::Assign(
                                    IrAssignmentTarget::simple(extra_ir_wire.into()),
                                    connection_value_ir_raw.inner,
                                ),
                            });
                            let process = IrCombinatorialProcess {
                                locals: ir_vars,
                                block: ir_block,
                            };
                            let child = IrModuleChild::CombinatorialProcess(process);
                            self.children.push(Spanned::new(connection_span, child));

                            IrSignal::Wire(extra_ir_wire)
                        };

                        IrPortConnection::Input(connection_signal_ir)
                    }
                    PortDirection::Output => {
                        // eval expr as dummy, wire or port
                        let build_error = || {
                            diags.report_error_simple(
                                "output port must be connected to wire or port",
                                value_expr.span,
                                "other value",
                            )
                        };

                        match refs.get_expr(value_expr) {
                            ExpressionKind::Dummy => IrPortConnection::Output(None),
                            &ExpressionKind::Id(id) => {
                                let id = ctx.eval_general_id(scope, flow_parent, id)?;
                                let id = id.as_ref().map_inner(ArcOrRef::as_ref);
                                let named = ctx.eval_named_or_value(scope, id)?;

                                let (signal_ir, signal_target, signal_domain, signal_ty) = match named.inner {
                                    NamedOrValue::Named(NamedValue::Wire(wire)) => {
                                        let wire_info = &mut ctx.wires[wire];

                                        let wire_domain =
                                            wire_info.suggest_domain(&mut ctx.wire_interfaces, connector_domain);
                                        let wire_ty = wire_info.suggest_ty(
                                            refs,
                                            &ctx.wire_interfaces,
                                            &mut self.ir_wires,
                                            ty.as_ref(),
                                        );

                                        let wire_domain = wire_domain?;
                                        let wire_ty = wire_ty?;

                                        (IrSignal::Wire(wire_ty.ir), Signal::Wire(wire), wire_domain, wire_ty.ty)
                                    }
                                    NamedOrValue::Named(NamedValue::Port(port)) => {
                                        let port_info = &ctx.ports[port];
                                        (
                                            IrSignal::Port(port_info.ir),
                                            Signal::Port(port),
                                            port_info.domain.map_inner(ValueDomain::from_port_domain),
                                            port_info.ty.as_ref(),
                                        )
                                    }
                                    _ => throw!(build_error()),
                                };

                                // check type
                                let mut any_err = Ok(());
                                let reason = TypeContainsReason::InstancePortOutput {
                                    span_connection_signal_id: value_expr.span,
                                    span_signal_ty: signal_ty.span,
                                };
                                any_err = any_err.and(check_type_contains_type(
                                    diags,
                                    elab,
                                    reason,
                                    &signal_ty.inner.as_type(),
                                    Spanned {
                                        span: connection_id.span,
                                        inner: &ty.inner.as_type(),
                                    },
                                ));

                                // check domain
                                any_err = any_err.and(ctx.check_valid_domain_crossing(
                                    connection_span,
                                    signal_domain,
                                    connector_domain,
                                    "output port connection",
                                ));

                                // report driver
                                self.report_driver(signal_target, DriverKind::InstanceOutputPort, value_expr.span);

                                // success, build connection
                                any_err?;
                                IrPortConnection::Output(Some(signal_ir))
                            }
                            _ => throw!(build_error()),
                        }
                    }
                };

                let spanned_ir_connection = Spanned {
                    span: connection_span,
                    inner: ir_connection,
                };
                Ok(vec![(single, signal, spanned_ir_connection)])
            }
            ConnectorKind::Interface {
                domain: connector_domain,
                view: connector_view,
                singles: connector_singles,
            } => {
                let connector_domain = connector_domain
                    .map_inner(|d| {
                        Ok(ValueDomain::from_domain_kind(map_domain_kind(
                            connector_domain.span,
                            d,
                        )?))
                    })
                    .transpose()?;

                // eval expr
                let mut flow_connection = flow_parent.new_child_isolated();
                let value = ctx.eval_expression_inner(scope, &mut flow_connection, &Type::Any, value_expr)?;

                // unwrap interface
                // TODO avoid cloning signals vec here
                let (value_interface, value_domain, value_signals) = match value {
                    ValueInner::PortInterface(port_interface) => {
                        let info = &ctx.port_interfaces[port_interface];
                        let port_interface = info.view.map_inner(|v| v.interface);
                        let port_domain = info
                            .domain
                            .map_inner(|d| ValueDomain::from_domain_kind(d.map_signal(|s| s.map_inner(Signal::Port))));
                        let port_signals = Signal::Port(&info.ports);
                        (port_interface, port_domain, port_signals)
                    }
                    ValueInner::WireInterface(wire_interface) => {
                        let info = &mut ctx.wire_interfaces[wire_interface];
                        let wire_domain = info.suggest_domain(connector_domain)?;
                        // reborrow immutably
                        let info = &ctx.wire_interfaces[wire_interface];
                        let wire_signals = Signal::Wire((&info.wires, &info.ir_wires));
                        (info.interface, wire_domain, wire_signals)
                    }
                    ValueInner::Value(_) => {
                        let diag = DiagnosticError::new(
                            "expected interface value",
                            value_expr.span,
                            "got non-interface expression",
                        )
                        .add_info(connector_id.span, "port defined as interface here")
                        .report(diags);
                        return Err(diag);
                    }
                };

                // check interface match (including generics)
                if value_interface.inner != connector_view.inner.interface {
                    let diag = DiagnosticError::new("interface mismatch", value_expr.span, "got mismatching interface")
                        .add_info(
                            connector_view.span,
                            format!(
                                "expected interface `{}` set here",
                                SimpleCompileValue::Interface(connector_view.inner.interface).value_string(elab)
                            ),
                        )
                        .add_info(
                            value_interface.span,
                            format!(
                                "actual interface `{}` set here",
                                SimpleCompileValue::Interface(value_interface.inner).value_string(elab)
                            ),
                        )
                        .report(diags);
                    return Err(diag);
                }

                // check directions and build connections
                let interface_info = refs
                    .shared
                    .elaboration_arenas
                    .interface_info(connector_view.inner.interface);
                let view_info = &interface_info.views[connector_view.inner.view_index];

                let mut any_input = false;
                let mut any_output = false;

                let mut result_connections = vec![];

                for port_index in 0..interface_info.ports.len() {
                    let (_, connector_dir) = &view_info.port_dirs.as_ref_ok()?[port_index];

                    // check direction
                    let (value_dir, value_signal, value_ir) = match value_signals {
                        Signal::Port(ports) => {
                            let port = ports[port_index];
                            let info = &ctx.ports[port];
                            (Some(info.direction), Signal::Port(port), IrSignal::Port(info.ir))
                        }
                        Signal::Wire((wires, ir_wires)) => {
                            let wire = wires[port_index];
                            (None, Signal::Wire(wire), IrSignal::Wire(ir_wires[port_index]))
                        }
                    };
                    if let Some(value_dir) = value_dir
                        && connector_dir.inner != value_dir.inner
                    {
                        let diag = DiagnosticError::new(
                            format!(
                                "direction mismatch for interface port `{}`",
                                interface_info.ports[port_index].id.str(source)
                            ),
                            value_expr.span,
                            format!("got direction `{}`", value_dir.inner.diagnostic_string()),
                        )
                        .add_info(
                            connection_id.span,
                            format!("expected direction `{}`", connector_dir.inner.diagnostic_string()),
                        )
                        .add_info(connector_dir.span, "expected direction set here")
                        .add_info(value_dir.span, "actual direction set here")
                        .report(diags);
                        return Err(diag);
                    }
                    let dir = connector_dir.inner;

                    // build connection
                    let ir_connection = match dir {
                        PortDirection::Input => {
                            any_input = true;
                            IrPortConnection::Input(value_ir)
                        }
                        PortDirection::Output => {
                            any_output = true;
                            self.report_driver(value_signal, DriverKind::InstanceOutputPort, value_expr.span);
                            IrPortConnection::Output(Some(value_ir))
                        }
                    };

                    // build signal
                    let signal = ConnectionSignal::Signal(Polarized {
                        inverted: false,
                        signal: value_signal,
                    });
                    result_connections.push((
                        connector_singles[port_index],
                        signal,
                        Spanned::new(connection_span, ir_connection),
                    ))
                }

                // check domains
                let mut any_err_domain = Ok(());
                if any_input {
                    let r = ctx.check_valid_domain_crossing(
                        connection_span,
                        value_domain,
                        connector_domain,
                        "interface connection with input port",
                    );
                    any_err_domain = any_err_domain.and(r);
                }
                if any_output {
                    let r = ctx.check_valid_domain_crossing(
                        connection_span,
                        connector_domain,
                        value_domain,
                        "interface connection with output port",
                    );
                    any_err_domain = any_err_domain.and(r);
                }
                any_err_domain?;

                Ok(result_connections)
            }
        }
    }

    fn report_driver(&mut self, signal: Signal, kind: DriverKind, span: Span) {
        self.drivers.entry(signal).or_default().push((kind, span))
    }
}
