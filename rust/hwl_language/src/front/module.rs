use crate::data::compiled::{Item, ModulePort, ModulePortInfo, Register, RegisterInfo, Wire, WireInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::data::module_body::{LowerBlock, LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleChecked, ModuleInstance, ModuleStatement};
use crate::front::block::AccessDirection;
use crate::front::checking::DomainUserControlled;
use crate::front::common::{ContextDomain, ExpressionContext, GenericContainer, GenericMap, ScopedEntry, ScopedEntryDirect, TypeOrValue, ValueDomain};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{Constructor, GenericArguments, PortConnections, Type};
use crate::front::values::{ModuleValueInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{ClockedBlock, CombinatorialBlock, Identifier, ModuleStatementKind, PortDirection, PortKind, RegDeclaration, Spanned, WireDeclaration};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use annotate_snippets::Level;
use indexmap::map::Entry;
use indexmap::IndexMap;
use itertools::{zip_eq, Itertools};
use std::cmp::min;
use std::hash::Hash;
use unwrap_match::unwrap_match;

impl<'d, 'a> CompileState<'d, 'a> {
    pub fn check_module_body(&mut self, module_item: Item, module_ast: &'a ast::ItemDefModule) -> ModuleChecked {
        let ast::ItemDefModule { span: _, vis: _, id: _, params: _, ports: _, body } = module_ast;
        let ast::Block { span: _, statements: module_statements } = body;

        // TODO do we event want to convert to some simpler IR here,
        //   or just leave the backend to walk the AST if it wants?
        let mut module_statements_lower = vec![];
        let mut module_regs = IndexMap::new();
        let mut module_wires = vec![];
        let mut output_port_regs = IndexMap::new();

        let scope_ports = self.compiled.module_info[&module_item].scope_ports;
        let scope_body = self.compiled.scopes.new_child(scope_ports, body.span, Visibility::Private);

        let mut drivers = Drivers::default();

        // create entry for each output port
        for &port in &self.compiled.module_info.get(&module_item).unwrap().ports {
            match self.compiled[port].direction {
                PortDirection::Input => {}
                PortDirection::Output => {
                    drivers.output_port_drivers.entry(port).or_default();
                }
            }
        }

        // first pass: populate scope with declarations
        // TODO fully implement graph-ness,
        //   in the current implementation eg. types and initializes still can't refer to future regs and wires
        for module_statement in module_statements {
            match &module_statement.inner {
                ModuleStatementKind::ConstDeclaration(decl) => {
                    self.process_and_declare_const(scope_body, decl, Visibility::Private);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let (reg, init) = self.process_module_declaration_reg(module_item, scope_body, decl);

                    // create entry, to ensure ordering
                    drivers.reg_drivers.entry(reg).or_default();

                    module_regs.insert_first(reg, init);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let (wire, value) = self.process_module_declaration_wire(module_item, scope_body, decl);

                    // create entry, to ensure ordering
                    drivers.wire_drivers.entry(wire).or_default();

                    // insert driver if there is a value
                    if value.is_some() {
                        drivers.wire_drivers
                            .entry(wire).or_default()
                            .entry(Driver::WireDeclaration).or_default()
                            .push(decl.id.span());
                    }

                    module_wires.push((wire, value));
                }
                ModuleStatementKind::RegOutPortMarker(marker) => {
                    self.process_reg_out_port_marker(module_item, module_ast, scope_body, &mut output_port_regs, marker);
                }
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
            }
        }

        // second pass: codegen for the actual blocks
        for module_statement in module_statements {
            let lower_statement_index = module_statements_lower.len();

            match &module_statement.inner {
                // declarations were already handled
                ModuleStatementKind::ConstDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                ModuleStatementKind::RegOutPortMarker(_) => {}
                // actual blocks
                ModuleStatementKind::CombinatorialBlock(ref comb_block) => {
                    let block = self.process_module_block_combinatorial(&mut drivers, scope_body, lower_statement_index, comb_block);
                    module_statements_lower.push(ModuleStatement::Combinatorial(block));
                }
                ModuleStatementKind::ClockedBlock(ref clocked_block) => {
                    let block = self.process_module_block_clocked(&mut drivers, scope_body, lower_statement_index, clocked_block);
                    module_statements_lower.push(ModuleStatement::Clocked(block));
                }
                // instances
                ModuleStatementKind::Instance(instance) => {
                    let block = self.process_module_instance(&mut drivers, scope_body, lower_statement_index, instance);
                    module_statements_lower.push(block);
                }
            }
        }

        let output_port_driver = self.check_driver_validness(&mut module_statements_lower, &module_regs, &drivers, &output_port_regs);

        ModuleChecked {
            statements: module_statements_lower,
            regs: module_regs.keys().copied().collect_vec(),
            wires: module_wires,
            output_port_driver,
        }
    }

    fn check_driver_validness(
        &self,
        module_statements_lower: &mut Vec<ModuleStatement>,
        module_regs: &IndexMap<Register, Spanned<Value>>,
        drivers: &Drivers,
        output_port_regs: &IndexMap<ModulePort, (&'a Identifier, Spanned<Value>)>,
    ) -> IndexMap<ModulePort, Driver> {
        let Drivers { output_port_drivers, reg_drivers, wire_drivers } = drivers;

        // check ports
        let mut output_port_driver_clean = IndexMap::new();

        for (&port, drivers) in output_port_drivers {
            let reg_def_id = &self.parsed.module_port_ast(self.compiled[port].ast).id;

            let reg_output_port = output_port_regs.get(&port);

            let mut any_err = None;

            for (driver, spans) in drivers {
                match (driver.kind(), reg_output_port) {
                    (DriverKind::WiredConnection, None) => {}
                    (DriverKind::ClockedBlock, Some(_)) => {}
                    (DriverKind::WiredConnection, Some((reg_marker_id, _))) => {
                        let mut diag = Diagnostic::new("register output port can only be driven by clocked block");
                        for &span in spans {
                            diag = diag.add_error(span, "driven incorrectly here");
                        }
                        diag = diag.add_info(reg_marker_id.span, "output port marked as register here");
                        any_err = Some(self.diags.report(diag.add_info(reg_def_id.span, "output port declared here").finish()));
                    }
                    (DriverKind::ClockedBlock, None) => {
                        let mut diag = Diagnostic::new("non-register output port cannot be driven by a clocked block");
                        for &span in spans {
                            diag = diag.add_error(span, "driven incorrectly here");
                        }

                        let help_message = format!(
                            "the port can be marked as a register by adding `reg out {} = <reset_value>;` to the module body",
                            reg_def_id.string
                        );
                        diag = diag.footer(Level::Help, help_message);

                        any_err = Some(self.diags.report(diag.add_info(reg_def_id.span, "output port declared here").finish()));
                    }
                }
            }

            if any_err.is_some() {
                continue;
            }

            let single_driver = self.check_exactly_one_driver("output port", reg_def_id.span, drivers);

            match single_driver {
                Ok(driver) => {
                    output_port_driver_clean.insert_first(port, driver);

                    if let Some((reg_marker_id, init)) = reg_output_port {
                        pull_reg_init_into_single_clocked_block_driver(
                            driver,
                            module_statements_lower,
                            reg_def_id.span,
                            Spanned { span: reg_marker_id.span, inner: Value::ModulePort(port) },
                            init.clone(),
                        );
                    }
                }
                Err(e) => {
                    let _: ErrorGuaranteed = e;
                }
            }
        }

        // check wires
        for (&wire, drivers) in wire_drivers {
            let def_span = self.compiled[wire].defining_id.span();

            for (driver, spans) in drivers {
                match driver.kind() {
                    DriverKind::WiredConnection => {}
                    DriverKind::ClockedBlock => {
                        let mut diag = Diagnostic::new("wire cannot be driven by clocked block");
                        for &span in spans {
                            diag = diag.add_error(span, "driven incorrectly here");
                        }
                        self.diags.report(
                            diag
                                .footer(Level::Help, "either drive the wire from a combinatorial block or change the wire into a register")
                                .add_info(def_span, "wire declared here")
                                .finish()
                        );
                    }
                }
            }

            let _ = self.check_exactly_one_driver("wire", def_span, drivers);
        }

        // check registers
        for (&reg, drivers) in reg_drivers {
            let reg_def_id_span = self.compiled[reg].defining_id.span();

            // check that register is only driven by clocked block
            let mut any_err = None;

            for (driver, spans) in drivers {
                match driver.kind() {
                    DriverKind::ClockedBlock => {}
                    DriverKind::WiredConnection => {
                        let mut diag = Diagnostic::new("register can only be driven by clocked block");
                        for &span in spans {
                            diag = diag.add_error(span, "driven incorrectly here");
                        }
                        any_err = Some(self.diags.report(diag.add_info(reg_def_id_span, "register declared here").finish()));
                    }
                }
            }

            if any_err.is_some() {
                continue;
            }

            // check that register has a single driver, and pull the reset value into the right block
            let single_driver = self.check_exactly_one_driver("register", reg_def_id_span, drivers);
            match single_driver {
                Ok(driver) => {
                    pull_reg_init_into_single_clocked_block_driver(
                        driver,
                        module_statements_lower,
                        reg_def_id_span,
                        Spanned { span: reg_def_id_span, inner: Value::Register(reg) },
                        module_regs.get(&reg).unwrap().clone(),
                    );
                }
                Err(e) => {
                    let _: ErrorGuaranteed = e;
                }
            }
        }

        output_port_driver_clean
    }

    fn check_exactly_one_driver(&self, kind: &str, declared_span: Span, drivers: &IndexMap<Driver, Vec<Span>>) -> Result<Driver, ErrorGuaranteed> {
        let diags = self.diags;
        match drivers.len() {
            // good, exactly one driver
            1 => Ok(*drivers.keys().next().unwrap()),
            // no drivers
            0 => {
                Err(diags.report_simple(format!("{kind} has no drivers"), declared_span, format!("{kind} declared here")))
            }
            // too many drivers
            _ => {
                let mut diag = Diagnostic::new(format!("{kind} has multiple drivers"));
                for (_, spans) in drivers {
                    diag = diag.add_error(spans[0], "driven here");
                }
                Err(diags.report(diag.add_info(declared_span, format!("{kind} declared here")).finish()))
            }
        }
    }

    fn process_reg_out_port_marker(
        &mut self,
        module_item: Item,
        module_ast: &'a ast::ItemDefModule,
        scope_body: Scope,
        output_port_regs: &mut IndexMap<ModulePort, (&'a Identifier, Spanned<Value>)>,
        marker: &'a ast::RegOutPortMarker,
    ) {
        let diags = self.diags;
        let ast::RegOutPortMarker { span: _, id, init } = marker;

        let init_span = init.span;
        let init_eval = self.eval_expression_as_value(
            &ExpressionContext::constant(init.span, scope_body),
            &mut MaybeDriverCollector::None,
            init,
        );

        let port = self.compiled.module_info[&module_item].ports
            .iter()
            .find(|&&port| {
                self.parsed.module_port_ast(self.compiled[port].ast).id.string == id.string
            });

        match port {
            None => {
                let diag = Diagnostic::new(format!("could not find port with name `{}`", id.string))
                    .add_error(id.span, "marker here")
                    .add_info(module_ast.ports.span, "module ports declared here")
                    .finish();
                diags.report(diag);
            }
            Some(&port) => {
                let port_info = &self.compiled[port];
                let port_ast = self.parsed.module_port_ast(port_info.ast);

                // check init value
                let port_ty_spanned = Spanned { span: port_ast.kind.span, inner: &port_info.kind.ty() };
                let init_eval = self.check_reg_init_value(id.span, port_ty_spanned, Spanned { span: init.span, inner: init_eval });

                let port_info = &self.compiled[port];
                let port_ast = self.parsed.module_port_ast(port_info.ast);

                // check port direction
                match port_info.direction {
                    PortDirection::Input => {
                        diags.report(Diagnostic::new("only output ports can be marked as registers")
                            .add_error(id.span, "marker applying to input port here")
                            .add_info(port_ast.direction.span, "port direction set to output here")
                            .finish()
                        );
                    }
                    PortDirection::Output => {
                        match output_port_regs.entry(port) {
                            Entry::Occupied(entry) => {
                                let (prev_id, _) = entry.get();
                                let diag = Diagnostic::new("output port already marked as register")
                                    .add_error(id.span, "marker here")
                                    .add_info(prev_id.span, "previous definition here")
                                    .finish();
                                diags.report(diag);
                            }
                            Entry::Vacant(entry) => {
                                entry.insert((id, Spanned { span: init_span, inner: init_eval }));
                            }
                        }
                    }
                }
            }
        }
    }

    fn process_module_declaration_reg(&mut self, module_item: Item, scope_body: Scope, decl: &RegDeclaration) -> (Register, Spanned<Value>) {
        let RegDeclaration { span: _, id, sync, ty, init } = decl;

        // eval everything
        let sync = sync.as_ref().map_inner(|sync| {
            sync.as_ref().map_inner(|v| &**v)
        });
        let sync = self.eval_sync_domain(scope_body, sync);

        let ty_eval = self.eval_expression_as_ty(scope_body, ty);
        let ty_spanned = Spanned { span: ty.span, inner: &ty_eval };

        let init_span = init.span;
        let init_eval = self.eval_expression_as_value(
            &ExpressionContext::constant(init.span, scope_body),
            &mut MaybeDriverCollector::None,
            init,
        );
        let init = Spanned { span: init.span, inner: init_eval };
        let init_eval = self.check_reg_init_value(id.span(), ty_spanned, init);

        // create register
        let reg = self.compiled.registers.push(RegisterInfo {
            defining_item: module_item,
            defining_id: id.clone(),
            domain: sync,
            ty: ty_eval,
        });

        let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Register(reg))));
        self.compiled[scope_body].maybe_declare(&self.diags, id.as_ref(), entry, Visibility::Private);

        let init_spanned = Spanned { span: init_span, inner: init_eval };
        (reg, init_spanned)
    }

    fn check_reg_init_value(&mut self, id_span: Span, ty: Spanned<&Type>, init: Spanned<Value>) -> Value {
        // check ty
        let init_eval = match self.check_type_contains(Some(ty.span), init.span, &ty.inner, &init.inner) {
            Ok(()) => init.inner,
            Err(e) => Value::Error(e),
        };

        // check domain
        let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
            id_span,
            &ValueDomain::CompileTime,
            init.span,
            &self.domain_of_value(init.span, &init_eval),
            DomainUserControlled::Source,
            "register initial value must be const",
        );

        init_eval
    }

    fn process_module_declaration_wire(
        &mut self, module_item: Item,
        scope_body: Scope,
        decl: &WireDeclaration,
    ) -> (Wire, Option<Value>) {
        let WireDeclaration { span: _, id, sync, ty, value } = decl;

        let sync = self.eval_domain(scope_body, sync.as_ref());
        let ty = self.eval_expression_as_ty(scope_body, ty);

        let value = match value {
            Some(value) => {
                let value_unchecked = self.eval_expression_as_value(
                    &ExpressionContext::passthrough(scope_body),
                    &mut MaybeDriverCollector::None,
                    value,
                );

                // check type
                let value_eval = match self.check_type_contains(Some(value.span), value.span, &ty, &value_unchecked) {
                    Ok(()) => value_unchecked,
                    Err(e) => Value::Error(e),
                };

                // check domain
                let _: Result<(), ErrorGuaranteed> = self.check_domain_crossing(
                    id.span(),
                    &ValueDomain::from_domain_kind(sync.clone()),
                    value.span,
                    &self.domain_of_value(value.span, &value_eval),
                    DomainUserControlled::Source,
                    "wire value must be assignable to the wire domain",
                );

                Some(value_eval)
            }
            None => None,
        };

        // create wire
        let wire = self.compiled.wires.push(WireInfo {
            defining_item: module_item,
            defining_id: id.clone(),
            domain: sync,
            ty,
            has_declaration_value: value.is_some(),
        });

        let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Wire(wire))));
        self.compiled[scope_body].maybe_declare(&self.diags, id.as_ref(), entry, Visibility::Private);

        (wire, value)
    }

    #[must_use]
    fn process_module_block_combinatorial(&mut self, drivers: &mut Drivers, scope_body: Scope, statement_index: usize, comb_block: &CombinatorialBlock) -> ModuleBlockCombinatorial {
        let &CombinatorialBlock { span, span_keyword: _, ref block } = comb_block;

        let mut collector = DriverCollector {
            driver: Driver::CombinatorialBlock(statement_index),
            drivers,
        };
        let lower_block = self.visit_block(
            &ExpressionContext::passthrough(scope_body),
            &mut MaybeDriverCollector::Some(&mut collector),
            block,
        );

        ModuleBlockCombinatorial {
            span,
            block: lower_block,
        }
    }

    #[must_use]
    fn process_module_block_clocked(
        &mut self,
        drivers: &mut Drivers,
        scopy_body: Scope,
        statement_index: usize,
        clocked_block: &ClockedBlock,
    ) -> ModuleBlockClocked {
        let &ClockedBlock {
            span, span_keyword: _, ref domain, ref block
        } = clocked_block;

        let domain_span = domain.span;

        let sync_domain = self.eval_sync_domain(scopy_body, domain.as_ref().map_inner(|v| {
            v.as_ref().map_inner(|v| &**v)
        }));
        let domain = Spanned { span: domain_span, inner: &ValueDomain::Sync(sync_domain.clone()) };

        let ctx = ExpressionContext {
            scope: scopy_body,
            domain: ContextDomain::Specific(domain),
            function_return_ty: None,
        };
        let mut driver_collector = DriverCollector {
            driver: Driver::ClockedBlock(statement_index),
            drivers,
        };
        let lower_block = self.visit_block(
            &ctx,
            &mut MaybeDriverCollector::Some(&mut driver_collector),
            block,
        );

        ModuleBlockClocked {
            span,
            domain: sync_domain,
            on_reset: LowerBlock { statements: vec![] },
            on_clock: lower_block,
        }
    }

    #[must_use]
    fn process_module_instance(
        &mut self,
        drivers: &mut Drivers,
        scope_body: Scope,
        statement_index: usize,
        instance: &ast::ModuleInstance,
    ) -> ModuleStatement {
        // TODO check that ID is unique, both with other IDs but also signals and wires
        //   (or do we leave that to backend?)
        let &ast::ModuleInstance {
            span: instance_span,
            span_keyword: keyword_span,
            ref name,
            ref module,
            ref generic_args,
            ref port_connections
        } = instance;

        // always evaluate generic args
        let ctx_generics = ExpressionContext::constant(instance_span, scope_body);
        let mut collector_generics = MaybeDriverCollector::None;
        let generic_args = generic_args.as_ref().map(|generic_args| {
            generic_args.map_inner(|arg| {
                self.eval_expression_as_ty_or_value(&ctx_generics, &mut collector_generics, arg)
            })
        });

        // find the module, fill in generics
        let module_with_generics = self.eval_expr_as_module_with_generics(
            keyword_span,
            &ctx_generics,
            &mut collector_generics,
            module,
            &generic_args,
        );

        // always evaluate ports, so they can emit errors even if the module or its generics are invalid
        let ctx_connections = ExpressionContext::passthrough(scope_body);
        let mut collector = DriverCollector {
            driver: Driver::InstancePortConnection(statement_index),
            drivers,
        };
        let mut collector_connections = MaybeDriverCollector::Some(&mut collector);

        let connections = port_connections.as_ref().map_inner(|port_connections| {
            port_connections.iter().map(|(id, expr)| {
                (id.clone(), expr.as_ref().map_inner(|_| {
                    self.eval_expression_as_value(&ctx_connections, &mut collector_connections, expr)
                }))
            }).collect_vec()
        });

        // to continue checking ports we need the module to be valid
        let (module_with_generics, generic_arguments) = match module_with_generics {
            Ok(m) => m,
            Err(e) => return ModuleStatement::Err(e),
        };

        // check port connections
        let port_connections = match self.check_module_instance_ports(
            &ctx_connections,
            &mut collector_connections,
            &module_with_generics.ports,
            &connections,
        ) {
            Ok(p) => p,
            Err(e) => return ModuleStatement::Err(e),
        };

        // successfully created instance
        ModuleStatement::Instance(ModuleInstance {
            module: module_with_generics.nominal_type_unique.item,
            name: name.as_ref().map(|s| s.string.clone()),
            generic_arguments,
            port_connections,
        })
    }

    // TODO allow different declaration and use orderings, be careful about interactions
    fn check_module_instance_ports(&mut self, ctx: &ExpressionContext, collector: &mut MaybeDriverCollector, ports: &[ModulePort], connections: &Spanned<Vec<(Identifier, Spanned<Value>)>>) -> Result<PortConnections, ErrorGuaranteed> {
        let diags = self.diags;

        let mut any_err = Ok(());
        if ports.len() != connections.inner.len() {
            let err = Diagnostic::new_simple(
                format!("constructor port count mismatch, expected {}, got {}", ports.len(), connections.inner.len()),
                connections.span,
                "connected here",
            );
            any_err = Err(diags.report(err));
        }
        let min_len = min(ports.len(), connections.inner.len());

        let mut map_port = IndexMap::new();
        let mut ordered_connections = vec![];

        for (&port, (connection_id, connection)) in zip_eq(&ports[..min_len], &connections.inner[..min_len]) {
            let port_ast = self.parsed.module_port_ast(self.compiled[port].ast);

            if port_ast.id.string != connection_id.string {
                let err = Diagnostic::new("port name mismatch")
                    .add_error(connection_id.span, format!("got {}, connected here", connection_id.string))
                    .add_info(port_ast.id.span, format!("expected {}, defined here", port_ast.id.string))
                    .footer(Level::Note, "different port and connection orderings are not yet supported")
                    .finish();
                any_err = Err(diags.report(err));

                // from now on port replacement is broken, so we have to stop the loop
                break;
            }

            // immediately use existing generic params to replace the current one
            // (actual generics are already replaced in this module instance, so no need to keep applying them)
            let map = GenericMap {
                generic_ty: Default::default(),
                generic_value: Default::default(),
                module_port: map_port,
            };
            let replaced = port.replace_generics(&mut self.compiled, &map);
            map_port = map.module_port;

            // kind/type/domain check
            let &Spanned { span: connection_value_span, inner: ref connection_value } = connection;

            let &ModulePortInfo {
                ast: _,
                direction: port_dir,
                kind: ref port_kind
            } = &self.compiled[replaced];

            // TODO centrally, at the end of every module: read/write checking for all regs,wires,ports
            //   specifically here: register reads and writes
            let port_replacement = match port_kind {
                PortKind::Clock => {
                    match self.check_type_contains(None, connection_value_span, &Type::Clock, connection_value) {
                        Ok(()) => connection_value.clone(),
                        Err(e) => Value::Error(e),
                    }
                }
                PortKind::Normal { domain: domain_port, ty: ty_port } => {
                    let domain_value = self.domain_of_value(connection_value_span, connection_value);

                    let (e_access, e_domain, e_ty) = match port_dir {
                        PortDirection::Input => {
                            let e_access = self.check_value_usable_as_direction(
                                ctx,
                                collector,
                                connection_value_span,
                                connection_value,
                                AccessDirection::Read,
                            );
                            let e_domain = self.check_domain_crossing(
                                port_ast.id.span,
                                &ValueDomain::from_domain_kind(domain_port.clone()),
                                connection_value_span,
                                &domain_value,
                                DomainUserControlled::Source,
                                "instance port connections must respect domains",
                            );
                            let e_ty = self.check_type_contains(
                                Some(port_ast.kind.span),
                                connection_value_span,
                                ty_port,
                                connection_value,
                            );
                            (e_access, e_domain, e_ty)
                        }
                        PortDirection::Output => {
                            let e_access = self.check_value_usable_as_direction(
                                ctx,
                                collector,
                                connection_value_span,
                                connection_value,
                                AccessDirection::Write,
                            );
                            let e_domain = self.check_domain_crossing(
                                connection_value_span,
                                &domain_value,
                                port_ast.id.span,
                                &ValueDomain::from_domain_kind(domain_port.clone()),
                                DomainUserControlled::Target,
                                "instance port connections must respect domains",
                            );
                            // TODO for port connections, should we assert that the types match exactly?
                            //   or should we disable the implicit expansion for all assignments,
                            //   and force exact type equalities?
                            let e_ty = self.check_type_contains(
                                Some(connection_value_span),
                                port_ast.kind.span,
                                &self.type_of_value(connection_value_span, &connection_value),
                                // TODO this is hacky, hopefully the next type system rewrite can fix this
                                &Value::ModulePort(port),
                            );
                            (e_access, e_domain, e_ty)
                        }
                    };

                    match (e_access, e_domain, e_ty) {
                        (Ok(()), Ok(()), Ok(())) => connection_value.clone(),
                        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => Value::Error(e),
                    }
                }
            };

            // add port to replacement
            map_port.insert_first(port, port_replacement.clone());
            ordered_connections.push(Spanned { span: connection_value_span, inner: port_replacement });
        }

        any_err.map(|()| PortConnections { vec: ordered_connections })
    }

    fn eval_expr_as_module_with_generics(
        &mut self,
        keyword_span: Span,
        ctx_generics: &ExpressionContext,
        collector_generics: &mut MaybeDriverCollector,
        module_expr: &ast::Expression,
        generic_args: &Option<ast::Args<TypeOrValue>>,
    ) -> Result<(ModuleValueInfo, Option<GenericArguments>), ErrorGuaranteed> {
        let diags = self.diags;

        let module_value = self.eval_expression(
            ctx_generics,
            collector_generics,
            module_expr,
        );

        let (constructor_params, module_inner) = match module_value {
            ScopedEntryDirect::Immediate(inner) => (Ok(None), inner),
            ScopedEntryDirect::Constructor(Constructor { parameters, inner }) => (Ok(Some(parameters)), inner),
            ScopedEntryDirect::Error(e) => (Err(e), TypeOrValue::Error(e)),
        };

        let constructor_str = match constructor_params {
            Ok(Some(_)) => " constructor",
            Ok(None) => "",
            // this error case should not show up in user output
            Err(_) => " error",
        };
        let module_inner_raw = match module_inner {
            TypeOrValue::Type(_) =>
                Err(diags.report_simple(format!("expected module, got type{}", constructor_str), module_expr.span, format!("type{}", constructor_str))),
            TypeOrValue::Value(value) => {
                match value {
                    Value::Error(e) => Err(e),
                    Value::Module(inner) => Ok(inner),
                    _ => Err(diags.report_simple(
                        format!("expected module, got other non-module{} {}", constructor_str, self.compiled.value_to_readable_str(self.source, self.parsed, &value)),
                        module_expr.span,
                        "value",
                    ))
                }
            }
            TypeOrValue::Error(e) => Err(e),
        };

        match (constructor_params, generic_args, module_inner_raw) {
            // immediate module
            (Ok(None), None, module_inner_raw) => {
                module_inner_raw.map(|inner| (inner, None))
            }
            // generic module with args
            (Ok(Some(constructor_params)), Some(generic_args), Ok(module_inner_raw)) => {
                self.eval_constructor_call(&constructor_params, &module_inner_raw, generic_args.clone(), false)
                    .map(|(inner, generic_args)| (inner, Some(generic_args)))
            }

            // error: we don't know if module is generic or not
            (Err(e), _, _) => Err(e),
            (_, _, Err(e)) => Err(e),

            // error: not generic but args provided
            (Ok(Some(_)), None, Ok(module_inner_raw)) => {
                let diag = Diagnostic::new("instance of generic module is missing generic args")
                    .add_error(keyword_span, "instance here does not provide generic args")
                    .add_info(module_expr.span, "module used here needs generics")
                    .add_info(self.compiled[module_inner_raw.nominal_type_unique.item].defining_id.span(), "module defined here")
                    .finish();
                Err(diags.report(diag))
            }
            (Ok(None), Some(_), Ok(module_inner_raw)) => {
                let diag = Diagnostic::new("instance of non-generic module has generic parameters")
                    .add_error(keyword_span, "instance here does provide generic args")
                    .add_info(module_expr.span, "module used here does not need generics")
                    .add_info(self.compiled[module_inner_raw.nominal_type_unique.item].defining_id.span(), "module defined here")
                    .finish();
                Err(diags.report(diag))
            }
        }
    }
}

fn pull_reg_init_into_single_clocked_block_driver(
    driver: Driver,
    module_statements_lower: &mut Vec<ModuleStatement>,
    def_span: Span,
    target: Spanned<Value>,
    init: Spanned<Value>,
) {
    let lower_statement_index = unwrap_match!(
            driver,
            Driver::ClockedBlock(i) => i
        );
    let block = unwrap_match!(
            &mut module_statements_lower[lower_statement_index],
            ModuleStatement::Clocked(block) => block
        );

    let stmt = LowerStatement::Assignment {
        target,
        value: init,
    };
    block.on_reset.statements.push(Spanned {
        span: def_span,
        inner: stmt,
    });
}

/// The usize fields are here to keep different drivers of the same type separate for Eq and Hash,
/// so we can correctly determining whether there are multiple drivers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Driver {
    WireDeclaration,
    InstancePortConnection(usize),
    CombinatorialBlock(usize),
    ClockedBlock(usize),
}

#[derive(Debug, Copy, Clone)]
pub enum DriverKind {
    WiredConnection,
    ClockedBlock,
}

impl Driver {
    fn kind(&self) -> DriverKind {
        match self {
            Driver::WireDeclaration | Driver::InstancePortConnection(_) | Driver::CombinatorialBlock(_) =>
                DriverKind::WiredConnection,
            Driver::ClockedBlock(_) =>
                DriverKind::ClockedBlock,
        }
    }
}

#[derive(Default, Debug)]
struct Drivers {
    output_port_drivers: IndexMap<ModulePort, IndexMap<Driver, Vec<Span>>>,
    reg_drivers: IndexMap<Register, IndexMap<Driver, Vec<Span>>>,
    wire_drivers: IndexMap<Wire, IndexMap<Driver, Vec<Span>>>,
}

pub enum MaybeDriverCollector<'m> {
    Some(&'m mut DriverCollector<'m>),
    None,
}

// TODO move the driver value into the context, where is is already almost known
pub struct DriverCollector<'m> {
    driver: Driver,
    drivers: &'m mut Drivers,
}

impl<'m> MaybeDriverCollector<'m> {
    fn report<K: Eq + Hash>(&mut self, diags: &Diagnostics, map: impl FnOnce(&mut Drivers) -> &mut IndexMap<K, IndexMap<Driver, Vec<Span>>>, key: K, span: Span) {
        match self {
            MaybeDriverCollector::Some(collector) => {
                map(collector.drivers).entry(key).or_default()
                    .entry(collector.driver).or_default()
                    .push(span);
            }
            MaybeDriverCollector::None => {
                let reason = "reporting driver in context where drivers are not being collected";
                diags.report_internal_error(span, reason);
            }
        }
    }

    pub fn report_write_output_port(&mut self, diags: &Diagnostics, port: ModulePort, span: Span) {
        self.report(diags, |d| &mut d.output_port_drivers, port, span);
    }

    pub fn report_write_reg(&mut self, diags: &Diagnostics, reg: Register, span: Span) {
        self.report(diags, |d| &mut d.reg_drivers, reg, span);
    }

    pub fn report_write_wire(&mut self, diags: &Diagnostics, wire: Wire, span: Span) {
        self.report(diags, |d| &mut d.wire_drivers, wire, span);
    }
}
