use crate::data::compiled::{Item, ModulePort, ModulePortInfo, Register, RegisterInfo, VariableInfo, Wire, WireInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::module_body::{LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleChecked, ModuleInstance, ModuleStatement};
use crate::front::common::{ExpressionContext, GenericContainer, GenericMap, ScopedEntry, ScopedEntryDirect, TypeOrValue, ValueDomainKind};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::{Constructor, GenericArguments, PortConnections, Type};
use crate::front::values::{ModuleValueInfo, Value};
use crate::syntax::ast;
use crate::syntax::ast::{BlockStatementKind, ClockedBlock, CombinatorialBlock, Identifier, ModuleStatementKind, PortDirection, PortKind, RegDeclaration, Spanned, SyncDomain, VariableDeclaration, WireDeclaration};
use crate::syntax::pos::Span;
use crate::util::data::IndexMapExt;
use annotate_snippets::Level;
use indexmap::IndexMap;
use itertools::{zip_eq, Itertools};
use std::cmp::min;

impl<'d, 'a> CompileState<'d, 'a> {
    pub fn check_module_body(&mut self, module_item: Item, module_ast: &ast::ItemDefModule) -> ModuleChecked {
        let ast::ItemDefModule { span: _, vis: _, id: _, params: _, ports: _, body } = module_ast;
        let ast::Block { span: _, statements } = body;

        // TODO do we event want to convert to some simpler IR here,
        //   or just leave the backend to walk the AST if it wants?
        let mut module_statements = vec![];
        let mut module_regs = vec![];
        let mut module_wires = vec![];

        let scope_ports = self.compiled.module_info[&module_item].scope_ports;
        let scope_body = self.compiled.scopes.new_child(scope_ports, body.span, Visibility::Private);

        let ctx_module = &ExpressionContext::NotFunctionBody;

        // first pass: populate scope with declarations
        for top_statement in statements {
            match &top_statement.inner {
                ModuleStatementKind::VariableDeclaration(decl) => {
                    self.process_module_declaration_variable(scope_body, ctx_module, decl);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let reg = self.process_module_declaration_reg(module_item, scope_body, ctx_module, decl);
                    module_regs.push(reg);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    let wire = self.process_module_declaration_wire(module_item, scope_body, ctx_module, decl);
                    module_wires.push(wire);
                }
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
                ModuleStatementKind::Instance(_) => {}
            }
        }

        // second pass: codegen for the actual blocks
        for top_statement in statements {
            match &top_statement.inner {
                // declarations were already handled
                ModuleStatementKind::VariableDeclaration(_) => {}
                ModuleStatementKind::RegDeclaration(_) => {}
                ModuleStatementKind::WireDeclaration(_) => {}
                // actual blocks
                ModuleStatementKind::CombinatorialBlock(ref comb_block) => {
                    let block = self.process_module_block_combinatorial(scope_body, comb_block);
                    module_statements.push(ModuleStatement::Combinatorial(block));
                }
                ModuleStatementKind::ClockedBlock(ref clocked_block) => {
                    let block = self.process_module_block_clocked(scope_body, clocked_block);
                    module_statements.push(ModuleStatement::Clocked(block));
                }
                // instances
                ModuleStatementKind::Instance(instance) => {
                    let block = self.process_module_instance(&ctx_module, scope_body, instance);
                    module_statements.push(block);
                }
            }
        }

        ModuleChecked {
            statements: module_statements,
            regs: module_regs,
            wires: module_wires,
        }
    }

    fn process_module_declaration_variable(&mut self, scope_body: Scope, ctx_module: &ExpressionContext, decl: &VariableDeclaration) {
        let &VariableDeclaration { span, mutable, ref id, ref ty, ref init } = decl;

        let entry = if mutable {
            let e = self.diags.report_todo(span, "mutable variable in module body");
            ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Error(e)))
        } else {
            match (ty, init) {
                (Some(ty), Some(init)) => {
                    let ty_eval = self.eval_expression_as_ty(scope_body, ty);
                    let init_eval = self.eval_expression_as_value(ctx_module, scope_body, init);

                    let _: Result<(), ErrorGuaranteed> = self.check_type_contains(Some(ty.span), init.span, &ty_eval, &init_eval);

                    let var = self.compiled.variables.push(VariableInfo {
                        defining_id: id.clone(),
                        ty: ty_eval.clone(),
                        mutable,
                    });
                    ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Variable(var)))
                }
                _ => {
                    let e = self.diags.report_todo(span, "variable declaration without type and/or init");
                    ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Error(e)))
                }
            }
        };
        self.compiled[scope_body].maybe_declare(&self.diags, &id, ScopedEntry::Direct(entry), Visibility::Private);
    }

    fn process_module_declaration_reg(&mut self, module_item: Item, scope_body: Scope, ctx_module: &ExpressionContext, decl: &RegDeclaration) -> (Register, Value) {
        let RegDeclaration { span: _, id, sync, ty, init } = decl;
        let ty_span = ty.span;
        let init_span = init.span;

        let sync = self.eval_sync_domain(scope_body, &sync.inner);
        let ty = self.eval_expression_as_ty(scope_body, ty);
        let init = self.eval_expression_as_value(ctx_module, scope_body, init);

        // check ty
        let init = match self.check_type_contains(Some(ty_span), init_span, &ty, &init) {
            Ok(()) => init,
            Err(e) => Value::Error(e),
        };

        // check domain
        let _: Result<(), ErrorGuaranteed> = self.check_domain_assign(
            id.span(),
            &ValueDomainKind::Const,
            init_span,
            &self.domain_of_value(init_span, &init),
            UserControlled::Source,
            "register initial value must be const",
        );

        // create register
        let reg = self.compiled.registers.push(RegisterInfo {
            defining_item: module_item,
            defining_id: id.clone(),
            domain: sync,
            ty,
        });

        let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Register(reg))));
        self.compiled[scope_body].maybe_declare(&self.diags, &id, entry, Visibility::Private);

        (reg, init)
    }

    fn process_module_declaration_wire(&mut self, module_item: Item, scope_body: Scope, ctx_module: &ExpressionContext, decl: &WireDeclaration) -> (Wire, Option<Value>) {
        let WireDeclaration { span: _, id, sync, ty, value } = decl;

        let sync = self.eval_domain(scope_body, &sync.inner);
        let ty = self.eval_expression_as_ty(scope_body, ty);

        let value = match value {
            Some(value) => {
                let value_span = value.span;
                let value = self.eval_expression_as_value(ctx_module, scope_body, value);

                // check type
                let value = match self.check_type_contains(Some(value_span), value_span, &ty, &value) {
                    Ok(()) => value,
                    Err(e) => Value::Error(e),
                };

                // check domain
                let _: Result<(), ErrorGuaranteed> = self.check_domain_assign(
                    id.span(),
                    &ValueDomainKind::from_domain_kind(sync.clone()),
                    value_span,
                    &self.domain_of_value(value_span, &value),
                    UserControlled::Source,
                    "wire value must be assignable to the wire domain",
                );

                Some(value)
            }
            None => None,
        };

        // create wire
        let wire = self.compiled.wires.push(WireInfo {
            defining_item: module_item,
            defining_id: id.clone(),
            domain: sync,
            ty,
        });

        let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Wire(wire))));
        self.compiled[scope_body].maybe_declare(&self.diags, &id, entry, Visibility::Private);

        (wire, value)
    }

    #[must_use]
    fn process_module_block_combinatorial(&mut self, scope_body: Scope, comb_block: &CombinatorialBlock) -> ModuleBlockCombinatorial {
        let &CombinatorialBlock { span, span_keyword: _, ref block } = comb_block;
        let ast::Block { span: _, statements } = block;

        let scope = self.compiled.scopes.new_child(scope_body, block.span, Visibility::Private);
        let ctx_comb = &ExpressionContext::NotFunctionBody;
        let ctx_sync = None;

        let mut result_statements = vec![];

        for statement in statements {
            match &statement.inner {
                BlockStatementKind::VariableDeclaration(_) => {
                    let err = self.diags.report_todo(statement.span, "combinatorial variable declaration");
                    result_statements.push(LowerStatement::Error(err));
                }
                BlockStatementKind::Assignment(ref assignment) => {
                    let &ast::Assignment { span: _, op, ref target, ref value } = assignment;
                    if op.inner.is_some() {
                        let err = self.diags.report_todo(statement.span, "combinatorial assignment with operator");
                        result_statements.push(LowerStatement::Error(err));
                    } else {
                        let target = self.eval_expression_as_value(ctx_comb, scope, target);
                        let value = self.eval_expression_as_value(ctx_comb, scope, value);

                        match (target, value) {
                            (Value::ModulePort(target), Value::ModulePort(value)) => {
                                let stmt = match self.check_assign_port_port(ctx_sync, assignment, target, value) {
                                    Ok(()) => LowerStatement::PortPortAssignment(target, value),
                                    Err(e) => LowerStatement::Error(e),
                                };
                                result_statements.push(stmt);
                            }
                            (Value::Error(e), _) | (_, Value::Error(e)) => {
                                result_statements.push(LowerStatement::Error(e));
                            }
                            _ => {
                                let err = self.diags.report_todo(statement.span, "general combinatorial assignment");
                                result_statements.push(LowerStatement::Error(err));
                            }
                        }
                    }
                }
                BlockStatementKind::Expression(_) => {
                    let err = self.diags.report_todo(statement.span, "combinatorial expression");
                    result_statements.push(LowerStatement::Error(err));
                }
            }
        }

        ModuleBlockCombinatorial {
            span,
            statements: result_statements,
        }
    }

    #[must_use]
    fn process_module_block_clocked(&mut self, scope_body: Scope, clocked_block: &ClockedBlock) -> ModuleBlockClocked {
        let &ClockedBlock {
            span, span_keyword: _, ref clock, ref reset, ref block
        } = clocked_block;
        let ast::Block { span: _, statements } = block;
        let span_domain = clock.span.join(reset.span);

        let scope = self.compiled.scopes.new_child(scope_body, block.span, Visibility::Private);
        let ctx_clocked = &ExpressionContext::NotFunctionBody;

        // TODO typecheck: clock must be a single-bit clock, reset must be a single-bit reset
        let clock_value_unchecked = self.eval_expression_as_value(ctx_clocked, scope, clock);
        let reset_value_unchecked = self.eval_expression_as_value(ctx_clocked, scope, reset);

        // check that clock is a clock
        let clock_domain = self.domain_of_value(clock.span, &clock_value_unchecked);
        let clock_value = match &clock_domain {
            ValueDomainKind::Clock => clock_value_unchecked,
            &ValueDomainKind::Error(e) => Value::Error(e),
            _ => {
                let title = format!("clock must be a clock, has domain {}", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, &clock_domain));
                let e = self.diags.report_simple(title, clock.span, "clock value");
                Value::Error(e)
            }
        };

        // check that reset is an async bool
        let reset_value_bool = match self.check_type_contains(None, reset.span, &Type::Boolean, &reset_value_unchecked) {
            Ok(()) => reset_value_unchecked,
            Err(e) => Value::Error(e),
        };
        let reset_domain = self.domain_of_value(reset.span, &reset_value_bool);
        let reset_value = match &reset_domain {
            ValueDomainKind::Async => reset_value_bool,
            &ValueDomainKind::Error(e) => Value::Error(e),
            _ => {
                let title = format!("reset must be an async boolean, has domain {}", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, &reset_domain));
                let e = self.diags.report_simple(title, reset.span, "reset value");
                Value::Error(e)
            }
        };

        let domain = SyncDomain { clock: clock_value, reset: reset_value };
        let ctx_sync = Some(Spanned { span: span_domain, inner: &domain });

        let mut result_statements = vec![];

        for statement in statements {
            match &statement.inner {
                BlockStatementKind::VariableDeclaration(_) => {
                    let err = self.diags.report_todo(statement.span, "clocked variable declaration");
                    result_statements.push(LowerStatement::Error(err));
                }
                BlockStatementKind::Assignment(assignment) => {
                    let &ast::Assignment { span: _, op, ref target, ref value } = assignment;
                    if op.inner.is_some() {
                        let err = self.diags.report_todo(statement.span, "clocked assignment with operator");
                        result_statements.push(LowerStatement::Error(err));
                    } else {
                        let target = self.eval_expression_as_value(ctx_clocked, scope, target);
                        let value = self.eval_expression_as_value(ctx_clocked, scope, value);

                        match (target, value) {
                            (Value::ModulePort(target), Value::ModulePort(value)) => {
                                let stmt = match self.check_assign_port_port(ctx_sync, assignment, target, value) {
                                    Ok(()) => LowerStatement::PortPortAssignment(target, value),
                                    Err(e) => LowerStatement::Error(e),
                                };
                                result_statements.push(stmt);
                            }
                            (Value::Error(e), _) | (_, Value::Error(e)) => {
                                result_statements.push(LowerStatement::Error(e));
                            }
                            _ => {
                                let err = self.diags.report_todo(statement.span, "general clocked assignment");
                                result_statements.push(LowerStatement::Error(err));
                            }
                        }
                    }
                }
                BlockStatementKind::Expression(_) => {
                    let err = self.diags.report_todo(statement.span, "expression inside clocked block");
                    result_statements.push(LowerStatement::Error(err));
                }
            }
        }

        ModuleBlockClocked {
            span,
            domain,
            on_reset: vec![],
            on_block: result_statements,
        }
    }

    #[must_use]
    fn process_module_instance(&mut self, ctx_module: &ExpressionContext, scope_body: Scope, instance: &ast::ModuleInstance) -> ModuleStatement {
        // TODO check that ID is unique, both with other IDs but also signals and wires
        //   (or do we leave that to backend?)
        let &ast::ModuleInstance {
            span: _,
            span_keyword: keyword_span,
            ref name,
            ref module,
            ref generic_args,
            ref port_connections
        } = instance;

        // always evaluate generic args
        let generic_args = generic_args.as_ref().map(|generic_args| {
            generic_args.map_inner(|arg| {
                self.eval_expression_as_ty_or_value(ctx_module, scope_body, arg)
            })
        });

        // find the module, fill in generics
        let module_with_generics = self.eval_expr_as_module_with_generics(keyword_span, ctx_module, scope_body, module, &generic_args);

        // always evaluate ports, so they can emit errors even if the module or its generics are invalid
        let connections = port_connections.as_ref().map_inner(|port_connections| {
            port_connections.iter().map(|(id, expr)| {
                (id.clone(), expr.as_ref().map_inner(|_| self.eval_expression_as_value(ctx_module, scope_body, expr)))
            }).collect_vec()
        });

        // to continue checking ports we need the module to be valid
        let (module_with_generics, generic_arguments) = match module_with_generics {
            Ok(m) => m,
            Err(e) => return ModuleStatement::Err(e),
        };

        // check port connections
        let port_connections = match self.check_module_instance_ports(&module_with_generics.ports, &connections) {
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
    fn check_module_instance_ports(&mut self, ports: &[ModulePort], connections: &Spanned<Vec<(Identifier, Spanned<Value>)>>) -> Result<PortConnections, ErrorGuaranteed> {
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
                    .add_info(port_ast.id.span, format!("expected {}, defined here", port_ast.id.string))
                    .add_error(connection_id.span, format!("got {}, connected here", connection_id.string))
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

                    let (e_domain, e_ty) = match port_dir {
                        PortDirection::Input => {
                            let e_domain = self.check_domain_assign(
                                port_ast.id.span,
                                &ValueDomainKind::from_domain_kind(domain_port.clone()),
                                connection_value_span,
                                &domain_value,
                                UserControlled::Source,
                                "instance port connections must respect domains",
                            );
                            let e_ty = self.check_type_contains(
                                Some(port_ast.kind.span),
                                connection_value_span,
                                ty_port,
                                connection_value,
                            );
                            (e_domain, e_ty)
                        }
                        PortDirection::Output => {
                            let e_domain = self.check_domain_assign(
                                connection_value_span,
                                &domain_value,
                                port_ast.id.span,
                                &ValueDomainKind::from_domain_kind(domain_port.clone()),
                                UserControlled::Target,
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
                            (e_domain, e_ty)
                        }
                    };

                    match (e_domain, e_ty) {
                        (Ok(()), Ok(())) => connection_value.clone(),
                        (Err(e), _) | (_, Err(e)) => Value::Error(e),
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
        ctx_module: &ExpressionContext,
        scope: Scope,
        module_expr: &ast::Expression,
        generic_args: &Option<ast::Args<TypeOrValue>>,
    ) -> Result<(ModuleValueInfo, Option<GenericArguments>), ErrorGuaranteed> {
        let diags = self.diags;

        let module_value = self.eval_expression(ctx_module, scope, module_expr);

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

    // TODO this will get significantly refactored, assignments between ports are just a singel special case
    fn check_assign_port_port(&mut self, block_sync: Option<Spanned<&SyncDomain<Value>>>, assignment: &ast::Assignment, target: ModulePort, value: ModulePort) -> Result<(), ErrorGuaranteed> {
        let span = assignment.span;
        let &ModulePortInfo { ast: target_ast, direction: target_dir, kind: ref target_kind } = &self.compiled[target];
        let &ModulePortInfo { ast: value_ast, direction: value_dir, kind: ref value_kind } = &self.compiled[value];

        // check item
        if target_ast.item != value_ast.item {
            return Err(self.diags.report_internal_error(span, "port assignment between different modules"));
        }

        // check direction
        if target_dir != PortDirection::Output {
            return Err(self.diags.report_internal_error(assignment.target.span, "port assignment to non-output"));
        }
        if value_dir != PortDirection::Input {
            // TODO allow read-back from output port under certain conditions
            //   (ie. if this is the same block that has already written to said output)
            return Err(self.diags.report_internal_error(assignment.value.span, "port assignment from non-input"));
        }

        // TODO check context: we should be in a combinatorial block,
        //   or a clocked block with the same clock domain

        match (target_kind, value_kind) {
            // TODO careful about delta cycles and the verilog equivalent!
            (PortKind::Clock, PortKind::Clock) =>
                Err(self.diags.report_todo(span, "clock assignment")),
            (PortKind::Normal { domain: target_sync, ty: target_ty }, PortKind::Normal { domain: value_sync, ty: value_ty }) => {
                match block_sync {
                    None => {
                        // async block, we just need source->target to be valid
                        self.check_domain_assign(
                            assignment.target.span,
                            &ValueDomainKind::from_domain_kind(target_sync.clone()),
                            assignment.value.span,
                            &ValueDomainKind::from_domain_kind(value_sync.clone()),
                            UserControlled::Both,
                            "in a combinatorial block, for each assignment, target and source must be in the same domain",
                        )?;
                    }
                    Some(Spanned { span: block_sync_span, inner: block_sync }) => {
                        // clocked block, we need source->block and block->target to be valid
                        let block_sync = ValueDomainKind::Sync(block_sync.clone());

                        let result_0 = self.check_domain_assign(
                            assignment.target.span,
                            &ValueDomainKind::from_domain_kind(target_sync.clone()),
                            block_sync_span,
                            &block_sync,
                            UserControlled::Target,
                            "in a clocked block, each assignment target must be in the same domain as the block",
                        );
                        let result_1 = self.check_domain_assign(
                            block_sync_span,
                            &block_sync,
                            assignment.value.span,
                            &ValueDomainKind::from_domain_kind(value_sync.clone()),
                            UserControlled::Source,
                            "in a clocked block, each source must be in the same domain as the block",
                        );

                        result_0?;
                        result_1?;
                    }
                }

                // TODO fix this once we support type-type checking
                let _ = value_ty;
                self.check_type_contains(None, assignment.value.span, &target_ty, &Value::ModulePort(value))?;

                Ok(())
            }
            _ => Err(self.diags.report_simple("port assignment between different port kinds", span, "assignment")),
        }
    }

    /// Checks whether the source sync domain can be assigned to the target sync domain.
    /// This is equivalent to checking whether source is more contained that target.
    fn check_domain_assign(
        &self,
        target_span: Span,
        target: &ValueDomainKind,
        source_span: Span,
        source: &ValueDomainKind,
        user_controlled: UserControlled,
        hint: &str,
    ) -> Result<(), ErrorGuaranteed> {
        let diags = self.diags;

        let invalid_reason = match (target, source) {
            // propagate errors
            (&ValueDomainKind::Error(e), _) | (_, &ValueDomainKind::Error(e)) =>
                return Err(e),
            // clock assignments are not yet implemented
            (ValueDomainKind::Clock, _) | (_, ValueDomainKind::Clock) =>
                return Err(self.diags.report_todo(target_span.join(source_span), "clock assignment")),
            // const target must have const source
            (ValueDomainKind::Const, ValueDomainKind::Const) => None,
            (ValueDomainKind::Const, ValueDomainKind::Async) => Some("async to const"),
            (ValueDomainKind::Const, ValueDomainKind::Sync(_)) => Some("sync to const"),
            // const can be the source of everything
            (ValueDomainKind::Async, ValueDomainKind::Const) => None,
            (ValueDomainKind::Sync(_), ValueDomainKind::Const) => None,
            // async can be the target of everything
            (ValueDomainKind::Async, _) => None,
            // sync cannot be the target of async
            (ValueDomainKind::Sync(_), ValueDomainKind::Async) => Some("async to sync"),
            // sync pair is allowed if clock and reset match
            (ValueDomainKind::Sync(target), ValueDomainKind::Sync(source)) => {
                let SyncDomain { clock: target_clock, reset: target_reset } = target;
                let SyncDomain { clock: source_clock, reset: source_reset } = source;

                // TODO equality is _probably_ the wrong operation for this
                let value_eq = |a: &Value, b: &Value| {
                    match (a, b) {
                        // optimistically assume they match
                        (&Value::Error(_), _) | (_, &Value::Error(_)) => true,
                        (a, b) => a == b,
                    }
                };

                match (value_eq(target_clock, source_clock), value_eq(target_reset, source_reset)) {
                    (false, false) => Some("different clock and reset"),
                    (false, true) => Some("different clock"),
                    (true, false) => Some("different reset"),
                    (true, true) => None,
                }
            }
        };

        let (target_level, source_level) = match user_controlled {
            UserControlled::Target => (Level::Error, Level::Info),
            UserControlled::Source => (Level::Info, Level::Error),
            UserControlled::Both => (Level::Error, Level::Error),
        };

        if let Some(invalid_reason) = invalid_reason {
            let err = Diagnostic::new(format!("unsafe domain crossing: {}", invalid_reason))
                .add(target_level, target_span, format!("target in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, target)))
                .add(source_level, source_span, format!("source in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, self.parsed, source)))
                .footer(Level::Help, hint)
                .finish();
            Err(diags.report(err))
        } else {
            Ok(())
        }
    }
}

enum UserControlled {
    Target,
    Source,
    Both,
}