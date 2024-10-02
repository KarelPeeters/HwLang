use crate::data::compiled::{Item, ModulePort, ModulePortInfo, Register, RegisterInfo, VariableInfo};
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, ErrorGuaranteed};
use crate::data::module_body::{LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleBlockInfo, ModuleChecked};
use crate::front::common::{ExpressionContext, ScopedEntry, ScopedEntryDirect, TypeOrValue, ValueDomainKind};
use crate::front::driver::CompileState;
use crate::front::scope::{Scope, Visibility};
use crate::front::types::Type;
use crate::front::values::Value;
use crate::syntax::ast;
use crate::syntax::ast::{BlockStatementKind, ClockedBlock, CombinatorialBlock, ModuleStatementKind, PortDirection, PortKind, RegDeclaration, Spanned, SyncDomain, VariableDeclaration};
use crate::syntax::pos::Span;
use annotate_snippets::Level;

impl<'d, 'a> CompileState<'d, 'a> {
    pub fn check_module_body(&mut self, module_item: Item, module_ast: &ast::ItemDefModule) -> ModuleChecked {
        let ast::ItemDefModule { span: _, vis: _, id: _, params: _, ports: _, body } = module_ast;
        let ast::Block { span: _, statements } = body;

        // TODO do we event want to convert to some simpler IR here,
        //   or just leave the backend to walk the AST if it wants?
        let mut module_blocks = vec![];
        let mut module_regs = vec![];

        let scope_ports = self.compiled.module_info[&module_item].scope_ports;
        let scope_body = self.compiled.scopes.new_child(scope_ports, body.span, Visibility::Private);

        let ctx_module = &ExpressionContext::ModuleTopLevel;

        // first pass: populate scope with declarations
        for top_statement in statements {
            match &top_statement.inner {
                ModuleStatementKind::VariableDeclaration(decl) => {
                    self.process_module_declaration_variable(scope_body, ctx_module, decl);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    self.process_module_declaration_reg(module_item, &mut module_regs, scope_body, ctx_module, decl);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    // TODO careful if/when implementing this, wire semantics are unclear
                    self.diags.report_todo(decl.span, "wire declaration");
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
                    self.process_module_block_combinatorial(&mut module_blocks, scope_body, comb_block);
                }
                ModuleStatementKind::ClockedBlock(ref clocked_block) => {
                    self.process_module_block_clocked(&mut module_blocks, scope_body, clocked_block);
                }
                // instances
                ModuleStatementKind::Instance(_) => {
                    self.diags.report_todo(top_statement.span, "module instance");
                }
            }
        }

        ModuleChecked {
            blocks: module_blocks,
            regs: module_regs,
        }
    }

    fn process_module_declaration_variable(&mut self, scope_body: Scope, ctx_module: &ExpressionContext, decl: &VariableDeclaration) {
        let &VariableDeclaration { span, mutable, ref id, ref ty, ref init } = decl;

        let entry = if mutable {
            let e = self.diags.report_todo(span, "mutable variable in module body");
            ScopedEntryDirect::Error(e)
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
                    ScopedEntryDirect::Error(e)
                }
            }
        };
        self.compiled[scope_body].maybe_declare(&self.diags, &id, ScopedEntry::Direct(entry), Visibility::Private);
    }

    fn process_module_declaration_reg(&mut self, module_item: Item, module_regs: &mut Vec<(Register, Value)>, scope_body: Scope, ctx_module: &ExpressionContext, decl: &RegDeclaration) {
        let RegDeclaration { span: _, id, sync, ty, init } = decl;

        let sync = self.eval_sync_domain(scope_body, &sync.inner);
        let ty = self.eval_expression_as_ty(scope_body, ty);

        // TODO assert that the init value is known at compile time (basically a kind of sync-ness)
        let init = self.eval_expression_as_value(ctx_module, scope_body, init);

        let reg = self.compiled.registers.push(RegisterInfo {
            defining_item: module_item,
            defining_id: id.clone(),
            domain: sync,
            ty,
        });
        module_regs.push((reg, init));

        let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Register(reg))));
        self.compiled[scope_body].maybe_declare(&self.diags, &id, entry, Visibility::Private);
    }

    fn process_module_block_combinatorial(&mut self, module_blocks: &mut Vec<ModuleBlockInfo>, scope_body: Scope, comb_block: &CombinatorialBlock) {
        let &CombinatorialBlock { span, span_keyword: _, ref block } = comb_block;
        let ast::Block { span: _, statements } = block;

        let scope = self.compiled.scopes.new_child(scope_body, block.span, Visibility::Private);
        let ctx_comb = &ExpressionContext::CombinatorialBlock;
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

        let result_block = ModuleBlockCombinatorial {
            span,
            statements: result_statements,
        };
        module_blocks.push(ModuleBlockInfo::Combinatorial(result_block));
    }

    fn process_module_block_clocked(&mut self, module_blocks: &mut Vec<ModuleBlockInfo>, scope_body: Scope, clocked_block: &ClockedBlock) {
        let &ClockedBlock {
            span, span_keyword: _, ref clock, ref reset, ref block
        } = clocked_block;
        let ast::Block { span: _, statements } = block;
        let span_domain = clock.span.join(reset.span);

        let scope = self.compiled.scopes.new_child(scope_body, block.span, Visibility::Private);
        let ctx_clocked = &ExpressionContext::ClockedBlock;

        // TODO typecheck: clock must be a single-bit clock, reset must be a single-bit reset
        let clock_value_unchecked = self.eval_expression_as_value(ctx_clocked, scope, clock);
        let reset_value_unchecked = self.eval_expression_as_value(ctx_clocked, scope, reset);

        // check that clock is a clock
        let clock_domain = self.domain_of_value(clock.span, &clock_value_unchecked);
        let clock_value = match &clock_domain {
            ValueDomainKind::Clock => clock_value_unchecked,
            &ValueDomainKind::Error(e) => Value::Error(e),
            _ => {
                let title = format!("clock must be a clock, has domain {}", self.compiled.sync_kind_to_readable_string(&self.source, &clock_domain));
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
                let title = format!("reset must be an async boolean, has domain {}", self.compiled.sync_kind_to_readable_string(&self.source, &reset_domain));
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

        let result_block = ModuleBlockClocked {
            span,
            domain,
            on_reset: vec![],
            on_block: result_statements,
        };
        module_blocks.push(ModuleBlockInfo::Clocked(result_block));
    }

    fn check_assign_port_port(&mut self, block_sync: Option<Spanned<&SyncDomain<Value>>>, assignment: &ast::Assignment, target: ModulePort, value: ModulePort) -> Result<(), ErrorGuaranteed> {
        let span = assignment.span;
        let &ModulePortInfo { defining_item: target_item, defining_id: _, direction: target_dir, kind: ref target_kind } = &self.compiled[target];
        let &ModulePortInfo { defining_item: value_item, defining_id: _, direction: value_dir, kind: ref value_kind } = &self.compiled[value];

        // check item
        if target_item != value_item {
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
                        self.check_sync_assign(
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

                        let result_0 = self.check_sync_assign(
                            assignment.target.span,
                            &ValueDomainKind::from_domain_kind(target_sync.clone()),
                            block_sync_span,
                            &block_sync,
                            UserControlled::Target,
                            "in a clocked block, each assignment target must be in the same domain as the block",
                        );
                        let result_1 = self.check_sync_assign(
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
    fn check_sync_assign(
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
                .add(target_level, target_span, format!("target in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, target)))
                .add(source_level, source_span, format!("source in domain {} ", self.compiled.sync_kind_to_readable_string(self.source, source)))
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