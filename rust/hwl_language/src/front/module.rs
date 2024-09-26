use crate::data::compiled::{Item, RegisterInfo};
use crate::data::module_body::{LowerStatement, ModuleBlockClocked, ModuleBlockCombinatorial, ModuleBlockInfo, ModuleChecked};
use crate::front::common::{ExpressionContext, ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::{CompileState, ResolveResult};
use crate::front::scope::Visibility;
use crate::front::values::Value;
use crate::syntax::ast;
use crate::syntax::ast::{BlockStatementKind, ClockedBlock, CombinatorialBlock, ModuleStatementKind, RegDeclaration, VariableDeclaration};

impl<'d, 'a> CompileState<'d, 'a> {
    pub fn check_module_body(&mut self, module_item: Item, module_ast: &ast::ItemDefModule) -> ResolveResult<ModuleChecked> {
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
                    let &VariableDeclaration { span, mutable, ref id, ref ty, ref init } = decl;

                    let entry = if mutable {
                        let e = self.diag.report_todo(span, "mutable variable in module body");
                        ScopedEntryDirect::Error(e)
                    } else {
                        match (ty, init) {
                            (Some(ty), Some(init)) => {
                                let ty_eval = self.eval_expression_as_ty(scope_body, ty)?;
                                let init_eval = self.eval_expression_as_value(ctx_module, scope_body, init)?;

                                // TODO this loses the type declaration information, which is not correct
                                match self.check_type_contains(ty.span, init.span, &ty_eval, &init_eval) {
                                    Ok(()) => ScopedEntryDirect::Immediate(TypeOrValue::Value(init_eval)),
                                    Err(e) => ScopedEntryDirect::Error(e),
                                }
                            }
                            _ => {
                                let e = self.diag.report_todo(span, "variable declaration without type and/or init");
                                ScopedEntryDirect::Error(e)
                            }
                        }
                    };
                    self.compiled[scope_body].maybe_declare(&self.diag, &id, ScopedEntry::Direct(entry), Visibility::Private);
                }
                ModuleStatementKind::RegDeclaration(decl) => {
                    let RegDeclaration { span: _, id, sync, ty, init } = decl;

                    let sync = self.eval_sync_domain(scope_body, &sync.inner)?;
                    let ty = self.eval_expression_as_ty(scope_body, ty)?;

                    // TODO assert that the init value is known at compile time (basically a kind of sync-ness)
                    let init = self.eval_expression_as_value(ctx_module, scope_body, init)?;

                    let reg = self.compiled.registers.push(RegisterInfo {
                        defining_item: module_item,
                        defining_id: id.clone(),
                        sync,
                        ty,
                    });
                    module_regs.push((reg, init));

                    let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(Value::Register(reg))));
                    self.compiled[scope_body].maybe_declare(&self.diag, &id, entry, Visibility::Private);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    // TODO careful if/when implementing this, wire semantics are unclear
                    self.diag.report_todo(decl.span, "wire declaration");
                }
                ModuleStatementKind::CombinatorialBlock(_) => {}
                ModuleStatementKind::ClockedBlock(_) => {}
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
                    let &CombinatorialBlock { span, span_keyword: _, ref block } = comb_block;
                    let ast::Block { span: _, statements } = block;

                    let scope = self.compiled.scopes.new_child(scope_body, block.span, Visibility::Private);
                    let ctx_comb = &ExpressionContext::CombinatorialBlock;
                    
                    let mut result_statements = vec![];

                    for statement in statements {
                        match &statement.inner {
                            BlockStatementKind::VariableDeclaration(_) => {
                                let err = self.diag.report_todo(statement.span, "combinatorial variable declaration");
                                result_statements.push(LowerStatement::Error(err));
                            }
                            BlockStatementKind::Assignment(ref assignment) => {
                                let &ast::Assignment { span: _, op, ref target, ref value } = assignment;
                                if op.inner.is_some() {
                                    let err = self.diag.report_todo(statement.span, "combinatorial assignment with operator");
                                    result_statements.push(LowerStatement::Error(err));
                                } else {
                                    // TODO type and sync checking
                                    let target = self.eval_expression_as_value(ctx_comb, scope, target)?;
                                    let value = self.eval_expression_as_value(ctx_comb, scope, value)?;

                                    match (target, value) {
                                        (Value::ModulePort(target), Value::ModulePort(value)) => {
                                            result_statements.push(LowerStatement::PortPortAssignment(target, value));
                                        }
                                        (Value::Error(e), _) | (_, Value::Error(e)) => {
                                            result_statements.push(LowerStatement::Error(e));
                                        }
                                        _ => {
                                            let err = self.diag.report_todo(statement.span, "general combinatorial assignment");
                                            result_statements.push(LowerStatement::Error(err));
                                        }
                                    }
                                }
                            }
                            BlockStatementKind::Expression(_) => {
                                let err = self.diag.report_todo(statement.span, "combinatorial expression");
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
                ModuleStatementKind::ClockedBlock(ref clocked_block) => {
                    let &ClockedBlock {
                        span, span_keyword: _, ref clock, ref reset, ref block
                    } = clocked_block;
                    let ast::Block { span: _, statements } = block;

                    let scope = self.compiled.scopes.new_child(scope_body, block.span, Visibility::Private);
                    let ctx_clocked = &ExpressionContext::ClockedBlock;

                    // TODO typecheck: clock must be a single-bit clock, reset must be a single-bit reset
                    let clock = self.eval_expression_as_value(ctx_clocked, scope, clock)?;
                    let reset = self.eval_expression_as_value(ctx_clocked, scope, reset)?;

                    let mut result_statements = vec![];

                    for statement in statements {
                        match &statement.inner {
                            BlockStatementKind::VariableDeclaration(_) => {
                                let err = self.diag.report_todo(statement.span, "clocked variable declaration");
                                result_statements.push(LowerStatement::Error(err));
                            }
                            BlockStatementKind::Assignment(_) => {
                                let err = self.diag.report_todo(statement.span, "assignment inside clocked block");
                                result_statements.push(LowerStatement::Error(err));
                            }
                            BlockStatementKind::Expression(_) => {
                                let err = self.diag.report_todo(statement.span, "expression inside clocked block");
                                result_statements.push(LowerStatement::Error(err));
                            }
                        }
                    }

                    let result_block = ModuleBlockClocked {
                        span,
                        clock,
                        reset,
                        on_reset: vec![],
                        on_block: result_statements,
                    };
                    module_blocks.push(ModuleBlockInfo::Clocked(result_block));
                }
            }
        }

        let result = ModuleChecked {
            blocks: module_blocks,
            regs: module_regs,
        };
        Ok(result)
    }
}