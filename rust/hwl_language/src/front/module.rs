use crate::data::compiled::Item;
use crate::data::diagnostic::Diagnostic;
use crate::data::module_body::{CombinatorialStatement, ModuleBlockCombinatorial, ModuleBlockInfo, ModuleBody, ModuleReg, ModuleRegInfo};
use crate::front::common::{ScopedEntry, ScopedEntryDirect, TypeOrValue};
use crate::front::driver::{CompileState, ResolveResult};
use crate::front::scope::Visibility;
use crate::front::values::Value;
use crate::syntax::ast;
use crate::syntax::ast::{BlockStatementKind, ModuleStatementKind, RegDeclaration};

impl<'d, 'a> CompileState<'d, 'a> {
    pub fn resolve_module_body(&mut self, module_item: Item, module_ast: &ast::ItemDefModule) -> ResolveResult<ModuleBody> {
        let ast::ItemDefModule { span: _, vis: _, id: _, params: _, ports: _, body } = module_ast;
        let ast::Block { span: _, statements } = body;

        // TODO add to this scope any locally-defined items first
        //   this might require spawning a new sub-driver instance,
        //     or can we avoid that and do everything in a single graph?
        //   do we _want_ to avoid splits? that's good for concurrency!
        let scope_ports = self.compiled.module_info[&module_item].scope_ports;
        let scope_body = self.compiled.scopes.new_child(scope_ports, body.span, Visibility::Private);

        // TODO do we event want to convert to some simpler IR here,
        //   or just leave the backend to walk the AST if it wants?
        let mut module_blocks = vec![];
        let mut module_regs = vec![];

        let mut module_reg_init = vec![];

        for top_statement in statements {
            match &top_statement.inner {
                ModuleStatementKind::RegDeclaration(decl) => {
                    let RegDeclaration { span: _, id, sync, ty, init } = decl;

                    let sync = self.eval_sync_domain(scope_body, &sync.inner)?;
                    let ty = self.eval_expression_as_ty(scope_body, ty)?;
                    let init = self.eval_expression_as_value(scope_body, init)?;

                    let index = module_regs.len();
                    module_regs.push(ModuleRegInfo { sync, ty });
                    let module_reg = ModuleReg { module_item, index };

                    // TODO assert that the init value is known at compile time (basically a kind of sync-ness)
                    module_reg_init.push(Some(init));

                    let value = Value::Reg(module_reg);
                    let entry = ScopedEntry::Direct(ScopedEntryDirect::Immediate(TypeOrValue::Value(value)));
                    self.compiled.scopes[scope_body].maybe_declare(&self.diag, &decl.id, entry, Visibility::Private);
                }
                ModuleStatementKind::WireDeclaration(decl) => {
                    // TODO careful if/when implementing this, wire semantics are unclear
                    self.diag.report_todo(decl.span, "wire declarataion");
                }
                ModuleStatementKind::CombinatorialBlock(ref comb_block) => {
                    let mut result_statements = vec![];

                    for statement in &comb_block.block.statements {
                        match &statement.inner {
                            BlockStatementKind::VariableDeclaration(_) => {
                                self.diag.report_todo(statement.span, "combinatorial variable declaration");
                            }
                            BlockStatementKind::RegDeclaration(decl) => {
                                let diag = Diagnostic::new_simple("register declaration inside combinatorial block", decl.span, "register declaration");
                                self.diag.report(diag);
                            }
                            BlockStatementKind::Assignment(ref assignment) => {
                                let &ast::Assignment { span: _, op, ref target, ref value } = assignment;
                                if op.inner.is_some() {
                                    self.diag.report_todo(statement.span, "combinatorial assignment with operator");
                                }

                                // TODO type and sync checking

                                let target = self.eval_expression_as_value(scope_body, target)?;
                                let value = self.eval_expression_as_value(scope_body, value)?;

                                if let (Value::ModulePort(target), Value::ModulePort(value)) = (target, value) {
                                    result_statements.push(CombinatorialStatement::PortPortAssignment(target, value));
                                } else {
                                    self.diag.report_todo(statement.span, "general combinatorial assignment");
                                }
                            }
                            BlockStatementKind::Expression(_) => {
                                self.diag.report_todo(statement.span, "combinatorial expression");
                            }
                        }
                    }

                    let comb_block = ModuleBlockCombinatorial {
                        statements: result_statements,
                    };
                    module_blocks.push(ModuleBlockInfo::Combinatorial(comb_block));
                }
                ModuleStatementKind::ClockedBlock(_) => {
                    self.diag.report_todo(top_statement.span, "clocked block");
                }
            }
        }

        let result = ModuleBody {
            blocks: module_blocks,
            regs: module_regs,
        };
        Ok(result)
    }
}