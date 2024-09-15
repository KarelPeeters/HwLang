use crate::data::compiled::Item;
use crate::data::diagnostic::{Diagnostic, DiagnosticAddable};
use crate::data::module_body::{CombinatorialStatement, ModuleBlock, ModuleBlockCombinatorial, ModuleBody};
use crate::front::driver::{CompileState, ResolveResult};
use crate::front::scope::Visibility;
use crate::front::values::Value;
use crate::syntax::ast;
use crate::syntax::ast::StatementKind;

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

        let mut module_blocks = vec![];

        for top_statement in statements {
            match top_statement.kind {
                StatementKind::Declaration(_) => {
                    self.diag.report_todo(top_statement.span, "module top-level declaration");
                }
                StatementKind::Assignment(_) => {
                    self.diag.report_todo(top_statement.span, "module top-level assignment");
                }
                StatementKind::Expression(_) => {
                    self.diag.report_todo(top_statement.span, "module top-level expression");
                }
                StatementKind::CombinatorialBlock(ref comb_block) => {
                    let mut result_statements = vec![];

                    for statement in &comb_block.block.statements {
                        match statement.kind {
                            StatementKind::Declaration(_) => {
                                self.diag.report_todo(statement.span, "combinatorial declaration");
                            }
                            StatementKind::Assignment(ref assignment) => {
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
                            StatementKind::Expression(_) => {
                                self.diag.report_todo(statement.span, "combinatorial expression");
                            }

                            StatementKind::CombinatorialBlock(ref comb_block_inner) => {
                                let err = Diagnostic::new("nested combinatorial block")
                                    .add_info(comb_block.span_keyword, "outer")
                                    .add_error(comb_block_inner.span_keyword, "inner")
                                    .finish();
                                self.diag.report(err);
                            }

                            StatementKind::ClockedBlock(ref clock_block_inner) => {
                                let err = Diagnostic::new("nested clock block in combinatorial block")
                                    .add_info(comb_block.span_keyword, "outer")
                                    .add_error(clock_block_inner.span_keyword, "inner")
                                    .finish();
                                self.diag.report(err);
                            }
                        }
                    }

                    let comb_block = ModuleBlockCombinatorial {
                        statements: result_statements,
                    };
                    module_blocks.push(ModuleBlock::Combinatorial(comb_block));
                }
                StatementKind::ClockedBlock(_) => {
                    self.diag.report_todo(top_statement.span, "clocked block");
                }
            }
        }

        let result = ModuleBody {
            blocks: module_blocks,
        };
        Ok(result)
    }
}