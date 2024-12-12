use crate::data::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::scope::{Scope, Visibility};
use crate::new::check::check_type_contains_value;
use crate::new::compile::{CompileState, VariableInfo};
use crate::new::expression::ExpressionContext;
use crate::new::ir::{IrAssignmentTarget, IrBlock, IrExpression, IrStatement, IrVariableInfo, IrVariables};
use crate::new::misc::{DomainSignal, ScopedEntry, ValueDomain};
use crate::new::types::{HardwareType, Type};
use crate::new::value::{AssignmentTarget, CompileValue, HardwareValueResult, MaybeCompile, NamedValue};
use crate::syntax::ast::{Assignment, Block, BlockStatement, BlockStatementKind, Expression, IfCondBlockPair, IfStatement, Spanned, SyncDomain, VariableDeclaration};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::data::VecExt;
use crate::util::result_pair;
use itertools::Itertools;

// TODO move
// TODO create some common ir process builder
pub struct IrContext<'b> {
    ir_locals: &'b mut IrVariables,
    ir_statements: &'b mut Vec<Result<Spanned<IrStatement>, ErrorGuaranteed>>,
}

impl ExpressionContext for IrContext<'_> {
    type T = TypedIrExpression;

    // TODO reduce the duplication here, const and param will always be evaluated exactly like this, right?
    fn eval_named(&mut self, s: &CompileState, _: Span, _: Span, named: NamedValue) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed> {
        match named {
            NamedValue::Constant(cst) =>
                Ok(MaybeCompile::Compile(s.constants[cst].value.clone())),
            NamedValue::Parameter(param) =>
                Ok(MaybeCompile::Compile(s.parameters[param].value.clone())),
            NamedValue::Variable(v) => {
                Ok(s.variables[v].current_value.clone())
            }
            NamedValue::Port(port) => {
                let port_info = &s.ports[port];
                let expr = TypedIrExpression {
                    ty: port_info.ty.inner.clone(),
                    domain: ValueDomain::from_port_domain(port_info.domain.inner.clone()),
                    expr: IrExpression::Port(port_info.ir),
                };
                Ok(MaybeCompile::Other(expr))
            }
            NamedValue::Wire(wire) => {
                let wire_info = &s.wires[wire];
                let expr = TypedIrExpression {
                    ty: wire_info.ty.inner.clone(),
                    domain: ValueDomain::from_domain_kind(wire_info.domain.inner.clone()),
                    expr: IrExpression::Wire(wire_info.ir),
                };
                Ok(MaybeCompile::Other(expr))
            }
            NamedValue::Register(reg) => {
                let reg_info = &s.registers[reg];
                let expr = TypedIrExpression {
                    ty: reg_info.ty.inner.clone(),
                    domain: ValueDomain::Sync(reg_info.domain.inner.clone()),
                    expr: IrExpression::Register(reg_info.ir),
                };
                Ok(MaybeCompile::Other(expr))
            }
        }
    }

    fn bool_not(&mut self, s: &CompileState, span_use: Span, v: TypedIrExpression) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed> {
        match v.ty {
            HardwareType::Clock | HardwareType::Bool => {}
            _ => return Err(s.diags.report_internal_error(span_use, "unexpected type for bool_not")),
        }
        Ok(MaybeCompile::Other(TypedIrExpression {
            ty: v.ty,
            domain: v.domain,
            expr: IrExpression::BoolNot(Box::new(v.expr)),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct TypedIrExpression {
    pub ty: HardwareType,
    pub domain: ValueDomain,
    pub expr: IrExpression,
}

#[derive(Debug)]
pub enum BlockDomain {
    Combinatorial,
    Clocked(Spanned<SyncDomain<DomainSignal>>),
}

impl CompileState<'_> {
    // TODO move
    pub fn eval_expression_as_ir(&mut self, ir_locals: &mut IrVariables, ir_statements: &mut Vec<Result<Spanned<IrStatement>, ErrorGuaranteed>>, scope: Scope, value: &Expression) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let mut ctx = IrContext { ir_locals, ir_statements };
        self.eval_expression(&mut ctx, scope, value)
    }

    pub fn elaborate_ir_block(
        &mut self,
        report_assignment: &mut impl FnMut(Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed>,
        ir_locals: &mut IrVariables,
        block_domain: &BlockDomain,
        condition_domains: &mut Vec<Spanned<ValueDomain>>,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> Result<IrBlock, ErrorGuaranteed> {
        let diags = self.diags;

        let Block { span: _, statements } = block;

        let scope = self.scopes.new_child(parent_scope, block.span, Visibility::Private);
        let mut ir_statements = vec![];

        for stmt in statements {
            let stmt_span = stmt.span;
            match &stmt.inner {
                BlockStatementKind::ConstDeclaration(decl) => {
                    self.const_eval_and_declare(scope, decl)
                }
                BlockStatementKind::VariableDeclaration(decl) => {
                    let VariableDeclaration { span: _, mutable, id, ty, init } = decl;
                    let mutable = *mutable;
                    let init_span = init.span;

                    let ty = ty.as_ref()
                        .map(|ty| Ok(Spanned { span: ty.span, inner: self.eval_expression_as_ty(scope, ty)? }))
                        .transpose();
                    let init = self.eval_expression_as_ir(ir_locals, &mut ir_statements, scope, init);

                    // check init fits in type
                    let entry = result_pair(ty, init).and_then(|(ty, init)| {
                        // check init fits in type
                        if let Some(ty) = &ty {
                            let init_spanned = Spanned { span: init_span, inner: &init };
                            check_type_contains_value(diags, stmt.span, ty.as_ref(), init_spanned, true)?;
                        }

                        // build variable
                        let info = VariableInfo {
                            id: id.clone(),
                            mutable,
                            ty,
                            current_value: init,
                        };
                        let variable = self.variables.push(info);
                        Ok(ScopedEntry::Direct(NamedValue::Variable(variable)))
                    });

                    self.scopes[scope].maybe_declare(diags, id.as_ref(), entry, Visibility::Private);
                }
                BlockStatementKind::Assignment(stmt) => {
                    let stmt = self.elaborate_assignment(report_assignment, ir_locals, block_domain, condition_domains, diags, scope, &mut ir_statements, stmt);

                    if let Some(stmt) = stmt.transpose() {
                        ir_statements.push(stmt.map(|stmt| Spanned { span: stmt_span, inner: stmt }));
                    }
                }
                BlockStatementKind::Expression(_) => throw!(diags.report_todo(stmt.span, "statement kind Expression")),
                BlockStatementKind::Block(inner) => {
                    let inner_block = self.elaborate_ir_block(report_assignment, ir_locals, block_domain, condition_domains, scope, inner)?;
                    ir_statements.push(Ok(Spanned { span: stmt.span, inner: IrStatement::Block(inner_block) }));
                }
                BlockStatementKind::If(stmt_if) => {
                    let IfStatement { initial_if, else_ifs, final_else } = stmt_if;

                    // TODO avoid allocation here
                    let mut all_ifs = vec![];
                    all_ifs.push(initial_if);
                    all_ifs.extend(else_ifs.iter());

                    let stmt_ir = self.elaborate_if_statement(report_assignment, block_domain, condition_domains, ir_locals, scope, &mut ir_statements, &all_ifs, final_else)?;
                    match stmt_ir {
                        LoweredIf::Nothing => {}
                        LoweredIf::SingleBlock(block) => {
                            let stmt_ir = IrStatement::Block(block);
                            ir_statements.push(Ok(Spanned { span: stmt.span, inner: stmt_ir }));
                        }
                        LoweredIf::IfStatement(if_stmt) => {
                            let stmt_ir = IrStatement::If(if_stmt);
                            ir_statements.push(Ok(Spanned { span: stmt.span, inner: stmt_ir }));
                        }
                    }
                }
                BlockStatementKind::While(_) => throw!(diags.report_todo(stmt.span, "statement kind While")),
                BlockStatementKind::For(_) => throw!(diags.report_todo(stmt.span, "statement kind For")),
                BlockStatementKind::Return(_) => throw!(diags.report_todo(stmt.span, "statement kind Return")),
                BlockStatementKind::Break(_) => throw!(diags.report_todo(stmt.span, "statement kind Break")),
                BlockStatementKind::Continue => throw!(diags.report_todo(stmt.span, "statement kind Continue")),
            };
        }

        let result = IrBlock {
            statements: ir_statements.into_iter().try_collect()?,
        };
        Ok(result)
    }

    fn elaborate_assignment(
        &mut self,
        report_assignment: &mut impl FnMut(Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed>,
        ir_locals: &mut IrVariables,
        block_domain: &BlockDomain,
        condition_domains: &mut Vec<Spanned<ValueDomain>>,
        diags: &Diagnostics,
        scope: Scope,
        ir_statements: &mut Vec<Result<Spanned<IrStatement>, ErrorGuaranteed>>,
        stmt: &Assignment,
    ) -> Result<Option<IrStatement>, ErrorGuaranteed> {
        let Assignment { span: _, op, target, value } = stmt;
        let target_span = target.span;
        let value_span = value.span;

        if op.inner.is_some() {
            throw!(diags.report_todo(stmt.span, "compound assignment"));
        }

        let target = self.eval_expression_as_assign_target(scope, target);
        let value = self.eval_expression_as_ir(ir_locals, ir_statements, scope, value);

        let target = target?;
        let value = value?;

        let condition_domains = &*condition_domains;

        let (target_ty, target_domain, ir_target) = match target {
            AssignmentTarget::Port(port) => {
                let info = &self.ports[port];
                let domain = ValueDomain::from_port_domain(info.domain.inner.clone());
                (&info.ty, domain, IrAssignmentTarget::Port(info.ir))
            }
            AssignmentTarget::Wire(wire) => {
                let info = &self.wires[wire];
                let domain = ValueDomain::from_domain_kind(info.domain.inner.clone());
                (&info.ty, domain, IrAssignmentTarget::Wire(info.ir))
            }
            AssignmentTarget::Register(reg) => {
                let info = &self.registers[reg];
                let domain = ValueDomain::Sync(info.domain.inner.clone());
                (&info.ty, domain, IrAssignmentTarget::Register(self.registers[reg].ir))
            }
            AssignmentTarget::Variable(var) => {
                let VariableInfo { id, mutable, ty, current_value } = &mut self.variables[var];

                // check mutable
                if !*mutable {
                    let diag = Diagnostic::new("assignment to immutable variable")
                        .add_error(target_span, "variable assigned to here")
                        .add_info(id.span(), "variable declared as immutable here")
                        .finish();
                    return Err(diags.report(diag));
                }

                // check type
                if let Some(ty) = ty {
                    let value_spanned = Spanned { span: value_span, inner: &value };
                    check_type_contains_value(diags, stmt.span, ty.as_ref(), value_spanned, true)?;
                }

                // store value to an ir variable, to turn this assignment into "by value"
                //   instead of some weird "by reference"
                let (ir_statement, stored_value) = match value {
                    MaybeCompile::Compile(value) => (None, MaybeCompile::Compile(value)),
                    MaybeCompile::Other(value) => {
                        let ir_variable = ir_locals.push(IrVariableInfo {
                            ty: value.ty.to_ir(),
                            debug_info_id: id.clone(),
                        });

                        let ir_statement = IrStatement::Assign(IrAssignmentTarget::Variable(ir_variable), value.expr);

                        let stored_value = TypedIrExpression {
                            ty: value.ty,
                            domain: value.domain,
                            expr: IrExpression::Variable(ir_variable),
                        };

                        (Some(ir_statement), MaybeCompile::Other(stored_value))
                    }
                };

                *current_value = stored_value;
                return Ok(ir_statement)
            }
        };

        let target_ty = target_ty.as_ref().map_inner(|ty| ty.as_type());
        let value_spanned = Spanned { span: value_span, inner: &value };
        let check_ty = check_type_contains_value(diags, stmt.span, target_ty.as_ref(), value_spanned, true);

        // check type and convert to value
        let (value_domain, ir_value) = match value {
            MaybeCompile::Compile(v) => {
                let ir_value = match v.as_hardware_value() {
                    HardwareValueResult::Success(v) =>
                        Ok(v),
                    HardwareValueResult::Undefined | HardwareValueResult::PartiallyUndefined =>
                        Err(diags.report_simple("undefined can only be used for register initialization", value_span, "value is undefined")),
                    HardwareValueResult::Unrepresentable => {
                        // TODO fix this duplication
                        let reason = "compile time value fits in hardware type but is not convertible to hardware value";
                        Err(diags.report_internal_error(value_span, reason))
                    }
                };
                (ValueDomain::CompileTime, ir_value)
            }
            MaybeCompile::Other(v) => {
                (v.domain, Ok(v.expr))
            }
        };

        // check domains
        // TODO better error messages with more explanation
        let target_domain = Spanned { span: target_span, inner: &target_domain };
        let value_domain = Spanned { span: value_span, inner: &value_domain };

        let check_domains = match block_domain {
            BlockDomain::Combinatorial => {
                let mut check = self.check_valid_domain_crossing(stmt.span, target_domain, value_domain, "value to target in combinatorial block");

                for condition_domain in condition_domains {
                    let c = self.check_valid_domain_crossing(stmt.span, target_domain, condition_domain.as_ref(), "condition to target in combinatorial block");
                    check = check.and(c);
                }

                check
            }
            BlockDomain::Clocked(block_domain) => {
                let block_domain = block_domain
                    .as_ref()
                    .map_inner(|d| ValueDomain::Sync(d.clone()));

                let check_target_domain = self.check_valid_domain_crossing(stmt.span, target_domain, block_domain.as_ref(), "clocked block to target");
                let check_value_domain = self.check_valid_domain_crossing(stmt.span, block_domain.as_ref(), value_domain, "value to clocked block");

                check_target_domain.and(check_value_domain)
            }
        };

        let ir_value = ir_value?;
        check_domains?;
        check_ty?;

        report_assignment(Spanned { span: target_span, inner: &target })?;
        Ok(Some(IrStatement::Assign(ir_target, ir_value)))
    }

    fn elaborate_if_statement(
        &mut self,
        report_assignment: &mut impl FnMut(Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed>,
        block_domain: &BlockDomain,
        condition_domains: &mut Vec<Spanned<ValueDomain>>,
        ir_locals: &mut IrVariables,
        scope: Scope,
        ir_statements: &mut Vec<Result<Spanned<IrStatement>, ErrorGuaranteed>>,
        ifs: &[&IfCondBlockPair<Box<Expression>, Block<BlockStatement>>],
        final_else: &Option<Block<BlockStatement>>,
    ) -> Result<LoweredIf, ErrorGuaranteed> {
        let diags = self.diags;

        let (initial_if, remaining_ifs) = match ifs.split_first() {
            Some(p) => p,
            None => {
                return match final_else {
                    None => Ok(LoweredIf::Nothing),
                    Some(final_else) => {
                        let block_ir = self.elaborate_ir_block(report_assignment, ir_locals, block_domain, condition_domains, scope, final_else)?;
                        Ok(LoweredIf::SingleBlock(block_ir))
                    }
                };
            }
        };

        let IfCondBlockPair { span: _, cond, block } = initial_if;
        let cond_eval = self.eval_expression_as_ir(ir_locals, ir_statements, scope, cond)?;

        let ty_bool = Spanned { span: cond.span, inner: &Type::Bool };
        let cond_eval_spanned = Spanned { span: cond.span, inner: &cond_eval };
        check_type_contains_value(diags, cond.span, ty_bool, cond_eval_spanned, false)?;

        match cond_eval {
            // evaluate the if at compile-time
            MaybeCompile::Compile(cond_eval) => {
                let cond_eval = match cond_eval {
                    CompileValue::Bool(b) => b,
                    _ => throw!(diags.report_internal_error(cond.span, "expected bool value")),
                };

                // only visit the selected branch
                if cond_eval {
                    let block_ir = self.elaborate_ir_block(report_assignment, ir_locals, block_domain, condition_domains, scope, block)?;
                    Ok(LoweredIf::SingleBlock(block_ir))
                } else {
                    self.elaborate_if_statement(report_assignment, block_domain, condition_domains, ir_locals, scope, ir_statements, remaining_ifs, final_else)
                }
            }
            // evaluate the if at runtime, generate the IR
            MaybeCompile::Other(cond_eval) => {
                // check condition domain
                // TODO extract this to a common function?
                let check_cond_domain = match block_domain {
                    BlockDomain::Combinatorial => Ok(()),
                    BlockDomain::Clocked(block_domain) => {
                        let cond_domain = Spanned { span: cond.span, inner: &cond_eval.domain };
                        let block_domain = block_domain
                            .as_ref()
                            .map_inner(|d| ValueDomain::Sync(d.clone()));
                        self.check_valid_domain_crossing(cond.span, block_domain.as_ref(), cond_domain, "condition used in clocked block")
                    }
                };

                // record condition domain
                let cond_domain = Spanned { span: cond.span, inner: cond_eval.domain };
                let (block_ir, else_ir) = condition_domains.with_pushed(cond_domain, |condition_domains| {
                    // lower both branches
                    let block_ir = self.elaborate_ir_block(report_assignment, ir_locals, block_domain, condition_domains, scope, block)?;
                    let else_ir = self.elaborate_if_statement(report_assignment, block_domain, condition_domains, ir_locals, scope, ir_statements, remaining_ifs, final_else)?;

                    Ok((block_ir, else_ir))
                })?;

                check_cond_domain?;

                let initial_if = IfCondBlockPair {
                    span: cond.span,
                    cond: cond_eval.expr,
                    block: block_ir,
                };

                let stmt = match else_ir {
                    LoweredIf::Nothing => {
                        // new simple if statement without any else-s
                        IfStatement {
                            initial_if,
                            else_ifs: vec![],
                            final_else: None,
                        }
                    }
                    LoweredIf::SingleBlock(else_ir) => {
                        // new simple if statement with opaque else
                        IfStatement {
                            initial_if,
                            else_ifs: vec![],
                            final_else: Some(else_ir),
                        }
                    }
                    LoweredIf::IfStatement(else_if_stmt) => {
                        // merge into a single bigger if statement
                        let IfStatement { initial_if: else_initial_if, else_ifs: mut combined_else_ifs, final_else } = else_if_stmt;
                        combined_else_ifs.insert(0, else_initial_if);

                        IfStatement {
                            initial_if,
                            else_ifs: combined_else_ifs,
                            final_else,
                        }
                    }
                };

                Ok(LoweredIf::IfStatement(stmt))
            }
        }
    }
}

#[derive(Debug)]
pub enum LoweredIf {
    Nothing,
    SingleBlock(IrBlock),
    IfStatement(IfStatement<IrExpression, IrBlock>),
}