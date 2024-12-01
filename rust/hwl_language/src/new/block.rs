use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::CompileState;
use crate::new::expression::ExpressionContext;
use crate::new::ir::{IrAssignmentTarget, IrBlock, IrExpression, IrLocals, IrStatement};
use crate::new::misc::{DomainSignal, ValueDomain};
use crate::new::types::HardwareType;
use crate::new::value::{AssignmentTarget, MaybeCompile, NamedValue};
use crate::syntax::ast::{Assignment, Block, BlockStatement, BlockStatementKind, Expression, Spanned, SyncDomain};
use crate::syntax::pos::Span;
use crate::throw;
use crate::util::result_pair;
use itertools::Itertools;

// TODO move
// TODO create some common ir process builder
pub struct IrContext<'b> {
    ir_locals: &'b mut IrLocals,
    ir_statements: &'b mut Vec<Result<IrStatement, ErrorGuaranteed>>,
}

impl ExpressionContext for IrContext<'_> {
    type T = TypedIrExpression;

    // TODO reduce the duplication here, const and param will always be evaluated exactly like this, right?
    fn eval_named(&mut self, s: &CompileState, span_use: Span, _: Span, named: NamedValue) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed> {
        match named {
            NamedValue::Constant(cst) =>
                Ok(MaybeCompile::Compile(s.constants[cst].value.clone())),
            NamedValue::Parameter(param) =>
                Ok(MaybeCompile::Compile(s.parameters[param].value.clone())),
            NamedValue::Variable(_) =>
                Err(s.diags.report_todo(span_use, "eval variable in IrContext")),
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
    pub fn eval_expression_as_ir(&mut self, ir_locals: &mut IrLocals, ir_statements: &mut Vec<Result<IrStatement, ErrorGuaranteed>>, scope: Scope, value: &Expression) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let mut ctx = IrContext { ir_locals, ir_statements };
        self.eval_expression(&mut ctx, scope, value)
    }

    // TODO how to implement compile-time variables?
    //   * always emit all reads/writes (if possible and using hardware types)
    //   * if not possible; record that this variable is compile-time only
    //   * keep a shadow map of variables to compile-time values,
    //       merge at the end of blocks that depend on runtime
    pub fn elaborate_ir_block(
        &mut self,
        ir_locals: &mut IrLocals,
        block_domain: &BlockDomain,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> Result<IrBlock, ErrorGuaranteed> {
        let Block { span: _, statements } = block;

        let scope = self.scopes.new_child(parent_scope, block.span, Visibility::Private);
        let mut ir_statements = vec![];

        for stmt in statements {
            match &stmt.inner {
                BlockStatementKind::ConstDeclaration(_) => throw!(self.diags.report_todo(stmt.span, "statement kind ConstDeclaration")),
                BlockStatementKind::VariableDeclaration(_) => throw!(self.diags.report_todo(stmt.span, "statement kind VariableDeclaration")),
                BlockStatementKind::Assignment(stmt) => {
                    let Assignment { span: _, op, target, value } = stmt;
                    let target_span = target.span;
                    let value_span = value.span;

                    if op.inner.is_some() {
                        throw!(self.diags.report_todo(stmt.span, "compound assignment"));
                    }

                    let target = self.eval_expression_as_assign_target(scope, target);
                    let mut ctx = IrContext {
                        ir_locals,
                        ir_statements: &mut ir_statements,
                    };
                    let value = self.eval_expression(&mut ctx, scope, value);

                    let stmt = result_pair(target, value).and_then(|(target, value)| {
                        // TODO extract this to a function
                        let (target_ty, target_domain, ir_target) = match target {
                            AssignmentTarget::Port(port) => {
                                let info = &self.ports[port];
                                let domain = ValueDomain::from_port_domain(info.domain.inner.clone());
                                (&info.ty, domain, IrAssignmentTarget::Port(info.ir))
                            },
                            AssignmentTarget::Wire(wire) => {
                                let info = &self.wires[wire];
                                let domain = ValueDomain::from_domain_kind(info.domain.inner.clone());
                                (&info.ty, domain, IrAssignmentTarget::Wire(info.ir))
                            },
                            AssignmentTarget::Register(reg) => {
                                let info = &self.registers[reg];
                                let domain = ValueDomain::Sync(info.domain.inner.clone());
                                (&info.ty, domain, IrAssignmentTarget::Register(self.registers[reg].ir))
                            },
                            AssignmentTarget::Variable(_) => throw!(self.diags.report_todo(stmt.span, "assignment to variable")),
                        };

                        // TODO record write to target
                        // TODO for variables, (also) do the assignment at compile-time (see above)

                        let target_ty = target_ty.as_ref().map_inner(|ty| ty.as_type());

                        // check type and convert to value
                        let (value_domain, ir_value) = match value {
                            MaybeCompile::Compile(v) => {
                                let value_spanned = Spanned { span: value_span, inner: &v };
                                let ir_value = self.check_type_contains_compile_value(stmt.span, target_ty.as_ref(), value_spanned).and_then(|()| {
                                    v.as_hardware_value()
                                        .ok_or_else(|| {
                                            let reason = "compile time value fits in hardware type but is not convertible to hardware value";
                                            self.diags.report_internal_error(value_span, reason)
                                        })
                                });
                                (ValueDomain::CompileTime, ir_value)
                            }
                            MaybeCompile::Other(v) => {
                                let value_ty = Spanned { span: value_span, inner: v.ty.as_type() };
                                let ir_value = self.check_type_contains_type(stmt.span, target_ty.as_ref(), value_ty.as_ref())
                                    .map(|()| v.expr);
                                (v.domain, ir_value)
                            },
                        };

                        // check domains
                        // TODO better error messages with more explanation
                        let target_domain = Spanned { span: target_span, inner: &target_domain };
                        let value_domain = Spanned { span: value_span, inner: &value_domain };

                        let check_domains = match block_domain {
                            BlockDomain::Combinatorial => {
                                self.check_valid_domain_crossing(stmt.span, target_domain, value_domain)
                            }
                            BlockDomain::Clocked(block_domain) => {
                                let block_domain = block_domain
                                    .as_ref()
                                    .map_inner(|d| ValueDomain::Sync(d.clone()));

                                let check_target_domain = self.check_valid_domain_crossing(stmt.span, target_domain, block_domain.as_ref());
                                let check_value_domain = self.check_valid_domain_crossing(stmt.span, block_domain.as_ref(), value_domain);

                                check_target_domain.and(check_value_domain)
                            }
                        };

                        let ir_value = ir_value?;
                        check_domains?;

                        Ok(IrStatement::Assign(ir_target, ir_value))
                    });
                    ir_statements.push(stmt);
                }
                BlockStatementKind::Expression(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Expression")),
                BlockStatementKind::Block(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Block")),
                BlockStatementKind::If(_) => throw!(self.diags.report_todo(stmt.span, "statement kind If")),
                BlockStatementKind::While(_) => throw!(self.diags.report_todo(stmt.span, "statement kind While")),
                BlockStatementKind::For(_) => throw!(self.diags.report_todo(stmt.span, "statement kind For")),
                BlockStatementKind::Return(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Return")),
                BlockStatementKind::Break(_) => throw!(self.diags.report_todo(stmt.span, "statement kind Break")),
                BlockStatementKind::Continue => throw!(self.diags.report_todo(stmt.span, "statement kind Continue")),
            };
        }

        let result = IrBlock {
            statements: ir_statements.into_iter().try_collect()?,
        };
        Ok(result)
    }
}
