use crate::data::diagnostic::ErrorGuaranteed;
use crate::front::scope::{Scope, Visibility};
use crate::new::compile::CompileState;
use crate::new::expression::ExpressionContext;
use crate::new::ir::{IrBlock, IrExpression, IrLocals, IrStatement};
use crate::new::misc::{DomainSignal, ValueDomain};
use crate::new::types::HardwareType;
use crate::new::value::{MaybeCompile, NamedValue};
use crate::syntax::ast::{Assignment, Block, BlockStatement, BlockStatementKind, DomainKind, Expression};
use crate::syntax::pos::Span;
use crate::throw;

// TODO move
// TODO create some common ir process builder
pub struct IrContext<'b> {
    ir_locals: &'b mut IrLocals,
    ir_statements: &'b mut Vec<IrStatement>,
}

impl ExpressionContext for IrContext<'_> {
    type T = TypedIrExpression;

    // TODO reduce the duplication here, const and param will always be evaluated exactly like this, right?
    fn eval_named(self, s: &CompileState, span_use: Span, _: Span, named: NamedValue) -> Result<MaybeCompile<Self::T>, ErrorGuaranteed> {
        match named {
            NamedValue::Constant(cst) =>
                Ok(MaybeCompile::Compile(s.constants[cst].value.clone())),
            NamedValue::Parameter(param) =>
                Ok(MaybeCompile::Compile(s.parameters[param].value.clone())),
            NamedValue::Variable(var) =>
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
}

#[derive(Debug, Clone)]
pub struct TypedIrExpression {
    pub ty: HardwareType,
    pub domain: ValueDomain,
    pub expr: IrExpression,
}

impl CompileState<'_> {
    // TODO move
    pub fn eval_expression_as_ir(&mut self, ir_locals: &mut IrLocals, ir_statements: &mut Vec<IrStatement>, scope: Scope, value: &Expression) -> Result<MaybeCompile<TypedIrExpression>, ErrorGuaranteed> {
        let ctx = IrContext { ir_locals, ir_statements };
        self.eval_expression(ctx, scope, value)
    }

    pub fn elaborate_ir_block(
        &mut self,
        ir_locals: &mut IrLocals,
        domain: DomainKind<DomainSignal>,
        parent_scope: Scope,
        block: &Block<BlockStatement>,
    ) -> Result<IrBlock, ErrorGuaranteed> {
        let Block { span: _, statements } = block;

        let scope = self.scopes.new_child(parent_scope, block.span, Visibility::Private);
        let mut statements_ir = vec![];

        for stmt in statements {
            match &stmt.inner {
                BlockStatementKind::ConstDeclaration(_) => throw!(self.diags.report_todo(stmt.span, "statement kind ConstDeclaration")),
                BlockStatementKind::VariableDeclaration(_) => throw!(self.diags.report_todo(stmt.span, "statement kind VariableDeclaration")),
                BlockStatementKind::Assignment(stmt) => {
                    let Assignment { span: _, op, target, value } = stmt;

                    if op.inner.is_some() {
                        throw!(self.diags.report_todo(stmt.span, "compound assignment"));
                    }

                    throw!(self.diags.report_todo(stmt.span, "assignment"));

                    // TODO how to implement compile-time variables?
                    //   * always emit all reads/writes (if possible and using hardware types)
                    //   * if not possible; record that this variable is compile-time only
                    //   * keep a shadow map of variables to compile-time values,
                    //       merge at the end of blocks that depend on runtime
                    // let target = self.eval_expression_as_assign_target(scope, target);
                    // let value = self.eval_expression::<IrContext>(scope, value);

                    // TODO check that we can read from valid domain
                    // TODO check that we can write to target domain
                    // TODO record write
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
            statements: statements_ir,
        };
        Ok(result)
    }
}
