use crate::data::diagnostic::{Diagnostics, ErrorGuaranteed};
use crate::new::block::BlockDomain;
use crate::new::ir::{IrBlock, IrStatement, IrVariable, IrVariableInfo, IrVariables};
use crate::new::misc::ValueDomain;
use crate::new::value::AssignmentTarget;
use crate::syntax::ast::Spanned;
use crate::syntax::pos::Span;
use crate::throw;
use std::fmt::Debug;

pub trait ExpressionContext {
    type Block: Debug;

    fn new_ir_block(&self) -> Self::Block;

    fn push_ir_statement(
        &self,
        diags: &Diagnostics,
        block: &mut Self::Block,
        stmt: Spanned<IrStatement>,
    ) -> Result<(), ErrorGuaranteed>;

    fn push_ir_statement_block(&self, block_parent: &mut Self::Block, block_inner: Spanned<Self::Block>);

    fn unwrap_ir_block(&self, diags: &Diagnostics, span: Span, block: Self::Block) -> Result<IrBlock, ErrorGuaranteed>;

    fn new_ir_variable(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        info: IrVariableInfo,
    ) -> Result<IrVariable, ErrorGuaranteed>;

    fn report_assignment(&mut self, target: Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed>;

    fn block_domain(&self) -> &BlockDomain;

    fn condition_domains(&self) -> &[Spanned<ValueDomain>];

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed>;
}

pub struct CompileTimeExpressionContext;

impl ExpressionContext for CompileTimeExpressionContext {
    type Block = ();

    fn new_ir_block(&self) -> () {
        ()
    }

    fn push_ir_statement(
        &self,
        diags: &Diagnostics,
        statements: &mut Self::Block,
        ir_statement: Spanned<IrStatement>,
    ) -> Result<(), ErrorGuaranteed> {
        let &mut () = statements;
        Err(diags.report_internal_error(ir_statement.span, "trying to push IR statement in compile-time context"))
    }

    fn push_ir_statement_block(&self, block_parent: &mut Self::Block, block_inner: Spanned<Self::Block>) {
        let &mut () = block_parent;
        let () = block_inner.inner;
        // do nothing
    }

    fn unwrap_ir_block(&self, diags: &Diagnostics, span: Span, block: Self::Block) -> Result<IrBlock, ErrorGuaranteed> {
        let _ = block;
        Err(diags.report_internal_error(span, "trying to unwrap IR block in compile-time context"))
    }

    fn new_ir_variable(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        info: IrVariableInfo,
    ) -> Result<IrVariable, ErrorGuaranteed> {
        let _ = info;
        Err(diags.report_internal_error(span, "trying to create IR variable in compile-time context"))
    }

    fn report_assignment(&mut self, target: Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed> {
        let _ = target;
        Ok(())
    }

    fn block_domain(&self) -> &BlockDomain {
        &BlockDomain::CompileTime
    }

    fn condition_domains(&self) -> &[Spanned<ValueDomain>] {
        &[]
    }

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        if domain.inner != ValueDomain::CompileTime {
            throw!(diags.report_internal_error(
                domain.span,
                "trying to push non-compile-time condition domain in compile-time context"
            ))
        }
        let _ = (diags, domain);
        f(self)
    }
}

pub struct IrBuilderExpressionContext<'a> {
    block_domain: &'a BlockDomain,
    report_assignment: &'a mut dyn FnMut(Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed>,
    condition_domains: Vec<Spanned<ValueDomain>>,
    ir_variables: IrVariables,
}

impl<'a> IrBuilderExpressionContext<'a> {
    pub fn new(
        block_domain: &'a BlockDomain,
        report_assignment: &'a mut dyn FnMut(Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed>,
    ) -> Self {
        Self {
            block_domain,
            report_assignment,
            condition_domains: vec![],
            ir_variables: IrVariables::default(),
        }
    }

    pub fn finish(self) -> IrVariables {
        assert!(self.condition_domains.is_empty());
        self.ir_variables
    }
}

impl ExpressionContext for IrBuilderExpressionContext<'_> {
    type Block = IrBlock;

    fn new_ir_block(&self) -> Self::Block {
        IrBlock { statements: vec![] }
    }

    fn push_ir_statement(
        &self,
        diags: &Diagnostics,
        block: &mut Self::Block,
        stmt: Spanned<IrStatement>,
    ) -> Result<(), ErrorGuaranteed> {
        let _ = diags;
        block.statements.push(stmt);
        Ok(())
    }

    fn push_ir_statement_block(&self, block_parent: &mut Self::Block, block_inner: Spanned<Self::Block>) {
        // skip pushing empty blocks to slightly clean up the resulting IR
        if block_inner.inner.statements.is_empty() {
            return;
        }

        block_parent
            .statements
            .push(block_inner.map_inner(|block_inner| IrStatement::Block(block_inner)));
    }

    fn unwrap_ir_block(&self, diags: &Diagnostics, span: Span, block: Self::Block) -> Result<IrBlock, ErrorGuaranteed> {
        let _ = (diags, span);
        Ok(block)
    }

    fn new_ir_variable(
        &mut self,
        diags: &Diagnostics,
        span: Span,
        info: IrVariableInfo,
    ) -> Result<IrVariable, ErrorGuaranteed> {
        let _ = (diags, span);
        Ok(self.ir_variables.push(info))
    }

    fn report_assignment(&mut self, target: Spanned<&AssignmentTarget>) -> Result<(), ErrorGuaranteed> {
        (self.report_assignment)(target)
    }

    fn block_domain(&self) -> &BlockDomain {
        self.block_domain
    }

    fn condition_domains(&self) -> &[Spanned<ValueDomain>] {
        &self.condition_domains
    }

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        let _ = diags;

        let len_before = self.condition_domains.len();
        self.condition_domains.push(domain);

        let result = f(self);

        assert_eq!(self.condition_domains.len(), len_before + 1);
        self.condition_domains.pop();

        result
    }
}
