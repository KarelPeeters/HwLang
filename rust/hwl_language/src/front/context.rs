use crate::front::block::BlockDomain;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::ir::{IrBlock, IrStatement, IrVariable, IrVariableInfo, IrVariables};
use crate::front::misc::{Signal, ValueDomain};
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

    fn report_assignment(&mut self, target: Spanned<Signal>) -> Result<(), ErrorGuaranteed>;

    fn block_domain(&self) -> &BlockDomain;

    fn condition_domains(&self) -> &[Spanned<ValueDomain>];

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed>;

    fn check_ir_context(&self, diags: &Diagnostics, span: Span, reason: &str) -> Result<(), ErrorGuaranteed>;
}

pub struct CompileTimeExpressionContext {
    pub span: Span,
    pub reason: String,
}

impl ExpressionContext for CompileTimeExpressionContext {
    type Block = ();

    fn new_ir_block(&self) {}

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
        let () = block;
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

    fn report_assignment(&mut self, target: Spanned<Signal>) -> Result<(), ErrorGuaranteed> {
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

    fn check_ir_context(&self, diags: &Diagnostics, span: Span, access: &str) -> Result<(), ErrorGuaranteed> {
        let diag = Diagnostic::new("trying access hardware in compile-time context")
            .add_error(span, format!("accessing {access} here"))
            .add_info(
                self.span,
                format!("compile-time context because {} must be compile-time", self.reason),
            )
            .finish();
        Err(diags.report(diag))
    }
}

pub struct IrBuilderExpressionContext<'a> {
    block_domain: &'a BlockDomain,
    report_assignment: &'a mut dyn FnMut(Spanned<Signal>) -> Result<(), ErrorGuaranteed>,
    condition_domains: Vec<Spanned<ValueDomain>>,
    ir_variables: IrVariables,
}

impl<'a> IrBuilderExpressionContext<'a> {
    pub fn new(
        block_domain: &'a BlockDomain,
        report_assignment: &'a mut dyn FnMut(Spanned<Signal>) -> Result<(), ErrorGuaranteed>,
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

        block_parent.statements.push(block_inner.map_inner(IrStatement::Block));
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

    fn report_assignment(&mut self, target: Spanned<Signal>) -> Result<(), ErrorGuaranteed> {
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

    fn check_ir_context(&self, _: &Diagnostics, _: Span, _: &str) -> Result<(), ErrorGuaranteed> {
        Ok(())
    }
}
