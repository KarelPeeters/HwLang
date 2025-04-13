use crate::front::block::BlockDomain;
use crate::front::compile::CompileItemContext;
use crate::front::diagnostic::{Diagnostic, DiagnosticAddable, Diagnostics, ErrorGuaranteed};
use crate::front::implication::Implication;
use crate::front::ir::{
    IrBlock, IrExpression, IrRegister, IrRegisterInfo, IrRegisters, IrStatement, IrVariable, IrVariableInfo,
    IrVariables,
};
use crate::front::misc::{DomainSignal, Signal, ValueDomain};
use crate::front::types::HardwareType;
use crate::front::variables::{ValueVersioned, VariableValues};
use crate::syntax::ast::{MaybeIdentifier, Spanned, SyncDomain};
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

    fn new_ir_register(
        &mut self,
        ctx: &CompileItemContext,
        diags: &Diagnostics,
        id: MaybeIdentifier,
        ty: HardwareType,
        init: Option<IrExpression>,
    ) -> Result<(SyncDomain<DomainSignal>, IrRegister), ErrorGuaranteed>;

    fn report_assignment(
        &mut self,
        diags: &Diagnostics,
        target: Spanned<Signal>,
        vars: &mut VariableValues,
    ) -> Result<(), ErrorGuaranteed>;

    fn block_domain(&self) -> &BlockDomain;

    fn condition_domains(&self) -> &[Spanned<ValueDomain>];

    fn with_condition_domain<T>(
        &mut self,
        diags: &Diagnostics,
        domain: Spanned<ValueDomain>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed>;

    fn with_implications<T>(
        &mut self,
        diags: &Diagnostics,
        implications: Vec<Implication>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed>;

    fn for_each_implication(&self, value: ValueVersioned, f: impl FnMut(&Implication));

    fn is_ir_context(&self) -> bool;

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

    fn new_ir_register(
        &mut self,
        ctx: &CompileItemContext,
        diags: &Diagnostics,
        id: MaybeIdentifier,
        ty: HardwareType,
        init: Option<IrExpression>,
    ) -> Result<(SyncDomain<DomainSignal>, IrRegister), ErrorGuaranteed> {
        let _ = (ctx, ty, init);
        Err(diags.report_internal_error(id.span(), "trying to create register in compile-time context"))
    }

    fn report_assignment(
        &mut self,
        diags: &Diagnostics,
        target: Spanned<Signal>,
        vars: &mut VariableValues,
    ) -> Result<(), ErrorGuaranteed> {
        let _ = vars;
        Err(diags.report_internal_error(
            target.span,
            "assigning to signal in compile-time context should not be possible",
        ))
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

    fn with_implications<T>(
        &mut self,
        diags: &Diagnostics,
        implications: Vec<Implication>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        if !implications.is_empty() {
            throw!(diags.report_internal_error(self.span, "trying to push implications in compile-time context"))
        }
        f(self)
    }

    fn for_each_implication(&self, _: ValueVersioned, _: impl FnMut(&Implication)) {
        // do nothing
    }

    fn is_ir_context(&self) -> bool {
        false
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

pub struct ExtraRegister {
    pub span: Span,
    pub reg: IrRegister,
    pub init: Option<IrExpression>,
}

// TODO rename to hardware
// TODO move to module?
// TODO make a shared enum between comb/clocked, there is a bunch of behavior that's different
pub struct IrBuilderExpressionContext<'a> {
    block_domain: &'a BlockDomain,
    report_assignment: &'a mut dyn FnMut(Spanned<Signal>) -> Result<(), ErrorGuaranteed>,
    condition_domains: Vec<Spanned<ValueDomain>>,
    implications: Vec<Implication>,
    ir_variables: IrVariables,
    ir_registers: Option<(&'a mut IrRegisters, &'a mut Vec<ExtraRegister>)>,
}

impl<'a> IrBuilderExpressionContext<'a> {
    pub fn new(
        block_domain: &'a BlockDomain,
        ir_registers: Option<(&'a mut IrRegisters, &'a mut Vec<ExtraRegister>)>,
        report_assignment: &'a mut dyn FnMut(Spanned<Signal>) -> Result<(), ErrorGuaranteed>,
    ) -> Self {
        Self {
            block_domain,
            report_assignment,
            condition_domains: vec![],
            implications: vec![],
            ir_variables: IrVariables::new(),
            ir_registers,
        }
    }

    pub fn finish(self) -> IrVariables {
        assert!(self.condition_domains.is_empty());
        self.ir_variables
    }
}

impl<'a> ExpressionContext for IrBuilderExpressionContext<'a> {
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

    fn new_ir_register(
        &mut self,
        ctx: &CompileItemContext,
        diags: &Diagnostics,
        id: MaybeIdentifier,
        ty: HardwareType,
        init: Option<IrExpression>,
    ) -> Result<(SyncDomain<DomainSignal>, IrRegister), ErrorGuaranteed> {
        match &mut self.ir_registers {
            None => Err(diags.report_simple(
                "defining new registers is only allowed in a clocked block",
                id.span(),
                "attempt to create a register here",
            )),
            Some((ir_registers, register_inits)) => {
                let domain = match self.block_domain {
                    BlockDomain::Clocked(domain) => domain.inner,
                    BlockDomain::CompileTime | BlockDomain::Combinatorial => {
                        return Err(diags.report_internal_error(id.span(), "expected clocked domain"))
                    }
                };

                let reg_info = IrRegisterInfo {
                    ty: ty.as_ir(),
                    debug_info_id: id.clone(),
                    debug_info_ty: ty,
                    debug_info_domain: domain.to_diagnostic_string(ctx),
                };

                let reg = ir_registers.push(reg_info);
                let extra = ExtraRegister {
                    span: id.span(),
                    reg,
                    init,
                };
                register_inits.push(extra);
                Ok((domain, reg))
            }
        }
    }

    fn report_assignment(
        &mut self,
        diags: &Diagnostics,
        target: Spanned<Signal>,
        vars: &mut VariableValues,
    ) -> Result<(), ErrorGuaranteed> {
        // TODO remove this callback indirection,
        //   there's only one user that cares about it (module) and they just want a recording
        let err1 = (self.report_assignment)(target);
        let err2 = vars.signal_report_write(diags, target);

        err1?;
        err2?;
        Ok(())
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

        let len_domains_before = self.condition_domains.len();
        self.condition_domains.push(domain);

        let result = f(self);

        assert_eq!(self.condition_domains.len(), len_domains_before + 1);
        self.condition_domains.pop();

        result
    }

    fn with_implications<T>(
        &mut self,
        diags: &Diagnostics,
        implications: Vec<Implication>,
        f: impl for<'s> FnOnce(&'s mut Self) -> Result<T, ErrorGuaranteed>,
    ) -> Result<T, ErrorGuaranteed> {
        let _ = diags;

        let len_implications_before = self.implications.len();
        let len_implications_added = implications.len();
        self.implications.extend(implications);

        let result = f(self);

        assert_eq!(
            self.implications.len(),
            len_implications_before + len_implications_added
        );
        self.implications.truncate(len_implications_before);

        result
    }

    fn for_each_implication(&self, value: ValueVersioned, mut f: impl FnMut(&Implication)) {
        for implication in &self.implications {
            if implication.value == value {
                f(implication);
            }
        }
    }

    fn is_ir_context(&self) -> bool {
        true
    }

    fn check_ir_context(&self, _: &Diagnostics, _: Span, _: &str) -> Result<(), ErrorGuaranteed> {
        Ok(())
    }
}
